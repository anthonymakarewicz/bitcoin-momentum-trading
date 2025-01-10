import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Backtester:
    def __init__(self, 
                 features, 
                 target,
                 close_price,
                 model,
                 params,
                 data_processor=None,
                 autoencoder=None, 
                 max_holding_period=15, 
                 stop_loss=0.005, 
                 take_profit=0.005,
                 transaction_costs=0.0008):
        self.features = features.copy()
        self.target = target.copy()
        self.close_price = close_price.copy()
        self.model = model
        self.data_processor = data_processor 
        self.autoencoder = autoencoder
        self.params = params
        self.max_holding_period = max_holding_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_costs = transaction_costs

        if "seq_len" in self.model.get_params():
            self.model_has_seq_len = True
        else:
            self.model_has_seq_len = False

    def _train_val_test_split(
        self,
        df,
        start_date,
        end_date,
        val_window_months=1,
        test_window_months=1,
        step_months=1
    ):
        """
        Yields (train_idx, val_idx, test_idx) for a monthly rolling scenario:
        - The first TEST window starts at `start_date` (1-month size by default).
        - The VAL window is the 1-month slice immediately before the test window.
        - The TRAIN window is everything before the val window.

        Args:
            df (pd.DataFrame): Must have a sorted DatetimeIndex or at least something you can sort by date.
            start_date (str): 'YYYY-MM-DD', the start of the FIRST test window.
            end_date   (str): 'YYYY-MM-DD', the last possible date for test (we stop if next test extends beyond).
            val_window_months (int): Size of the validation window in months. (Default 1)
            test_window_months (int): Size of the test window in months. (Default 1)
            step_months (int): By how many months we shift forward each iteration. Typically 1 for monthly stepping.

        Yields:
            (train_index, val_index, test_index): index labels for each split.
        """
        date_index = df.index
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        df_filtered = df.loc[date_index <= end_ts]
        if df_filtered.empty:
            raise ValueError("No data in the requested [start_date, end_date] range.")

        current_test_start = start_ts

        while True:
            test_start = current_test_start
            test_end = test_start + relativedelta(months=test_window_months) - pd.Timedelta(days=1)
            if test_end > end_ts:
                break

            val_end = test_start - pd.Timedelta(days=1)
            val_start = val_end - relativedelta(months=val_window_months) + pd.Timedelta(days=1)

            if val_start < df.index[0]:
                val_start = df.index[0]

            train_end = val_start - pd.Timedelta(days=1)
            train_start = df.index[0]  # expanding from the beginning

            train_mask = (df.index >= train_start) & (df.index <= train_end)
            val_mask = (df.index >= val_start) & (df.index <= val_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_idx = df.index[train_mask]
            val_idx = df.index[val_mask]
            test_idx = df.index[test_mask]

            if len(test_idx) == 0:
                break

            yield (train_idx, val_idx, test_idx)

            current_test_start = current_test_start + relativedelta(months=step_months)
            if current_test_start > end_ts:
                break

    def _optimize_params(self, X_train, y_train, X_val, y_val):
        best_params = None
        best_score = -np.inf

        param_grid_iter = tqdm(ParameterGrid(self.params), desc="Hyperparameter Tuning")
        for params in param_grid_iter:
            self.model.set_params(**params)
            self.model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())

            y_pred = self.model.predict(X_val.to_numpy()).ravel()

            if self.model_has_seq_len:
                y_val_adj = y_val.iloc[self.model.get_params("seq_len"):].copy()
            else:
                y_val_adj = y_val

            score = accuracy_score(y_val_adj, y_pred)

            param_grid_iter.set_postfix({
                "Accuracy": f"{score:.2f}",
                **params
            })

            if score > best_score:
                best_score = score
                best_params = params

        return best_params

    def _apply_processor_and_autoencoder(self, X, fit=False):
        X_proc = X.copy()
        
        # DataProcessor
        if self.data_processor is not None:
            if fit:
                X_proc = self.data_processor.fit_transform(X_proc)
            else:
                X_proc = self.data_processor.transform(X_proc)

        # Autoencoder
        if self.autoencoder is not None:
            if fit:
                self.autoencoder.fit(X_proc)
            X_proc = self.autoencoder.transform(X_proc)

        return X_proc

    def run_backtest(self, start_date, end_date):
        self.y_pred = []
        self.y_test = []

        for (train_idx, val_idx, test_idx) in self._train_val_test_split(
            df=self.features, 
            start_date=start_date, 
            end_date=end_date
        ):
            X_train = self.features.loc[train_idx]
            y_train = self.target.loc[train_idx]

            X_val = self.features.loc[val_idx]
            y_val = self.target.loc[val_idx]

            X_test = self.features.loc[test_idx]
            y_test = self.target.loc[test_idx]

            X_train_proc = self._apply_processor_and_autoencoder(X_train, fit=True)
            X_val_proc = self._apply_processor_and_autoencoder(X_val, fit=False)

            best_params = self._optimize_params(X_train_proc, y_train, X_val_proc, y_val)
            self.model.set_params(**best_params)

            X_train_val = pd.concat([X_train, X_val], axis=0)
            y_train_val = pd.concat([y_train, y_val], axis=0)

            X_train_val_proc = self._apply_processor_and_autoencoder(X_train_val, fit=True)
            X_test_proc = self._apply_processor_and_autoencoder(X_test, fit=False)

            self.model.fit(X_train_val_proc.to_numpy(), y_train_val.to_numpy().ravel())
            preds = self.model.predict_proba(X_test_proc.to_numpy())[:, -1]
            self.y_pred.append(preds)

            if self.model_has_seq_len:
                self.y_test.append(y_test.iloc[self.model.get_params("seq_len"):])
            else:
                self.y_test.append(y_test)

        self.y_test = pd.concat(self.y_test)
        self.y_pred = pd.DataFrame(np.concatenate(self.y_pred), index=self.y_test.index)
        self.results = self._calculate_results()
        
    def _calculate_results(self):
        results = pd.DataFrame(index=self.y_pred.index)
        results["prediction"] = self.y_pred
        results["position"] = 0
        results["pnl"] = 0.0
        results["take_profit"] = np.nan
        results["stop_loss"]  = np.nan
        results["holding_time"] = 0

        returns = self.close_price.pct_change().fillna(0)
        curr_pos = 0
        tp_value = np.nan
        sl_value = np.nan
        hold_time = 0

        for i in range(len(results)):
            curr_close_price = self.close_price.loc[results.index[i]]
            # If no position, check entry
            if curr_pos == 0:
                if results.at[results.index[i], "prediction"] > 0.7:
                    tp_value = curr_close_price * (1 + self.take_profit)
                    sl_value = curr_close_price * (1 - self.stop_loss)
                    curr_pos = 1
                    hold_time = 0
                    
                elif results.at[results.index[i], "prediction"] < 0.3:
                    tp_value = curr_close_price * (1 - self.take_profit)
                    sl_value = curr_close_price * (1 + self.stop_loss)
                    curr_pos = -1
                    hold_time = 0

            else:  # Already in a position, check exit
                if curr_pos == 1:
                    if (curr_close_price >= tp_value or
                        curr_close_price <= sl_value or
                        hold_time >= self.max_holding_period
                    ):
                        curr_pos = 0
                        hold_time = 0
                        tp_value = np.nan
                        sl_value = np.nan
                    else:
                        hold_time += 1

                elif curr_pos == -1:
                    if (curr_close_price <= tp_value or
                        curr_close_price >= sl_value or
                        hold_time >= self.max_holding_period
                    ):
                        curr_pos = 0
                        hold_time = 0
                        tp_value = np.nan
                        sl_value = np.nan
                    else:
                        hold_time += 1

            results.at[results.index[i], "position"] = curr_pos
            results.at[results.index[i], "take_profit"] = tp_value
            results.at[results.index[i], "stop_loss"]  = sl_value
            results.at[results.index[i], "holding_time"] = hold_time

            # PnL calculation
            if i > 0:
                results.at[results.index[i], "gross_pnl"] = (
                    results.at[results.index[i-1], "position"] * returns.loc[results.index[i]]
                )
                # Transaction costs
                if results.at[results.index[i], "position"] != results.at[results.index[i-1], "position"]:
                    results.at[results.index[i], "net_pnl"] = results.at[results.index[i], "gross_pnl"] - self.transaction_costs

        results["cum_gross_pnl"] = 1 + results["gross_pnl"].cumsum()
        if "net_pnl" in results.columns:
            results["cum_net_pnl"] = 1 + results["net_pnl"].cumsum()
        
        return results