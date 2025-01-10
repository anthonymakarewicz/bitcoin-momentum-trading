import numpy as np
import polars as pl
import talib as ta
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn style class for creating technical & time-based features
    on a Polars DataFrame with OHLCV columns.

    - windows: list of integers for rolling calculations (SMA, RSI, etc.).
    """

    def __init__(self, windows=[7, 14, 21, 50, 100], drop_original=True):
        """
        Args:
            windows (list): Time periods for rolling/technical indicators.
            drop_original (bool): Whether to drop original OHLCV columns at the end.
        """
        self.windows = windows
        self.drop_original = drop_original
    
    def fit(self, X, y=None):
        """
        No training needed, just return self.
        """
        return self
    
    def transform(self, X):
        """
        Applies all feature-creation functions to the Polars DataFrame X.
        Returns a new Polars DataFrame with the generated features.
        """
        # Make a copy if you prefer not to modify X in place
        df = X.clone()

        # Keep track of original columns if you plan to drop them
        orig_cols = df.columns

        # 1) Create returns-based features
        df = self._create_return_features(df, self.windows)

        # 2) Create technical indicators
        df = self._create_technical_indicators_features(df, self.windows)

        # 3) Create volume features
        df = self._create_volume_features(df, self.windows)

        # 4) Create time-based features
        df = self._create_time_features(df)

        # Clean up (drop NaNs, nulls)
        df = df.drop_nulls()

        # Optionally drop original OHLCV columns
        if self.drop_original:
            keep_cols = set(df.columns) - set(["Open", "High", "Low", "Close", "Volume", "Date"])
            df = df.select(list(keep_cols))

        return df
    
    def fit_transform(self, X, y=None):
        """
        Convenience method = fit + transform.
        """
        return self.fit(X, y).transform(X)

    # ----------------------------------------------------------------
    # Below are static or internal helper methods for each feature block
    # ----------------------------------------------------------------

    def _create_return_features(self, prices: pl.DataFrame, windows):
        """
        Creates:
          - 'log_returns'
          - rolling mean/std of log returns for each window
        """
        # 1) log returns
        prices = prices.with_columns(
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias('log_returns')
        )

        # 2) rolling mean/std of log returns
        features = []
        for w in windows:
            features.extend([
                pl.col('log_returns').rolling_mean(window_size=w).alias(f'mean_log_returns_{w}'),
                pl.col('log_returns').rolling_std(window_size=w).alias(f'std_log_returns_{w}'),
            ])
        
        prices = prices.with_columns(features)
        return prices

    def _create_technical_indicators_features(self, prices: pl.DataFrame, windows):
        """
        Extract standard TA indicators: MACD, SMA/EMA, RSI, ADX, ATR, BBANDS, ROC, etc.
        """
        close = prices["Close"]
        high = prices['High']
        low = prices['Low']

        # MACD
        macd, macd_signal, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicator_columns = [
            pl.Series('MACD', macd),
            pl.Series('MACD_signal', macd_signal)
        ]

        for w in windows:
            # SMA, EMA
            sma = ta.SMA(close, timeperiod=w)
            ema = ta.EMA(close, timeperiod=w)

            # Bollinger Bands
            bb_upper, _, bb_lower = ta.BBANDS(close, timeperiod=w, nbdevup=2, nbdevdn=2)

            # RSI
            rsi = ta.RSI(close, timeperiod=w)

            # ADX
            adx = ta.ADX(high, low, close, timeperiod=w)

            # ATR
            atr = ta.ATR(high, low, close, timeperiod=w)

            # ROC
            roc = ta.ROCP(close, timeperiod=w)

            indicator_columns.extend([
                pl.Series(f'SMA_{w}', sma),
                pl.Series(f'EMA_{w}', ema),
                pl.Series(f'RSI_{w}', rsi),
                pl.Series(f'ADX_{w}', adx),
                pl.Series(f'ATR_{w}', atr),
                pl.Series(f'ROC_{w}', roc),
                pl.Series(f'BB_upper_{w}', bb_upper),
                pl.Series(f'BB_lower_{w}', bb_lower)
            ])

        # Add them to DF
        prices = prices.with_columns(indicator_columns)

        # Example cross feature
        if "SMA_7" in prices.columns and "SMA_21" in prices.columns:
            prices = prices.with_columns(
                (pl.col("SMA_7") - pl.col("SMA_21")).alias("SMA_cross_7_21")
            )

        return prices

    def _create_volume_features(self, prices: pl.DataFrame, windows):
        """
        Volume-based features: VWAP, OBV, AD line, MFI, rolling volume stats.
        """
        volume_features = []

        # VWAP
        vwap = (prices['Close'] * prices['Volume']).cum_sum() / prices['Volume'].cum_sum()

        # OBV
        obv = ta.OBV(prices['Close'], prices['Volume'])

        # ADL
        adl = ta.AD(prices['High'], prices['Low'], prices['Close'], prices['Volume'])

        volume_features.extend([
            pl.Series('VWAP', vwap),
            pl.Series('OBV', obv),
            pl.Series('ADL', adl)
        ])

        for w in windows:
            # MFI
            mfi = ta.MFI(prices['High'], prices['Low'], prices['Close'], prices['Volume'], timeperiod=w)

            rolling_vol_mean = prices['Volume'].rolling_mean(window_size=w)
            rolling_vol_std = prices['Volume'].rolling_std(window_size=w)

            volume_features.extend([
                mfi.alias(f'MFI_{w}'),
                rolling_vol_mean.alias(f'mean_volume_{w}'),
                rolling_vol_std.alias(f'std_volume_{w}')
            ])

        prices = prices.with_columns(volume_features)
        return prices

    def _create_time_features(self, prices: pl.DataFrame):
        """
        Create cyclical/time-of-day features plus session flags.
        Removes intermediate hour/minute columns after calculations.
        """
        # Extract raw time info
        prices = prices.with_columns(
            pl.col("Date").dt.hour().alias("hour"),
            pl.col("Date").dt.minute().alias("minute"),
            pl.col("Date").dt.weekday().alias("day_of_week")
        )

        # fractional hour
        prices = prices.with_columns(
            (pl.col("hour") + pl.col("minute") / 60.0).alias("fractional_hour")
        )

        # cyclical hour
        prices = prices.with_columns(
            ((pl.col("fractional_hour") * 2.0 * np.pi / 24.0).sin()).alias("hour_sin"),
            ((pl.col("fractional_hour") * 2.0 * np.pi / 24.0).cos()).alias("hour_cos")
        )

        # cyclical day
        prices = prices.with_columns(
            (
                ((pl.col("day_of_week") + pl.col("fractional_hour") / 24.0) 
                 * 2.0 * np.pi / 7.0).sin()
            ).alias("day_sin"),
            (
                ((pl.col("day_of_week") + pl.col("fractional_hour") / 24.0)
                 * 2.0 * np.pi / 7.0).cos()
            ).alias("day_cos")
        )

        # session flags
        prices = prices.with_columns(
            ((pl.col("hour") >= 9) & (pl.col("hour") <= 16)).cast(pl.Int8).alias("us_session"),
            ((pl.col("hour") >= 1) & (pl.col("hour") <= 8)).cast(pl.Int8).alias("asia_session"),
            ((pl.col("hour") >= 7) & (pl.col("hour") <= 15)).cast(pl.Int8).alias("eu_session")
        )

        # weekend
        prices = prices.with_columns(
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("weekend")
        )

        # Drop temp columns
        prices = prices.drop(["hour", "minute", "fractional_hour", "day_of_week"])
        return prices