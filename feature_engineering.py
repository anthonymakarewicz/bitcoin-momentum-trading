import numpy as np
import polars as pl
import talib as ta
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[7, 14, 21, 50, 100], drop_original=True):
        self.windows = windows
        self.drop_original = drop_original
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.clone()
        orig_cols = df.columns

        df = self._create_return_features(df, self.windows)
        df = self._create_technical_indicators_features(df, self.windows)
        df = self._create_volume_features(df, self.windows)
        df = self._create_time_features(df)

        df = df.drop_nans().drop_nulls()

        if self.drop_original:
            keep_cols = set(df.columns) - set(["Open", "High", "Low", "Close", "Volume"])
            df = df.select(list(keep_cols))

        return df
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _create_return_features(self, prices: pl.DataFrame, windows):
        # Log returns
        prices = prices.with_columns(
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias('log_returns')
        )

        # Rolling mean/std of log returns
        features = []
        for window in windows:
            features.extend([
                pl.col('log_returns').rolling_mean(window_size=window).alias(f'mean_log_returns_{window}'),
                pl.col('log_returns').rolling_std(window_size=window).alias(f'std_log_returns_{window}'),
            ])
        
        prices = prices.with_columns(features)
        return prices

    def _create_technical_indicators_features(self, prices: pl.DataFrame, windows):
        close = prices["Close"]
        high = prices['High']
        low = prices['Low']

        # Moving Average Convergence Divergence
        macd, macd_signal, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicator_columns = [
            pl.Series('MACD', macd),
            pl.Series('MACD_signal', macd_signal)
        ]

        for window in windows:
            # Simple and Exponential Moving Averages
            sma = ta.SMA(close, timeperiod=window)
            ema = ta.EMA(close, timeperiod=window)

            # Bollinger Bands
            bb_upper, _, bb_lower = ta.BBANDS(close, timeperiod=window, nbdevup=2, nbdevdn=2)

            # Relative Strength Index
            rsi = ta.RSI(close, timeperiod=window)

            # Average Directional Index
            adx = ta.ADX(high, low, close, timeperiod=window)

            # Average True Range
            atr = ta.ATR(high, low, close, timeperiod=window)

            # Rate of Change
            roc = ta.ROCP(close, timeperiod=window)

            indicator_columns.extend([
                pl.Series(f'SMA_{window}', sma),
                pl.Series(f'EMA_{window}', ema),
                pl.Series(f'RSI_{window}', rsi),
                pl.Series(f'ADX_{window}', adx),
                pl.Series(f'ATR_{window}', atr),
                pl.Series(f'ROC_{window}', roc),
                pl.Series(f'BB_upper_{window}', bb_upper),
                pl.Series(f'BB_lower_{window}', bb_lower)
            ])

        prices = prices.with_columns(indicator_columns)
        return prices

    def _create_volume_features(self, prices: pl.DataFrame, windows):
        """
        Volume-based features: VWAP, OBV, AD line, MFI, rolling volume stats.
        """
        volume_features = []

        # Volume Weighted Average Price
        vwap = (prices['Close'] * prices['Volume']).cum_sum() / prices['Volume'].cum_sum()

        # On Balance Volume
        obv = ta.OBV(prices['Close'], prices['Volume'])

        # Accumulation/Distribution Line
        adl = ta.AD(prices['High'], prices['Low'], prices['Close'], prices['Volume'])

        volume_features.extend([
            pl.Series('VWAP', vwap),
            pl.Series('OBV', obv),
            pl.Series('ADL', adl)
        ])

        for window in windows:
            # Money Flow Index
            mfi = ta.MFI(prices['High'], prices['Low'], prices['Close'], prices['Volume'], timeperiod=window)

            # Mean and standard deviation of volume
            rolling_vol_mean = prices['Volume'].rolling_mean(window_size=window)
            rolling_vol_std = prices['Volume'].rolling_std(window_size=window)

            volume_features.extend([
                mfi.alias(f'MFI_{window}'),
                rolling_vol_mean.alias(f'mean_volume_{window}'),
                rolling_vol_std.alias(f'std_volume_{window}')
            ])

        prices = prices.with_columns(volume_features)
        return prices

    def _create_time_features(self, prices: pl.DataFrame):
        """
        Create cyclical/time-of-day features plus session flags.
        Removes intermediate hour/minute columns after calculations.
        """
        prices = prices.with_columns(
            pl.col("Date").dt.hour().alias("hour"),
            pl.col("Date").dt.minute().alias("minute"),
            pl.col("Date").dt.weekday().alias("day_of_week")
        )

        prices = prices.with_columns(
            (pl.col("hour") + pl.col("minute") / 60.0).alias("fractional_hour")
        )

        # Cyclical hour
        prices = prices.with_columns(
            ((pl.col("fractional_hour") * 2.0 * np.pi / 24.0).sin()).alias("hour_sin"),
            ((pl.col("fractional_hour") * 2.0 * np.pi / 24.0).cos()).alias("hour_cos")
        )

        # Cyclical day
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

        # Session flags
        prices = prices.with_columns(
            ((pl.col("hour") >= 9) & (pl.col("hour") <= 16)).cast(pl.Int8).alias("us_session"),
            ((pl.col("hour") >= 1) & (pl.col("hour") <= 8)).cast(pl.Int8).alias("asia_session"),
            ((pl.col("hour") >= 7) & (pl.col("hour") <= 15)).cast(pl.Int8).alias("eu_session")
        )

        # Weekend flag
        prices = prices.with_columns(
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("weekend")
        )

        prices = prices.drop(["hour", "minute", "fractional_hour", "day_of_week"])
        return prices