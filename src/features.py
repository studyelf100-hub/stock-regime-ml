"""
features.py
-----------
Feature engineering for market regime classification.
Computes technical indicators from raw OHLCV price data.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    print(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    print(f"  Got {len(df)} trading days.")
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — measures momentum."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index — measures trend strength.
    High ADX = strong trend (either up or down).
    Low ADX = no trend / choppy market.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Zero out where the other is larger
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask = minus_dm < plus_dm
    minus_dm[mask] = 0

    atr = compute_atr(df, period)

    plus_di = 100 * (plus_dm.ewm(com=period - 1, min_periods=period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(com=period - 1, min_periods=period).mean() / (atr + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw OHLCV data.

    Returns a DataFrame with one row per trading day containing
    all features used for regime classification.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]

    # ----- Returns -----
    feat["return_1d"] = np.log(close / close.shift(1))
    feat["return_5d"] = np.log(close / close.shift(5))
    feat["return_20d"] = np.log(close / close.shift(20))

    # ----- Momentum -----
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["rsi_zscore"] = (feat["rsi_14"] - feat["rsi_14"].rolling(60).mean()) / (
        feat["rsi_14"].rolling(60).std() + 1e-9
    )

    # ----- Volatility -----
    feat["atr_14"] = compute_atr(df, 14)
    feat["atr_pct"] = feat["atr_14"] / close  # normalized by price
    feat["realized_vol_20"] = feat["return_1d"].rolling(20).std() * np.sqrt(252)

    # ----- Bollinger Bands -----
    bb_window = 20
    bb_mid = close.rolling(bb_window).mean()
    bb_std = close.rolling(bb_window).std()
    feat["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)
    feat["bb_position"] = (close - bb_mid) / (bb_std + 1e-9)  # z-score within bands

    # ----- Trend Strength -----
    feat["adx_14"] = compute_adx(df, 14)
    feat["trend_strength"] = feat["adx_14"] / 100.0  # normalize to [0, 1]

    # ----- Volume -----
    if "volume" in df.columns:
        feat["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    else:
        feat["volume_ratio"] = 1.0

    # ----- Statistical Moments -----
    feat["skew_20"] = feat["return_1d"].rolling(20).skew()
    feat["kurt_20"] = feat["return_1d"].rolling(20).kurt()

    # ----- Price Relative to Moving Averages -----
    feat["price_vs_ma50"] = close / close.rolling(50).mean() - 1
    feat["price_vs_ma200"] = close / close.rolling(200).mean() - 1

    # Drop rows with NaN (from rolling windows at start of series)
    feat.dropna(inplace=True)

    print(f"  Built {len(feat.columns)} features across {len(feat)} trading days.")
    return feat


if __name__ == "__main__":
    df = fetch_data("SPY", "2015-01-01", "2024-01-01")
    features = build_features(df)
    print("\nFeature sample:")
    print(features.tail())
    print("\nFeature stats:")
    print(features.describe().round(3))
