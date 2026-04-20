"""
tests/test_features.py
----------------------
Basic unit tests for feature engineering and utilities.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest
from features import compute_rsi, compute_atr, build_features


def make_dummy_df(n=300):
    """Create a small fake OHLCV dataframe for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    high = close * (1 + np.random.uniform(0, 0.01, n))
    low = close * (1 - np.random.uniform(0, 0.01, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({"close": close, "high": high, "low": low,
                          "open": close, "volume": volume}, index=dates)


def test_rsi_range():
    """RSI should always be between 0 and 100."""
    df = make_dummy_df()
    rsi = compute_rsi(df["close"])
    rsi_valid = rsi.dropna()
    assert (rsi_valid >= 0).all(), "RSI below 0 found"
    assert (rsi_valid <= 100).all(), "RSI above 100 found"


def test_atr_positive():
    """ATR should always be non-negative."""
    df = make_dummy_df()
    atr = compute_atr(df)
    assert (atr.dropna() >= 0).all(), "Negative ATR found"


def test_build_features_shape():
    """build_features should return a DataFrame with expected columns."""
    df = make_dummy_df(400)
    features = build_features(df)

    expected_cols = [
        "return_1d", "return_5d", "rsi_14", "atr_pct",
        "bb_width", "adx_14", "trend_strength", "volume_ratio",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_no_lookahead_in_features():
    """
    Rolling features should not use future data.
    We test by checking that the last row can be computed
    without knowing tomorrow's price.
    """
    df = make_dummy_df(400)
    features = build_features(df)
    # If we trim 1 day and recompute, last row should be unchanged
    features_trimmed = build_features(df.iloc[:-1])
    shared_idx = features.index.intersection(features_trimmed.index)
    last_common = shared_idx[-1]
    pd.testing.assert_series_equal(
        features.loc[last_common],
        features_trimmed.loc[last_common],
        check_names=False,
    )


def test_rsi_oversold():
    """RSI should be low after a strong downward trend."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Price drops every day for 20 days at the end
    close = np.ones(n) * 100
    close[-20:] = np.linspace(100, 60, 20)
    series = pd.Series(close, index=dates)
    rsi = compute_rsi(series, 14).dropna()
    assert rsi.iloc[-1] < 40, "RSI should be in oversold territory after strong decline"
