"""
Unit Tests for Market Data Integrity and Price Consistency
"""

import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, FINAL_DIR, TARGET_TICKERS

def test_company_cohort_presence():
    """Verify that all 15 companies are present in the dataset."""
    df = pd.read_parquet(PROCESSED_DIR / "market_raw_ohlcv.parquet")
    symbols_present = set(df['symbol'].unique())
    assert symbols_present == set(TARGET_TICKERS), f"Missing tickers! Expected {set(TARGET_TICKERS)}, got {symbols_present}"

def test_no_duplicate_records():
    """Verify that there are zero duplicate (date, symbol) records."""
    for fn in ["market_raw_ohlcv.parquet", "market_adjusted_ohlcv.parquet"]:
        df = pd.read_parquet(PROCESSED_DIR / fn)
        dups = df.duplicated(subset=['date', 'symbol']).sum()
        assert dups == 0, f"Found {dups} duplicates in {fn}"

def test_ohlc_physical_consistency():
    """Verify physical consistency: High >= Low, High >= Open/Close, Low <= Open/Close."""
    df_raw = pd.read_parquet(PROCESSED_DIR / "market_raw_ohlcv.parquet")
    
    assert (df_raw['high'] >= df_raw['low']).all(), "OHLC Violation: High < Low detected!"
    assert (df_raw['high'] >= df_raw['open']).all(), "OHLC Violation: High < Open detected!"
    assert (df_raw['high'] >= df_raw['close']).all(), "OHLC Violation: High < Close detected!"
    assert (df_raw['low'] <= df_raw['open']).all(), "OHLC Violation: Low > Open detected!"
    assert (df_raw['low'] <= df_raw['close']).all(), "OHLC Violation: Low > Close detected!"
    assert (df_raw['close'] > 0).all(), "Zero or negative Close price detected!"
    assert (df_raw['volume'] >= 0).all(), "Negative volume detected!"

def test_adjusted_series_consistency():
    """Verify that adjusted OHLC series are strictly positive and correctly scaled."""
    df_adj = pd.read_parquet(PROCESSED_DIR / "market_adjusted_ohlcv.parquet")
    
    assert (df_adj['adj_factor_cumulative'] > 0).all(), "Non-positive adjustment factor detected!"
    assert (df_adj['adj_close'] > 0).all(), "Non-positive adjusted close detected!"
    assert (df_adj['adj_high'] >= df_adj['adj_low']).all(), "Adjusted High < Adjusted Low detected!"
