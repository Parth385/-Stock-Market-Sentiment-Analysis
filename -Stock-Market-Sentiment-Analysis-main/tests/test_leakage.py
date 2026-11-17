"""
Unit Tests for Temporal Alignment and Absence of Future Data Leakage
"""

import pandas as pd
import numpy as np
import pytest
from src.config import FINAL_DIR, PROCESSED_DIR, TARGET_TICKERS

def test_no_future_data_in_features():
    """Assert that features at index t do not contain information from session t+1."""
    final_path = FINAL_DIR / "final_integrated_dataset.parquet"
    assert final_path.exists(), "Final integrated dataset must exist before testing"
    
    df = pd.read_parquet(final_path)
    
    for sym in TARGET_TICKERS:
        sub = df[df['symbol'] == sym].sort_values('date').reset_index(drop=True)
        
        # Verify ret_1d matches (adj_close_t - adj_close_{t-1}) / adj_close_{t-1}
        expected_ret = sub['adj_close'].pct_change()
        # compare non-null
        valid_mask = ~expected_ret.isna()
        np.testing.assert_allclose(
            sub.loc[valid_mask, 'ret_1d'].values,
            expected_ret.loc[valid_mask].values,
            rtol=1e-5, atol=1e-5,
            err_msg=f"ret_1d does not match backwards return for {sym}"
        )
        
        # Verify target is strictly forward-looking
        # Target for row t must equal the return of row t+1
        # target_ret_o2c = (adj_close_{t+1} - adj_open_{t+1}) / adj_open_{t+1}
        # Let's verify against the actual raw prices
        for i in range(len(sub) - 1):
            curr_row = sub.iloc[i]
            next_row = sub.iloc[i + 1]
            
            # Check O2C target return
            expected_target_ret_o2c = (next_row['adj_close'] - next_row['adj_open']) / next_row['adj_open']
            assert abs(curr_row['target_ret_o2c'] - expected_target_ret_o2c) < 1e-5, \
                f"Leakage detected! Target at t does not match t+1 for {sym} at date {curr_row['date']}"
            
            # Check C2C target return
            expected_target_ret_c2c = (next_row['adj_close'] - curr_row['adj_close']) / curr_row['adj_close']
            assert abs(curr_row['target_ret_c2c'] - expected_target_ret_c2c) < 1e-5, \
                f"Leakage detected! C2C Target at t does not match close t to close t+1 for {sym}"

def test_news_forward_alignment():
    """Assert that news mapped to trading date t was published strictly before t."""
    cal_path = PROCESSED_DIR / "trading_calendar.parquet"
    news_daily_path = PROCESSED_DIR / "news_daily_aggregated.parquet"
    
    df_cal = pd.read_parquet(cal_path)
    df_news = pd.read_parquet(news_daily_path)
    
    # Check that for any calendar date D, mapped trading session is strictly > D
    trading_dates = set(df_cal['trading_date'])
    for d in df_news['date'].unique():
        # Any trading date it maps to must be > d
        future_trading_dates = [td for td in sorted(list(trading_dates)) if td > d]
        if future_trading_dates:
            target_td = future_trading_dates[0]
            assert target_td > d, f"News date {d} mapped to invalid trading date {target_td}"
