"""
Unit Tests for News Mapping, Non-Substitution, and No-News Indicators
"""

import pandas as pd
from src.config import PROCESSED_DIR, FINAL_DIR

def test_no_news_indicators():
    """Verify that has_direct_news and has_sector_news correctly match counts."""
    df = pd.read_parquet(FINAL_DIR / "final_integrated_dataset.parquet")
    
    # has_direct_news must be 1 if direct_news_count > 0 else 0
    expected_direct_flag = (df['direct_news_count'] > 0).astype(int)
    assert (df['has_direct_news'] == expected_direct_flag).all(), "Mismatch in has_direct_news indicator!"
    
    expected_sector_flag = (df['sectoral_news_count'] > 0).astype(int)
    assert (df['has_sector_news'] == expected_sector_flag).all(), "Mismatch in has_sector_news indicator!"

def test_non_substitution_policy():
    """Verify that missing direct news is NOT substituted with sector or market sentiment."""
    df = pd.read_parquet(FINAL_DIR / "final_integrated_dataset.parquet")
    
    zero_direct = df[df['has_direct_news'] == 0]
    # For all zero direct days, direct polarity must be exactly 0.0
    assert (zero_direct['direct_finbert_polarity'] == 0.0).all(), "Violation: Direct polarity non-zero on zero-direct days!"
    assert (zero_direct['direct_news_pos'] == 0.0).all(), "Violation: Direct pos non-zero on zero-direct days!"
    assert (zero_direct['direct_news_neg'] == 0.0).all(), "Violation: Direct neg non-zero on zero-direct days!"
    
    # Sector polarity should NOT be zero everywhere (proving they are independent)
    assert (df['sectoral_finbert_polarity'] != 0.0).any(), "Sector polarity is unexpectedly all zero!"
    assert (df['global_finbert_polarity'] != 0.0).any(), "Global polarity is unexpectedly all zero!"
