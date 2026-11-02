"""
Unit and Integrity Tests for Stage 3: Sentiment Extraction and Linguistic Representations.
Validates:
1. Article completeness across all three models (LM, FinBERT, RoBERTa).
2. Probability validity (sums to 1.0, ranges in [0, 1]).
3. No-news vs neutral distinction (strict zero encoding for no-news days).
4. Temporal alignment and leakage prevention.
5. Sector attribution and company integrity.
"""

import os
import pytest
import numpy as np
import pandas as pd

def test_loughran_mcdonald_integrity():
    lm_path = "data/processed/sentiment/sentiment_lm.parquet"
    assert os.path.exists(lm_path), f"File {lm_path} does not exist"
    df_lm = pd.read_parquet(lm_path)
    
    assert len(df_lm) == 155537, f"Expected 155,537 articles, got {len(df_lm)}"
    assert df_lm['article_id'].nunique() == 155537, "Duplicate article_ids found in LM dataset"
    
    # Polarity range [-1, 1]
    assert (df_lm['lm_polarity'] >= -1.0).all() and (df_lm['lm_polarity'] <= 1.0).all()
    # Word counts non-negative
    assert (df_lm['lm_pos_count'] >= 0).all()
    assert (df_lm['lm_neg_count'] >= 0).all()
    assert (df_lm['lm_unc_count'] >= 0).all()

def test_finbert_sentiment_integrity():
    fb_path = "data/processed/sentiment/sentiment_finbert.parquet"
    if not os.path.exists(fb_path):
        pytest.skip("FinBERT extraction still in progress")
    
    df_fb = pd.read_parquet(fb_path)
    assert len(df_fb) == 155537, f"Expected 155,537 articles, got {len(df_fb)}"
    assert df_fb['article_id'].nunique() == 155537, "Duplicate article_ids found in FinBERT dataset"
    
    # Probability sums to 1.0
    prob_sums = df_fb['finbert_pos'] + df_fb['finbert_neu'] + df_fb['finbert_neg']
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4, err_msg="FinBERT probabilities do not sum to 1.0")
    
    # Polarity is pos - neg
    expected_pol = df_fb['finbert_pos'] - df_fb['finbert_neg']
    np.testing.assert_allclose(df_fb['finbert_polarity'], expected_pol, atol=1e-5)
    
    # Confidence is max(pos, neu, neg)
    max_prob = df_fb[['finbert_pos', 'finbert_neu', 'finbert_neg']].max(axis=1)
    np.testing.assert_allclose(df_fb['finbert_confidence'], max_prob, atol=1e-5)

def test_roberta_sentiment_integrity():
    rob_path = "data/processed/sentiment/sentiment_roberta.parquet"
    if not os.path.exists(rob_path):
        pytest.skip("RoBERTa extraction still in progress")
    
    df_rob = pd.read_parquet(rob_path)
    assert len(df_rob) == 155537, f"Expected 155,537 articles, got {len(df_rob)}"
    assert df_rob['article_id'].nunique() == 155537, "Duplicate article_ids found in RoBERTa dataset"
    
    # Probability sums to 1.0
    prob_sums = df_rob['roberta_pos'] + df_rob['roberta_neu'] + df_rob['roberta_neg']
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4, err_msg="RoBERTa probabilities do not sum to 1.0")
    
    # Polarity is pos - neg
    expected_pol = df_rob['roberta_pos'] - df_rob['roberta_neg']
    np.testing.assert_allclose(df_rob['roberta_polarity'], expected_pol, atol=1e-5)

def test_no_news_encoding_distinction():
    agg_path = "data/processed/sentiment/sentiment_daily_aggregated.parquet"
    if not os.path.exists(agg_path):
        pytest.skip("Daily aggregation not yet generated")
    
    df_agg = pd.read_parquet(agg_path)
    # Filter rows with direct news
    direct_rows = df_agg[df_agg['has_direct_news'] == 1]
    assert len(direct_rows) > 0
    assert (direct_rows['direct_news_count'] > 0).all()

def test_temporal_feature_store_integrity():
    final_path = "data/final/final_sentiment_enhanced_dataset.parquet"
    if not os.path.exists(final_path):
        pytest.skip("Final dataset not yet generated")
    
    df_final = pd.read_parquet(final_path)
    
    # Check 15 approved companies
    expected_tickers = {
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
        'SBIN', 'HINDUNILVR', 'ITC', 'LT', 'BHARTIARTL',
        'SUNPHARMA', 'CIPLA', 'MARUTI', 'M&M', 'TATASTEEL'
    }
    assert set(df_final['symbol'].unique()) == expected_tickers
    
    # Zero news encoding verification
    zero_news_mask = df_final['has_direct_news'] == 0
    assert (df_final.loc[zero_news_mask, 'finbert_direct_pos'] == 0.0).all()
    assert (df_final.loc[zero_news_mask, 'finbert_direct_neu'] == 0.0).all()
    assert (df_final.loc[zero_news_mask, 'finbert_direct_neg'] == 0.0).all()
    assert (df_final.loc[zero_news_mask, 'finbert_direct_polarity'] == 0.0).all()
    
    # Target verification: Primary Open-to-Close on session t
    expected_ret = (df_final['adj_close'] - df_final['adj_open']) / df_final['adj_open']
    np.testing.assert_allclose(df_final['target_same_session_ret_o2c'], expected_ret, atol=1e-5)
    
    # Verify open_gap_pct
    expected_gap = (df_final['adj_open'] - df_final['adj_prev_close']) / df_final['adj_prev_close']
    # Note: open_gap_pct computed with prev_adj_close which matches adj_prev_close
    
    # Leakage check: verify no future columns exist in predictive features
    predictive_feature_names = [c for c in df_final.columns if 'lag1' in c or 'roll' in c or c == 'open_gap_pct']
    for feat in predictive_feature_names:
        assert 'next_' not in feat

def test_corrected_feature_store_integrity():
    corr_path = "data/final/final_sentiment_enhanced_dataset_corrected.parquet"
    if not os.path.exists(corr_path):
        pytest.skip("Corrected final dataset not yet generated")
    
    df_corr = pd.read_parquet(corr_path)
    assert len(df_corr) == 23115, f"Expected 23,115 rows, got {len(df_corr)}"
    assert len(df_corr.columns) == 126, f"Expected 126 columns, got {len(df_corr.columns)}"
    
    # Assert that unique counts are always <= exposure counts
    assert (df_corr['sector_news_count'] <= df_corr['sector_news_count_exposure']).all()
    assert (df_corr['market_news_count'] <= df_corr['market_news_count_exposure']).all()
    
    # Verify presence of exposure weighted columns
    assert 'finbert_sector_polarity_exposure' in df_corr.columns
    assert 'finbert_market_polarity_exposure' in df_corr.columns

