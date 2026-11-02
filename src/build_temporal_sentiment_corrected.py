"""
Build Corrected Temporal Sentiment Features.
Carries through both:
- Unique-article sector & market sentiment features (primary)
- Exposure-weighted sector & market sentiment features (secondary)
Across same-session (t), lag-1 (t-1), rolling-3 (t, t-1, t-2), and rolling-7 (t ... t-6).

Output: data/processed/sentiment/sentiment_temporal_features_corrected.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from src.config import TARGET_TICKERS, COMPANIES

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def build_forward_calendar_mapping(trading_dates, min_date, max_date):
    trading_dates = sorted(pd.to_datetime(trading_dates).date.tolist())
    all_cal_dates = pd.date_range(min_date, max_date).date
    
    mapping = {}
    for cd in all_cal_dates:
        future_dates = [td for td in trading_dates if td > cd]
        mapping[cd] = future_dates[0] if future_dates else None
    return mapping

def main():
    daily_path = "data/processed/sentiment/sentiment_daily_aggregated_corrected.parquet"
    calendar_path = "data/processed/trading_calendar.parquet"
    output_path = "data/processed/sentiment/sentiment_temporal_features_corrected.parquet"

    print(f"Loading corrected daily sentiment from {daily_path}...")
    df_daily = pd.read_parquet(daily_path)
    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.date

    print(f"Loading trading calendar from {calendar_path}...")
    df_cal = pd.read_parquet(calendar_path)
    df_cal['trading_date'] = pd.to_datetime(df_cal['trading_date']).dt.date
    trading_dates = sorted(df_cal['trading_date'].unique())

    min_date = df_daily['date'].min()
    max_date = df_daily['date'].max()
    cal_map = build_forward_calendar_mapping(trading_dates, min_date, max_date)

    df_daily['forward_trading_date'] = df_daily['date'].map(cal_map)
    df_daily = df_daily.dropna(subset=['forward_trading_date']).copy()

    agg_dict = {
        # Direct
        'direct_news_count': 'sum',
        'finbert_direct_pos': 'mean',
        'finbert_direct_neu': 'mean',
        'finbert_direct_neg': 'mean',
        'finbert_direct_polarity': 'mean',
        'finbert_direct_confidence': 'mean',
        'roberta_direct_pos': 'mean',
        'roberta_direct_neu': 'mean',
        'roberta_direct_neg': 'mean',
        'roberta_direct_polarity': 'mean',
        'roberta_direct_confidence': 'mean',
        'lm_direct_pos_count': 'sum',
        'lm_direct_neg_count': 'sum',
        'lm_direct_unc_count': 'sum',
        'lm_direct_polarity': 'mean',
        'lm_direct_subjectivity': 'mean',
        # Unique Sector
        'sector_news_count': 'sum',
        'finbert_sector_pos': 'mean',
        'finbert_sector_neu': 'mean',
        'finbert_sector_neg': 'mean',
        'finbert_sector_polarity': 'mean',
        'roberta_sector_pos': 'mean',
        'roberta_sector_neu': 'mean',
        'roberta_sector_neg': 'mean',
        'roberta_sector_polarity': 'mean',
        'lm_sector_pos_count': 'sum',
        'lm_sector_neg_count': 'sum',
        'lm_sector_polarity': 'mean',
        # Exposure-Weighted Sector
        'sector_news_count_exposure': 'sum',
        'finbert_sector_polarity_exposure': 'mean',
        'roberta_sector_polarity_exposure': 'mean',
        'lm_sector_polarity_exposure': 'mean',
        # Unique Market
        'market_news_count': 'sum',
        'finbert_market_pos': 'mean',
        'finbert_market_neu': 'mean',
        'finbert_market_neg': 'mean',
        'finbert_market_polarity': 'mean',
        'roberta_market_pos': 'mean',
        'roberta_market_neu': 'mean',
        'roberta_market_neg': 'mean',
        'roberta_market_polarity': 'mean',
        'lm_market_polarity': 'mean',
        # Exposure-Weighted Market
        'market_news_count_exposure': 'sum',
        'finbert_market_polarity_exposure': 'mean',
        'roberta_market_polarity_exposure': 'mean',
        'lm_market_polarity_exposure': 'mean'
    }

    df_forward = df_daily.groupby(['forward_trading_date', 'symbol', 'sector']).agg(agg_dict).reset_index()
    df_forward = df_forward.rename(columns={'forward_trading_date': 'date'})

    # Full grid of (trading_date, symbol)
    grid_rows = []
    for td in trading_dates:
        for sym in TARGET_TICKERS:
            grid_rows.append({
                'date': td,
                'symbol': sym,
                'sector': COMPANIES[sym]['sector']
            })
    df_grid = pd.DataFrame(grid_rows)

    df_panel = df_grid.merge(df_forward, on=['date', 'symbol', 'sector'], how='left')

    df_panel['has_direct_news'] = (df_panel['direct_news_count'].fillna(0) > 0).astype(int)
    df_panel['has_sector_news'] = (df_panel['sector_news_count'].fillna(0) > 0).astype(int)
    df_panel['direct_news_count'] = df_panel['direct_news_count'].fillna(0).astype(int)
    df_panel['sector_news_count'] = df_panel['sector_news_count'].fillna(0).astype(int)
    df_panel['sector_news_count_exposure'] = df_panel['sector_news_count_exposure'].fillna(0).astype(int)
    df_panel['market_news_count'] = df_panel['market_news_count'].fillna(0).astype(int)
    df_panel['market_news_count_exposure'] = df_panel['market_news_count_exposure'].fillna(0).astype(int)

    sentiment_cols = [c for c in agg_dict.keys() if 'count' not in c]
    for sc in sentiment_cols:
        df_panel[sc] = df_panel[sc].fillna(0.0)

    # Compute temporal features per company
    company_list = []
    for sym in TARGET_TICKERS:
        sub = df_panel[df_panel['symbol'] == sym].sort_values('date').copy()

        # Direct
        sub['finbert_direct_polarity_lag1'] = sub['finbert_direct_polarity'].shift(1).fillna(0.0)
        sub['roberta_direct_polarity_lag1'] = sub['roberta_direct_polarity'].shift(1).fillna(0.0)
        sub['lm_direct_polarity_lag1'] = sub['lm_direct_polarity'].shift(1).fillna(0.0)
        sub['direct_news_count_lag1'] = sub['direct_news_count'].shift(1).fillna(0).astype(int)

        sub['finbert_direct_polarity_roll3'] = sub['finbert_direct_polarity'].rolling(3, min_periods=1).mean()
        sub['roberta_direct_polarity_roll3'] = sub['roberta_direct_polarity'].rolling(3, min_periods=1).mean()
        sub['lm_direct_polarity_roll3'] = sub['lm_direct_polarity'].rolling(3, min_periods=1).mean()
        sub['direct_news_count_roll3'] = sub['direct_news_count'].rolling(3, min_periods=1).sum()

        sub['finbert_direct_polarity_roll7'] = sub['finbert_direct_polarity'].rolling(7, min_periods=1).mean()
        sub['roberta_direct_polarity_roll7'] = sub['roberta_direct_polarity'].rolling(7, min_periods=1).mean()
        sub['lm_direct_polarity_roll7'] = sub['lm_direct_polarity'].rolling(7, min_periods=1).mean()
        sub['direct_news_count_roll7'] = sub['direct_news_count'].rolling(7, min_periods=1).sum()

        # Unique Sector
        sub['finbert_sector_polarity_lag1'] = sub['finbert_sector_polarity'].shift(1).fillna(0.0)
        sub['roberta_sector_polarity_lag1'] = sub['roberta_sector_polarity'].shift(1).fillna(0.0)
        sub['lm_sector_polarity_lag1'] = sub['lm_sector_polarity'].shift(1).fillna(0.0)

        sub['finbert_sector_polarity_roll3'] = sub['finbert_sector_polarity'].rolling(3, min_periods=1).mean()
        sub['roberta_sector_polarity_roll3'] = sub['roberta_sector_polarity'].rolling(3, min_periods=1).mean()
        sub['lm_sector_polarity_roll3'] = sub['lm_sector_polarity'].rolling(3, min_periods=1).mean()

        sub['finbert_sector_polarity_roll7'] = sub['finbert_sector_polarity'].rolling(7, min_periods=1).mean()
        sub['roberta_sector_polarity_roll7'] = sub['roberta_sector_polarity'].rolling(7, min_periods=1).mean()
        sub['lm_sector_polarity_roll7'] = sub['lm_sector_polarity'].rolling(7, min_periods=1).mean()

        # Unique Market
        sub['finbert_market_polarity_lag1'] = sub['finbert_market_polarity'].shift(1).fillna(0.0)
        sub['roberta_market_polarity_lag1'] = sub['roberta_market_polarity'].shift(1).fillna(0.0)
        sub['lm_market_polarity_lag1'] = sub['lm_market_polarity'].shift(1).fillna(0.0)

        sub['finbert_market_polarity_roll3'] = sub['finbert_market_polarity'].rolling(3, min_periods=1).mean()
        sub['roberta_market_polarity_roll3'] = sub['roberta_market_polarity'].rolling(3, min_periods=1).mean()
        sub['lm_market_polarity_roll3'] = sub['lm_market_polarity'].rolling(3, min_periods=1).mean()

        sub['finbert_market_polarity_roll7'] = sub['finbert_market_polarity'].rolling(7, min_periods=1).mean()
        sub['roberta_market_polarity_roll7'] = sub['roberta_market_polarity'].rolling(7, min_periods=1).mean()
        sub['lm_market_polarity_roll7'] = sub['lm_market_polarity'].rolling(7, min_periods=1).mean()

        company_list.append(sub)

    df_temporal = pd.concat(company_list, ignore_index=True).sort_values(['symbol', 'date']).reset_index(drop=True)

    print(f"Saving corrected temporal sentiment features to {output_path} ({len(df_temporal):,} rows × {len(df_temporal.columns)} cols)...")
    df_temporal.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
