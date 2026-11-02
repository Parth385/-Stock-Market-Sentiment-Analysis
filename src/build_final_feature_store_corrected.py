"""
Construct Master Sentiment-Enhanced Feature Store (Corrected Version).
Merges market indicators with corrected temporal sentiment features:
- Unique-article sector & market sentiment features (primary)
- Explicitly labeled exposure-weighted sector & market features (secondary)
- Preserves original final_sentiment_enhanced_dataset.parquet and final_integrated_dataset.parquet without overwriting.

Output: data/final/final_sentiment_enhanced_dataset_corrected.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from src.config import TARGET_TICKERS, COMPANIES

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def main():
    mkt_raw_path = "data/processed/market_raw_ohlcv.parquet"
    mkt_adj_path = "data/processed/market_adjusted_ohlcv.parquet"
    sentiment_path = "data/processed/sentiment/sentiment_temporal_features_corrected.parquet"
    output_path = "data/final/final_sentiment_enhanced_dataset_corrected.parquet"
    sample_csv_path = "data/final/final_sentiment_corrected_sample_200.csv"

    print("Loading market data...")
    df_raw = pd.read_parquet(mkt_raw_path)
    df_adj = pd.read_parquet(mkt_adj_path)

    df_mkt = df_raw.rename(columns={
        'open': 'raw_open', 'high': 'raw_high', 'low': 'raw_low',
        'close': 'raw_close', 'prev_close': 'raw_prev_close',
        'volume': 'raw_volume', 'turnover': 'raw_turnover', 'trades': 'raw_trades'
    })

    adj_cols = ['date', 'symbol', 'adj_factor_cumulative', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_prev_close']
    df_mkt = pd.merge(df_mkt, df_adj[adj_cols], on=['date', 'symbol'], how='inner')
    df_mkt['sector'] = df_mkt['symbol'].map(lambda s: COMPANIES[s]['sector'])
    df_mkt['date'] = pd.to_datetime(df_mkt['date']).dt.date

    comp_list = []
    for sym in TARGET_TICKERS:
        sub = df_mkt[df_mkt['symbol'] == sym].sort_values('date').copy()

        # Session t Close indicators (for next-session robustness benchmark)
        sub['ret_1d_t0'] = sub['adj_close'].pct_change()
        sub['rsi_14_t0'] = compute_rsi(sub['adj_close'], period=14)
        macd, sig = compute_macd(sub['adj_close'])
        sub['macd_t0'] = macd
        sub['macd_signal_t0'] = sig
        sub['macd_diff_t0'] = macd - sig
        sub['rolling_vol_5d_t0'] = sub['ret_1d_t0'].rolling(5, min_periods=3).std().fillna(0.0)
        sub['rolling_vol_21d_t0'] = sub['ret_1d_t0'].rolling(21, min_periods=10).std().fillna(0.0)
        sma_20 = sub['adj_close'].rolling(20, min_periods=10).mean()
        sub['adj_close_sma_ratio_20_t0'] = ((sub['adj_close'] / sma_20) - 1.0).fillna(0.0)
        sub['intraday_range_pct_t0'] = ((sub['raw_high'] - sub['raw_low']) / sub['raw_close']).fillna(0.0)
        vol_med = sub['rolling_vol_21d_t0'].median()
        sub['market_regime_t0'] = np.where(sub['rolling_vol_21d_t0'] >= vol_med, 'HIGH_VOL', 'LOW_VOL')

        # Session t Open indicators (available at market open)
        sub['prev_adj_close'] = sub['adj_close'].shift(1)
        sub['open_gap_pct'] = ((sub['adj_open'] - sub['prev_adj_close']) / sub['prev_adj_close']).fillna(0.0)

        # Lag-1 indicators (strictly available prior to session t open)
        sub['ret_1d_lag1'] = sub['ret_1d_t0'].shift(1).fillna(0.0)
        sub['rsi_14_lag1'] = sub['rsi_14_t0'].shift(1).fillna(50.0)
        sub['macd_lag1'] = sub['macd_t0'].shift(1).fillna(0.0)
        sub['macd_signal_lag1'] = sub['macd_signal_t0'].shift(1).fillna(0.0)
        sub['macd_diff_lag1'] = sub['macd_diff_t0'].shift(1).fillna(0.0)
        sub['rolling_vol_5d_lag1'] = sub['rolling_vol_5d_t0'].shift(1).fillna(0.0)
        sub['rolling_vol_21d_lag1'] = sub['rolling_vol_21d_t0'].shift(1).fillna(0.0)
        sub['adj_close_sma_ratio_20_lag1'] = sub['adj_close_sma_ratio_20_t0'].shift(1).fillna(0.0)
        sub['intraday_range_pct_lag1'] = sub['intraday_range_pct_t0'].shift(1).fillna(0.0)
        sub['market_regime_lag1'] = sub['market_regime_t0'].shift(1).fillna('LOW_VOL')

        # TARGETS
        # 1. Primary Target: Intraday Open-to-Close on session t
        sub['target_same_session_ret_o2c'] = (sub['adj_close'] - sub['adj_open']) / sub['adj_open']
        sub['target_same_session_dir_o2c'] = (sub['target_same_session_ret_o2c'] > 0).astype(int)

        # 2. Robustness Target: Next-session Open-to-Close on session t+1
        sub['next_adj_open'] = sub['adj_open'].shift(-1)
        sub['next_adj_close'] = sub['adj_close'].shift(-1)
        sub['target_next_session_ret_o2c'] = (sub['next_adj_close'] - sub['next_adj_open']) / sub['next_adj_open']
        sub['target_next_session_dir_o2c'] = (sub['target_next_session_ret_o2c'] > 0).astype(int)

        # 3. Robustness Target: Next-session Close-to-Close on session t+1
        sub['target_next_session_ret_c2c'] = (sub['next_adj_close'] - sub['adj_close']) / sub['adj_close']
        sub['target_next_session_dir_c2c'] = (sub['target_next_session_ret_c2c'] > 0).astype(int)

        comp_list.append(sub)

    df_mkt_all = pd.concat(comp_list, ignore_index=True)

    print("Loading corrected temporal sentiment features...")
    df_sent = pd.read_parquet(sentiment_path)
    df_sent['date'] = pd.to_datetime(df_sent['date']).dt.date

    print("Merging market panel with corrected sentiment feature store...")
    df_final = df_mkt_all.merge(df_sent, on=['date', 'symbol', 'sector'], how='inner')

    # Drop terminal day 2026-03-30
    df_final = df_final.dropna(subset=['target_next_session_ret_o2c']).copy()
    df_final = df_final.drop(columns=['prev_adj_close', 'next_adj_open', 'next_adj_close'])
    df_final = df_final.sort_values(['symbol', 'date']).reset_index(drop=True)

    print(f"Saving corrected final sentiment dataset to {output_path} ({len(df_final):,} rows × {len(df_final.columns)} cols)...")
    df_final.to_parquet(output_path, index=False)
    df_final.head(200).to_csv(sample_csv_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
