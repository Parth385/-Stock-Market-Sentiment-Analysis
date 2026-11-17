"""
Master Integration and Target Construction Pipeline
Executes forward-session news alignment (T -> T+1), rolling weekend/holiday aggregation,
computes technical indicators exclusively on adjusted prices, and generates mathematically
consistent next-session prediction targets (Primary: Open-to-Close; Secondary: Close-to-Close).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src.config import PROCESSED_DIR, FINAL_DIR, TARGET_TICKERS, COMPANIES

def compute_rsi(series, period=14):
    """Compute Relative Strength Index on price series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD and MACD signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def build_forward_calendar_mapping(df_cal, min_date, max_date):
    """
    Build a complete lookup mapping from any calendar date to the next available trading session.
    Rule: tau(D) = min { t in TradingCalendar | t > D }
    """
    trading_dates = sorted(df_cal['trading_date'].tolist())
    all_cal_dates = pd.date_range(min_date, max_date).date
    
    mapping = {}
    for cd in all_cal_dates:
        # Find the smallest trading date strictly greater than cd
        future_trading_dates = [td for td in trading_dates if td > cd]
        if future_trading_dates:
            mapping[cd] = future_trading_dates[0]
        else:
            mapping[cd] = None # Past the final trading session
            
    return mapping

def aggregate_news_to_forward_sessions(df_news_daily, calendar_map):
    """
    Map daily calendar news records to forward trading sessions.
    When multiple days roll forward to the same trading day (e.g. Fri + Sat + Sun -> Mon),
    we aggregate counts (sum) and compute weighted average sentiment polarities.
    """
    print("Mapping news records to forward trading sessions (T -> T+1)...")
    df = df_news_daily.copy()
    df['forward_trading_date'] = df['date'].map(calendar_map)
    
    # Drop news that has no future trading session in calendar
    df = df.dropna(subset=['forward_trading_date']).copy()
    
    # Aggregate by (forward_trading_date, symbol)
    agg_funcs = {
        'direct_news_count': 'sum',
        'sectoral_news_count': 'sum',
        'global_news_count': 'sum',
        'direct_news_pos': 'mean',
        'direct_news_neu': 'mean',
        'direct_news_neg': 'mean',
        'sectoral_news_pos': 'mean',
        'sectoral_news_neu': 'mean',
        'sectoral_news_neg': 'mean',
        'global_news_pos': 'mean',
        'global_news_neu': 'mean',
        'global_news_neg': 'mean',
        'direct_finbert_polarity': 'mean',
        'sectoral_finbert_polarity': 'mean',
        'global_finbert_polarity': 'mean'
    }
    
    if 'extracted_direct_count' in df.columns:
        agg_funcs['extracted_direct_count'] = 'sum'
    if 'aggregated_titles' in df.columns:
        agg_funcs['aggregated_titles'] = lambda titles: ' | '.join([t for t in titles if t][:3])
        
    df_forward = df.groupby(['forward_trading_date', 'symbol']).agg(agg_funcs).reset_index()
    df_forward = df_forward.rename(columns={'forward_trading_date': 'date'})
    
    # Recompute explicit No-News indicator flags
    df_forward['has_direct_news'] = (df_forward['direct_news_count'] > 0).astype(int)
    df_forward['has_sector_news'] = (df_forward['sectoral_news_count'] > 0).astype(int)
    
    # Strictly zero out direct sentiment if has_direct_news == 0
    zero_mask = df_forward['has_direct_news'] == 0
    df_forward.loc[zero_mask, 'direct_finbert_pos'] = 0.0
    df_forward.loc[zero_mask, 'direct_finbert_neu'] = 0.0
    df_forward.loc[zero_mask, 'direct_finbert_neg'] = 0.0
    df_forward.loc[zero_mask, 'direct_finbert_polarity'] = 0.0
    
    print(f"Aggregated {len(df_news_daily)} daily news records into {len(df_forward)} forward-session company records.")
    return df_forward

def integrate_and_build_features():
    """Master function to integrate market and news data, generate indicators and targets."""
    print("=" * 60)
    print("Building Master Integrated Dataset for Beyond Sentiment...")
    print("=" * 60)
    
    # 1. Load clean processed files
    df_raw_mkt = pd.read_parquet(PROCESSED_DIR / "market_raw_ohlcv.parquet")
    df_adj_mkt = pd.read_parquet(PROCESSED_DIR / "market_adjusted_ohlcv.parquet")
    df_cal = pd.read_parquet(PROCESSED_DIR / "trading_calendar.parquet")
    df_news_daily = pd.read_parquet(PROCESSED_DIR / "news_daily_aggregated.parquet")
    
    # 2. Build forward calendar mapping
    min_cal_date = df_news_daily['date'].min()
    max_cal_date = df_news_daily['date'].max()
    cal_map = build_forward_calendar_mapping(df_cal, min_cal_date, max_cal_date)
    
    # 3. Aggregate news forward to trading sessions
    df_news_forward = aggregate_news_to_forward_sessions(df_news_daily, cal_map)
    
    # 4. Merge market data (Raw and Adjusted)
    print("\nMerging raw and adjusted market series...")
    # Merge raw and adjusted columns
    df_mkt = df_raw_mkt.rename(columns={
        'open': 'raw_open', 'high': 'raw_high', 'low': 'raw_low', 
        'close': 'raw_close', 'prev_close': 'raw_prev_close',
        'volume': 'raw_volume', 'turnover': 'raw_turnover', 'trades': 'raw_trades'
    })
    
    adj_cols = ['date', 'symbol', 'adj_factor_cumulative', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_prev_close']
    df_mkt = pd.merge(df_mkt, df_adj_mkt[adj_cols], on=['date', 'symbol'], how='inner')
    
    # Add Sector from config
    df_mkt['sector'] = df_mkt['symbol'].map(lambda s: COMPANIES[s]['sector'])
    
    # 5. Compute technical indicators per company (chronologically sorted)
    print("Computing technical features exclusively on adjusted price series...")
    all_companies = []
    
    for sym in TARGET_TICKERS:
        sub = df_mkt[df_mkt['symbol'] == sym].sort_values('date').copy()
        
        # Returns
        sub['ret_1d'] = sub['adj_close'].pct_change()
        
        # Technical indicators on adj_close
        sub['rsi_14'] = compute_rsi(sub['adj_close'], period=14)
        macd, signal = compute_macd(sub['adj_close'])
        sub['macd'] = macd
        sub['macd_signal'] = signal
        sub['macd_diff'] = macd - signal
        
        # Volatility
        sub['rolling_vol_5d'] = sub['ret_1d'].rolling(5, min_periods=3).std()
        sub['rolling_vol_21d'] = sub['ret_1d'].rolling(21, min_periods=10).std()
        
        # Moving average ratios
        sma_20 = sub['adj_close'].rolling(20, min_periods=10).mean()
        sub['adj_close_sma_ratio_20'] = (sub['adj_close'] / sma_20) - 1.0
        
        # Intraday range on raw prices (scalar cancels out)
        sub['intraday_range_pct'] = (sub['raw_high'] - sub['raw_low']) / sub['raw_close']
        
        # Market Regime indicator (21-day median volatility split)
        vol_median = sub['rolling_vol_21d'].median()
        sub['market_regime'] = np.where(sub['rolling_vol_21d'] >= vol_median, 'HIGH_VOL', 'LOW_VOL')
        
        # Prediction Targets for next trading session t+1
        # Primary Target: Open-to-Close direction on session t+1
        # Next-session return: (Adj Close_{t+1} - Adj Open_{t+1}) / Adj Open_{t+1}
        sub['next_adj_open'] = sub['adj_open'].shift(-1)
        sub['next_adj_close'] = sub['adj_close'].shift(-1)
        
        sub['target_ret_o2c'] = (sub['next_adj_close'] - sub['next_adj_open']) / sub['next_adj_open']
        sub['target_o2c_direction'] = (sub['target_ret_o2c'] > 0).astype(int)
        
        # Secondary Target: Close-to-Close direction on session t+1 (Informational)
        sub['target_ret_c2c'] = (sub['next_adj_close'] - sub['adj_close']) / sub['adj_close']
        sub['target_c2c_direction'] = (sub['target_ret_c2c'] > 0).astype(int)
        
        all_companies.append(sub)
        
    df_mkt_feat = pd.concat(all_companies, ignore_index=True)
    
    # 6. Merge with forward-mapped news sentiment features
    print("Joining forward-aligned news features with market features...")
    df_integrated = pd.merge(df_mkt_feat, df_news_forward, on=['date', 'symbol'], how='left')
    
    # Fill missing news days with explicit defaults
    df_integrated['direct_news_count'] = df_integrated['direct_news_count'].fillna(0).astype(int)
    df_integrated['sectoral_news_count'] = df_integrated['sectoral_news_count'].fillna(0).astype(int)
    df_integrated['global_news_count'] = df_integrated['global_news_count'].fillna(0).astype(int)
    df_integrated['has_direct_news'] = df_integrated['has_direct_news'].fillna(0).astype(int)
    df_integrated['has_sector_news'] = df_integrated['has_sector_news'].fillna(0).astype(int)
    
    sentiment_cols = [
        'direct_news_pos', 'direct_news_neu', 'direct_news_neg', 'direct_finbert_polarity',
        'sectoral_news_pos', 'sectoral_news_neu', 'sectoral_news_neg', 'sectoral_finbert_polarity',
        'global_news_pos', 'global_news_neu', 'global_news_neg', 'global_finbert_polarity'
    ]
    for sc in sentiment_cols:
        df_integrated[sc] = df_integrated[sc].fillna(0.0)
        
    # 7. Compute rolling sentiment features per company (Experiment E)
    print("Computing rolling temporal sentiment windows (3-day and 7-day)...")
    comp_dfs = []
    for sym in TARGET_TICKERS:
        sub = df_integrated[df_integrated['symbol'] == sym].sort_values('date').copy()
        sub['direct_polarity_roll_3d'] = sub['direct_finbert_polarity'].rolling(3, min_periods=1).mean()
        sub['direct_polarity_roll_7d'] = sub['direct_finbert_polarity'].rolling(7, min_periods=1).mean()
        sub['sectoral_polarity_roll_3d'] = sub['sectoral_finbert_polarity'].rolling(3, min_periods=1).mean()
        sub['sectoral_polarity_roll_7d'] = sub['sectoral_finbert_polarity'].rolling(7, min_periods=1).mean()
        comp_dfs.append(sub)
        
    df_final = pd.concat(comp_dfs, ignore_index=True).sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 8. Drop the very last observation for each company because next_day target is physically unknown
    df_final = df_final.dropna(subset=['next_adj_close']).copy()
    
    # Final cleanup of temporary helper columns
    drop_temp = ['next_adj_open', 'next_adj_close']
    df_final = df_final.drop(columns=drop_temp)
    
    # Save Final Dataset
    final_parquet = FINAL_DIR / "final_integrated_dataset.parquet"
    df_final.to_parquet(final_parquet, index=False)
    print(f"\nFinal Integrated Dataset successfully saved: {final_parquet}")
    print(f"Total Rows: {len(df_final):,} | Total Columns: {len(df_final.columns)}")
    
    # Also save CSV sample (first 200 rows)
    sample_csv = FINAL_DIR / "final_dataset_sample_200.csv"
    df_final.head(200).to_csv(sample_csv, index=False)
    print(f"Sample CSV saved: {sample_csv}")
    
    return df_final

if __name__ == "__main__":
    df_final = integrate_and_build_features()
