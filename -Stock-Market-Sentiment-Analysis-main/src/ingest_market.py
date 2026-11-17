"""
Market Data Ingestion Pipeline for TejHQ Indian Markets
Downloads and filters prices and corporate actions for the 15 approved companies.
Preserves Raw OHLCV, Adjusted OHLCV, Corporate Actions, and Trading Calendar separately.
"""

import io
import requests
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

from src.config import (
    TEJHQ_BASE_URL,
    TARGET_TICKERS,
    START_DATE,
    END_DATE,
    RAW_MARKET_DIR,
    RAW_ACTIONS_DIR,
    PROCESSED_DIR
)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def fetch_and_save_raw_market():
    """Download yearly Parquet files from TejHQ and save filtered slices to data/raw/."""
    all_prices = []
    all_actions = []
    
    print("=" * 60)
    print("Ingesting TejHQ Market Prices & Corporate Actions (2020 - 2026)...")
    print("=" * 60)
    
    for year in YEARS:
        # 1. Fetch adjusted prices
        price_url = f"{TEJHQ_BASE_URL}/prices_adjusted/nse_{year}.parquet"
        print(f"Fetching prices for year {year} from {price_url}...")
        r = requests.get(price_url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch {price_url}: status {r.status_code}")
        
        table = pq.read_table(io.BytesIO(r.content))
        df_year = table.to_pandas()
        
        # Filter for series == 'EQ' and target tickers
        df_filtered = df_year[(df_year['series'] == 'EQ') & (df_year['symbol'].isin(TARGET_TICKERS))].copy()
        df_filtered['date'] = pd.to_datetime(df_filtered['date']).dt.date
        
        # Filter date boundary
        start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()
        df_filtered = df_filtered[(df_filtered['date'] >= start_d) & (df_filtered['date'] <= end_d)]
        
        # Save raw slice
        raw_price_file = RAW_MARKET_DIR / f"nse_prices_{year}.parquet"
        df_filtered.to_parquet(raw_price_file, index=False)
        print(f"  -> Saved {len(df_filtered)} rows to {raw_price_file.name}")
        all_prices.append(df_filtered)
        
        # 2. Fetch corporate actions
        actions_url = f"{TEJHQ_BASE_URL}/actions/nse_{year}.parquet"
        print(f"Fetching corporate actions for year {year} from {actions_url}...")
        r_act = requests.get(actions_url)
        if r_act.status_code == 200:
            df_act = pd.read_parquet(io.BytesIO(r_act.content))
            df_act_filtered = df_act[df_act['symbol'].isin(TARGET_TICKERS)].copy()
            df_act_filtered['ex_date'] = pd.to_datetime(df_act_filtered['ex_date']).dt.date
            df_act_filtered = df_act_filtered[(df_act_filtered['ex_date'] >= start_d) & (df_act_filtered['ex_date'] <= end_d)]
            raw_act_file = RAW_ACTIONS_DIR / f"nse_actions_{year}.parquet"
            df_act_filtered.to_parquet(raw_act_file, index=False)
            print(f"  -> Saved {len(df_act_filtered)} corporate actions to {raw_act_file.name}")
            all_actions.append(df_act_filtered)
        else:
            print(f"  -> No actions file or status {r_act.status_code} for {year}")

    # Combine all years
    df_prices_all = pd.concat(all_prices, ignore_index=True).sort_values(['symbol', 'date']).reset_index(drop=True)
    df_actions_all = pd.concat(all_actions, ignore_index=True).sort_values(['symbol', 'ex_date']).reset_index(drop=True)
    
    print("\nTotal Market Records Ingested:", len(df_prices_all))
    print("Total Corporate Actions Ingested:", len(df_actions_all))
    return df_prices_all, df_actions_all

def build_processed_market_datasets(df_prices, df_actions):
    """Separate and build Raw OHLCV, Adjusted OHLCV, and Corporate Actions."""
    print("\nBuilding clean processed datasets...")
    
    # 1. Raw OHLCV dataset
    raw_cols = [
        'date', 'symbol', 'series', 'isin', 'name', 
        'open', 'high', 'low', 'close', 'last', 'prev_close', 
        'volume', 'turnover', 'trades'
    ]
    df_raw = df_prices[raw_cols].copy()
    raw_path = PROCESSED_DIR / "market_raw_ohlcv.parquet"
    df_raw.to_parquet(raw_path, index=False)
    print(f"Saved Raw OHLCV: {raw_path} ({len(df_raw)} rows)")
    
    # 2. Adjusted OHLCV dataset (Scaling ALL price fields consistently by adj_factor_cumulative)
    df_adj = df_prices[['date', 'symbol', 'series', 'isin', 'name']].copy()
    factor = df_prices['adj_factor_cumulative']
    df_adj['adj_factor_cumulative'] = factor
    df_adj['adj_open'] = df_prices['open'] * factor
    df_adj['adj_high'] = df_prices['high'] * factor
    df_adj['adj_low'] = df_prices['low'] * factor
    df_adj['adj_close'] = df_prices['close'] * factor
    df_adj['adj_prev_close'] = df_prices['prev_close'] * factor
    df_adj['volume'] = df_prices['volume']
    df_adj['turnover'] = df_prices['turnover']
    df_adj['trades'] = df_prices['trades']
    
    adj_path = PROCESSED_DIR / "market_adjusted_ohlcv.parquet"
    df_adj.to_parquet(adj_path, index=False)
    print(f"Saved Adjusted OHLCV: {adj_path} ({len(df_adj)} rows)")
    
    # 3. Corporate Actions dataset
    actions_path = PROCESSED_DIR / "corporate_actions.parquet"
    df_actions.to_parquet(actions_path, index=False)
    print(f"Saved Corporate Actions: {actions_path} ({len(df_actions)} records)")
    
    # Also save CSV preview
    df_actions.to_csv(PROCESSED_DIR / "corporate_actions.csv", index=False)
    
    return df_raw, df_adj

def build_trading_calendar(df_prices):
    """Extract official unique NSE trading days and build calendar mapping."""
    unique_dates = sorted(df_prices['date'].unique())
    cal_rows = []
    
    for idx, d in enumerate(unique_dates):
        prev_d = unique_dates[idx - 1] if idx > 0 else None
        next_d = unique_dates[idx + 1] if idx < len(unique_dates) - 1 else None
        cal_rows.append({
            'session_index': idx,
            'trading_date': d,
            'day_of_week': d.strftime('%A'),
            'prev_trading_date': prev_d,
            'next_trading_date': next_d
        })
    
    df_cal = pd.DataFrame(cal_rows)
    cal_parquet = PROCESSED_DIR / "trading_calendar.parquet"
    cal_csv = PROCESSED_DIR / "trading_calendar.csv"
    df_cal.to_parquet(cal_parquet, index=False)
    df_cal.to_csv(cal_csv, index=False)
    print(f"Saved Trading Calendar: {cal_parquet} ({len(df_cal)} trading sessions)")
    return df_cal

if __name__ == "__main__":
    df_prices, df_actions = fetch_and_save_raw_market()
    build_processed_market_datasets(df_prices, df_actions)
    build_trading_calendar(df_prices)
    print("\nMarket ingestion completed successfully!")
