"""
News Data Ingestion and Processing Pipeline
Downloads and filters financial news from mveen3 Six-Year Indian Stock Market Dataset.
Preserves original text, headlines, dates, URLs, source publishers, and multi-tier classifications.
Extracts benchmark FinBERT reference probabilities and news volume features.
"""

import sys
import re
import csv
import io
import requests
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime

from src.config import (
    MVEEN3_BASE_URL,
    COMPANIES,
    TARGET_TICKERS,
    START_DATE,
    END_DATE,
    RAW_NEWS_DIR,
    PROCESSED_DIR
)

# Compile regex patterns for company aliases (case-insensitive with word boundaries)
ALIAS_PATTERNS = {}
for ticker, info in COMPANIES.items():
    # Build regex matching any alias with word boundary
    escaped_aliases = [re.escape(a) for a in info['aliases']]
    pattern = re.compile(r'\b(' + '|'.join(escaped_aliases) + r')\b', re.IGNORECASE)
    ALIAS_PATTERNS[ticker] = pattern

# Sector keyword patterns
SECTOR_PATTERNS = {
    "Energy": re.compile(r'\b(oil|gas|crude|petroleum|refinery|renewable|power|energy)\b', re.IGNORECASE),
    "Information Technology": re.compile(r'\b(IT|software|tech|cloud|digital|SaaS|cybersecurity|AI)\b', re.IGNORECASE),
    "Financial Services": re.compile(r'\b(bank|banking|lender|credit|NPA|RBI|repo rate|monetary|NBFC|finance)\b', re.IGNORECASE),
    "Fast Moving Consumer Goods": re.compile(r'\b(FMCG|consumer goods|packaged food|soap|detergent|retail consumption)\b', re.IGNORECASE),
    "Construction & Infrastructure": re.compile(r'\b(infra|infrastructure|construction|engineering|cement|highways|railways)\b', re.IGNORECASE),
    "Telecommunications": re.compile(r'\b(telecom|spectrum|5G|4G|tariff|TRAI|broadband|subscriber)\b', re.IGNORECASE),
    "Healthcare & Pharmaceuticals": re.compile(r'\b(pharma|pharmaceutical|drug|API|FDA|clinical|vaccine|healthcare)\b', re.IGNORECASE),
    "Automobile": re.compile(r'\b(auto|automobile|EV|electric vehicle|passenger vehicle|commercial vehicle|car sales)\b', re.IGNORECASE),
    "Metals & Mining": re.compile(r'\b(steel|metal|mining|iron ore|aluminum|copper|smelting)\b', re.IGNORECASE)
}

def clean_publisher_source(url):
    """Extract publisher domain name cleanly from URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if "economictimes" in netloc:
            return "The Economic Times"
        elif "business-standard" in netloc:
            return "Business Standard"
        elif "financialexpress" in netloc:
            return "Financial Express"
        elif "moneycontrol" in netloc:
            return "Moneycontrol"
        else:
            return netloc
    except:
        return "Unknown"

def ingest_benchmark_sentiment_and_counts():
    """Ingest mveen3/dataset/news_sentiment.csv for exact benchmark probabilities and counts."""
    print("Ingesting benchmark news_sentiment.csv...")
    url = f"{MVEEN3_BASE_URL}/dataset/news_sentiment.csv"
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url}: status {r.status_code}")
    
    df = pd.read_csv(io.StringIO(r.text))
    # Filter target tickers
    df = df[df['Symbol'].isin(TARGET_TICKERS)].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    df = df[(df['Date'] >= start_d) & (df['Date'] <= end_d)].copy()
    
    # Save raw slice
    raw_file = RAW_NEWS_DIR / "benchmark_news_sentiment.parquet"
    df.to_parquet(raw_file, index=False)
    print(f"Saved benchmark sentiment slice: {raw_file} ({len(df)} rows)")
    return df

def stream_and_filter_processed_news():
    """
    Stream processed_news_dataset.csv, extract articles mentioning the 15 companies,
    preserve full text, headline, date, url, source, and assign news tier.
    """
    print("\nStreaming processed_news_dataset.csv from Hugging Face...")
    url = f"{MVEEN3_BASE_URL}/dataset/processed_news_dataset.csv"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to stream {url}: status {r.status_code}")
    
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    
    matched_articles = []
    seen_articles = set() # (date, title) to remove duplicates
    
    line_iter = r.iter_lines(chunk_size=131072, decode_unicode=True)
    reader = csv.reader(line_iter)
    header = next(reader)
    print(f"Ingesting with headers: {header}")
    
    article_idx = 0
    direct_match_count = 0
    total_parsed = 0
    
    for row in reader:
        if len(row) < 4:
            continue
        total_parsed += 1
        
        date_str = row[0].strip()
        title = row[1].strip()
        news_text = row[2].strip()
        article_url = row[3].strip()
        
        try:
            art_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            continue
            
        if art_date < start_d or art_date > end_d:
            continue
            
        # Deduplication check
        dedup_key = (art_date, title.lower())
        if dedup_key in seen_articles:
            continue
        seen_articles.add(dedup_key)
        
        # Check company mentions in title and body
        combined_text = f"{title}. {news_text}"
        matched_tickers = []
        for ticker, pattern in ALIAS_PATTERNS.items():
            if pattern.search(combined_text):
                matched_tickers.append(ticker)
        
        # Determine publisher source
        publisher = clean_publisher_source(article_url)
        
        if matched_tickers:
            direct_match_count += 1
            for t in matched_tickers:
                article_idx += 1
                matched_articles.append({
                    'article_id': article_idx,
                    'date': art_date,
                    'symbol': t,
                    'sector': COMPANIES[t]['sector'],
                    'news_tier': 'DIRECT',
                    'title': title,
                    'news_text': news_text,
                    'url': article_url,
                    'publisher': publisher
                })
        
        if total_parsed % 50000 == 0:
            print(f"  Parsed {total_parsed:,} articles | Direct matches: {direct_match_count:,} | Kept rows: {len(matched_articles):,}")
    
    print(f"\nFinished parsing {total_parsed:,} articles.")
    print(f"Total direct company news records extracted: {len(matched_articles):,}")
    
    df_news = pd.DataFrame(matched_articles)
    
    # Save clean article-level news dataset
    clean_news_path = PROCESSED_DIR / "news_clean.parquet"
    df_news.to_parquet(clean_news_path, index=False)
    print(f"Saved Clean Article Dataset: {clean_news_path} ({len(df_news)} records)")
    
    # Save sample CSV for inspection
    df_news.head(100).to_csv(PROCESSED_DIR / "news_sample_100.csv", index=False)
    return df_news

def build_daily_news_features(df_news, df_benchmark_sentiment):
    """
    Build daily company-level aggregated sentiment, volume counts, and no-news flags.
    Merges extracted article volume with benchmark FinBERT probabilities.
    """
    print("\nBuilding daily company-level news features...")
    
    # 1. Group extracted articles by (date, symbol) to get direct article counts and titles
    df_direct_counts = df_news.groupby(['date', 'symbol']).agg(
        extracted_direct_count=('article_id', 'count'),
        aggregated_titles=('title', lambda titles: ' | '.join(titles[:5]))
    ).reset_index()
    
    # Rename Date and Symbol in benchmark sentiment
    df_bench = df_benchmark_sentiment.rename(columns={'Date': 'date', 'Symbol': 'symbol'}).copy()
    
    # Merge on (date, symbol)
    df_daily = pd.merge(df_bench, df_direct_counts, on=['date', 'symbol'], how='left')
    df_daily['extracted_direct_count'] = df_daily['extracted_direct_count'].fillna(0).astype(int)
    df_daily['aggregated_titles'] = df_daily['aggregated_titles'].fillna("")
    
    # Add explicit No-News indicator flags
    df_daily['has_direct_news'] = (df_daily['direct_news_count'] > 0).astype(int)
    df_daily['has_sector_news'] = (df_daily['sectoral_news_count'] > 0).astype(int)
    
    # Compute polarities (Pos - Neg)
    df_daily['direct_finbert_polarity'] = df_daily['direct_news_pos'] - df_daily['direct_news_neg']
    df_daily['sectoral_finbert_polarity'] = df_daily['sectoral_news_pos'] - df_daily['sectoral_news_neg']
    df_daily['global_finbert_polarity'] = df_daily['global_news_pos'] - df_daily['global_news_neg']
    
    # When no direct news, polarity is explicitly 0.0 (already 0.0 - 0.0)
    
    daily_news_path = PROCESSED_DIR / "news_daily_aggregated.parquet"
    df_daily.to_parquet(daily_news_path, index=False)
    print(f"Saved Daily News Aggregates: {daily_news_path} ({len(df_daily)} rows)")
    return df_daily

if __name__ == "__main__":
    df_bench = ingest_benchmark_sentiment_and_counts()
    df_news = stream_and_filter_processed_news()
    df_daily = build_daily_news_features(df_news, df_bench)
    print("\nNews ingestion pipeline completed successfully!")
