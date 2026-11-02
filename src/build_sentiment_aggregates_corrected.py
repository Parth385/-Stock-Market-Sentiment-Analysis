"""
Corrected Multi-tier Sentiment Aggregation.
Addresses multi-company article duplication:
1. Calculates UNIQUE article-level sector aggregation by deduplicating on (date, sector, url).
2. Calculates UNIQUE article-level broad market aggregation by deduplicating on (date, url).
3. Preserves company-exposure-weighted metrics as separate, explicitly named features:
   - finbert_sector_polarity_exposure_weighted, sector_news_count_exposure
   - finbert_market_polarity_exposure_weighted, market_news_count_exposure
4. Preserves direct company-level attribution untouched.
5. Saves to separate corrected files without overwriting existing Stage 3 outputs:
   - Output: data/processed/sentiment/sentiment_daily_aggregated_corrected.parquet
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    article_path = "data/processed/sentiment/sentiment_all_models_articles.parquet"
    output_path = "data/processed/sentiment/sentiment_daily_aggregated_corrected.parquet"

    print(f"Loading unified article dataset from {article_path}...")
    df_articles = pd.read_parquet(article_path)
    print(f"Total article records: {len(df_articles):,}")

    # =========================================================================
    # 1. DIRECT COMPANY NEWS AGGREGATION (Preserved exactly as-is)
    # =========================================================================
    print("Computing direct company daily sentiment...")
    df_direct = df_articles.groupby(["date", "symbol", "sector"]).agg(
        direct_news_count=("article_id", "count"),
        finbert_direct_pos=("finbert_pos", "mean"),
        finbert_direct_neu=("finbert_neu", "mean"),
        finbert_direct_neg=("finbert_neg", "mean"),
        finbert_direct_polarity=("finbert_polarity", "mean"),
        finbert_direct_confidence=("finbert_confidence", "mean"),
        roberta_direct_pos=("roberta_pos", "mean"),
        roberta_direct_neu=("roberta_neu", "mean"),
        roberta_direct_neg=("roberta_neg", "mean"),
        roberta_direct_polarity=("roberta_polarity", "mean"),
        roberta_direct_confidence=("roberta_confidence", "mean"),
        lm_direct_pos_count=("lm_pos_count", "sum"),
        lm_direct_neg_count=("lm_neg_count", "sum"),
        lm_direct_unc_count=("lm_unc_count", "sum"),
        lm_direct_polarity=("lm_polarity", "mean"),
        lm_direct_subjectivity=("lm_subjectivity", "mean")
    ).reset_index()
    df_direct["has_direct_news"] = 1

    # =========================================================================
    # 2. UNIQUE SECTOR NEWS AGGREGATION (Deduplicated by (date, sector, url))
    # =========================================================================
    print("Computing UNIQUE sector-level daily sentiment (deduplicated by (date, sector, url))...")
    df_sec_unique_articles = df_articles.drop_duplicates(subset=["date", "sector", "url"])
    print(f"  Unique (date, sector, url) article records: {len(df_sec_unique_articles):,} (from {len(df_articles):,})")

    df_sector_unique = df_sec_unique_articles.groupby(["date", "sector"]).agg(
        sector_news_count=("url", "count"),
        finbert_sector_pos=("finbert_pos", "mean"),
        finbert_sector_neu=("finbert_neu", "mean"),
        finbert_sector_neg=("finbert_neg", "mean"),
        finbert_sector_polarity=("finbert_polarity", "mean"),
        roberta_sector_pos=("roberta_pos", "mean"),
        roberta_sector_neu=("roberta_neu", "mean"),
        roberta_sector_neg=("roberta_neg", "mean"),
        roberta_sector_polarity=("roberta_polarity", "mean"),
        lm_sector_pos_count=("lm_pos_count", "sum"),
        lm_sector_neg_count=("lm_neg_count", "sum"),
        lm_sector_polarity=("lm_polarity", "mean")
    ).reset_index()
    df_sector_unique["has_sector_news"] = 1

    # Also compute company-exposure-weighted sector metrics as separate features
    print("Computing separate company-exposure-weighted sector metrics...")
    df_sector_exposure = df_articles.groupby(["date", "sector"]).agg(
        sector_news_count_exposure=("article_id", "count"),
        finbert_sector_polarity_exposure=("finbert_polarity", "mean"),
        roberta_sector_polarity_exposure=("roberta_polarity", "mean"),
        lm_sector_polarity_exposure=("lm_polarity", "mean")
    ).reset_index()

    df_sector = df_sector_unique.merge(df_sector_exposure, on=["date", "sector"], how="left")

    # =========================================================================
    # 3. UNIQUE BROAD MARKET NEWS AGGREGATION (Deduplicated by (date, url))
    # =========================================================================
    print("Computing UNIQUE broad market daily sentiment (deduplicated by (date, url))...")
    df_mkt_unique_articles = df_articles.drop_duplicates(subset=["date", "url"])
    print(f"  Unique (date, url) article records: {len(df_mkt_unique_articles):,} (from {len(df_articles):,})")

    df_market_unique = df_mkt_unique_articles.groupby("date").agg(
        market_news_count=("url", "count"),
        finbert_market_pos=("finbert_pos", "mean"),
        finbert_market_neu=("finbert_neu", "mean"),
        finbert_market_neg=("finbert_neg", "mean"),
        finbert_market_polarity=("finbert_polarity", "mean"),
        roberta_market_pos=("roberta_pos", "mean"),
        roberta_market_neu=("roberta_neu", "mean"),
        roberta_market_neg=("roberta_neg", "mean"),
        roberta_market_polarity=("roberta_polarity", "mean"),
        lm_market_polarity=("lm_polarity", "mean")
    ).reset_index()

    # Also compute company-exposure-weighted market metrics as separate features
    print("Computing separate company-exposure-weighted broad market metrics...")
    df_market_exposure = df_articles.groupby("date").agg(
        market_news_count_exposure=("article_id", "count"),
        finbert_market_polarity_exposure=("finbert_polarity", "mean"),
        roberta_market_polarity_exposure=("roberta_polarity", "mean"),
        lm_market_polarity_exposure=("lm_polarity", "mean")
    ).reset_index()

    df_market = df_market_unique.merge(df_market_exposure, on="date", how="left")

    # =========================================================================
    # 4. MERGE DIRECT WITH SECTOR AND MARKET
    # =========================================================================
    print("Integrating multi-tier sentiment...")
    df_merged = df_direct.merge(df_sector, on=["date", "sector"], how="left")
    df_merged = df_merged.merge(df_market, on="date", how="left")

    print(f"Saving corrected daily sentiment to {output_path} ({len(df_merged):,} rows)...")
    df_merged.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
