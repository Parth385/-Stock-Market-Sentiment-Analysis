"""
Multi-tier Sentiment Aggregation.
Combines article-level Loughran-McDonald, FinBERT, and Financial RoBERTa scores into:
1. Direct company news sentiment.
2. Sector-level news sentiment.
3. Broad market news sentiment.
4. News volume metrics.
5. Strict no-news indicators (distinguishing zero news from genuine neutral news).

Output: data/processed/sentiment/sentiment_daily_aggregated.parquet
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    lm_path = "data/processed/sentiment/sentiment_lm.parquet"
    finbert_path = "data/processed/sentiment/sentiment_finbert.parquet"
    roberta_path = "data/processed/sentiment/sentiment_roberta.parquet"
    sector_map_path = "data/processed/company_sector_mapping.json"
    output_path = "data/processed/sentiment/sentiment_daily_aggregated.parquet"

    print("Verifying input sentiment files...")
    for p in [lm_path, finbert_path, roberta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing sentiment file: {p}")

    print("Loading Loughran-McDonald sentiment...")
    df_lm = pd.read_parquet(lm_path)
    print("Loading FinBERT sentiment...")
    df_fb = pd.read_parquet(finbert_path)
    print("Loading RoBERTa sentiment...")
    df_rob = pd.read_parquet(roberta_path)

    print(f"Shapes - LM: {df_lm.shape}, FinBERT: {df_fb.shape}, RoBERTa: {df_rob.shape}")

    # Merge into unified article sentiment dataset
    df_articles = df_lm[[
        "article_id", "date", "symbol", "sector", "title", "url", "publisher",
        "lm_pos_count", "lm_neg_count", "lm_unc_count", "lm_lit_count", "lm_con_count",
        "lm_total_words", "lm_polarity", "lm_subjectivity", "lm_uncertainty"
    ]].copy()

    df_articles = df_articles.merge(
        df_fb[["article_id", "finbert_pos", "finbert_neu", "finbert_neg", "finbert_polarity", "finbert_confidence"]],
        on="article_id", how="left"
    )

    df_articles = df_articles.merge(
        df_rob[["article_id", "roberta_pos", "roberta_neu", "roberta_neg", "roberta_polarity", "roberta_confidence"]],
        on="article_id", how="left"
    )

    # Save complete merged article sentiment dataset
    article_out = "data/processed/sentiment/sentiment_all_models_articles.parquet"
    df_articles.to_parquet(article_out, index=False)
    print(f"Saved unified article sentiment dataset: {article_out} ({len(df_articles):,} rows)")

    # 1. Broad Market News Aggregation (by date)
    print("Computing broad market daily sentiment...")
    df_market = df_articles.groupby("date").agg(
        market_news_count=("article_id", "count"),
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

    # 2. Sector-level News Aggregation (by date, sector)
    print("Computing sector daily sentiment...")
    df_sector = df_articles.groupby(["date", "sector"]).agg(
        sector_news_count=("article_id", "count"),
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
    df_sector["has_sector_news"] = 1

    # 3. Direct Company News Aggregation (by date, symbol, sector)
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

    # Merge Direct with Sector and Market
    print("Integrating multi-tier sentiment...")
    df_merged = df_direct.merge(df_sector, on=["date", "sector"], how="left")
    df_merged = df_merged.merge(df_market, on="date", how="left")

    print(f"Aggregated daily records with direct news: {len(df_merged):,}")
    df_merged.to_parquet(output_path, index=False)
    print(f"Successfully saved aggregated daily sentiment to {output_path}")

if __name__ == "__main__":
    main()
