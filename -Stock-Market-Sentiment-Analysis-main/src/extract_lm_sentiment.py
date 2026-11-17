"""
Stage 3: Loughran-McDonald Financial Dictionary Sentiment Extraction
Extracts Positive, Negative, Uncertainty, Litigious, and Constraining word counts and polarities.
Saves to data/processed/sentiment/sentiment_lm.parquet
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DIR, DATA_DIR

LM_DICT_PATH = DATA_DIR / "raw" / "sentiment" / "loughran_mcdonald_master_dictionary.csv"
OUTPUT_DIR = PROCESSED_DIR / "sentiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_lm_wordlists():
    """Load Loughran-McDonald word lists into sets."""
    print(f"Loading Loughran-McDonald Master Dictionary from {LM_DICT_PATH}...")
    df_lm = pd.read_csv(LM_DICT_PATH)
    
    # Extract categories where indicator column > 0
    categories = {
        'positive': set(df_lm[df_lm['Positive'] > 0]['Word'].str.upper()),
        'negative': set(df_lm[df_lm['Negative'] > 0]['Word'].str.upper()),
        'uncertainty': set(df_lm[df_lm['Uncertainty'] > 0]['Word'].str.upper()),
        'litigious': set(df_lm[df_lm['Litigious'] > 0]['Word'].str.upper()),
        'constraining': set(df_lm[df_lm['Constraining'] > 0]['Word'].str.upper())
    }
    
    print(f"Loaded LM Words: {len(categories['positive'])} Positive, {len(categories['negative'])} Negative, {len(categories['uncertainty'])} Uncertainty")
    return categories

def extract_lm_sentiment():
    """Run dictionary-based sentiment extraction across unique news articles."""
    categories = load_lm_wordlists()
    pos_words = categories['positive']
    neg_words = categories['negative']
    unc_words = categories['uncertainty']
    lit_words = categories['litigious']
    con_words = categories['constraining']
    
    clean_news_path = PROCESSED_DIR / "news_clean.parquet"
    print(f"Loading cleaned news articles from {clean_news_path}...")
    df_news = pd.read_parquet(clean_news_path)
    
    # Process unique articles (date, title) to optimize execution
    df_unique = df_news[['date', 'title', 'news_text']].drop_duplicates(subset=['date', 'title']).copy()
    print(f"Total unique articles to score: {len(df_unique):,}")
    
    tokenizer_pattern = re.compile(r'\b[A-Za-z]+\b')
    
    results = []
    t0 = pd.Timestamp.now()
    
    for idx, row in enumerate(df_unique.itertuples(index=False)):
        combined_text = f"{row.title}. {row.news_text}"
        tokens = [w.upper() for w in tokenizer_pattern.findall(combined_text)]
        n_tokens = len(tokens)
        
        pos_c = sum(1 for w in tokens if w in pos_words)
        neg_c = sum(1 for w in tokens if w in neg_words)
        unc_c = sum(1 for w in tokens if w in unc_words)
        lit_c = sum(1 for w in tokens if w in lit_words)
        con_c = sum(1 for w in tokens if w in con_words)
        
        # Normalized sentiment scores
        denom = pos_c + neg_c
        polarity = (pos_c - neg_c) / (denom + 1e-6) if denom > 0 else 0.0
        subjectivity = (pos_c + neg_c) / (n_tokens + 1e-6) if n_tokens > 0 else 0.0
        uncertainty_score = unc_c / (n_tokens + 1e-6) if n_tokens > 0 else 0.0
        
        results.append({
            'date': row.date,
            'title': row.title,
            'lm_pos_count': pos_c,
            'lm_neg_count': neg_c,
            'lm_unc_count': unc_c,
            'lm_lit_count': lit_c,
            'lm_con_count': con_c,
            'lm_total_words': n_tokens,
            'lm_polarity': polarity,
            'lm_subjectivity': subjectivity,
            'lm_uncertainty': uncertainty_score
        })
        
        if (idx + 1) % 25000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df_unique):,} articles...")
            
    df_lm_scores = pd.DataFrame(results)
    print(f"Completed LM scoring in {(pd.Timestamp.now() - t0).total_seconds():.1f}s")
    
    # Merge back to full news dataset (preserving article_id, symbol, sector, news_tier)
    df_merged = pd.merge(
        df_news[['article_id', 'date', 'symbol', 'sector', 'news_tier', 'title', 'url', 'publisher']],
        df_lm_scores,
        on=['date', 'title'],
        how='left'
    )
    
    out_path = OUTPUT_DIR / "sentiment_lm.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"Saved Loughran-McDonald Sentiment Dataset: {out_path} ({len(df_merged):,} records)")
    return df_merged

if __name__ == "__main__":
    extract_lm_sentiment()
