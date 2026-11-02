"""
Extract Financial Sentiment using ProsusAI FinBERT.
Model: ProsusAI/finbert
Tokenizer: BertTokenizer (AutoTokenizer)
Input: News Headlines (titles) with max sequence length 64 (0% truncation rate).
Output: data/processed/sentiment/sentiment_finbert.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    start_time = time.time()
    input_path = "data/processed/news_clean.parquet"
    output_dir = "data/processed/sentiment"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sentiment_finbert.parquet")

    print(f"Loading cleaned news from {input_path}...")
    df_news = pd.read_parquet(input_path)
    print(f"Total articles: {len(df_news):,}")

    # Unique titles for maximum inference efficiency
    unique_titles = df_news['title'].drop_duplicates().tolist()
    total_unique = len(unique_titles)
    print(f"Unique headlines to score: {total_unique:,}")

    model_name = "ProsusAI/finbert"
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Verify label mapping: {0: 'positive', 1: 'negative', 2: 'neutral'}
    id2label = model.config.id2label
    print(f"Model id2label mapping: {id2label}")
    pos_idx = [k for k, v in id2label.items() if 'pos' in v.lower()][0]
    neg_idx = [k for k, v in id2label.items() if 'neg' in v.lower()][0]
    neu_idx = [k for k, v in id2label.items() if 'neu' in v.lower()][0]

    # CPU optimization
    num_threads = torch.get_num_threads()
    print(f"PyTorch using {num_threads} CPU threads")

    batch_size = 128
    results = []

    print(f"Beginning batch inference (batch_size={batch_size}, max_length=64)...")
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, total_unique, batch_size):
            batch_titles = unique_titles[i : i + batch_size]
            inputs = tokenizer(
                batch_titles,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            b_pos = probs[:, pos_idx]
            b_neg = probs[:, neg_idx]
            b_neu = probs[:, neu_idx]
            b_pol = b_pos - b_neg
            b_conf = np.max(probs, axis=1)

            for j, title in enumerate(batch_titles):
                results.append({
                    "title": title,
                    "finbert_pos": float(b_pos[j]),
                    "finbert_neu": float(b_neu[j]),
                    "finbert_neg": float(b_neg[j]),
                    "finbert_polarity": float(b_pol[j]),
                    "finbert_confidence": float(b_conf[j])
                })

            if (i // batch_size) % 50 == 0 or (i + batch_size) >= total_unique:
                elapsed = time.time() - t0
                done = min(i + batch_size, total_unique)
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_unique - done) / rate if rate > 0 else 0
                print(f"  Processed {done:,}/{total_unique:,} ({done/total_unique*100:.1f}%) | "
                      f"Speed: {rate:.1f} titles/s | ETA: {eta/60:.1f} min")

    df_scored = pd.DataFrame(results)

    # Join back to full article dataset
    print("Mapping FinBERT scores back to all article records...")
    df_final = df_news.merge(df_scored, on="title", how="left")

    cols_to_keep = [
        "article_id", "date", "symbol", "sector", "news_tier", "title", "url", "publisher",
        "finbert_pos", "finbert_neu", "finbert_neg", "finbert_polarity", "finbert_confidence"
    ]
    df_final = df_final[cols_to_keep]

    print(f"Saving FinBERT sentiment dataset to {output_path}...")
    df_final.to_parquet(output_path, index=False)
    total_time = time.time() - start_time
    print(f"Successfully generated FinBERT dataset: {len(df_final):,} rows in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
