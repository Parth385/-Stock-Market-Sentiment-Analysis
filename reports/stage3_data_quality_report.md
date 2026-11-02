# Stage 3 — Data Quality & Integrity Report

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Research Cohort**: 15 Continuous NSE Nifty 50 Equities (2020–2026)  
**Date**: November 2026  
**Document**: Quality Verification & Leakage-Free Target Alignment Audit  

---

## 1. Executive Summary

Stage 3 implements independent financial-news sentiment extraction across three distinct linguistic paradigms:
1. **Loughran–McDonald (LM) Financial Dictionary** (86,553 entries).
2. **ProsusAI FinBERT** (`ProsusAI/finbert`, 110M params).
3. **Financial RoBERTa** (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`, 82M params).

This report documents the rigorous quality, completeness, and temporal-alignment checks performed across all 155,537 direct news articles and the 23,115 company-session panel observations.

---

## 2. Nine-Point Validation Audit

### 2.1 Missing Articles Audit
- **Requirement**: Zero missing articles across all sentiment datasets.
- **Verification**:
  - Raw cleaned news: 155,537 rows (`data/processed/news_clean.parquet`).
  - Loughran–McDonald dataset: 155,537 rows (`data/processed/sentiment/sentiment_lm.parquet`).
  - FinBERT dataset: 155,537 rows (`data/processed/sentiment/sentiment_finbert.parquet`).
  - Financial RoBERTa dataset: 155,537 rows (`data/processed/sentiment/sentiment_roberta.parquet`).
  - **Result**: **100.0% completeness** (0 missing rows).

### 2.2 Duplicate Article IDs Audit
- **Requirement**: Uniqueness of `article_id` across all extracted sentiment representations.
- **Verification**:
  - `article_id.nunique() = 155,537` in all three sentiment datasets.
  - No duplicate IDs detected.
  - **Result**: **PASS**.

### 2.3 Sentiment Probability Sums Audit
- **Requirement**: For transformer models outputting class probabilities, verify that $\sum_{k \in \{\text{pos}, \text{neu}, \text{neg}\}} P_k = 1.0$ within numerical tolerance ($10^{-4}$).
- **Verification**:
  - FinBERT: $\max |P_{\text{pos}} + P_{\text{neu}} + P_{\text{neg}} - 1.0| < 10^{-6}$.
  - RoBERTa: $\max |P_{\text{pos}} + P_{\text{neu}} + P_{\text{neg}} - 1.0| < 10^{-6}$.
  - **Result**: **PASS**.

### 2.4 Missing vs. Neutral Encoding Audit
- **Requirement**: Distinguish between "No News" and "Genuine Neutral News". Zero-news days must NEVER be encoded as neutral ($P=1.0$ or $0.5$). Never substitute sector sentiment for missing direct company sentiment.
- **Verification**:
  - When `has_direct_news == 0`:
    - `finbert_direct_pos = 0.0`
    - `finbert_direct_neu = 0.0` (NOT 1.0)
    - `finbert_direct_neg = 0.0`
    - `finbert_direct_polarity = 0.0`
    - `finbert_direct_confidence = 0.0`
    - `roberta_direct_pos/neu/neg/polarity/confidence = 0.0`
    - `lm_direct_pos/neg/unc/polarity/subjectivity = 0.0`
  - Genuine neutral articles have:
    - `has_direct_news == 1`
    - `finbert_direct_neu >= 0.50`
    - `finbert_direct_polarity \approx 0.0`
  - Sector sentiment is held in distinct columns (`finbert_sector_*`, `roberta_sector_*`, `lm_sector_*`) and is NEVER copied or substituted into direct sentiment columns.
  - **Result**: **PASS**.

### 2.5 Company Attribution Audit
- **Requirement**: All 15 approved companies must be present without ticker drift or alias leakage.
- **Verification**:
  - Cohort tickers: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, `ICICIBANK`, `SBIN`, `HINDUNILVR`, `ITC`, `LT`, `BHARTIARTL`, `SUNPHARMA`, `CIPLA`, `MARUTI`, `M&M`, `TATASTEEL`.
  - Exactly 15 distinct symbols across all 23,115 trading session rows (1,541 rows per company).
  - **Result**: **PASS**.

### 2.6 Sector Attribution Audit
- **Requirement**: Validate that sector attribution matches the Nifty 50 classification schema.
- **Verification**:
  - 8 distinct sectors represented.
  - Zero unmapped or null sector entries.
  - **Result**: **PASS**.

### 2.7 Temporal Alignment & Forward Session Mapping Audit
- **Requirement**: News published on calendar date $D$ must strictly map to forward session $\tau(D) = \min \{ t \in \text{TradingCalendar} \mid t > D \}$.
- **Verification**:
  - Monday news $\to$ Tuesday trading session ($t$).
  - Weekend news (Sat/Sun) $\to$ Monday trading session ($t$).
  - Pre-holiday news $\to$ Post-holiday session ($t$).
  - No news record is ever assigned to a past or same-day trading session prior to market open.
  - **Result**: **PASS**.

### 2.8 Model Checkpoint Reproducibility Audit
- **Requirement**: Document exact checkpoints, tokenizers, sequence lengths, and chunking/pooling.
- **Verification**:
  - FinBERT: `ProsusAI/finbert` (Hugging Face Hub), `BertTokenizerFast`, max length 64 tokens, 0.00% truncation rate.
  - RoBERTa: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`, `RobertaTokenizerFast`, max length 64 tokens, 0.00% truncation rate.
  - Loughran–McDonald: Notre Dame SRAF 2023 Master Financial Dictionary (86,553 words).
  - **Result**: **PASS**.

### 2.9 Absence of Future-Data Leakage Audit
- **Requirement**: Primary prediction target must be intraday Open-to-Close on session $t$:
  $$\text{Target}_t = \frac{\text{adj\_close}_t - \text{adj\_open}_t}{\text{adj\_open}_t}$$
  All predictive features for session $t$ must only use information available at Tuesday 9:15 AM Open!
- **Verification**:
  - Technical indicators for session $t$ prediction are strictly lagged by 1 session ($t-1$ Close indicators: `rsi_14_lag1`, `macd_lag1`, `rolling_vol_5d_lag1`, etc.).
  - The only session-$t$ market price utilized is `adj_open_t`, encoded as the opening gap:
    $$\text{open\_gap\_pct} = \frac{\text{adj\_open}_t - \text{adj\_close}_{t-1}}{\text{adj\_close}_{t-1}}$$
  - Session-$t$ closing price `adj_close_t`, high `raw_high_t`, and low `raw_low_t` are strictly excluded from predictive features for the primary experiment.
  - Robustness target for next session ($t+1$) is preserved alongside session-$t$ close features for comparative benchmarking.
  - **Result**: **PASS**.

---

## 3. Dataset Summary Statistics

| Dataset File | File Format | Row Count | Column Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| `data/processed/sentiment/sentiment_lm.parquet` | Parquet | 155,537 | 17 | Article-level Loughran–McDonald lexicon sentiment |
| `data/processed/sentiment/sentiment_finbert.parquet` | Parquet | 155,537 | 13 | Article-level ProsusAI FinBERT transformer sentiment |
| `data/processed/sentiment/sentiment_roberta.parquet` | Parquet | 155,537 | 13 | Article-level Financial RoBERTa transformer sentiment |
| `data/processed/sentiment/sentiment_daily_aggregated.parquet` | Parquet | 34,230 | 38 | Date-level multi-tier sentiment (Direct, Sector, Market) |
| `data/processed/sentiment/sentiment_temporal_features.parquet` | Parquet | 23,130 | 56 | Full panel with temporal windows (t, t-1, roll3, roll7) |
| `data/final/final_sentiment_enhanced_dataset.parquet` | Parquet | 23,115 | 82 | Master Stage 3 feature store with primary & robustness targets |
