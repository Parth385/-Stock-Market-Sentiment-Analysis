# Stage 3 — Master Data Dictionary

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Scope**: Stage 3 Datasets and Sentiment-Enhanced Feature Store  
**Version**: 3.0.0 (Stage 3 Complete)  
**Date**: November 2026  

---

## 1. Directory Structure

```text
Beyond_Sentiment/
├── data/
│   ├── raw/
│   │   └── sentiment/
│   │       └── loughran_mcdonald_master_dictionary.csv (86,553 terms)
│   ├── processed/
│   │   ├── news_clean.parquet (155,537 rows)
│   │   ├── market_raw_ohlcv.parquet (23,130 rows)
│   │   ├── market_adjusted_ohlcv.parquet (23,130 rows)
│   │   ├── trading_calendar.parquet (1,542 sessions)
│   │   └── sentiment/
│   │       ├── sentiment_lm.parquet (155,537 rows × 17 cols)
│   │       ├── sentiment_finbert.parquet (155,537 rows × 13 cols)
│   │       ├── sentiment_roberta.parquet (155,537 rows × 13 cols)
│   │       ├── sentiment_daily_aggregated.parquet (Date-level multi-tier aggregations)
│   │       └── sentiment_temporal_features.parquet (Panel: 23,130 company-sessions)
│   └── final/
│       ├── final_integrated_dataset.parquet (Preserved Stage 2 baseline: 23,115 rows × 61 cols)
│       └── final_sentiment_enhanced_dataset.parquet (Master Stage 3 Feature Store: 23,115 rows)
```

---

## 2. Article-Level Sentiment Datasets (`data/processed/sentiment/`)

### 2.1 Loughran–McDonald Dataset (`sentiment_lm.parquet`)
| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `article_id` | `int64` | 0 | Unique identifier linking to original article record. |
| `date` | `date` | 0 | Publication calendar date of the news article. |
| `symbol` | `string` | 0 | NSE ticker symbol of the mentioned company (1 of 15 cohort). |
| `sector` | `string` | 0 | Sector assignment according to Nifty 50 classification. |
| `news_tier` | `string` | 0 | Provenance tier: `DIRECT`. |
| `title` | `string` | 0 | Original headline text of the news article. |
| `url` | `string` | 0 | Source URL / web reference. |
| `publisher` | `string` | 0 | News publisher source. |
| `lm_pos_count` | `int64` | 0 | Number of words matching the LM Positive word list. |
| `lm_neg_count` | `int64` | 0 | Number of words matching the LM Negative word list. |
| `lm_unc_count` | `int64` | 0 | Number of words matching the LM Uncertainty word list. |
| `lm_lit_count` | `int64` | 0 | Number of words matching the LM Litigious word list. |
| `lm_con_count` | `int64` | 0 | Number of words matching the LM Constraining word list. |
| `lm_total_words` | `int64` | 0 | Total tokenized word count in the article headline. |
| `lm_polarity` | `float64` | 0 | Normalized polarity: $(N_{\text{pos}} - N_{\text{neg}}) / (N_{\text{pos}} + N_{\text{neg}} + 10^{-5}) \in [-1.0, 1.0]$. |
| `lm_subjectivity` | `float64` | 0 | Lexical subjectivity: $(N_{\text{pos}} + N_{\text{neg}}) / (\text{Total Words} + 10^{-5})$. |
| `lm_uncertainty` | `float64` | 0 | Uncertainty density: $N_{\text{unc}} / (\text{Total Words} + 10^{-5})$. |

### 2.2 FinBERT Dataset (`sentiment_finbert.parquet`)
| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `article_id` | `int64` | 0 | Unique identifier linking to original article record. |
| `date` | `date` | 0 | Publication calendar date. |
| `symbol` | `string` | 0 | NSE ticker symbol. |
| `sector` | `string` | 0 | Sector assignment. |
| `news_tier` | `string` | 0 | Provenance tier (`DIRECT`). |
| `title` | `string` | 0 | Original headline text. |
| `url` | `string` | 0 | Source URL. |
| `publisher` | `string` | 0 | News publisher source. |
| `finbert_pos` | `float64` | 0 | FinBERT positive class probability $P(\text{pos}) \in [0.0, 1.0]$. |
| `finbert_neu` | `float64` | 0 | FinBERT neutral class probability $P(\text{neu}) \in [0.0, 1.0]$. |
| `finbert_neg` | `float64` | 0 | FinBERT negative class probability $P(\text{neg}) \in [0.0, 1.0]$. |
| `finbert_polarity` | `float64` | 0 | Net polarity: $P(\text{pos}) - P(\text{neg}) \in [-1.0, 1.0]$. |
| `finbert_confidence` | `float64` | 0 | Prediction confidence: $\max(P_{\text{pos}}, P_{\text{neu}}, P_{\text{neg}}) \in [0.33, 1.0]$. |

### 2.3 Financial RoBERTa Dataset (`sentiment_roberta.parquet`)
| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `article_id` | `int64` | 0 | Unique identifier linking to original article record. |
| `date` | `date` | 0 | Publication calendar date. |
| `symbol` | `string` | 0 | NSE ticker symbol. |
| `sector` | `string` | 0 | Sector assignment. |
| `news_tier` | `string` | 0 | Provenance tier (`DIRECT`). |
| `title` | `string` | 0 | Original headline text. |
| `url` | `string` | 0 | Source URL. |
| `publisher` | `string` | 0 | News publisher source. |
| `roberta_pos` | `float64` | 0 | RoBERTa positive class probability $P(\text{pos}) \in [0.0, 1.0]$. |
| `roberta_neu` | `float64` | 0 | RoBERTa neutral class probability $P(\text{neu}) \in [0.0, 1.0]$. |
| `roberta_neg` | `float64` | 0 | RoBERTa negative class probability $P(\text{neg}) \in [0.0, 1.0]$. |
| `roberta_polarity` | `float64` | 0 | Net polarity: $P(\text{pos}) - P(\text{neg}) \in [-1.0, 1.0]$. |
| `roberta_confidence` | `float64` | 0 | Prediction confidence: $\max(P_{\text{pos}}, P_{\text{neu}}, P_{\text{neg}}) \in [0.33, 1.0]$. |

---

## 3. Master Sentiment-Enhanced Feature Store (`data/final/final_sentiment_enhanced_dataset.parquet`)

### 3.1 Identification & Market Panel Columns
- `date`: NSE Trading session date $t$.
- `symbol`: NSE Stock ticker (15 cohort).
- `sector`: Sector classification.
- `raw_open`, `raw_high`, `raw_low`, `raw_close`, `raw_volume`, `raw_turnover`, `raw_trades`: TejHQ unadjusted Bhavcopy metrics for session $t$.
- `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_prev_close`, `adj_factor_cumulative`: Split/bonus/dividend adjusted metrics.

### 3.2 Target Variables
1. **`target_same_session_ret_o2c` (PRIMARY TARGET)**:
   $$\frac{\text{adj\_close}_t - \text{adj\_open}_t}{\text{adj\_open}_t}$$
   Intraday return on session $t$ (buying at 9:15 AM Open and selling at 3:30 PM Close).
2. **`target_same_session_dir_o2c` (PRIMARY BINARY TARGET)**:
   $$1 \text{ if } \text{target\_same\_session\_ret\_o2c} > 0 \text{ else } 0$$
3. **`target_next_session_ret_o2c` (ROBUSTNESS TARGET)**:
   $$\frac{\text{adj\_close}_{t+1} - \text{adj\_open}_{t+1}}{\text{adj\_open}_{t+1}}$$
4. **`target_next_session_dir_o2c` (ROBUSTNESS TARGET)**:
   $$1 \text{ if } \text{target\_next\_session\_ret\_o2c} > 0 \text{ else } 0$$
5. **`target_next_session_ret_c2c` (ROBUSTNESS TARGET)**:
   $$\frac{\text{adj\_close}_{t+1} - \text{adj\_close}_t}{\text{adj\_close}_t}$$
6. **`target_next_session_dir_c2c` (ROBUSTNESS TARGET)**:
   $$1 \text{ if } \text{target\_next\_session\_ret\_c2c} > 0 \text{ else } 0$$

### 3.3 Leakage-Free Predictive Market Features (Available at Session $t$ 9:15 AM Open)
- `open_gap_pct`: $(\text{adj\_open}_t - \text{adj\_close}_{t-1}) / \text{adj\_close}_{t-1}$.
- `ret_1d_lag1`: Prior trading session return $(\text{adj\_close}_{t-1} - \text{adj\_close}_{t-2}) / \text{adj\_close}_{t-2}$.
- `rsi_14_lag1`: 14-period RSI calculated as of session $t-1$ close.
- `macd_lag1`, `macd_signal_lag1`, `macd_diff_lag1`: MACD indicators as of session $t-1$ close.
- `rolling_vol_5d_lag1`, `rolling_vol_21d_lag1`: Historical volatility as of session $t-1$ close.
- `adj_close_sma_ratio_20_lag1`: Price-to-20SMA ratio as of session $t-1$ close.
- `intraday_range_pct_lag1`: Prior session high-low range as of session $t-1$.
- `market_regime_lag1`: `HIGH_VOL` / `LOW_VOL` regime classification as of session $t-1$.

### 3.4 Multi-Tier Sentiment Features
- **Volume & Presence Flags**:
  - `has_direct_news`: Binary flag (1 if $\ge 1$ direct article rolled to session $t$, 0 otherwise).
  - `direct_news_count`: Total direct articles for the stock rolled to session $t$.
  - `has_sector_news`: Binary flag for sector news.
  - `sector_news_count`: Total sector articles rolled to session $t$.
  - `market_news_count`: Total broad market articles rolled to session $t$.
- **Direct Sentiment (Zero-encoded when `has_direct_news == 0`)**:
  - FinBERT: `finbert_direct_pos`, `finbert_direct_neu`, `finbert_direct_neg`, `finbert_direct_polarity`, `finbert_direct_confidence`.
  - RoBERTa: `roberta_direct_pos`, `roberta_direct_neu`, `roberta_direct_neg`, `roberta_direct_polarity`, `roberta_direct_confidence`.
  - Loughran–McDonald: `lm_direct_pos_count`, `lm_direct_neg_count`, `lm_direct_unc_count`, `lm_direct_polarity`, `lm_direct_subjectivity`.
- **Sector Sentiment (Separate columns, NEVER substituted for direct sentiment)**:
  - `finbert_sector_pos`, `finbert_sector_neu`, `finbert_sector_neg`, `finbert_sector_polarity`.
  - `roberta_sector_pos`, `roberta_sector_neu`, `roberta_sector_neg`, `roberta_sector_polarity`.
  - `lm_sector_pos_count`, `lm_sector_neg_count`, `lm_sector_polarity`.
- **Market Sentiment**:
  - `finbert_market_pos`, `finbert_market_neu`, `finbert_market_neg`, `finbert_market_polarity`.
  - `roberta_market_pos`, `roberta_market_neu`, `roberta_market_neg`, `roberta_market_polarity`.
  - `lm_market_polarity`.

### 3.5 Temporal Sentiment Windows
- **1-Session Lag ($t-1$)**:
  - `finbert_direct_polarity_lag1`, `roberta_direct_polarity_lag1`, `lm_direct_polarity_lag1`, `direct_news_count_lag1`.
  - `finbert_sector_polarity_lag1`, `roberta_sector_polarity_lag1`, `lm_sector_polarity_lag1`.
  - `finbert_market_polarity_lag1`, `roberta_market_polarity_lag1`, `lm_market_polarity_lag1`.
- **3-Session Rolling Average ($t, t-1, t-2$)**:
  - `finbert_direct_polarity_roll3`, `roberta_direct_polarity_roll3`, `lm_direct_polarity_roll3`, `direct_news_count_roll3`.
  - `finbert_sector_polarity_roll3`, `roberta_sector_polarity_roll3`, `lm_sector_polarity_roll3`.
  - `finbert_market_polarity_roll3`, `roberta_market_polarity_roll3`, `lm_market_polarity_roll3`.
- **7-Session Rolling Average ($t \dots t-6$)**:
  - `finbert_direct_polarity_roll7`, `roberta_direct_polarity_roll7`, `lm_direct_polarity_roll7`, `direct_news_count_roll7`.
  - `finbert_sector_polarity_roll7`, `roberta_sector_polarity_roll7`, `lm_sector_polarity_roll7`.
  - `finbert_market_polarity_roll7`, `roberta_market_polarity_roll7`, `lm_market_polarity_roll7`.
