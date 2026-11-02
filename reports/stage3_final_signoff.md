# Stage 3 — Final Methodology Sign-Off & Verification Report

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Research Cohort**: 15 Continuous NSE Nifty 50 Equities (2020–2026)  
**Date**: November 2026  
**Document**: Final Stage 3 Verification, Deduplication Key Audit & Sign-Off Dossier  
**Status**: Stage 3 Final Sign-Off Complete — Awaiting User Approval to Begin Stage 4  

---

## 1. Article Deduplication Keys Verification (`article_id` vs. Canonical `url`)

### 1.1 Structural Taxonomy of Identifiers in `news_clean.parquet`
To verify the deduplication mechanism, we inspected all 155,537 records in [`data/processed/news_clean.parquet`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/data/processed/news_clean.parquet):

| Identifier Attribute | Metric Count | Audit Verification & Findings |
| :--- | :--- | :--- |
| **Total Record Rows** | **155,537** | Every row represents a single **company-mention news observation**. |
| **Unique `article_id`s** | **155,537** | `article_id` is a sequential row identifier ($1 \dots 155,537$). Each company mention receives a distinct `article_id`. |
| **Unique `url`s** | **71,330** | Represents the true **canonical web address** of each physical news article crawled. |
| **Null or Missing URLs** | **0 (0.00%)** | Zero nulls, zero NaNs, zero empty strings (`url != ''`). |
| **Unique `(date, title)` Pairs** | **71,330** | Exactly 1-to-1 with unique URLs. |
| **Unique `(date, url)` Pairs** | **71,330** | Exactly 1-to-1 with unique URLs. |
| **`article_id`s with >1 URL** | **0 (0.00%)** | Each `article_id` maps strictly to a single URL. |
| **URLs with >1 `article_id`** | **30,631** | **42.9% of news stories** mention multiple companies in the 15-stock cohort (up to 14 companies in market roundup articles). |
| **Max `article_id`s per URL** | **14 mentions** | Broad market roundup articles discussing multiple stocks simultaneously. |
| **URL Normalization Audit** | **0 discrepancies** | URLs are strictly canonical (lowercased, scheme `https://`, netloc `economictimes.indiatimes.com`, no trailing slashes, no query parameter drift). |

---

### 1.2 Comparison: `article_id`-Based vs. `url`-Based Deduplication

1. **Why Direct `article_id` Deduplication Fails**:
   Because `article_id` was assigned as a unique primary key to each company-article row ($N = 155,537$), attempting to deduplicate by `subset=['date', 'article_id']` retains all 155,537 rows. **Zero multi-company articles are deduplicated.**
2. **Canonical Story Identification**:
   To establish an integer-based unique story key, we construct:
   $$\text{story\_id} = \min_{i \in \text{Articles}(u)} (\text{article\_id}_i)$$
   where $u$ is the canonical URL. This assigns a unique integer ID ($1 \dots 71,330$) to each distinct news story.
3. **Equivalence Proof**:
   - Deduplication by `(date, story_id)` yields **exactly 71,330 unique broad market records**.
   - Deduplication by `(date, url)` yields **exactly 71,330 unique broad market records**.
   - Row-for-row index equivalence is **100.0% identical** (`(dedup_story.index == dedup_url.index).all() == True`).
   - Sector-level deduplication by `(date, sector, story_id)` and `(date, sector, url)` both yield **exactly 131,800 unique sector records**.

#### Conclusion on Keys:
- **Direct Company News**: Uses `article_id` (155,537 company-specific observations).
- **Unique Sector News**: Uses canonical story key `(date, sector, story_id)` / `(date, sector, url)` (131,800 unique sector stories).
- **Unique Broad Market News**: Uses canonical story key `(date, story_id)` / `(date, url)` (71,330 unique global stories).
- **Exposure-Weighted News**: Separately preserved under `*_exposure` column names.

---

## 2. Corrected Feature Store Validation Dossier

The corrected feature store [`data/final/final_sentiment_enhanced_dataset_corrected.parquet`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/data/final/final_sentiment_enhanced_dataset_corrected.parquet) was verified against all 6 project criteria:

```text
=== CORRECTED FEATURE STORE VALIDATION AUDIT ===
Dimensions: 23,115 rows × 126 columns
Symbols Count: Exactly 15 approved continuous Nifty 50 companies
Observations per Symbol: Exactly 1,541 rows for every single company
Duplicate (date, symbol) Rows: Exactly 0
Primary Target (target_same_session_ret_o2c) Null Count: Exactly 0
Predictive Pre-Open Features Tested for Future Leakage: 43 features -> 0 leakage detected
Stage 2 Original Dataset (final_integrated_dataset.parquet): 10,825,215 bytes -> Unchanged
Stage 3 Original Dataset (final_sentiment_enhanced_dataset.parquet): 16,822,101 bytes -> Unchanged
Corrected Dataset (final_sentiment_enhanced_dataset_corrected.parquet): 17,980,442 bytes -> Generated
```

### 2.1 Complete 126-Column Catalog of Corrected Feature Store
1. **Identification (5 cols)**: `date`, `symbol`, `series`, `isin`, `name`.
2. **Raw Market Prices (9 cols)**: `raw_open`, `raw_high`, `raw_low`, `raw_close`, `last`, `raw_prev_close`, `raw_volume`, `raw_turnover`, `raw_trades`.
3. **Adjusted Prices (6 cols)**: `adj_factor_cumulative`, `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_prev_close`.
4. **Sector & Regime (2 cols)**: `sector`, `market_regime_t0`.
5. **Session $t$ Close Technicals (9 cols, for robustness benchmarking)**: `ret_1d_t0`, `rsi_14_t0`, `macd_t0`, `macd_signal_t0`, `macd_diff_t0`, `rolling_vol_5d_t0`, `rolling_vol_21d_t0`, `adj_close_sma_ratio_20_t0`, `intraday_range_pct_t0`.
6. **Leakage-Free Session $t$ Pre-Open Technicals (11 cols, for primary intraday target)**: `open_gap_pct`, `ret_1d_lag1`, `rsi_14_lag1`, `macd_lag1`, `macd_signal_lag1`, `macd_diff_lag1`, `rolling_vol_5d_lag1`, `rolling_vol_21d_lag1`, `adj_close_sma_ratio_20_lag1`, `intraday_range_pct_lag1`, `market_regime_lag1`.
7. **Prediction Targets (6 cols)**: 
   - `target_same_session_ret_o2c` (**PRIMARY INTRADAY RETURN**: $(\text{adj\_close}_t - \text{adj\_open}_t)/\text{adj\_open}_t$)
   - `target_same_session_dir_o2c` (**PRIMARY INTRADAY DIRECTION**: binary $\{0, 1\}$)
   - `target_next_session_ret_o2c` (ROBUSTNESS)
   - `target_next_session_dir_o2c` (ROBUSTNESS)
   - `target_next_session_ret_c2c` (ROBUSTNESS)
   - `target_next_session_dir_c2c` (ROBUSTNESS)
8. **Direct Company Sentiment (16 cols)**: `direct_news_count`, `finbert_direct_pos`, `finbert_direct_neu`, `finbert_direct_neg`, `finbert_direct_polarity`, `finbert_direct_confidence`, `roberta_direct_pos`, `roberta_direct_neu`, `roberta_direct_neg`, `roberta_direct_polarity`, `roberta_direct_confidence`, `lm_direct_pos_count`, `lm_direct_neg_count`, `lm_direct_unc_count`, `lm_direct_polarity`, `lm_direct_subjectivity`.
9. **Unique Sector Sentiment (12 cols, Primary)**: `sector_news_count`, `finbert_sector_pos`, `finbert_sector_neu`, `finbert_sector_neg`, `finbert_sector_polarity`, `roberta_sector_pos`, `roberta_sector_neu`, `roberta_sector_neg`, `roberta_sector_polarity`, `lm_sector_pos_count`, `lm_sector_neg_count`, `lm_sector_polarity`.
10. **Exposure-Weighted Sector Sentiment (4 cols, Secondary)**: `sector_news_count_exposure`, `finbert_sector_polarity_exposure`, `roberta_sector_polarity_exposure`, `lm_sector_polarity_exposure`.
11. **Unique Broad Market Sentiment (10 cols, Primary)**: `market_news_count`, `finbert_market_pos`, `finbert_market_neu`, `finbert_market_neg`, `finbert_market_polarity`, `roberta_market_pos`, `roberta_market_neu`, `roberta_market_neg`, `roberta_market_polarity`, `lm_market_polarity`.
12. **Exposure-Weighted Market Sentiment (4 cols, Secondary)**: `market_news_count_exposure`, `finbert_market_polarity_exposure`, `roberta_market_polarity_exposure`, `lm_market_polarity_exposure`.
13. **Presence Indicators (2 cols)**: `has_direct_news`, `has_sector_news`.
14. **Temporal Lag-1 Features (10 cols)**: `finbert_direct_polarity_lag1`, `roberta_direct_polarity_lag1`, `lm_direct_polarity_lag1`, `direct_news_count_lag1`, `finbert_sector_polarity_lag1`, `roberta_sector_polarity_lag1`, `lm_sector_polarity_lag1`, `finbert_market_polarity_lag1`, `roberta_market_polarity_lag1`, `lm_market_polarity_lag1`.
15. **Temporal Rolling-3 Features (10 cols)**: `finbert_direct_polarity_roll3`, `roberta_direct_polarity_roll3`, `lm_direct_polarity_roll3`, `direct_news_count_roll3`, `finbert_sector_polarity_roll3`, `roberta_sector_polarity_roll3`, `lm_sector_polarity_roll3`, `finbert_market_polarity_roll3`, `roberta_market_polarity_roll3`, `lm_market_polarity_roll3`.
16. **Temporal Rolling-7 Features (10 cols)**: `finbert_direct_polarity_roll7`, `roberta_direct_polarity_roll7`, `lm_direct_polarity_roll7`, `direct_news_count_roll7`, `finbert_sector_polarity_roll7`, `roberta_sector_polarity_roll7`, `lm_sector_polarity_roll7`, `finbert_market_polarity_roll7`, `roberta_market_polarity_roll7`, `lm_market_polarity_roll7`.

---

## 3. Methodological Comparison Limitation: Input Scope Asymmetry

### Critical Research Caveat
In any comparative discussion between the three sentiment approaches in this project, the following structural asymmetry must be explicitly acknowledged:

1. **Loughran–McDonald (LM)** evaluates **Headline Plus Full Article Body** (`f"{row.title}. {row.news_text}"`):
   - Text scope: Mean 569 words, maximum 13,058 words.
   - Bag-of-words tokenization captures entire body disclosures, regulatory legal boilerplate, historical earnings, and management commentary.
2. **ProsusAI FinBERT & Financial RoBERTa** evaluate **Headline Only** (`row.title`):
   - Text scope: Mean 13.2 words, maximum 41 words (maximum token length = 61 WordPieces / 38 BPE tokens).
   - Deep transformer self-attention captures short, highly condensed, forward-looking editorial summaries.

### Methodological Confounder:
**Differences in empirical performance, correlations, or predictive power between Loughran–McDonald and the transformer models cannot be attributed solely to the underlying NLP methodology (lexicon vs. transformer).** 

They represent a **compound effect** of:
- **Algorithm Type**: Static unigram dictionary matching vs. contextual bidirectional self-attention embeddings; AND
- **Information Scope**: Broad full-article narrative context vs. condensed headline-only framing.

This distinction is documented as an explicit research characteristic of the dataset representations for Stage 4 modeling.

---

## 4. Final Quality Verification Summary

| Verification Item | Status | Verification Evidence |
| :--- | :--- | :--- |
| **Multi-Company Deduplication** | **VERIFIED** | Unique story keys (`(date, story_id)` / `(date, url)`) eliminate duplication distortion across 2,248 market dates and 6,274 sector dates. Exposure metrics retained separately. |
| **Temporal Autocorrelation** | **VERIFIED** | Genuine non-mechanical autocorrelations established: Lag 1 = **0.1161**, Lag 3 = **0.0475**, Lag 7 = **0.0304**. Linear correlation with primary return target confirmed near zero ($|r| < 0.035$). |
| **Target Alignment** | **VERIFIED** | Primary target is same-session Open-to-Close intraday return $(\text{adj\_close}_t - \text{adj\_open}_t)/\text{adj\_open}_t$. All predictive technical features use prices strictly $\le t-1$. |
| **Company Observations** | **VERIFIED** | Exactly 1,541 rows for all 15 companies ($23,115$ total panel rows). Exactly 0 duplicate pairs. |
| **Dataset Preservation** | **VERIFIED** | Stage 2 (`final_integrated_dataset.parquet`) and Stage 3 (`final_sentiment_enhanced_dataset.parquet`) are 100% unchanged. Corrected store saved separately as `final_sentiment_enhanced_dataset_corrected.parquet`. |
| **Automated Test Suite** | **VERIFIED** | **14 out of 14 unit tests passed** in 4.54s (`tests/test_sentiment_integrity.py`). |

---

## 5. Formal Request for Stage 4 Authorization

All Stage 3 requirements, audits, and methodology corrections are complete, verified, and saved to [`reports/stage3_final_signoff.md`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/reports/stage3_final_signoff.md).

Per strict guidelines, execution has stopped. I am prepared to begin Stage 4 (Baseline Models, Market Features Benchmark, and Machine Learning Evaluation) upon your explicit authorization.
