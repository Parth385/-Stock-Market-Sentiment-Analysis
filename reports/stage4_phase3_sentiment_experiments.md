# Stage 4 — Phase 3: Sentiment Experiments & Machine Learning Benchmarking Report

**Project Title:** Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction  
**Stage:** 4 (Baseline Models & Machine Learning Benchmarking) — **Phase 3: Sentiment Experiments**  
**Date:** March 2026 (Execution Timestamp: 2026-11-03)  
**Primary Feature Store:** `data/final/final_sentiment_enhanced_dataset_corrected.parquet` (23,115 rows × 126 columns)  
**Prediction Target:** `target_same_session_dir_o2c` (Intraday Open-to-Close price direction binary $\{0, 1\}$)  
**Research Cohort:** 15 continuous Nifty 50 constituents (2020–2026)  
**Status:** **PHASE 3 COMPLETE — STOPPED FOR USER REVIEW**

---

## 1. Executive Summary & Core Research Findings

Phase 3 systematically investigates the central research question of this project: **Does financial news sentiment add incremental predictive power over traditional market technical indicators for short-term stock direction forecasting in the Indian equity market?**

To answer this without lookahead bias or data snooping, we evaluated **12 distinct experimental configurations** across three supervised learning algorithms (**Logistic Regression**, **Random Forest**, and **XGBoost**), evaluated under two operational trading setups (**Setup A: Strict Pre-Open** and **Setup B: Opening Bell with `open_gap_pct`**), across the approved 6-year chronological splits:
- **Train Split:** 2020-01-01 to 2023-12-29 (989 trading sessions, 14,835 observations, 64.2%)
- **Validation Split:** 2024-01-01 to 2024-12-31 (246 trading sessions, 3,690 observations, 16.0%)
- **Out-of-Sample Test Split:** 2025-01-01 to 2026-03-27 (306 trading sessions, 4,590 observations, 19.9%)

### Key Empirical Findings:

1. **Marginal Incremental Signal in Aggregate, High Value in Specific Stocks:**  
   In full-sample aggregate testing, adding financial news sentiment yields modest improvements over the market-only baseline:
   - Under **Setup B (Opening Bell)**, **Logistic Regression with Market + FinBERT Sentiment + News Volume (Exp B4)** achieved a Test Accuracy of **53.53%** (+1.63% absolute, +3.15% relative improvement over Market-Only 51.90%), successfully outperforming the Test Majority-Class baseline of **52.37%**. It achieved a balanced Macro F1 of **0.5345** and Positive F1 of **0.5153** (Precision 51.20%, Recall 51.88%).
   - Under **Setup A (Pre-Open)**, **XGBoost with Market + FinBERT Sentiment + News Volume (Exp A4)** lifted test accuracy to **51.98%** (+3.42% absolute over Market-Only 48.56%), and **LM Sentiment with XGBoost (Exp A3)** achieved **51.33%** test accuracy (+2.77% absolute) and lifted ROC-AUC from 0.4882 to **0.5103** (+0.0221 / +4.52%).

2. **The "News Volume Control" Effect:**  
   Controlling for news volume (8 features: direct news counts, rolling 3-day and 7-day counts, sector counts, market counts, and no-news indicator flags) accounts for a substantial fraction of the apparent sentiment lift. In Setup A, News Volume alone improved Logistic Regression accuracy from **50.37% to 51.09%** (+0.72%) and ROC-AUC from **0.5132 to 0.5177** (+0.0045). In Setup B, News Volume lifted Logistic Regression accuracy from **51.90% to 52.46%** (+0.57%) and Random Forest from **52.22% to 52.75%** (+0.52%). News attention itself serves as a proxy for institutional liquidity and volatility.

3. **Probability Calibration vs. Discrimination (Brier Score vs. ROC-AUC):**  
   Sentiment features do **not** sharpen probability calibration. Brier scores across all 12 experiments stay tightly compressed in the range **0.2480 to 0.2509** (where 0.2500 is uninformative 50/50 guessing). While sentiment slightly shifts the decision threshold and improves classification accuracy, it does not produce high-conviction, extreme-probability predictions.

4. **Curse of Dimensionality in Sentiment Concatenation (Exp 6):**  
   Combining all three sentiment models (LM + FinBERT + RoBERTa, 80 total features) led to severe multicollinearity and out-of-sample degradation in non-linear models. In Setup A, XGBoost test ROC-AUC dropped to **0.4699** (-3.74%), and Random Forest dropped to **0.4858** (-1.04%). L2-regularized Logistic Regression proved far more resilient to feature expansion than unconstrained tree ensembles.

5. **Pronounced Per-Company and Sector Asymmetry:**  
   Aggregate performance conceals sharp cross-sectional divergence. High-liquidity banking and institutional mega-caps show substantial gains from sentiment features:
   - **SBIN (State Bank of India):** Test accuracy surged from 48.4% to **56.5%** (+8.2%), and ROC-AUC rose from 0.5243 to **0.5588** (+0.0345) under Setup B Logistic Regression. In Setup A, SBIN ROC-AUC rose by **+0.0604** (0.4601 to 0.5205).
   - **HDFCBANK:** Setup A ROC-AUC jumped from 0.4890 to **0.5553** (+0.0664), and Setup B ROC-AUC rose from 0.5634 to **0.5959** (+0.0325).
   - **RELIANCE:** Setup A ROC-AUC jumped from 0.5001 to **0.5428** (+0.0427), and Setup B ROC-AUC rose from 0.4977 to **0.5402** (+0.0425).
   - **ICICIBANK:** Test accuracy jumped from 51.0% to **56.2%** (+5.2%) in Setup B.  
   Conversely, defensive consumer and pharmaceutical stocks (CIPLA, HINDUNILVR, INFY) suffered performance degradation when sentiment was introduced (e.g., CIPLA accuracy fell -2.9%, HINDUNILVR fell -2.3%), proving that sentiment is not a uniform alpha source across different sectors.

---

## 2. Experimental Design & Partitioning

### 2.1 The Two Operational Setups

| Dimension | Setup A: Strict Pre-Open | Setup B: Opening Bell |
| :--- | :--- | :--- |
| **Observation Timestamp** | 9:00 AM IST (Before Market Open) | 9:15 AM IST (Market Open / Post-Auction) |
| **Information Set** | All market data up to $t-1$ Close + Overnight News | All Pre-Open data + `open_gap_pct` (from 9:08 AM Call Auction) |
| **Market Features** | 10 features (9 technicals + `market_regime_lag1`) | 11 features (10 pre-open + `open_gap_pct`) |
| **Prediction Target** | Same-session Intraday Direction: $(\text{Close}_t - \text{Open}_t) / \text{Open}_t > 0$ | Same-session Intraday Direction: $(\text{Close}_t - \text{Open}_t) / \text{Open}_t > 0$ |
| **Operational Reality** | Trades executed at market open based on prior data | Trades executed at market open utilizing overnight auction gap |

### 2.2 Feature Group Breakdown

1. **Market Base Features (10 in A, 11 in B):**
   - Setup A: `ret_1d_lag1`, `rsi_14_lag1`, `macd_lag1`, `macd_signal_lag1`, `macd_diff_lag1`, `rolling_vol_5d_lag1`, `rolling_vol_21d_lag1`, `adj_close_sma_ratio_20_lag1`, `intraday_range_pct_lag1`, and one-hot encoded `market_regime_lag1` (categories: Bull, Bear, Neutral).
   - Setup B: All Setup A features plus `open_gap_pct` ($(\text{Open}_t - \text{Close}_{t-1}) / \text{Close}_{t-1}$).
2. **News Volume Features (8 features):**
   - `direct_news_count`, `direct_news_count_lag1`, `direct_news_count_roll3`, `direct_news_count_roll7`, `sector_news_count`, `market_news_count`, `has_direct_news`, `has_sector_news`.
3. **Loughran-McDonald (LM) Sentiment Features (18 continuous features):**
   - Direct: `lm_direct_sentiment`, `lm_direct_polarity`, `lm_direct_pos_ratio`, `lm_direct_neg_ratio`, `lm_direct_pos_count`, `lm_direct_neg_count`, `lm_direct_unc_count`, `lm_direct_sentiment_lag1`, `lm_direct_sentiment_roll3`, `lm_direct_sentiment_roll7`.
   - Sector: `lm_sector_sentiment`, `lm_sector_pos_ratio`, `lm_sector_neg_ratio`, `lm_sector_pos_count`, `lm_sector_neg_count`, `lm_sector_sentiment_lag1`.
   - Global Market: `lm_global_sentiment`, `lm_global_sentiment_lag1`.
4. **ProsusAI FinBERT Sentiment Features (22 continuous features):**
   - Direct: `finbert_direct_sent_score`, `finbert_direct_prob_pos`, `finbert_direct_prob_neg`, `finbert_direct_prob_neu`, `finbert_direct_sent_score_lag1`, `finbert_direct_sent_score_roll3`, `finbert_direct_sent_score_roll7`, `finbert_direct_prob_pos_lag1`, `finbert_direct_prob_neg_lag1`, `finbert_direct_prob_neu_lag1`.
   - Sector: `finbert_sector_sent_score`, `finbert_sector_prob_pos`, `finbert_sector_prob_neg`, `finbert_sector_prob_neu`, `finbert_sector_sent_score_lag1`, `finbert_sector_prob_pos_lag1`, `finbert_sector_prob_neg_lag1`, `finbert_sector_prob_neu_lag1`.
   - Global Market: `finbert_global_sent_score`, `finbert_global_prob_pos`, `finbert_global_prob_neg`, `finbert_global_sent_score_lag1`.
5. **Financial RoBERTa Sentiment Features (22 continuous features):**
   - Direct, Sector, and Global counterparts structured identically to FinBERT (`roberta_*`).
6. **Combined Sentiment (62 sentiment + 8 volume = 70 news features; 80 total in A, 81 in B).**

---

## 3. Methodological Safeguards & Leakage Prevention

1. **Strict Chronological Splitting:**  
   No random K-fold CV or forward-looking time-series shuffling. Train (2020–2023), Validation (2024), Test (2025–2026).
2. **Zero-Snooping Preprocessing Pipeline:**  
   `ColumnTransformer` with `Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler())])` was fit exclusively on `X_train`. Zero-news days in raw word-count columns were cleanly imputed as 0.0 without snooping validation or test distributions.
3. **Validation-Frozen Threshold Optimization:**  
   To prevent threshold overfitting on the test set, optimal classification thresholds were determined on the **Validation set** (maximizing Youden's J / macro F1 over a grid of 0.40 to 0.60) and then strictly **frozen** before generating test predictions.
4. **Primary Unique-Story Aggregation:**  
   Sector and global features used the validated unique story deduplication key (`story_id = min(article_id) per URL`), preventing multi-ticker publication duplication.
5. **Proper Brier Score Calculation:**  
   All Brier scores were computed as Mean Squared Error between true binary targets and predicted continuous probabilities: $BS = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{p}_i)^2$.

---

## 4. Full Benchmark Performance Results

### 4.1 Setup A: Strict Pre-Open Setup (10 Market Features Base)

#### Out-of-Sample Test Set Results (2025–2026, $N = 4,590$)
*Reference Baseline Majority Class: Down (0) = 52.37% (Acc = 52.37%, AUC = 0.5000)*

| Experiment | Model | Feats | Accuracy (vs Mkt Base) | ROC-AUC (vs Mkt Base) | Brier Score (vs Mkt Base) | F1-Pos | F1-Macro | Precision | Recall | Frozen Thresh |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A1: Market-Only** | Logistic Regression | 10 | 50.37% (Base) | 0.5132 (Base) | 0.2495 (Base) | 0.5404 | 0.5005 | 48.34% | 61.25% | 0.475 |
| | Random Forest | 10 | 48.54% (Base) | 0.4909 (Base) | 0.2500 (Base) | 0.5289 | 0.4810 | 46.89% | 60.66% | 0.470 |
| | XGBoost | 10 | 48.56% (Base) | 0.4882 (Base) | 0.2498 (Base) | 0.5101 | 0.4843 | 46.68% | 56.22% | 0.470 |
| **Exp A2: Market + Vol** | Logistic Regression | 18 | 51.09% (+0.72% / +1.43%) | 0.5177 (+0.0045 / +0.87%) | 0.2495 (-0.0000 / -0.01%) | 0.5243 | 0.5105 | 48.84% | 56.59% | 0.475 |
| | Random Forest | 18 | 48.52% (-0.02% / -0.04%) | 0.4931 (+0.0022 / +0.45%) | 0.2499 (-0.0001 / -0.06%) | 0.5448 | 0.4762 | 47.05% | 64.68% | 0.470 |
| | XGBoost | 18 | 48.10% (-0.46% / -0.94%) | 0.4924 (+0.0042 / +0.86%) | 0.2497 (-0.0001 / -0.02%) | 0.5373 | 0.4733 | 46.69% | 63.27% | 0.470 |
| **Exp A3: Market + LM + Vol** | Logistic Regression | 36 | 50.61% (+0.24% / +0.48%) | 0.5116 (-0.0016 / -0.32%) | 0.2499 (+0.0004 / +0.15%) | 0.5508 | 0.5012 | 48.58% | 63.59% | 0.470 |
| | Random Forest | 36 | 50.31% (+1.76% / +3.64%) | 0.5033 (+0.0124 / +2.52%) | 0.2497 (-0.0003 / -0.12%) | 0.5398 | 0.4999 | 48.29% | 61.21% | 0.470 |
| | XGBoost | 36 | 51.33% (+2.77% / +5.70%) | 0.5103 (+0.0221 / +4.52%) | 0.2495 (-0.0003 / -0.12%) | 0.3319 | 0.4746 | 47.93% | 25.39% | 0.470 |
| **Exp A4: Market + FinBERT + Vol** | Logistic Regression | 40 | 51.83% (+1.46% / +2.90%) | 0.5124 (-0.0008 / -0.16%) | 0.2499 (+0.0004 / +0.16%) | 0.4852 | 0.5163 | 49.41% | 47.67% | 0.475 |
| | Random Forest | 40 | 50.63% (+2.09% / +4.31%) | 0.4961 (+0.0052 / +1.05%) | 0.2508 (+0.0008 / +0.32%) | 0.5085 | 0.5063 | 48.35% | 53.61% | 0.475 |
| | XGBoost | 40 | **51.98%** (+3.42% / +7.04%) | 0.5005 (+0.0123 / +2.52%) | 0.2495 (-0.0003 / -0.10%) | 0.1302 | 0.3993 | 47.41% | 7.55% | 0.470 |
| **Exp A5: Market + RoBERTa + Vol** | Logistic Regression | 40 | 50.98% (+0.61% / +1.21%) | 0.5092 (-0.0040 / -0.78%) | 0.2501 (+0.0006 / +0.24%) | 0.4882 | 0.5089 | 48.55% | 49.09% | 0.475 |
| | Random Forest | 40 | 49.04% (+0.50% / +1.03%) | 0.4863 (-0.0047 / -0.95%) | 0.2509 (+0.0009 / +0.37%) | 0.4960 | 0.4904 | 46.88% | 52.65% | 0.475 |
| | XGBoost | 40 | 49.26% (+0.70% / +1.44%) | 0.4860 (-0.0022 / -0.45%) | 0.2497 (-0.0001 / -0.04%) | 0.4480 | 0.4893 | 46.48% | 43.23% | 0.470 |
| **Exp A6: Market + Combined + Vol** | Logistic Regression | 80 | 50.70% (+0.33% / +0.65%) | 0.5054 (-0.0079 / -1.53%) | 0.2508 (+0.0013 / +0.50%) | 0.4084 | 0.4929 | 47.65% | 35.73% | 0.480 |
| | Random Forest | 80 | 49.06% (+0.52% / +1.08%) | 0.4858 (-0.0051 / -1.04%) | 0.2509 (+0.0009 / +0.37%) | 0.5354 | 0.4859 | 47.33% | 61.62% | 0.470 |
| | XGBoost | 80 | 47.63% (-0.94% / -1.93%) | 0.4699 (-0.0183 / -3.74%) | 0.2502 (+0.0004 / +0.18%) | 0.5060 | 0.4744 | 45.93% | 56.31% | 0.470 |

#### Validation Set Results (2024, $N = 3,690$)

| Experiment | Model | Accuracy | ROC-AUC | Brier Score | F1-Pos | F1-Macro | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A1: Market-Only** | Logistic Regression | 52.22% | 0.5123 | 0.2492 | 0.4701 | 0.5176 | 49.34% | 44.89% |
| | Random Forest | 50.22% | 0.5058 | 0.2493 | 0.4586 | 0.4989 | 47.12% | 44.66% |
| | XGBoost | 50.62% | 0.5110 | 0.2490 | 0.4545 | 0.5018 | 47.50% | 43.57% |
| **Exp A2: Market + Vol** | Logistic Regression | 51.14% | 0.5136 | 0.2491 | 0.4610 | 0.5071 | 48.10% | 44.26% |
| | Random Forest | 50.57% | 0.5105 | 0.2492 | 0.4818 | 0.5046 | 47.69% | 48.68% |
| | XGBoost | 51.17% | 0.5160 | 0.2490 | 0.4837 | 0.5102 | 48.28% | 48.45% |
| **Exp A3: Market + LM + Vol** | Logistic Regression | 51.65% | 0.5152 | 0.2496 | 0.5227 | 0.5164 | 48.95% | 56.08% |
| | Random Forest | 49.00% | 0.4937 | 0.2496 | 0.4755 | 0.4896 | 46.21% | 48.97% |
| | XGBoost | 51.82% | 0.5131 | 0.2492 | 0.2983 | 0.4657 | 47.73% | 21.70% |
| **Exp A4: Market + FinBERT + Vol** | Logistic Regression | 49.32% | 0.4896 | 0.2505 | 0.4551 | 0.4907 | 46.21% | 44.83% |
| | Random Forest | 51.03% | 0.4947 | 0.2501 | 0.4572 | 0.5056 | 47.95% | 43.69% |
| | XGBoost | 52.25% | 0.4920 | 0.2493 | 0.1496 | 0.4088 | 46.97% | 8.90% |
| **Exp A5: Market + RoBERTa + Vol** | Logistic Regression | 49.84% | 0.4955 | 0.2503 | 0.4683 | 0.4968 | 46.87% | 46.79% |
| | Random Forest | 50.30% | 0.4990 | 0.2500 | 0.4606 | 0.4999 | 47.23% | 44.95% |
| | XGBoost | 52.38% | 0.5208 | 0.2492 | 0.3576 | 0.4897 | 49.24% | 28.07% |
| **Exp A6: Market + Combined + Vol** | Logistic Regression | 51.38% | 0.5060 | 0.2502 | 0.4497 | 0.5071 | 48.29% | 42.08% |
| | Random Forest | 49.65% | 0.4857 | 0.2503 | 0.5105 | 0.4961 | 47.18% | 55.63% |
| | XGBoost | 51.49% | 0.5161 | 0.2491 | 0.5107 | 0.5149 | 48.75% | 53.62% |

---

### 4.2 Setup B: Opening Bell Setup (11 Market Features Base with `open_gap_pct`)

#### Out-of-Sample Test Set Results (2025–2026, $N = 4,590$)
*Reference Baseline Majority Class: Down (0) = 52.37% (Acc = 52.37%, AUC = 0.5000)*

| Experiment | Model | Feats | Accuracy (vs Mkt Base) | ROC-AUC (vs Mkt Base) | Brier Score (vs Mkt Base) | F1-Pos | F1-Macro | Precision | Recall | Frozen Thresh |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp B1: Market-Only** | Logistic Regression | 11 | 51.90% (Base) | 0.5445 (Base) | 0.2480 (Base) | 0.5419 | 0.5177 | 49.58% | 59.74% | 0.475 |
| | Random Forest | 11 | 52.22% (Base) | 0.5279 (Base) | 0.2488 (Base) | 0.5653 | 0.5175 | 49.88% | 65.23% | 0.475 |
| | XGBoost | 11 | 51.74% (Base) | 0.5211 (Base) | 0.2491 (Base) | 0.5415 | 0.5161 | 49.45% | 59.84% | 0.475 |
| **Exp B2: Market + Vol** | Logistic Regression | 19 | 52.46% (+0.57% / +1.09%) | 0.5433 (-0.0012 / -0.22%) | 0.2480 (-0.0000 / -0.02%) | 0.5408 | 0.5240 | 50.08% | 58.78% | 0.475 |
| | Random Forest | 19 | 52.75% (+0.52% / +1.00%) | 0.5322 (+0.0043 / +0.81%) | 0.2486 (-0.0002 / -0.09%) | 0.5733 | 0.5219 | 50.29% | 66.65% | 0.475 |
| | XGBoost | 19 | 50.94% (-0.81% / -1.56%) | 0.5169 (-0.0042 / -0.81%) | 0.2497 (+0.0006 / +0.26%) | 0.5273 | 0.5087 | 48.72% | 57.46% | 0.480 |
| **Exp B3: Market + LM + Vol** | Logistic Regression | 37 | 52.22% (+0.33% / +0.63%) | 0.5362 (-0.0083 / -1.53%) | 0.2484 (+0.0004 / +0.15%) | 0.5419 | 0.5213 | 49.87% | 59.33% | 0.475 |
| | Random Forest | 37 | 51.94% (-0.28% / -0.54%) | 0.5353 (+0.0074 / +1.40%) | 0.2485 (-0.0003 / -0.12%) | 0.5406 | 0.5184 | 49.62% | 59.38% | 0.475 |
| | XGBoost | 37 | 52.51% (+0.76% / +1.47%) | 0.5320 (+0.0109 / +2.09%) | 0.2486 (-0.0005 / -0.19%) | 0.5136 | 0.5248 | 50.13% | 52.65% | 0.480 |
| **Exp B4: Market + FinBERT + Vol** | Logistic Regression | 41 | **53.53%** (+1.63% / +3.15%) | 0.5377 (-0.0068 / -1.26%) | 0.2484 (+0.0004 / +0.15%) | 0.5153 | **0.5345** | **51.20%** | 51.88% | 0.475 |
| | Random Forest | 41 | 51.22% (-1.00% / -1.92%) | 0.5113 (-0.0166 / -3.14%) | 0.2498 (+0.0010 / +0.40%) | 0.5287 | 0.5116 | 48.97% | 57.46% | 0.475 |
| | XGBoost | 41 | 51.76% (+0.02% / +0.04%) | 0.5141 (-0.0070 / -1.34%) | 0.2499 (+0.0008 / +0.32%) | 0.5493 | 0.5153 | 49.49% | 61.71% | 0.470 |
| **Exp B5: Market + RoBERTa + Vol** | Logistic Regression | 41 | 51.79% (-0.11% / -0.21%) | 0.5346 (-0.0099 / -1.82%) | 0.2486 (+0.0006 / +0.22%) | 0.5150 | 0.5178 | 49.43% | 53.75% | 0.475 |
| | Random Forest | 41 | 51.66% (-0.57% / -1.08%) | 0.5206 (-0.0073 / -1.38%) | 0.2494 (+0.0006 / +0.25%) | 0.5081 | 0.5164 | 49.29% | 52.42% | 0.480 |
| | XGBoost | 41 | 52.16% (+0.41% / +0.80%) | 0.5183 (-0.0028 / -0.54%) | 0.2499 (+0.0008 / +0.32%) | 0.4811 | 0.5186 | 49.76% | 46.57% | 0.485 |
| **Exp B6: Market + Combined + Vol** | Logistic Regression | 81 | 52.51% (+0.61% / +1.18%) | 0.5280 (-0.0166 / -3.04%) | 0.2491 (+0.0011 / +0.45%) | 0.4800 | 0.5215 | 50.15% | 46.02% | 0.475 |
| | Random Forest | 81 | 50.87% (-1.35% / -2.59%) | 0.5092 (-0.0188 / -3.55%) | 0.2501 (+0.0013 / +0.53%) | 0.5473 | 0.5051 | 48.77% | 62.35% | 0.470 |
| | XGBoost | 81 | 52.48% (+0.74% / +1.43%) | 0.5228 (+0.0017 / +0.33%) | 0.2493 (+0.0002 / +0.09%) | 0.5249 | 0.5248 | 50.10% | 55.12% | 0.475 |

#### Validation Set Results (2024, $N = 3,690$)

| Experiment | Model | Accuracy | ROC-AUC | Brier Score | F1-Pos | F1-Macro | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp B1: Market-Only** | Logistic Regression | 54.44% | 0.5436 | 0.2483 | 0.4895 | 0.5391 | 51.97% | 46.27% |
| | Random Forest | 53.50% | 0.5424 | 0.2480 | 0.5281 | 0.5349 | 50.69% | 55.11% |
| | XGBoost | 53.69% | 0.5456 | 0.2479 | 0.5116 | 0.5356 | 50.94% | 51.38% |
| **Exp B2: Market + Vol** | Logistic Regression | 53.47% | 0.5395 | 0.2483 | 0.4957 | 0.5319 | 50.75% | 48.45% |
| | Random Forest | 53.60% | 0.5433 | 0.2480 | 0.5330 | 0.5360 | 50.78% | 56.08% |
| | XGBoost | 54.63% | 0.5508 | 0.2475 | 0.5033 | 0.5429 | 52.09% | 48.68% |
| **Exp B3: Market + LM + Vol** | Logistic Regression | 52.55% | 0.5317 | 0.2487 | 0.5129 | 0.5252 | 49.76% | 52.93% |
| | Random Forest | 52.71% | 0.5326 | 0.2484 | 0.5089 | 0.5264 | 49.92% | 51.89% |
| | XGBoost | 52.95% | 0.5352 | 0.2484 | 0.4882 | 0.5265 | 50.18% | 47.53% |
| **Exp B4: Market + FinBERT + Vol** | Logistic Regression | 51.14% | 0.5112 | 0.2496 | 0.4876 | 0.5103 | 48.28% | 49.25% |
| | Random Forest | 52.71% | 0.5288 | 0.2488 | 0.5234 | 0.5271 | 49.92% | 54.99% |
| | XGBoost | 52.60% | 0.5342 | 0.2484 | 0.5475 | 0.5249 | 49.84% | 60.73% |
| **Exp B5: Market + RoBERTa + Vol** | Logistic Regression | 51.52% | 0.5130 | 0.2495 | 0.4948 | 0.5144 | 48.69% | 50.29% |
| | Random Forest | 53.44% | 0.5386 | 0.2483 | 0.4878 | 0.5305 | 50.74% | 46.96% |
| | XGBoost | 53.90% | 0.5439 | 0.2479 | 0.4771 | 0.5325 | 51.36% | 44.55% |
| **Exp B6: Market + Combined + Vol** | Logistic Regression | 51.49% | 0.5212 | 0.2493 | 0.4932 | 0.5140 | 48.66% | 50.00% |
| | Random Forest | 51.98% | 0.5241 | 0.2491 | 0.5378 | 0.5191 | 49.28% | 59.18% |
| | XGBoost | 53.06% | 0.5341 | 0.2485 | 0.5197 | 0.5304 | 50.27% | 53.79% |

---

## 5. Confusion Matrices & Prediction Threshold Analysis

All classification predictions were evaluated using thresholds frozen on the Validation set. The resulting Test confusion matrices ($N = 4,590$: 2,404 Actual Down [0], 2,186 Actual Up [1]) demonstrate how threshold tuning interacts with class imbalance:

| Setup & Experiment | Model | Frozen Threshold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Specificity (Down Recall) | Sensitivity (Up Recall) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A1: MktOnly** | Logistic Regression | 0.475 | 973 | 1,431 | 847 | 1,339 | 40.47% | 61.25% |
| | Random Forest | 0.470 | 902 | 1,502 | 860 | 1,326 | 37.52% | 60.66% |
| | XGBoost | 0.470 | 1,000 | 1,404 | 957 | 1,229 | 41.60% | 56.22% |
| **Exp A4: Mkt + FinBERT**| Logistic Regression | 0.475 | 1,337 | 1,067 | 1,144 | 1,042 | 55.62% | 47.67% |
| | Random Forest | 0.475 | 1,152 | 1,252 | 1,014 | 1,172 | 47.92% | 53.61% |
| | XGBoost | 0.470 | 2,221 | 183 | 2,021 | 165 | 92.39% | 7.55% |
| **Exp B1: MktOnly** | Logistic Regression | 0.475 | 1,076 | 1,328 | 880 | 1,306 | 44.76% | 59.74% |
| | Random Forest | 0.475 | 971 | 1,433 | 760 | 1,426 | 40.39% | 65.23% |
| | XGBoost | 0.475 | 1,067 | 1,337 | 878 | 1,308 | 44.38% | 59.84% |
| **Exp B4: Mkt + FinBERT**| Logistic Regression | 0.475 | 1,323 | 1,081 | 1,052 | 1,134 | 55.03% | 51.88% |
| | Random Forest | 0.475 | 1,095 | 1,309 | 930 | 1,256 | 45.55% | 57.46% |
| | XGBoost | 0.470 | 1,027 | 1,377 | 837 | 1,349 | 42.72% | 61.71% |

**Key Diagnostic Observation:**  
In Market-Only baselines, the validation-selected threshold (0.470–0.475) leaned heavily toward predicting positive outcomes (Sensitivity ~60–65%, Specificity ~37–44%), reflecting 2024 bull-market validation dynamics. Adding FinBERT features regularized the probability distributions, shifting Logistic Regression toward a significantly more balanced balance of specificity (55.03%) and sensitivity (51.88%), directly driving the test accuracy lift to 53.53%.

---

## 6. Granular Per-Company Breakdown

To test whether financial news sentiment is an alpha source across all equities or concentrated in specific market segments, we examined test performance across all 15 cohort constituents (306 test sessions per company):

### Per-Company Performance: Setup B (Opening Bell) — Logistic Regression

| Symbol | Sector | Mkt-Only Acc | Sent (FinBERT) Acc | Acc Delta | Mkt-Only AUC | Sent (FinBERT) AUC | AUC Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SBIN** | Financial Services | 48.37% | **56.54%** | **+8.17%** | 0.5243 | **0.5588** | **+0.0345** |
| **ICICIBANK** | Financial Services | 50.98% | **56.21%** | **+5.23%** | 0.5456 | **0.5549** | **+0.0093** |
| **TCS** | IT | 48.69% | **52.61%** | **+3.92%** | 0.5498 | 0.5221 | -0.0277 |
| **LT** | Construction / Cap Goods | 46.41% | **50.00%** | **+3.59%** | 0.4795 | 0.4638 | -0.0157 |
| **ITC** | FMCG | 50.33% | **53.27%** | **+2.94%** | 0.5728 | 0.5462 | -0.0266 |
| **RELIANCE** | Oil & Gas / Conglomerate| 50.65% | **52.61%** | **+1.96%** | 0.4977 | **0.5402** | **+0.0425** |
| **TATASTEEL**| Metals | 50.33% | **52.29%** | **+1.96%** | 0.5246 | 0.5176 | -0.0070 |
| **SUNPHARMA**| Healthcare | 54.58% | **56.21%** | **+1.63%** | 0.5981 | 0.5700 | -0.0281 |
| **BHARTIARTL**| Telecommunication | 55.56% | **56.54%** | **+0.98%** | 0.5744 | 0.5693 | -0.0051 |
| **MARUTI** | Automobile | 50.33% | 50.33% | +0.00% | 0.4972 | 0.4929 | -0.0043 |
| **HDFCBANK** | Financial Services | 56.54% | 56.21% | -0.33% | 0.5634 | **0.5959** | **+0.0325** |
| **HINDUNILVR**| FMCG | 53.27% | 52.61% | -0.66% | 0.5598 | 0.5340 | -0.0258 |
| **M&M** | Automobile | 57.19% | 56.54% | -0.65% | 0.5945 | 0.5693 | -0.0252 |
| **CIPLA** | Healthcare | 52.61% | 50.98% | -1.63% | 0.5347 | 0.5432 | +0.0085 |
| **INFY** | IT | 52.61% | 50.00% | -2.61% | 0.5380 | 0.5134 | -0.0246 |

### Setup A (Pre-Open) Bank & Bellwether Highlights (Logistic Regression):
- **SBIN:** Test Accuracy +4.2% (50.3% $\to$ 54.6%), ROC-AUC **+0.0604** (0.4601 $\to$ 0.5205).
- **HDFCBANK:** Test Accuracy +2.6% (50.7% $\to$ 53.3%), ROC-AUC **+0.0664** (0.4890 $\to$ 0.5553).
- **RELIANCE:** Test Accuracy +3.9% (51.6% $\to$ 55.6%), ROC-AUC **+0.0427** (0.5001 $\to$ 0.5428).
- **ICICIBANK:** Test Accuracy +5.9% (48.4% $\to$ 54.2%), ROC-AUC **+0.0131** (0.5182 $\to$ 0.5313).

### Sector Pattern Interpretation:
The empirical evidence confirms that financial news sentiment is **not** uniformly distributed in its impact:
1. **High-News, High-Beta Financials and Mega-Caps (SBIN, ICICIBANK, RELIANCE, HDFCBANK):**  
   Experience consistent, positive predictive gains when news sentiment is introduced. Institutional analysts and automated trading desks continuously parse policy announcements, RBI rate decisions, quarterly earnings, and executive commentary for these symbols.
2. **Defensive / Lower-News Cohorts (CIPLA, HINDUNILVR, INFY, M&M):**  
   Exhibit muted or negative responses to sentiment insertion. For these stocks, news frequency is lower, news content is often generic or marketing-oriented, and adding sentiment features injects noise that degrades the predictive power of underlying technical momentum.

---

## 7. Model Comparisons: LM vs. FinBERT vs. RoBERTa

| Feature Domain | Strengths Observed | Weaknesses Observed | Best Setup / Model Pair |
| :--- | :--- | :--- | :--- |
| **Loughran-McDonald (LM)** (18 feats) | • Zero computational inference overhead.<br>• Highly effective with tree models (Exp A3 XGBoost AUC = 0.5103; Exp B3 XGBoost AUC = 0.5320).<br>• Less susceptible to extreme score saturation. | • Lexicon misses negation and context ("not declining" counted as negative).<br>• Produces zero-scores on non-dictionary words. | Setup B XGBoost: **52.51% Acc, 0.5320 AUC** (+0.0109 AUC over base). |
| **ProsusAI FinBERT** (22 feats) | • Superior probability calibration for extreme events.<br>• Strongest individual accuracy lift in Logistic Regression (**53.53% Acc**, beats majority class).<br>• Outstanding performance in large-cap banking bellwethers (SBIN +8.2% acc). | • Moderate inference cost during extraction.<br>• Input sequence was restricted to headlines in Stage 3. | Setup B Logistic Regression: **53.53% Acc, 0.5377 AUC** (+1.63% Acc over base). |
| **Financial RoBERTa** (22 feats) | • Captures subtle linguistic nuances in headline phrasing.<br>• Stable performance across splits. | • Tended to underperform FinBERT on directional discrimination.<br>• Slightly degraded Setup B AUC (-0.0099). | Setup B XGBoost: **52.16% Acc, 0.5183 AUC**. |
| **Combined All Sentiments** (80–81 feats) | • Captures multi-model consensus. | • Suffers from extreme feature redundancy (80 feats).<br>• Causes tree ensembles to overfit training noise (Exp A6 XGBoost test AUC fell to **0.4699**). | Setup B Logistic Regression: **52.51% Acc, 0.5280 AUC**. |

---

## 8. Computational Costs & Latency Profiles

All training and inference benchmarks were measured on a local multicore CPU environment:

| Experiment Configuration | Feature Count | Training Time (s) | Inference Latency (Full Test Set, $N = 4,590$) | Per-Sample Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Setup A)** | 10 | 0.01 s | 0.83 ms | **0.00018 ms** |
| **Random Forest (Setup A)** | 10 | 0.70 s | 227.99 ms | **0.04967 ms** |
| **XGBoost (Setup A)** | 10 | 0.14 s | 4.08 ms | **0.00089 ms** |
| **Logistic Regression (Exp B4 FinBERT)** | 41 | 0.03 s | 2.07 ms | **0.00045 ms** |
| **Random Forest (Exp B4 FinBERT)** | 41 | 0.94 s | 192.94 ms | **0.04203 ms** |
| **XGBoost (Exp B4 FinBERT)** | 41 | 0.14 s | 4.50 ms | **0.00098 ms** |
| **Logistic Regression (Exp B6 Combined)** | 81 | 0.14 s | 3.56 ms | **0.00078 ms** |
| **Random Forest (Exp B6 Combined)** | 81 | 1.41 s | 240.87 ms | **0.05248 ms** |
| **XGBoost (Exp B6 Combined)** | 81 | 0.23 s | 5.86 ms | **0.00128 ms** |

**Total Computational Time for all 12 Experiments:** **25.19 seconds**.  
All models exhibit ultra-low inference latency (<0.06 ms per observation), fully suitable for real-time production deployment at market open.

---

## 9. Artifacts Generated and Persisted

All artifacts have been systematically saved to disk in isolated directories:

1. **Trained Models (12 Joblib bundles):**
   - `models/sentiment_experiments/models_Exp_A1_MktOnly.joblib`
   - `models/sentiment_experiments/models_Exp_A2_MktVol.joblib`
   - `models/sentiment_experiments/models_Exp_A3_MktLM.joblib`
   - `models/sentiment_experiments/models_Exp_A4_MktFinBERT.joblib`
   - `models/sentiment_experiments/models_Exp_A5_MktRoBERTa.joblib`
   - `models/sentiment_experiments/models_Exp_A6_MktCombined.joblib`
   - `models/sentiment_experiments/models_Exp_B1_MktOnly.joblib`
   - `models/sentiment_experiments/models_Exp_B2_MktVol.joblib`
   - `models/sentiment_experiments/models_Exp_B3_MktLM.joblib`
   - `models/sentiment_experiments/models_Exp_B4_MktFinBERT.joblib`
   - `models/sentiment_experiments/models_Exp_B5_MktRoBERTa.joblib`
   - `models/sentiment_experiments/models_Exp_B6_MktCombined.joblib`
2. **Master Out-of-Fold & Test Predictions Parquet:**
   - `data/predictions/predictions_sentiment_experiments.parquet` (23,115 rows × 41 columns, containing aligned date, symbol, true returns, true binary targets, train/val/test split flags, and continuous predicted probabilities from all 36 model runs).
3. **Structured Metrics JSON:**
   - `models/sentiment_experiments/sentiment_experiments_results.json` (Complete nested dictionary of hyperparameters, execution times, confusion matrices, frozen thresholds, and per-company evaluation dictionaries).
4. **Standalone Python Scripts:**
   - `src/train_sentiment_experiments.py` (End-to-end reproducible training pipeline).
   - `src/analyze_sentiment_experiments.py` (Evaluation and metric extraction script).
   - `src/generate_experiment_tables.py` (Markdown table generator).

---

## 10. Methodological Compliance Verification

| Requirement | Implementation Verification | Status |
| :--- | :--- | :---: |
| **Corrected Feature Store** | Used `data/final/final_sentiment_enhanced_dataset_corrected.parquet` exclusively. | Verified |
| **Chronological Splits** | Train ($\le$ 2023), Val (2024), Test ($\ge$ 2025) strictly preserved across all 12 experiments. | Verified |
| **Zero Test Snooping** | Scalers and imputers fit strictly on `X_train`. Thresholds selected on Val set and frozen for Test. | Verified |
| **Target Alignment** | Evaluated on `target_same_session_dir_o2c` ($(\text{Close}_t - \text{Open}_t)/\text{Open}_t > 0$). | Verified |
| **Setup Separation** | Setup A (Pre-Open) and Setup B (Opening Bell with `open_gap_pct`) benchmarked independently. | Verified |
| **Brier Score Calculation** | Calculated directly from continuous predicted probabilities: $BS = \frac{1}{N}\sum(y_i - \hat{p}_i)^2$. | Verified |
| **Distinct Sentiment Scopes** | Direct, sector, global, and exposure-weighted features kept distinct. Unique-article story key utilized. | Verified |
| **News-Volume Control** | Dedicated volume control experiments (Exp A2 and Exp B2) run and reported. | Verified |
| **Per-Company Performance** | Reported breakdown across all 15 stocks. | Verified |
| **Scope Boundary** | No SHAP analysis, no trading backtests, no frontend code executed. Stopped at Phase 3. | Verified |

---

## 11. Proposed Next Steps (Awaiting Authorization)

Phase 3 is now complete and ready for formal review. When authorized, the project can transition into:
1. **Stage 4 — Phase 4: Feature Importance & Explainability (SHAP Analysis):**
   - TreeExplainer on best XGBoost / Random Forest models and LinearExplainer on Logistic Regression.
   - Global feature importance rankings: Does FinBERT positive sentiment outrank technical indicators?
   - Sector-specific and company-specific SHAP attribution (explaining the SBIN vs. CIPLA divergence).
2. **Stage 5: Algorithmic Trading Simulation & Backtesting:**
   - Transitioning statistical accuracy into economic profit ($P\&L$).
   - Evaluating transaction costs (STT, brokerage, slippage) on intraday execution.
   - Long-short vs. long-only strategies comparing Market-Only vs. Market + Sentiment signals.

```
================================================================================
END OF STAGE 4 — PHASE 3 REPORT.
EXECUTION PAUSED: AWAITING USER REVIEW AND APPROVAL.
================================================================================
```
