# Stage 4 — Phase 2: Market-Only Baseline Models Evaluation Report

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Research Cohort**: 15 Continuous NSE Nifty 50 Equities (2020–2026)  
**Primary Dataset**: `data/final/final_sentiment_enhanced_dataset_corrected.parquet` (23,115 rows × 126 columns)  
**Date**: November 2026  
**Document**: Stage 4 Phase 2 Empirical Evaluation Report for Market-Only Baseline Models  
**Status**: Phase 2 Complete — Awaiting User Approval to Begin Phase 3 Sentiment Experiments  

---

## 1. Executive Summary & Experimental Protocol

Phase 2 establishes the empirical market-only benchmark for predicting intraday Indian equity price direction:
$$\text{target\_same\_session\_dir\_o2c}_t = \mathbb{I}\left( \frac{\text{adj\_close}_t - \text{adj\_open}_t}{\text{adj\_open}_t} > 0 \right) \in \{0, 1\}$$

In strict compliance with the project guidelines:
1. **Zero Sentiment Features**: All news sentiment polarities, word counts, and volume indicators were strictly excluded.
2. **Leakage-Free Temporal Splits**: Models were trained on **2020–2023** ($N = 14,835$), tuned on **2024** ($N = 3,690$), and evaluated out-of-sample on **2025–2026** ($N = 4,590$).
3. **Zero Data Snooping**: All scalers and encoders were fitted **exclusively on the training split (`fit(X_train)`)**.
4. **Categorical Handling**: `market_regime_lag1` was treated as a categorical feature via `OneHotEncoder`.
5. **Two Execution Setups Benchmarked**:
   - **Setup A (Pure Pre-Open)**: Uses 10 market features strictly observable before market open.
   - **Setup B (Opening Bell)**: Uses 11 market features, adding the 9:08 AM call auction gap (`open_gap_pct`).

---

## 2. Feature Definitions & Exclusions

### 2.1 Market-Only Input Features Used

| Feature Name | Category | Theoretical Function & Construction | Availability Timestamp |
| :--- | :--- | :--- | :--- |
| `ret_1d_lag1` | Continuous | Prior session Close-to-Close return: $(\text{adj\_close}_{t-1} - \text{adj\_close}_{t-2})/\text{adj\_close}_{t-2}$ | Session $t-1$ Close (3:30 PM) |
| `rsi_14_lag1` | Continuous | 14-period Relative Strength Index as of session $t-1$ | Session $t-1$ Close |
| `macd_lag1` | Continuous | 12/26 EMA Difference as of session $t-1$ | Session $t-1$ Close |
| `macd_signal_lag1` | Continuous | 9-day EMA Signal line as of session $t-1$ | Session $t-1$ Close |
| `macd_diff_lag1` | Continuous | MACD Histogram ($\text{macd} - \text{signal}$) as of session $t-1$ | Session $t-1$ Close |
| `rolling_vol_5d_lag1` | Continuous | 5-session annualized historical volatility as of session $t-1$ | Session $t-1$ Close |
| `rolling_vol_21d_lag1` | Continuous | 21-session annualized historical volatility as of session $t-1$ | Session $t-1$ Close |
| `adj_close_sma_ratio_20_lag1` | Continuous | Price-to-20SMA ratio as of session $t-1$: $(\text{adj\_close}_{t-1} / \text{SMA}_{20}) - 1.0$ | Session $t-1$ Close |
| `intraday_range_pct_lag1` | Continuous | Prior session normalized range: $(\text{high}_{t-1} - \text{low}_{t-1})/\text{close}_{t-1}$ | Session $t-1$ Close |
| `market_regime_lag1` | Categorical | Binary volatility regime (`HIGH_VOL` vs `LOW_VOL`) via `OneHotEncoder` | Session $t-1$ Close |
| `open_gap_pct` *(Setup B Only)*| Continuous | Pre-open auction gap: $(\text{adj\_open}_t - \text{adj\_close}_{t-1})/\text{adj\_close}_{t-1}$ | Session $t$ Pre-Open (9:08 AM) |

### 2.2 Excluded Columns (Strict Zero Leakage)
- **Targets**: `target_same_session_dir_o2c`, `target_same_session_ret_o2c`, and all next-session targets.
- **Identifiers**: `date`, `symbol`, `series`, `isin`, `name`, `sector`.
- **Concurrent Session $t$ Prices**: `raw_open`, `raw_high`, `raw_low`, `raw_close`, `adj_open`, `adj_close`, `ret_1d_t0`, `rsi_14_t0`, `macd_t0`, `rolling_vol_5d_t0`.
- **All Sentiment Features**: LM (20 cols), FinBERT (24 cols), RoBERTa (24 cols), Volume & Flags (10 cols).

---

## 3. Model Configurations & Hyperparameters

1. **Logistic Regression (L2 Regularization)**:
   - Penalty: `l2`, Solver: `lbfgs`, `max_iter=1000`, `random_state=42`.
   - Regularization Strength $C$ tuned on Validation set over grid $[10^{-4}, 10^{-3}, 10^{-2}, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]$:
     - Setup A Selected $C$: **$10.0$** (Val AUC = 0.5123)
     - Setup B Selected $C$: **$0.01$** (Val AUC = 0.5436)
2. **Random Forest Classifier**:
   - `n_estimators=300`, `max_depth=6`, `min_samples_split=20`, `min_samples_leaf=10`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`.
3. **XGBoost Classifier**:
   - `learning_rate=0.03`, `max_depth=4`, `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `eval_metric='logloss'`, `random_state=42`, `early_stopping_rounds=30`.
   - Setup A Best Iteration: **24 trees** (Early stopping triggered at step 54)
   - Setup B Best Iteration: **64 trees** (Early stopping triggered at step 94)

---

## 4. Overall Empirical Evaluation Results

### 4.1 Setup A: Pure Pre-Open Market Models (10 Features, Excluding Open Gap)

| Model Architecture | Partition Split | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Macro F1 | ROC-AUC | Brier Score | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | Validation (2024)<br>Test (2025–26) | 52.79%<br>52.37% | 0.00%<br>0.00% | 0.00%<br>0.00% | 0.00%<br>0.00% | 34.37%<br>34.37% | 0.5000<br>0.5000 | 0.2492<br>0.2498 | [1948, 0, 1742, 0]<br>[2404, 0, 2186, 0] |
| **1-Day Momentum Baseline** | Validation (2024)<br>Test (2025–26) | 47.45%<br>48.78% | 44.76%<br>46.36% | 48.28%<br>48.12% | 46.45%<br>47.23% | 47.41%<br>48.72% | 0.4635<br>0.4762 | 0.5255<br>0.5122 | [910, 1038, 901, 841]<br>[1188, 1216, 1134, 1052] |
| **Logistic Regression (L2)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 53.26%<br>52.85%<br>**52.55%** | 52.05%<br>51.61%<br>**52.94%** | 1.64%<br>1.84%<br>**3.29%** | 3.18%<br>3.55%<br>**6.20%** | 36.19%<br>36.17%<br>**37.22%** | 0.5189<br>0.5123<br>**0.5132** | 0.2487<br>0.2492<br>**0.2495** | [7787, 105, 6829, 114]<br>[1918, 30, 1710, 32]<br>**[2340, 64, 2114, 72]** |
| **Random Forest (300t, d=6)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 57.56%<br>53.04%<br>**51.53%** | 78.55%<br>52.00%<br>**46.61%** | 12.82%<br>6.72%<br>**12.26%** | 22.04%<br>11.90%<br>**19.41%** | 46.44%<br>39.94%<br>**42.38%** | 0.6613<br>0.5058<br>**0.4909** | 0.2424<br>0.2493<br>**0.2500** | [7649, 243, 6053, 890]<br>[1840, 108, 1625, 117]<br>**[2097, 307, 1918, 268]** |
| **XGBoost (lr=0.03, d=4)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 53.90%<br>52.90%<br>**52.24%** | 77.08%<br>57.14%<br>**44.83%** | 2.13%<br>0.92%<br>**1.19%** | 4.15%<br>1.81%<br>**2.32%** | 36.90%<br>35.41%<br>**35.36%** | 0.6011<br>0.5110<br>**0.4882** | 0.2467<br>0.2490<br>**0.2498** | [7848, 44, 6795, 148]<br>[1936, 12, 1726, 16]<br>**[2372, 32, 2160, 26]** |

---

### 4.2 Setup B: Opening Bell Market Models (11 Features, Including Open Gap)

| Model Architecture | Partition Split | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Macro F1 | ROC-AUC | Brier Score | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (L2)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 53.06%<br>53.01%<br>**53.77%** | 48.90%<br>51.77%<br>**56.93%** | 6.74%<br>6.72%<br>**12.03%** | 11.85%<br>11.89%<br>**19.86%** | 39.93%<br>39.93%<br>**43.69%** | 0.5373<br>0.5436<br>**0.5445** | 0.2482<br>0.2483<br>**0.2480** | [7403, 489, 6475, 468]<br>[1839, 109, 1625, 117]<br>**[2205, 199, 1923, 263]** |
| **Random Forest (300t, d=6)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 60.06%<br>53.74%<br>**51.26%** | 69.53%<br>52.43%<br>**48.13%** | 26.10%<br>21.70%<br>**29.96%** | 37.95%<br>30.69%<br>**36.93%** | 54.25%<br>47.99%<br>**48.61%** | 0.6450<br>0.5424<br>**0.5279** | 0.2405<br>0.2480<br>**0.2488** | [7098, 794, 5131, 1812]<br>[1605, 343, 1364, 378]<br>**[1698, 706, 1531, 655]** |
| **XGBoost (lr=0.03, d=4)** | Training (2020–23)<br>Validation (2024)<br>**Test (2025–26)** | 57.95%<br>53.96%<br>**51.31%** | 65.15%<br>53.18%<br>**47.83%** | 21.83%<br>20.67%<br>**24.70%** | 32.71%<br>29.76%<br>**32.58%** | 51.06%<br>47.76%<br>**47.24%** | 0.6184<br>0.5456<br>**0.5211** | 0.2425<br>0.2479<br>**0.2491** | [7081, 811, 5427, 1516]<br>[1631, 317, 1382, 360]<br>**[1815, 589, 1646, 540]** |

---

## 5. In-Depth Empirical Findings & Technical Insights

### 5.1 Pre-Open Market Features Have Near-Zero Standalone Power (Setup A)
In Setup A, where only traditional technical indicators (yesterday's RSI, MACD, returns, historical volatility) are available before market open:
1. **Tree Models Fail to Beat Majority Class**: Random Forest (51.53%) and XGBoost (52.24%) fail to beat the zero-intelligence majority class baseline (52.37%) on the forward out-of-sample test set.
2. **Sub-0.50 Test AUCs**: Test ROC-AUCs for tree models collapse to **0.4909 (RF)** and **0.4882 (XGB)**, indicating that complex non-linear combinations of lagged technical indicators overfit in training (train AUC ~0.60–0.66) and degrade to slight noise/inversion out-of-sample.
3. **Logistic Regression Conservatism**: Logistic Regression achieves 52.55% test accuracy only because strong L2 regularization forces it to predict Down on 97% of observations, mimicking the majority class.

### 5.2 The Pre-Open Call Auction Gap (`open_gap_pct`) Injects Real Predictive Signal (Setup B)
When the opening call auction price discovery gap (`open_gap_pct`) is added at 9:08 AM:
1. **Logistic Regression Outperformed Baseline**: Test accuracy reaches **53.77%**, beating the majority-class baseline by **+1.40 percentage points**, with test precision of **56.93%** and test ROC-AUC rising to **0.5445**.
2. **Tree Models Gain Significant AUC**: Test ROC-AUC rises from 0.4909 to **0.5279** (+3.70 AUC points) for Random Forest, and from 0.4882 to **0.5211** (+3.29 AUC points) for XGBoost.
3. **Financial Mechanism**: The overnight auction gap reflects institutional order imbalance resolved between 9:00 AM and 9:08 AM. Stocks gapping sharply up or down exhibit partial intraday continuation/reversal patterns that linear and tree models can systematically exploit.

---

## 6. Per-Company Performance Breakdown (Out-of-Sample Test Set)

Performance varies substantially across the 15 Nifty 50 constituents, reflecting differing institutional market structures:

### Out-of-Sample Test Breakdown ($N = 306$ sessions per company = 4,590 total):

| Company Symbol | Sector | Setup A (Pre-Open) LR AUC | Setup A RF AUC | Setup A XGB AUC | Setup B (Opening Bell) LR AUC | Setup B RF AUC | Setup B XGB AUC | Setup B LR Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SUNPHARMA` | Healthcare | 0.5233 | 0.5652 | 0.5613 | **0.5981** | **0.6067** | **0.5983** | **56.54%** |
| `M&M` | Automobile | **0.5912** | 0.5238 | 0.4954 | **0.5945** | 0.5339 | 0.5283 | 55.88% |
| `HDFCBANK` | Financial Services | 0.4890 | 0.5245 | 0.5230 | **0.5634** | **0.5796** | **0.5896** | 49.02% |
| `BHARTIARTL` | Telecom | 0.5288 | 0.5003 | 0.5182 | **0.5744** | 0.5471 | 0.5535 | 53.59% |
| `ITC` | FMCG | 0.5205 | 0.4619 | 0.4419 | **0.5728** | 0.5195 | 0.5104 | **61.11%** |
| `HINDUNILVR` | FMCG | 0.5364 | 0.5250 | 0.5127 | **0.5598** | **0.5688** | 0.5398 | 53.27% |
| `TCS` | Information Tech | 0.5502 | 0.5260 | 0.5103 | 0.5498 | 0.5188 | 0.5134 | 58.82% |
| `ICICIBANK` | Financial Services | 0.5182 | 0.4626 | 0.4746 | 0.5456 | 0.5225 | 0.5073 | 51.96% |
| `INFY` | Information Tech | 0.5256 | 0.4449 | 0.4440 | 0.5380 | 0.5196 | 0.5132 | 52.94% |
| `CIPLA` | Healthcare | 0.5281 | 0.4764 | 0.4790 | 0.5347 | 0.5216 | 0.4992 | 49.67% |
| `TATASTEEL` | Metals & Mining | 0.4833 | 0.4765 | 0.4868 | 0.5247 | 0.5286 | 0.5261 | 52.94% |
| `SBIN` | Financial Services | 0.4601 | 0.4892 | 0.4975 | 0.5243 | 0.5273 | 0.5280 | 51.63% |
| `LT` | Construction & Infra | 0.4809 | 0.5369 | 0.5136 | 0.4795 | 0.5174 | 0.5166 | 55.23% |
| `RELIANCE` | Energy | 0.5001 | 0.4440 | 0.4596 | 0.4977 | 0.4527 | 0.4509 | 52.29% |
| `MARUTI` | Automobile | 0.4912 | 0.4532 | 0.4535 | 0.4972 | 0.4800 | 0.4603 | 51.63% |

#### Cross-Company Observations:
- **Highest Predictability**: `SUNPHARMA` achieved the highest test AUC across all models (**0.6067 in RF**, 0.5983 in XGB, 0.5981 in LR). `HDFCBANK` reached **0.5896 AUC in XGBoost**. `ITC` achieved the highest raw directional accuracy (**61.11% in LR**).
- **Lowest Predictability**: Mega-caps with the deepest algorithmic and institutional liquidity (`RELIANCE` AUC ~0.45–0.50, `MARUTI` AUC ~0.46–0.49) proved closest to a pure random walk, showing that the most heavily traded index heavyweights exhibit the highest intraday efficiency.

---

## 7. Computational Efficiency & Execution Profile

All baseline training and evaluation executed on CPU across all 15 stocks ($N = 23,115$ observations):

| Model Architecture | Setup A Train Time | Setup A Inference Latency (Full Panel) | Setup B Train Time | Setup B Inference Latency (Full Panel) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.008 seconds** | **0.9 ms** | **0.008 seconds** | **0.8 ms** |
| **Random Forest (300 trees)**| **0.748 seconds** | **214.6 ms** | **0.716 seconds** | **240.2 ms** |
| **XGBoost (300 trees max)** | **0.228 seconds** | **3.1 ms** | **0.120 seconds** | **4.8 ms** |
| **Total Pipeline Run** | — | — | — | **3.28 seconds** |

---

## 8. Artifacts & Outputs Generated

The following reproducible artifacts have been saved:
1. **Trained Models**:
   - [`models/market_baselines/models_setup_a_pre_open.joblib`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/models/market_baselines/models_setup_a_pre_open.joblib) (Fitted preprocessor, Logistic Regression, Random Forest, XGBoost)
   - [`models/market_baselines/models_setup_b_opening_bell.joblib`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/models/market_baselines/models_setup_b_opening_bell.joblib) (Fitted preprocessor, Logistic Regression, Random Forest, XGBoost)
2. **Model Predictions Table**:
   - [`data/predictions/predictions_market_baselines.parquet`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/data/predictions/predictions_market_baselines.parquet) (23,115 rows: includes `date`, `symbol`, split indicator, true target, and predicted probabilities for all 3 models across both setups).
3. **Machine-Readable Results**:
   - [`models/market_baselines/market_baselines_results.json`](file:///c:/Users/Parth%20Sharma/Desktop/Beyond_Sentiment/models/market_baselines/market_baselines_results.json) (Exact metrics, hyperparameters, confusion matrices, and per-company results).

---

## 9. Baseline Benchmark Established for Phase 3 Sentiment Experiments

Phase 2 sets the definitive hurdle that financial news sentiment must surpass in Phase 3:

$$\mathbf{\text{Pre-Open Baseline Hurdle (Setup A)}}: \quad \text{Test Accuracy} = \mathbf{52.55\%}, \quad \text{Test AUC} = \mathbf{0.5132}$$
$$\mathbf{\text{Opening Bell Baseline Hurdle (Setup B)}}: \quad \text{Test Accuracy} = \mathbf{53.77\%}, \quad \text{Test AUC} = \mathbf{0.5445}$$

In Phase 3, we will independently test:
1. `Market + LM Sentiment`
2. `Market + FinBERT Sentiment`
3. `Market + RoBERTa Sentiment`
4. `Market + Multi-Model Sentiment Combinations`

To definitively answer the central research question of **Beyond Sentiment**: *Does financial news sentiment add incremental predictive power over traditional market technical indicators for short-term Indian stock market prediction?*

---

### Awaiting Authorization for Phase 3

In strict adherence to instructions, **execution has stopped after Phase 2.** No sentiment models have been trained, no SHAP analysis was run, and no trading backtests were conducted. 

I await your review and explicit approval to begin **Stage 4 — Phase 3: Sentiment Experiments**.
