# Stage 5 — Experimental Design: Explainability, Interpretation & Error Analysis

**Project Title:** Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction  
**Stage:** 5 (Explainability, Interpretation & Error Analysis) — **Methodological Experimental Design**  
**Date:** March 2026 (Execution Timestamp: 2026-11-03)  
**Deliverable File:** `reports/stage5_experimental_design.md`  
**Status:** **DESIGN PROPOSAL — AWAITING FORMAL USER APPROVAL BEFORE EXECUTION**

---

## 1. Executive Summary & Purpose

Stage 4 established benchmark predictive baselines and evaluated the incremental directional value of financial news sentiment representations (Loughran-McDonald, ProsusAI FinBERT, and Financial RoBERTa) against market technical indicators across 15 continuous Nifty 50 constituents (2020–2026).

Stage 5 shifts focus from **predictive benchmarking** to **interpretability, diagnostic explanation, and failure analysis**. The primary research questions of Stage 5 are:
1. **Linguistic Representation Attribution:** How do different sentiment representations (lexicon vs. transformer embeddings) alter the internal decision logic of linear and non-linear classifiers?
2. **Feature-Group Importance:** What is the relative predictive attribution of technical price signals (`open_gap_pct`, momentum, volatility) versus news volume (attention) versus linguistic sentiment (polarity, probabilities)?
3. **Cross-Sectional Heterogeneity:** Why do high-liquidity banking bellwethers (e.g., SBIN, HDFCBANK, ICICIBANK) exhibit substantial accuracy and discrimination gains from sentiment, while defensive consumer/pharma stocks (e.g., CIPLA, HINDUNILVR) experience performance attenuation?
4. **Error Characterization:** What market conditions, volatility regimes, or news configurations systematically trigger catastrophic model errors (False Positives vs. False Negatives)?

### Strict Methodological Boundaries for Stage 5:
- **Frozen Models:** Stage 5 will strictly utilize frozen models trained and serialized in Stage 4 (`models/sentiment_experiments/*.joblib`). **Zero retraining** will occur.
- **Dataset Preservation:** The corrected Stage 3 master feature store (`final_sentiment_enhanced_dataset_corrected.parquet`) and Stage 4 predictions will remain unaltered.
- **No Causal Inferences:** SHAP (Shapley Additive Explanations) values quantify marginal feature attribution within the statistical mapping learned by the model; they do **not** establish real-world econometric or financial causality.
- **Zero Leakage / No Selection Snooping:** Test-set explanations will be conducted purely as post-hoc diagnostics. No feature selection, threshold tuning, or hyperparameter adjustment will be informed by Stage 5 test-set evaluations.
- **No Downstream Tasks:** Trading backtesting, execution cost simulations, and frontend development remain strictly unauthorized until Stage 5 is approved and concluded.

---

## 2. Model Selection: Representative Model Architecture

Stage 4 produced 36 model instances across 12 experiment configurations. To maximize interpretability depth without redundancy, Stage 5 selects **four primary representative model packages** capturing the key architectural trade-offs:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 5 REPRESENTATIVE MODEL MATRIX                              │
├───────────────────────┬──────────────┬───────────────┬────────────────────────────────────────┤
│ Model Identifier      │ Setup        │ Algorithm     │ Primary Research Role                  │
├───────────────────────┼──────────────┼───────────────┼────────────────────────────────────────┤
│ 1. Exp_B1_MktOnly_LR  │ Setup B      │ Logistic Reg  │ Market-Only Reference Linear Baseline  │
│ 2. Exp_B4_FinBERT_LR  │ Setup B      │ Logistic Reg  │ Top Sentiment Linear Classifier (Acc)  │
│ 3. Exp_B1_MktOnly_XGB │ Setup B      │ XGBoost       │ Market-Only Non-Linear Baseline        │
│ 4. Exp_B3_LM_XGB      │ Setup B      │ XGBoost       │ Top Lexicon Non-Linear Model (AUC)     │
│ 5. Exp_B4_FinBERT_RF  │ Setup B      │ Random Forest │ Bagged Ensemble Comparator             │
│ 6. Exp_A4_FinBERT_XGB │ Setup A      │ XGBoost       │ Pure Pre-Open Benchmark (No Gap)       │
└───────────────────────┴──────────────┴───────────────┴────────────────────────────────────────┘
```

### Detailed Rationale for Focal Models:

1. **`Exp_B4_FinBERT` with Logistic Regression (`models/sentiment_experiments/models_Exp_B4_MktFinBERT.joblib`):**
   - *Why Selected:* Achieved the highest out-of-sample test accuracy (**53.53%**, +1.63% over market-only, beating majority class 52.37%) and highest Macro F1 (**0.5345**).
   - *Analytical Focus:* SHAP will isolate why linear coefficients placed weight on `finbert_direct_prob_pos`, `finbert_direct_sent_score`, or volume flags, and how these features modified the decision boundary relative to `open_gap_pct`.
2. **`Exp_B3_LM` with XGBoost (`models/sentiment_experiments/models_Exp_B3_MktLM.joblib`):**
   - *Why Selected:* Demonstrated the highest non-linear discrimination gain among tree models, lifting Test ROC-AUC from $0.5211$ to **$0.5320$** (+0.0109 / +2.09%) and accuracy to **52.51%**.
   - *Analytical Focus:* Tree SHAP will uncover non-linear interactions between Loughran-McDonald uncertainty/polarity counts (`lm_direct_unc_count`, `lm_direct_polarity`) and pre-existing volatility (`rolling_vol_5d_lag1`).
3. **Market-Only Counterparts (`Exp_B1_MktOnly` LR and XGBoost):**
   - *Why Selected:* Essential counterfactual controls. Comparing SHAP attributions of `open_gap_pct` and `macd_diff_lag1` before and after the introduction of sentiment proves whether sentiment features substitute for, complement, or attenuate technical indicators.

---

## 3. Global Explainability Methodology

We adopt the cooperative game-theoretic framework of Shapley Additive Explanations (SHAP), which guarantees efficiency, symmetry, and additivity:
$$g(z') = \phi_0 + \sum_{j=1}^M \phi_j z'_j$$
where $\phi_j$ is the marginal contribution of feature $j$, and $\phi_0 = \mathbb{E}[f(x)]$ is the base expected model prediction.

### 3.1 Algorithm-Specific SHAP Estimators

| Algorithm | SHAP Estimator | Background Reference Set | Mathematical Handling of Correlation |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `shap.LinearExplainer` | Training Split ($N = 14,835$) with empirical covariance matrix $\Sigma_X$ | Directly accounts for feature inter-correlations using the training feature covariance matrix, preventing probability distortion from collinear variables. |
| **Random Forest** | `shap.TreeExplainer` | `feature_perturbation="tree_path_dependent"` | Follows tree decision paths without requiring an explicit background sample; calculates conditional expectation $\mathbb{E}[f(x) \mid x_S]$ based on node sample counts. |
| **XGBoost** | `shap.TreeExplainer` | `feature_perturbation="tree_path_dependent"` / interventional background | Calculates exact polynomial-time Shapley values using Lundberg et al. (2020) tree SHAP algorithm. Explains log-odds margin output before logistic sigmoid transformation. |

### 3.2 Evaluation Outputs:
1. **Mean Absolute SHAP Summary Plots (Global Importance):**
   $$\text{Importance}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j^{(i)}|$$
2. **SHAP Beeswarm Plots:** Visualizing feature directionality (e.g., does high `finbert_direct_sent_score` shift log-odds positively or negatively across the distribution?).
3. **SHAP Dependence & Interaction Plots:** Examining non-linear thresholds and interaction terms (e.g., `open_gap_pct` vs. `finbert_direct_prob_pos`).

---

## 4. Feature-Group Partitioning & Attribution Aggregation

Because individual sentiment representations produce multiple collinear features (e.g., direct score, positive probability, negative probability, 3-day rolling score, sector aggregate), interpreting features in isolation can dilute perceived importance. 

Stage 5 will implement **Grouped Shapley Attribution**, summing absolute attributions across mutually exclusive, pre-defined functional partitions:
$$\Phi_G^{(i)} = \sum_{j \in G} |\phi_j^{(i)}|$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 5 FUNCTIONAL FEATURE GROUPS                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Core Market Features (9 continuous + 1 categorical):                     │
│    • ret_1d_lag1, rsi_14_lag1, macd_lag1, macd_signal_lag1, macd_diff_lag1  │
│    • rolling_vol_5d_lag1, rolling_vol_21d_lag1, intraday_range_pct_lag1     │
│    • adj_close_sma_ratio_20_lag1, market_regime_lag1                        │
│                                                                             │
│ 2. Opening Auction Signal (1 continuous, Setup B only):                     │
│    • open_gap_pct                                                           │
│                                                                             │
│ 3. News Volume & Market Attention (8 features):                             │
│    • direct_news_count, direct_news_count_lag1, direct_news_count_roll3/7   │
│    • sector_news_count, market_news_count, has_direct_news, has_sector_news │
│                                                                             │
│ 4. Direct Company Sentiment (Linguistic Representation):                    │
│    • FinBERT: finbert_direct_sent_score, finbert_direct_prob_pos/neg/neu    │
│    • LM: lm_direct_sentiment, lm_direct_polarity, lm_direct_pos/neg_ratio   │
│    • RoBERTa: roberta_direct_sent_score, roberta_direct_prob_pos/neg/neu    │
│                                                                             │
│ 5. Temporal Sentiment Dynamics (Persistence & Moving Averages):             │
│    • Lags: direct_sentiment_lag1, prob_pos_lag1, prob_neg_lag1             │
│    • Rolling Windows: direct_sentiment_roll3, direct_sentiment_roll7        │
│                                                                             │
│ 6. Unique-Story Sector Sentiment:                                           │
│    • sector_sentiment, sector_prob_pos/neg/neu, sector_pos/neg_count        │
│                                                                             │
│ 7. Global Macro Sentiment:                                                  │
│    • global_sentiment, global_prob_pos/neg/neu                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

This partitioning allows Stage 5 to report the exact percentage share of total predictive attribution captured by:
$$\text{Share}(G) = \frac{\sum_{i=1}^N \sum_{j \in G} |\phi_j^{(i)}|}{\sum_{i=1}^N \sum_{k=1}^M |\phi_k^{(i)}|} \times 100\%$$

---

## 5. Cross-Sectional & Company-Level Analysis

Stage 4 revealed dramatic performance dispersion across constituents:
- Banking bellwethers (**SBIN**, **ICICIBANK**, **HDFCBANK**) and institutional conglomerates (**RELIANCE**) demonstrated large test accuracy (+5% to +8%) and AUC (+0.03 to +0.06) gains from FinBERT sentiment.
- Defensive equities (**CIPLA**, **HINDUNILVR**, **INFY**) suffered accuracy degradation (-1.5% to -2.9%).

### 5.1 Systematic 15-Company Evaluation (No Cherry-Picking)
Stage 5 will **not** restrict explainability to "winning" stocks. All 15 companies will be systematically evaluated and categorized into three comparative cohort tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          15-COMPANY COMPARATIVE TIERS                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: High News Volume / High Institutional Beta                          │
│   • SBIN, HDFCBANK, ICICIBANK, RELIANCE, TCS                                │
│ Tier 2: Cyclical & Industrial Large-Caps                                    │
│   • LT, TATASTEEL, MARUTI, M&M, BHARTIARTL                                  │
│ Tier 3: Defensive & Low News Intensity                                      │
│   • HINDUNILVR, ITC, SUNPHARMA, CIPLA, INFY                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Company-Level Attribution Hypotheses to Test:
1. **The Signal vs. Noise Hypothesis:** In Tier 1 stocks, is the mean absolute SHAP value of direct sentiment ($\Phi_{\text{direct}}$) significantly higher than in Tier 3 stocks?
2. **The Attention Filter Hypothesis:** In Tier 3 stocks, does the presence of `has_direct_news == 0` force models to rely excessively on noisy sector/global sentiment, leading to misclassification?
3. **The Gap Overwrite Hypothesis:** In companies with high overnight price discovery, does `open_gap_pct` dominate 80%+ of the SHAP attribution, leaving direct sentiment statistically irrelevant?

---

## 6. Error & Failure Analysis Design

To understand when and why sentiment models fail, Stage 5 will dissect predictions into four diagnostic quadrants based on the out-of-sample confusion matrix:

```
                      ACTUAL CLASS (Same-Session Direction)
                      Down (0)                     Up (1)
             ┌───────────────────────────┬───────────────────────────┐
    Down (0) │ True Negatives (TN)       │ False Negatives (FN)      │
PREDICTED    │ Consistent Bearish Signal │ Missed Bullish Moves      │
CLASS        ├───────────────────────────┼───────────────────────────┤
    Up (1)   │ False Positives (FP)      │ True Positives (TP)       │
             │ The "Bull Trap" Errors    │ Successful Long Trades    │
             └───────────────────────────┴───────────────────────────┘
```

### 6.1 Diagnostic Error Investigations:
1. **False Positive ("Bull Trap") Attribution:**  
   Examine instances where models predicted upward direction ($\hat{y} = 1$) but stocks closed lower ($y = 0$).
   - Did extreme positive sentiment (`finbert_direct_prob_pos > 0.85`) induce a false buy signal when pre-open market technicals indicated overbought exhaustion?
   - Did the news represent a "sell the news" corporate event (e.g., strong quarterly earnings already priced in)?
2. **False Negative ("Missed Breakout") Attribution:**  
   Examine instances where models predicted downward direction ($\hat{y} = 0$) but stocks closed higher ($y = 1$).
   - Did absence of direct news (`has_direct_news == 0`) lead to false pessimism?
3. **Boundary Misses vs. High-Conviction Catastrophes:**  
   Distinguish marginal errors ($0.470 \le \hat{p} \le 0.485$) from extreme high-conviction failures ($\hat{p} > 0.65$ with $y = 0$, or $\hat{p} < 0.35$ with $y = 1$).
4. **Stress Period Analysis:**  
   Identify model behavior during the top 10 most volatile trading sessions in the 2025–2026 Test split (e.g., Union Budget presentation, global central bank rate shocks, major geopolitical escalations). Evaluate whether sentiment SHAP attribution acted as a stabilizing factor or amplified prediction error during macro shocks.

---

## 7. Market Regime Analysis

Stage 4 established that market regime is a significant state variable. Stage 5 will analyze feature attribution conditioned on the lagged market regime (`market_regime_lag1`):
1. **Bull Regime ($+1$):** Nifty 50 above its 20-day and 50-day moving averages with positive momentum.
2. **Bear Regime ($-1$):** Nifty 50 below key moving averages with elevated realized volatility.
3. **Neutral / Range-Bound Regime ($0$):** Oscillating prices within Bollinger bands.

### Analytical Design:
- Calculate regime-stratified SHAP matrices: $\Phi_{\text{Bull}}$, $\Phi_{\text{Bear}}$, and $\Phi_{\text{Neutral}}$.
- Test whether **negative sentiment features** (`finbert_direct_prob_neg`, `lm_direct_neg_ratio`) exert higher absolute attribution in Bear regimes than in Bull regimes (asymmetric loss aversion).
- Test whether technical indicators dominate in Neutral regimes while sentiment dominates during Regime transitions.

---

## 8. Robustness, Collinearity & Leakage Safeguards

### 8.1 Mitigating Multicollinearity in SHAP
Financial news sentiment representations exhibit inherent correlations:
- `finbert_direct_prob_pos` is negatively correlated with `finbert_direct_prob_neg` ($r \approx -0.78$).
- Direct sentiment is positively correlated with rolling 3-day sentiment ($r \approx 0.52$).

To prevent spurious attribution:
1. **Grouped Aggregation:** As defined in Section 4, reporting functional group attributions resolves individual collinearity by aggregating complementary facets of the same underlying signal.
2. **Covariance-Aware Linear SHAP:** `shap.LinearExplainer` will be initialized with the empirical feature covariance matrix $\Sigma_{\text{train}}$ computed strictly on the training set, correctly accounting for correlation structures.
3. **Consistency Verification:** Attributions will be cross-checked against standard permutation feature importance to verify that tree SHAP has not over-allocated importance to redundant collinear features.

### 8.2 Strict Avoidance of Test-Set Leakage
- **Frozen Explanations:** SHAP values will be computed on the test set **purely to explain what the frozen models learned from historical data**.
- No parameters, thresholds, feature sets, or model architectures will be adjusted based on Stage 5 observations.
- All background distributions (reference expected values $\mathbb{E}[X]$) will be derived strictly from the **Training Set** ($N = 14,835$), preventing any test-set distribution statistics from leaking into the baseline expectation $\phi_0$.

### 8.3 Causal Discipline
- All reporting in Stage 5 will explicitly describe SHAP values as **predictive importance within the model's functional mapping**, never as causal financial mechanisms.

---

## 9. Implementation Roadmap & Required Environment

### 9.1 Software Requirements
- Package: `shap` (version $\ge 0.44.0$).
- Existing libraries: `scikit-learn` 1.9.0, `xgboost` 3.3.0, `joblib`, `pandas`, `numpy`, `matplotlib`.

### 9.2 Execution Architecture (Upon Approval)
1. **Phase 1: Computation Pipeline (`src/explainability_pipeline.py`)**  
   Load frozen models from `models/sentiment_experiments/*.joblib`. Compute SHAP values for `Exp_B4_FinBERT_LR`, `Exp_B3_LM_XGB`, `Exp_B1_MktOnly_LR`, `Exp_B1_MktOnly_XGB`. Save computed SHAP values to `data/predictions/shap_values_stage5.parquet`.
2. **Phase 2: Global & Grouped Attribution Analysis**  
   Generate summary tables, beeswarm charts, and group share percentages.
3. **Phase 3: Company & Regime Segmentation Analysis**  
   Compute per-company and per-regime SHAP breakdowns across the 15 stocks and 3 regimes.
4. **Phase 4: Error & Failure Case Dissection**  
   Analyze false positives, false negatives, and high-conviction boundary failures.
5. **Phase 5: Synthesis Report (`reports/stage5_explainability_report.md`)**  
   Author the final Stage 5 report integrating all empirical findings.

---

```
================================================================================
STAGE 5 EXPERIMENTAL DESIGN COMPLETE.
EXECUTION PAUSED: NO SHAP ANALYSIS EXECUTED.
AWAITING FORMAL USER REVIEW AND AUTHORIZATION TO PROCEED.
================================================================================
```
