# Beyond Sentiment: Executive Presentation & Project Dossier
### A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction

> **Evaluation Dossier for Faculty & Project Reviewers**  
> **Domain**: Financial Machine Learning, Natural Language Processing & Quantitative Trading  
> **Target Market**: National Stock Exchange of India (NSE Nifty 50)  
> **Investigation Period**: January 2020 – March 2026 (6+ Years; 1,541 Supervised Trading Sessions)  
> **Research Cohort**: 15 Continuous Liquid Large-Cap Equities across 9 Core Sectors  

---

## 1. Executive Summary & Research Problem

### The Core Research Question
*Does incorporating financial news sentiment extracted from state-of-the-art transformer models (FinBERT, RoBERTa) and financial lexicons (Loughran-McDonald) provide incremental predictive power over traditional technical market indicators for predicting short-term intraday stock price movements in the Indian equity market?*

### The Central Empirical Discovery: The Statistical vs. Economic Viability Paradox
1. **Statistically**: Financial news sentiment delivers a modest, measurable predictive lift over market-only indicators:
   - In **Setup B (Opening Bell Benchmark)**, FinBERT + Logistic Regression achieved **53.53% out-of-sample accuracy** (+1.63% absolute lift over Market-Only 51.90%, outperforming the majority-class baseline of 52.37%).
   - In **Setup A (Pre-Open Executable)**, FinBERT + XGBoost lifted accuracy from **48.56% to 51.98%** (+3.42% absolute lift).
   - In specific high-beta banking equities (`SBIN`, `HDFCBANK`, `ICICIBANK`), directional accuracy reached **56.5%** with ROC-AUC gains of $+0.06$.
2. **Economically**: In real-world market execution net of Indian statutory charges, taxes, exchange fees, and bid-ask slippage ($\approx 15.77\text{ bps}$ round-trip):
   - The thin statistical edge is **consumed by transaction friction and whipsaw turnover**.
   - Mandatory daily trading generated 7,955% annual turnover, dragging capital by 12.54% per year and yielding a net return of **$-3.92\%$ CAGR** (Net Sharpe $-2.51$).
3. **The Breakthrough Solution**: Deploying **Selective Classification ($\delta$-Conviction Thresholding)**:
   - Trading only on high-conviction days ($|p - 0.5| \ge \delta$) filters out uninformative 50/50 noise.
   - Slashes annual turnover from 7,955% to 3,400%, saving 7.1% in annual friction drag.
   - Scales directional accuracy from **53.5% up to 64.7% – 69.2%**, converting a net loss into a **$+11.7\%$ to $+16.0\%$ net CAGR**.

---

## 2. Research Universe & Data Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA ENGINEERING WORKFLOW                              │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Financial News Dataset   │ 668,000 articles (mveen3 / Hugging Face, 2020–2026)         │
│ Market Bhavcopy Data     │ Official daily OHLCV + Corporate Actions (TejHQ / NSE)      │
│ Total Cohort Records     │ 155,537 mapped company-news events                          │
│ Integrated Panel         │ 23,115 company-session observations (15 stocks x 1,541 days)│
│ Features Engineered      │ 126 features (Technicals, News Volume, LM, FinBERT, RoBERTa)│
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### The 15 Supervised Constituents
* **Financial Services**: `HDFCBANK`, `ICICIBANK`, `SBIN`
* **Information Technology**: `TCS`, `INFY`
* **Energy**: `RELIANCE`
* **Fast Moving Consumer Goods (FMCG)**: `HINDUNILVR`, `ITC`
* **Construction & Infrastructure**: `LT`
* **Telecommunications**: `BHARTIARTL`
* **Healthcare & Pharmaceuticals**: `SUNPHARMA`, `CIPLA`
* **Automobile**: `MARUTI`, `M&M`
* **Metals & Mining**: `TATASTEEL`

### Strict Anti-Leakage Temporal Alignment
To eliminate lookahead bias without intraday timestamps:
$$\tau(D) = \min \{ t \in \text{TradingCalendar} \mid t > D \}$$
* News on calendar date $D$ maps strictly to session $t+1$.
* All indicators are calculated exclusively on corporate-action adjusted prices.
* Prediction Target: Same-session Open-to-Close direction $y_t = \mathbb{I}((\text{Close}_t - \text{Open}_t) / \text{Open}_t > 0)$.

---

## 3. NLP Sentiment Methodologies Compared

| Model | Architecture | Text Scope | Key Strengths & Observations |
| :--- | :--- | :--- | :--- |
| **Loughran–McDonald (LM)** | Financial Domain Lexicon | Headlines + Body | Stable baseline; high persistence in word counts; zero black-box risk. |
| **ProsusAI FinBERT** | Domain Transformer (`bert-base`) | Headlines | Superior financial semantics; best probability discrimination; drives linear models. |
| **Financial RoBERTa** | Domain Transformer (`roberta-base`) | Headlines | Saturated class probability distributions; highly correlated with FinBERT. |

### Deduplication Correction (Mathematical Rigor)
* Multi-ticker articles (e.g. *"Sensex rallies led by HDFC Bank and ICICI Bank"*) artificially distort sector averages if duplicated.
* Deduplicated by `(date, sector, url)` for sector and `(date, url)` for market, resolving polarity distortions up to $0.67$ on 98.5% of dates.

---

## 4. Machine Learning Benchmarks (Stage 4)

Chronological Partitioning:
* **Train Set**: 2020–2023 (14,835 observations, 64.2%)
* **Validation Set**: 2024 (3,690 observations, 16.0%) — *Used to freeze classification thresholds*
* **Test Set (Out-of-Sample)**: 2025–March 2026 (4,590 observations, 19.9%)

### Out-of-Sample Test Results Matrix

| Experiment | Setup | Algorithm | Features | Test Accuracy | ROC-AUC | F1-Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Drift** | Both | Benchmark | 0 | 52.37% | 0.5000 | 0.3437 |
| **Market-Only (Reference)** | Setup A | Logistic Reg | 10 | 50.37% | 0.5132 | 0.5005 |
| **Market-Only (Reference)** | Setup A | XGBoost | 10 | 48.56% | 0.4882 | 0.4843 |
| **FinBERT + News Volume** | Setup A | XGBoost | 40 | **51.98%** (+3.42%) | 0.5005 | 0.3993 |
| **LM Sentiment + Volume** | Setup A | XGBoost | 36 | **51.33%** (+2.77%) | **0.5103** | 0.4746 |
| **Market-Only (Reference)** | Setup B | Logistic Reg | 11 | 51.90% | 0.5445 | 0.5177 |
| **FinBERT + News Volume** | Setup B | Logistic Reg | 41 | **53.53%** (+1.63%) | **0.5377** | **0.5345** |
| **All Sentiment Concatenated** | Setup A | XGBoost | 80 | 47.63% (Degraded) | 0.4699 | 0.4744 |

### Critical Takeaways
1. **Linear vs Non-Linear Models**: L2-regularized Logistic Regression proved resilient to feature expansion, while XGBoost suffered split dilution and overfitting when 80 correlated sentiment features were concatenated (Exp 6).
2. **News Attention Volume as a Proxy**: News volume alone accounts for ~40% of the apparent sentiment lift by proxying institutional liquidity and volatility.

---

## 5. Model Explainability & SHAP Analysis (Stage 5)

Using TreeSHAP and LinearSHAP across 7 functional feature groups:
* **Temporal Sentiment Dominates Linear Classifiers (41.15% Attribution)**: 3-day and 7-day rolling sentiment moving averages filter daily headline noise.
* **Opening Auction Gap Dominates Tree Models (32.78% Attribution)**: The 9:08 AM clearing price gap serves as the primary root partition in tree depth.
* **Attention vs. Sentiment Orthogonality**: Empirical correlation between `direct_news_count` and `finbert_direct_polarity` is $r = +0.0023$, confirming that news attention and linguistic polarity provide independent, complementary signals.
* **Bull Trap Dynamics**: High-confidence false positives occur when overnight gap-down opens combine with positive headlines, inducing a false expectation of an intraday mean-reversion bounce that fails to materialize.

---

## 6. Quantitative Backtesting & Real-World Friction (Stage 6)

* **Realistic Cost Model**: $15.77\text{ bps}$ round-trip (STT, SEBI turnover fees, exchange charges, stamp duty, brokerage, and bid-ask slippage).
* **Mandatory Intraday Execution**: 9:15 AM entry, 3:20 PM mandatory cash liquidation (100% overnight cash reset).

### Out-of-Sample Performance Comparison (2025–2026)

| Strategy | Operational Setup | CAGR (%) | Net Sharpe | Max Drawdown | Annual Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PreOpen_Mkt_LR** | Executable (Setup A) | **+2.84%** | **-1.92** | **1.69%** | 2,240% |
| **PreOpen_Sent_LR** | Executable (Setup A) | **-3.92%** | **-2.51** | **7.12%** | 7,955% |
| **SetupB_Mkt_LR** | Bell Benchmark (Setup B) | **+10.34%** | **+0.54** | **3.12%** | 21,085% |
| **SetupB_LM_XGB** | Bell Benchmark (Setup B) | **+10.08%** | **+0.65** | **2.69%** | 11,941% |
| **Equal-Weight Buy & Hold** | Reference Benchmark | **-9.64%** | **-1.59** | **13.34%** | 0% |
| **Intraday 1D Momentum** | Reference Benchmark | **-37.07%** | **-9.31** | **43.05%** | 48,539% |

---

## 7. The Accuracy Scaling Breakthrough

### Why Baseline Models Plateaued at ~53%
In the baseline study, the model was forced to predict **every stock every day** (4,590 trades). On days with zero or neutral news, predicted probabilities hover around 50/50, diluting the score with noise.

### The Solution: Selective Conviction Thresholding ($|p - 0.5| \ge \delta$)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      ACCURACY & TURNOVER SCALING UNDER CONVICTION                      │
├───────────────────┬──────────────┬──────────────┬──────────────────┬───────────────────┤
│ Conviction (δ)    │ Coverage (%) │ Active Trades│ Test Accuracy    │ Projected Net CAGR│
├───────────────────┼──────────────┼──────────────┼──────────────────┼───────────────────┤
│ δ = 0.00 (All)    │ 100.0%       │ 4,590 trades │ 51.98% / 53.53%  │ -3.92% (Loss)     │
│ δ = 0.02          │  82.7%       │ 3,795 trades │ 56.74% (+3.2%)   │ +1.24%            │
│ δ = 0.04          │  68.4%       │ 3,139 trades │ 60.53% (+7.0%)   │ +6.49%            │
│ δ = 0.06 (Optimal)│  56.6%       │ 2,597 trades │ 64.70% (+11.2%)  │ +11.71%           │
│ δ = 0.08 (Decile) │  46.8%       │ 2,148 trades │ 69.19% (+15.7%)  │ +16.02%           │
└───────────────────┴──────────────┴──────────────┴──────────────────┴───────────────────┘
```

### Sector Partitioning Alpha
* By training a dedicated **Financials Model** for `HDFCBANK`, `ICICIBANK`, and `SBIN`, baseline accuracy immediately jumps to **56.32%**, and reaches **72.3%** under $\delta = 0.08$ conviction.

---

## 8. Presentation Deliverables Manifest

This directory contains everything required for presentation and defense:

1. **`00_EXECUTIVE_PROJECT_OVERVIEW.md`** (This master evaluation document)
2. **`01_INTERACTIVE_DASHBOARD.html`** (Self-contained interactive dashboard with sliders and live charts)
3. **`launch_dashboard.bat`** (1-click script to open the dashboard in your default browser)
4. **`02_Visual_Figures_and_Plots/`** (6 high-resolution research figures):
   - `01_Feature_Group_Importance_SHAP.png`
   - `02_Top_15_Features_FinBERT_LR.png`
   - `03_Company_Sentiment_Shares.png`
   - `04_Trading_Strategy_Equity_Curves.png`
   - `05_Portfolio_Drawdowns.png`
   - `06_Transaction_Cost_Sensitivity.png`
5. **`01_Research_Reports/`** (Numbered technical reports for Stages 3, 4, 5, 6, and Accuracy Scaling)
6. **`03_Presentation_Slides_Outline.md`** (Slide-by-slide defense script)
7. **`04_Structured_Data_and_Metrics/`** (Raw numerical JSON results for verification)
