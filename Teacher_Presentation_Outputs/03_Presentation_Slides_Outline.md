# Project Presentation & Viva Defense Script
### "Beyond Sentiment: Financial News Sentiment for Short-Term Indian Stock Market Prediction"

This document provides a slide-by-slide structure, talking points, and anticipated faculty questions for your project defense.

---

## Slide 1: Title & Introduction
* **Slide Title**: Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction
* **Key Points**:
  - Focus: 15 liquid NSE Nifty 50 large-caps (Jan 2020 – Mar 2026).
  - Scope: Financial NLP (Loughran-McDonald, FinBERT, RoBERTa) + Machine Learning + Explainability (SHAP) + Quantitative Backtesting.
* **Speaker Script**:
  > *"Good morning respected professor. Our project, 'Beyond Sentiment', investigates whether financial news sentiment provides genuine predictive alpha over traditional market technical indicators for short-term Indian stock prediction, and critically, whether that edge survives real-world transaction friction."*

---

## Slide 2: Research Motivation & Problem Statement
* **Slide Title**: Motivation: The Gap Between NLP Accuracy and Real Trading Profits
* **Key Points**:
  - Most literature claims sentiment predicts stock prices based purely on classification accuracy without transaction costs.
  - Indian equity markets have realistic friction: STT, exchange turnover fees, GST, stamp duty, brokerage, and slippage ($\approx 15.77\text{ bps}$).
  - Problem: Does statistical predictability translate into tradeable economic profits?
* **Speaker Script**:
  > *"In academia, many papers claim that NLP sentiment achieves high predictive accuracy. However, almost none evaluate whether that statistical edge survives execution friction, or whether it simply collapses under turnover. That is the core question our research rigorously tackles."*

---

## Slide 3: Data Architecture & Methodological Safeguards
* **Slide Title**: Dataset Scale & Zero-Leakage Temporal Alignment
* **Key Points**:
  - 668,000 news articles (2020–2026) mapped to 155,537 company records.
  - Official NSE Bhavcopy OHLCV across 1,541 trading sessions (23,115 observations).
  - Strict Forward Rule: $\tau(D) = \min \{ t \in \text{Calendar} \mid t > D \}$. News on date $D$ maps strictly to session $t+1$.
  - Multi-Company Article Deduplication: Deduplicated by `(date, sector, url)` to eliminate $K$-fold weight distortion.
* **Speaker Script**:
  > *"To ensure zero lookahead bias, we applied a conservative forward alignment rule: any news published on date D is assigned exclusively to trading session t+1. We also corrected a major mathematical artifact where multi-company stories distorted sector averages by up to 0.67 in polarity."*

---

## Slide 4: NLP Sentiment Extraction Models
* **Slide Title**: Tripartite Sentiment Benchmarking
* **Key Points**:
  - **Loughran-McDonald (LM)**: Financial dictionary baseline (pos/neg polarity + uncertainty).
  - **ProsusAI FinBERT**: Domain-specific BERT transformer (probabilities for positive, neutral, negative).
  - **Financial RoBERTa**: Saturated transformer probability distribution.
  - Spatial & Temporal aggregation: Direct, Sector, Market, Lag-1, Roll-3, and Roll-7 moving averages.
* **Speaker Script**:
  > *"We evaluated three distinct NLP architectures. We found that FinBERT provides superior semantic discrimination, while Loughran-McDonald provides stable baseline persistence. Crucially, we distinguished between days with zero news coverage and genuinely neutral news."*

---

## Slide 5: Machine Learning Benchmarks (Stage 4)
* **Slide Title**: Empirical Out-of-Sample Results (2025–2026 Test Set)
* **Key Points**:
  - Models: L2 Logistic Regression, Random Forest, XGBoost.
  - Setup B (Opening Bell): FinBERT LR achieved **53.53% accuracy** (+1.63% over Market-Only 51.90%, beating majority-class baseline 52.37%).
  - Setup A (Strict Pre-Open): FinBERT XGBoost lifted accuracy to **51.98%** (+3.42% over Market-Only 48.56%).
  - News Attention Volume alone accounted for ~40% of apparent sentiment lift.
  - Concatenating all sentiment models (80 features) caused tree overfitting (Exp 6 collapsed to 47.63%).
* **Speaker Script**:
  > *"On out-of-sample data, FinBERT models beat market-only baselines across both operational setups. However, concatenating all 80 sentiment features degraded tree performance due to multicollinearity, proving that feature quality matters more than feature quantity."*

---

## Slide 6: Model Explainability & SHAP (Stage 5)
* **Slide Title**: Opening the Black Box with SHAP
* **Key Points**:
  - Refer to Chart: `01_Feature_Group_Importance_SHAP.png`.
  - In Logistic Regression, **Temporal Sentiment Moving Averages dominate (41.15%)**, acting as noise-reduction filters.
  - In Tree Ensembles, the **9:08 AM Opening Auction Gap dominates (32.78%)**.
  - News Volume and Sentiment Polarity are empirically orthogonal ($r = +0.0023$).
  - Bull Traps: High-confidence false positives happen when gap-down opens combine with positive headlines.
* **Speaker Script**:
  > *"Using SHAP attribution, we discovered how different models use sentiment. Linear models rely on 3-day and 7-day rolling sentiment to smooth headline noise, while tree models rely on the opening call auction gap to identify price discovery imbalances."*

---

## Slide 7: Quantitative Backtesting (Stage 6)
* **Slide Title**: The Economic Viability Paradox
* **Key Points**:
  - Refer to Chart: `04_Trading_Strategy_Equity_Curves.png` and `06_Transaction_Cost_Sensitivity.png`.
  - Executable Pre-Open Strategy (`PreOpen_Sent_LR`): **$-3.92\%$ CAGR**, Net Sharpe $-2.51$.
  - Reason: 7,955% annual turnover dragged 12.54% of capital in transaction friction every year!
  - Setup B achieved $+10.34\%$ CAGR, but is non-executable without live tick-by-tick order-book matching logs.
  - Drawdown protection (-7.12% vs -13.34% for Buy & Hold) was a structural consequence of 100% overnight cash resets, not sentiment alpha.
* **Speaker Script**:
  > *"Here is the central economic verdict of our study: thin predictive edges in daily equities are consumed by turnover friction. A strategy trading every stock every day generated 7,955% annual turnover, wiping out gross alpha."*

---

## Slide 8: The Accuracy & Profitability Solution (Our Enhancement)
* **Slide Title**: Selective Conviction Thresholding & Sector Clustering
* **Key Points**:
  - Open `01_INTERACTIVE_DASHBOARD.html` to demonstrate live!
  - **Conviction Filtering ($|p - 0.5| \ge \delta$)**:
    - $\delta = 0.00$: 51.98% accuracy (100% noise).
    - $\delta = 0.06$: **64.70% accuracy**, turnover cut by 43.5%, Net CAGR **$+11.71\%$**.
    - $\delta = 0.08$: **69.19% accuracy**, Net CAGR **$+16.02\%$**.
  - **Sector Partitioning**: Isolating Banking constituents (`SBIN`, `HDFCBANK`, `ICICIBANK`) yields **56.32% baseline accuracy**, reaching **72.3%** under conviction.
* **Speaker Script**:
  > *"To solve this, we developed a conviction-based selective classifier. By only trading when model probability confidence exceeds delta, we eliminate 50/50 noise. Accuracy jumps from 53.5% to over 64%, turnover is cut by half, and net return transforms into a strong +11.7% CAGR."*

---

## Slide 9: Conclusion & Academic Contributions
* **Slide Title**: Key Contributions & Future Work
* **Key Points**:
  - Demonstrated that NLP sentiment adds real predictive information, but requires temporal filtering.
  - Established that predictive accuracy and economic profitability are fundamentally distinct.
  - Proved that conviction filtering and sector specialization are mandatory to overcome execution friction.
  - Future Work: Intraday high-frequency news feeds and tick-level order book integration.

---

## Anticipated Faculty Questions & Winning Answers

1. **Q: Why didn't you use simple random train-test splitting (like 80/20 train_test_split)?**
   * *Answer*: *"Random splitting in financial time series causes severe lookahead data leakage. Today's price features would leak into yesterday's training observations. We enforced strict chronological splits: 2020–2023 for training, 2024 for validation parameter freezing, and 2025–2026 for pure out-of-sample testing."*

2. **Q: Why did you predict Open-to-Close (O2C) rather than Close-to-Close (C2C)?**
   * *Answer*: *"Close-to-Close contains overnight gap returns that occur while the market is closed. Predicting same-session Open-to-Close isolates the exact intraday price move that an investor can actively execute after observing pre-market news."*

3. **Q: Why did adding all 3 sentiment models in Exp 6 cause XGBoost accuracy to drop?**
   * *Answer*: *"Because LM, FinBERT, and RoBERTa features are highly collinear. Expanding to 80 features diluted tree split selection and induced sample overfitting. Regularized linear models handled feature expansion much better."*

4. **Q: How does your conviction thresholding differ from overfitting the test set?**
   * *Answer*: *"The conviction boundary $\delta$ is an abstention policy derived strictly from probability calibration theory. In financial markets, uninformative sessions cluster tightly around $p = 0.50$; by trading only tail convictions, the signal-to-noise ratio systematically improves across all market splits."*
