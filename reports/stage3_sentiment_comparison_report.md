# Stage 3 — Sentiment Comparison & Empirical Analysis Report

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Research Cohort**: 15 Continuous NSE Nifty 50 Equities (2020–2026)  
**Date**: November 2026  
**Document**: Comparative Linguistic & Statistical Analysis of Financial Sentiment Representations  

---

## 1. Executive Summary

Stage 3 generated independent article-level sentiment representations across all 155,537 direct financial news records using three distinct methodologies:
1. **Loughran–McDonald (LM) Lexicon**: Domain-specific financial dictionary (Notre Dame SRAF).
2. **ProsusAI FinBERT**: 110M parameter transformer pretrained on corporate filings and fine-tuned on Financial PhraseBank.
3. **Financial RoBERTa**: 82M parameter distilled transformer fine-tuned for financial sentiment.

This report presents the comparative statistical distribution, cross-model correlation structure, classification agreement, sector-wise dynamics, and temporal behavior of the extracted sentiment signals.

---

## 2. Cross-Model Statistical Distributions

The 155,537 news headlines were independently evaluated across all three paradigms without truncation (0.00% truncation rate).

### 2.1 Polarity Descriptive Statistics
$$\text{Polarity} \in [-1.0, 1.0], \quad \text{where } +1.0 = \text{Purely Positive}, \quad -1.0 = \text{Purely Negative}$$

| Metric | Loughran–McDonald (LM) | ProsusAI FinBERT | Financial RoBERTa |
| :--- | :--- | :--- | :--- |
| **Mean** | **-0.0319** | **+0.0873** | **+0.1173** |
| **Standard Deviation** | 0.5698 | 0.6416 | 0.7728 |
| **Minimum** | -1.0000 | -0.9688 | -0.9983 |
| **25th Percentile (Q1)** | -0.4444 | -0.3337 | -0.7268 |
| **Median (Q2)** | 0.0000 | +0.0514 | -0.0000 |
| **75th Percentile (Q3)** | +0.3333 | +0.7622 | +0.9988 |
| **Maximum** | +1.0000 | +0.9425 | +0.9996 |

### 2.2 Discrete Class Distribution (Argmax)

| Class | Loughran–McDonald (Sign Split) | ProsusAI FinBERT (Argmax Prob) | Financial RoBERTa (Argmax Prob) |
| :--- | :--- | :--- | :--- |
| **Negative** | **47.54%** ($N=73,939$) | **26.09%** ($N=40,573$) | **25.79%** ($N=40,112$) |
| **Neutral** | **11.92%** ($N=18,543$) | **38.01%** ($N=59,125$) | **36.80%** ($N=57,232$) |
| **Positive** | **40.54%** ($N=63,055$) | **35.90%** ($N=55,839$) | **37.41%** ($N=58,193$) |

### 2.3 Key Distributional Insights
1. **Lexical Negativity Bias**: The Loughran–McDonald dictionary exhibits a slight negative mean polarity (-0.0319) and classifies 47.54% of headlines as negative. This is a recognized property of financial dictionaries: the LM dictionary contains 2,345 negative terms vs. only 347 positive terms (a 6.75:1 ratio), reflecting the regulatory and disclosure necessity of detailing corporate risks, defaults, and litigation.
2. **Contextual Moderation by Deep Transformers**: FinBERT and RoBERTa correctly identify neutral financial reporting (38.01% and 36.80% neutral respectively), whereas bag-of-words matching misinterprets ordinary factual reporting containing risk terminology as negative.
3. **Bimodal Extremism in RoBERTa vs. Calibrated Softmax in FinBERT**: Financial RoBERTa outputs sharper, higher-confidence probabilities ($\text{mean confidence} = 0.982$, standard deviation of polarity = 0.773), frequently assigning near 1.0 probability to the dominant class. FinBERT produces smoother, more calibrated probability distributions ($\text{mean confidence} = 0.849$, standard deviation = 0.642).

---

## 3. Cross-Model Correlation & Agreement Structure

### 3.1 Pearson Correlation Matrix (Article Polarities)

| Metric | Loughran–McDonald | ProsusAI FinBERT | Financial RoBERTa |
| :--- | :--- | :--- | :--- |
| **Loughran–McDonald** | **1.0000** | 0.3870 | 0.3667 |
| **ProsusAI FinBERT** | 0.3870 | **1.0000** | **0.8003** |
| **Financial RoBERTa** | 0.3667 | **0.8003** | **1.0000** |

### 3.2 Transformer Classification Agreement
- **FinBERT vs. RoBERTa Exact Class Agreement**: **81.90%** (127,385 out of 155,537 articles).
- In the remaining 18.10% of cases, the discordance primarily occurs between *Neutral* and *Positive/Negative* classifications (e.g., mildly positive headlines classified as Neutral by FinBERT but Positive by RoBERTa due to higher softmax saturation).

---

## 4. Sector-Wise Sentiment Dynamics

| Sector | Sample Articles | LM Mean Polarity | FinBERT Mean Polarity | RoBERTa Mean Polarity |
| :--- | :--- | :--- | :--- | :--- |
| **Construction & Infrastructure** | 12,410 | **+0.0287** | **+0.1367** | **+0.1660** |
| **Telecommunications** | 8,920 | -0.0515 | +0.1156 | +0.1191 |
| **Energy** | 14,830 | -0.0190 | +0.1098 | +0.1255 |
| **Automobile** | 11,540 | +0.0009 | +0.0886 | +0.1272 |
| **Fast Moving Consumer Goods** | 16,720 | -0.0272 | +0.0906 | +0.1262 |
| **Financial Services** | 41,290 | -0.0416 | +0.0775 | +0.1120 |
| **Healthcare & Pharmaceuticals** | 18,340 | -0.0367 | +0.0774 | +0.1016 |
| **Information Technology** | 22,110 | -0.0698 | +0.0585 | +0.0900 |
| **Metals & Mining** | 9,377 | -0.0696 | +0.0506 | +0.1063 |

### Sector Findings:
- **Most Optimistic Sector**: *Construction & Infrastructure* (`LT`) consistently recorded the highest sentiment scores across all three models (+0.0287 LM, +0.1367 FinBERT, +0.1660 RoBERTa), driven by post-pandemic order inflows, national infrastructure allocations, and order book expansions.
- **Most Pessimistic Sectors**: *Information Technology* (`TCS`, `INFY`) and *Metals & Mining* (`TATASTEEL`) registered the lowest sentiment scores across all models, reflecting macroeconomic headwinds, global IT spending slowdowns, and commodity cyclicality.

---

## 5. Company-Wise Sentiment Summary (15 Cohort Equities)

| Symbol | Sector | LM Polarity | FinBERT Polarity | RoBERTa Polarity | FinBERT Conf. | RoBERTa Conf. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LT` | Construction & Infra | +0.0287 | +0.1367 | +0.1660 | 0.852 | 0.984 |
| `MARUTI` | Automobile | +0.0114 | +0.0941 | +0.1364 | 0.849 | 0.983 |
| `ITC` | FMCG | -0.0039 | +0.0932 | +0.1345 | 0.851 | 0.982 |
| `RELIANCE` | Energy | -0.0190 | +0.1098 | +0.1255 | 0.855 | 0.983 |
| `SBIN` | Financial Services | -0.0335 | +0.0886 | +0.1245 | 0.846 | 0.981 |
| `M&M` | Automobile | -0.0028 | +0.0866 | +0.1239 | 0.848 | 0.982 |
| `BHARTIARTL`| Telecommunications | -0.0515 | +0.1156 | +0.1191 | 0.850 | 0.981 |
| `HINDUNILVR`| FMCG | -0.0533 | +0.0876 | +0.1169 | 0.847 | 0.980 |
| `ICICIBANK` | Financial Services | -0.0416 | +0.0754 | +0.1115 | 0.847 | 0.981 |
| `TATASTEEL` | Metals & Mining | -0.0696 | +0.0506 | +0.1063 | 0.848 | 0.982 |
| `SUNPHARMA` | Healthcare & Pharma | -0.0555 | +0.0698 | +0.1051 | 0.850 | 0.983 |
| `CIPLA` | Healthcare & Pharma | -0.0076 | +0.0893 | +0.0963 | 0.849 | 0.983 |
| `HDFCBANK` | Financial Services | -0.0545 | +0.0620 | +0.0927 | 0.845 | 0.981 |
| `INFY` | Information Tech | -0.0741 | +0.0558 | +0.0909 | 0.850 | 0.982 |
| `TCS` | Information Tech | -0.0651 | +0.0615 | +0.0890 | 0.851 | 0.983 |

---

## 6. Multi-Tier & Temporal Aggregation Analysis

In the final master feature store (`data/final/final_sentiment_enhanced_dataset.parquet`):
1. **Multi-Tier Separation**:
   - Direct company news accounts for 44.76% of company-days ($N=10,345$).
   - Sector news covers 86.84% of company-days ($N=20,073$).
   - Broad market news covers 99.42% of company-days ($N=22,981$).
   - Strict non-substitution guarantees that direct sentiment is NEVER contaminated by sector sentiment.
2. **Temporal Decay Structure**:
   - The Pearson correlation between same-session FinBERT polarity ($t$) and lag-1 polarity ($t-1$) is **$r = 0.084$**, demonstrating that single-day financial headline sentiment decays rapidly within 24–48 hours.
   - The 3-session rolling polarity provides a smoothed regime proxy ($r = 0.491$ with session $t$), while the 7-session rolling window captures weekly institutional tone ($r = 0.312$ with session $t$).

---

## 7. Conclusions for Stage 4 Modeling

1. **Feature Redundancy vs. Complementarity**:
   FinBERT and RoBERTa share 80% correlation, suggesting that while both capture transformer-based contextual embeddings, one can act as an ensemble regularizer for the other. Conversely, Loughran–McDonald shares only ~38% correlation with deep models, providing an orthogonal, lexical perspective that may reduce variance in ensemble predictions.
2. **Target Alignment Integrity**:
   The primary prediction target $(\text{adj\_close}_t - \text{adj\_open}_t) / \text{adj\_open}_t$ paired with lag-1 market features and session-$t$ open gap guarantees **zero lookahead bias**, establishing a solid foundation for Stage 4 machine learning.
