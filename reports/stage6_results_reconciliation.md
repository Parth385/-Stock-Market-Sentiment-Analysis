# Stage 6 — Quantitative Results & Methodology Reconciliation Audit

**Project Title:** Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction  
**Stage:** 6 (Trading Strategy & Backtesting Simulation) — **Post-Execution Results Reconciliation Audit**  
**Date:** March 2026 (Execution Timestamp: 2026-11-03)  
**Deliverable File:** `reports/stage6_results_reconciliation.md`  
**Related Report:** `reports/stage6_backtesting_report.md`  
**Status:** **OFFICIAL AUDIT COMPLETE — ARTIFACTS PRESERVED WITHOUT RERUNNING**

---

## 1. Executive Summary & Audit Mandate

Following the completion of Stage 6 backtesting execution, this audit was conducted to resolve five specific methodological, reporting, and statistical discrepancies:
1. **Test-Set Row Accounting:** Reconciling the reported $N = 4,050$ figure against the approved $N = 4,590$ dataset.
2. **PreOpen_LM_XGB Inactivity:** Auditing the $+8.21\%$ cumulative return ($+6.72\%$ CAGR, $0\%$ volatility, $0\%$ turnover) and confirming zero active trade execution.
3. **Stage 4 Statistical Significance Restraint:** Aligning report claims with the rigorous clustered block permutation test ($p = 0.0820$) from Stage 4.
4. **Sharpe Ratio & Risk-Free Accounting:** Documenting the mathematical excess-return formula, zero-volatility edge case, and why nominal positive returns generate negative Sharpe ratios against the $6.50\%$ cash hurdle.
5. **Downside Capital Preservation Attribution:** De-confounding structural overnight cash insulation from genuine sentiment predictive alpha.

---

## 2. Point 1: Test-Set Row Count Reconciliation

### 2.1 The Issue
- **Approved Dataset Specification:** $15\text{ companies} \times 306\text{ trading sessions} = \mathbf{4,590\text{ observations}}$ in the 2025–2026 Out-of-Sample Test Split.
- **Reporting Discrepancy:** The narrative text in Section 1 of `reports/stage6_backtesting_report.md` (line 17) and Section 1 of `reports/stage6_backtesting_design.md` (line 26) referenced $N = 4,050$ stock-days.

### 2.2 Forensic Data Audit
We audited the underlying serialized prediction and backtest files:
1. **Primary Prediction Store (`data/predictions/predictions_sentiment_experiments.parquet`):**
   ```python
   df_pred = pd.read_parquet('data/predictions/predictions_sentiment_experiments.parquet')
   test_df = df_pred[df_pred['split'] == 'TEST']
   print(test_df.shape)  # Output: (4590, 41)
   print(test_df['symbol'].value_counts())  # Exactly 306 observations for ALL 15 companies
   ```
2. **Backtesting Execution Engine (`src/run_stage6_backtesting.py`):**
   ```python
   # Execution Log Confirmation:
   # Splits: Train=14,835, Val=3,690, Test=4,590
   ```
3. **Daily Equity Curves (`data/predictions/backtest_equity_curves.parquet`):**
   Every single one of the 8 strategies simulated all **306 calendar sessions** in the test split ($1,541$ sessions total across Train, Val, and Test).
4. **Per-Company Coverage in Test Split:**
   - All 15 companies (BHARTIARTL, CIPLA, HDFCBANK, HINDUNILVR, ICICIBANK, INFY, ITC, LT, M&M, MARUTI, RELIANCE, SBIN, SUNPHARMA, TATASTEEL, TCS) have **exactly 306 session records (100.0% coverage)**.
   - Zero observations were dropped due to missing market data, missing predictions, or alignment filters.

### 2.3 Origin of the "4,050" Figure & Reconciliation Verdict
- The figure **$4,050$** was an early draft typographical legacy based on an initial estimate of 270 test trading sessions ($270 \times 15 = 4,050$).
- When the full 2025–2026 test calendar was verified, it spanned 306 trading sessions ($306 \times 15 = 4,590$).
- While Section 2 of the report correctly specified $4,590$ rows and the simulation engine ingested all $4,590$ observations, Section 1 accidentally retained the stale $4,050$ text.
- **Correction:** Section 1 of `reports/stage6_backtesting_report.md` is updated to state $N = 4,590$ stock-days ($306\text{ sessions} \times 15\text{ companies}$). No simulation rerun is required.

---

## 3. Point 2: Audit of PreOpen_LM_XGB Inactivity

### 3.1 The Issue
`PreOpen_LM_XGB` reported:
- Cumulative Net Return: $+8.21\%$
- CAGR: $+6.72\%$
- Annualized Volatility: $0.00\%$
- Net Sharpe Ratio: $0.00$
- Max Drawdown: $0.00\%$
- Annual Traded Turnover: $0.0\%$

### 3.2 Mechanism Audit
We audited the predicted probability distribution of `Exp_A3_MktLM_xgboost_prob` in `predictions_sentiment_experiments.parquet`:
- **Validation Split ($N = 3,690$):**
  - Minimum probability: $0.4560$
  - Maximum probability: $0.4933$
  - Mean probability: $0.4682$, Standard Deviation: $0.0032$
- **Test Split ($N = 4,590$):**
  - Minimum probability: $0.4522$
  - Maximum probability: $0.4933$
  - Mean probability: $0.4684$, Standard Deviation: $0.0035$

### 3.3 The Zero-Trade Explanation
1. During the validation grid search over the 22 valid parameter pairs ($\theta_{\text{long}} \in [0.48, 0.56]$, $\theta_{\text{short}} \in [0.42, 0.50]$ with $\theta_{\text{short}} < \theta_{\text{long}}$), the optimizer selected:
   $$\theta^*_{\text{long}} = 0.50, \quad \theta^*_{\text{short}} = 0.42$$
2. Because the highest predicted probability in both Validation and Test splits was **$0.4933$** ($< 0.50$) and the lowest was **$0.4522$** ($> 0.42$), **100% of all observations fell inside the neutral cash band**:
   $$\theta^*_{\text{short}} (0.42) < \hat{p}_{i, t} < \theta^*_{\text{long}} (0.50) \quad \forall (i, t)$$
3. As a result:
   - **Active Trades Executed:** **0**
   - **Portfolio Capital in Cash:** **100.0% every single day**
   - **Daily Portfolio Return:** $R_{p, t} = \frac{R_f}{252} = \frac{0.0650}{252} = 0.0002579365$
   - **Compounded 306-day Return:** $(1 + \frac{0.065}{252})^{306} - 1 = \mathbf{+8.2116\%}$
   - **CAGR:** $(1.082116)^{252 / 306} - 1 = \mathbf{+6.7150\%}$

### 3.4 Reconciliation Verdict
`PreOpen_LM_XGB` is **NOT an active trading strategy**. It is an **inactive, zero-trade system** that generated pure cash yield identical to the risk-free cash hurdle (`risk_free_cash_6.5pct`). It must be explicitly labeled as **`INACTIVE / ZERO TRADES (100% CASH)`** to prevent confusing risk-free cash accrual with predictive quantitative alpha.

---

## 4. Point 3: Stage 4 Statistical Significance Restraint

### 3.1 The Issue
`reports/stage6_backtesting_report.md` initially claimed that Stage 4 models demonstrated "statistically significant directional accuracy gains ($+1.63\%$ to $+3.42\%$)".

### 3.2 Audit of Stage 4 Statistical Testing (`reports/stage4_phase3_benchmark_reconciliation.md`)
We reviewed the formal hypothesis tests conducted in Stage 4:
1. **Unadjusted McNemar's Test (Assuming i.i.d. observations):**
   - Setup B FinBERT LR vs. Market-Only LR ($+1.63\%$ lift): $\chi^2 = 3.8809, p = 0.0488$ (marginally significant).
   - Setup A FinBERT XGB vs. Market-Only XGB ($+3.42\%$ lift): $\chi^2 = 9.7932, p = 0.00175$ (significant under i.i.d.).
2. **Session-Clustered Block Permutation Test (Correcting for Cross-Stock Contemporaneous Correlation):**
   - Stock returns and model errors across the 15 stocks within the same trading session are **heavily correlated** due to macroeconomic, policy, and market-wide factors.
   - Stage 4 conducted a 2,000-iteration session-clustered block permutation test (permuting entire daily prediction vectors across all 15 stocks simultaneously).
   - **Empirical Clustered p-value:**
     $$\mathbf{p = 0.0820}$$
   - **Formal Statistical Standard:** At the canonical $\alpha = 0.05$ threshold, $p = 0.0820$ is **NOT statistically significant** (it is significant only at the relaxed $\alpha = 0.10$ level).

### 3.3 Reconciliation Verdict
- The claim of "statistically significant directional accuracy gains" in `reports/stage6_backtesting_report.md` is **unsupported under rigorous clustered testing**.
- The $+1.63\%$ accuracy gain must be formally described as **statistically suggestive / marginally significant at $\alpha = 0.10$ ($p = 0.0820$), but non-significant at the conventional $\alpha = 0.05$ level**.
- Other improvements (e.g., $+3.42\%$ for Setup A XGBoost) must be explicitly identified as **descriptive sample improvements** where clustered permutation tests were not evaluated.
- `reports/stage6_backtesting_report.md` is updated to reflect this methodological discipline.

---

## 5. Point 4: Sharpe Ratio & Risk-Free Accounting Methodology

### 5.1 Mathematical Formulations
The simulation engine (`src/backtest_engine.py`) defines performance metrics on daily return series as follows:

1. **Daily Excess Return:**
   $$R_{p, t}^{\text{excess}} = R_{p, t}^{\text{net}} - \frac{R_f}{252}, \quad \text{where } R_f = 0.0650 \text{ (6.50\% p.a. illustrative cash hurdle)}$$
2. **Annualized Net Sharpe Ratio:**
   $$\text{Sharpe} = \frac{\overline{R_{p}^{\text{excess}}}}{\sigma(R_{p}^{\text{net}})} \times \sqrt{252} = \frac{\frac{1}{N}\sum_{t=1}^N (R_{p, t}^{\text{net}} - \frac{R_f}{252})}{\sqrt{\frac{1}{N-1}\sum_{t=1}^N (R_{p, t}^{\text{net}} - \overline{R_{p}^{\text{net}}})^2}} \times \sqrt{252}$$
3. **Annualized Net Sortino Ratio:**
   $$\text{Sortino} = \frac{\overline{R_{p}^{\text{excess}}}}{\sqrt{\frac{1}{N}\sum_{t=1}^N \min\left(0, R_{p, t}^{\text{excess}}\right)^2}} \times \sqrt{252}$$
4. **Handling of Zero-Volatility Edge Cases:**
   If $\sigma(R_{p}^{\text{net}}) \le 10^{-9}$ (as occurs for 100% cash strategies earning constant daily risk-free interest), the denominator approaches zero and excess return is exactly zero. The engine explicitly sets:
   $$\text{Sharpe} = 0.00, \quad \text{Sortino} = 0.00$$
   This prevents division-by-zero errors or false infinite Sharpe ratios.

### 5.2 Resolution: Why Positive Nominal Returns Yield Negative Sharpe Ratios
A key question is why strategies with positive nominal CAGR report negative Sharpe ratios:
- `PreOpen_Mkt_LR`: Nominal CAGR $= \mathbf{+2.84\%}$, Net Sharpe $= \mathbf{-1.92}$
- `PreOpen_Sent_XGB`: Nominal CAGR $= \mathbf{+4.06\%}$, Net Sharpe $= \mathbf{-1.18}$

**Economic Explanation:**
- In institutional finance, Sharpe measures **excess return above the benchmark cash hurdle per unit of risk**.
- Because the cash hurdle is $R_f = 6.50\%$:
  $$\text{CAGR} < R_f \implies \overline{R_{p}^{\text{excess}}} < 0 \implies \text{Sharpe} < 0$$
- A strategy earning $+2.84\%$ or $+4.06\%$ while exposing capital to equity market volatility has delivered an economic return **lower than holding risk-free cash (6.50%)**. The negative Sharpe ratio correctly penalizes the strategy for taking risk to underperform cash.

---

## 6. Point 5: Downside Capital Preservation Interpretation

### 6.1 The Issue
The initial report noted that Setup A strategies preserved capital during the 2025–2026 market downturn (MDD of $1.3\%$ to $7.1\%$ vs. $13.3\%$ for Buy & Hold). It is critical to separate the effect of **overnight cash insulation** from the **predictive contribution of sentiment**.

### 6.2 Forensic Disaggregation: Sentiment vs. Overnight Cash Reset
To isolate the true driver of drawdown reduction, we compare matching systems under identical execution rules:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            DOWNSIDE DRAWDOWN DECOMPOSITION (TEST SET)                       │
├─────────────────────┬──────────────────────────┬──────────┬──────────┬──────────────────────┤
│ Strategy / System   │ Overnight Cash Policy    │ CAGR (%) │ MDD (%)  │ Primary Driver       │
├─────────────────────┼──────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Buy & Hold 15 Eq    │ 0% Cash (24h Exposure)   │  -9.64%  │  13.34%  │ Market Beta Exposure │
│ Intraday Momentum   │ 100% Cash Overnight      │ -37.07%  │  43.05%  │ Whipsawed Execution  │
│ PreOpen_Mkt_LR      │ 100% Cash Overnight      │  +2.84%  │   1.69%  │ Cash Reset + Market  │
│ PreOpen_Sent_LR     │ 100% Cash Overnight      │  -3.92%  │   7.12%  │ Cash Reset + FinBERT │
│ PreOpen_Sent_XGB    │ 100% Cash Overnight      │  +4.06%  │   1.30%  │ Cash Reset + XGBoost │
└─────────────────────┴──────────────────────────┴──────────┴──────────┴──────────────────────┘
```

### 6.3 Critical Analytical Insights
1. **Drawdown Reduction is Structural, Not Predictive:**
   - Buy & Hold lost $-11.58\%$ ($13.34\%$ MDD) primarily because it was exposed to overnight gap-downs and compounding multi-day market drift.
   - The intraday strategies closed all positions at 3:20 PM and held 100% cash overnight, completely eliminating overnight inventory risk.
2. **Sentiment Did NOT Reduce Drawdown Relative to Market-Only:**
   - Comparing `PreOpen_Sent_LR` (FinBERT) directly to its matching market-only baseline `PreOpen_Mkt_LR`:
     - `PreOpen_Mkt_LR` suffered an MDD of only **$1.69\%$**.
     - `PreOpen_Sent_LR` suffered an MDD of **$7.12\%$** (over 4x larger!).
   - Adding FinBERT sentiment actually **increased drawdown** because false short triggers (especially in `MARUTI`) caused unnecessary whipsaw losses and increased friction.
3. **Reconciliation Verdict:**
   `reports/stage6_backtesting_report.md` must **not** credit sentiment with drawdown reduction. The report is updated to explicitly state that drawdown mitigation was a **structural property of the intraday holding period and overnight cash reset**, whereas adding sentiment increased drawdown relative to the market-only baseline.

---

## 7. Audit Summary & Implemented Report Revisions

The following five revisions are implemented in `reports/stage6_backtesting_report.md`:

| Audit Point | Original Text / Issue | Corrected & Reconciled Content |
|---|---|---|
| **1. Row Count** | Stated $N = 4,050$ stock-days in Section 1. | Corrected to $N = 4,590$ stock-days ($306\text{ sessions} \times 15\text{ companies}$, 100% complete coverage). |
| **2. PreOpen_LM_XGB** | Reported $+8.21\%$ cumulative return without operational context. | Explicitly labeled as **`INACTIVE / ZERO TRADES (100% CASH)`**, clarifying that return is 100% illustrative risk-free cash yield, not trading alpha. |
| **3. Significance Claims** | Claimed "statistically significant accuracy improvements ($+1.63\%$ to $+3.42\%$)". | Revised to cite the $p = 0.0820$ session-clustered block permutation test (not significant at $\alpha = 0.05$; suggestive at $\alpha = 0.10$); other lifts classified as descriptive. |
| **4. Sharpe Methodology** | Unexplained negative Sharpe ratios on positive-return strategies. | Documented mathematical formulas, $6.50\%$ cash hurdle excess-return definition, zero-volatility handling ($0.0$), and explained why $\text{CAGR} < 6.5\%$ yields negative Sharpe. |
| **5. Downside Attribution** | Praised sentiment models for capital preservation vs. Buy & Hold. | Clarified that drawdown containment was driven structurally by the overnight cash reset, and that adding FinBERT sentiment actually increased drawdown from $1.69\%$ to $7.12\%$ vs. Market-Only. |

---

```
================================================================================
STAGE 6 RESULTS RECONCILIATION AUDIT COMPLETED.
ALL FINDINGS RECONCILED AGAINST EXISTING ARTIFACTS.
ZERO RE-RUNS EXECUTED. DATASETS AND PREDICTIONS FULLY PRESERVED.
AWAITING FINAL USER APPROVAL.
================================================================================
```
