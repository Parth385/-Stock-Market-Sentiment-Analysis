# Stage 6 — Experimental Design: Trading Strategy, Portfolio Construction & Backtesting Methodology (Final Revision)

**Project Title:** Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction  
**Stage:** 6 (Trading Strategy & Backtesting Simulation) — **Methodological Experimental Design (Final Revision)**  
**Date:** March 2026 (Execution Timestamp: 2026-11-03)  
**Deliverable File:** `reports/stage6_backtesting_design.md`  
**Status:** **FINAL REVISED METHODOLOGY — AWAITING FORMAL USER APPROVAL BEFORE EXECUTION**

---

## 1. Executive Summary & Core Objectives

Stages 1 through 5 established rigorous data engineering, leakage-free feature aggregation, machine learning benchmarking, and explainability analysis. Stage 4 proved that financial news sentiment models (particularly **FinBERT Logistic Regression** and **Loughran-McDonald XGBoost**) achieve directional accuracy improvements (+1.63% to +3.42%) and discrimination gains over market-only baselines.

However, a fundamental tenet of quantitative finance is: **Statistical classification accuracy does not guarantee economic profitability.**

A model with 53.5% directional accuracy can easily generate net trading losses if:
1. Average losses on incorrect days exceed average gains on correct days (asymmetric payoff tails).
2. Statutory transaction charges, exchange turnover fees, Securities Transaction Tax (STT), Goods and Services Tax (GST), and bid-ask slippage consume the thin daily edge.
3. Market execution assumes unexecutable fill prices, double-counts friction, or ignores intraday liquidity realities.

The primary objective of Stage 6 is to bridge the gap between **statistical forecasting** and **economic viability**. Stage 6 designs a realistic, execution-conscious, transaction-cost-adjusted backtesting framework to determine whether financial news sentiment generates authentic, tradeable economic alpha in the Indian stock market (NSE).

### Strict Methodological Boundaries for Stage 6:
- **Frozen Models & Pre-Computed Predictions:** Stage 6 strictly utilizes the predictions already serialized in Stage 4 (`data/predictions/predictions_sentiment_experiments.parquet`). **Zero retraining** or modifications to model weights or datasets will occur.
- **Strict Chronological Separation:** Parameter tuning (thresholds, position sizing, stop-loss rules) is conducted strictly on the **2024 Validation Split** ($N = 3,690$). The **2025–2026 Out-of-Sample Test Set** ($N = 4,590$; 306 sessions $\times$ 15 companies) will be evaluated only once with frozen strategy parameters.
- **Realistic Execution Realities:** All trades reflect actual NSE cash market mechanics: 9:15 AM opening entry, same-session intraday square-off before market close, conservative slippage modeling, and date-appropriate statutory Indian equity taxes.
- **No Downstream Tasks:** Frontend UI development remains unauthorized until Stage 6 is reviewed and approved.

---

## 2. Market Timing, Data Availability & Execution Feasibility

### 2.1 Critical Data Availability Audit: Setup A vs. Setup B

A critical finding from our methodology audit concerns the exact timing and data availability of features in the Stage 3 feature store:

1. **Setup A (Strict Pre-Open Setup) — Primary Executable System:**
   - **Signal Generation Timestamp:** 9:00:00 AM IST.
   - **Information Set:** Strictly pre-market data. All technical indicators computed through $t-1$ Close, plus overnight news published up to 9:00 AM IST on day $t$.
   - **Execution Feasibility:** **100% Executable.** Signals are finalized 15 minutes before the 9:15 AM opening bell. Market-on-Open (MOO) limit or market orders can be placed directly into the trading terminal.
2. **Setup B (Opening Bell Setup) — Non-Executable Exploratory Benchmark:**
   - **Feature Construction Reality:** In `final_sentiment_enhanced_dataset_corrected.parquet`, `open_gap_pct` was computed as:
     $$\text{open\_gap\_pct}_t = \frac{\text{adj\_open}_t - \text{adj\_close}_{t-1}}{\text{adj\_close}_{t-1}}$$
   - **Data Origin:** The variable $\text{adj\_open}_t$ originates from historical Bhavcopy daily records, which record the **final official clearing opening price** published by the exchange post-auction.
   - **Data Availability Reality:** In live trading, between 9:00 AM and 9:08 AM, the pre-open call auction calculates an *indicative* equilibrium price; the actual executed open price is published only after order matching concludes at 9:08 AM. However, our dataset comes from end-of-day Bhavcopy daily records, not tick-by-tick timestamped 9:08 AM call auction order-book snapshots.
   - **Methodological Verdict:** Treating the final opening price as available before market open assumes information that was not genuinely timestamped in the historical daily dataset. Therefore, **Setup B is formally classified as NON-EXECUTABLE for live forward trading with the current daily Bhavcopy dataset**.
   - **Role in Stage 6:** Setup B is retained strictly as an **exploratory upper-bound benchmark** to measure the theoretical value of overnight price gap information, but **Setup A (Strict Pre-Open)** serves as the **sole primary realistic, executable backtest**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             SETUP FEASIBILITY & EXECUTION MATRIX                            │
├──────────────────────────┬─────────────────────────────┬────────────────────────────────────┤
│ Dimension                │ Setup A: Strict Pre-Open    │ Setup B: Opening Bell (Benchmark)  │
├──────────────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ Operational Status       │ PRIMARY EXECUTABLE STRATEGY │ EXPLORATORY BENCHMARK ONLY         │
│ Signal Timestamp         │ 9:00:00 AM IST              │ 9:14:00 AM IST                     │
│ Information Boundary     │ Lagged Market (t-1) + News  │ Final Bhavcopy adj_open_t          │
│ Live Feasibility         │ Fully Feasible              │ Non-Executable (No Tick Timestamps)│
│ Execution Time           │ 9:15:00 AM IST              │ 9:15:00 AM IST                     │
│ Fill Pricing             │ Unadjusted adj_open_t       │ Unadjusted adj_open_t              │
│ Friction Accounting      │ Deducted from traded volume │ Deducted from traded volume        │
│ Holding Period           │ Exactly 1 Trading Session   │ Exactly 1 Trading Session          │
│ Square-off Time          │ 3:20 PM IST (Assumed RMS)   │ 3:20 PM IST (Assumed RMS)          │
│ Overnight Inventory Risk │ Zero (100% Cash Overnight)  │ Zero (100% Cash Overnight)         │
└──────────────────────────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## 3. Signal Generation & Finite Parameter Grid

Each trading session $t$, the frozen model outputs continuous probabilities $\hat{p}_{i, t} \in [0, 1]$ representing the forecast probability of an upward intraday move ($R_{i, t} > 0$).

### 3.1 Strategy Archetypes & Signal Boundary Definitions

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   STRATEGY ARCHETYPES                                       │
├─────────────────────┬───────────────────────────────────────────────────────────────────────┤
│ Strategy 1:         │ • Long-only intraday positions on high-conviction bullish stocks.     │
│ Long-Only Absolute  │ • Signal: Long if p_hat >= theta_long; Cash if p_hat < theta_long.     │
│ Threshold           │ • Zero short exposure. Capital in cash earns illustrative risk-free.  │
├─────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Strategy 2:         │ • Long high conviction, Short low conviction, Cash in neutral band.   │
│ Long/Short Dynamic  │ • Signal: Long if p_hat >= theta_long; Short if p_hat <= theta_short; │
│ Threshold           │   Cash if theta_short < p_hat < theta_long.                           │
│                     │ • Intraday short-selling executed via NSE MIS mode.                   │
├─────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Strategy 3:         │ • Cross-sectional ranking: Long Top K, Short Bottom K.                │
│ Rank-Based          │ • Dollar-neutral gross allocation (50% long notional, 50% short).     │
│ Long/Short (Top-K)  │ • Strict 1.0x gross exposure cap. Note: Dollar neutrality does not    │
│                     │   imply zero market beta (beta asymmetry documented below).           │
└─────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

### 3.2 Finite Validation Search Grid & Boundary Resolution (22 Valid Pairs)

To prevent threshold overfitting and eliminate test-set snooping, thresholds are tuned exclusively on the **2024 Validation Split** ($N = 3,690$).

We specify a discrete grid of candidate thresholds:
$$\theta_{\text{long}} \in \{0.48, 0.50, 0.52, 0.54, 0.56\}$$
$$\theta_{\text{short}} \in \{0.42, 0.44, 0.46, 0.48, 0.50\}$$

#### Grid Sizing Analysis:
- The unrestricted Cartesian product contains $5 \times 5 = 25$ pairs.
- Imposing a weak inequality $\theta_{\text{short}} \le \theta_{\text{long}}$ eliminates only 1 pair where $\theta_{\text{short}} = 0.50$ and $\theta_{\text{long}} = 0.48$, leaving **24 pairs**.
- However, allowing $\theta_{\text{short}} = \theta_{\text{long}}$ (which occurs at $(0.48, 0.48)$ and $(0.50, 0.50)$) introduces **overlapping boundary signals** where an observation with $\hat{p} = \theta$ simultaneously satisfies both long and short entry criteria, completely collapsing the neutral cash zone.
- Therefore, we enforce the **strict inequality constraint**:
  $$\theta_{\text{short}} < \theta_{\text{long}}$$
  This eliminates both conflicting equality points, guarantees an unambiguous neutral cash band of strictly positive width, and produces exactly **22 valid parameter pairs**.

#### Exact Grid Enumeration (22 Valid Pairs):
- For $\theta_{\text{long}} = 0.48$: $\theta_{\text{short}} \in \{0.42, 0.44, 0.46\}$ (**3 pairs**)
- For $\theta_{\text{long}} = 0.50$: $\theta_{\text{short}} \in \{0.42, 0.44, 0.46, 0.48\}$ (**4 pairs**)
- For $\theta_{\text{long}} = 0.52$: $\theta_{\text{short}} \in \{0.42, 0.44, 0.46, 0.48, 0.50\}$ (**5 pairs**)
- For $\theta_{\text{long}} = 0.54$: $\theta_{\text{short}} \in \{0.42, 0.44, 0.46, 0.48, 0.50\}$ (**5 pairs**)
- For $\theta_{\text{long}} = 0.56$: $\theta_{\text{short}} \in \{0.42, 0.44, 0.46, 0.48, 0.50\}$ (**5 pairs**)
$$\text{Total Valid Configurations} = 3 + 4 + 5 + 5 + 5 = \mathbf{22 \text{ discrete parameter pairs}}$$

#### Explicit Signal Boundary Handling:
- **Long Signal ($S_{i, t} = +1$):** $\hat{p}_{i, t} \ge \theta_{\text{long}}$
- **Short Signal ($S_{i, t} = -1$):** $\hat{p}_{i, t} \le \theta_{\text{short}}$
- **Neutral / Cash ($S_{i, t} = 0$):** $\theta_{\text{short}} < \hat{p}_{i, t} < \theta_{\text{long}}$

Because $\theta_{\text{short}} < \theta_{\text{long}}$, the intervals are mutually exclusive and exhaustive over $[0, 1]$. The parameter pair $(\theta^*_{\text{long}}, \theta^*_{\text{short}})$ maximizing the **Validation Net Sharpe Ratio** is permanently frozen before evaluating the 2025–2026 Test set. Zero test-set metrics will be inspected during parameter selection.

---

## 4. Position Sizing, Capital Allocation & Execution Realities

### 4.1 Capital Allocation & Portfolio Base
- **Initial Portfolio Capital:** $C_0 = \text{₹}10,000,000$ ($1\text{ Crore INR}$, illustrative institutional scale).
- **Allocation Scheme:** If $K_{\text{long}, t}$ stocks trigger long signals and $K_{\text{short}, t}$ trigger short signals on day $t$:
  $$w_{i, t}^{\text{long}} = \frac{0.50}{K_{\text{long}, t}} \quad (\text{if } K_{\text{short}, t} > 0), \quad w_{j, t}^{\text{short}} = -\frac{0.50}{K_{\text{short}, t}}$$
  If only long signals trigger, $w_{i, t} = \frac{1.0}{K_{\text{long}, t}}$ subject to concentration caps.

### 4.2 Portfolio Risk & Concentration Limits
1. **Gross Leverage Cap:** Maximum 1.0x gross exposure at all times:
   $$\sum_{i \in \text{Long}} w_{i, t} + \sum_{j \in \text{Short}} |w_{j, t}| \le 1.0$$
2. **Single-Stock Cap:** Maximum $20\%$ allocation per constituent ($|w_{i, t}| \le 0.20$).
3. **Sector Concentration Cap:** Maximum $40\%$ allocation in any single sector to prevent over-concentration in Financial Services (SBIN, HDFCBANK, ICICIBANK).
4. **Volume Participation Limit & Dataset Limitation:**  
   We specify an execution constraint capping single-order size at $\le 1.0\%$ of constituent opening volume.  
   *Explicit Dataset Limitation Disclosure:* The current historical daily Bhavcopy dataset contains only day-level aggregate volume and cannot establish 5-minute opening volume or empirical order-book queue depth. This constraint cannot be empirically verified with the present daily data. Consequently, this backtest **does not empirically establish strategy capacity for a ₹1 Crore portfolio**. Capacity verification requires tick-level order book data.
5. **Cash Yield (Fixed Illustrative Assumption):** Unallocated capital earns a **fixed illustrative annualized yield of 6.50% p.a.** ($0.02579\%$ per trading day). This is explicitly documented as a fixed benchmark hurdle rate, not a dynamic historical interest rate.

### 4.3 Practical Short-Position Execution on NSE
- In Indian cash equities, intraday short selling is legally permitted via Margin Intraday Square-off (MIS).
- Short selling does **not** require Securities Lending & Borrowing (SLBM) fees because positions are not held overnight.
- **Assumed RMS Square-Off Policy:** We model square-off at 3:20 PM IST based on typical Indian broker Risk Management System (RMS) automated square-off policies (to ensure liquidation before the official 3:30 PM exchange close). This is an assumed broker operational policy, not a universal statutory exchange rule.
- All positions must be closed intraday; under Indian cash equity regulations, carryover of naked short positions is strictly prohibited.
- **Dollar Neutrality vs. Beta Neutrality:** In Strategy 3 (Rank-Based Long/Short), allocating 50% notional to Top-K long and 50% notional to Bottom-K short achieves **dollar neutrality, not beta neutrality**:
  $$\beta_{\text{portfolio}, t} = \sum_{i \in \text{Long}} w_{i, t} \beta_i - \sum_{j \in \text{Short}} |w_{j, t}| \beta_j \ne 0$$
  The portfolio remains exposed to net systematic beta depending on sector representation.

---

## 5. Friction Accounting & Intraday Turnover

### 5.1 Consistent Cost Accounting (Avoiding Double-Counting Slippage)

To avoid double-counting slippage, we establish a single consistent accounting method:
1. **Gross Returns:** Calculated directly using unadjusted official daily Bhavcopy prices:
   $$R_{i, t}^{\text{gross}} = \begin{cases} \frac{\text{adj\_close}_{i, t} - \text{adj\_open}_{i, t}}{\text{adj\_open}_{i, t}}, & \text{for Long positions } (w_{i, t} > 0) \\ \frac{\text{adj\_open}_{i, t} - \text{adj\_close}_{i, t}}{\text{adj\_open}_{i, t}}, & \text{for Short positions } (w_{i, t} < 0) \end{cases}$$
2. **Fill Prices:** Modeled strictly at unadjusted $\text{adj\_open}_{i, t}$ (entry) and $\text{adj\_close}_{i, t}$ (exit). **Fill prices are NOT artificially adjusted for slippage.**
3. **Total Transaction Friction:** Deducted as an explicit, separate charge from actual traded portfolio turnover, combining both statutory/brokerage charges and bid-ask slippage. Slippage is not deducted twice.

### 5.2 Date-Appropriate Statutory Fee Schedule & Historical Validation Costs

The 2024 Validation split spans dates before and after the **October 1, 2024 NSE transaction charge revision**. The simulation engine dynamically applies date-appropriate statutory rates:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          STATUTORY & EXECUTION FRICTION SCHEDULE                            │
├───────────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Fee / Tax Component           │ Pre-Oct 1, 2024 Rate        │ Post-Oct 1, 2024 Rate (Test)  │
├───────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 1. Brokerage                  │ Buy: 2.00 bps / Sell: 2.00  │ Buy: 2.00 bps / Sell: 2.00    │
│ 2. Securities Transaction Tax │ Buy: 0.00 / Sell: 2.50 bps  │ Buy: 0.00 / Sell: 2.50 bps    │
│ 3. NSE Exchange Turnover Fee  │ Buy: 0.325 bps / Sell: 0.325│ Buy: 0.297 bps / Sell: 0.297  │
│ 4. SEBI Turnover Fee          │ Buy: 0.010 bps / Sell: 0.010│ Buy: 0.010 bps / Sell: 0.010  │
│ 5. NSE IPFT Charge            │ Buy: 0.010 bps / Sell: 0.010│ Buy: 0.010 bps / Sell: 0.010  │
│ 6. Stamp Duty (Indian Stamp)  │ Buy: 0.300 bps / Sell: 0.000│ Buy: 0.300 bps / Sell: 0.000  │
│ 7. GST (18% on Taxable)       │ Round-Trip: 0.841 bps       │ Round-Trip: 0.831 bps         │
├───────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ STATUTORY + BROKERAGE TOTAL   │ 8.331 bps (~8.33 bps)       │ 8.265 bps (~8.27 bps)         │
├───────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 8. Bid-Ask Slippage           │ 3.75 bps Entry / 3.75 Exit  │ 3.75 bps Entry / 3.75 Exit    │
│    (Separately Stated)        │ = 7.50 bps Round-Trip       │ = 7.50 bps Round-Trip         │
├───────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ TOTAL ROUND-TRIP FRICTION     │ 15.831 bps (~15.83 bps)     │ 15.765 bps (~15.77 bps)       │
└───────────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

#### Detailed Itemized Breakdown & Historical Disclosure:
- **Pre-Oct 1, 2024 (Validation Split Jan 1 – Sep 30, 2024):**
  - NSE cash market turnover charge was ₹3.25 per lakh (0.325 bps per side, 0.650 bps round-trip).
  - Taxable value for GST = Brokerage (4.00 bps) + NSE Fee (0.650 bps) + SEBI Fee (0.020 bps) = 4.670 bps. GST @ 18% = 0.8406 bps.
  - Statutory + Brokerage = $4.00 + 2.50 + 0.650 + 0.020 + 0.020 + 0.300 + 0.8406 = \mathbf{8.331\text{ bps}}$.
  - With 7.50 bps bid-ask slippage, total round-trip friction is $\mathbf{15.831\text{ bps} \approx 15.83\text{ bps} \text{ (0.1583\%)}}$.
- **Post-Oct 1, 2024 (Validation Split Oct 1 – Dec 31, 2024 & Out-of-Sample Test Split 2025–2026):**
  - NSE cash market turnover charge revised to flat ₹2.97 per lakh (0.297 bps per side, 0.594 bps round-trip).
  - Taxable value for GST = Brokerage (4.00 bps) + NSE Fee (0.594 bps) + SEBI Fee (0.020 bps) = 4.614 bps. GST @ 18% = 0.8305 bps.
  - Statutory + Brokerage = $4.00 + 2.50 + 0.594 + 0.020 + 0.020 + 0.300 + 0.8305 = \mathbf{8.265\text{ bps} \approx 8.27\text{ bps}}$.
  - With 7.50 bps bid-ask slippage, total round-trip friction is $\mathbf{15.765\text{ bps} \approx 15.77\text{ bps} \text{ (0.1577\%)}}$.
- **Implementation & Sensitivity Disclosure:**  
  The simulation engine dynamically applies the historical split:
  $$\text{Friction}_{\text{round-trip}, t} = \begin{cases} 15.831\text{ bps (0.15831\%)}, & \text{for } t \le 2024\text{-}09\text{-}30 \\ 15.765\text{ bps (0.15765\%)}, & \text{for } t \ge 2024\text{-}10\text{-}01 \end{cases}$$
  The delta between the two regimes is approximately $0.066\text{ bps}$ ($0.00066\%$). Applying historically exact charges eliminates distortion across the 2024 validation window.

### 5.3 Correct Intraday Turnover Formulation

Because the portfolio is **100% in cash overnight**, every active position taken on day $t$ involves an **opening entry at 9:15 AM** AND a **closing square-off at 3:20 PM**.

We explicitly calculate entry and exit turnover for each session:
- **Entry Turnover Leg (9:15 AM):** $\text{Turnover}_{\text{entry}, t} = \sum_{i=1}^{15} |w_{i, t}|$
- **Exit Turnover Leg (3:20 PM):** $\text{Turnover}_{\text{exit}, t} = \sum_{i=1}^{15} |w_{i, t}|$
- **Total Daily Traded Turnover:**
  $$\text{Turnover}_{\text{traded}, t} = \text{Turnover}_{\text{entry}, t} + \text{Turnover}_{\text{exit}, t} = 2 \times \sum_{i=1}^{15} |w_{i, t}|$$

**Key Operational Realization:**  
A fully opened and closed 10% position ($|w_{i, t}| = 0.10$) generates **20% of portfolio-equity turnover** ($10\%$ entry $+ 10\%$ exit).  
We strictly avoid using a simple interday weight difference $|w_{i, t} - w_{i, t-1}|$, which would falsely assume positions are held across sessions and miss the overnight cash reset.

### 5.4 Net Daily Portfolio Return Formulation & Accounting Clarification

Transaction friction must be deducted without double-counting. We verify the accounting with a concrete example:

**Concrete Verification Example:**
Consider a 10% position opened at 9:15 AM and closed at 3:20 PM ($|w_{i, t}| = 0.10$).
- Traded turnover is $10\% \text{ (entry)} + 10\% \text{ (exit)} = 20\%$ of portfolio equity.
- The applicable round-trip friction is $15.765\text{ bps}$ ($0.0015765$).
- **Method 1 (Recommended):** Round-trip cost rate $\times$ one-way position notional (sum of absolute position weights):
  $$\text{Friction Cost} = 15.765\text{ bps} \times 10\% = 0.015765\% \text{ of portfolio equity}$$
- **Method 2:** One-way cost rate ($f_{\text{one-way}} = \frac{1}{2} \times 15.765\text{ bps} = 7.8825\text{ bps}$) $\times$ actual traded turnover ($20\%$):
  $$\text{Friction Cost} = 7.8825\text{ bps} \times 20\% = 0.015765\% \text{ of portfolio equity}$$

Both formulations yield the exact same economic friction. The implementation **must never multiply the full round-trip rate by total traded turnover** (which would incorrectly charge $15.765\text{ bps} \times 20\% = 0.03153\%$, doubling the true friction).

The net portfolio return on day $t$ is therefore:
$$R_{p, t}^{\text{net}} = \sum_{i=1}^{15} w_{i, t} R_{i, t}^{\text{gross}} - \left( \sum_{i=1}^{15} |w_{i, t}| \times \text{Friction}_{\text{round-trip}, t} \right) + \left( 1 - \sum_{i=1}^{15} |w_{i, t}| \right) \frac{R_f}{252}$$

Where:
- $w_{i, t} > 0$ for long positions, $w_{i, t} < 0$ for short positions.
- Gross stock return $R_{i, t}^{\text{gross}}$ is computed from unadjusted Bhavcopy open and close prices.
- $\text{Friction}_{\text{round-trip}, t}$ is $15.831\text{ bps}$ for $t \le 2024\text{-}09\text{-}30$ and $15.765\text{ bps}$ for $t \ge 2024\text{-}10\text{-}01$.
- Traded turnover is tracked separately as $\text{Turnover}_{\text{traded}, t} = 2 \times \sum_{i=1}^{15} |w_{i, t}|$.
- Uninvested cash earns the fixed illustrative annualized yield of $R_f = 6.50\%$.

---

## 6. Strategy Configurations & Benchmark Hierarchy

Stage 6 evaluates **8 distinct strategy configurations**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 STAGE 6 STRATEGY MATRIX                                     │
├─────────────────────┬──────────────┬───────────────┬────────────────────────────────────────┤
│ Strategy Identifier │ Operational  │ Model & Type  │ Information Set & Features             │
├─────────────────────┼──────────────┼───────────────┼────────────────────────────────────────┤
│ 1. PreOpen_Mkt_LR   │ EXECUTABLE   │ Logistic Reg  │ Setup A: Pre-Open Technicals Only      │
│ 2. PreOpen_Sent_LR  │ EXECUTABLE   │ Logistic Reg  │ Setup A: Pre-Open Market + FinBERT     │
│ 3. PreOpen_Sent_XGB │ EXECUTABLE   │ XGBoost       │ Setup A: Pre-Open Market + FinBERT     │
│ 4. PreOpen_LM_XGB   │ EXECUTABLE   │ XGBoost       │ Setup A: Pre-Open Market + LM Sent     │
│ 5. SetupB_Mkt_LR    │ BENCHMARK    │ Logistic Reg  │ Setup B: Market + open_gap_pct         │
│ 6. SetupB_FinBERT_LR│ BENCHMARK    │ Logistic Reg  │ Setup B: Market + FinBERT + Gap        │
│ 7. SetupB_LM_XGB    │ BENCHMARK    │ XGBoost       │ Setup B: Market + LM Sent + Gap        │
│ 8. SetupB_FinBERT_RF│ BENCHMARK    │ Random Forest │ Setup B: Market + FinBERT + Gap        │
└─────────────────────┴──────────────┴───────────────┴────────────────────────────────────────┘
```

### Benchmark Interpretation Discipline:

1. **Primary Incremental-News Comparison (Matching Market-Only ML Baselines):**  
   The primary test of economic value is comparing `PreOpen_Sent_LR` against `PreOpen_Mkt_LR` under identical execution rules, fee structures, and turnover models.
2. **Buy-and-Hold Nifty 50 Constituent Portfolio (Exposure-Mismatched Reference):**  
   Equal-weighted buy-and-hold across the 15 stocks.  
   *Critical Interpretation:* Buy-and-hold holds equities 24 hours a day, bearing 100% overnight gap risk and compounding broad equity beta. Intraday strategies hold equities for only ~6 hours and sit in cash overnight. Buy-and-hold is **not exposure-matched** and serves solely as an unmanaged equity reference.
3. **Intraday 1-Day Momentum Baseline:**  
   Intraday strategy trading purely on previous-session direction ($R_{t-1} > 0$).
4. **Risk-Free Cash Hurdle:**  
   Illustrative fixed benchmark rate of $6.5\%$ p.a.

---

## 7. Quantitative Performance Metrics

All strategies will be evaluated across twelve financial performance dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            QUANTITATIVE PERFORMANCE METRICS                                 │
├──────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│ Metric Category          │ Indicator                     │ Mathematical Formula             │
├──────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
│ 1. Cumulative Return     │ Cumulative Net Return (%)     │ (V_T - V_0) / V_0 * 100%         │
│ 2. Annualized Return     │ CAGR (%)                      │ (V_T / V_0)^(252 / N_days) - 1   │
│ 3. Volatility            │ Annualized Volatility (%)     │ std(R_daily) * sqrt(252)         │
│ 4. Risk-Adjusted Return  │ Sharpe Ratio                  │ (CAGR - R_f) / Volatility_ann    │
│ 5. Downside Risk         │ Sortino Ratio                 │ (CAGR - R_f) / Downside_dev_ann  │
│ 6. Drawdown Profile      │ Maximum Drawdown (MDD %)      │ max_t (Peak_t - V_t) / Peak_t    │
│ 7. Drawdown Recovery     │ Calmar Ratio                  │ CAGR / |MDD|                     │
│ 8. Trade Accuracy        │ Win Rate (%)                  │ N_profitable_days / N_active_days│
│ 9. Profit Quality        │ Profit Factor                 │ Sum(Gains) / |Sum(Losses)|       │
│ 10. Payoff Asymmetry     │ Gain-to-Pain Ratio            │ Mean(Gains) / |Mean(Losses)|     │
│ 11. Trading Intensity    │ Annual Turnover (%)           │ Mean(Turnover_traded) * 252      │
│ 12. Market Exposure      │ Time-in-Market (%)            │ N_invested_days / N_total_days   │
└──────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
```

---

## 8. Sub-Group & Diagnostic Performance Breakdowns

1. **Per-Company Economic Attribution:** Cumulative net P&L, Sharpe ratio, and win rate reported for all 15 stocks individually.
2. **Market Regime Conditioning:** Net P&L, Sharpe, and drawdown stratified across Bullish Trend ($> +1\%$ SMA), Bearish Trend ($< -1\%$ SMA), Neutral Trend ($\pm 1\%$ SMA), and Volatility Regimes (`LOW_VOL` vs. `HIGH_VOL`).
3. **Stress Period Dissection:** Daily net returns and maximum drawdown during the top 5 market dislocation days in 2025–2026.
4. **Documented Survivorship Bias:** Explicit recognition that the 15 approved stocks remained established, continuous Nifty 50 constituents throughout 2020–2026.

---

## 9. Implementation Architecture (Stage 6 Roadmap)

Upon user approval of this final design, Stage 6 will execute the following modular pipeline:
1. **`src/backtest_engine.py`:** Vectorized, zero-lookahead backtesting simulation engine implementing the exact itemized statutory fee schedule (15.83 bps pre-Oct 2024, 15.77 bps post-Oct 2024), 22-pair validation parameter grid, correct intraday turnover doubling ($2 \times \sum |w_i|$), and portfolio constructors.
2. **`src/run_stage6_backtesting.py`:** Execution script evaluating all 8 strategy configurations across Train, Validation, and Test splits.
3. **Persisted Artifacts:**
   - Trade logs: `data/predictions/backtest_trade_log.parquet`.
   - Equity curves: `data/predictions/backtest_equity_curves.parquet`.
   - Structured metrics: `models/sentiment_experiments/stage6_backtesting_results.json`.
   - Figures: `reports/figures/stage6_equity_curves.png`, `reports/figures/stage6_drawdowns.png`, `reports/figures/stage6_cost_sensitivity.png`.
4. **Final Deliverable:** `reports/stage6_backtesting_report.md`.

---

```
================================================================================
STAGE 6 FINAL REVISED EXPERIMENTAL DESIGN COMPLETE.
EXECUTION PAUSED: NO BACKTESTS RUN. NO CODE EXECUTED.
AWAITING USER REVIEW AND FORMAL AUTHORIZATION TO PROCEED.
================================================================================
```
