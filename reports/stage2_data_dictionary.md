# BEYOND SENTIMENT: STAGE 2 DATA DICTIONARY

**Document Type:** Formal Data Architecture & Column Specification  
**Dataset Name:** `final_integrated_dataset.parquet`  
**File Location:** `data/final/final_integrated_dataset.parquet`  
**Version:** 2.0 (Stage 2 Integration)  
**Entities Covered:** 15 Approved Nifty 50 Large-Cap Companies  
**Observation Window:** January 1, 2020 – March 31, 2026  

---

### 1. Primary Keys & Entity Identifiers

| Column Name | Physical Data Type | Logical Type | Description & Domain |
| :--- | :--- | :--- | :--- |
| `date` | `date32` | Date | Official NSE Trading Session Date ($t$). Format: `YYYY-MM-DD`. |
| `symbol` | `string` | Categorical | Official NSE equity trading ticker symbol (e.g. `RELIANCE`, `TCS`). |
| `isin` | `string` | Identifier | Permanent 12-character International Securities Identification Number. |
| `sector` | `string` | Categorical | Primary economic sector classification of the issuer. |

---

### 2. Raw Market Observations (Unscaled Exchange Bhavcopy)

All fields in this section represent nominal, unadjusted values as published in the official NSE Bhavcopy.

| Column Name | Physical Data Type | Logical Type | Description & Domain |
| :--- | :--- | :--- | :--- |
| `raw_open` | `float64` | Currency (INR) | Official session opening price. |
| `raw_high` | `float64` | Currency (INR) | Highest price executed during trading session. |
| `raw_low` | `float64` | Currency (INR) | Lowest price executed during trading session. |
| `raw_close` | `float64` | Currency (INR) | Official volume-weighted closing price (last 30 minutes). |
| `raw_prev_close` | `float64` | Currency (INR) | Previous session's closing price as recorded by the exchange. |
| `raw_volume` | `int64` | Count | Total quantity of shares traded during session. |
| `raw_turnover` | `float64` | Currency (INR) | Gross trading value executed across all trades in session. |
| `raw_trades` | `int64` | Count | Total number of discrete trades executed in session. |

---

### 3. Corporate-Action-Adjusted Price Series (Total Return Framework)

Prices continuously adjusted backward for all subsequent stock splits, bonus issues, and cash dividends. Scaled consistently using the cumulative factor.

| Column Name | Physical Data Type | Logical Type | Description & Domain |
| :--- | :--- | :--- | :--- |
| `adj_factor_cumulative` | `float64` | Multiplier | Cumulative backward multiplier ($\prod_{\tau > t} \text{Factor}_\tau$). Value is $1.0$ when no subsequent actions exist. |
| `adj_open` | `float64` | Currency (INR) | $\text{raw\_open} \times \text{adj\_factor\_cumulative}$. |
| `adj_high` | `float64` | Currency (INR) | $\text{raw\_high} \times \text{adj\_factor\_cumulative}$. |
| `adj_low` | `float64` | Currency (INR) | $\text{raw\_low} \times \text{adj\_factor\_cumulative}$. |
| `adj_close` | `float64` | Currency (INR) | $\text{raw\_close} \times \text{adj\_factor\_cumulative}$. Total return closing price. |
| `adj_prev_close` | `float64` | Currency (INR) | $\text{raw\_prev\_close} \times \text{adj\_factor\_cumulative}$. |

---

### 4. Technical Market Indicators (Computed Exclusively on Adjusted Prices)

| Column Name | Physical Data Type | Logical Type | Mathematical Definition & Operational Notes |
| :--- | :--- | :--- | :--- |
| `ret_1d` | `float64` | Ratio | Daily percentage return: $(\text{adj\_close}_t - \text{adj\_close}_{t-1}) / \text{adj\_close}_{t-1}$. |
| `rsi_14` | `float64` | Index $[0, 100]$ | 14-session Relative Strength Index computed on `adj_close`. Standard Wilder smoothing. |
| `macd` | `float64` | Momentum | Moving Average Convergence Divergence line: $\text{EMA}_{12}(\text{adj\_close}) - \text{EMA}_{26}(\text{adj\_close})$. |
| `macd_signal` | `float64` | Momentum | 9-session exponential moving average of the `macd` line. |
| `macd_diff` | `float64` | Momentum | MACD Histogram: $\text{macd} - \text{macd\_signal}$. |
| `rolling_vol_5d` | `float64` | Volatility | 5-session rolling sample standard deviation of `ret_1d`. |
| `rolling_vol_21d` | `float64` | Volatility | 21-session rolling sample standard deviation of `ret_1d` (Market Regime proxy). |
| `adj_close_sma_ratio_20` | `float64` | Trend Ratio | Percentage deviation from 20-day Simple Moving Average: $(\text{adj\_close} / \text{SMA}_{20}) - 1$. |
| `intraday_range_pct` | `float64` | Ratio | Normalized intraday range: $(\text{raw\_high} - \text{raw\_low}) / \text{raw\_close}$. Identical on raw or adjusted. |
| `market_regime` | `string` | Categorical | Regime classification: `'HIGH_VOL'` if `rolling_vol_21d` $\ge \text{Median}$, else `'LOW_VOL'`. |

---

### 5. Multi-Tier Sentiment Features (Forward-Aligned $T \to T+1$)

All news features mapped to trading session $t$ were published on calendar dates strictly prior to $t$.

#### Group A: Direct Company Sentiment
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `direct_news_pos` | `float64` | $[0.0, 1.0]$ | FinBERT positive sentiment probability for direct company news. $0.0$ if no direct news. |
| `direct_news_neu` | `float64` | $[0.0, 1.0]$ | FinBERT neutral sentiment probability for direct company news. $0.0$ if no direct news. |
| `direct_news_neg` | `float64` | $[0.0, 1.0]$ | FinBERT negative sentiment probability for direct company news. $0.0$ if no direct news. |
| `direct_finbert_polarity`| `float64` | $[-1.0, 1.0]$ | Net direct polarity: $\text{direct\_news\_pos} - \text{direct\_news\_neg}$. Explicitly $0.0$ if no direct news. |

#### Group B: Sector-Level Sentiment
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `sectoral_news_pos` | `float64` | $[0.0, 1.0]$ | FinBERT positive probability for sector-level industry news. |
| `sectoral_news_neu` | `float64` | $[0.0, 1.0]$ | FinBERT neutral probability for sector-level industry news. |
| `sectoral_news_neg` | `float64` | $[0.0, 1.0]$ | FinBERT negative probability for sector-level industry news. |
| `sectoral_finbert_polarity`| `float64` | $[-1.0, 1.0]$ | Net sector polarity: $\text{sectoral\_news\_pos} - \text{sectoral\_news\_neg}$. |

#### Group C: Broad Market Sentiment
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `global_news_pos` | `float64` | $[0.0, 1.0]$ | FinBERT positive probability for macro/international news. |
| `global_news_neu` | `float64` | $[0.0, 1.0]$ | FinBERT neutral probability for macro/international news. |
| `global_news_neg` | `float64` | $[0.0, 1.0]$ | FinBERT negative probability for macro/international news. |
| `global_finbert_polarity` | `float64` | $[-1.0, 1.0]$ | Net macro polarity: $\text{global\_news\_pos} - \text{global\_news\_neg}$. |

#### Group D: News Volume Counts
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `direct_news_count` | `int64` | $\ge 0$ | Total number of direct articles aggregated into session $t$. |
| `sectoral_news_count`| `int64` | $\ge 0$ | Total number of sector articles aggregated into session $t$. |
| `global_news_count` | `int64` | $\ge 0$ | Total number of macro articles aggregated into session $t$. |
| `extracted_direct_count`| `int64` | $\ge 0$ | Direct articles verified by our entity alias extractor. |

#### Group E: No-News Indicator Flags
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `has_direct_news` | `int32` | $\{0, 1\}$ | Binary indicator: $1$ if `direct_news_count` $> 0$, else $0$. |
| `has_sector_news` | `int32` | $\{0, 1\}$ | Binary indicator: $1$ if `sectoral_news_count` $> 0$, else $0$. |

#### Group F: Temporal Aggregation Windows
| Column Name | Physical Data Type | Domain | Description |
| :--- | :--- | :--- | :--- |
| `direct_polarity_roll_3d` | `float64` | $[-1.0, 1.0]$ | 3-session rolling average of `direct_finbert_polarity`. |
| `direct_polarity_roll_7d` | `float64` | $[-1.0, 1.0]$ | 7-session rolling average of `direct_finbert_polarity`. |
| `sectoral_polarity_roll_3d`| `float64` | $[-1.0, 1.0]$ | 3-session rolling average of `sectoral_finbert_polarity`. |
| `sectoral_polarity_roll_7d`| `float64` | $[-1.0, 1.0]$ | 7-session rolling average of `sectoral_finbert_polarity`. |

---

### 6. Forward-Looking Prediction Targets (Session $t+1$)

| Column Name | Physical Data Type | Logical Domain | Mathematical Definition & Timing Protocol |
| :--- | :--- | :--- | :--- |
| `target_ret_o2c` | `float64` | Return | Intraday return for session $t+1$: $(\text{adj\_close}_{t+1} - \text{adj\_open}_{t+1}) / \text{adj\_open}_{t+1}$. |
| `target_o2c_direction` | `int32` | Binary $\{0, 1\}$ | **Primary Target:** $1$ if $\text{target\_ret\_o2c} > 0$ (UP), else $0$ (DOWN). |
| `target_ret_c2c` | `float64` | Return | Interday close-to-close return: $(\text{adj\_close}_{t+1} - \text{adj\_close}_t) / \text{adj\_close}_t$. |
| `target_c2c_direction` | `int32` | Binary $\{0, 1\}$ | **Secondary Informational Target:** $1$ if $\text{target\_ret\_c2c} > 0$ (UP), else $0$ (DOWN). |
