# BEYOND SENTIMENT: STAGE 2 DATA QUALITY REPORT

**Status:** ALL CHECKS PASSED
**Generated:** 2026-11-03 01:37:02

## 1. Summary of Integrity Verifications

| Integrity Dimension | Validation Rule | Actual Result | Verification Status |
| :--- | :--- | :--- | :---: |
| **Entity Consistency** | Exactly 15 approved companies present | 15 / 15 verified | **PASSED** |
| **Date Continuity** | Continuous trading 2020-01-01 to 2026-03-31 | 1,541 sessions / company | **PASSED** |
| **Duplicate Records** | 0 duplicate (date, symbol) pairs | 0 duplicates found | **PASSED** |
| **Missing Values** | 0 nulls in OHLCV, targets, and flags | 0 unexpected nulls | **PASSED** |
| **OHLC Consistency** | High >= Low, High >= Open/Close, Low <= Open/Close | 100% compliant | **PASSED** |
| **Price Positivity** | All prices > 0, Volume >= 0 | 100% compliant | **PASSED** |
| **Corporate Actions** | Fully scaled without mixing adjusted/raw | 150 actions integrated | **PASSED** |
| **Leakage Prevention** | Features at t strictly depend on <= t; Targets = t+1 | Exact match verified | **PASSED** |
| **Non-Substitution** | Zero direct news days strictly neutral (0.0) | Verified across all rows | **PASSED** |

## 2. Target Distribution Breakdown

| Prediction Target | Definition | UP Class (%) | DOWN Class (%) | Total Count |
| :--- | :--- | :---: | :---: | :---: |
| **Primary: Open-to-Close** | `Adj Close(t+1) > Adj Open(t+1)` | 47.02% | 52.98% | 23,115 |
| **Secondary: Close-to-Close** | `Adj Close(t+1) > Adj Close(t)` | 51.16% | 48.84% | 23,115 |

Both targets demonstrate a healthy ~50/50 balance without severe class imbalance.
