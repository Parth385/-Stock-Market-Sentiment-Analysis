"""
Stage 6 Backtesting Pipeline Execution Script.
Author: Antigravity AI / DeepMind Team
Project: Beyond Sentiment

Executes Stage 6 Trading Strategy & Backtesting Simulation:
1. Loads frozen Stage 4 predictions and market regime metadata.
2. Selects optimal thresholds exclusively on the 2024 Validation split across 22 valid parameter pairs.
3. Simulates 8 strategy configurations + reference baselines across Train, Validation, and Test splits.
4. Computes all 12 quantitative financial metrics.
5. Performs per-company attribution, market regime breakdowns, stress period analysis, and cost sensitivity analysis.
6. Generates publication-ready figures.
7. Persists trade logs, equity curves, and structured JSON results.
"""

import json
import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest_engine import (
    simulate_daily_portfolio,
    simulate_buy_and_hold_baseline,
    simulate_intraday_momentum_baseline,
    calculate_performance_metrics,
    optimize_thresholds_on_validation,
    get_valid_threshold_pairs,
    get_roundtrip_friction,
    DAILY_RF,
)

# Set styling for plots
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    print("=" * 80)
    print("STAGE 6: TRADING STRATEGY & BACKTESTING EXECUTION")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/6] Loading predictions and feature store...")
    df_pred = pd.read_parquet("data/predictions/predictions_sentiment_experiments.parquet")
    df_feat = pd.read_parquet("data/final/final_sentiment_enhanced_dataset_corrected.parquet")
    
    # Merge regime indicators
    df = pd.merge(
        df_pred,
        df_feat[["date", "symbol", "market_regime_lag1", "adj_close_sma_ratio_20_lag1", "ret_1d_lag1"]],
        on=["date", "symbol"],
        how="inner",
    )
    df["date"] = pd.to_datetime(df["date"])
    print(f"Dataset successfully merged. Total observations: {len(df):,}")

    df_train = df[df["split"] == "TRAIN"].copy()
    df_val = df[df["split"] == "VAL"].copy()
    df_test = df[df["split"] == "TEST"].copy()
    print(f"Splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

    # 2. Define Strategy Configurations
    strategies_spec = [
        # Setup A (Primary Executable)
        {
            "id": "PreOpen_Mkt_LR",
            "name": "Setup A: Market-Only (Logistic Reg)",
            "setup": "Setup A",
            "operational": "EXECUTABLE",
            "prob_col": "Exp_A1_MktOnly_logreg_prob",
            "model": "Logistic Regression",
            "features": "Market-Only Pre-Open Technicals",
        },
        {
            "id": "PreOpen_Sent_LR",
            "name": "Setup A: FinBERT Sentiment (Logistic Reg)",
            "setup": "Setup A",
            "operational": "EXECUTABLE",
            "prob_col": "Exp_A4_MktFinBERT_logreg_prob",
            "model": "Logistic Regression",
            "features": "Market + FinBERT Sentiment",
        },
        {
            "id": "PreOpen_Sent_XGB",
            "name": "Setup A: FinBERT Sentiment (XGBoost)",
            "setup": "Setup A",
            "operational": "EXECUTABLE",
            "prob_col": "Exp_A4_MktFinBERT_xgboost_prob",
            "model": "XGBoost",
            "features": "Market + FinBERT Sentiment",
        },
        {
            "id": "PreOpen_LM_XGB",
            "name": "Setup A: LM Sentiment (XGBoost)",
            "setup": "Setup A",
            "operational": "EXECUTABLE",
            "prob_col": "Exp_A3_MktLM_xgboost_prob",
            "model": "XGBoost",
            "features": "Market + LM Sentiment",
        },
        # Setup B (Exploratory Benchmarks)
        {
            "id": "SetupB_Mkt_LR",
            "name": "Setup B: Market-Only (Logistic Reg)",
            "setup": "Setup B",
            "operational": "BENCHMARK_ONLY",
            "prob_col": "Exp_B1_MktOnly_logreg_prob",
            "model": "Logistic Regression",
            "features": "Market + Open Gap %",
        },
        {
            "id": "SetupB_FinBERT_LR",
            "name": "Setup B: FinBERT Sentiment (Logistic Reg)",
            "setup": "Setup B",
            "operational": "BENCHMARK_ONLY",
            "prob_col": "Exp_B4_MktFinBERT_logreg_prob",
            "model": "Logistic Regression",
            "features": "Market + FinBERT + Open Gap %",
        },
        {
            "id": "SetupB_LM_XGB",
            "name": "Setup B: LM Sentiment (XGBoost)",
            "setup": "Setup B",
            "operational": "BENCHMARK_ONLY",
            "prob_col": "Exp_B3_MktLM_xgboost_prob",
            "model": "XGBoost",
            "features": "Market + LM + Open Gap %",
        },
        {
            "id": "SetupB_FinBERT_RF",
            "name": "Setup B: FinBERT Sentiment (Random Forest)",
            "setup": "Setup B",
            "operational": "BENCHMARK_ONLY",
            "prob_col": "Exp_B4_MktFinBERT_random_forest_prob",
            "model": "Random Forest",
            "features": "Market + FinBERT + Open Gap %",
        },
    ]

    # 3. Step 2: Validation Threshold Optimization (Strict Zero-Lookahead)
    print("\n[Step 2/6] Optimizing Dynamic Thresholds on 2024 Validation Split across 22 valid pairs...")
    strategy_results = {}
    optimal_params = {}

    for strat in strategies_spec:
        sid = strat["id"]
        prob_col = strat["prob_col"]
        print(f"  Optimizing {sid} ({prob_col})...")
        best_tl, best_ts, best_val_sh, grid_info = optimize_thresholds_on_validation(
            df_val, prob_col=prob_col, strategy_type="dynamic_threshold"
        )
        optimal_params[sid] = {
            "theta_long": best_tl,
            "theta_short": best_ts,
            "val_sharpe": best_val_sh,
        }
        print(f"    -> Selected: theta_long={best_tl}, theta_short={best_ts} | Val Net Sharpe: {best_val_sh:.4f}")

    # 4. Step 3: Run Full Simulations Across Splits
    print("\n[Step 3/6] Running simulations across Train, Validation, and Out-of-Sample Test splits...")
    all_daily_equity = {}
    all_trade_logs = []

    for strat in strategies_spec:
        sid = strat["id"]
        prob_col = strat["prob_col"]
        tl = optimal_params[sid]["theta_long"]
        ts = optimal_params[sid]["theta_short"]

        # Run for each split
        split_metrics = {}
        daily_dfs_by_split = {}

        for sname, sdata in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
            # Dynamic Long/Short (Primary)
            d_df, t_df = simulate_daily_portfolio(
                sdata, prob_col=prob_col, strategy_type="dynamic_threshold", theta_long=tl, theta_short=ts
            )
            d_df["strategy_id"] = sid
            d_df["split"] = sname
            daily_dfs_by_split[sname] = d_df

            t_df["strategy_id"] = sid
            t_df["split"] = sname
            all_trade_logs.append(t_df)

            metrics = calculate_performance_metrics(d_df["net_return"], d_df["traded_turnover"])
            split_metrics[sname] = metrics

        # Also run Long-Only Archetype on Test
        d_df_longonly, _ = simulate_daily_portfolio(
            df_test, prob_col=prob_col, strategy_type="long_only", theta_long=tl, theta_short=ts
        )
        longonly_metrics = calculate_performance_metrics(d_df_longonly["net_return"], d_df_longonly["traded_turnover"])

        # Also run Rank-Based (Top-3 / Bottom-3) Archetype on Test
        d_df_rank, _ = simulate_daily_portfolio(
            df_test, prob_col=prob_col, strategy_type="rank_based", top_k=3
        )
        rank_metrics = calculate_performance_metrics(d_df_rank["net_return"], d_df_rank["traded_turnover"])

        # Combine daily equity across all splits for charting
        combined_daily = pd.concat([daily_dfs_by_split["TRAIN"], daily_dfs_by_split["VAL"], daily_dfs_by_split["TEST"]], ignore_index=True)
        combined_daily["cum_equity"] = (1.0 + combined_daily["net_return"]).cumprod()
        all_daily_equity[sid] = combined_daily

        strategy_results[sid] = {
            "metadata": strat,
            "optimal_params": optimal_params[sid],
            "train_metrics": split_metrics["TRAIN"],
            "val_metrics": split_metrics["VAL"],
            "test_metrics": split_metrics["TEST"],
            "test_archetypes": {
                "dynamic_long_short": split_metrics["TEST"],
                "long_only": longonly_metrics,
                "rank_based_top3": rank_metrics,
            },
        }

    # Reference Baselines on Test Set
    print("  Simulating reference baselines (Buy & Hold, Intraday Momentum, Risk-Free)...")
    bh_test = simulate_buy_and_hold_baseline(df_test)
    bh_metrics = calculate_performance_metrics(bh_test["net_return"], bh_test["traded_turnover"])

    mom_test = simulate_intraday_momentum_baseline(df, "TEST")
    mom_metrics = calculate_performance_metrics(mom_test["net_return"], mom_test["traded_turnover"])

    # Risk-free daily return series
    n_test_days = len(df_test["date"].unique())
    rf_test_series = pd.Series([DAILY_RF] * n_test_days)
    rf_turnover_series = pd.Series([0.0] * n_test_days)
    rf_metrics = calculate_performance_metrics(rf_test_series, rf_turnover_series)

    reference_baselines = {
        "buy_and_hold_15": bh_metrics,
        "intraday_momentum": mom_metrics,
        "risk_free_cash_6.5pct": rf_metrics,
    }

    # Combine trade logs into single dataframe
    df_all_trades = pd.concat(all_trade_logs, ignore_index=True)
    os.makedirs("data/predictions", exist_ok=True)
    df_all_trades.to_parquet("data/predictions/backtest_trade_log.parquet", index=False)
    print(f"Trade log saved. Total trade events: {len(df_all_trades):,}")

    # Build and persist consolidated equity curves dataframe
    equity_curve_dfs = []
    for sid, ddf in all_daily_equity.items():
        sub = ddf[["date", "split", "strategy_id", "net_return", "gross_return", "friction_cost", "traded_turnover", "cum_equity"]].copy()
        equity_curve_dfs.append(sub)
    df_all_equity = pd.concat(equity_curve_dfs, ignore_index=True)
    df_all_equity.to_parquet("data/predictions/backtest_equity_curves.parquet", index=False)
    print(f"Equity curves saved. Total daily records: {len(df_all_equity):,}")

    # 5. Step 4: Diagnostic Breakdowns on Out-of-Sample Test Set
    print("\n[Step 4/6] Conducting diagnostic breakdowns on Test Set...")
    
    # 5.1 Per-Company Attribution for Primary Strategies
    primary_trades = df_all_trades[
        (df_all_trades["strategy_id"] == "PreOpen_Sent_LR") & (df_all_trades["split"] == "TEST")
    ].copy()
    
    company_attribution = {}
    for sym, cdf in primary_trades.groupby("symbol"):
        net_pnl = float(cdf["net_trade_pnl"].sum())
        gross_pnl = float(cdf["gross_trade_pnl"].sum())
        friction = float(cdf["friction_cost"].sum())
        n_trades = len(cdf)
        win_trades = int((cdf["net_trade_pnl"] > 0).sum())
        win_rate = (win_trades / n_trades * 100.0) if n_trades > 0 else 0.0
        company_attribution[sym] = {
            "n_trades": n_trades,
            "win_rate_pct": win_rate,
            "gross_pnl": gross_pnl,
            "friction_cost": friction,
            "net_pnl": net_pnl,
        }

    # 5.2 Market Regime Breakdown
    test_daily_primary = all_daily_equity["PreOpen_Sent_LR"]
    test_daily_primary = test_daily_primary[test_daily_primary["split"] == "TEST"].copy()
    
    # Merge regime information by date (taking first symbol's market regime since it's an index/market-wide variable)
    market_regimes_by_date = df_test[["date", "market_regime_lag1", "adj_close_sma_ratio_20_lag1"]].drop_duplicates(subset=["date"])
    test_daily_regime = pd.merge(test_daily_primary, market_regimes_by_date, on="date", how="left")

    def get_trend_regime(sma_ratio):
        if sma_ratio > 1.01:
            return "Bullish (> +1% SMA)"
        elif sma_ratio < 0.99:
            return "Bearish (< -1% SMA)"
        return "Neutral (+/- 1% SMA)"

    test_daily_regime["trend_regime"] = test_daily_regime["adj_close_sma_ratio_20_lag1"].apply(get_trend_regime)

    regime_breakdown = {}
    # Volatility Regimes
    for vol_reg, r_df in test_daily_regime.groupby("market_regime_lag1"):
        reg_metrics = calculate_performance_metrics(r_df["net_return"], r_df["traded_turnover"])
        regime_breakdown[f"Volatility_{vol_reg}"] = reg_metrics

    # Trend Regimes
    for trend_reg, r_df in test_daily_regime.groupby("trend_regime"):
        reg_metrics = calculate_performance_metrics(r_df["net_return"], r_df["traded_turnover"])
        regime_breakdown[f"Trend_{trend_reg}"] = reg_metrics

    # 5.3 Top 5 Market Dislocation Days in Test Set
    # Find days with largest absolute Nifty/portfolio movements
    dislocation_dates = test_daily_regime.sort_values(by="gross_return", ascending=True).head(5)
    stress_period_events = []
    for _, row in dislocation_dates.iterrows():
        stress_period_events.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "gross_return_pct": float(row["gross_return"] * 100.0),
            "friction_cost_pct": float(row["friction_cost"] * 100.0),
            "net_return_pct": float(row["net_return"] * 100.0),
            "n_long": int(row["n_long"]),
            "n_short": int(row["n_short"]),
            "market_regime": row["market_regime_lag1"],
        })

    # 5.4 Transaction Cost Sensitivity Analysis
    friction_levels_bps = [0.0, 5.0, 10.0, 15.765, 20.0, 25.0, 30.0]
    sensitivity_results = {}
    for sid in ["PreOpen_Sent_LR", "PreOpen_Mkt_LR", "SetupB_FinBERT_LR", "SetupB_Mkt_LR"]:
        prob_col = [s["prob_col"] for s in strategies_spec if s["id"] == sid][0]
        tl = optimal_params[sid]["theta_long"]
        ts = optimal_params[sid]["theta_short"]
        sens_curves = []
        for f_bps in friction_levels_bps:
            d_df_sens, _ = simulate_daily_portfolio(
                df_test,
                prob_col=prob_col,
                strategy_type="dynamic_threshold",
                theta_long=tl,
                theta_short=ts,
                friction_override_bps=f_bps,
            )
            m = calculate_performance_metrics(d_df_sens["net_return"], d_df_sens["traded_turnover"])
            sens_curves.append({
                "friction_bps": f_bps,
                "sharpe_ratio": m["sharpe_ratio"],
                "cagr_pct": m["cagr_pct"],
                "cumulative_return_pct": m["cumulative_return_pct"],
            })
        sensitivity_results[sid] = sens_curves

    # 6. Step 5: Generate Publication Figures
    print("\n[Step 5/6] Generating publication figures...")
    os.makedirs("reports/figures", exist_ok=True)

    # Figure 1: Cumulative Equity Curves (Test Set Focus)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    colors = {
        "PreOpen_Sent_LR": "#1f77b4",   # Deep blue
        "PreOpen_Mkt_LR": "#7f7f7f",    # Gray
        "PreOpen_Sent_XGB": "#2ca02c",  # Green
        "PreOpen_LM_XGB": "#ff7f0e",    # Orange
        "SetupB_FinBERT_LR": "#9467bd", # Purple (dashed)
        "SetupB_Mkt_LR": "#8c564b",     # Brown (dashed)
    }

    test_dates = df_test["date"].drop_duplicates().sort_values().values
    start_date = test_dates[0]

    for sid, color in colors.items():
        sub = all_daily_equity[sid]
        sub_test = sub[sub["split"] == "TEST"].copy()
        sub_test["test_cum_equity"] = (1.0 + sub_test["net_return"]).cumprod()
        linestyle = "--" if "SetupB" in sid else "-"
        label = f"{sid} (Sharpe: {strategy_results[sid]['test_metrics']['sharpe_ratio']:.2f})"
        ax.plot(pd.to_datetime(sub_test["date"]), sub_test["test_cum_equity"], label=label, color=color, linestyle=linestyle, linewidth=1.8)

    # Buy and hold test curve
    bh_test_cum = (1.0 + bh_test["net_return"]).cumprod()
    ax.plot(pd.to_datetime(bh_test["date"]), bh_test_cum, label=f"Buy & Hold 15 (Sharpe: {bh_metrics['sharpe_ratio']:.2f})", color="#d62728", linestyle=":", linewidth=1.5)

    ax.axhline(1.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_title("Out-of-Sample Equity Curves (2025–2026 Test Set) — Net of Realistic Friction", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio Value (Base = 1.0)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig("reports/figures/stage6_equity_curves.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/stage6_equity_curves.png")

    # Figure 2: Drawdown Underwater Plot
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    for sid in ["PreOpen_Sent_LR", "PreOpen_Mkt_LR", "SetupB_FinBERT_LR"]:
        sub = all_daily_equity[sid]
        sub_test = sub[sub["split"] == "TEST"].copy()
        cum = (1.0 + sub_test["net_return"]).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak * 100.0
        color = colors[sid]
        ax.plot(pd.to_datetime(sub_test["date"]), dd, label=f"{sid} (MDD: {abs(strategy_results[sid]['test_metrics']['max_drawdown_pct']):.1f}%)", color=color, linewidth=1.6)

    ax.fill_between(pd.to_datetime(sub_test["date"]), 0, dd, color=colors["PreOpen_Sent_LR"], alpha=0.1)
    ax.set_title("Drawdown Underwater Profiles (2025–2026 Test Set)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="lower left")
    plt.tight_layout()
    fig.savefig("reports/figures/stage6_drawdowns.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/stage6_drawdowns.png")

    # Figure 3: Transaction Cost Sensitivity Curves
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for sid, color in [("PreOpen_Sent_LR", "#1f77b4"), ("PreOpen_Mkt_LR", "#7f7f7f"), ("SetupB_FinBERT_LR", "#9467bd")]:
        f_vals = [pt["friction_bps"] for pt in sensitivity_results[sid]]
        sh_vals = [pt["sharpe_ratio"] for pt in sensitivity_results[sid]]
        ax.plot(f_vals, sh_vals, marker="o", label=sid, color=color, linewidth=1.8)

    ax.axvline(15.765, color="#d62728", linestyle="--", linewidth=1.2, label="Base Friction (15.77 bps)")
    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_title("Strategy Robustness: Net Sharpe Ratio vs. Round-Trip Friction (bps)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Total Round-Trip Friction (bps)", fontsize=10)
    ax.set_ylabel("Net Out-of-Sample Sharpe Ratio", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig.savefig("reports/figures/stage6_cost_sensitivity.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/stage6_cost_sensitivity.png")

    # 7. Step 6: Serialize Structured Results JSON
    print("\n[Step 6/6] Saving structured results JSON...")
    final_output = {
        "status": "COMPLETED",
        "timestamp": "2026-11-03",
        "strategies": strategy_results,
        "reference_baselines": reference_baselines,
        "company_attribution_PreOpen_Sent_LR": company_attribution,
        "regime_breakdown_PreOpen_Sent_LR": regime_breakdown,
        "stress_period_events": stress_period_events,
        "cost_sensitivity": sensitivity_results,
    }

    os.makedirs("models/sentiment_experiments", exist_ok=True)
    with open("models/sentiment_experiments/stage6_backtesting_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("  -> Saved models/sentiment_experiments/stage6_backtesting_results.json")

    print("\n" + "=" * 80)
    print("STAGE 6 BACKTESTING EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
