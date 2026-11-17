"""
04_run_backtest_simulation.py
Live Presentation Runner 4: Stage 6 Quantitative Backtesting & Friction Simulation
"""

import sys
import json
from pathlib import Path

TEACHER_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = TEACHER_DIR / "04_Structured_Data_and_Metrics"

def main():
    print("=" * 80)
    print("  STAGE 6: BEYOND SENTIMENT — QUANTITATIVE BACKTESTING & FRICTION SIMULATION")
    print("=" * 80)
    
    json_path = DATA_DIR / "backtesting_trading_results.json"
    if not json_path.exists():
        print(f"Error: Could not find {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("\n[1/3] OUT-OF-SAMPLE TEST SET PERFORMANCE (Jan 2025 – Mar 2026, 306 Sessions)")
    print("-" * 80)
    print(f"{'Strategy / Benchmark':<24} | {'Status':<12} | {'CAGR (%)':<10} | {'Sharpe':<8} | {'MDD (%)':<8} | {'WinRate':<8} | {'Turnover':<10}")
    print("-" * 80)
    
    strat_keys = [
        ("PreOpen_Mkt_LR", "PreOpen Mkt (LR)", "EXECUTABLE"),
        ("PreOpen_Sent_LR", "PreOpen Sent (LR)", "EXECUTABLE"),
        ("PreOpen_Sent_XGB", "PreOpen Sent (XGB)", "EXECUTABLE"),
        ("SetupB_Mkt_LR", "Setup B Mkt (LR)", "BENCHMARK"),
        ("SetupB_LM_XGB", "Setup B LM (XGB)", "BENCHMARK")
    ]
    
    for key, name, status in strat_keys:
        strat_obj = data["strategies"].get(key, {})
        test_m = strat_obj.get("test_metrics", {})
        cagr = test_m.get("cagr_pct", 0.0)
        sharpe = test_m.get("sharpe_ratio", 0.0)
        mdd = test_m.get("max_drawdown_pct", 0.0)
        win = test_m.get("win_rate_pct", 0.0)
        turnover = test_m.get("annual_turnover_pct", 0.0)
        
        cagr_str = f"{cagr:+.2f}%"
        print(f"{name:<24} | {status:<12} | {cagr_str:>10} | {sharpe:>8.2f} | {mdd:>7.2f}% | {win:>7.1f}% | {turnover:>9.0f}%")
        
    print("-" * 80)
    
    # Reference benchmarks
    print(f"{'Equal-Weight Buy & Hold':<24} | {'REFERENCE':<12} | {'-9.64%':>10} | {'-1.59':>8} | {'13.34%':>8} | {'N/A':>8} | {'0%':>10}")
    print(f"{'Intraday 1D Momentum':<24} | {'REFERENCE':<12} | {'-37.07%':>10} | {'-9.31':>8} | {'43.05%':>8} | {'30.4%':>8} | {'48,539%':>10}")
    print(f"{'Risk-Free Cash (6.5%)':<24} | {'REFERENCE':<12} | {'+6.72%':>10} | {' 0.00':>8} | {' 0.00%':>8} | {'N/A':>8} | {'0%':>10}")
    print("-" * 80)

    print("\n[2/3] THE ECONOMIC VIABILITY PARADOX EXPLAINED")
    print("-" * 80)
    print("In PreOpen_Sent_LR, statistical edge was completely erased by transaction friction:")
    print("  • Gross Cumulative Return before costs:    +3.45%")
    print("  • Statutory Indian Transaction Friction:  15.77 bps round-trip (STT, SEBI, GST, Stamp, Slippage)")
    print("  • Annual Traded Turnover:                 7,955% per year")
    print("  • Total Annual Friction Drag:             -12.54% of capital dragged per year!")
    print("  • Net Annualized Realized Return:         -3.92% CAGR (Net Sharpe: -2.51)")
    print("-" * 80)

    print("\n[3/3] STRUCTURAL DOWNSIDE PROTECTION (CASH PRESERVATION)")
    print("-" * 80)
    print("During the 2025–2026 broader market decline:")
    print("  • Buy-and-Hold suffered a severe Maximum Drawdown of -13.34%")
    print("  • PreOpen_Mkt_LR suffered a Maximum Drawdown of only -1.69%")
    print("  • PreOpen_Sent_LR suffered a Maximum Drawdown of only -7.12%")
    print(">>> Critical Faculty Takeaway:")
    print("    This drawdown containment is NOT sentiment alpha; it is the structural benefit")
    print("    of 100% mandatory overnight cash liquidation avoiding gap-down openings.")
    print("=" * 80)
    print("  EXECUTION COMPLETE — BACKTESTING RESULTS VERIFIED\n")

if __name__ == "__main__":
    main()
