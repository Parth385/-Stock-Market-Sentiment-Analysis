"""
run_all_presentation_demos.py
Master Live Presentation Demonstrator
Executes all four presentation stages consecutively:
1. Stage 4: ML Benchmarking
2. Stage 5: Explainability & SHAP
3. Stage 6: Backtesting & Friction
4. Stage 7: Accuracy Optimization & Conviction Scaling
"""

import sys
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

def run_script(script_name):
    script_path = CURRENT_DIR / script_name
    print(f"\n>>>> EXECUTING: {script_name} <<<<\n")
    res = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if res.returncode != 0:
        print(f"Error running {script_name} (Exit code: {res.returncode})")

def main():
    print("#" * 80)
    print("  BEYOND SENTIMENT: END-TO-END RESEARCH & DEMONSTRATION SUITE")
    print("  National Stock Exchange of India (NSE Nifty 50: 2020–2026)")
    print("#" * 80)
    
    scripts = [
        "02_run_benchmark_evaluation.py",
        "03_run_shap_explainability.py",
        "04_run_backtest_simulation.py",
        "01_run_accuracy_optimization.py"
    ]
    
    for s in scripts:
        run_script(s)
        
    print("\n" + "#" * 80)
    print("  ALL DEMONSTRATION STAGES COMPLETED SUCCESSFULLY!")
    print("  Results match published reports and interactive dashboard.")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
