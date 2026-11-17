import json
import pandas as pd
import numpy as np

def load_results():
    with open("models/sentiment_experiments/sentiment_experiments_results.json", "r") as f:
        return json.load(f)

def summarize():
    data = load_results()
    
    # Baselines for Setup A and Setup B
    base_a = data['Exp_A1_MktOnly']
    base_b = data['Exp_B1_MktOnly']
    
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    print("=" * 80)
    print("STAGE 4 — PHASE 3: SENTIMENT EXPERIMENTS SUMMARY")
    print("=" * 80)
    
    # Table 1: Setup A Test Metrics
    setup_a_exps = [
        ('Exp_A1_MktOnly', '1. Mkt-Only (Ref)'),
        ('Exp_A2_MktVol', '2. Mkt + Vol'),
        ('Exp_A3_MktLM', '3. Mkt + LM + Vol'),
        ('Exp_A4_MktFinBERT', '4. Mkt + FinBERT + Vol'),
        ('Exp_A5_MktRoBERTa', '5. Mkt + RoBERTa + Vol'),
        ('Exp_A6_MktCombined', '6. Mkt + All Sent + Vol')
    ]
    
    setup_b_exps = [
        ('Exp_B1_MktOnly', '1. Mkt-Only (Ref)'),
        ('Exp_B2_MktVol', '2. Mkt + Vol'),
        ('Exp_B3_MktLM', '3. Mkt + LM + Vol'),
        ('Exp_B4_MktFinBERT', '4. Mkt + FinBERT + Vol'),
        ('Exp_B5_MktRoBERTa', '5. Mkt + RoBERTa + Vol'),
        ('Exp_B6_MktCombined', '6. Mkt + All Sent + Vol')
    ]
    
    for split_name in ['test', 'val']:
        print(f"\n--- SETUP A: PRE-OPEN ({split_name.upper()} SET) ---")
        rows = []
        for exp_key, exp_label in setup_a_exps:
            exp_data = data[exp_key]
            for m in models:
                res = exp_data[m][split_name]
                base_res = base_a[m][split_name]
                
                acc = res['accuracy'] * 100
                auc = res['roc_auc']
                f1_pos = res['f1_positive']
                f1_macro = res['f1_macro']
                brier = res['brier_score']
                prec = res['precision'] * 100
                rec = res['recall'] * 100
                
                # Deltas vs base_a
                d_acc = acc - (base_res['accuracy'] * 100)
                d_auc = auc - base_res['roc_auc']
                d_brier = brier - base_res['brier_score']
                
                rows.append({
                    'Experiment': exp_label,
                    'Model': m,
                    'FeatCount': exp_data['feature_count'],
                    'Accuracy': f"{acc:.2f}% ({d_acc:+.2f}%)",
                    'ROC_AUC': f"{auc:.4f} ({d_auc:+.4f})",
                    'Brier': f"{brier:.4f} ({d_brier:+.4f})",
                    'F1_Pos': f"{f1_pos:.4f}",
                    'F1_Macro': f"{f1_macro:.4f}",
                    'Prec': f"{prec:.2f}%",
                    'Rec': f"{rec:.2f}%",
                    'Thresh': f"{exp_data[m]['threshold']:.4f}"
                })
        df_a = pd.DataFrame(rows)
        print(df_a.to_string(index=False))

    for split_name in ['test', 'val']:
        print(f"\n--- SETUP B: OPENING BELL ({split_name.upper()} SET) ---")
        rows = []
        for exp_key, exp_label in setup_b_exps:
            exp_data = data[exp_key]
            for m in models:
                res = exp_data[m][split_name]
                base_res = base_b[m][split_name]
                
                acc = res['accuracy'] * 100
                auc = res['roc_auc']
                f1_pos = res['f1_positive']
                f1_macro = res['f1_macro']
                brier = res['brier_score']
                prec = res['precision'] * 100
                rec = res['recall'] * 100
                
                # Deltas vs base_b
                d_acc = acc - (base_res['accuracy'] * 100)
                d_auc = auc - base_res['roc_auc']
                d_brier = brier - base_res['brier_score']
                
                rows.append({
                    'Experiment': exp_label,
                    'Model': m,
                    'FeatCount': exp_data['feature_count'],
                    'Accuracy': f"{acc:.2f}% ({d_acc:+.2f}%)",
                    'ROC_AUC': f"{auc:.4f} ({d_auc:+.4f})",
                    'Brier': f"{brier:.4f} ({d_brier:+.4f})",
                    'F1_Pos': f"{f1_pos:.4f}",
                    'F1_Macro': f"{f1_macro:.4f}",
                    'Prec': f"{prec:.2f}%",
                    'Rec': f"{rec:.2f}%",
                    'Thresh': f"{exp_data[m]['threshold']:.4f}"
                })
        df_b = pd.DataFrame(rows)
        print(df_b.to_string(index=False))

if __name__ == '__main__':
    summarize()
