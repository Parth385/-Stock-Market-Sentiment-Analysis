import json
import pandas as pd
import numpy as np

def generate_markdown_tables():
    with open("models/sentiment_experiments/sentiment_experiments_results.json", "r") as f:
        data = json.load(f)

    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    setup_a_keys = [
        ('Exp_A1_MktOnly', 'Market-Only (Reference Baseline)'),
        ('Exp_A2_MktVol', 'Market + News Volume Control'),
        ('Exp_A3_MktLM', 'Market + LM Sentiment + Volume'),
        ('Exp_A4_MktFinBERT', 'Market + FinBERT Sentiment + Volume'),
        ('Exp_A5_MktRoBERTa', 'Market + RoBERTa Sentiment + Volume'),
        ('Exp_A6_MktCombined', 'Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume')
    ]
    
    setup_b_keys = [
        ('Exp_B1_MktOnly', 'Market-Only (Reference Baseline)'),
        ('Exp_B2_MktVol', 'Market + News Volume Control'),
        ('Exp_B3_MktLM', 'Market + LM Sentiment + Volume'),
        ('Exp_B4_MktFinBERT', 'Market + FinBERT Sentiment + Volume'),
        ('Exp_B5_MktRoBERTa', 'Market + RoBERTa Sentiment + Volume'),
        ('Exp_B6_MktCombined', 'Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume')
    ]

    out_file = "reports/stage4_phase3_tables_generated.md"
    with open(out_file, "w", encoding="utf-8") as out:
        for setup_label, exps, base_key in [
            ('Setup A (Strict Pre-Open: 10 Market Features Base)', setup_a_keys, 'Exp_A1_MktOnly'),
            ('Setup B (Opening Bell: 11 Market Features Base with open_gap_pct)', setup_b_keys, 'Exp_B1_MktOnly')
        ]:
            out.write(f"\n# {setup_label}\n\n")
            for split in ['test', 'val']:
                out.write(f"### {split.upper()} SET RESULTS\n\n")
                out.write("| Experiment | Model | Feats | Accuracy (Delta / %Delta) | ROC-AUC (Delta / %Delta) | Brier Score (Delta / %Delta) | F1-Pos | F1-Macro | Precision | Recall | Threshold |\n")
                out.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                
                for exp_key, exp_name in exps:
                    exp_res = data[exp_key]
                    base_res = data[base_key]
                    f_count = exp_res['feature_count']
                    
                    for m in models:
                        m_res = exp_res[m][split]
                        b_res = base_res[m][split]
                        th = exp_res[m]['threshold']
                        
                        acc = m_res['accuracy'] * 100
                        b_acc = b_res['accuracy'] * 100
                        d_acc = acc - b_acc
                        pct_acc = (d_acc / b_acc) * 100
                        
                        auc = m_res['roc_auc']
                        b_auc = b_res['roc_auc']
                        d_auc = auc - b_auc
                        pct_auc = (d_auc / b_auc) * 100
                        
                        brier = m_res['brier_score']
                        b_brier = b_res['brier_score']
                        d_brier = brier - b_brier
                        pct_brier = (d_brier / b_brier) * 100
                        
                        f1_p = m_res['f1_positive']
                        f1_m = m_res['f1_macro']
                        prec = m_res['precision'] * 100
                        rec = m_res['recall'] * 100
                        
                        d_acc_str = f"{d_acc:+.2f}% ({pct_acc:+.2f}%)" if exp_key != base_key else "Base"
                        d_auc_str = f"{d_auc:+.4f} ({pct_auc:+.2f}%)" if exp_key != base_key else "Base"
                        d_brier_str = f"{d_brier:+.4f} ({pct_brier:+.2f}%)" if exp_key != base_key else "Base"
                        
                        out.write(f"| {exp_name} | {m} | {f_count} | {acc:.2f}% <br><small>({d_acc_str})</small> | {auc:.4f} <br><small>({d_auc_str})</small> | {brier:.4f} <br><small>({d_brier_str})</small> | {f1_p:.4f} | {f1_m:.4f} | {prec:.2f}% | {rec:.2f}% | {th:.3f} |\n")
                out.write("\n")
    print(f"Generated tables written to {out_file}")

if __name__ == '__main__':
    generate_markdown_tables()
