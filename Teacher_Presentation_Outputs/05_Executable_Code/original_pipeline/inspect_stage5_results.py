import json
import pandas as pd

def inspect_stage5():
    with open('models/sentiment_experiments/stage5_explainability_results.json', 'r') as f:
        d = json.load(f)

    print("=" * 80)
    print("STAGE 5 NUMERICAL FINDINGS AUDIT")
    print("=" * 80)

    # 1. Feature Group Contributions across all models
    print("\n--- 1. AGGREGATED ABSOLUTE SHAP IMPORTANCE BY FEATURE GROUP ---")
    rows = []
    for m_id, m_dict in d['model_explainability'].items():
        scale = m_dict['output_scale']
        tot = m_dict['total_abs_shap']
        for g in m_dict['group_attribution']:
            rows.append({
                'Model': m_id,
                'Output_Scale': scale,
                'Group': g['group'],
                'Agg_Abs_SHAP': round(g['aggregated_abs_shap'], 6),
                'Pct_Share_%': round(g['percentage_contribution'], 2)
            })
    df_groups = pd.DataFrame(rows)
    print(df_groups.pivot(index='Group', columns='Model', values='Pct_Share_%').to_string())

    # 2. Top 10 Features for Primary Models
    for m_id in ['Exp_B4_FinBERT_LR', 'Exp_B3_LM_XGB', 'Exp_B1_MktOnly_LR', 'Exp_B1_MktOnly_XGB']:
        print(f"\n--- TOP 10 FEATURES: {m_id} ({d['model_explainability'][m_id]['output_scale']}) ---")
        top_feats = pd.DataFrame(d['model_explainability'][m_id]['top_15_features'][:10])
        print(top_feats[['feature', 'group', 'mean_abs_shap']].to_string(index=False))

    # 3. Company-Level Analysis for Exp B4 FinBERT LR
    print("\n--- 3. COMPANY-LEVEL ATTRIBUTION: Exp_B4_FinBERT_LR (All 15 Companies) ---")
    comp_df = pd.DataFrame(d['company_attribution']['Exp_B4_FinBERT_LR'])
    print(comp_df[['symbol', 'accuracy', 'total_abs_shap', 'opening_gap_share_%', 'all_sentiment_share_%', 'direct_sentiment_share_%', 'news_volume_share_%']].to_string(index=False))

    # 4. Error Analysis
    print("\n--- 4. ERROR ANALYSIS: Exp_B4_FinBERT_LR ---")
    err_df = pd.DataFrame(d['error_analysis']['Exp_B4_FinBERT_LR']).T
    print(err_df[['count', 'pct_of_test', 'mean_predicted_prob', 'mean_open_gap_pct', 'has_direct_news_pct', 'opening_gap_share_%', 'direct_sentiment_share_%']].to_string())

    # 5. Regime Analysis
    print("\n--- 5. REGIME ANALYSIS: Exp_B4_FinBERT_LR ---")
    reg_df = pd.DataFrame(d['regime_analysis']['Exp_B4_FinBERT_LR']).T
    print(reg_df[['count', 'pct_of_test', 'accuracy', 'opening_gap_share_%', 'all_sentiment_share_%', 'direct_sentiment_share_%', 'core_market_share_%']].to_string())

    # 6. Feature Correlations
    print("\n--- 6. FEATURE CORRELATIONS AMONG FINBERT SENTIMENT FEATURES ---")
    corr_df = pd.DataFrame(d['feature_correlations'])
    print(corr_df.to_string())

if __name__ == '__main__':
    inspect_stage5()
