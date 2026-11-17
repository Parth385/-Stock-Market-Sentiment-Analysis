"""
Stage 5 — Explainability, Interpretation & Error Analysis Pipeline
Computes and aggregates SHAP values for Stage 4 frozen models:
1. Exp_B4_FinBERT Logistic Regression (Setup B)
2. Exp_B3_LM XGBoost (Setup B)
3. Exp_B1_MktOnly Logistic Regression (Setup B reference control)
4. Exp_B1_MktOnly XGBoost (Setup B reference control)
5. Exp_B4_FinBERT Random Forest (Setup B comparator)
6. Exp_A4_FinBERT XGBoost (Setup A comparator)

Adheres strictly to the four approved clarifications:
- Uses 'aggregated absolute SHAP importance' for group sums.
- Defines FP/FN using actual model predictions and observed labels.
- Documents output space (log-odds vs. probability).
- Evaluates all 15 companies individually before exploratory grouping.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

TRAIN_END = date(2023, 12, 31)
VAL_END = date(2024, 12, 31)

def assign_feature_group(feature_name):
    """Categorize transformed feature names into functional groups."""
    fn = feature_name.lower()
    # Strip transformer prefixes if present (e.g. 'num__', 'cat__')
    if '__' in fn:
        fn_clean = fn.split('__')[1]
    else:
        fn_clean = fn
        
    if fn_clean == 'open_gap_pct':
        return 'Opening Gap'
        
    # News volume features
    volume_keys = ['news_count', 'has_direct_news', 'has_sector_news']
    if any(k in fn_clean for k in volume_keys):
        return 'News Volume'
        
    is_sent = ('finbert_' in fn_clean or 'lm_' in fn_clean or 'roberta_' in fn_clean)
    
    # Temporal sentiment (lags and rolling averages of sentiment)
    temporal_keys = ['_lag1', '_roll3', '_roll7']
    if is_sent and any(fn_clean.endswith(k) for k in temporal_keys):
        return 'Temporal Sentiment'

    # Global / Market macro sentiment
    if is_sent and ('global' in fn_clean or 'market' in fn_clean):
        return 'Global Sentiment'
        
    # Sector sentiment
    if is_sent and 'sector' in fn_clean:
        return 'Sector Sentiment'
        
    # Direct company sentiment
    if is_sent and 'direct' in fn_clean:
        return 'Direct Sentiment'
        
    # Core market features
    market_keys = [
        'ret_1d', 'rsi_14', 'macd', 'rolling_vol', 'intraday_range',
        'adj_close_sma_ratio', 'market_regime'
    ]
    if any(k in fn_clean for k in market_keys):
        return 'Core Market'
        
    return 'Other'

def clean_feature_label(name):
    if '__' in name:
        return name.split('__')[1]
    return name

def run_stage5():
    print("=" * 80)
    print("STAGE 5: EXPLAINABILITY, INTERPRETATION & ERROR ANALYSIS")
    print("=" * 80)
    
    input_path = "data/final/final_sentiment_enhanced_dataset_corrected.parquet"
    models_dir = "models/sentiment_experiments"
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading master dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    train_mask = df['date'] <= TRAIN_END
    val_mask = (df['date'] > TRAIN_END) & (df['date'] <= VAL_END)
    test_mask = df['date'] > VAL_END
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    y_test = df_test['target_same_session_dir_o2c'].values
    n_test = len(df_test)
    print(f"Dataset split: Train = {len(df_train)}, Val = {len(df_val)}, Test = {n_test}")
    
    # Load Stage 4 frozen threshold results
    results_path = os.path.join(models_dir, "sentiment_experiments_results.json")
    with open(results_path, 'r') as f:
        stage4_results = json.load(f)
        
    # Model configs to explain
    explain_configs = [
        {
            'name': 'Exp_B4_FinBERT_LR',
            'exp_id': 'Exp_B4_MktFinBERT',
            'model_key': 'logreg',
            'explainer_type': 'linear',
            'output_scale': 'Log-Odds Margin Space (logit)'
        },
        {
            'name': 'Exp_B3_LM_XGB',
            'exp_id': 'Exp_B3_MktLM',
            'model_key': 'xgboost',
            'explainer_type': 'tree',
            'output_scale': 'Log-Odds Margin Space (raw margin)'
        },
        {
            'name': 'Exp_B1_MktOnly_LR',
            'exp_id': 'Exp_B1_MktOnly',
            'model_key': 'logreg',
            'explainer_type': 'linear',
            'output_scale': 'Log-Odds Margin Space (logit)'
        },
        {
            'name': 'Exp_B1_MktOnly_XGB',
            'exp_id': 'Exp_B1_MktOnly',
            'model_key': 'xgboost',
            'explainer_type': 'tree',
            'output_scale': 'Log-Odds Margin Space (raw margin)'
        },
        {
            'name': 'Exp_B4_FinBERT_RF',
            'exp_id': 'Exp_B4_MktFinBERT',
            'model_key': 'random_forest',
            'explainer_type': 'tree_rf',
            'output_scale': 'Probability Space (class 1 probability)'
        },
        {
            'name': 'Exp_A4_FinBERT_XGB',
            'exp_id': 'Exp_A4_MktFinBERT',
            'model_key': 'xgboost',
            'explainer_type': 'tree',
            'output_scale': 'Log-Odds Margin Space (raw margin)'
        }
    ]
    
    shap_results = {}
    shap_arrays = {}
    
    for cfg in explain_configs:
        cfg_name = cfg['name']
        exp_id = cfg['exp_id']
        m_key = cfg['model_key']
        ex_type = cfg['explainer_type']
        out_scale = cfg['output_scale']
        
        print(f"\n--- Computing SHAP for [{cfg_name}] ({out_scale}) ---")
        joblib_path = os.path.join(models_dir, f"models_{exp_id}.joblib")
        pkg = joblib.load(joblib_path)
        
        preprocessor = pkg['preprocessor']
        model = pkg[m_key]
        frozen_thresh = stage4_results[exp_id]['Logistic Regression' if m_key == 'logreg' else ('Random Forest' if m_key == 'random_forest' else 'XGBoost')]['threshold']
        
        # Continuous feature definitions
        feats_list = stage4_results[exp_id]['continuous_features'] + ['market_regime_lag1']
        
        X_train_raw = df_train[feats_list]
        X_test_raw = df_test[feats_list]
        
        X_train_trans = preprocessor.transform(X_train_raw)
        X_test_trans = preprocessor.transform(X_test_raw)
        
        feature_names = preprocessor.get_feature_names_out()
        cleaned_fnames = [clean_feature_label(fn) for fn in feature_names]
        feature_groups = [assign_feature_group(fn) for fn in feature_names]
        
        # Compute SHAP
        if ex_type == 'linear':
            # LinearExplainer with background sample
            np.random.seed(42)
            bg_indices = np.random.choice(X_train_trans.shape[0], size=500, replace=False)
            background = X_train_trans[bg_indices]
            explainer = shap.LinearExplainer(model, background)
            shap_vals = explainer.shap_values(X_test_trans)
            # shap_vals for linear model on 2D input is 2D: (N, M)
            if isinstance(shap_vals, list):
                shap_mat = shap_vals[1]
            else:
                shap_mat = shap_vals
                
        elif ex_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_trans)
            if isinstance(shap_vals, list):
                shap_mat = shap_vals[1]
            else:
                shap_mat = shap_vals
                
        elif ex_type == 'tree_rf':
            # Random forest in shap gives [N, M, 2] or list of 2
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_trans)
            if isinstance(shap_vals, list):
                shap_mat = shap_vals[1]
            elif len(shap_vals.shape) == 3:
                shap_mat = shap_vals[:, :, 1]
            else:
                shap_mat = shap_vals
                
        shap_arrays[cfg_name] = shap_mat
        
        # 1. Feature Level Importance
        mean_abs_feat = np.mean(np.abs(shap_mat), axis=0)
        feat_df = pd.DataFrame({
            'feature': cleaned_fnames,
            'raw_feature': feature_names,
            'group': feature_groups,
            'mean_abs_shap': mean_abs_feat
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        # 2. Group Level Attribution (Aggregated Absolute SHAP Importance)
        total_abs_shap = np.sum(mean_abs_feat)
        group_df = feat_df.groupby('group')['mean_abs_shap'].sum().reset_index()
        group_df.columns = ['group', 'aggregated_abs_shap']
        group_df['percentage_contribution'] = (group_df['aggregated_abs_shap'] / total_abs_shap) * 100.0
        group_df = group_df.sort_values('percentage_contribution', ascending=False).reset_index(drop=True)
        
        print(f"Group Attribution for {cfg_name}:")
        for _, r in group_df.iterrows():
            print(f"  {r['group']:<22}: {r['aggregated_abs_shap']:.6f} ({r['percentage_contribution']:.2f}%)")
            
        shap_results[cfg_name] = {
            'output_scale': out_scale,
            'frozen_threshold': frozen_thresh,
            'total_abs_shap': float(total_abs_shap),
            'group_attribution': group_df.to_dict(orient='records'),
            'top_15_features': feat_df.head(15).to_dict(orient='records')
        }
        
    # Save SHAP arrays
    np.savez_compressed("data/predictions/shap_values_stage5.npz", **shap_arrays)
    print("\nSaved all SHAP arrays to data/predictions/shap_values_stage5.npz")
    
    # Detailed Analysis on Primary Models: Exp_B4_FinBERT_LR and Exp_B3_LM_XGB
    primary_models = [
        ('Exp_B4_FinBERT_LR', 'Exp_B4_MktFinBERT', 'logreg', 'Log-Odds Margin Space'),
        ('Exp_B3_LM_XGB', 'Exp_B3_MktLM', 'xgboost', 'Log-Odds Margin Space')
    ]
    
    company_results = {}
    error_results = {}
    regime_results = {}
    
    for cfg_name, exp_id, m_key, scale in primary_models:
        shap_mat = shap_arrays[cfg_name]
        pkg = joblib.load(os.path.join(models_dir, f"models_{exp_id}.joblib"))
        preprocessor = pkg['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        feature_groups = np.array([assign_feature_group(fn) for fn in feature_names])
        
        pred_col = f'{exp_id}_{m_key}_prob'
        pred_df = pd.read_parquet("data/predictions/predictions_sentiment_experiments.parquet")
        test_pred_prob = pred_df.loc[test_mask, pred_col].values
        
        frozen_th = shap_results[cfg_name]['frozen_threshold']
        hard_pred = (test_pred_prob >= frozen_th).astype(int)
        
        # C. Company Analysis: All 15 companies individually first
        comp_records = []
        for sym in df_test['symbol'].unique():
            idx_sym = np.where(df_test['symbol'].values == sym)[0]
            sub_shap = shap_mat[idx_sym]
            sub_y = y_test[idx_sym]
            sub_pred = hard_pred[idx_sym]
            sub_prob = test_pred_prob[idx_sym]
            
            acc = float(np.mean(sub_pred == sub_y)) * 100.0
            
            # Group shares for this company
            sub_mean_abs = np.mean(np.abs(sub_shap), axis=0)
            sub_tot = np.sum(sub_mean_abs)
            
            # Aggregated group shares
            group_shares = {}
            for g in np.unique(feature_groups):
                g_idx = np.where(feature_groups == g)[0]
                g_sum = np.sum(sub_mean_abs[g_idx])
                group_shares[g] = float((g_sum / sub_tot) * 100.0) if sub_tot > 0 else 0.0
                
            sent_total_share = group_shares.get('Direct Sentiment', 0.0) + group_shares.get('Temporal Sentiment', 0.0) + group_shares.get('Sector Sentiment', 0.0) + group_shares.get('Global Sentiment', 0.0)
            
            comp_records.append({
                'symbol': sym,
                'n_obs': int(len(idx_sym)),
                'accuracy': acc,
                'mean_predicted_prob': float(np.mean(sub_prob)),
                'total_abs_shap': float(sub_tot),
                'opening_gap_share_%': group_shares.get('Opening Gap', 0.0),
                'core_market_share_%': group_shares.get('Core Market', 0.0),
                'news_volume_share_%': group_shares.get('News Volume', 0.0),
                'direct_sentiment_share_%': group_shares.get('Direct Sentiment', 0.0),
                'all_sentiment_share_%': sent_total_share
            })
            
        company_df = pd.DataFrame(comp_records).sort_values('accuracy', ascending=False)
        company_results[cfg_name] = company_df.to_dict(orient='records')
        
        # D. Error Analysis
        # Defined by actual predictions and observed targets:
        # False Positives: hard_pred == 1 and y_test == 0
        # False Negatives: hard_pred == 0 and y_test == 1
        # True Positives: hard_pred == 1 and y_test == 1
        # True Negatives: hard_pred == 0 and y_test == 0
        is_fp = (hard_pred == 1) & (y_test == 0)
        is_fn = (hard_pred == 0) & (y_test == 1)
        is_tp = (hard_pred == 1) & (y_test == 1)
        is_tn = (hard_pred == 0) & (y_test == 0)
        
        # High confidence errors:
        # High-confidence FP: test_pred_prob >= 0.55 & y == 0
        # High-confidence FN: test_pred_prob <= 0.40 & y == 1
        is_hi_fp = (test_pred_prob >= 0.55) & (y_test == 0)
        is_hi_fn = (test_pred_prob <= 0.40) & (y_test == 1)
        
        error_subsets = {
            'True_Positive': is_tp,
            'True_Negative': is_tn,
            'False_Positive_Bull_Trap': is_fp,
            'False_Negative_Missed_Up': is_fn,
            'High_Confidence_FP': is_hi_fp,
            'High_Confidence_FN': is_hi_fn
        }
        
        err_metrics = {}
        for err_label, mask in error_subsets.items():
            count = int(np.sum(mask))
            if count == 0:
                continue
            sub_shap = shap_mat[mask]
            sub_mean_abs = np.mean(np.abs(sub_shap), axis=0)
            sub_tot = np.sum(sub_mean_abs)
            
            # Check opening gap and sentiment averages in raw dataframe
            sub_df = df_test[mask]
            mean_gap = float(sub_df['open_gap_pct'].mean())
            mean_vol = float(sub_df['direct_news_count'].mean())
            has_news_pct = float(sub_df['has_direct_news'].mean() * 100.0)
            
            # Key feature attributions
            g_shares = {}
            for g in np.unique(feature_groups):
                g_idx = np.where(feature_groups == g)[0]
                g_sum = np.sum(sub_mean_abs[g_idx])
                g_shares[g] = float((g_sum / sub_tot) * 100.0) if sub_tot > 0 else 0.0
                
            err_metrics[err_label] = {
                'count': count,
                'pct_of_test': float(count / n_test * 100.0),
                'mean_predicted_prob': float(np.mean(test_pred_prob[mask])),
                'mean_open_gap_pct': mean_gap,
                'mean_direct_news_count': mean_vol,
                'has_direct_news_pct': has_news_pct,
                'opening_gap_share_%': g_shares.get('Opening Gap', 0.0),
                'core_market_share_%': g_shares.get('Core Market', 0.0),
                'direct_sentiment_share_%': g_shares.get('Direct Sentiment', 0.0),
                'news_volume_share_%': g_shares.get('News Volume', 0.0)
            }
        error_results[cfg_name] = err_metrics
        
        # Analysis of Top 10 High Volatility Difficult Sessions
        # Session volatility defined by intraday range mean across all 15 stocks
        session_vols = df_test.groupby('date')['intraday_range_pct_lag1'].mean().reset_index()
        top_10_dates = session_vols.sort_values('intraday_range_pct_lag1', ascending=False).head(10)['date'].values
        
        volatile_mask = df_test['date'].isin(top_10_dates).values
        vol_acc = float(np.mean(hard_pred[volatile_mask] == y_test[volatile_mask]) * 100.0)
        vol_tot_obs = int(np.sum(volatile_mask))
        
        vol_shap_mean = np.mean(np.abs(shap_mat[volatile_mask]), axis=0)
        vol_tot_shap = np.sum(vol_shap_mean)
        vol_g_shares = {}
        for g in np.unique(feature_groups):
            g_idx = np.where(feature_groups == g)[0]
            g_sum = np.sum(vol_shap_mean[g_idx])
            vol_g_shares[g] = float((g_sum / vol_tot_shap) * 100.0) if vol_tot_shap > 0 else 0.0
            
        error_results[cfg_name]['Top_10_Volatile_Sessions'] = {
            'dates': [str(d) for d in top_10_dates],
            'obs_count': vol_tot_obs,
            'accuracy': vol_acc,
            'opening_gap_share_%': vol_g_shares.get('Opening Gap', 0.0),
            'core_market_share_%': vol_g_shares.get('Core Market', 0.0),
            'direct_sentiment_share_%': vol_g_shares.get('Direct Sentiment', 0.0),
            'news_volume_share_%': vol_g_shares.get('News Volume', 0.0)
        }
        
        # E. Market Regime Conditioning
        # 1. Trend Regimes based on 20-day SMA ratio:
        # Bull: ratio > +0.01 (+1% above SMA-20)
        # Bear: ratio < -0.01 (-1% below SMA-20)
        # Neutral: within +/- 1% of SMA-20
        sma_ratio = df_test['adj_close_sma_ratio_20_lag1'].values
        reg_masks = {
            'Bull_Trend': sma_ratio > 0.01,
            'Bear_Trend': sma_ratio < -0.01,
            'Neutral_Trend': (sma_ratio >= -0.01) & (sma_ratio <= 0.01),
            'Low_Volatility': (df_test['market_regime_lag1'] == 'LOW_VOL').values,
            'High_Volatility': (df_test['market_regime_lag1'] == 'HIGH_VOL').values
        }

        reg_records = {}
        for reg_label, reg_mask in reg_masks.items():
            count = int(np.sum(reg_mask))
            if count == 0:
                continue
            sub_acc = float(np.mean(hard_pred[reg_mask] == y_test[reg_mask]) * 100.0)
            sub_shap = shap_mat[reg_mask]
            sub_mean_abs = np.mean(np.abs(sub_shap), axis=0)
            sub_tot = np.sum(sub_mean_abs)
            
            reg_g_shares = {}
            for g in np.unique(feature_groups):
                g_idx = np.where(feature_groups == g)[0]
                g_sum = np.sum(sub_mean_abs[g_idx])
                reg_g_shares[g] = float((g_sum / sub_tot) * 100.0) if sub_tot > 0 else 0.0
                
            reg_records[reg_label] = {
                'count': count,
                'pct_of_test': float(count / n_test * 100.0),
                'accuracy': sub_acc,
                'opening_gap_share_%': reg_g_shares.get('Opening Gap', 0.0),
                'core_market_share_%': reg_g_shares.get('Core Market', 0.0),
                'direct_sentiment_share_%': reg_g_shares.get('Direct Sentiment', 0.0),
                'all_sentiment_share_%': reg_g_shares.get('Direct Sentiment', 0.0) + reg_g_shares.get('Temporal Sentiment', 0.0) + reg_g_shares.get('Sector Sentiment', 0.0) + reg_g_shares.get('Global Sentiment', 0.0),
                'news_volume_share_%': reg_g_shares.get('News Volume', 0.0)
            }
        regime_results[cfg_name] = reg_records

    # F. Robustness: Feature Correlations among Top FinBERT Sentiment Features
    finbert_cols = [
        'finbert_direct_pos', 'finbert_direct_neu', 'finbert_direct_neg', 'finbert_direct_polarity',
        'finbert_direct_polarity_lag1', 'finbert_direct_polarity_roll3',
        'finbert_sector_polarity', 'finbert_market_polarity', 'direct_news_count', 'open_gap_pct'
    ]
    corr_mat = df_test[finbert_cols].corr().round(4).to_dict()

    # Save full JSON results
    full_output = {
        'model_explainability': shap_results,
        'company_attribution': company_results,
        'error_analysis': error_results,
        'regime_analysis': regime_results,
        'feature_correlations': corr_mat
    }
    
    json_path = os.path.join(models_dir, "stage5_explainability_results.json")
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved complete Stage 5 structured results to {json_path}")
    
    # Generate Publication-Ready Figures
    generate_stage5_figures(shap_arrays, shap_results, df_test, company_results)

def generate_stage5_figures(shap_arrays, shap_results, df_test, company_results):
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Figure 1: Aggregated Absolute SHAP Importance by Feature Group (Exp B4 LR vs Exp B3 XGB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (m_id, ax, title) in enumerate([
        ('Exp_B4_FinBERT_LR', axes[0], 'Exp B4: FinBERT Logistic Regression (Setup B)\nOutput Scale: Log-Odds Margin'),
        ('Exp_B3_LM_XGB', axes[1], 'Exp B3: LM XGBoost (Setup B)\nOutput Scale: Log-Odds Margin')
    ]):
        g_data = shap_results[m_id]['group_attribution']
        groups = [r['group'] for r in g_data]
        pcts = [r['percentage_contribution'] for r in g_data]
        
        # Sort ascending for horizontal bar chart
        sort_idx = np.argsort(pcts)
        sorted_groups = [groups[i] for i in sort_idx]
        sorted_pcts = [pcts[i] for i in sort_idx]
        
        colors = ['#1f77b4' if g == 'Opening Gap' else ('#2ca02c' if 'Sentiment' in g else ('#ff7f0e' if 'Volume' in g else '#7f7f7f')) for g in sorted_groups]
        
        ax.barh(sorted_groups, sorted_pcts, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xlabel('Percentage Contribution to Total Absolute Attribution (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(sorted_pcts) * 1.15)
        for i, v in enumerate(sorted_pcts):
            ax.text(v + 0.8, i, f"{v:.1f}%", va='center', fontsize=10, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    f1_path = os.path.join(fig_dir, "stage5_feature_group_importance.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {f1_path}")
    
    # Figure 2: Top 15 Individual Feature Importance for Exp B4 LR
    b4_top15 = shap_results['Exp_B4_FinBERT_LR']['top_15_features']
    f_names = [r['feature'] for r in b4_top15][::-1]
    f_scores = [r['mean_abs_shap'] for r in b4_top15][::-1]
    f_groups = [r['group'] for r in b4_top15][::-1]
    
    plt.figure(figsize=(10, 7))
    bar_colors = ['#1f77b4' if g == 'Opening Gap' else ('#2ca02c' if 'Sentiment' in g else ('#ff7f0e' if 'Volume' in g else '#7f7f7f')) for g in f_groups]
    plt.barh(f_names, f_scores, color=bar_colors, edgecolor='black', alpha=0.85)
    plt.xlabel('Mean Absolute SHAP Value (Log-Odds Scale)', fontsize=11)
    plt.title('Top 15 Most Influential Features: Exp B4 FinBERT Logistic Regression\n(Log-Odds Output Margin)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    f2_path = os.path.join(fig_dir, "stage5_top15_finbert_lr.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {f2_path}")

    # Figure 3: Company-Level Sentiment Attribution Share vs. Test Accuracy
    comp_list = company_results['Exp_B4_FinBERT_LR']
    comp_df = pd.DataFrame(comp_list).sort_values('all_sentiment_share_%', ascending=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(comp_df['symbol'], comp_df['all_sentiment_share_%'], color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.ylabel('Aggregated Sentiment Attribution Share (%)', fontsize=11)
    plt.title('Aggregated Sentiment Attribution Share across all 15 NSE Cohort Stocks\n(Exp B4 FinBERT Logistic Regression)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, acc in zip(bars, comp_df['accuracy']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    f3_path = os.path.join(fig_dir, "stage5_company_sentiment_shares.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {f3_path}")
    print("All Stage 5 figures successfully rendered and saved.")

if __name__ == '__main__':
    run_stage5()
