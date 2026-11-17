"""
Stage 4 — Phase 3: Sentiment Experiments Pipeline
Benchmarks incremental predictive power of financial news sentiment representations against market-only baselines:
1. Market-Only (Reference baseline)
2. Market + News Volume control
3. Market + LM sentiment + News Volume
4. Market + FinBERT sentiment + News Volume
5. Market + RoBERTa sentiment + News Volume
6. Market + Combined (LM, FinBERT, RoBERTa) + News Volume

Evaluates both Setup A (Pre-Open) and Setup B (Opening Bell).
Models: Logistic Regression (L2), Random Forest (300 trees, d=6), XGBoost (lr=0.03, d=4).
Chronological splits: Train (2020-2023), Val (2024), Test (2025-2026).
Zero-snooping scaling: StandardScaler fit strictly on Train split.
Thresholding: Threshold frozen from Validation set maximizing Youden's J / F1.

Outputs:
- models/sentiment_experiments/*.joblib
- data/predictions/predictions_sentiment_experiments.parquet
- models/sentiment_experiments/sentiment_experiments_results.json
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from src.config import TARGET_TICKERS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

TRAIN_END = date(2023, 12, 31)
VAL_END = date(2024, 12, 31)

def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes Validation macro F1-score."""
    thresholds = np.linspace(0.40, 0.60, 41)
    best_thresh = 0.50
    best_f1 = -1.0
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        score = f1_score(y_true, pred, average='macro', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = th
    return float(best_thresh)

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_positive': float(f1_score(y_true, y_pred, zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def evaluate_per_company(df_split, prob_col, threshold=0.5, y_col='target_same_session_dir_o2c'):
    """Compute per-company metrics."""
    results = {}
    for sym in TARGET_TICKERS:
        sub = df_split[df_split['symbol'] == sym]
        if len(sub) == 0:
            continue
        y_true = sub[y_col].values
        y_prob = sub[prob_col].values
        y_pred = (y_prob >= threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        results[sym] = {'accuracy': float(acc), 'roc_auc': float(auc), 'n_obs': len(sub)}
    return results

def train_and_eval_experiment(df, continuous_features, exp_id, exp_name):
    print("=" * 75)
    print(f"RUNNING: [{exp_id}] {exp_name}")
    print(f"Total continuous features: {len(continuous_features)}")
    print("=" * 75)

    categorical_features = ['market_regime_lag1']
    feature_names = continuous_features + categorical_features

    train_mask = df['date'] <= TRAIN_END
    val_mask = (df['date'] > TRAIN_END) & (df['date'] <= VAL_END)
    test_mask = df['date'] > VAL_END

    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]

    X_train_raw = df_train[feature_names]
    y_train = df_train['target_same_session_dir_o2c'].values

    X_val_raw = df_val[feature_names]
    y_val = df_val['target_same_session_dir_o2c'].values

    X_test_raw = df_test[feature_names]
    y_test = df_test['target_same_session_dir_o2c'].values

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, continuous_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    exp_results = {'feature_count': int(X_train.shape[1]), 'continuous_features': continuous_features}
    models = {'preprocessor': preprocessor}
    pred_series = {}

    # 1. Logistic Regression
    c_grid = [1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_c = 1.0
    best_val_auc = -1.0
    for c in c_grid:
        clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c

    t0 = time.time()
    best_logreg = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    best_logreg.fit(X_train, y_train)
    lr_train_time = time.time() - t0

    t0 = time.time()
    lr_p_train = best_logreg.predict_proba(X_train)[:, 1]
    lr_p_val = best_logreg.predict_proba(X_val)[:, 1]
    lr_p_test = best_logreg.predict_proba(X_test)[:, 1]
    lr_infer_time = time.time() - t0

    # Freeze threshold on validation
    lr_thresh = find_best_threshold(y_val, lr_p_val)
    models['logreg'] = best_logreg
    pred_series['logreg'] = (lr_p_train, lr_p_val, lr_p_test)
    exp_results['Logistic Regression'] = {
        'best_hyperparameters': {'C': best_c, 'penalty': 'l2', 'solver': 'lbfgs'},
        'threshold': lr_thresh,
        'train_time_sec': float(lr_train_time),
        'infer_time_ms': float(lr_infer_time * 1000),
        'train': compute_metrics(y_train, lr_p_train, lr_thresh),
        'val': compute_metrics(y_val, lr_p_val, lr_thresh),
        'test': compute_metrics(y_test, lr_p_test, lr_thresh)
    }

    # 2. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_split=20, min_samples_leaf=10,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_p_train = rf.predict_proba(X_train)[:, 1]
    rf_p_val = rf.predict_proba(X_val)[:, 1]
    rf_p_test = rf.predict_proba(X_test)[:, 1]
    rf_infer_time = time.time() - t0

    rf_thresh = find_best_threshold(y_val, rf_p_val)
    models['random_forest'] = rf
    pred_series['random_forest'] = (rf_p_train, rf_p_val, rf_p_test)
    exp_results['Random Forest'] = {
        'best_hyperparameters': {'n_estimators': 300, 'max_depth': 6, 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'random_state': 42},
        'threshold': rf_thresh,
        'train_time_sec': float(rf_train_time),
        'infer_time_ms': float(rf_infer_time * 1000),
        'train': compute_metrics(y_train, rf_p_train, rf_thresh),
        'val': compute_metrics(y_val, rf_p_val, rf_thresh),
        'test': compute_metrics(y_test, rf_p_test, rf_thresh)
    }

    # 3. XGBoost
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss', early_stopping_rounds=30, n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_train_time = time.time() - t0

    t0 = time.time()
    xgb_p_train = xgb.predict_proba(X_train)[:, 1]
    xgb_p_val = xgb.predict_proba(X_val)[:, 1]
    xgb_p_test = xgb.predict_proba(X_test)[:, 1]
    xgb_infer_time = time.time() - t0

    xgb_thresh = find_best_threshold(y_val, xgb_p_val)
    models['xgboost'] = xgb
    pred_series['xgboost'] = (xgb_p_train, xgb_p_val, xgb_p_test)
    exp_results['XGBoost'] = {
        'best_hyperparameters': {'n_estimators': 300, 'best_iteration': int(xgb.best_iteration), 'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42},
        'threshold': xgb_thresh,
        'train_time_sec': float(xgb_train_time),
        'infer_time_ms': float(xgb_infer_time * 1000),
        'train': compute_metrics(y_train, xgb_p_train, xgb_thresh),
        'val': compute_metrics(y_val, xgb_p_val, xgb_thresh),
        'test': compute_metrics(y_test, xgb_p_test, xgb_thresh)
    }

    # Per-company test evaluation
    test_eval_df = df_test[['symbol', 'target_same_session_dir_o2c']].copy()
    test_eval_df['prob_lr'] = lr_p_test
    test_eval_df['prob_rf'] = rf_p_test
    test_eval_df['prob_xgb'] = xgb_p_test

    company_breakdown = {
        'Logistic Regression': evaluate_per_company(test_eval_df, 'prob_lr', lr_thresh),
        'Random Forest': evaluate_per_company(test_eval_df, 'prob_rf', rf_thresh),
        'XGBoost': evaluate_per_company(test_eval_df, 'prob_xgb', xgb_thresh)
    }
    exp_results['per_company_test'] = company_breakdown

    print(f"Results for [{exp_id}] on TEST:")
    print(f"  LR:  Acc = {exp_results['Logistic Regression']['test']['accuracy']*100:.2f}%, AUC = {exp_results['Logistic Regression']['test']['roc_auc']:.4f}, Brier = {exp_results['Logistic Regression']['test']['brier_score']:.4f}")
    print(f"  RF:  Acc = {exp_results['Random Forest']['test']['accuracy']*100:.2f}%, AUC = {exp_results['Random Forest']['test']['roc_auc']:.4f}, Brier = {exp_results['Random Forest']['test']['brier_score']:.4f}")
    print(f"  XGB: Acc = {exp_results['XGBoost']['test']['accuracy']*100:.2f}%, AUC = {exp_results['XGBoost']['test']['roc_auc']:.4f}, Brier = {exp_results['XGBoost']['test']['brier_score']:.4f}")

    return exp_results, models, pred_series

def main():
    start_total = time.time()
    input_path = "data/final/final_sentiment_enhanced_dataset_corrected.parquet"
    models_dir = "models/sentiment_experiments"
    preds_dir = "data/predictions"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"Loading master corrected dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Base feature definitions
    pre_open_mkt = [
        'ret_1d_lag1', 'rsi_14_lag1', 'macd_lag1', 'macd_signal_lag1', 'macd_diff_lag1',
        'rolling_vol_5d_lag1', 'rolling_vol_21d_lag1', 'adj_close_sma_ratio_20_lag1',
        'intraday_range_pct_lag1'
    ]
    opening_bell_mkt = pre_open_mkt + ['open_gap_pct']

    volume_features = [
        'direct_news_count', 'direct_news_count_lag1', 'direct_news_count_roll3', 'direct_news_count_roll7',
        'sector_news_count', 'market_news_count',
        'has_direct_news', 'has_sector_news'
    ]

    lm_features = [c for c in df.columns if c.startswith('lm_') and not c.endswith('_exposure')]
    finbert_features = [c for c in df.columns if c.startswith('finbert_') and not c.endswith('_exposure')]
    roberta_features = [c for c in df.columns if c.startswith('roberta_') and not c.endswith('_exposure')]
    all_sentiment_features = lm_features + finbert_features + roberta_features

    # Master Experiment Catalog (Exps 1 to 6 across Setups A and B)
    experiments_config = [
        # Setup A: Pre-Open
        ('Exp_A1_MktOnly', 'Setup A: Market-Only Reference', pre_open_mkt),
        ('Exp_A2_MktVol', 'Setup A: Market + News Volume Control', pre_open_mkt + volume_features),
        ('Exp_A3_MktLM', 'Setup A: Market + LM Sentiment + Volume', pre_open_mkt + lm_features + volume_features),
        ('Exp_A4_MktFinBERT', 'Setup A: Market + FinBERT Sentiment + Volume', pre_open_mkt + finbert_features + volume_features),
        ('Exp_A5_MktRoBERTa', 'Setup A: Market + RoBERTa Sentiment + Volume', pre_open_mkt + roberta_features + volume_features),
        ('Exp_A6_MktCombined', 'Setup A: Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume', pre_open_mkt + all_sentiment_features + volume_features),

        # Setup B: Opening Bell
        ('Exp_B1_MktOnly', 'Setup B: Market-Only Reference', opening_bell_mkt),
        ('Exp_B2_MktVol', 'Setup B: Market + News Volume Control', opening_bell_mkt + volume_features),
        ('Exp_B3_MktLM', 'Setup B: Market + LM Sentiment + Volume', opening_bell_mkt + lm_features + volume_features),
        ('Exp_B4_MktFinBERT', 'Setup B: Market + FinBERT Sentiment + Volume', opening_bell_mkt + finbert_features + volume_features),
        ('Exp_B5_MktRoBERTa', 'Setup B: Market + RoBERTa Sentiment + Volume', opening_bell_mkt + roberta_features + volume_features),
        ('Exp_B6_MktCombined', 'Setup B: Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume', opening_bell_mkt + all_sentiment_features + volume_features)
    ]

    all_results = {}
    preds_df = df[['date', 'symbol', 'target_same_session_dir_o2c', 'target_same_session_ret_o2c']].copy()
    train_mask = df['date'] <= TRAIN_END
    val_mask = (df['date'] > TRAIN_END) & (df['date'] <= VAL_END)
    preds_df['split'] = np.where(train_mask, 'TRAIN', np.where(val_mask, 'VAL', 'TEST'))

    for exp_id, exp_name, feats in experiments_config:
        res, trained_models, p_series = train_and_eval_experiment(df, feats, exp_id, exp_name)
        all_results[exp_id] = res

        # Save model joblib
        joblib.dump(trained_models, os.path.join(models_dir, f"models_{exp_id}.joblib"))

        # Add predictions to master predictions df
        for m_name, (pt, pv, pte) in p_series.items():
            full_arr = np.zeros(len(df))
            full_arr[train_mask] = pt
            full_arr[val_mask] = pv
            full_arr[df['date'] > VAL_END] = pte
            preds_df[f'{exp_id}_{m_name}_prob'] = full_arr

    # Save predictions parquet
    pred_out_path = os.path.join(preds_dir, "predictions_sentiment_experiments.parquet")
    preds_df.to_parquet(pred_out_path, index=False)
    print(f"\nSaved all experiment predictions to {pred_out_path} ({len(preds_df):,} rows × {len(preds_df.columns)} cols)")

    # Save JSON results
    json_path = os.path.join(models_dir, "sentiment_experiments_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full experiment metrics to {json_path}")
    print(f"All Stage 4 Phase 3 experiments completed in {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    main()
