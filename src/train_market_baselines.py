"""
Stage 4 — Phase 2: Market-Only Baseline Models
Trains and evaluates:
1. Logistic Regression (L2 Regularization)
2. Random Forest (300 trees, max_depth=6)
3. XGBoost (learning_rate=0.03, max_depth=4)

Evaluates:
- Setup A: Pure Pre-Open (10 market features, excluding open_gap_pct)
- Setup B: Opening Bell (11 market features, including open_gap_pct)

Features: Leakage-free lag-1 market indicators only.
Categorical handling: market_regime_lag1 encoded via OneHotEncoder.
Scaling: StandardScaler fitted strictly on Train split (2020-2023).
Validation: 2024 calendar year for model selection & hyperparameter tuning.
Test: 2025-01-01 to 2026-03-27 for out-of-sample forward evaluation.

Outputs:
- models/market_baselines/*.joblib
- data/predictions/predictions_market_baselines.parquet
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from src.config import TARGET_TICKERS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Date boundaries approved in Phase 1
TRAIN_END = date(2023, 12, 31)
VAL_END = date(2024, 12, 31)

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_positive': float(f1_score(y_true, y_pred, zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def evaluate_per_company(df_split, prob_col, y_col='target_same_session_dir_o2c'):
    """Compute per-company accuracy and AUC."""
    results = {}
    for sym in TARGET_TICKERS:
        sub = df_split[df_split['symbol'] == sym]
        if len(sub) == 0:
            continue
        y_true = sub[y_col].values
        y_prob = sub[prob_col].values
        y_pred = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        results[sym] = {'accuracy': float(acc), 'roc_auc': float(auc), 'n_obs': len(sub)}
    return results

def train_and_evaluate_setup(df, continuous_features, setup_name):
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {setup_name}")
    print(f"Continuous features ({len(continuous_features)}): {continuous_features}")
    print("=" * 70)

    categorical_features = ['market_regime_lag1']
    feature_names = continuous_features + categorical_features

    # 1. Split data chronologically
    train_mask = df['date'] <= TRAIN_END
    val_mask = (df['date'] > TRAIN_END) & (df['date'] <= VAL_END)
    test_mask = df['date'] > VAL_END

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    X_train_raw = df_train[feature_names]
    y_train = df_train['target_same_session_dir_o2c'].values

    X_val_raw = df_val[feature_names]
    y_val = df_val['target_same_session_dir_o2c'].values

    X_test_raw = df_test[feature_names]
    y_test = df_test['target_same_session_dir_o2c'].values

    print(f"Split shapes: Train = {len(df_train):,}, Val = {len(df_val):,}, Test = {len(df_test):,}")

    # 2. Preprocessor: fit STRICTLY on Train
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    print("Fitting preprocessor strictly on training split (fit(X_train))...")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    transformed_feature_names = continuous_features + [f"regime_{cat}" for cat in preprocessor.named_transformers_['cat'].categories_[0][1:]]
    print(f"Transformed feature dimension: {X_train.shape[1]}")

    results_dict = {}
    models_dict = {'preprocessor': preprocessor, 'feature_names': transformed_feature_names}
    predictions_df = df[['date', 'symbol', 'target_same_session_dir_o2c', 'target_same_session_ret_o2c']].copy()

    # =========================================================================
    # MODEL 1: LOGISTIC REGRESSION (L2 Regularization)
    # =========================================================================
    print("\n--- Training Model 1: Logistic Regression ---")
    c_grid = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_c = 1.0
    best_val_auc = -1.0
    
    # Validation tuning loop
    for c in c_grid:
        clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c

    print(f"Best C selected via validation AUC: {best_c} (Val AUC = {best_val_auc:.4f})")

    t0 = time.time()
    best_logreg = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    best_logreg.fit(X_train, y_train)
    logreg_train_time = time.time() - t0

    t0 = time.time()
    prob_train_lr = best_logreg.predict_proba(X_train)[:, 1]
    prob_val_lr = best_logreg.predict_proba(X_val)[:, 1]
    prob_test_lr = best_logreg.predict_proba(X_test)[:, 1]
    logreg_infer_time = time.time() - t0

    models_dict['logreg'] = best_logreg
    results_dict['Logistic Regression'] = {
        'best_hyperparameters': {'C': best_c, 'penalty': 'l2', 'solver': 'lbfgs'},
        'train_time_sec': float(logreg_train_time),
        'infer_time_ms': float(logreg_infer_time * 1000),
        'train': compute_metrics(y_train, prob_train_lr),
        'val': compute_metrics(y_val, prob_val_lr),
        'test': compute_metrics(y_test, prob_test_lr)
    }

    # =========================================================================
    # MODEL 2: RANDOM FOREST (300 trees, max_depth=6)
    # =========================================================================
    print("\n--- Training Model 2: Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    prob_train_rf = rf.predict_proba(X_train)[:, 1]
    prob_val_rf = rf.predict_proba(X_val)[:, 1]
    prob_test_rf = rf.predict_proba(X_test)[:, 1]
    rf_infer_time = time.time() - t0

    models_dict['random_forest'] = rf
    results_dict['Random Forest'] = {
        'best_hyperparameters': {
            'n_estimators': 300, 'max_depth': 6, 'min_samples_split': 20,
            'min_samples_leaf': 10, 'max_features': 'sqrt', 'random_state': 42
        },
        'train_time_sec': float(rf_train_time),
        'infer_time_ms': float(rf_infer_time * 1000),
        'train': compute_metrics(y_train, prob_train_rf),
        'val': compute_metrics(y_val, prob_val_rf),
        'test': compute_metrics(y_test, prob_test_rf)
    }

    # =========================================================================
    # MODEL 3: XGBOOST (learning_rate=0.03, max_depth=4)
    # =========================================================================
    print("\n--- Training Model 3: XGBoost ---")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30,
        n_jobs=-1
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_train_time = time.time() - t0
    best_iteration = xgb.best_iteration
    print(f"XGBoost best iteration from validation early stopping: {best_iteration} / 300")

    t0 = time.time()
    prob_train_xgb = xgb.predict_proba(X_train)[:, 1]
    prob_val_xgb = xgb.predict_proba(X_val)[:, 1]
    prob_test_xgb = xgb.predict_proba(X_test)[:, 1]
    xgb_infer_time = time.time() - t0

    models_dict['xgboost'] = xgb
    results_dict['XGBoost'] = {
        'best_hyperparameters': {
            'n_estimators': 300, 'best_iteration': int(best_iteration),
            'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42
        },
        'train_time_sec': float(xgb_train_time),
        'infer_time_ms': float(xgb_infer_time * 1000),
        'train': compute_metrics(y_train, prob_train_xgb),
        'val': compute_metrics(y_val, prob_val_xgb),
        'test': compute_metrics(y_test, prob_test_xgb)
    }

    # Store predictions
    split_col = np.where(train_mask, 'TRAIN', np.where(val_mask, 'VAL', 'TEST'))
    predictions_df['split'] = split_col
    
    # Full prob arrays
    full_lr = np.zeros(len(df))
    full_rf = np.zeros(len(df))
    full_xgb = np.zeros(len(df))

    full_lr[train_mask] = prob_train_lr
    full_lr[val_mask] = prob_val_lr
    full_lr[test_mask] = prob_test_lr

    full_rf[train_mask] = prob_train_rf
    full_rf[val_mask] = prob_val_rf
    full_rf[test_mask] = prob_test_rf

    full_xgb[train_mask] = prob_train_xgb
    full_xgb[val_mask] = prob_val_xgb
    full_xgb[test_mask] = prob_test_xgb

    clean_setup = setup_name.lower().replace(" ", "_")
    predictions_df[f'prob_logreg_{clean_setup}'] = full_lr
    predictions_df[f'prob_rf_{clean_setup}'] = full_rf
    predictions_df[f'prob_xgb_{clean_setup}'] = full_xgb

    # Per-company breakdown on Test set
    test_df_preds = predictions_df[predictions_df['split'] == 'TEST'].copy()
    company_breakdown = {
        'Logistic Regression': evaluate_per_company(test_df_preds, f'prob_logreg_{clean_setup}'),
        'Random Forest': evaluate_per_company(test_df_preds, f'prob_rf_{clean_setup}'),
        'XGBoost': evaluate_per_company(test_df_preds, f'prob_xgb_{clean_setup}')
    }
    results_dict['per_company_test'] = company_breakdown

    return results_dict, models_dict, predictions_df

def main():
    start_total = time.time()
    input_path = "data/final/final_sentiment_enhanced_dataset_corrected.parquet"
    models_dir = "models/market_baselines"
    preds_dir = "data/predictions"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"Loading master corrected dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    # 1. SETUP A: PURE PRE-OPEN (10 market features, strictly before market open)
    pre_open_continuous = [
        'ret_1d_lag1', 'rsi_14_lag1', 'macd_lag1', 'macd_signal_lag1', 'macd_diff_lag1',
        'rolling_vol_5d_lag1', 'rolling_vol_21d_lag1', 'adj_close_sma_ratio_20_lag1',
        'intraday_range_pct_lag1'
    ]
    results_setup_a, models_setup_a, preds_a = train_and_evaluate_setup(
        df, pre_open_continuous, "Setup A Pre-Open"
    )

    # Save Setup A models
    joblib.dump(models_setup_a, os.path.join(models_dir, "models_setup_a_pre_open.joblib"))

    # 2. SETUP B: OPENING BELL (11 market features, including open_gap_pct)
    opening_bell_continuous = pre_open_continuous + ['open_gap_pct']
    results_setup_b, models_setup_b, preds_b = train_and_evaluate_setup(
        df, opening_bell_continuous, "Setup B Opening Bell"
    )

    # Save Setup B models
    joblib.dump(models_setup_b, os.path.join(models_dir, "models_setup_b_opening_bell.joblib"))

    # Combine predictions
    preds_combined = preds_a.merge(
        preds_b[['date', 'symbol', 'prob_logreg_setup_b_opening_bell', 'prob_rf_setup_b_opening_bell', 'prob_xgb_setup_b_opening_bell']],
        on=['date', 'symbol']
    )
    pred_path = os.path.join(preds_dir, "predictions_market_baselines.parquet")
    preds_combined.to_parquet(pred_path, index=False)
    print(f"\nSaved all predictions to {pred_path} ({len(preds_combined):,} rows)")

    # Save combined results json
    full_results = {
        'setup_a_pre_open': results_setup_a,
        'setup_b_opening_bell': results_setup_b
    }
    json_path = os.path.join(models_dir, "market_baselines_results.json")
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved evaluation metrics to {json_path}")
    print(f"Total Phase 2 execution completed in {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    main()
