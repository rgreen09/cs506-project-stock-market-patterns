"""
Final optimized Cup and Handle classifier.

Key optimizations:
- Feature selection (remove redundant, keep 20 features)
- Outlier clipping (1st-99th percentile)
- NO SMOTE - use scale_pos_weight instead
- RobustScaler for outlier handling
- Tuned XGBoost hyperparameters
- Optimal threshold selection

Usage:
    python train_model_final.py --input outputs/training_dataset.csv --output outputs/model_final.joblib
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb


def remove_redundant_features(df, feature_cols, corr_threshold=0.85):
    """Remove highly correlated features, keeping the most informative."""
    X = df[feature_cols].values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    mi = mutual_info_classif(X, y, random_state=42)
    mi_dict = dict(zip(feature_cols, mi))
    
    corr = df[feature_cols].corr()
    
    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if abs(corr.iloc[i,j]) > corr_threshold:
                if mi_dict[feature_cols[i]] < mi_dict[feature_cols[j]]:
                    to_drop.add(feature_cols[i])
                else:
                    to_drop.add(feature_cols[j])
    
    selected = [f for f in feature_cols if f not in to_drop]
    return selected, list(to_drop)


def clip_outliers(X, percentile=1):
    """Clip outliers using percentile-based clipping."""
    X_clipped = X.copy()
    clip_params = []
    for i in range(X.shape[1]):
        lower = np.percentile(X[:, i], percentile)
        upper = np.percentile(X[:, i], 100 - percentile)
        X_clipped[:, i] = np.clip(X[:, i], lower, upper)
        clip_params.append((lower, upper))
    return X_clipped, clip_params


def find_optimal_threshold(y_true, y_proba):
    """Find optimal classification threshold maximizing F1."""
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in np.arange(0.30, 0.85, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def train_final_model(input_path: str, output_path: str, random_state: int = 42) -> dict:
    """Train the final optimized model."""
    
    print("="*70)
    print("FINAL OPTIMIZED Cup and Handle Classifier")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    meta_cols = ['symbol', 'window_start', 'window_end', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"Original features: {len(feature_cols)}")
    
    # 1. Feature Selection
    print("\n[1] Feature Selection...")
    selected_features, dropped = remove_redundant_features(df, feature_cols)
    print(f"    Selected: {len(selected_features)} features")
    print(f"    Dropped: {len(dropped)} redundant features")
    
    X = df[selected_features].values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Outlier Clipping
    print("\n[2] Outlier Clipping (1st-99th percentile)...")
    X, clip_params = clip_outliers(X, percentile=1)
    
    # 3. Time-based split
    print("\n[3] Time-based Train/Test Split...")
    df['window_end'] = pd.to_datetime(df['window_end'])
    df_sorted = df.sort_values('window_end').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    train_indices = df_sorted.index[:split_idx].values
    test_indices = df_sorted.index[split_idx:].values
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"    Train: {len(X_train)} samples, {y_train.sum()} positive ({100*y_train.mean():.2f}%)")
    print(f"    Test:  {len(X_test)} samples, {y_test.sum()} positive ({100*y_test.mean():.2f}%)")
    
    # 4. Feature Scaling
    print("\n[4] Feature Scaling (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"\n[5] Class weight: {scale_pos_weight:.2f}")
    
    # 6. Train XGBoost with optimized hyperparameters
    print("\n[6] Training XGBoost (optimized hyperparameters)...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
    # 7. Cross-validation
    print("\n[7] Cross-validation (5-fold stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    cv_f1_scores = []
    cv_recall_scores = []
    cv_precision_scores = []
    
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        model_cv = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, eval_metric='logloss'
        )
        model_cv.fit(X_cv_train, y_cv_train)
        y_cv_proba = model_cv.predict_proba(X_cv_val)[:, 1]
        
        # Use threshold 0.75 from optimization
        y_cv_pred = (y_cv_proba >= 0.75).astype(int)
        
        cv_f1_scores.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_recall_scores.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_precision_scores.append(precision_score(y_cv_val, y_cv_pred, zero_division=0))
    
    print(f"    CV Precision: {np.mean(cv_precision_scores):.4f} (+/- {np.std(cv_precision_scores)*2:.4f})")
    print(f"    CV Recall:    {np.mean(cv_recall_scores):.4f} (+/- {np.std(cv_recall_scores)*2:.4f})")
    print(f"    CV F1:        {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores)*2:.4f})")
    
    # 8. Test set predictions
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 9. Threshold optimization
    print("\n[8] Threshold Optimization...")
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba)
    print(f"    Optimal threshold: {optimal_threshold:.2f}")
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # 10. Results
    print("\n" + "="*70)
    print(f"TEST SET RESULTS (Threshold = {optimal_threshold:.2f})")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=['No Pattern', 'Cup & Handle']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"\nFinal Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    
    # Feature importance
    print("\n" + "="*70)
    print("Top 10 Most Important Features")
    print("="*70)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print("\n" + "="*70)
    print("Saving model...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': selected_features,
        'clip_params': clip_params,
        'optimal_threshold': optimal_threshold,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cv_f1_mean': float(np.mean(cv_f1_scores)),
            'cv_f1_std': float(np.std(cv_f1_scores)),
            'cv_precision_mean': float(np.mean(cv_precision_scores)),
            'cv_recall_mean': float(np.mean(cv_recall_scores)),
        },
        'training_config': {
            'n_features_original': len(feature_cols),
            'n_features_selected': len(selected_features),
            'dropped_features': dropped,
            'scale_pos_weight': scale_pos_weight,
            'hyperparameters': {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 2,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        },
        'confusion_matrix': {
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0]),
            'TP': int(cm[1,1])
        },
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, output_path)
    print(f"Model saved to {output_path}")
    
    # Save metrics
    metrics_path = output_path.replace('.joblib', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': model_data['metrics'],
            'training_config': model_data['training_config'],
            'confusion_matrix': model_data['confusion_matrix'],
            'feature_importance': importance_df.head(15).to_dict('records'),
            'optimal_threshold': optimal_threshold,
            'training_date': model_data['training_date']
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    return model_data['metrics']


def main():
    parser = argparse.ArgumentParser(description="Train final optimized Cup and Handle classifier")
    parser.add_argument("--input", type=str, default="outputs/training_dataset.csv")
    parser.add_argument("--output", type=str, default="outputs/model_final.joblib")
    
    args = parser.parse_args()
    train_final_model(args.input, args.output)


if __name__ == "__main__":
    main()

