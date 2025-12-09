"""
Flags and Pennants Pattern ML Model Training
--
Trains models to detect flag and pennant continuation patterns using
engineered features from sliding windows.

Handles class imbalance using:
- SMOTE (Synthetic Minority Oversampling)
- Class weights
- Stratified sampling

Models trained:
- Random Forest
- XGBoost

Usage:
    python Train_flags_pennants_model.py --input AAPL_flags_pennants_15m_windows.csv
    python Train_flags_pennants_model.py --input AAPL_flags_pennants_15m_windows.csv --use-smote
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not installed. Install with 'pip install imbalanced-learn' for SMOTE support.")

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print(f"Warning: XGBoost not available ({str(e)[:100]}). Skipping XGBoost model.")


def load_and_prepare_data(input_path, test_size=0.2, random_state=42):
    """
    Load dataset and prepare features and labels.

    Args:
        input_path: Path to CSV file
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Dataset shape: {df.shape}")

    # Separate features and labels
    exclude_cols = ['symbol', 'start_timestamp', 'end_timestamp', 'label_flags_pennants']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['label_flags_pennants'].values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Total samples: {len(X):,}")
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = 100 * count / len(y)
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")

    # Stratified train/test split
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def apply_smote(X_train, y_train, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE to balance the training set.

    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    if not HAS_SMOTE:
        print("SMOTE not available. Skipping resampling.")
        return X_train, y_train

    print("\nApplying SMOTE oversampling...")
    print(f"Before SMOTE - Class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count:,}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Class distribution:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count:,}")

    return X_resampled, y_resampled


def train_random_forest(X_train, y_train, use_class_weight=True, random_state=42):
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        use_class_weight: Whether to use class weights
        random_state: Random seed

    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)

    class_weight = 'balanced' if use_class_weight else None

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    print(f"\nTraining with {len(X_train):,} samples...")
    model.fit(X_train, y_train)

    print("Training complete!")
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """
    Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained model
    """
    if not HAS_XGB:
        print("\nXGBoost not available. Skipping XGBoost training.")
        return None

    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)

    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )

    print(f"\nTraining with {len(X_train):,} samples...")
    model.fit(X_train, y_train, verbose=False)

    print("Training complete!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for display

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Pattern', 'Flag/Pennant']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0, 0]:,}")
    print(f"False Positives: {cm[0, 1]:,}")
    print(f"False Negatives: {cm[1, 0]:,}")
    print(f"True Positives:  {cm[1, 1]:,}")

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Average Precision
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")

    # Precision at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Find threshold for 90% precision
    idx_90 = np.where(precision >= 0.90)[0]
    if len(idx_90) > 0:
        threshold_90 = thresholds[idx_90[0]]
        recall_90 = recall[idx_90[0]]
        print(f"\nAt 90% precision: threshold={threshold_90:.4f}, recall={recall_90:.4f}")

    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot feature importance.

    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Model name for title
        top_n: Number of top features to show
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nFeature importance plot saved: {filename}")
        plt.close()


def plot_roc_curve(y_test, y_pred_proba, model_name):
    """
    Plot ROC curve.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name: Model name for title
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved: {filename}")
    plt.close()


def plot_precision_recall_curve(y_test, y_pred_proba, model_name):
    """
    Plot precision-recall curve.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name: Model name for title
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Precision-Recall curve saved: {filename}")
    plt.close()


def save_model(model, scaler, feature_names, model_name):
    """
    Save trained model and associated artifacts.

    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        model_name: Name for saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    scaler_filename = f"scaler_{timestamp}.pkl"
    features_filename = f"features_{timestamp}.txt"

    # Save model
    joblib.dump(model, model_filename)
    print(f"\nModel saved: {model_filename}")

    # Save scaler
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved: {scaler_filename}")

    # Save feature names
    with open(features_filename, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    print(f"Features saved: {features_filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ML models for flags and pennants pattern detection'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='AAPL_flags_pennants_15m_windows.csv',
        help='Input CSV file with labeled data'
    )
    parser.add_argument(
        '--use-smote',
        action='store_true',
        help='Use SMOTE for oversampling minority class'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--no-xgboost',
        action='store_true',
        help='Skip XGBoost training'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Flags and Pennants Pattern ML Training")
    print("="*60)

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data(
        args.input,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Apply SMOTE if requested
    if args.use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=args.random_state)

    # Train Random Forest
    rf_model = train_random_forest(
        X_train, y_train,
        use_class_weight=not args.use_smote,
        random_state=args.random_state
    )

    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_feature_importance(rf_model, feature_names, "Random Forest")
    plot_roc_curve(y_test, rf_metrics['y_pred_proba'], "Random Forest")
    plot_precision_recall_curve(y_test, rf_metrics['y_pred_proba'], "Random Forest")
    save_model(rf_model, scaler, feature_names, "Random Forest")

    # Train XGBoost
    if not args.no_xgboost and HAS_XGB:
        xgb_model = train_xgboost(X_train, y_train, random_state=args.random_state)

        if xgb_model is not None:
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            plot_feature_importance(xgb_model, feature_names, "XGBoost")
            plot_roc_curve(y_test, xgb_metrics['y_pred_proba'], "XGBoost")
            plot_precision_recall_curve(y_test, xgb_metrics['y_pred_proba'], "XGBoost")
            save_model(xgb_model, scaler, feature_names, "XGBoost")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nAll models, plots, and artifacts have been saved.")


if __name__ == "__main__":
    main()
