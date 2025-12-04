"""
Machine Learning Pipeline for Double Top Pattern Detection
Round One Modeling

This script trains and evaluates multiple ML models for detecting double-top patterns
in stock price data.
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

# Gradient boosting imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# For saving models
import joblib


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = "data/combined_double_top_15m_windows.csv"
MODELS_DIR = "best_model"
LOG_FILE = "best_model/ML_log.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns to drop (metadata)
METADATA_COLS = ['symbol', 'start_timestamp', 'end_timestamp']

# Columns that need price normalization (divide by close_mean)
COLS_TO_NORMALIZE = [
    'close_std',
    'slope_entire_window',
    'slope_last_30',
    'rolling_std_20',
    'rolling_std_60',
    'true_range_mean_20',
    'peak1_sharpness',
    'peak2_sharpness',
    'sma_50'
]

# Columns to drop after normalization (absolute values)
COLS_TO_DROP_AFTER_NORM = [
    'close_last',
    'close_mean',
    'close_std',
    'price_range_abs',
    'slope_entire_window',
    'slope_last_30',
    'rolling_std_20',
    'rolling_std_60',
    'true_range_mean_20',
    'peak1_sharpness',
    'peak2_sharpness',
    'sma_20',
    'sma_50'
]

TARGET_COL = 'label_double_top'


# =============================================================================
# Logging Utility
# =============================================================================

class MLLogger:
    """Logger for ML experiment results."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.log_lines = []
        
    def log(self, message: str = ""):
        """Add a line to the log."""
        self.log_lines.append(message)
        print(message)
        
    def log_section(self, title: str):
        """Add a section header."""
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)
        
    def save(self):
        """Write all logs to file."""
        # Ensure directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_lines))
            print(f"\nLog saved to {self.log_file}")
        except Exception as e:
            print(f"\nError saving log file: {e}")
            raise


# =============================================================================
# Data Loading and Feature Engineering
# =============================================================================

def load_and_engineer_features(data_path: str, logger: MLLogger) -> pd.DataFrame:
    """
    Load dataset and create price-normalized features.
    
    Args:
        data_path: Path to the CSV file
        logger: MLLogger instance
        
    Returns:
        Processed DataFrame ready for ML
    """
    logger.log_section("DATA LOADING AND FEATURE ENGINEERING")
    
    # Load data
    logger.log(f"Loading data from {data_path}...")
    start_time = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    logger.log(f"Loaded {len(df):,} rows in {load_time:.2f} seconds")
    logger.log(f"Original columns: {len(df.columns)}")
    
    # Show class distribution
    class_dist = df[TARGET_COL].value_counts()
    logger.log(f"\nTarget distribution:")
    logger.log(f"  Class 0 (No Double Top): {class_dist[0]:,} ({100*class_dist[0]/len(df):.1f}%)")
    logger.log(f"  Class 1 (Double Top):    {class_dist[1]:,} ({100*class_dist[1]/len(df):.1f}%)")
    
    # Drop metadata columns
    logger.log(f"\nDropping metadata columns: {METADATA_COLS}")
    df = df.drop(columns=METADATA_COLS)
    
    # Create price-normalized features
    logger.log("\nCreating price-normalized features (dividing by close_mean):")
    close_mean = df['close_mean']
    
    # Handle potential division by zero (shouldn't happen with stock prices, but be safe)
    close_mean = close_mean.replace(0, np.nan)
    
    normalized_features = {}
    for col in COLS_TO_NORMALIZE:
        if col in df.columns:
            if col == 'sma_50':
                new_col = 'sma_50_ratio'
            else:
                new_col = f'{col}_pct'
            normalized_features[new_col] = df[col] / close_mean
            logger.log(f"  {col} -> {new_col}")
    
    # Add normalized features to dataframe
    for col_name, col_data in normalized_features.items():
        df[col_name] = col_data
    
    # Drop absolute value columns
    logger.log(f"\nDropping absolute value columns: {len(COLS_TO_DROP_AFTER_NORM)} columns")
    cols_to_drop = [c for c in COLS_TO_DROP_AFTER_NORM if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Handle any inf/nan values from division
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    nan_count = df.isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        logger.log(f"\nHandling invalid values: {inf_count} inf, {nan_count} NaN")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
    
    # Final feature list
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    logger.log(f"\nFinal feature count: {len(feature_cols)}")
    logger.log(f"Features: {feature_cols}")
    
    return df


# =============================================================================
# Model Training and Evaluation
# =============================================================================

def get_models(n_pos: int, n_neg: int) -> dict:
    """
    Get dictionary of models to train.
    
    Args:
        n_pos: Number of positive samples in training set
        n_neg: Number of negative samples in training set
        
    Returns:
        Dictionary of model name -> model instance
    """
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )
    }
    
    return models


def evaluate_model(model, X: np.ndarray, y: np.ndarray, set_name: str = "") -> dict:
    """
    Evaluate a trained model on data.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        set_name: Name of the dataset ('train' or 'test')
        
    Returns:
        Dictionary of metric name -> value
    """
    start_time = time.time()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    pred_time = time.time() - start_time
    
    metrics = {
        f'{set_name}_accuracy': accuracy_score(y, y_pred),
        f'{set_name}_precision': precision_score(y, y_pred),
        f'{set_name}_recall': recall_score(y, y_pred),
        f'{set_name}_f1': f1_score(y, y_pred),
        f'{set_name}_roc_auc': roc_auc_score(y, y_prob),
        f'{set_name}_prediction_time': pred_time
    }
    
    return metrics


def train_and_evaluate_models(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    logger: MLLogger
) -> dict:
    """
    Train and evaluate all models.
    
    Args:
        X_train, X_test: Train/test features
        y_train, y_test: Train/test labels
        logger: MLLogger instance
        
    Returns:
        Dictionary of model name -> (trained model, metrics dict)
    """
    logger.log_section("MODEL TRAINING AND EVALUATION")
    
    # Get class counts for weighting
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    logger.log(f"Training set class distribution: {n_neg:,} negative, {n_pos:,} positive")
    
    models = get_models(n_pos, n_neg)
    results = {}
    
    for name, model in models.items():
        logger.log(f"\n--- {name} ---")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.log(f"Training time: {train_time:.2f} seconds")
        
        # Evaluate on training set
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test, 'test')
        
        # Combine metrics
        metrics = {**train_metrics, **test_metrics, 'training_time': train_time}
        
        # Log training metrics
        logger.log(f"\nTraining Set Metrics:")
        logger.log(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
        logger.log(f"  Precision: {metrics['train_precision']:.4f}")
        logger.log(f"  Recall:    {metrics['train_recall']:.4f}")
        logger.log(f"  F1 Score:  {metrics['train_f1']:.4f}")
        logger.log(f"  ROC-AUC:   {metrics['train_roc_auc']:.4f}")
        
        # Log test metrics
        logger.log(f"\nTest Set Metrics:")
        logger.log(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
        logger.log(f"  Precision: {metrics['test_precision']:.4f}")
        logger.log(f"  Recall:    {metrics['test_recall']:.4f}")
        logger.log(f"  F1 Score:  {metrics['test_f1']:.4f}")
        logger.log(f"  ROC-AUC:   {metrics['test_roc_auc']:.4f}")
        
        results[name] = (model, metrics)
    
    return results


# =============================================================================
# Model Saving
# =============================================================================

def save_models(results: dict, scaler: StandardScaler, models_dir: str, logger: MLLogger):
    """
    Save all trained models and scaler to disk.
    
    Args:
        results: Dictionary of model name -> (model, metrics)
        scaler: Fitted StandardScaler
        models_dir: Directory to save models
        logger: MLLogger instance
    """
    logger.log_section("SAVING MODELS")
    
    # Ensure models directory exists
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.log(f"Saved scaler to {scaler_path}")
    
    # Save each model
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'LightGBM': 'lightgbm_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Neural Network': 'neural_network_model.pkl'
    }
    
    for name, (model, _) in results.items():
        filename = model_files.get(name, f"{name.lower().replace(' ', '_')}_model.pkl")
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model, filepath)
        logger.log(f"Saved {name} to {filepath}")


# =============================================================================
# Summary and Comparison
# =============================================================================

def log_summary(results: dict, logger: MLLogger):
    """
    Log summary comparison table of all models.
    
    Args:
        results: Dictionary of model name -> (model, metrics)
        logger: MLLogger instance
    """
    logger.log_section("MODEL COMPARISON SUMMARY - TRAINING SET")
    
    # Training set comparison table
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    logger.log(header)
    logger.log("-" * len(header))
    
    for name, (_, metrics) in results.items():
        row = f"{name:<25} {metrics['train_accuracy']:>10.4f} {metrics['train_precision']:>10.4f} {metrics['train_recall']:>10.4f} {metrics['train_f1']:>10.4f} {metrics['train_roc_auc']:>10.4f}"
        logger.log(row)
    
    logger.log("")
    logger.log_section("MODEL COMPARISON SUMMARY - TEST SET")
    
    # Test set comparison table
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'Train(s)':>10}"
    logger.log(header)
    logger.log("-" * len(header))
    
    best_f1 = 0
    best_model = None
    
    for name, (_, metrics) in results.items():
        row = f"{name:<25} {metrics['test_accuracy']:>10.4f} {metrics['test_precision']:>10.4f} {metrics['test_recall']:>10.4f} {metrics['test_f1']:>10.4f} {metrics['test_roc_auc']:>10.4f} {metrics['training_time']:>10.2f}"
        logger.log(row)
        
        if metrics['test_f1'] > best_f1:
            best_f1 = metrics['test_f1']
            best_model = name
    
    logger.log("")
    logger.log(f"Best model by Test F1 score: {best_model} (Test F1 = {best_f1:.4f})")
    
    # Log overfitting analysis
    logger.log("")
    logger.log("Overfitting Analysis (Train F1 - Test F1):")
    for name, (_, metrics) in results.items():
        diff = metrics['train_f1'] - metrics['test_f1']
        logger.log(f"  {name:<25} {diff:>10.4f}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Main ML pipeline execution."""
    
    # Initialize logger
    logger = MLLogger(LOG_FILE)
    
    # Log header
    logger.log("=" * 70)
    logger.log("ROUND ONE MODELING - Double Top Pattern Detection")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 70)
    
    # Step 1: Load and engineer features
    df = load_and_engineer_features(DATA_PATH, logger)
    
    # Step 2: Prepare features and target
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    
    logger.log_section("TRAIN/TEST SPLIT")
    logger.log(f"Test size: {TEST_SIZE} ({int(TEST_SIZE*100)}%)")
    logger.log(f"Random state: {RANDOM_STATE}")
    logger.log("Stratified split: Yes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.log(f"\nTraining set: {len(X_train):,} samples")
    logger.log(f"Test set: {len(X_test):,} samples")
    
    # Step 3: Feature scaling
    logger.log_section("FEATURE SCALING")
    logger.log("Using StandardScaler (fit on training data only)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.log(f"Scaled {len(feature_cols)} features")
    logger.log(f"Training data mean (after scaling): {X_train_scaled.mean():.6f}")
    logger.log(f"Training data std (after scaling): {X_train_scaled.std():.6f}")
    
    # Step 4 & 5: Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, logger)
    
    # Step 6: Save models
    save_models(results, scaler, MODELS_DIR, logger)
    
    # Step 7: Log summary
    log_summary(results, logger)
    
    # Save log file
    logger.log("")
    logger.log("=" * 70)
    logger.log("ROUND ONE MODELING COMPLETE")
    logger.log("=" * 70)
    logger.save()
    
    return results


if __name__ == "__main__":
    main()

