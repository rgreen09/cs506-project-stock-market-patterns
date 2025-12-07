"""
Machine Learning Pipeline for Triangle Pattern Detection

Optimized for:
- Huge datasets (tens of millions of rows)
- Extreme class imbalance (label_triangle ~ very small %)

Steps:
1. Stream combined_dataset_triangle.csv in chunks.
2. Keep ALL positive (label_triangle=1) windows.
3. Randomly keep a small fraction of label_triangle=0 windows.
4. Do feature engineering (drop metadata, optional normalization).
5. Train multiple ML models.
6. Save models, scaler, ROC curve, and a detailed log.

This mirrors the Head & Shoulders training pipeline, but for TRIANGLES.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

import joblib
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "combined_dataset_triangle.csv"   # combined triangle dataset
MODELS_DIR = "best_model_triangle"
LOG_FILE = os.path.join(MODELS_DIR, "ML_log_triangle.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Chunked loading settings (tune if needed)
CHUNK_SIZE = 200_000          # rows per chunk when streaming CSV
NEGATIVE_SAMPLE_FRAC = 0.001  # keep 0.1% of negatives; adjust as needed

TARGET_COL = "label_triangle"

# Metadata / non-feature columns to drop (we ONLY use numeric features)
METADATA_COLS = [
    "symbol",
    "timestamp",
    "start_ts",
    "end_ts",
    "start_timestamp",
    "end_timestamp",
    "start_date",
    "end_date",
    "date",
]

# These are kept for structural similarity with the HS script.
# For your triangle dataset, most of these columns probably DON'T exist,
# and the code will simply skip normalization if 'close_mean' is missing.
COLS_TO_NORMALIZE = [
    "close_std",
    "slope_entire_window",
    "slope_last_30",
    "rolling_std_20",
    "rolling_std_60",
    "true_range_mean_20",
    "peak1_sharpness",
    "peak2_sharpness",
    "sma_50",
]

COLS_TO_DROP_AFTER_NORM = [
    "close_last",
    "close_mean",
    "close_std",
    "price_range_abs",
    "slope_entire_window",
    "rolling_std_20",
    "rolling_std_60",
    "true_range_mean_20",
    "peak1_sharpness",
    "peak2_sharpness",
    "sma_20",
    "sma_50",
]

# =============================================================================
# LOGGER
# =============================================================================

class MLLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lines = []

    def log(self, msg: str = ""):
        self.lines.append(msg)
        print(msg)

    def section(self, title: str):
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    def save(self):
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n[LOG] Saved to {self.log_file}")


# =============================================================================
# DATA LOADING (CHUNKED) + SAMPLING
# =============================================================================

def load_sampled_data(data_path: str, logger: MLLogger) -> pd.DataFrame:
    """
    Stream the huge CSV in chunks.
    - Keep ALL label_triangle=1 rows.
    - Randomly keep a small fraction of label_triangle=0 rows.

    Returns a manageable DataFrame for ML.
    """
    logger.section("STREAMING CSV + SAMPLING (TRIANGLES)")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    total_rows = 0
    total_pos = 0
    total_neg = 0

    pos_chunks = []
    neg_chunks = []

    t0 = time.time()
    logger.log(f"Reading from {data_path} in chunks of {CHUNK_SIZE:,} rows...")
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=CHUNK_SIZE)):
        total_rows += len(chunk)

        if TARGET_COL not in chunk.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

        # Split into positive & negative
        pos = chunk[chunk[TARGET_COL] == 1]
        neg = chunk[chunk[TARGET_COL] == 0]

        total_pos += len(pos)
        total_neg += len(neg)

        if len(pos) > 0:
            pos_chunks.append(pos)

        # Sample a fraction of negatives
        if len(neg) > 0 and NEGATIVE_SAMPLE_FRAC > 0.0:
            frac = min(1.0, NEGATIVE_SAMPLE_FRAC)
            neg_sample = neg.sample(
                frac=frac,
                random_state=RANDOM_STATE + i,
                replace=False,
            )
            neg_chunks.append(neg_sample)

        if (i + 1) % 10 == 0:
            logger.log(
                f"  Processed {total_rows:,} rows so far "
                f"(Pos: {total_pos:,}, Neg: {total_neg:,})"
            )

    t1 = time.time()
    logger.log("")
    logger.log(f"Finished streaming in {t1 - t0:.1f} seconds.")
    logger.log(f"Total rows seen: {total_rows:,}")
    logger.log(f"Total positives (label_triangle=1): {total_pos:,}")
    logger.log(f"Total negatives (label_triangle=0): {total_neg:,}")

    if not pos_chunks and not neg_chunks:
        raise RuntimeError("No data collected. Check label_triangle values and CSV format.")

    # Concatenate collected samples
    logger.log("\nConcatenating sampled data into a single DataFrame...")
    df_pos = pd.concat(pos_chunks, ignore_index=True) if pos_chunks else pd.DataFrame()
    df_neg = pd.concat(neg_chunks, ignore_index=True) if neg_chunks else pd.DataFrame()

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    logger.log(f"Sampled dataset size: {len(df):,} rows")

    # Show sampled class distribution
    class_counts = df[TARGET_COL].value_counts()
    for label in sorted(class_counts.index):
        pct = 100.0 * class_counts[label] / len(df)
        logger.log(f"  {TARGET_COL} = {label}: {class_counts[label]:,} rows ({pct:.2f}%)")

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame, logger: MLLogger) -> pd.DataFrame:
    """
    Drop metadata columns, create price-normalized features (if possible),
    and clean NaNs/infs.

    For your triangle builder output, you likely have columns:
        ['timestamp', 'symbol', 'close', 'label_triangle']

    In that case:
      - metadata columns will drop 'timestamp', 'symbol'
      - 'close' will remain as the primary numeric feature.
      - If you later add more engineered features (close_mean, etc.),
        the normalization section will automatically kick in.
    """
    logger.section("FEATURE ENGINEERING (TRIANGLES)")

    logger.log(f"Initial columns: {len(df.columns)}")
    logger.log(f"Columns: {list(df.columns)}")

    # Drop metadata columns that actually exist
    cols_to_drop_meta = [c for c in METADATA_COLS if c in df.columns]
    if cols_to_drop_meta:
        logger.log(f"\nDropping metadata columns: {cols_to_drop_meta}")
        df = df.drop(columns=cols_to_drop_meta)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not in DataFrame after drops.")

    # Create normalized features using close_mean (if present)
    if "close_mean" not in df.columns:
        logger.log("\n[WARN] 'close_mean' not found; skipping normalization step.")
    else:
        logger.log("\nCreating price-normalized features (divide by close_mean)...")
        close_mean = df["close_mean"].replace(0, np.nan)

        for col in COLS_TO_NORMALIZE:
            if col not in df.columns:
                continue
            if col == "sma_50":
                new_col = "sma_50_ratio"
            else:
                new_col = f"{col}_pct"

            df[new_col] = df[col] / close_mean
            logger.log(f"  {col} -> {new_col}")

        # Drop absolute magnitude columns if they exist
        cols_to_drop_abs = [c for c in COLS_TO_DROP_AFTER_NORM if c in df.columns]
        if cols_to_drop_abs:
            logger.log(f"\nDropping absolute-scale columns: {cols_to_drop_abs}")
            df = df.drop(columns=cols_to_drop_abs)

    # Replace inf / -inf / NaN with 0
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    nan_count = df.isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        logger.log(f"\nCleaning invalid values: {inf_count} inf, {nan_count} NaN")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

    # Final feature list
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    logger.log(f"\nFinal feature count: {len(feature_cols)}")
    logger.log(f"Final features: {feature_cols}")

    return df


# =============================================================================
# MODELS
# =============================================================================

def get_models():
    """Return a dict of models to train (same structure as HS pipeline)."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_model(model, X, y, prefix: str) -> dict:
    """Compute metrics for a given dataset."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = y_pred.astype(float)

    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, y_pred),
        f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y, y_prob),
    }
    return metrics, y_prob


def train_and_evaluate(X_train, X_test, y_train, y_test, logger: MLLogger):
    logger.section("MODEL TRAINING & EVALUATION (TRIANGLES)")

    models = get_models()
    results = {}

    for name, model in models.items():
        logger.log(f"\n--- {name} ---")

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        logger.log(f"Training time: {train_time:.2f}s")

        # Train set metrics
        train_metrics, _ = evaluate_model(model, X_train, y_train, "train")

        # Test set metrics
        test_metrics, y_test_prob = evaluate_model(model, X_test, y_test, "test")

        # Log
        logger.log("Train metrics:")
        logger.log(
            f"  Acc={train_metrics['train_accuracy']:.4f} "
            f"Prec={train_metrics['train_precision']:.4f} "
            f"Rec={train_metrics['train_recall']:.4f} "
            f"F1={train_metrics['train_f1']:.4f} "
            f"AUC={train_metrics['train_roc_auc']:.4f}"
        )
        logger.log("Test metrics:")
        logger.log(
            f"  Acc={test_metrics['test_accuracy']:.4f} "
            f"Prec={test_metrics['test_precision']:.4f} "
            f"Rec={test_metrics['test_recall']:.4f} "
            f"F1={test_metrics['test_f1']:.4f} "
            f"AUC={test_metrics['test_roc_auc']:.4f}"
        )

        results[name] = {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_time": train_time,
            "y_test_prob": y_test_prob,
        }

    return results


# =============================================================================
# SAVE MODELS + ROC CURVE
# =============================================================================

def save_all(results, scaler, X_test, y_test, logger: MLLogger):
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_triangle.pkl")
    joblib.dump(scaler, scaler_path)
    logger.log(f"\nSaved scaler to {scaler_path}")

    # Save models
    for name, info in results.items():
        filename = name.lower().replace(" ", "_") + "_triangle.pkl"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(info["model"], path)
        logger.log(f"Saved {name} model to {path}")

    # Plot ROC curve for the best model (by test ROC-AUC)
    best_name = None
    best_auc = -1.0
    for name, info in results.items():
        auc = info["test_metrics"]["test_roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_name = name

    if best_name is not None:
        logger.log(f"\nBest model by ROC-AUC: {best_name} (AUC={best_auc:.4f})")
        best_info = results[best_name]
        y_prob = best_info["y_test_prob"]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"{best_name} (AUC={best_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Triangle Pattern Detection")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_path = os.path.join(MODELS_DIR, "roc_curve_triangle.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.log(f"Saved ROC curve to {roc_path}")


def log_summary(results, logger: MLLogger):
    logger.section("SUMMARY (TEST SET) — TRIANGLES")

    header = f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Train s':>10}"
    logger.log(header)
    logger.log("-" * len(header))

    for name, info in results.items():
        m = info["test_metrics"]
        row = (
            f"{name:<20} "
            f"{m['test_accuracy']:>8.4f} "
            f"{m['test_precision']:>8.4f} "
            f"{m['test_recall']:>8.4f} "
            f"{m['test_f1']:>8.4f} "
            f"{m['test_roc_auc']:>8.4f} "
            f"{info['train_time']:>10.2f}"
        )
        logger.log(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = MLLogger(LOG_FILE)
    logger.log("=" * 70)
    logger.log("TRIANGLE PATTERN — MODEL TRAINING (OPTIMIZED)")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 70)

    # 1) Load sampled data
    df_sample = load_sampled_data(DATA_PATH, logger)

    # 2) Feature engineering
    df_sample = engineer_features(df_sample, logger)

    # 3) Train/test split
    logger.section("TRAIN / TEST SPLIT")
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]
    X = df_sample[feature_cols].values
    y = df_sample[TARGET_COL].values

    logger.log(f"Total samples (after sampling): {len(X):,}")
    logger.log(f"Features: {len(feature_cols)}")
    logger.log(f"Test size: {TEST_SIZE} ({int(TEST_SIZE*100)}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.log(f"Train set: {len(X_train):,} samples")
    logger.log(f"Test set:  {len(X_test):,} samples")

    # 4) Scaling
    logger.section("SCALING")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.log("Scaling done (StandardScaler).")

    # 5) Train & evaluate models
    results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, logger)

    # 6) Save models and ROC curve
    save_all(results, scaler, X_test_scaled, y_test, logger)

    # 7) Summary
    log_summary(results, logger)

    logger.log("")
    logger.log("=" * 70)
    logger.log("TRAINING COMPLETE — TRIANGLES")
    logger.log("=" * 70)
    logger.save()


if __name__ == "__main__":
    main()
