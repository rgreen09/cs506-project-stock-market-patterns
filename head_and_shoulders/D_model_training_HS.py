"""
Head & Shoulders Pattern - ML Modeling (Coarse Features Only)

- Reads combined_dataset_HS.csv in streaming fashion
- Strongly subsamples negatives to control class imbalance
- Uses only coarse, non-shape features to reduce label leakage:
    * vol_ratio   = close_std / |close_mean|
    * range_ratio = price_range / |close_mean|
    * abs_ret1    = |return_1|
    * abs_ret20   = |return_20|
    * ret20_sign  = sign(return_20)

- Trains:
    * Random Forest (regularized)
    * Logistic Regression (with class_weight="balanced")

- Saves:
    * best_model_HS/scaler_HS.pkl
    * best_model_HS/random_forest_HS.pkl
    * best_model_HS/logistic_regression_HS.pkl
    * best_model_HS/roc_curve_HS.png
    * best_model_HS/ML_log_HS.txt
"""

import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "combined_dataset_HS.csv"  # combined HS dataset you just created
MODELS_DIR = "best_model_HS"
LOG_FILE = os.path.join(MODELS_DIR, "ML_log_HS.txt")
ROC_PLOT_PATH = os.path.join(MODELS_DIR, "roc_curve_HS.png")

TARGET_COL = "label_hs"

# Streaming + sampling parameters
CHUNK_SIZE = 200_000
NEG_PER_POS = 10          # keep at most this many negatives per positive
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# LOGGER
# =============================================================================

class Logger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lines = []

    def log(self, msg: str = ""):
        self.lines.append(msg)
        print(msg)

    def header(self, title: str):
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\nLog saved to {self.path}")


# =============================================================================
# STREAMING LOAD + SAMPLING
# =============================================================================

def stream_and_sample(data_path: str, logger: Logger) -> pd.DataFrame:
    """
    Stream through the huge CSV and:
      - keep ALL positives
      - randomly sample at most NEG_PER_POS negatives per positive
    Returns a reasonably sized DataFrame for modeling.
    """
    logger.header("STREAMING CSV + SAMPLING")
    logger.log(f"Reading from {data_path} in chunks of {CHUNK_SIZE:,} rows...")

    pos_chunks = []
    neg_chunks = []

    total_rows = 0
    total_pos = 0
    total_neg = 0

    t0 = time.time()

    for chunk in pd.read_csv(data_path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        # ensure target exists
        if TARGET_COL not in chunk.columns:
            raise ValueError(f"Expected column '{TARGET_COL}' not found in data.")

        pos_chunk = chunk[chunk[TARGET_COL] == 1]
        neg_chunk = chunk[chunk[TARGET_COL] == 0]

        n_pos = len(pos_chunk)
        n_neg = len(neg_chunk)

        if n_pos > 0:
            pos_chunks.append(pos_chunk)

        # only sample some negatives, proportional to positives
        # we aim for NEG_PER_POS negatives per positive overall
        if n_pos > 0:
            # how many negatives can we keep from this chunk?
            # approximate target negatives so far:
            target_neg_total = NEG_PER_POS * (total_pos + n_pos)
            remaining_neg_allowance = max(0, target_neg_total - total_neg)

            if remaining_neg_allowance > 0 and n_neg > 0:
                sample_size = min(n_neg, remaining_neg_allowance)
                neg_sample = neg_chunk.sample(
                    n=sample_size, random_state=RANDOM_STATE
                )
                neg_chunks.append(neg_sample)
                total_neg += sample_size

        total_pos += n_pos

        if total_rows % (2_000_000) == 0:
            logger.log(
                f"  Processed {total_rows:,} rows so far "
                f"(Pos: {total_pos:,}, Neg kept: {total_neg:,})"
            )

    elapsed = time.time() - t0
    logger.log(f"\nFinished streaming in {elapsed:.1f} seconds.")
    logger.log(f"Total rows seen: {total_rows:,}")
    logger.log(f"Total positives: {total_pos:,}")
    logger.log(f"Total negatives kept: {total_neg:,}")

    if total_pos == 0:
        raise ValueError("No positive (label_hs=1) samples found in the dataset!")

    # Concatenate sampled data
    logger.log("\nConcatenating sampled data into a single DataFrame...")
    df_pos = pd.concat(pos_chunks, ignore_index=True)
    df_neg = pd.concat(neg_chunks, ignore_index=True) if neg_chunks else pd.DataFrame()

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.log(f"Sampled dataset size: {len(df):,} rows")
    vc = df[TARGET_COL].value_counts()
    for label, count in vc.items():
        pct = 100.0 * count / len(df)
        logger.log(f"  {TARGET_COL} = {label}: {count:,} rows ({pct:.2f}%)")

    return df


# =============================================================================
# FEATURE ENGINEERING (COARSE FEATURES ONLY)
# =============================================================================

def engineer_features(df: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    """
    Take the sampled dataframe with columns:
        symbol, start_ts, end_ts,
        close_last, close_mean, close_std,
        price_range, return_1, return_20, label_hs

    and create ONLY coarse, non-leaky features:
        vol_ratio   = close_std / |close_mean|
        range_ratio = price_range / |close_mean|
        abs_ret1    = |return_1|
        abs_ret20   = |return_20|
        ret20_sign  = sign(return_20)

    Then drops all raw price/return/timestamp columns.
    """
    logger.header("FEATURE ENGINEERING")

    logger.log(f"Initial columns: {len(df.columns)}")
    logger.log(f"Columns: {list(df.columns)}")

    # Drop metadata: symbol, timestamps (never used as features)
    meta_to_drop = [col for col in ["symbol", "start_ts", "end_ts"] if col in df.columns]
    if meta_to_drop:
        logger.log(f"\nDropping metadata columns: {meta_to_drop}")
        df = df.drop(columns=meta_to_drop)

    # Sanity check required numeric columns
    required = [
        "close_last",
        "close_mean",
        "close_std",
        "price_range",
        "return_1",
        "return_20",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not in dataframe.")

    logger.log("\nCreating COARSE, scale-free features (to reduce leakage)...")
    eps = 1e-8
    cm_abs = df["close_mean"].abs() + eps

    # Volatility relative to price level
    df["vol_ratio"] = df["close_std"] / cm_abs

    # Range relative to price level
    df["range_ratio"] = df["price_range"] / cm_abs

    # Coarse momentum features (absolute returns)
    df["abs_ret1"] = df["return_1"].abs()
    df["abs_ret20"] = df["return_20"].abs()

    # Direction of 20-step move only (very coarse)
    df["ret20_sign"] = np.sign(df["return_20"]).astype(np.float32)

    logger.log("  Added features: ['vol_ratio', 'range_ratio', 'abs_ret1', 'abs_ret20', 'ret20_sign']")

    # Drop the original scale-dependent columns to avoid leakage
    drop_raw = [
        "close_last",
        "close_mean",
        "close_std",
        "price_range",
        "return_1",
        "return_20",
    ]
    logger.log(f"\nDropping raw scale/return columns: {drop_raw}")
    df = df.drop(columns=drop_raw)

    # Handle any NaN/inf
    num_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(num_df).sum().sum()
    nan_count = num_df.isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        logger.log(f"\nHandling invalid values: {inf_count} inf, {nan_count} NaN")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    logger.log(f"\nFinal feature count: {len(feature_cols)}")
    logger.log(f"Final features: {feature_cols}")

    return df


# =============================================================================
# MODELS
# =============================================================================

def get_models():
    """
    Return a dict of model_name -> model, using
    relatively conservative hyperparameters.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,           # restrict depth → less overfitting
            min_samples_leaf=20,   # each leaf must have decent support
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate(model, X, y, prefix: str):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        f"{prefix}_acc": accuracy_score(y, y_pred),
        f"{prefix}_prec": precision_score(y, y_pred),
        f"{prefix}_rec": recall_score(y, y_pred),
        f"{prefix}_f1": f1_score(y, y_pred),
        f"{prefix}_auc": roc_auc_score(y, y_prob),
        f"{prefix}_y_prob": y_prob,  # keep probs for ROC plotting later
    }


# =============================================================================
# ROC PLOTTING
# =============================================================================

def plot_roc_curve(y_test, y_prob_best, model_name: str, out_path: str, logger: Logger):
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    auc_val = roc_auc_score(y_test, y_prob_best)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Head & Shoulders Detection (Coarse Features)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.log(f"\nROC curve saved to {out_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger = Logger(LOG_FILE)

    logger.log("=" * 70)
    logger.log("HEAD & SHOULDERS — MODEL TRAINING (COARSE FEATURES ONLY)")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 70)

    # 1) Streaming load + sampling
    df_sampled = stream_and_sample(DATA_PATH, logger)

    # 2) Feature engineering (drop leak-prone stuff, keep only coarse ratios)
    df_fe = engineer_features(df_sampled, logger)

    # 3) Prepare X, y
    feature_cols = [c for c in df_fe.columns if c != TARGET_COL]
    X = df_fe[feature_cols].values
    y = df_fe[TARGET_COL].values

    logger.header("TRAIN / TEST SPLIT")
    logger.log(f"Total samples (after sampling): {len(X):,}")
    logger.log(f"Features: {len(feature_cols)}")
    logger.log(f"Test size: {TEST_SIZE} ({int(TEST_SIZE * 100)}%)")

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
    logger.header("SCALING")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.log("Applied StandardScaler to all features.")

    # 5) Train models
    logger.header("MODEL TRAINING & EVAL")
    models = get_models()
    results = {}

    for name, model in models.items():
        logger.log(f"\n-- {name} ---")
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        logger.log(f"Training time: {train_time:.2f} seconds")

        # Train metrics
        train_metrics = evaluate(model, X_train_scaled, y_train, "train")
        # Test metrics
        test_metrics = evaluate(model, X_test_scaled, y_test, "test")

        # Log
        logger.log("Train Metrics:")
        logger.log(
            f"  Acc: {train_metrics['train_acc']:.4f}, "
            f"Prec: {train_metrics['train_prec']:.4f}, "
            f"Rec: {train_metrics['train_rec']:.4f}, "
            f"F1: {train_metrics['train_f1']:.4f}, "
            f"AUC: {train_metrics['train_auc']:.4f}"
        )
        logger.log("Test Metrics:")
        logger.log(
            f"  Acc: {test_metrics['test_acc']:.4f}, "
            f"Prec: {test_metrics['test_prec']:.4f}, "
            f"Rec: {test_metrics['test_rec']:.4f}, "
            f"F1: {test_metrics['test_f1']:.4f}, "
            f"AUC: {test_metrics['test_auc']:.4f}"
        )

        results[name] = {
            "model": model,
            "train": train_metrics,
            "test": test_metrics,
            "train_time": train_time,
        }

    # 6) Summary + choose best by test F1
    logger.header("MODEL COMPARISON SUMMARY (TEST SET)")
    logger.log(
        f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
        f"{'F1':>8} {'AUC':>8} {'Train(s)':>10}"
    )
    logger.log("-" * 70)

    best_name = None
    best_f1 = -1.0
    best_probs = None

    for name, res in results.items():
        tm = res["test"]
        row = (
            f"{name:<20} {tm['test_acc']:>8.4f} {tm['test_prec']:>8.4f} "
            f"{tm['test_rec']:>8.4f} {tm['test_f1']:>8.4f} {tm['test_auc']:>8.4f} "
            f"{res['train_time']:>10.2f}"
        )
        logger.log(row)

        if tm["test_f1"] > best_f1:
            best_f1 = tm["test_f1"]
            best_name = name
            best_probs = tm["test_y_prob"]

    logger.log("")
    logger.log(f"Best model by Test F1: {best_name} (F1={best_f1:.4f})")

    # 7) ROC curve for best model
    if best_probs is not None:
        plot_roc_curve(y_test, best_probs, best_name, ROC_PLOT_PATH, logger)

    # 8) Save models + scaler
    logger.header("SAVING MODELS & SCALER")
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    scaler_path = os.path.join(MODELS_DIR, "scaler_HS.pkl")
    joblib.dump(scaler, scaler_path)
    logger.log(f"Saved scaler to {scaler_path}")

    for name, res in results.items():
        fname = (
            "random_forest_HS.pkl"
            if name == "Random Forest"
            else "logistic_regression_HS.pkl"
        )
        fpath = os.path.join(MODELS_DIR, fname)
        joblib.dump(res["model"], fpath)
        logger.log(f"Saved {name} model to {fpath}")

    logger.header("TRAINING COMPLETE")
    logger.save()


if __name__ == "__main__":
    main()
