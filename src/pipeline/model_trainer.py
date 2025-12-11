import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config.settings import load_config
from src.utils.logging import MLLogger

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def _build_models(model_names: list, pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for name in model_names:
        if name == "random_forest":
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
        elif name == "xgboost" and XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "lightgbm" and LGBMClassifier is not None:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "logistic_regression":
            models["Logistic Regression"] = LogisticRegression(
                max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42
            )
        elif name == "neural_network":
            models["Neural Network"] = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
    return models


class ModelTrainer:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path) if config_path else load_config()

    def train(self, pattern: str, dataset_path: str | None = None) -> str:
        cfg = self.config
        pattern_cfg = cfg["patterns"][pattern]
        data_cfg = cfg["data"]
        training_cfg = pattern_cfg.get("training", {})

        models_dir = Path(data_cfg["models_dir"]) / pattern
        models_dir.mkdir(parents=True, exist_ok=True)

        if dataset_path is None:
            dataset_path = os.path.join(data_cfg["output_dir"], f"combined_{pattern}_windows.csv")

        df = pd.read_csv(dataset_path)
        label_col = pattern_cfg["label_column"]
        drop_cols = {"symbol", "start_timestamp", "end_timestamp", label_col}
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].values
        y = df[label_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=training_cfg.get("test_size", 0.2),
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        pos_weight = neg / pos if pos > 0 else 1.0

        model_names = training_cfg.get("models", ["random_forest"])
        models = _build_models(model_names, pos_weight)

        logger = MLLogger(str(models_dir / "training_log.txt"))
        logger.section(f"MODEL TRAINING - {pattern}")
        logger.log(f"Samples: train={len(X_train)}, test={len(X_test)}")
        logger.log(f"Features: {len(feature_cols)}")

        best_model_name = None
        best_f1 = -1.0
        for name, model in models.items():
            logger.section(name)
            model.fit(X_train_scaled, y_train)
            metrics = self._evaluate(model, X_test_scaled, y_test)
            logger.log(
                f"Accuracy={metrics['accuracy']:.4f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = name
            model_path = models_dir / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
            logger.log(f"Saved {name} to {model_path}")

        scaler_path = models_dir / "scaler.pkl"
        features_path = models_dir / "features.txt"
        joblib.dump(scaler, scaler_path)
        with open(features_path, "w", encoding="utf-8") as f:
            for col in feature_cols:
                f.write(f"{col}\n")

        logger.log(f"Best model: {best_model_name} (F1={best_f1:.4f})")
        logger.save()
        return str(models_dir)

    def _evaluate(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_prob_full = model.predict_proba(X)
            if y_prob_full.shape[1] > 1:
                y_prob = y_prob_full[:, 1]
                try:
                    roc_auc = roc_auc_score(y, y_prob)
                except ValueError:
                    roc_auc = 0.0
            else:
                # Model trained on a single class; treat AUC as undefined/zero
                roc_auc = 0.0
        else:
            roc_auc = 0.0
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc,
        }

