# Cup and Handle Pattern Detection & ML Classification

This module implements algorithmic detection and ML-based classification of the **Cup and Handle** chart pattern.

## Overview

The Cup and Handle is a bullish continuation pattern consisting of:
1. **Cup**: A rounded bottom formation (U-shape) with similar price levels at both edges
2. **Handle**: A smaller consolidation/pullback after the cup
3. **Breakout**: Price breaking above the resistance level formed by the cup's rim

## Project Structure

```
patterns/cup-and-handle/
├── aggregate_to_daily.py    # Convert tick data to daily OHLCV
├── detect_patterns.py       # Run pattern detection
├── build_dataset.py         # Create training dataset with features
├── train_model_final.py     # Train optimized XGBoost classifier
├── cup_and_handle/          # Core detection module
│   ├── detector.py          # CupAndHandleDetector class
│   ├── utils.py             # Helper functions
│   └── visualize.py         # Pattern visualization
└── outputs/
    ├── daily_data.csv       # Aggregated daily OHLCV
    ├── detected_patterns.csv # Detected patterns
    ├── training_dataset.csv # Features + labels
    ├── model_final.joblib   # Trained model
    └── model_final_metrics.json
```

## Pipeline

### 1. Data Aggregation
```bash
python aggregate_to_daily.py --input ../../combined_dataset.csv --output outputs/daily_data.csv
```

### 2. Pattern Detection
```bash
python detect_patterns.py --input outputs/daily_data.csv --output outputs/detected_patterns.csv
```

### 3. Feature Engineering
```bash
python build_dataset.py --daily outputs/daily_data.csv --patterns outputs/detected_patterns.csv --output outputs/training_dataset.csv
```

### 4. Model Training
```bash
python train_model_final.py --input outputs/training_dataset.csv --output outputs/model_final.joblib
```

## Model Results (Final Optimized)

| Metric | Value |
|--------|-------|
| **Precision** | **0.81** |
| **Recall** | **0.70** |
| **F1 Score** | **0.75** |
| **ROC-AUC** | **0.94** |
| **PR-AUC** | **0.66** |
| CV F1 (5-fold) | 0.98 |

### Confusion Matrix
```
              Predicted
            No      Yes
Actual No  1602     11   (99% specificity)
Actual Yes   20     47   (70% sensitivity)
```

### Model Evolution
| Version | F1 | Recall | AUC | Changes |
|---------|-----|--------|-----|---------|
| v1 (baseline) | 0.28 | 0.17 | 0.78 | Standard XGBoost |
| v2 (SMOTE) | 0.67 | 0.63 | 0.93 | + SMOTE, feature selection |
| **v3 (final)** | **0.75** | **0.70** | **0.94** | - SMOTE, + scale_pos_weight, tuned params |

### Key Optimizations Applied
1. **Feature Selection**: 30 → 20 features (removed 10 redundant via correlation + mutual information)
2. **Outlier Clipping**: 1st-99th percentile
3. **NO SMOTE**: scale_pos_weight (8.99) works better than oversampling
4. **RobustScaler**: Better for data with outliers
5. **Regularization**: L1=0.1, L2=1.0, gamma=0.1, min_child_weight=2
6. **Threshold Optimization**: 0.75 (vs default 0.50)

### Top Features
1. `price_range` (13.7%) - High-low price range
2. `max_drawdown_pct` (9.5%) - Maximum drawdown percentage
3. `hl_spread_mean` (8.6%) - Average high-low spread
4. `close_max` (7.0%) - Maximum closing price
5. `close_min` (6.8%) - Minimum closing price

## Usage Example

```python
import joblib
import numpy as np

# Load model
model_data = joblib.load('outputs/model_final.joblib')
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']
threshold = model_data['optimal_threshold']  # 0.75
clip_params = model_data['clip_params']

# Prepare new data (same features as training)
X_new = ...  # Extract features using build_dataset.py logic

# Apply clipping
for i, (lower, upper) in enumerate(clip_params):
    X_new[:, i] = np.clip(X_new[:, i], lower, upper)

# Scale and predict
X_scaled = scaler.transform(X_new)
probabilities = model.predict_proba(X_scaled)[:, 1]
predictions = (probabilities >= threshold).astype(int)
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Source | `combined_dataset.csv` (3-second ticks) |
| Date Range | 2020-01-26 to 2025-10-31 |
| Symbols | MSFT, AAPL, NVDA, SPY, QQQ |
| Daily Bars | 8,720 (1,744 days × 5 symbols) |
| Patterns Detected | 27 Cup and Handle formations |
| Training Samples | 8,400 windows |
| Positive Samples | 740 (8.81%) |
| Window Size | 65 days |

## Detection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Cup Duration | 7-65 days | Min-max cup formation time |
| Cup Depth | 12-33% | Acceptable depth range |
| Handle Duration | 3-25 days | Handle consolidation time (relaxed from default 5) |
| Handle Depth | ≤18% | Maximum handle retracement |
| Peak Similarity | ≤8% | Max difference between cup edges (relaxed from default 5%) |
| Volume Breakout | ≥1.0x | Volume ratio on breakout (relaxed for proxy volume) |

> Note: Parameters are relaxed from standard values due to limited data (5 symbols only)

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0
xgboost>=1.7.0
joblib>=1.2.0
```

## Notes

- The model uses **temporal split** (train: 2020-2024, test: 2024-2025) to prevent data leakage
- Cup and Handle is a **rare pattern** - class imbalance is handled via scale_pos_weight
- The 0.75 threshold prioritizes precision over recall (fewer false positives)
