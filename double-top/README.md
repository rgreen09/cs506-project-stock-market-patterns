# Double-Top Pattern Detection

End-to-end workflow for detecting double-top chart patterns: rule-based detection for labeling, sliding-window dataset generation, dataset combination, and ML modeling with saved artifacts and visuals.

## Components
- `A_double_top_detector.py`: Peak/trough-based detector with smoothing and configurable thresholds (peak tolerance %, minimum neckline drop, gap between peaks, optional confirmation break below neckline).
- `B_build_double_top_dataset.py`: Builds labeled 15-minute (300-bar, 3s bars) windows per symbol, filters out non-trading hours/day spans, computes ~30 engineered features, labels via relaxed detector settings.
- `C_combine_datasets.py`: Merges per-symbol window CSVs into `data/combined_double_top_15m_windows.csv` in memory-safe chunks.
- `D_model_training.py`: Loads combined dataset, price-normalizes key columns, drops absolutes, scales features, trains/evaluates multiple models, saves scaler/models, and logs metrics.
- Visuals under `double-top/visualizations/double_top/`: sample detected patterns across symbols.

## Detection Logic (rule-based)
1. Smooth prices (rolling mean, default 5 bars).
2. Find local peaks/troughs with minimum prominence.
3. Iterate peak pairs respecting min/max gap; require similar heights within `peak_tolerance`.
4. Compute neckline (min between peaks) and ensure drop ≥ `min_drop_pct`.
5. Optional confirmation: price breaks below neckline (with buffer) after second peak; configurable `min_bars_after_peak`.
6. Returns detailed `DoubleTopResult` (indices, prices, drop %, height diff %, bars between peaks, confirmation flag, failure reason).

Quick use:
```
from A_double_top_detector import DoubleTopDetector
result = DoubleTopDetector().detect(close_series, window_bars=len(close_series))
print(result.is_pattern, result.failure_reason)
```

## Dataset Builder
- Input: intraday CSV with `TimeStamp` and symbol columns (default `data/combined_dataset.csv`).
- Windows: 300 bars (~15 minutes). Filters outside 09:30–16:00 and windows spanning days (configurable).
- Labeling: relaxed detector config (`peak_tolerance=0.04`, `min_drop_pct=0.002`, `min_gap=5`, `max_gap_ratio=0.6`, no confirmation, smoothing=7).
- Features (~30): price stats, returns/momentum (1/5/20/30), slopes (full/last30), peak geometry (counts, positions, gap, height diff %, neckline drop %, drawdown, sharpness), volatility (rolling std 20/60, true range mean 20), MAs/oscillators (SMA20, SMA50, close_over_sma20, RSI14, %B).

Run:
```
python B_build_double_top_dataset.py --symbol AAPL --input data/combined_dataset.csv --output AAPL_double_top_15m_windows.csv
```

## Combine Datasets
Merge per-symbol outputs:
```
python C_combine_datasets.py -- (defaults: data dir = data, output = data/combined_double_top_15m_windows.csv)
```
Processes in 10k-row chunks to stay memory-safe and preserves header ordering.

## ML Training Pipeline
Steps (`D_model_training.py`):
1. Load `data/combined_double_top_15m_windows.csv`.
2. Drop metadata (`symbol`, `start_timestamp`, `end_timestamp`).
3. Price-normalize select columns (e.g., `close_std_pct`, `slope_entire_window_pct`, `true_range_mean_20_pct`, `peak1_sharpness_pct`, `sma_50_ratio`); drop absolute-value counterparts.
4. Split stratified train/test (80/20, random_state=42).
5. Scale features with `StandardScaler` (fit on train).
6. Train and evaluate: Random Forest, XGBoost, LightGBM, Logistic Regression, MLP (class-balanced where applicable).
7. Save artifacts (scaler + models) and log metrics.

Run:
```
python D_model_training.py
```

Artifacts (per Round One run): saved under `double-top/machine_learning/models/` with `scaler.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl`, `lightgbm_model.pkl`, `logistic_regression_model.pkl`, `neural_network_model.pkl`; logs in `best_model/ML_log.txt` (script default) and `ML-output.txt` summary.

## Round One Results (from ML-output.txt)
- Data: 602,217 windows; train 481,773 / test 120,444; class balance ~66.7% negative / 33.3% positive.
- Features after processing: 26.

Test-set metrics:

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC | Train(s) |
| ------------------- | -------- | --------- | ------ | ----- | ------- | -------- |
| Random Forest       | 0.9910   | 0.9772    | 0.9961 | 0.9866| 0.9994  | 65.57    |
| XGBoost             | 0.9872   | 0.9665    | 0.9961 | 0.9811| 0.9984  | 3.03     |
| LightGBM            | 0.9822   | 0.9552    | 0.9932 | 0.9738| 0.9973  | 2.53     |
| Logistic Regression | 0.9731   | 0.9418    | 0.9797 | 0.9604| 0.9950  | 4.38     |
| Neural Network      | 0.9896   | 0.9782    | 0.9909 | 0.9845| 0.9976  | 378.35   |

- Best by F1 (test): Random Forest (0.9866). Smallest overfit gap belongs to Logistic Regression (-0.0006); tree/boosting models show minimal gaps (≤0.0134).

## Visualizations
Sample detected double-top windows saved under `double-top/visualizations/double_top/` for symbols such as AAPL, MSFT, NVDA, QQQ, SPY (files named `{SYMBOL}_double_top_{n}.png`).

