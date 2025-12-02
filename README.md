# Stock Pattern Detection - Flags & Pennants

Machine learning system for detecting flag and pennant continuation patterns in stock price data using Random Forest classification.

## Features

- **Pattern Detection**: Identifies bullish/bearish flag and pennant continuation patterns
- **High Accuracy**: 95% recall with 0.9998 ROC-AUC score
- **Real-time Scanning**: Clean interface for scanning recent price data
- **Class Imbalance Handling**: Uses balanced class weights for optimal performance

## Project Structure

```
Stock/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── src/                         # Source code
│   ├── build_dataset.py         # Generate labeled training datasets
│   ├── train_model.py           # Train ML models
│   ├── live_scanner.py          # Scan recent data for patterns
│   ├── pattern_detector.py      # Pattern detection logic
│   └── predict_backtest.py      # Backtesting predictions
├── visualizations/              # Model performance plots
│   ├── random_forest_feature_importance.png
│   ├── random_forest_roc_curve.png
│   ├── random_forest_pr_curve.png
│   └── flags_pennants_test.png
├── outputs/                     # Example detection results
│   └── AAPL_detections_20251201_144210.csv
├── models/                      # Trained models (gitignored)
└── data/                        # Datasets (gitignored)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rgreen09/cs506-project-stock-market-patterns.git
cd cs506-project-stock-market-patterns
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Build Training Dataset

Generate labeled data from raw 3-second intraday price data:

```bash
python src/build_dataset.py --symbol AAPL --sample-rate 50
```

Options:
- `--symbol`: Stock symbol to process
- `--sample-rate`: Process every Nth window (default: 50)
- `--input`: Path to CSV file with price data

### 2. Train Model

Train Random Forest classifier on labeled data:

```bash
python src/train_model.py --input AAPL_flags_pennants_15m_windows.csv
```

Options:
- `--input`: Path to labeled dataset CSV
- `--use-smote`: Apply SMOTE oversampling (optional)
- `--test-size`: Fraction for test set (default: 0.2)
- `--no-xgboost`: Skip XGBoost training

Outputs:
- Trained model (.pkl)
- Feature scaler (.pkl)
- Feature names (.txt)
- Performance visualizations (.png)

### 3. Scan Recent Data

Use trained model to detect patterns in recent price data:

```bash
python src/live_scanner.py --symbol AAPL --days 10 --threshold 0.75 --data data/combined_dataset.csv
```

Options:
- `--symbol`: Stock symbol to scan
- `--days`: Number of recent days to scan (default: 10)
- `--threshold`: Minimum confidence threshold (default: 0.75)
- `--stride`: Window stride for faster scanning (default: 50)
- `--data`: Path to price data CSV
- `--model`: Path to trained model (default: auto-detect latest)

Output:
```
======================================================================
Live Pattern Scanner - AAPL
======================================================================
Detected 8 pattern(s):

[1] Flag/Pennant
    Time:       2025-10-31 09:41:02
    Price:      $270.11
    Confidence: 96.4%
    Duration:   2025-10-31 09:26:03 to 2025-10-31 09:41:02
...
```

Results are automatically saved to CSV in the outputs directory.

## Model Performance

- **Recall**: 95% (catches 118/124 patterns)
- **Precision**: 75%
- **ROC-AUC**: 0.9998
- **Average Precision**: 0.9458
- **Training Data**: 216,828 windows with 621 positive examples (0.29%)

### Visualizations

Model performance visualizations are available in the `visualizations/` directory:

- **Feature Importance** (`random_forest_feature_importance.png`): Shows top 20 most influential features for pattern detection
- **ROC Curve** (`random_forest_roc_curve.png`): Displays classifier performance with AUC=0.9998
- **Precision-Recall Curve** (`random_forest_pr_curve.png`): Shows precision-recall tradeoff (AP=0.9458)
- **Example Pattern** (`flags_pennants_test.png`): Visualization of detected flag/pennant pattern

### Example Output

See `outputs/AAPL_detections_20251201_144210.csv` for an example detection run showing 8 patterns detected from Oct 21-31, 2025 AAPL data with confidence scores ranging from 75-96%.

## Data Format

The system expects CSV files with the following columns:
- `timestamp`: Datetime
- `symbol`: Stock ticker (or symbols as separate columns in wide format)
- `close`: Closing price

## Technical Details

### Pattern Detection
- **Window Size**: 300 bars (15 minutes of 3-second data)
- **Features**: 35 engineered features including:
  - Price momentum and trends
  - Support/resistance levels
  - Volume analysis
  - Pattern shape metrics

### Class Imbalance Handling
- Balanced class weights in Random Forest
- Optional SMOTE oversampling
- Stratified train/test splits

## Requirements

See `requirements.txt` for full list. Key dependencies:
- pandas
- numpy
- scikit-learn
- xgboost (optional)
- matplotlib
- seaborn
- imbalanced-learn (for SMOTE)

