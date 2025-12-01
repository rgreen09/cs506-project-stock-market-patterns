"""
Live Pattern Scanner - Clean Interface for Flag/Pennant Detection

Uses trained Random Forest model to scan recent price data for patterns.
No verbose logging, clean output only.

Usage:
    python Live_scanner.py --symbol AAPL --days 10
    python Live_scanner.py --symbol AAPL --days 30 --threshold 0.8
"""

import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import importlib.util

# Suppress all warnings and verbose output
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Disable parallel logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings if present

# Redirect verbose sklearn output
import io
import contextlib


def load_model_artifacts(model_path, scaler_path, features_path):
    """Load trained model, scaler, and feature names."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f]

    return model, scaler, feature_names


def load_build_module():
    """Dynamically load the feature engineering module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(script_dir, "build_dataset.py")

    spec = importlib.util.spec_from_file_location("build_module", build_path)
    build_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_module)

    return build_module.compute_window_features


def load_recent_data(csv_path, symbol, days=10):
    """Load recent N days of price data."""
    df = pd.read_csv(csv_path)

    # Handle wide format (symbols as columns)
    if 'symbol' not in df.columns:
        timestamp_col = 'TimeStamp' if 'TimeStamp' in df.columns else 'timestamp'
        df = pd.DataFrame({
            'timestamp': df[timestamp_col],
            'symbol': symbol,
            'close': df[symbol]
        })
    else:
        df = df[df['symbol'] == symbol].copy()

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get recent N days
    cutoff_date = df['timestamp'].max() - timedelta(days=days)
    df = df[df['timestamp'] >= cutoff_date].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def scan_for_patterns(df, compute_features_fn, model, scaler, feature_names, threshold=0.75, window_size=300, stride=50):
    """Scan data for patterns using sliding windows."""
    results = []
    n_windows = len(df) - window_size + 1

    if n_windows <= 0:
        return pd.DataFrame()

    # Progress indicator
    n_scanned = len(range(0, n_windows, stride))
    print(f"\nScanning {n_scanned:,} windows (stride={stride})...", end='', flush=True)

    for i in range(0, n_windows, stride):
        # Extract window
        window = df.iloc[i:i + window_size]

        # Compute features (suppress any output)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            features = compute_features_fn(window['close'])

        if features is None:
            continue

        # Prepare features for model
        feature_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)

        # Predict (suppress verbose output)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            proba = model.predict_proba(feature_scaled)[0, 1]

        # Only keep predictions above threshold
        if proba >= threshold:
            results.append({
                'timestamp': window['timestamp'].iloc[-1],
                'start_timestamp': window['timestamp'].iloc[0],
                'price': window['close'].iloc[-1],
                'confidence': proba,
                'pattern_type': 'Flag/Pennant'
            })

    print(" Done!")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Scan recent price data for flag/pennant patterns'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol to scan (default: AAPL)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=10,
        help='Number of recent days to scan (default: 10)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Minimum confidence threshold (default: 0.75)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=50,
        help='Window stride for faster scanning (default: 50)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='intraday_3sec_data.csv',
        help='Path to price data CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest_20251201_003050.pkl',
        help='Path to trained model'
    )

    args = parser.parse_args()

    # Auto-detect scaler and features files from model filename
    parts = args.model.replace('.pkl', '').split('_')
    if len(parts) >= 3:
        timestamp = '_'.join(parts[-2:])
    else:
        timestamp = parts[-1]

    scaler_path = f'scaler_{timestamp}.pkl'
    features_path = f'features_{timestamp}.txt'

    # Header
    print("=" * 70)
    print(f"Live Pattern Scanner - {args.symbol}")
    print("=" * 70)
    print(f"Scanning last {args.days} days (threshold: {args.threshold:.0%})")

    # Check files exist
    for path, name in [(args.model, "Model"), (scaler_path, "Scaler"), (features_path, "Features"), (args.data, "Data")]:
        if not os.path.exists(path):
            print(f"\nError: {name} file not found: {path}")
            return 1

    # Load artifacts
    print(f"\nLoading model artifacts...")
    model, scaler, feature_names = load_model_artifacts(args.model, scaler_path, features_path)

    # Load feature engineering function
    compute_features_fn = load_build_module()

    # Load recent data
    print(f"Loading recent {args.days} days of data for {args.symbol}...")
    df = load_recent_data(args.data, args.symbol, args.days)
    print(f"Loaded {len(df):,} price records from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Scan for patterns
    detections = scan_for_patterns(
        df, compute_features_fn, model, scaler, feature_names,
        threshold=args.threshold,
        stride=args.stride
    )

    # Display results
    print("\n" + "=" * 70)
    if len(detections) == 0:
        print(f"No patterns detected above {args.threshold:.0%} confidence threshold")
    else:
        print(f"Detected {len(detections)} pattern(s):")
        print("=" * 70)

        for idx, row in detections.iterrows():
            print(f"\n[{idx + 1}] {row['pattern_type']}")
            print(f"    Time:       {row['timestamp']}")
            print(f"    Price:      ${row['price']:.2f}")
            print(f"    Confidence: {row['confidence']:.1%}")
            print(f"    Duration:   {row['start_timestamp']} to {row['timestamp']}")

        # Save to CSV
        output_file = f"{args.symbol}_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        detections.to_csv(output_file, index=False)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_file}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
