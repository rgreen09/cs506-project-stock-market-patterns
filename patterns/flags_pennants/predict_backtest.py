"""
Flags and Pennants Pattern Prediction and Backtesting
--
Uses trained ML model to:
1. Predict flag/pennant patterns in real-time sliding windows
2. Generate trading signals
3. Backtest the strategy
4. Visualize results

Usage:
    # Make predictions on new data
    python Predict_and_backtest.py --input data/combined_dataset.csv --symbol AAPL --model random_forest_*.pkl

    # Run with custom threshold for higher precision
    python Predict_and_backtest.py --input data/combined_dataset.csv --symbol AAPL --model random_forest_*.pkl --threshold 0.8

    # Backtest specific date range
    python Predict_and_backtest.py --input data/combined_dataset.csv --symbol AAPL --model random_forest_*.pkl --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

# Import feature computation from dataset builder
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import compute_window_features directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Build flags_pennants_dataset.py")
)
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
compute_window_features = build_module.compute_window_features


def load_model_artifacts(model_path, scaler_path, features_path):
    """
    Load trained model, scaler, and feature names.

    Args:
        model_path: Path to model pickle file
        scaler_path: Path to scaler pickle file
        features_path: Path to features text file

    Returns:
        model, scaler, feature_names
    """
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    print(f"Loading features from: {features_path}")
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"Model loaded successfully with {len(feature_names)} features")
    return model, scaler, feature_names


def load_and_prepare_symbol_data(input_path, symbol, start_date=None, end_date=None):
    """
    Load data for a specific symbol and date range.

    Args:
        input_path: Path to input CSV
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD) or None
        end_date: End date (YYYY-MM-DD) or None

    Returns:
        DataFrame with timestamp and close price
    """
    print(f"\nLoading data for {symbol}...")
    df = pd.read_csv(input_path)

    # Handle wide format (symbol as column) or long format (symbol as row value)
    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for symbol '{symbol}'")
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        # Wide format
        if symbol not in df.columns:
            available_symbols = [col for col in df.columns if col not in ['ID', 'TimeStamp', 'timestamp', 'Date', 'Time']]
            raise ValueError(f"Symbol '{symbol}' not found. Available: {', '.join(available_symbols)}")

        timestamp_col = 'TimeStamp' if 'TimeStamp' in df.columns else 'timestamp'
        df = pd.DataFrame({
            'timestamp': df[timestamp_col],
            'close': df[symbol]
        })
        df = df.dropna(subset=['close']).reset_index(drop=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])

    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['datetime'] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df['datetime'] <= end_dt]

    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def create_sliding_windows(df, window_size=300, stride=50):
    """
    Create sliding windows from price data.

    Args:
        df: DataFrame with timestamp and close columns
        window_size: Window size in bars
        stride: Stride between windows

    Returns:
        List of (window_df, start_time, end_time) tuples
    """
    print(f"\nCreating sliding windows (size={window_size}, stride={stride})...")

    windows = []
    n_total = len(df)

    for i in range(0, n_total - window_size + 1, stride):
        window_df = df.iloc[i:i+window_size]
        start_time = window_df.iloc[0]['timestamp']
        end_time = window_df.iloc[-1]['timestamp']
        windows.append((window_df, start_time, end_time))

    print(f"Created {len(windows):,} windows")
    return windows


def predict_patterns(windows, model, scaler, feature_names, threshold=0.5):
    """
    Predict flag/pennant patterns for all windows.

    Args:
        windows: List of window tuples
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        threshold: Prediction probability threshold

    Returns:
        DataFrame with predictions
    """
    print(f"\nPredicting patterns (threshold={threshold})...")

    predictions = []

    for i, (window_df, start_time, end_time) in enumerate(windows):
        if i % 1000 == 0:
            print(f"  Processing window {i+1:,}/{len(windows):,}...")

        # Compute features
        close_window = window_df['close']
        features = compute_window_features(close_window)

        # Create feature vector in correct order
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)

        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)

        # Predict
        pred_proba = model.predict_proba(feature_vector_scaled)[0, 1]
        pred_label = 1 if pred_proba >= threshold else 0

        predictions.append({
            'start_time': start_time,
            'end_time': end_time,
            'start_price': close_window.iloc[0],
            'end_price': close_window.iloc[-1],
            'probability': pred_proba,
            'prediction': pred_label
        })

    df_predictions = pd.DataFrame(predictions)

    n_patterns = df_predictions['prediction'].sum()
    pct_patterns = 100 * n_patterns / len(df_predictions)
    print(f"\nPredicted {n_patterns:,} patterns ({pct_patterns:.2f}% of windows)")

    return df_predictions


def generate_trading_signals(df_predictions, hold_periods=5):
    """
    Generate trading signals from pattern predictions.

    Args:
        df_predictions: DataFrame with predictions
        hold_periods: Number of periods to hold after pattern detected

    Returns:
        DataFrame with trading signals
    """
    print(f"\nGenerating trading signals (hold_periods={hold_periods})...")

    # Filter to only predicted patterns
    signals = df_predictions[df_predictions['prediction'] == 1].copy()

    # Add signal metadata
    signals['signal_type'] = 'BUY'  # Buy at breakout
    signals['entry_price'] = signals['end_price']
    signals['confidence'] = signals['probability']

    print(f"Generated {len(signals):,} trading signals")

    return signals


def backtest_strategy(df, signals, hold_periods=5, position_size=1.0):
    """
    Simple backtest of the pattern detection strategy.

    Args:
        df: Full price DataFrame
        signals: Trading signals DataFrame
        hold_periods: Bars to hold position
        position_size: Fraction of capital per trade

    Returns:
        Backtest results DataFrame
    """
    print(f"\nBacktesting strategy...")

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('datetime')

    trades = []

    for _, signal in signals.iterrows():
        entry_time = pd.to_datetime(signal['end_time'])
        entry_price = signal['entry_price']

        # Find exit time (hold_periods after entry)
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = min(entry_idx + hold_periods, len(df) - 1)

        exit_time = df.index[exit_idx]
        exit_price = df.iloc[exit_idx]['close']

        # Calculate return
        pct_return = (exit_price - entry_price) / entry_price
        profit = pct_return * position_size

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': pct_return * 100,
            'profit': profit,
            'confidence': signal['confidence']
        })

    df_trades = pd.DataFrame(trades)

    if len(df_trades) > 0:
        # Calculate statistics
        total_return = df_trades['profit'].sum()
        avg_return = df_trades['return_pct'].mean()
        win_rate = (df_trades['return_pct'] > 0).sum() / len(df_trades) * 100

        print(f"\nBacktest Results:")
        print(f"  Total trades: {len(df_trades):,}")
        print(f"  Win rate: {win_rate:.2f}%")
        print(f"  Average return per trade: {avg_return:.2f}%")
        print(f"  Total return: {total_return:.2f}x")
        print(f"  Best trade: {df_trades['return_pct'].max():.2f}%")
        print(f"  Worst trade: {df_trades['return_pct'].min():.2f}%")
    else:
        print("No trades executed")

    return df_trades


def plot_predictions_timeline(df, predictions, symbol, max_points=10000):
    """
    Plot price with predicted patterns highlighted.

    Args:
        df: Price DataFrame
        predictions: Predictions DataFrame
        symbol: Stock symbol for title
        max_points: Maximum data points to plot
    """
    print(f"\nCreating prediction timeline plot...")

    # Subsample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df.copy()

    df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'])

    # Get pattern predictions
    patterns = predictions[predictions['prediction'] == 1].copy()
    patterns['end_datetime'] = pd.to_datetime(patterns['end_time'])

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Price line
    ax.plot(df_plot['datetime'], df_plot['close'], label='Close Price', linewidth=1, alpha=0.7)

    # Mark predicted patterns
    for _, pattern in patterns.iterrows():
        ax.axvline(pattern['end_datetime'], color='red', alpha=0.3, linewidth=1)

    # Add pattern markers
    ax.scatter(patterns['end_datetime'], patterns['end_price'],
              color='red', s=100, marker='^', label='Predicted Patterns', zorder=5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{symbol} - Flag/Pennant Pattern Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{symbol}_predictions_timeline.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Timeline plot saved: {filename}")
    plt.close()


def plot_probability_distribution(predictions):
    """
    Plot distribution of prediction probabilities.

    Args:
        predictions: Predictions DataFrame
    """
    print(f"\nCreating probability distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of all probabilities
    ax1.hist(predictions['probability'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Pattern Probabilities')
    ax1.grid(True, alpha=0.3)

    # Box plot comparing predicted vs not predicted
    patterns = predictions[predictions['prediction'] == 1]['probability']
    no_patterns = predictions[predictions['prediction'] == 0]['probability']

    ax2.boxplot([no_patterns, patterns], labels=['No Pattern', 'Pattern Detected'])
    ax2.set_ylabel('Prediction Probability')
    ax2.set_title('Probability Distribution by Prediction')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'probability_distribution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Probability distribution plot saved: {filename}")
    plt.close()


def plot_backtest_results(trades):
    """
    Plot backtest performance.

    Args:
        trades: Trades DataFrame
    """
    if len(trades) == 0:
        print("No trades to plot")
        return

    print(f"\nCreating backtest results plot...")

    # Calculate cumulative returns
    trades = trades.sort_values('entry_time')
    trades['cumulative_return'] = (1 + trades['return_pct'] / 100).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Cumulative returns
    ax1.plot(trades['entry_time'], trades['cumulative_return'], linewidth=2)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Strategy Cumulative Returns')
    ax1.grid(True, alpha=0.3)

    # Individual trade returns
    colors = ['green' if x > 0 else 'red' for x in trades['return_pct']]
    ax2.bar(range(len(trades)), trades['return_pct'], color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Individual Trade Returns')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'backtest_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Backtest results plot saved: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Predict flag/pennant patterns and backtest trading strategy'
    )
    parser.add_argument('--input', type=str, default='data/combined_dataset.csv',
                       help='Input CSV file with price data')
    parser.add_argument('--symbol', type=str, required=True,
                       help='Stock symbol to analyze')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model pickle file')
    parser.add_argument('--scaler', type=str, default=None,
                       help='Path to scaler pickle file (auto-detected if not provided)')
    parser.add_argument('--features', type=str, default=None,
                       help='Path to features text file (auto-detected if not provided)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction probability threshold (default: 0.5)')
    parser.add_argument('--window-size', type=int, default=300,
                       help='Window size in bars (default: 300)')
    parser.add_argument('--stride', type=int, default=50,
                       help='Stride between windows (default: 50)')
    parser.add_argument('--hold-periods', type=int, default=5,
                       help='Periods to hold position after pattern (default: 5)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for analysis (YYYY-MM-DD)')

    args = parser.parse_args()

    # Auto-detect scaler and features files if not provided
    if args.scaler is None:
        # Extract timestamp from model filename (e.g., random_forest_20251201_003050.pkl)
        parts = args.model.replace('.pkl', '').split('_')
        if len(parts) >= 3:
            timestamp = '_'.join(parts[-2:])  # Get last two parts (date_time)
        else:
            timestamp = parts[-1]
        args.scaler = f'scaler_{timestamp}.pkl'
    if args.features is None:
        parts = args.model.replace('.pkl', '').split('_')
        if len(parts) >= 3:
            timestamp = '_'.join(parts[-2:])
        else:
            timestamp = parts[-1]
        args.features = f'features_{timestamp}.txt'

    print("="*60)
    print("Flag/Pennant Pattern Prediction & Backtesting")
    print("="*60)

    # Load model
    model, scaler, feature_names = load_model_artifacts(
        args.model, args.scaler, args.features
    )

    # Load data
    df = load_and_prepare_symbol_data(
        args.input, args.symbol, args.start_date, args.end_date
    )

    # Create windows
    windows = create_sliding_windows(df, args.window_size, args.stride)

    # Predict patterns
    predictions = predict_patterns(windows, model, scaler, feature_names, args.threshold)

    # Generate signals
    signals = generate_trading_signals(predictions, args.hold_periods)

    # Backtest
    if len(signals) > 0:
        trades = backtest_strategy(df, signals, args.hold_periods)

        # Save results
        predictions.to_csv(f'{args.symbol}_predictions.csv', index=False)
        print(f"\nPredictions saved: {args.symbol}_predictions.csv")

        if len(trades) > 0:
            trades.to_csv(f'{args.symbol}_trades.csv', index=False)
            print(f"Trades saved: {args.symbol}_trades.csv")

            # Plot results
            plot_backtest_results(trades)

    # Visualization
    plot_predictions_timeline(df, predictions, args.symbol)
    plot_probability_distribution(predictions)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
