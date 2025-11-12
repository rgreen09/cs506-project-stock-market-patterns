"""
Double-Top Pattern Dataset Builder

This script processes intraday stock data to build a labeled dataset for detecting
double-top patterns using sliding 15-minute windows (300 bars at 3-second intervals).

Usage:
    Minimal (uses defaults):
        python build_double_top_dataset.py --symbol AAPL
    
    Full example:
        python build_double_top_dataset.py --input data/combined_dataset.csv --output AAPL_double_top_15m_windows.csv --symbol AAPL --window-bars 300

Arguments:
    --symbol (required): Stock symbol to process (e.g., AAPL, MSFT, NVDA)
    --input (optional): Path to input CSV file (default: data/combined_dataset.csv)
    --output (optional): Path to output CSV file (default: {symbol}_double_top_15m_windows.csv)
    --window-bars (optional): Size of sliding window in bars (default: 300)
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bar.")

# Constants for double-top detection
PEAK_TOL = 0.02       # peaks within 2% in height
MIN_DROP_PCT = 0.005  # at least 0.5% drop to neckline
BREAK_BUFFER = 0.001  # small extra buffer below neckline
MIN_GAP = 10          # minimum bars between peaks


def find_local_peaks(prices: np.ndarray) -> List[int]:
    """
    Find local maxima in a price array.
    
    A local peak at index i is defined as:
    prices[i] > prices[i-1] and prices[i] > prices[i+1]
    
    Args:
        prices: 1D numpy array of prices
        
    Returns:
        List of indices where local peaks occur
    """
    peaks = []
    n = len(prices)
    
    if n < 3:
        return peaks
    
    for i in range(1, n - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append(i)
    
    return peaks


def find_local_troughs(prices: np.ndarray) -> List[int]:
    """
    Find local minima in a price array.
    
    A local trough at index i is defined as:
    prices[i] < prices[i-1] and prices[i] < prices[i+1]
    
    Args:
        prices: 1D numpy array of prices
        
    Returns:
        List of indices where local troughs occur
    """
    troughs = []
    n = len(prices)
    
    if n < 3:
        return troughs
    
    for i in range(1, n - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append(i)
    
    return troughs


def compute_window_features(close_window: pd.Series) -> Dict[str, float]:
    """
    Compute 30 engineered features from a window of closing prices.
    
    Args:
        close_window: pandas Series of closing prices (length should be 300)
        
    Returns:
        Dictionary mapping feature names to their values
    """
    c = close_window.values
    n = len(c)
    
    # Create smoothed price series for peak/trough detection
    smooth_series = pd.Series(c).rolling(window=5, center=True, min_periods=1).mean()
    smooth = smooth_series.values
    
    # Find peaks and troughs
    peaks = find_local_peaks(smooth)
    troughs = find_local_troughs(smooth)
    
    # Initialize feature dictionary
    features = {}
    
    # ===== Core price stats (5 features) =====
    features['close_last'] = float(c[-1])
    features['close_mean'] = float(np.mean(c))
    features['close_std'] = float(np.std(c))
    features['price_range_abs'] = float(np.max(c) - np.min(c))
    features['price_range_pct'] = float((features['price_range_abs'] / c[-1]) if c[-1] != 0 else 0.0)
    
    # ===== Returns & momentum (7 features) =====
    features['cumulative_return_window'] = float((c[-1] / c[0] - 1) if c[0] != 0 else 0.0)
    features['ret_1'] = float((c[-1] / c[-2] - 1) if n > 1 and c[-2] != 0 else 0.0)
    features['ret_5'] = float((c[-1] / c[-6] - 1) if n > 5 and c[-6] != 0 else 0.0)
    features['ret_20'] = float((c[-1] / c[-21] - 1) if n > 20 and c[-21] != 0 else 0.0)
    features['momentum_last_30'] = float((c[-1] - c[-31]) / c[-31] if n > 30 and c[-31] != 0 else 0.0)
    
    # Slope of entire window
    x_full = np.arange(n)
    if n > 1:
        slope_full = np.polyfit(x_full, c, 1)[0]
    else:
        slope_full = 0.0
    features['slope_entire_window'] = float(slope_full)
    
    # Slope of last 30 points
    if n >= 30:
        x_last30 = np.arange(30)
        slope_last30 = np.polyfit(x_last30, c[-30:], 1)[0]
    else:
        x_last30 = np.arange(n)
        slope_last30 = np.polyfit(x_last30, c, 1)[0] if n > 1 else 0.0
    features['slope_last_30'] = float(slope_last30)
    
    # ===== Peak / trough geometry (10 features) =====
    features['num_peaks_window'] = float(len(peaks))
    features['num_troughs_window'] = float(len(troughs))
    
    if len(peaks) >= 2:
        i1, i2 = peaks[-2], peaks[-1]
        h1, h2 = smooth[i1], smooth[i2]
        neckline = float(np.min(smooth[i1:i2+1]))
        
        features['peak1_rel_pos'] = float(i1 / (n - 1) if n > 1 else 0.0)
        features['peak2_rel_pos'] = float(i2 / (n - 1) if n > 1 else 0.0)
        features['bars_between_last_two_peaks'] = float(i2 - i1)
        
        max_peak = max(h1, h2)
        features['peak_height_diff_pct'] = float(abs(h1 - h2) / max_peak if max_peak != 0 else 0.0)
        
        min_peak = min(h1, h2)
        features['neckline_drop_pct'] = float((min_peak - neckline) / min_peak if min_peak != 0 else 0.0)
        
        features['drawdown_from_last_peak'] = float((h2 - c[-1]) / h2 if h2 != 0 else 0.0)
        
        # Peak sharpness
        if 1 <= i1 < len(smooth) - 1:
            features['peak1_sharpness'] = float(h1 - (smooth[i1-1] + smooth[i1+1]) / 2)
        else:
            features['peak1_sharpness'] = 0.0
        
        if 1 <= i2 < len(smooth) - 1:
            features['peak2_sharpness'] = float(h2 - (smooth[i2-1] + smooth[i2+1]) / 2)
        else:
            features['peak2_sharpness'] = 0.0
    else:
        # Set all peak-related features to 0 if fewer than 2 peaks
        features['peak1_rel_pos'] = 0.0
        features['peak2_rel_pos'] = 0.0
        features['bars_between_last_two_peaks'] = 0.0
        features['peak_height_diff_pct'] = 0.0
        features['neckline_drop_pct'] = 0.0
        features['drawdown_from_last_peak'] = 0.0
        features['peak1_sharpness'] = 0.0
        features['peak2_sharpness'] = 0.0
    
    # ===== Volatility / local range (3 features) =====
    if n >= 20:
        features['rolling_std_20'] = float(np.std(c[-20:]))
    else:
        features['rolling_std_20'] = float(np.std(c))
    
    if n >= 60:
        features['rolling_std_60'] = float(np.std(c[-60:]))
    else:
        features['rolling_std_60'] = float(np.std(c))
    
    if n >= 20:
        diffs = np.diff(c[-20:])
        features['true_range_mean_20'] = float(np.mean(np.abs(diffs)) if len(diffs) > 0 else 0.0)
    elif n > 1:
        diffs = np.diff(c)
        features['true_range_mean_20'] = float(np.mean(np.abs(diffs)))
    else:
        features['true_range_mean_20'] = 0.0
    
    # ===== Moving average & oscillator features (5 features) =====
    if n >= 20:
        sma_20 = float(np.mean(c[-20:]))
    else:
        sma_20 = float(np.mean(c))
    features['sma_20'] = sma_20
    features['close_over_sma20'] = float(c[-1] / sma_20 if sma_20 != 0 else 1.0)
    
    if n >= 50:
        features['sma_50'] = float(np.mean(c[-50:]))
    else:
        features['sma_50'] = float(np.mean(c))
    
    # RSI_14 calculation
    if n >= 15:
        # Use last 14 periods for RSI
        delta = np.diff(c[-15:])  # Need 15 to get 14 deltas
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1 + rs))
        features['rsi_14'] = float(rsi)
    else:
        # Not enough data for RSI
        features['rsi_14'] = 50.0  # Neutral RSI
    
    # Bollinger %B
    if n >= 20:
        ma20 = float(np.mean(c[-20:]))
        std20 = float(np.std(c[-20:]))
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        if upper > lower:
            features['percent_b'] = float((c[-1] - lower) / (upper - lower))
        else:
            features['percent_b'] = 0.0
    else:
        features['percent_b'] = 0.0
    
    return features


def is_double_top_in_window(close_window: pd.Series, window_bars: int = 300) -> bool:
    """
    Detect if a double-top pattern exists in the given window.
    
    A double-top pattern requires:
    1. Two distinct peaks of similar height (within 2% tolerance)
    2. Peaks separated by appropriate spacing (MIN_GAP to MAX_GAP bars)
    3. Significant drop to neckline (at least 0.5%)
    4. Confirmation: price breaks below neckline after second peak
    
    Args:
        close_window: pandas Series of closing prices
        window_bars: size of the window (default 300)
        
    Returns:
        True if double-top pattern is detected, False otherwise
    """
    # Smooth prices
    smooth_series = pd.Series(close_window).rolling(window=5, center=True, min_periods=1).mean()
    smooth = smooth_series.values
    
    # Find local peaks
    peaks = find_local_peaks(smooth)
    
    # Need at least 2 peaks
    if len(peaks) < 2:
        return False
    
    # Get last two peaks
    i1, i2 = peaks[-2], peaks[-1]
    MAX_GAP = window_bars // 2
    
    # Check spacing
    if i2 - i1 < MIN_GAP or i2 - i1 > MAX_GAP:
        return False
    
    # Get peak heights
    h1, h2 = smooth[i1], smooth[i2]
    if max(h1, h2) == 0:
        return False
    
    # Check height similarity
    height_diff_pct = abs(h1 - h2) / max(h1, h2)
    if height_diff_pct > PEAK_TOL:
        return False
    
    # Find neckline (minimum between peaks)
    segment_between = smooth[i1:i2+1]
    neck_price = np.min(segment_between)
    if neck_price <= 0:
        return False
    
    # Check drop from peaks to neckline
    min_peak = min(h1, h2)
    drop_pct = (min_peak - neck_price) / min_peak
    if drop_pct < MIN_DROP_PCT:
        return False
    
    # Confirmation: price must break below neckline after second peak
    if i2 + 1 < len(smooth):
        after_second = smooth[i2+1:]
        confirmed = np.any(after_second < neck_price * (1 - BREAK_BUFFER))
        if not confirmed:
            return False
    else:
        # No data after second peak
        return False
    
    return True


def build_double_top_dataset_for_symbol(
    input_csv_path: str,
    output_csv_path: str,
    symbol: str,
    window_bars: int = 300
) -> None:
    """
    Build a labeled dataset for double-top pattern detection for a single symbol.
    
    Args:
        input_csv_path: Path to input CSV file with columns: ID, TimeStamp, and stock symbols
        output_csv_path: Path where output CSV will be saved
        symbol: Stock symbol to process (e.g., "AAPL")
        window_bars: Size of sliding window in bars (default 300 for 15 minutes)
    """
    # Load CSV
    print(f"\n{'='*60}")
    print(f"Loading data from {input_csv_path}...")
    start_time = time.time()
    try:
        df = pd.read_csv(input_csv_path)
        load_time = time.time() - start_time
        print(f"✓ Loaded {len(df):,} rows in {load_time:.2f} seconds")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Parse TimeStamp as datetime
    if 'TimeStamp' not in df.columns:
        raise ValueError("CSV must contain a 'TimeStamp' column")
    
    print("Parsing timestamps and sorting data...")
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Sort by TimeStamp ascending
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    print(f"✓ Data sorted. Time range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
    
    # Validate symbol exists
    if symbol not in df.columns:
        available_symbols = [col for col in df.columns if col not in ['ID', 'TimeStamp']]
        raise ValueError(
            f"Symbol '{symbol}' not found in CSV. Available symbols: {available_symbols}"
        )
    
    # Extract price series for the symbol
    print(f"Extracting price series for {symbol}...")
    symbol_series = df[symbol].copy()
    
    # Check for missing values
    n_missing = symbol_series.isna().sum()
    if n_missing > 0:
        print(f"⚠ Warning: Found {n_missing:,} missing values in {symbol} series. Filling with forward fill.")
        symbol_series = symbol_series.ffill().bfill()
        print(f"✓ Missing values filled")
    
    # Create sliding windows and compute features
    n_windows = len(df) - window_bars + 1
    print(f"\n{'='*60}")
    print(f"Processing {n_windows:,} windows (window size: {window_bars} bars = 15 minutes)")
    print(f"Data range: {df['TimeStamp'].iloc[0]} to {df['TimeStamp'].iloc[-1]}")
    print(f"{'='*60}\n")
    
    rows = []
    double_top_count = 0
    process_start = time.time()
    
    # Create progress bar or use simple iteration
    window_range = range(window_bars - 1, len(df))
    if HAS_TQDM:
        pbar = tqdm(window_range, desc="Processing windows", unit="window", ncols=120)
        window_range = pbar
    else:
        pbar = None
    
    for end_idx in window_range:
        start_idx = end_idx - window_bars + 1
        
        # Extract window
        close_window = symbol_series.iloc[start_idx:end_idx+1]
        start_timestamp = df.iloc[start_idx]['TimeStamp']
        end_timestamp = df.iloc[end_idx]['TimeStamp']
        
        # Skip if window has insufficient data
        if len(close_window) < window_bars:
            continue
        
        # Compute features
        features = compute_window_features(close_window)
        
        # Compute label
        is_double_top = is_double_top_in_window(close_window, window_bars)
        label_double_top = 1 if is_double_top else 0
        
        if label_double_top == 1:
            double_top_count += 1
        
        # Create row
        row = {
            'symbol': symbol,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            **features,
            'label_double_top': label_double_top
        }
        rows.append(row)
        
        # Update progress bar with additional info (every 100 windows to avoid overhead)
        if pbar is not None and len(rows) % 100 == 0:
            elapsed = time.time() - process_start
            rate = len(rows) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'Double-tops': double_top_count,
                'Rate': f"{rate:.0f}/s"
            })
        
        # Progress indicator (only if not using tqdm)
        if not HAS_TQDM and len(rows) % 1000 == 0:
            elapsed = time.time() - process_start
            rate = len(rows) / elapsed if elapsed > 0 else 0
            remaining = (n_windows - len(rows)) / rate if rate > 0 else 0
            print(f"  Processed {len(rows):,}/{n_windows:,} windows ({len(rows)/n_windows*100:.1f}%) | "
                  f"Double-tops: {double_top_count} | "
                  f"Rate: {rate:.0f} windows/sec | "
                  f"ETA: {remaining/60:.1f} min")
    
    # Convert to DataFrame and save
    process_time = time.time() - process_start
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Processed {len(rows):,} windows in {process_time:.1f} seconds ({len(rows)/process_time:.0f} windows/sec)")
    print(f"  Double-top patterns found: {double_top_count:,} ({double_top_count/len(rows)*100:.2f}%)")
    print(f"Creating output DataFrame with {len(rows):,} rows...")
    output_df = pd.DataFrame(rows)
    
    # Ensure consistent column order
    feature_cols = [
        'close_last', 'close_mean', 'close_std', 'price_range_abs', 'price_range_pct',
        'cumulative_return_window', 'ret_1', 'ret_5', 'ret_20', 'momentum_last_30',
        'slope_entire_window', 'slope_last_30',
        'num_peaks_window', 'num_troughs_window', 'peak1_rel_pos', 'peak2_rel_pos',
        'bars_between_last_two_peaks', 'peak_height_diff_pct', 'neckline_drop_pct',
        'drawdown_from_last_peak', 'peak1_sharpness', 'peak2_sharpness',
        'rolling_std_20', 'rolling_std_60', 'true_range_mean_20',
        'sma_20', 'close_over_sma20', 'sma_50', 'rsi_14', 'percent_b'
    ]
    
    column_order = ['symbol', 'start_timestamp', 'end_timestamp'] + feature_cols + ['label_double_top']
    output_df = output_df[column_order]
    
    print(f"Saving to {output_csv_path}...")
    save_start = time.time()
    output_df.to_csv(output_csv_path, index=False)
    save_time = time.time() - save_start
    
    # Print summary statistics
    n_positive = output_df['label_double_top'].sum()
    pct_positive = 100 * n_positive / len(output_df) if len(output_df) > 0 else 0
    total_time = time.time() - start_time
    
    print(f"✓ Saved in {save_time:.2f} seconds")
    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"  Total windows: {len(output_df):,}")
    print(f"  Double-top patterns: {n_positive:,} ({pct_positive:.2f}%)")
    print(f"  Non-patterns: {len(output_df) - n_positive:,} ({100 - pct_positive:.2f}%)")
    print(f"\n  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Output file: {output_csv_path}")
    print(f"{'='*60}\n")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Build labeled dataset for double-top pattern detection'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/combined_dataset.csv',
        help='Path to input CSV file (default: data/combined_dataset.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: {symbol}_double_top_15m_windows.csv)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Stock symbol to process (e.g., AAPL, MSFT, NVDA)'
    )
    parser.add_argument(
        '--window-bars',
        type=int,
        default=300,
        help='Size of sliding window in bars (default: 300 for 15 minutes at 3s per bar)'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f"{args.symbol}_double_top_15m_windows.csv"
    
    # Run the dataset builder
    build_double_top_dataset_for_symbol(
        input_csv_path=args.input,
        output_csv_path=args.output,
        symbol=args.symbol,
        window_bars=args.window_bars
    )


if __name__ == "__main__":
    main()

