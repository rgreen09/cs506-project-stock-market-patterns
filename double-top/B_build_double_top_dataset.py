"""
Double-Top Pattern Dataset Builder

This script processes intraday stock data to build a labeled dataset for detecting
double-top patterns using sliding 15-minute windows (300 bars at 3-second intervals).

Usage:
    Minimal (uses defaults):
        python B_build_double_top_dataset.py --symbol AAPL
    
    Full example:
        python B_build_double_top_dataset.py --input data/combined_dataset.csv --output AAPL_double_top_15m_windows.csv --symbol AAPL --window-bars 300
    
    With custom trading hours:
        python B_build_double_top_dataset.py --symbol AAPL --market-open 09:30 --market-close 16:00
    
    Disable filtering:
        python B_build_double_top_dataset.py --symbol AAPL --no-filter-trading-hours --no-filter-day-boundaries

Arguments:
    --symbol (required): Stock symbol to process (e.g., AAPL, MSFT, NVDA)
    --input (optional): Path to input CSV file (default: data/combined_dataset.csv)
    --output (optional): Path to output CSV file (default: {symbol}_double_top_15m_windows.csv)
    --window-bars (optional): Size of sliding window in bars (default: 300)
    --filter-trading-hours (default: True): Filter out windows outside trading hours
    --no-filter-trading-hours: Disable trading hours filtering
    --market-open (optional): Market open time in HH:MM format (default: 09:30)
    --market-close (optional): Market close time in HH:MM format (default: 16:00)
    --filter-day-boundaries (default: True): Filter out windows spanning multiple days
    --no-filter-day-boundaries: Disable day boundary filtering
"""

import argparse
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import time as dt_time
import warnings
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bar.")

from A_double_top_detector import (
    is_double_top_in_window,
    DoubleTopDetector,
    DoubleTopConfig
)

# Import helper functions for feature computation
find_local_peaks = DoubleTopDetector.find_local_peaks
find_local_troughs = DoubleTopDetector.find_local_troughs


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


def is_within_trading_hours(
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    market_open: str = '09:30',
    market_close: str = '16:00'
) -> bool:
    """
    Check if a window is within trading hours.
    
    Args:
        start_timestamp: Start timestamp of the window
        end_timestamp: End timestamp of the window
        market_open: Market open time in HH:MM format (default: '09:30')
        market_close: Market close time in HH:MM format (default: '16:00')
        
    Returns:
        True if both start and end timestamps are within trading hours on the same day
    """
    # Extract date and time components
    start_date = start_timestamp.date()
    end_date = end_timestamp.date()
    
    # Must be on the same day
    if start_date != end_date:
        return False
    
    # Parse market hours
    open_hour, open_minute = map(int, market_open.split(':'))
    close_hour, close_minute = map(int, market_close.split(':'))
    
    # Get time components
    start_time = start_timestamp.time()
    end_time = end_timestamp.time()
    
    # Create time objects for market hours
    market_open_time = dt_time(open_hour, open_minute)
    market_close_time = dt_time(close_hour, close_minute)
    
    # Check if both timestamps are within trading hours
    start_in_hours = market_open_time <= start_time <= market_close_time
    end_in_hours = market_open_time <= end_time <= market_close_time
    
    return start_in_hours and end_in_hours


def spans_multiple_days(
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp
) -> bool:
    """
    Check if a window spans multiple calendar days.
    
    Args:
        start_timestamp: Start timestamp of the window
        end_timestamp: End timestamp of the window
        
    Returns:
        True if timestamps are on different calendar days
    """
    return start_timestamp.date() != end_timestamp.date()


def build_double_top_dataset_for_symbol(
    input_csv_path: str,
    output_csv_path: str,
    symbol: str,
    window_bars: int = 300,
    filter_trading_hours: bool = True,
    market_open: str = '09:30',
    market_close: str = '16:00',
    filter_day_boundaries: bool = True
) -> None:
    """
    Build a labeled dataset for double-top pattern detection for a single symbol.
    
    Args:
        input_csv_path: Path to input CSV file with columns: ID, TimeStamp, and stock symbols
        output_csv_path: Path where output CSV will be saved
        symbol: Stock symbol to process (e.g., "AAPL")
        window_bars: Size of sliding window in bars (default 300 for 15 minutes)
        filter_trading_hours: If True, filter out windows outside trading hours (default: True)
        market_open: Market open time in HH:MM format (default: '09:30')
        market_close: Market close time in HH:MM format (default: '16:00')
        filter_day_boundaries: If True, filter out windows spanning multiple days (default: True)
    """
    # Load CSV
    print(f"Loading data from {input_csv_path}...")
    start_time = time.time()
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Parse TimeStamp as datetime
    if 'TimeStamp' not in df.columns:
        raise ValueError("CSV must contain a 'TimeStamp' column")
    
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)
    
    # Validate symbol exists
    if symbol not in df.columns:
        available_symbols = [col for col in df.columns if col not in ['ID', 'TimeStamp']]
        raise ValueError(
            f"Symbol '{symbol}' not found in CSV. Available symbols: {available_symbols}"
        )
    
    # Extract price series for the symbol
    symbol_series = df[symbol].copy()
    
    # Check for missing values
    n_missing = symbol_series.isna().sum()
    if n_missing > 0:
        symbol_series = symbol_series.ffill().bfill()
    
    # Pre-filter data to trading hours and single-day windows to reduce iterations
    # This dramatically reduces the number of windows we need to check
    if filter_trading_hours or filter_day_boundaries:
        print("Pre-filtering data to trading hours and single-day windows...")
        original_len = len(df)
        
        # Parse market hours
        open_hour, open_minute = map(int, market_open.split(':'))
        close_hour, close_minute = map(int, market_close.split(':'))
        market_open_time = dt_time(open_hour, open_minute)
        market_close_time = dt_time(close_hour, close_minute)
        
        # Create mask for rows to keep
        mask = pd.Series(True, index=df.index)
        
        if filter_trading_hours:
            # Keep only rows within trading hours
            df_time = df['TimeStamp'].dt.time
            mask = mask & (df_time >= market_open_time) & (df_time <= market_close_time)
        
        if filter_day_boundaries:
            # Filter out rows that are within window_bars of a day boundary
            # This prevents windows from spanning multiple days
            df['date'] = df['TimeStamp'].dt.date
            date_changes = df['date'] != df['date'].shift(1)
            # Find indices where date changes (first row of each new day)
            date_change_indices = df[date_changes].index.tolist()
            
            # Create mask to exclude rows within window_bars of day boundaries
            exclude_mask = pd.Series(False, index=df.index)
            for change_idx in date_change_indices:
                if change_idx > 0:  # Skip first row (no previous day)
                    # Exclude window_bars rows before and after each day boundary
                    start_exclude = max(0, change_idx - window_bars)
                    end_exclude = min(len(df), change_idx + window_bars)
                    exclude_mask.iloc[start_exclude:end_exclude] = True
            
            mask = mask & ~exclude_mask
        
        # Apply mask
        df = df[mask].reset_index(drop=True)
        
        # Drop temporary date column if it was added
        if filter_day_boundaries and 'date' in df.columns:
            df = df.drop(columns=['date'])
        symbol_series = df[symbol].copy()
        
        # Re-check for missing values after filtering
        n_missing = symbol_series.isna().sum()
        if n_missing > 0:
            symbol_series = symbol_series.ffill().bfill()
        
        filtered_rows = original_len - len(df)
        print(f"  Filtered out {filtered_rows:,} rows ({100*filtered_rows/original_len:.1f}%)")
        print(f"  Remaining: {len(df):,} rows")
    
    # Group data by day to create windows only within each day
    # This ensures no windows span day boundaries and reduces redundant checks
    if filter_day_boundaries:
        df['date'] = df['TimeStamp'].dt.date
        grouped_by_day = df.groupby('date')
        print(f"Data grouped into {len(grouped_by_day)} trading days")
        # Calculate total windows more accurately (sum of windows per day)
        n_windows = sum(max(0, len(group) - window_bars + 1) for _, group in grouped_by_day)
    else:
        # If not filtering day boundaries, treat all data as one group
        # Create a list of tuples to match the groupby interface
        grouped_by_day = [(None, df)]
        n_windows = len(df) - window_bars + 1
    
    print(f"Processing {n_windows:,} windows for {symbol}...")
    
    # Initialize CSV file with header (write incrementally to avoid memory issues)
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
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
    
    # Batch writing to reduce I/O overhead while keeping memory bounded
    BATCH_SIZE = 1000
    rows_batch = []
    total_rows_written = 0
    double_top_count = 0
    filtered_trading_hours = 0
    filtered_day_boundaries = 0
    process_start = time.time()
    
    # Create progress bar or use simple iteration
    if HAS_TQDM:
        pbar = tqdm(
            total=n_windows,
            desc="Processing windows",
            unit="window",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            miniters=1,
            mininterval=0.1
        )
    else:
        pbar = None
    
    # Process windows grouped by day to avoid day boundary issues
    for date, day_df in grouped_by_day:
        if len(day_df) < window_bars:
            # Skip days with insufficient data
            continue
        
        # Get symbol series for this day
        day_symbol_series = day_df[symbol].copy()
        
        # Create windows only within this day
        day_window_range = range(window_bars - 1, len(day_df))
        
        for end_idx in day_window_range:
            start_idx = end_idx - window_bars + 1
            
            # Extract window
            close_window = day_symbol_series.iloc[start_idx:end_idx+1]
            start_timestamp = day_df.iloc[start_idx]['TimeStamp']
            end_timestamp = day_df.iloc[end_idx]['TimeStamp']
            
            # Skip if window has insufficient data (shouldn't happen, but safety check)
            if len(close_window) < window_bars:
                if pbar is not None:
                    pbar.update(1)
                continue
            
            # Since we grouped by day, windows can't span multiple days
            # But we still check trading hours as a safety check (in case pre-filtering missed something)
            if filter_trading_hours and not is_within_trading_hours(start_timestamp, end_timestamp, market_open, market_close):
                filtered_trading_hours += 1
                if pbar is not None:
                    pbar.update(1)
                continue
            
            # Compute features
            features = compute_window_features(close_window)
            
            # Compute label - use relaxed parameters optimized for intraday 3-second data
            # Disable confirmation requirement since we're using sliding windows
            # and patterns may not have time to confirm within the window
            config = DoubleTopConfig(
                peak_tolerance=0.04,        # Relaxed: 4% tolerance (was 2%)
                min_drop_pct=0.002,         # Relaxed: 0.2% minimum drop (was 0.5%)
                min_gap=5,                  # Reduced: 5 bars = 15 seconds (was 10 bars = 30 seconds)
                max_gap_ratio=0.6,          # Increased: 60% of window (was 50%)
                require_confirmation=False,  # Disable for sliding windows
                smoothing_window=7          # Slightly more smoothing for noise reduction
            )
            detector = DoubleTopDetector(config)
            is_double_top = detector.detect_simple(close_window, window_bars)
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
            rows_batch.append(row)
            
            # Write batch to CSV when it reaches BATCH_SIZE
            if len(rows_batch) >= BATCH_SIZE:
                with open(output_csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=column_order)
                    for r in rows_batch:
                        writer.writerow(r)
                total_rows_written += len(rows_batch)
                rows_batch = []
            
            # Update progress bar frequently
            if pbar is not None:
                elapsed = time.time() - process_start
                rate = (total_rows_written + len(rows_batch)) / elapsed if elapsed > 0 else 0
                pbar.update(1)
                postfix = {
                    'Double-tops': double_top_count,
                    'Kept': total_rows_written + len(rows_batch),
                    'Rate': f"{rate:.0f}/s"
                }
                if filter_trading_hours or filter_day_boundaries:
                    postfix['Filtered'] = f"{filtered_trading_hours + filtered_day_boundaries}"
                pbar.set_postfix(postfix)
            
            # Progress indicator (only if not using tqdm)
            if not HAS_TQDM and (total_rows_written + len(rows_batch)) % 1000 == 0:
                elapsed = time.time() - process_start
                rate = (total_rows_written + len(rows_batch)) / elapsed if elapsed > 0 else 0
                # Estimate windows processed (approximate)
                windows_processed = total_rows_written + len(rows_batch) + filtered_trading_hours + filtered_day_boundaries
                remaining = (n_windows - windows_processed) / (windows_processed / elapsed) if windows_processed > 0 else 0
                filtered_total = filtered_trading_hours + filtered_day_boundaries
                print(f"  Kept {total_rows_written + len(rows_batch):,} windows | "
                      f"Double-tops: {double_top_count} | "
                      f"Filtered: {filtered_total:,} | "
                      f"Rate: {rate:.0f} windows/sec | "
                      f"ETA: {remaining/60:.1f} min")
    
    # Write remaining rows in batch
    if rows_batch:
        with open(output_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            for r in rows_batch:
                writer.writerow(r)
        total_rows_written += len(rows_batch)
    
    if pbar is not None:
        pbar.close()
    
    # Load final DataFrame for statistics (only for summary, then release)
    process_time = time.time() - process_start
    output_df = pd.read_csv(output_csv_path)
    
    # Print summary statistics
    n_positive = output_df['label_double_top'].sum()
    pct_positive = 100 * n_positive / len(output_df) if len(output_df) > 0 else 0
    total_time = time.time() - start_time
    
    total_filtered = filtered_trading_hours + filtered_day_boundaries
    pct_filtered = 100 * total_filtered / n_windows if n_windows > 0 else 0
    pct_kept = 100 * len(output_df) / n_windows if n_windows > 0 else 0
    
    print(f"\nComplete: {len(output_df):,} windows kept | Double-tops: {n_positive:,} ({pct_positive:.2f}%) | Time: {total_time:.1f}s")
    if filter_trading_hours or filter_day_boundaries:
        print(f"Filtering: {total_filtered:,} windows filtered ({pct_filtered:.2f}%)")
        if filter_trading_hours:
            pct_th = 100 * filtered_trading_hours / n_windows if n_windows > 0 else 0
            print(f"  - Outside trading hours: {filtered_trading_hours:,} ({pct_th:.2f}%)")
        if filter_day_boundaries:
            pct_db = 100 * filtered_day_boundaries / n_windows if n_windows > 0 else 0
            print(f"  - Spans multiple days: {filtered_day_boundaries:,} ({pct_db:.2f}%)")
        print(f"Kept: {len(output_df):,} windows ({pct_kept:.2f}% of {n_windows:,} total)")
    print(f"Output: {output_csv_path}")


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
    parser.add_argument(
        '--filter-trading-hours',
        action='store_true',
        default=True,
        help='Filter out windows outside trading hours (default: True)'
    )
    parser.add_argument(
        '--no-filter-trading-hours',
        dest='filter_trading_hours',
        action='store_false',
        help='Disable filtering of windows outside trading hours'
    )
    parser.add_argument(
        '--market-open',
        type=str,
        default='09:30',
        help='Market open time in HH:MM format (default: 09:30)'
    )
    parser.add_argument(
        '--market-close',
        type=str,
        default='16:00',
        help='Market close time in HH:MM format (default: 16:00)'
    )
    parser.add_argument(
        '--filter-day-boundaries',
        action='store_true',
        default=True,
        help='Filter out windows spanning multiple days (default: True)'
    )
    parser.add_argument(
        '--no-filter-day-boundaries',
        dest='filter_day_boundaries',
        action='store_false',
        help='Disable filtering of windows spanning multiple days'
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
        window_bars=args.window_bars,
        filter_trading_hours=args.filter_trading_hours,
        market_open=args.market_open,
        market_close=args.market_close,
        filter_day_boundaries=args.filter_day_boundaries
    )


if __name__ == "__main__":
    main()

