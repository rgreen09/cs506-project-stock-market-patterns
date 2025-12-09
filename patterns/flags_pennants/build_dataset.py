"""
Flags and Pennants Pattern Dataset Builder --

This script processes intraday stock data to build a labeled dataset for detecting
flag and pennant continuation patterns using sliding 15-minute windows (300 bars at 3-second intervals).

Usage:
    Minimal (uses defaults):
        python build_flags_pennants_dataset.py --symbol AAPL
    
    Full example:
        python build_flags_pennants_dataset.py --input data/combined_dataset.csv --output AAPL_flags_pennants_15m_windows.csv --symbol AAPL --window-bars 300
    
    With custom trading hours:
        python build_flags_pennants_dataset.py --symbol AAPL --market-open 09:30 --market-close 16:00
    
    Disable filtering:
        python build_flags_pennants_dataset.py --symbol AAPL --no-filter-trading-hours --no-filter-day-boundaries

Arguments:
    --symbol (required): Stock symbol to process (e.g., AAPL, MSFT, NVDA)
    --input (optional): Path to input CSV file (default: data/combined_dataset.csv)
    --output (optional): Path to output CSV file (default: {symbol}_flags_pennants_15m_windows.csv)
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
from typing import Dict
from datetime import time as dt_time
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bar.")

from Flags_pennants_detector import (
    is_flags_pennants_in_window,
    FlagsPennantsDetector,
    FlagsPennantsConfig
)


def calculate_linear_trend(prices: np.ndarray) -> float:
    """Calculate linear regression slope for price array."""
    n = len(prices)
    if n < 2:
        return 0.0
    x_axis = np.arange(n)
    slope = np.polyfit(x_axis, prices, 1)[0]
    return float(slope)


def find_strongest_move(prices: np.ndarray, max_bars: int = 100) -> Dict[str, float]:
    """
    Find the strongest directional price move in the window.
    
    Returns:
        Dictionary with move statistics
    """
    n = len(prices)
    if n < 3:
        return {
            'max_move_pct': 0.0,
            'max_move_bars': 0,
            'max_move_direction': 0  # 1 for up, -1 for down
        }
    
    max_move_pct = 0.0
    max_move_bars = 0
    max_move_direction = 0
    
    # Search for strongest consecutive move
    for start_idx in range(n - 2):
        for end_idx in range(start_idx + 2, min(n, start_idx + max_bars)):
            if prices[start_idx] == 0:
                continue
            
            move_pct = (prices[end_idx] - prices[start_idx]) / prices[start_idx]
            abs_move = abs(move_pct)
            
            if abs_move > abs(max_move_pct):
                max_move_pct = move_pct
                max_move_bars = end_idx - start_idx
                max_move_direction = 1 if move_pct > 0 else -1
    
    return {
        'max_move_pct': float(max_move_pct),
        'max_move_bars': float(max_move_bars),
        'max_move_direction': float(max_move_direction)
    }


def compute_consolidation_features(prices: np.ndarray, pole_end: int) -> Dict[str, float]:
    """
    Compute features related to consolidation patterns.
    
    Args:
        prices: Price array
        pole_end: Index where pole ends
        
    Returns:
        Dictionary of consolidation features
    """
    n = len(prices)
    
    # Default values if consolidation analysis not possible
    if pole_end >= n - 10 or pole_end < 10:
        return {
            'consolidation_range_pct': 0.0,
            'consolidation_volatility': 0.0,
            'consolidation_slope': 0.0,
            'price_compression_ratio': 0.0
        }
    
    # Analyze period after pole
    after_pole = prices[pole_end:]
    
    if len(after_pole) < 10:
        return {
            'consolidation_range_pct': 0.0,
            'consolidation_volatility': 0.0,
            'consolidation_slope': 0.0,
            'price_compression_ratio': 0.0
        }
    
    # Take up to 150 bars for consolidation analysis
    consolidation_segment = after_pole[:min(150, len(after_pole))]
    
    # Calculate range during consolidation
    cons_min = np.min(consolidation_segment)
    cons_max = np.max(consolidation_segment)
    cons_range = cons_max - cons_min
    
    # Normalize by pole end price
    pole_price = prices[pole_end]
    range_pct = cons_range / pole_price if pole_price > 0 else 0.0
    
    # Calculate volatility during consolidation
    cons_volatility = np.std(consolidation_segment)
    
    # Calculate slope during consolidation
    cons_slope = calculate_linear_trend(consolidation_segment)
    
    # Calculate compression ratio (first third vs last third)
    n_cons = len(consolidation_segment)
    if n_cons >= 30:
        first_third = consolidation_segment[:n_cons//3]
        last_third = consolidation_segment[-n_cons//3:]
        
        first_range = np.max(first_third) - np.min(first_third)
        last_range = np.max(last_third) - np.min(last_third)
        
        compression_ratio = last_range / first_range if first_range > 0 else 1.0
    else:
        compression_ratio = 1.0
    
    return {
        'consolidation_range_pct': float(range_pct),
        'consolidation_volatility': float(cons_volatility),
        'consolidation_slope': float(cons_slope),
        'price_compression_ratio': float(compression_ratio)
    }


def compute_window_features(close_window: pd.Series) -> Dict[str, float]:
    """
    Compute 35 engineered features from a window of closing prices.
    Features designed specifically for flag and pennant pattern detection.
    
    Args:
        close_window: pandas Series of closing prices (length should be 300)
        
    Returns:
        Dictionary mapping feature names to their values
    """
    c = close_window.values
    n = len(c)
    
    # Create smoothed price series
    smooth_series = pd.Series(c).rolling(window=5, center=True, min_periods=1).mean()
    smooth = smooth_series.values
    
    # Initialize feature dictionary
    features = {}
    
    # ===== Core price statistics (5 features) =====
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
    features['slope_entire_window'] = float(calculate_linear_trend(c))
    
    # Slope of last 30 points
    if n >= 30:
        features['slope_last_30'] = float(calculate_linear_trend(c[-30:]))
    else:
        features['slope_last_30'] = float(calculate_linear_trend(c) if n > 1 else 0.0)
    
    # ===== Pole detection features (10 features) =====
    # Find strongest directional move in window
    move_stats = find_strongest_move(smooth, max_bars=100)
    features['max_move_pct'] = move_stats['max_move_pct']
    features['max_move_bars'] = move_stats['max_move_bars']
    features['max_move_direction'] = move_stats['max_move_direction']
    
    # Calculate pole strength relative to window
    abs_max_move = abs(move_stats['max_move_pct'])
    features['pole_strength_ratio'] = float(abs_max_move / features['price_range_pct'] if features['price_range_pct'] > 0 else 0.0)
    
    # Directional momentum in different sections
    if n >= 100:
        first_third = c[:n//3]
        middle_third = c[n//3:2*n//3]
        last_third = c[2*n//3:]
        
        ret_first = (first_third[-1] / first_third[0] - 1) if first_third[0] != 0 else 0.0
        ret_middle = (middle_third[-1] / middle_third[0] - 1) if middle_third[0] != 0 else 0.0
        ret_last = (last_third[-1] / last_third[0] - 1) if last_third[0] != 0 else 0.0
        
        features['return_first_third'] = float(ret_first)
        features['return_middle_third'] = float(ret_middle)
        features['return_last_third'] = float(ret_last)
    else:
        features['return_first_third'] = 0.0
        features['return_middle_third'] = 0.0
        features['return_last_third'] = 0.0
    
    # Estimate pole end position (where strongest move ends)
    if move_stats['max_move_bars'] > 0:
        # Simplified: assume pole is in first half of window
        estimated_pole_end = min(int(move_stats['max_move_bars']), n//2)
    else:
        estimated_pole_end = n//3
    
    features['estimated_pole_end_idx'] = float(estimated_pole_end)
    features['estimated_pole_end_rel_pos'] = float(estimated_pole_end / n if n > 0 else 0.0)
    
    # Pole velocity (move per bar)
    features['pole_velocity'] = float(
        move_stats['max_move_pct'] / move_stats['max_move_bars'] 
        if move_stats['max_move_bars'] > 0 else 0.0
    )
    
    # ===== Consolidation pattern features (8 features) =====
    # Analyze consolidation characteristics
    cons_features = compute_consolidation_features(smooth, estimated_pole_end)
    features.update(cons_features)
    
    # Additional consolidation metrics
    if estimated_pole_end < n - 10:
        after_pole_segment = c[estimated_pole_end:]
        
        # Check if price stays in range after pole
        pole_end_price = c[estimated_pole_end]
        max_deviation_pct = np.max(np.abs(after_pole_segment - pole_end_price)) / pole_end_price if pole_end_price > 0 else 0.0
        features['max_deviation_after_pole_pct'] = float(max_deviation_pct)
        
        # Average distance from pole end price
        avg_deviation_pct = np.mean(np.abs(after_pole_segment - pole_end_price)) / pole_end_price if pole_end_price > 0 else 0.0
        features['avg_deviation_after_pole_pct'] = float(avg_deviation_pct)
        
        # Trend consistency after pole
        after_pole_slope = calculate_linear_trend(after_pole_segment)
        features['slope_after_pole'] = float(after_pole_slope)
        
        # Compare volatility before and after pole
        before_pole_vol = np.std(c[:estimated_pole_end]) if estimated_pole_end > 1 else 0.0
        after_pole_vol = np.std(after_pole_segment)
        vol_ratio = after_pole_vol / before_pole_vol if before_pole_vol > 0 else 1.0
        features['volatility_ratio_after_pole'] = float(vol_ratio)
    else:
        features['max_deviation_after_pole_pct'] = 0.0
        features['avg_deviation_after_pole_pct'] = 0.0
        features['slope_after_pole'] = 0.0
        features['volatility_ratio_after_pole'] = 1.0
    
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
        delta = np.diff(c[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        features['rsi_14'] = float(rsi)
    else:
        features['rsi_14'] = 50.0
    
    # MACD-like feature (fast MA - slow MA)
    if n >= 26:
        ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().iloc[-1]
        features['macd_line'] = float(ema_fast - ema_slow)
    else:
        features['macd_line'] = 0.0
    
    return features


def is_within_trading_hours(
    start_timestamp: str,
    end_timestamp: str,
    market_open: str = '09:30',
    market_close: str = '16:00'
) -> bool:
    """
    Check if a window falls within regular trading hours.
    
    Args:
        start_timestamp: ISO format timestamp for window start
        end_timestamp: ISO format timestamp for window end
        market_open: Market open time in HH:MM format
        market_close: Market close time in HH:MM format
        
    Returns:
        True if entire window is within trading hours, False otherwise
    """
    # Parse timestamps
    start_dt = pd.to_datetime(start_timestamp)
    end_dt = pd.to_datetime(end_timestamp)
    
    # Parse market hours
    open_parts = market_open.split(':')
    open_time = dt_time(int(open_parts[0]), int(open_parts[1]))
    close_parts = market_close.split(':')
    close_time = dt_time(int(close_parts[0]), int(close_parts[1]))
    
    # Check if both start and end are within market hours
    start_time = start_dt.time()
    end_time = end_dt.time()
    
    return (start_time >= open_time and end_time <= close_time)


def build_flags_pennants_dataset_for_symbol(
    input_csv_path: str,
    output_csv_path: str,
    symbol: str,
    window_bars: int = 300,
    sample_rate: int = 1,
    filter_trading_hours: bool = True,
    market_open: str = '09:30',
    market_close: str = '16:00',
    filter_day_boundaries: bool = True
):
    """
    Build a labeled dataset for a single symbol by sliding a window and computing features.
    
    Args:
        input_csv_path: Path to input CSV with columns [timestamp, symbol, close]
        output_csv_path: Path where output CSV will be written
        symbol: Stock symbol to process
        window_bars: Size of sliding window in number of bars
        filter_trading_hours: If True, exclude windows outside trading hours
        market_open: Market open time in HH:MM format
        market_close: Market close time in HH:MM format
        filter_day_boundaries: If True, exclude windows spanning multiple days
    """
    start_time = time.time()
    
    print(f"Loading data for {symbol}...")
    df = pd.read_csv(input_csv_path)

    # Check if data is in wide format (symbol as column) or long format (symbol as row value)
    if 'symbol' in df.columns:
        # Long format: filter for the specified symbol
        df = df[df['symbol'] == symbol].copy()

        if len(df) == 0:
            print(f"Error: No data found for symbol '{symbol}'")
            return

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        # Wide format: convert to long format for the specified symbol
        if symbol not in df.columns:
            print(f"Error: Symbol '{symbol}' not found in dataset columns")
            available_symbols = [col for col in df.columns if col not in ['ID', 'TimeStamp', 'timestamp', 'Date', 'Time']]
            print(f"Available symbols: {', '.join(available_symbols)}")
            return

        # Create long format DataFrame
        timestamp_col = 'TimeStamp' if 'TimeStamp' in df.columns else 'timestamp'
        df = pd.DataFrame({
            'timestamp': df[timestamp_col],
            'symbol': symbol,
            'close': df[symbol]
        })

        # Remove rows where close price is NaN
        df = df.dropna(subset=['close']).reset_index(drop=True)

        if len(df) == 0:
            print(f"Error: No valid data found for symbol '{symbol}'")
            return

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Found {len(df):,} records for {symbol}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Window size: {window_bars} bars (15 minutes at 3-second intervals)")
    if sample_rate > 1:
        print(f"Sampling: Processing every {sample_rate} window(s) for faster execution")
    
    # Pre-filter data by trading hours if enabled
    if filter_trading_hours:
        print(f"Pre-filtering data to trading hours ({market_open} - {market_close})...")
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['time'] = df['datetime'].dt.time

        open_parts = market_open.split(':')
        open_time = dt_time(int(open_parts[0]), int(open_parts[1]))
        close_parts = market_close.split(':')
        close_time = dt_time(int(close_parts[0]), int(close_parts[1]))
        
        mask_trading_hours = (df['time'] >= open_time) & (df['time'] <= close_time)
        df = df[mask_trading_hours].reset_index(drop=True)
        
        print(f"After trading hours filter: {len(df):,} records remaining")
        
        if len(df) == 0:
            print("Error: No data remaining after trading hours filter")
            return
    
    # Calculate total possible windows
    n_windows = len(df) - window_bars + 1
    print(f"Total possible windows: {n_windows:,}")
    
    if n_windows <= 0:
        print("Error: Not enough data for even one window")
        return
    
    # Define column order for output CSV
    sample_features = compute_window_features(pd.Series(df['close'].iloc[:window_bars]))
    column_order = [
        'symbol',
        'start_timestamp',
        'end_timestamp'
    ] + list(sample_features.keys()) + ['label_flags_pennants']
    
    # Write CSV header
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
    
    # Process windows in batches
    BATCH_SIZE = 100
    rows_batch = []
    total_rows_written = 0
    flags_pennants_count = 0
    filtered_trading_hours = 0
    filtered_day_boundaries = 0
    
    print(f"\nProcessing windows...")
    process_start = time.time()
    
    # Create progress bar if tqdm available
    if HAS_TQDM:
        pbar = tqdm(total=n_windows, desc="Processing", unit="windows")
    else:
        pbar = None
        print(f"Processing {n_windows:,} windows (progress updates every 1000 windows)...")
    
    # Slide window
    for i in range(n_windows):
        # Skip windows based on sample_rate
        if i % sample_rate != 0:
            if pbar is not None:
                pbar.update(1)
            continue

        window_df = df.iloc[i:i+window_bars]
        close_window = window_df['close']

        start_timestamp = window_df.iloc[0]['timestamp']
        end_timestamp = window_df.iloc[-1]['timestamp']

        # Check if window spans multiple days
        if filter_day_boundaries:
            start_date = pd.to_datetime(start_timestamp).date()
            end_date = pd.to_datetime(end_timestamp).date()
            
            if start_date != end_date:
                filtered_day_boundaries += 1
                if pbar is not None:
                    pbar.update(1)
                continue
        
        # Double-check trading hours (safety check)
        if filter_trading_hours and not is_within_trading_hours(start_timestamp, end_timestamp, market_open, market_close):
            filtered_trading_hours += 1
            if pbar is not None:
                pbar.update(1)
            continue
        
        # Compute features
        features = compute_window_features(close_window)
        
        # Compute label - use relaxed parameters for intraday 3-second data
        config = FlagsPennantsConfig(
            pole_min_move_pct=0.015,         # Relaxed: 1.5% minimum pole (was 2%)
            pole_max_bars=120,               # Extended: allow up to 120 bars for pole
            consolidation_min_bars=8,        # Reduced: 8 bars = 24 seconds
            consolidation_max_bars=180,      # Extended: allow longer consolidation
            consolidation_max_range_pct=0.015,  # Relaxed: 1.5% max range (was 1%)
            flag_slope_tolerance=0.4,        # More tolerant of slope
            pennant_converge_ratio=0.65,     # Slightly relaxed convergence requirement
            breakout_threshold_pct=0.003,    # Reduced: 0.3% breakout threshold
            require_breakout=False,          # Disable for sliding windows
            smoothing_window=7               # Slightly more smoothing
        )
        detector = FlagsPennantsDetector(config)
        is_flags_pennants = detector.detect_simple(close_window, window_bars)
        label_flags_pennants = 1 if is_flags_pennants else 0
        
        if label_flags_pennants == 1:
            flags_pennants_count += 1
        
        # Create row
        row = {
            'symbol': symbol,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            **features,
            'label_flags_pennants': label_flags_pennants
        }
        rows_batch.append(row)
        
        # Write batch to CSV
        if len(rows_batch) >= BATCH_SIZE:
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=column_order)
                for r in rows_batch:
                    writer.writerow(r)
            total_rows_written += len(rows_batch)
            rows_batch = []
        
        # Update progress bar
        if pbar is not None:
            elapsed = time.time() - process_start
            rate = (total_rows_written + len(rows_batch)) / elapsed if elapsed > 0 else 0
            pbar.update(1)
            postfix = {
                'Flags/Pennants': flags_pennants_count,
                'Kept': total_rows_written + len(rows_batch),
                'Rate': f"{rate:.0f}/s"
            }
            if filter_trading_hours or filter_day_boundaries:
                postfix['Filtered'] = f"{filtered_trading_hours + filtered_day_boundaries}"
            pbar.set_postfix(postfix)
        
        # Progress indicator without tqdm
        if not HAS_TQDM and (total_rows_written + len(rows_batch)) % 1000 == 0:
            elapsed = time.time() - process_start
            rate = (total_rows_written + len(rows_batch)) / elapsed if elapsed > 0 else 0
            windows_processed = total_rows_written + len(rows_batch) + filtered_trading_hours + filtered_day_boundaries
            remaining = (n_windows - windows_processed) / (windows_processed / elapsed) if windows_processed > 0 else 0
            filtered_total = filtered_trading_hours + filtered_day_boundaries
            print(f"  Kept {total_rows_written + len(rows_batch):,} windows | "
                  f"Flags/Pennants: {flags_pennants_count} | "
                  f"Filtered: {filtered_total:,} | "
                  f"Rate: {rate:.0f} windows/sec | "
                  f"ETA: {remaining/60:.1f} min")
    
    # Write remaining rows
    if rows_batch:
        with open(output_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            for r in rows_batch:
                writer.writerow(r)
        total_rows_written += len(rows_batch)
    
    if pbar is not None:
        pbar.close()
    
    # Load final DataFrame for statistics
    process_time = time.time() - process_start
    output_df = pd.read_csv(output_csv_path)
    
    # Print summary statistics
    n_positive = output_df['label_flags_pennants'].sum()
    pct_positive = 100 * n_positive / len(output_df) if len(output_df) > 0 else 0
    total_time = time.time() - start_time
    
    total_filtered = filtered_trading_hours + filtered_day_boundaries
    pct_filtered = 100 * total_filtered / n_windows if n_windows > 0 else 0
    pct_kept = 100 * len(output_df) / n_windows if n_windows > 0 else 0
    
    print(f"\nComplete: {len(output_df):,} windows kept | Flags/Pennants: {n_positive:,} ({pct_positive:.2f}%) | Time: {total_time:.1f}s")
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
        description='Build labeled dataset for flags and pennants pattern detection'
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
        help='Path to output CSV file (default: {symbol}_flags_pennants_15m_windows.csv)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=False,
        default=None,
        help='Stock symbol to process (e.g., AAPL, MSFT, NVDA)'
    )
    parser.add_argument(
        '--window-bars',
        type=int,
        default=300,
        help='Size of sliding window in bars (default: 300 for 15 minutes at 3s per bar)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=1,
        help='Process every Nth window (default: 1 for all windows, use 10-100 for faster processing)'
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

    # If no symbol provided, list available symbols
    if args.symbol is None:
        print("No symbol specified. Checking available symbols in the dataset...")
        try:
            df = pd.read_csv(args.input)
            if 'symbol' in df.columns:
                symbols = sorted(df['symbol'].unique())
                print(f"\nAvailable symbols in {args.input}:")
                for sym in symbols:
                    count = len(df[df['symbol'] == sym])
                    print(f"  - {sym}: {count:,} records")
            else:
                # Handle wide format (columns are symbol names)
                exclude_cols = ['ID', 'TimeStamp', 'timestamp', 'Date', 'Time']
                symbols = [col for col in df.columns if col not in exclude_cols]
                if symbols:
                    print(f"\nAvailable symbols in {args.input}:")
                    for sym in symbols:
                        non_null_count = df[sym].notna().sum()
                        print(f"  - {sym}: {non_null_count:,} records")
                else:
                    print(f"Error: Could not identify symbol columns in {args.input}")
                    return

            print("\nPlease run the script with --symbol <SYMBOL>")
            print(f"Example: python \"Build flags_pennants_dataset.py\" --symbol {symbols[0]}")
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found")
        except Exception as e:
            print(f"Error reading input file: {e}")
        return

    # Set default output path if not provided
    if args.output is None:
        args.output = f"{args.symbol}_flags_pennants_15m_windows.csv"

    # Run the dataset builder
    build_flags_pennants_dataset_for_symbol(
        input_csv_path=args.input,
        output_csv_path=args.output,
        symbol=args.symbol,
        window_bars=args.window_bars,
        sample_rate=args.sample_rate,
        filter_trading_hours=args.filter_trading_hours,
        market_open=args.market_open,
        market_close=args.market_close,
        filter_day_boundaries=args.filter_day_boundaries
    )


if __name__ == "__main__":
    main()