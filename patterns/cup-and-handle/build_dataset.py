"""
Build labeled dataset for Cup and Handle pattern classification.

This script creates sliding windows of daily price data with engineered features
and binary labels indicating whether the window contains a Cup and Handle pattern.

Usage:
    python build_dataset.py --daily outputs/daily_data.csv --patterns outputs/detected_patterns.csv --output outputs/training_dataset.csv
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class WindowConfig:
    """Configuration for sliding window feature extraction."""
    window_size: int = 65  # Max cup duration
    min_window_size: int = 30  # Minimum window to consider
    label_overlap_threshold: float = 0.5  # Minimum overlap to label as positive


def find_local_peaks(values: np.ndarray) -> List[int]:
    """Return indices of simple local maxima."""
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(i)
    return peaks


def find_local_troughs(values: np.ndarray) -> List[int]:
    """Return indices of simple local minima."""
    troughs = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            troughs.append(i)
    return troughs


def linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y ~ x."""
    if len(x) < 2:
        return 0.0
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    slope = result[0][0]
    return float(slope)


def compute_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute engineered features from a window of OHLCV data.
    
    Features are designed to capture Cup and Handle characteristics:
    - Price movements and volatility
    - Peak/trough patterns
    - Volume patterns
    - Shape characteristics
    """
    features: Dict[str, float] = {}
    
    closes = window_df['Close'].values.astype(float)
    highs = window_df['High'].values.astype(float)
    lows = window_df['Low'].values.astype(float)
    volumes = window_df['Volume'].values.astype(float)
    n = len(closes)
    
    if n < 5:
        return {}
    
    # === Basic Price Statistics ===
    features['close_mean'] = float(np.mean(closes))
    features['close_std'] = float(np.std(closes))
    features['close_min'] = float(np.min(closes))
    features['close_max'] = float(np.max(closes))
    features['price_range'] = float(np.max(closes) - np.min(closes))
    features['price_range_pct'] = float(features['price_range'] / closes[0]) if closes[0] != 0 else 0.0
    
    # === Returns ===
    features['total_return'] = float((closes[-1] / closes[0] - 1.0)) if closes[0] != 0 else 0.0
    
    daily_returns = np.diff(closes) / closes[:-1]
    daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
    features['return_mean'] = float(np.mean(daily_returns))
    features['return_std'] = float(np.std(daily_returns))
    features['return_skew'] = float(pd.Series(daily_returns).skew()) if len(daily_returns) > 2 else 0.0
    
    # === Trend Features (Linear Regression Slopes) ===
    x_full = np.arange(n)
    features['slope_full'] = linear_regression_slope(x_full, closes)
    
    # First half vs second half
    mid = n // 2
    features['slope_first_half'] = linear_regression_slope(x_full[:mid], closes[:mid])
    features['slope_second_half'] = linear_regression_slope(x_full[mid:], closes[mid:])
    features['slope_diff'] = features['slope_second_half'] - features['slope_first_half']
    
    # === Cup-Specific Features ===
    # Find peaks and troughs
    smooth = pd.Series(closes).rolling(window=3, center=True, min_periods=1).mean().values
    peaks = find_local_peaks(smooth)
    troughs = find_local_troughs(smooth)
    
    features['num_peaks'] = float(len(peaks))
    features['num_troughs'] = float(len(troughs))
    
    # Peak characteristics
    if len(peaks) >= 2:
        first_peak_idx = peaks[0]
        last_peak_idx = peaks[-1]
        first_peak_price = closes[first_peak_idx]
        last_peak_price = closes[last_peak_idx]
        
        features['peak_price_diff_pct'] = abs(first_peak_price - last_peak_price) / first_peak_price if first_peak_price != 0 else 0.0
        features['peak_distance'] = float(last_peak_idx - first_peak_idx)
    else:
        features['peak_price_diff_pct'] = 0.0
        features['peak_distance'] = 0.0
    
    # Cup depth (max drawdown from first local high)
    if len(peaks) > 0 and len(troughs) > 0:
        first_peak_price = closes[peaks[0]]
        min_trough_price = min(closes[t] for t in troughs)
        features['max_drawdown_pct'] = (first_peak_price - min_trough_price) / first_peak_price if first_peak_price != 0 else 0.0
    else:
        features['max_drawdown_pct'] = (closes[0] - np.min(closes)) / closes[0] if closes[0] != 0 else 0.0
    
    # Position of minimum (ideally around middle for cup shape)
    min_idx = np.argmin(closes)
    features['min_position_ratio'] = float(min_idx / n)
    
    # === Volume Features ===
    features['volume_mean'] = float(np.mean(volumes))
    features['volume_std'] = float(np.std(volumes))
    
    # Volume trend
    features['volume_slope'] = linear_regression_slope(x_full, volumes)
    
    # Volume at start vs end
    vol_first_quarter = np.mean(volumes[:n//4]) if n >= 4 else volumes[0]
    vol_last_quarter = np.mean(volumes[-n//4:]) if n >= 4 else volumes[-1]
    features['volume_ratio_end_start'] = float(vol_last_quarter / vol_first_quarter) if vol_first_quarter != 0 else 0.0
    
    # === Shape Features (U-shape detection) ===
    # Ratio of close prices at edges vs center
    edge_avg = (closes[0] + closes[-1]) / 2
    center_avg = np.mean(closes[n//3:2*n//3])
    features['edge_to_center_ratio'] = float(edge_avg / center_avg) if center_avg != 0 else 0.0
    
    # Symmetry measure
    first_half_avg = np.mean(closes[:mid])
    second_half_avg = np.mean(closes[mid:])
    features['half_symmetry'] = abs(first_half_avg - second_half_avg) / first_half_avg if first_half_avg != 0 else 0.0
    
    # === High-Low Spread ===
    hl_spread = highs - lows
    features['hl_spread_mean'] = float(np.mean(hl_spread))
    features['hl_spread_std'] = float(np.std(hl_spread))
    
    # Volatility clustering
    features['hl_spread_first_half'] = float(np.mean(hl_spread[:mid]))
    features['hl_spread_second_half'] = float(np.mean(hl_spread[mid:]))
    
    return features


def is_window_in_pattern(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    patterns_df: pd.DataFrame,
    symbol: str,
    overlap_threshold: float = 0.5
) -> int:
    """
    Check if a window overlaps with any detected pattern.
    
    Args:
        window_start: Start date of window
        window_end: End date of window
        patterns_df: DataFrame with detected patterns
        symbol: Stock symbol
        overlap_threshold: Minimum overlap ratio to label as positive
    
    Returns:
        1 if window contains pattern, 0 otherwise
    """
    symbol_patterns = patterns_df[patterns_df['ticker'] == symbol]
    
    if len(symbol_patterns) == 0:
        return 0
    
    window_days = (window_end - window_start).days + 1
    
    for _, pattern in symbol_patterns.iterrows():
        pattern_start = pd.to_datetime(pattern['pattern_start_date'])
        pattern_end = pd.to_datetime(pattern['pattern_end_date'])
        
        # Calculate overlap
        overlap_start = max(window_start, pattern_start)
        overlap_end = min(window_end, pattern_end)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days + 1
            overlap_ratio = overlap_days / window_days
            
            if overlap_ratio >= overlap_threshold:
                return 1
    
    return 0


def build_training_dataset(
    daily_path: str,
    patterns_path: str,
    output_path: str,
    config: Optional[WindowConfig] = None
) -> pd.DataFrame:
    """
    Build training dataset with sliding windows and labels.
    
    Args:
        daily_path: Path to daily OHLCV data
        patterns_path: Path to detected patterns
        output_path: Path to save training dataset
        config: Window configuration
    
    Returns:
        DataFrame with features and labels
    """
    if config is None:
        config = WindowConfig()
    
    print(f"Loading daily data from {daily_path}...")
    daily_df = pd.read_csv(daily_path)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    print(f"Loading patterns from {patterns_path}...")
    patterns_df = pd.read_csv(patterns_path)
    
    symbols = daily_df['Symbol'].unique()
    print(f"Symbols: {symbols.tolist()}")
    print(f"Window size: {config.window_size} days")
    
    all_rows = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        symbol_df = daily_df[daily_df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date').reset_index(drop=True)
        n_days = len(symbol_df)
        
        # Sliding window
        for end_idx in range(config.window_size - 1, n_days):
            start_idx = end_idx - config.window_size + 1
            
            window_df = symbol_df.iloc[start_idx:end_idx + 1]
            window_start = window_df['Date'].iloc[0]
            window_end = window_df['Date'].iloc[-1]
            
            # Compute features
            features = compute_window_features(window_df)
            
            if not features:
                continue
            
            # Get label
            label = is_window_in_pattern(
                window_start, window_end, 
                patterns_df, symbol,
                config.label_overlap_threshold
            )
            
            # Create row
            row = {
                'symbol': symbol,
                'window_start': window_start,
                'window_end': window_end,
                **features,
                'label': label
            }
            all_rows.append(row)
    
    # Create DataFrame
    dataset_df = pd.DataFrame(all_rows)
    
    # Summary
    n_positive = dataset_df['label'].sum()
    n_negative = len(dataset_df) - n_positive
    
    print(f"\n{'='*50}")
    print(f"Dataset built successfully!")
    print(f"Total samples: {len(dataset_df)}")
    print(f"Positive (pattern): {n_positive} ({100*n_positive/len(dataset_df):.2f}%)")
    print(f"Negative (no pattern): {n_negative} ({100*n_negative/len(dataset_df):.2f}%)")
    print(f"Features: {len([c for c in dataset_df.columns if c not in ['symbol', 'window_start', 'window_end', 'label']])}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return dataset_df


def main():
    parser = argparse.ArgumentParser(description="Build training dataset for Cup and Handle classification")
    parser.add_argument(
        "--daily",
        type=str,
        default="outputs/daily_data.csv",
        help="Path to daily OHLCV data"
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="outputs/detected_patterns.csv",
        help="Path to detected patterns"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/training_dataset.csv",
        help="Path to save training dataset"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=65,
        help="Sliding window size in days"
    )
    
    args = parser.parse_args()
    
    config = WindowConfig(window_size=args.window_size)
    build_training_dataset(args.daily, args.patterns, args.output, config)


if __name__ == "__main__":
    main()

