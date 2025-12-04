"""
Visualize Double Top Patterns

This script creates visualizations of successful double top patterns from each stock.
For each example, it shows price data with 100 bars before and after the window,
and annotates the peaks and neckline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Add double-top directory to path to import detector
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from A_double_top_detector import DoubleTopDetector, DoubleTopConfig


def load_successful_double_tops(csv_path: str, num_examples: int = 5) -> pd.DataFrame:
    """
    Load and filter successful double tops from a stock CSV file.
    
    Args:
        csv_path: Path to the stock's double top CSV file
        num_examples: Number of examples to return
        
    Returns:
        DataFrame with successful double top windows
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for successful double tops
    successful = df[df['label_double_top'] == 1].copy()
    
    if len(successful) == 0:
        print(f"  No successful double tops found in {csv_path}")
        return pd.DataFrame()
    
    # Sample num_examples (or all if fewer available)
    num_to_sample = min(num_examples, len(successful))
    sampled = successful.sample(n=num_to_sample, random_state=42).reset_index(drop=True)
    
    # Parse timestamps
    sampled['start_timestamp'] = pd.to_datetime(sampled['start_timestamp'])
    sampled['end_timestamp'] = pd.to_datetime(sampled['end_timestamp'])
    
    print(f"  Found {len(successful)} successful double tops, sampling {num_to_sample}")
    return sampled


def extract_price_data(
    original_data: pd.DataFrame,
    symbol: str,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    bars_before: int = 100,
    bars_after: int = 100,
    window_bars: int = 300
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Extract price data from original dataset using window timestamps.
    
    Args:
        original_data: DataFrame with TimeStamp and symbol columns
        symbol: Stock symbol to extract
        start_timestamp: Start timestamp of the window
        end_timestamp: End timestamp of the window
        bars_before: Number of bars to include before the window
        bars_after: Number of bars to include after the window
        window_bars: Size of the window in bars
        
    Returns:
        Tuple of (full_price_series, window_price_series, timestamps, window_indices)
        Returns (None, None, None, None) if data cannot be extracted
    """
    # Find the window in the original data
    # Get all rows for this symbol
    symbol_data = original_data[['TimeStamp', symbol]].copy()
    symbol_data = symbol_data.dropna(subset=[symbol])
    symbol_data = symbol_data.sort_values('TimeStamp').reset_index(drop=True)
    
    # Find indices matching the window timestamps
    # Try exact match first
    start_matches = symbol_data[symbol_data['TimeStamp'] == start_timestamp].index
    end_matches = symbol_data[symbol_data['TimeStamp'] == end_timestamp].index
    
    # If exact match not found, find closest timestamps
    if len(start_matches) == 0:
        # Find closest timestamp to start
        time_diffs = (symbol_data['TimeStamp'] - start_timestamp).abs()
        start_idx = time_diffs.idxmin()
        # Check if the difference is reasonable (within 10 seconds)
        if time_diffs[start_idx] > pd.Timedelta(seconds=10):
            return None, None, None, None
    else:
        start_idx = start_matches[0]
    
    if len(end_matches) == 0:
        # Find closest timestamp to end
        time_diffs = (symbol_data['TimeStamp'] - end_timestamp).abs()
        end_idx = time_diffs.idxmin()
        # Check if the difference is reasonable (within 10 seconds)
        if time_diffs[end_idx] > pd.Timedelta(seconds=10):
            return None, None, None, None
    else:
        end_idx = end_matches[0]
    
    # Verify the window size matches
    actual_window_size = end_idx - start_idx + 1
    if actual_window_size != window_bars:
        # Try to find the closest match
        print(f"  Warning: Window size mismatch. Expected {window_bars}, got {actual_window_size}")
        # Use the actual window size
        window_bars = actual_window_size
    
    # Extract window data
    window_start = max(0, start_idx - bars_before)
    window_end = min(len(symbol_data), end_idx + bars_after + 1)
    
    extracted_data = symbol_data.iloc[window_start:window_end].copy()
    
    if len(extracted_data) == 0:
        return None, None, None, None
    
    # Get price series and timestamps
    full_price_series = extracted_data[symbol].copy()
    timestamps = extracted_data['TimeStamp'].copy()
    
    # Calculate window indices within the extracted data
    window_start_in_extracted = start_idx - window_start
    window_end_in_extracted = end_idx - window_start + 1
    
    window_price_series = full_price_series.iloc[window_start_in_extracted:window_end_in_extracted].copy()
    
    window_indices = (window_start_in_extracted, window_end_in_extracted)
    
    return full_price_series, window_price_series, timestamps, window_indices


def detect_pattern_annotations(
    window_prices: pd.Series,
    window_bars: int = 300
) -> Optional[dict]:
    """
    Use DoubleTopDetector to detect pattern and extract peak/neckline information.
    
    Args:
        window_prices: Price series for the window
        window_bars: Size of the window
        
    Returns:
        Dictionary with peak/neckline information, or None if detection fails
    """
    # Use the same config as in build_double_top_dataset.py
    config = DoubleTopConfig(
        peak_tolerance=0.04,
        min_drop_pct=0.002,
        min_gap=5,
        max_gap_ratio=0.6,
        require_confirmation=False,
        smoothing_window=7
    )
    detector = DoubleTopDetector(config)
    result = detector.detect(window_prices, window_bars)
    
    if not result.is_pattern:
        return None
    
    return {
        'peak1_idx': result.peak1_idx,
        'peak2_idx': result.peak2_idx,
        'peak1_price': result.peak1_price,
        'peak2_price': result.peak2_price,
        'neckline_price': result.neckline_price,
        'neckline_idx': result.neckline_idx,
        'drop_pct': result.drop_pct,
        'height_diff_pct': result.height_diff_pct,
        'bars_between_peaks': result.bars_between_peaks
    }


def create_visualization(
    full_prices: pd.Series,
    timestamps: pd.Series,
    window_indices: Tuple[int, int],
    annotations: dict,
    symbol: str,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    output_path: str,
    bars_before: int = 100
):
    """
    Create matplotlib visualization of double top pattern.
    
    Args:
        full_prices: Full price series including before/after window
        timestamps: Timestamps for the full price series
        window_indices: Tuple of (window_start_idx, window_end_idx) in full_prices
        annotations: Dictionary with peak/neckline information
        symbol: Stock symbol
        start_timestamp: Window start timestamp
        end_timestamp: Window end timestamp
        output_path: Path to save the visualization
        bars_before: Number of bars before window (for offset calculation)
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot the full price series
    x_full = np.arange(len(full_prices))
    ax.plot(x_full, full_prices.values, 'b-', linewidth=1.5, label='Price', alpha=0.7)
    
    # Highlight the window region
    window_start_idx, window_end_idx = window_indices
    window_x = x_full[window_start_idx:window_end_idx]
    window_prices = full_prices.iloc[window_start_idx:window_end_idx]
    ax.plot(window_x, window_prices.values, 'g-', linewidth=2, label='Window', alpha=0.9)
    
    # Add vertical lines for window boundaries
    ax.axvline(x=window_start_idx, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Window Start')
    ax.axvline(x=window_end_idx - 1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Window End')
    
    # Annotate peaks (adjust indices to account for bars_before offset)
    if annotations:
        peak1_idx_in_window = annotations['peak1_idx']
        peak2_idx_in_window = annotations['peak2_idx']
        
        # Map window indices to full series indices
        peak1_idx_full = window_start_idx + peak1_idx_in_window
        peak2_idx_full = window_start_idx + peak2_idx_in_window
        
        # Mark peaks
        ax.plot(peak1_idx_full, annotations['peak1_price'], 'ro', markersize=12, 
                markeredgecolor='darkred', markeredgewidth=2, label='Peak 1', zorder=5)
        ax.plot(peak2_idx_full, annotations['peak2_price'], 'ro', markersize=12, 
                markeredgecolor='darkred', markeredgewidth=2, label='Peak 2', zorder=5)
        
        # Draw neckline
        neckline_price = annotations['neckline_price']
        neckline_start = window_start_idx
        neckline_end = window_end_idx
        ax.axhline(y=neckline_price, color='purple', linestyle='-', linewidth=2, 
                  alpha=0.7, label=f'Neckline (${neckline_price:.2f})')
        
        # Add text annotations for peaks
        ax.annotate(f'Peak 1\n${annotations["peak1_price"]:.2f}',
                   xy=(peak1_idx_full, annotations['peak1_price']),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Peak 2\n${annotations["peak2_price"]:.2f}',
                   xy=(peak2_idx_full, annotations['peak2_price']),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Formatting
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Create title with pattern metrics
    title = f'{symbol} Double Top Pattern\n'
    title += f'Window: {start_timestamp.strftime("%Y-%m-%d %H:%M:%S")} to {end_timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
    if annotations:
        title += f'\nDrop: {annotations["drop_pct"]*100:.2f}% | '
        title += f'Height Diff: {annotations["height_diff_pct"]*100:.2f}% | '
        title += f'Bars Between: {annotations["bars_between_peaks"]}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {output_path}")


def main():
    """Main function to process all stocks and generate visualizations."""
    # Configuration
    data_dir = Path('data')
    original_data_path = Path('data/old_data/combined_dataset.csv')
    output_dir = Path('visualizations')
    num_examples_per_stock = 5
    bars_before = 100
    bars_after = 100
    window_bars = 300
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load original price data
    print("Loading original price data...")
    print(f"  Reading {original_data_path}...")
    original_data = pd.read_csv(original_data_path)
    original_data['TimeStamp'] = pd.to_datetime(original_data['TimeStamp'])
    original_data = original_data.sort_values('TimeStamp').reset_index(drop=True)
    print(f"  Loaded {len(original_data):,} rows")
    
    # Stock symbols to process
    symbols = ['AAPL', 'MSFT', 'NVDA', 'QQQ', 'SPY']
    
    # Process each stock
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        csv_path = data_dir / f"{symbol}_double_top_15m_windows.csv"
        
        if not csv_path.exists():
            print(f"  File not found: {csv_path}")
            continue
        
        # Load successful double tops
        successful_tops = load_successful_double_tops(str(csv_path), num_examples_per_stock)
        
        if len(successful_tops) == 0:
            continue
        
        # Process each example
        for idx, row in successful_tops.iterrows():
            print(f"\n  Example {idx + 1}/{len(successful_tops)}")
            print(f"    Window: {row['start_timestamp']} to {row['end_timestamp']}")
            
            # Extract price data
            full_prices, window_prices, timestamps, window_indices = extract_price_data(
                original_data,
                symbol,
                row['start_timestamp'],
                row['end_timestamp'],
                bars_before,
                bars_after,
                window_bars
            )
            
            if full_prices is None or window_prices is None:
                print(f"    Could not extract price data, skipping...")
                continue
            
            # Detect pattern annotations
            annotations = detect_pattern_annotations(window_prices, window_bars)
            
            if annotations is None:
                print(f"    Could not detect pattern in window, skipping...")
                continue
            
            # Create visualization
            output_path = output_dir / f"{symbol}_double_top_{idx + 1}.png"
            create_visualization(
                full_prices,
                timestamps,
                window_indices,
                annotations,
                symbol,
                row['start_timestamp'],
                row['end_timestamp'],
                str(output_path),
                bars_before
            )
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

