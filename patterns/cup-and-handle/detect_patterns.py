"""
Detect Cup and Handle patterns in daily aggregated data.

This script runs the Cup and Handle detector on daily OHLCV data
and outputs detected patterns to a CSV file.

Usage:
    python detect_patterns.py --input outputs/daily_data.csv --output outputs/detected_patterns.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add cup_and_handle module to path (works from multiple locations)
_module_paths = [
    Path(__file__).parent / "cup_and_handle",  # Local copy
    Path(__file__).parent.parent.parent / "labeling" / "cup_and_handle",  # Original location
]
for _path in _module_paths:
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

from detector import CupAndHandleDetector


def detect_patterns_in_daily_data(
    input_path: str,
    output_path: str,
    min_cup_duration: int = 7,
    max_cup_duration: int = 65,
    min_cup_depth: float = 12,
    max_cup_depth: float = 33
) -> pd.DataFrame:
    """
    Detect Cup and Handle patterns in daily data.
    
    Args:
        input_path: Path to daily_data.csv
        output_path: Path to save detected patterns
        min_cup_duration: Minimum cup duration in days
        max_cup_duration: Maximum cup duration in days
        min_cup_depth: Minimum cup depth percentage
        max_cup_depth: Maximum cup depth percentage
    
    Returns:
        DataFrame with detected patterns
    """
    print(f"Loading daily data from {input_path}...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    symbols = df['Symbol'].unique()
    print(f"Symbols to process: {symbols.tolist()}")
    
    # Initialize detector with relaxed parameters for limited data
    detector = CupAndHandleDetector(
        min_cup_duration=min_cup_duration,
        max_cup_duration=max_cup_duration,
        min_cup_depth=min_cup_depth,
        max_cup_depth=max_cup_depth,
        peak_similarity_threshold=8,  # Relaxed for limited data
        min_handle_duration=3,
        max_handle_duration=25,
        max_handle_depth=18,
        volume_breakout_threshold=1.0,  # Relaxed since volume is proxy
        extrema_order=3
    )
    
    all_patterns = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date').reset_index(drop=True)
        
        print(f"  Date range: {symbol_df['Date'].min()} to {symbol_df['Date'].max()}")
        print(f"  Total days: {len(symbol_df)}")
        
        # Detect patterns
        patterns = detector.detect_patterns(symbol_df)
        
        # Add ticker to each pattern
        for pattern in patterns:
            pattern['ticker'] = symbol
        
        all_patterns.extend(patterns)
        print(f"  Patterns found: {len(patterns)}")
    
    # Create DataFrame
    if all_patterns:
        patterns_df = pd.DataFrame(all_patterns)
        
        # Reorder columns
        columns_order = [
            'ticker', 'pattern_start_date', 'pattern_end_date',
            'cup_start_date', 'cup_end_date',
            'handle_start_date', 'handle_end_date', 'breakout_date',
            'cup_depth_pct', 'handle_depth_pct', 'breakout_price', 'confidence_score'
        ]
        patterns_df = patterns_df[[c for c in columns_order if c in patterns_df.columns]]
        
        print(f"\n{'='*50}")
        print(f"Total patterns detected: {len(patterns_df)}")
        print(f"\nPatterns by ticker:")
        print(patterns_df['ticker'].value_counts())
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        patterns_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        
        return patterns_df
    else:
        print("\nNo patterns detected with current parameters.")
        print("This is expected with only 5 symbols - Cup and Handle is a rare pattern.")
        
        # Create empty DataFrame with correct structure
        patterns_df = pd.DataFrame(columns=[
            'ticker', 'pattern_start_date', 'pattern_end_date',
            'cup_start_date', 'cup_end_date',
            'handle_start_date', 'handle_end_date', 'breakout_date',
            'cup_depth_pct', 'handle_depth_pct', 'breakout_price', 'confidence_score'
        ])
        patterns_df.to_csv(output_path, index=False)
        print(f"Saved empty patterns file to {output_path}")
        
        return patterns_df


def main():
    parser = argparse.ArgumentParser(description="Detect Cup and Handle patterns in daily data")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/daily_data.csv",
        help="Path to daily_data.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/detected_patterns.csv",
        help="Path to save detected patterns"
    )
    parser.add_argument(
        "--min-cup-duration",
        type=int,
        default=7,
        help="Minimum cup duration in days"
    )
    parser.add_argument(
        "--max-cup-duration",
        type=int,
        default=65,
        help="Maximum cup duration in days"
    )
    
    args = parser.parse_args()
    detect_patterns_in_daily_data(
        args.input,
        args.output,
        args.min_cup_duration,
        args.max_cup_duration
    )


if __name__ == "__main__":
    main()

