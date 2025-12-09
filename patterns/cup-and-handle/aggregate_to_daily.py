"""
Aggregate 3-second tick data to daily OHLCV bars.

This script converts the combined_dataset.csv (tick data) to daily bars
suitable for Cup and Handle pattern detection.

Usage:
    python aggregate_to_daily.py --input ../../combined_dataset.csv --output outputs/daily_data.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_ticks_to_daily(input_path: str, output_path: str, chunk_size: int = 500000) -> pd.DataFrame:
    """
    Aggregate tick data to daily OHLCV bars.
    
    Since the input only has price (not full OHLCV), we create pseudo-OHLCV:
    - Open: first price of the day
    - High: max price of the day
    - Low: min price of the day
    - Close: last price of the day
    - Volume: count of ticks (proxy for activity)
    
    Args:
        input_path: Path to combined_dataset.csv
        output_path: Path to save daily data
        chunk_size: Number of rows to process at once (for memory efficiency)
    
    Returns:
        DataFrame with daily OHLCV data
    """
    print(f"Loading data from {input_path}...")
    print("Processing in chunks due to large file size...")
    
    # Read first chunk to get column names
    sample = pd.read_csv(input_path, nrows=5)
    symbols = [col for col in sample.columns if col not in ['ID', 'TimeStamp']]
    print(f"Found symbols: {symbols}")
    
    # Process in chunks
    all_daily = []
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"Processing chunk {chunk_num}...")
        
        # Parse timestamp and extract date
        chunk['TimeStamp'] = pd.to_datetime(chunk['TimeStamp'])
        chunk['Date'] = chunk['TimeStamp'].dt.date
        
        # Aggregate each symbol
        for symbol in symbols:
            if symbol not in chunk.columns:
                continue
                
            daily = chunk.groupby('Date').agg(
                Open=(symbol, 'first'),
                High=(symbol, 'max'),
                Low=(symbol, 'min'),
                Close=(symbol, 'last'),
                Volume=(symbol, 'count')
            ).reset_index()
            
            daily['Symbol'] = symbol
            all_daily.append(daily)
    
    print("Combining all chunks...")
    combined = pd.concat(all_daily, ignore_index=True)
    
    # Aggregate again to handle chunks that span same day
    print("Final aggregation...")
    final_daily = combined.groupby(['Symbol', 'Date']).agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum')
    ).reset_index()
    
    # Sort by symbol and date
    final_daily = final_daily.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Convert Date to datetime
    final_daily['Date'] = pd.to_datetime(final_daily['Date'])
    
    print(f"\nAggregation complete!")
    print(f"Total rows: {len(final_daily)}")
    print(f"Date range: {final_daily['Date'].min()} to {final_daily['Date'].max()}")
    print(f"Symbols: {final_daily['Symbol'].unique().tolist()}")
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_daily.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return final_daily


def main():
    parser = argparse.ArgumentParser(description="Aggregate tick data to daily OHLCV bars")
    parser.add_argument(
        "--input",
        type=str,
        default="../../combined_dataset.csv",
        help="Path to combined_dataset.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/daily_data.csv",
        help="Path to save daily data"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Chunk size for processing large files"
    )
    
    args = parser.parse_args()
    aggregate_ticks_to_daily(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()

