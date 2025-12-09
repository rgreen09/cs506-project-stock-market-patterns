"""
Script to combine multiple double-top dataset CSV files into one combined dataset.
Handles large files by processing them in chunks to avoid memory issues.
"""

import pandas as pd
import os
from pathlib import Path

def combine_datasets(data_dir: str = "data", output_file: str = "data/combined_double_top_15m_windows.csv"):
    """
    Combine all double-top dataset CSV files in the data directory into one file.
    
    Args:
        data_dir: Directory containing the CSV files
        output_file: Path to the output combined CSV file
    """
    data_path = Path(data_dir)
    
    # Find all double_top CSV files
    csv_files = sorted(data_path.glob("*_double_top_15m_windows.csv"))
    
    if not csv_files:
        print(f"No double-top CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} files to combine:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Get the header from the first file
    print("\nReading header from first file...")
    first_file = csv_files[0]
    sample_df = pd.read_csv(first_file, nrows=1)
    columns = sample_df.columns.tolist()
    
    print(f"Columns: {', '.join(columns)}")
    
    # Open output file and write header
    print(f"\nWriting combined dataset to {output_file}...")
    first_chunk = True
    
    total_rows = 0
    chunk_size = 10000  # Process in chunks of 10k rows
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        file_rows = 0
        
        # Read and write in chunks
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            file_rows += len(chunk)
            total_rows += len(chunk)
            
            # Write chunk to output file
            if first_chunk:
                chunk.to_csv(output_file, mode='w', index=False, header=True)
                first_chunk = False
            else:
                chunk.to_csv(output_file, mode='a', index=False, header=False)
        
        print(f"  Added {file_rows:,} rows from {csv_file.name}")
    
    print(f"\n✓ Successfully combined {len(csv_files)} files")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output file: {output_file}")
    
    # Verify the combined file
    print(f"\nVerifying combined file...")
    verify_df = pd.read_csv(output_file, nrows=5)
    print(f"  Sample rows: {len(verify_df)}")
    print(f"  Columns: {len(verify_df.columns)}")
    print(f"  Unique symbols: {verify_df['symbol'].unique() if 'symbol' in verify_df.columns else 'N/A'}")

if __name__ == "__main__":
    combine_datasets()

