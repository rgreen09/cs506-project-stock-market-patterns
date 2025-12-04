"""
Script to combine multiple double-top dataset CSV files into one combined dataset.
Performs stratified sampling: keeps all double-top rows and samples 2x that count
from non-double-top rows for each dataset before combining.
Handles large files by processing them in chunks to avoid memory issues.
"""

import pandas as pd
import os
from pathlib import Path
import random

def combine_datasets(
    data_dir: str = "data", 
    output_file: str = "data/combined_double_top_15m_windows.csv",
    random_state: int = 42
):
    """
    Combine all double-top dataset CSV files with stratified sampling.
    
    For each dataset:
    - Keep ALL rows where label_double_top = 1 (has double top)
    - Randomly sample 2x that count from rows where label_double_top = 0 (no double top)
    - If insufficient non-double-top rows exist, take all available
    
    Args:
        data_dir: Directory containing the CSV files
        output_file: Path to the output combined CSV file
        random_state: Random seed for reproducibility
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
    
    # Verify label_double_top column exists
    if 'label_double_top' not in columns:
        raise ValueError("CSV files must contain a 'label_double_top' column")
    
    # Initialize output file with header
    print(f"\nWriting combined dataset to {output_file}...")
    first_file_processed = True
    
    total_rows_combined = 0
    chunk_size = 10000  # Process in chunks of 10k rows
    
    # Process each file
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        # Accumulate rows for this file
        double_top_rows = []
        non_double_top_rows = []
        
        # Read file in chunks and separate by label
        print("  Reading file and separating rows by label...")
        chunk_count = 0
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_count += 1
            
            # Separate rows by label_double_top
            double_top_chunk = chunk[chunk['label_double_top'] == 1]
            non_double_top_chunk = chunk[chunk['label_double_top'] == 0]
            
            # Append to accumulators
            if len(double_top_chunk) > 0:
                double_top_rows.append(double_top_chunk)
            if len(non_double_top_chunk) > 0:
                non_double_top_rows.append(non_double_top_chunk)
            
            if chunk_count % 100 == 0:
                print(f"    Processed {chunk_count} chunks...")
        
        # Combine accumulated chunks into DataFrames
        print("  Combining chunks...")
        if double_top_rows:
            double_top_df = pd.concat(double_top_rows, ignore_index=True)
        else:
            double_top_df = pd.DataFrame(columns=columns)
        
        if non_double_top_rows:
            non_double_top_df = pd.concat(non_double_top_rows, ignore_index=True)
        else:
            non_double_top_df = pd.DataFrame(columns=columns)
        
        double_top_count = len(double_top_df)
        non_double_top_count = len(non_double_top_df)
        
        print(f"  Found {double_top_count:,} double-top rows")
        print(f"  Found {non_double_top_count:,} non-double-top rows")
        
        # Sample non-double-top rows: 2x the double-top count (or all available)
        target_sample_size = 2 * double_top_count
        if non_double_top_count >= target_sample_size:
            print(f"  Sampling {target_sample_size:,} non-double-top rows (2x double-top count)...")
            sampled_non_double_top_df = non_double_top_df.sample(
                n=target_sample_size, 
                random_state=random_state
            )
        else:
            print(f"  Warning: Only {non_double_top_count:,} non-double-top rows available "
                  f"(need {target_sample_size:,}). Taking all available.")
            sampled_non_double_top_df = non_double_top_df
        
        # Combine double-top and sampled non-double-top rows
        combined_df = pd.concat([double_top_df, sampled_non_double_top_df], ignore_index=True)
        
        # Shuffle the combined dataset to mix double-top and non-double-top rows
        combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        file_rows = len(combined_df)
        total_rows_combined += file_rows
        
        # Write to output file
        print(f"  Writing {file_rows:,} sampled rows to output file...")
        if first_file_processed:
            combined_df.to_csv(output_file, mode='w', index=False, header=True)
            first_file_processed = False
        else:
            combined_df.to_csv(output_file, mode='a', index=False, header=False)
        
        print(f"  ✓ Added {file_rows:,} rows from {csv_file.name}")
        print(f"    - Double-tops: {double_top_count:,}")
        print(f"    - Non-double-tops sampled: {len(sampled_non_double_top_df):,}")
    
    print(f"\n✓ Successfully combined {len(csv_files)} files")
    print(f"  Total rows: {total_rows_combined:,}")
    print(f"  Output file: {output_file}")
    
    # Verify the combined file
    print(f"\nVerifying combined file...")
    verify_df = pd.read_csv(output_file, nrows=5)
    print(f"  Sample rows: {len(verify_df)}")
    print(f"  Columns: {len(verify_df.columns)}")
    
    # Check label distribution in sample
    if 'label_double_top' in verify_df.columns:
        sample_labels = verify_df['label_double_top'].value_counts()
        print(f"  Label distribution in sample: {dict(sample_labels)}")
    
    # Get full statistics
    print(f"\nCalculating full dataset statistics...")
    full_df = pd.read_csv(output_file)
    total_double_tops = len(full_df[full_df['label_double_top'] == 1])
    total_non_double_tops = len(full_df[full_df['label_double_top'] == 0])
    print(f"  Total double-tops: {total_double_tops:,}")
    print(f"  Total non-double-tops: {total_non_double_tops:,}")
    print(f"  Ratio: {total_non_double_tops/total_double_tops:.2f}:1" if total_double_tops > 0 else "  Ratio: N/A")
    
    if 'symbol' in full_df.columns:
        unique_symbols = full_df['symbol'].unique()
        print(f"  Unique symbols: {', '.join(sorted(unique_symbols))}")

if __name__ == "__main__":
    combine_datasets()
