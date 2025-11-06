"""
Build Combined Dataset from Zip Files

Processes zip files (2020-2025) containing weekly CSV files,
filters to keep only ID, TimeStamp, MSFT, AAPL, NVDA, SPY, QQQ columns,
and combines into a single dataset sorted by timestamp.
"""

import zipfile
import pandas as pd
from pathlib import Path
from io import StringIO
from typing import List

# Columns to keep in the final dataset
KEEP_COLUMNS = ['ID', 'TimeStamp', 'MSFT', 'AAPL', 'NVDA', 'SPY', 'QQQ']

# Zip files in chronological order
ZIP_FILES = [
    'Zipped data 2020.zip',
    'Zipped data 2021.zip',
    'Zipped data 2022.zip',
    'Zipped data 2023.zip',
    'Zipped data 2024.zip',
    'Zipped data 2025.zip',
]


def process_zip_file(zip_path: Path) -> pd.DataFrame:
    """
    Process a single zip file and return a DataFrame with filtered columns.
    
    Args:
        zip_path: Path to the zip file
        
    Returns:
        DataFrame with filtered columns from all CSV files in the zip
    """
    print(f"Processing {zip_path.name}...")
    
    all_dataframes: List[pd.DataFrame] = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all CSV files in the zip
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        csv_files.sort()  # Sort to ensure consistent processing order
        
        print(f"  Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Read CSV directly from zip
                content = zip_ref.read(csv_file).decode('utf-8')
                df = pd.read_csv(StringIO(content))
                
                # Check which columns exist
                available_columns = [col for col in KEEP_COLUMNS if col in df.columns]
                
                if 'TimeStamp' not in df.columns:
                    print(f"  Warning: {csv_file} missing TimeStamp column, skipping")
                    continue
                
                # Filter to keep only desired columns
                df_filtered = df[available_columns].copy()
                
                # Only add if we have at least ID and TimeStamp
                if len(df_filtered.columns) >= 2:
                    all_dataframes.append(df_filtered)
                else:
                    print(f"  Warning: {csv_file} has insufficient columns, skipping")
                    
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")
                continue
    
    if not all_dataframes:
        print(f"  No valid data found in {zip_path.name}")
        return pd.DataFrame(columns=KEEP_COLUMNS)
    
    # Combine all DataFrames from this zip file
    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"  Combined {len(combined)} rows from {zip_path.name}")
    
    return combined


def build_combined_dataset(data_dir: Path, output_path: Path) -> None:
    """
    Build combined dataset from all zip files.
    
    Args:
        data_dir: Directory containing the zip files
        output_path: Path to save the combined dataset
    """
    print("=" * 60)
    print("Building Combined Dataset")
    print("=" * 60)
    print()
    
    all_dataframes: List[pd.DataFrame] = []
    
    # Process each zip file in chronological order
    for zip_filename in ZIP_FILES:
        zip_path = data_dir / zip_filename
        
        if not zip_path.exists():
            print(f"Warning: {zip_filename} not found, skipping")
            continue
        
        df = process_zip_file(zip_path)
        if not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("Error: No data found in any zip files")
        return
    
    # Combine all DataFrames
    print()
    print("Combining all datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total rows before sorting: {len(combined_df)}")
    
    # Convert TimeStamp to datetime (handle mixed formats)
    print("Converting TimeStamp to datetime...")
    combined_df['TimeStamp'] = pd.to_datetime(combined_df['TimeStamp'], format='mixed', errors='coerce')
    
    # Remove rows with invalid timestamps
    invalid_count = combined_df['TimeStamp'].isna().sum()
    if invalid_count > 0:
        print(f"Warning: Removing {invalid_count} rows with invalid timestamps")
        combined_df = combined_df.dropna(subset=['TimeStamp'])
    
    # Sort by TimeStamp
    print("Sorting by TimeStamp...")
    combined_df = combined_df.sort_values('TimeStamp').reset_index(drop=True)
    
    # Ensure all required columns exist (fill missing with NaN)
    for col in KEEP_COLUMNS:
        if col not in combined_df.columns:
            combined_df[col] = None
    
    # Reorder columns to match KEEP_COLUMNS
    combined_df = combined_df[KEEP_COLUMNS]
    
    # Save to CSV
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print()
    print("=" * 60)
    print("Dataset Build Complete!")
    print("=" * 60)
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df['TimeStamp'].min()} to {combined_df['TimeStamp'].max()}")
    print(f"Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_path = data_dir / "combined_dataset.csv"
    
    build_combined_dataset(data_dir, output_path)

