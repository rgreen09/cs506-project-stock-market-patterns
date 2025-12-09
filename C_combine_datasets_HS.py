"""
Combine all Head & Shoulders dataset CSV files in the SAME directory.
Looks for files matching: *_HS_dataset.csv
"""

import pandas as pd
import os
from pathlib import Path

def combine_hs_datasets(output_file="combined_dataset_HS.csv"):
    """
    Combines all *_HS_dataset.csv files in the SAME folder as this script.
    """
    script_dir = Path(__file__).resolve().parent
    csv_files = sorted(script_dir.glob("*_HS_dataset.csv"))

    if not csv_files:
        print("❌ No HS dataset CSVs found in this folder.")
        print(f"Checked folder: {script_dir}")
        return

    print(f"✅ Found {len(csv_files)} HS dataset files:")
    for f in csv_files:
        print(f"   - {f.name}")

    # Read header from first file
    first_file = csv_files[0]
    sample_df = pd.read_csv(first_file, nrows=1)
    columns = sample_df.columns.tolist()

    print("\n📝 Columns detected:")
    print(columns)

    print(f"\n📌 Writing combined dataset → {output_file}")

    first_chunk = True
    total_rows = 0
    chunk_size = 10000

    for csv_file in csv_files:
        print(f"\n➡️ Processing {csv_file.name}...")
        file_rows = 0

        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            file_rows += len(chunk)
            total_rows += len(chunk)

            if first_chunk:
                chunk.to_csv(output_file, mode='w', index=False, header=True)
                first_chunk = False
            else:
                chunk.to_csv(output_file, mode='a', index=False, header=False)

        print(f"   ✓ Added {file_rows:,} rows")

    print("\n🎉 DONE combining datasets!")
    print(f"   Total rows combined: {total_rows:,}")
    print(f"   Output file: {output_file}")

    # Quick verification
    preview = pd.read_csv(output_file, nrows=5)
    print("\n🔍 Preview of combined file:")
    print(preview)

if __name__ == "__main__":
    combine_hs_datasets()
