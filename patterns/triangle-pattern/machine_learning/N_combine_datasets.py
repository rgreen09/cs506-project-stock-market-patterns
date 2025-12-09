"""
Combine all Triangle dataset CSV files in the SAME directory.

Looks for files matching: *_triangle_dataset.csv

Example:
    AAPL_triangle_dataset.csv
    MSFT_triangle_dataset.csv
    NVDA_triangle_dataset.csv
    ...

Output:
    combined_dataset_triangle.csv   (by default)

Usage:
    python combine_triangle_datasets.py
    # or:
    # python combine_triangle_datasets.py --output my_combined_triangle.csv
"""

import pandas as pd
from pathlib import Path
import argparse


def combine_triangle_datasets(output_file: str = "combined_dataset_triangle.csv") -> None:
    """
    Combines all *_triangle_dataset.csv files in the SAME folder as this script.

    Assumes each per-symbol file was created by build_triangle_dataset.py and
    contains columns like:
        ['timestamp', 'symbol', 'close', 'label_triangle']
    """
    script_dir = Path(__file__).resolve().parent
    csv_files = sorted(script_dir.glob("*_triangle_dataset.csv"))

    if not csv_files:
        print("❌ No triangle dataset CSVs found in this folder.")
        print(f"Checked folder: {script_dir}")
        return

    print(f"✅ Found {len(csv_files)} triangle dataset files:")
    for f in csv_files:
        print(f"   - {f.name}")

    # Read header from first file
    first_file = csv_files[0]
    sample_df = pd.read_csv(first_file, nrows=1)
    columns = sample_df.columns.tolist()

    print("\n📝 Columns detected in first file:")
    print(columns)

    print(f"\n📌 Writing combined dataset → {output_file}")

    first_chunk = True
    total_rows = 0
    chunk_size = 10000

    for csv_file in csv_files:
        print(f"\n➡️ Processing {csv_file.name}...")
        file_rows = 0

        # Stream file in chunks to avoid memory blowup for large datasets
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            file_rows += len(chunk)
            total_rows += len(chunk)

            # Keep column order consistent with first file
            chunk = chunk[columns]

            if first_chunk:
                # Write header only once
                chunk.to_csv(output_file, mode="w", index=False, header=True)
                first_chunk = False
            else:
                chunk.to_csv(output_file, mode="a", index=False, header=False)

        print(f"   ✓ Added {file_rows:,} rows")

    print("\n🎉 DONE combining triangle datasets!")
    print(f"   Total rows combined: {total_rows:,}")
    print(f"   Output file: {output_file}")

    # Quick verification
    try:
        preview = pd.read_csv(output_file, nrows=5)
        print("\n🔍 Preview of combined file (first 5 rows):")
        print(preview)
    except Exception as e:
        print(f"\n⚠️ Could not read preview from {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all *_triangle_dataset.csv files in this directory "
                    "into a single combined_dataset_triangle.csv file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_dataset_triangle.csv",
        help="Output CSV file name (default: combined_dataset_triangle.csv)",
    )
    args = parser.parse_args()

    combine_triangle_datasets(output_file=args.output)


if __name__ == "__main__":
    main()
