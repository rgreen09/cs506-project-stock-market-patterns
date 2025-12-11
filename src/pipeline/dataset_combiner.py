import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import load_config


class DatasetCombiner:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path) if config_path else load_config()

    def combine(self, pattern: str, output_file: Optional[str] = None) -> str:
        data_cfg = self.config["data"]
        output_dir = data_cfg["output_dir"]
        pattern_cfg = self.config["patterns"][pattern]
        label_col = pattern_cfg["label_column"]

        if output_file is None:
            output_file = os.path.join(output_dir, f"combined_{pattern}_windows.csv")

        csv_files = sorted(glob.glob(os.path.join(output_dir, f"*_{pattern}_windows.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for pattern '{pattern}' in {output_dir}")

        first_file = csv_files[0]
        sample_df = pd.read_csv(first_file, nrows=1)
        columns = sample_df.columns.tolist()

        first_chunk = True
        chunk_size = 10000
        total_rows = 0
        for csv_file in csv_files:
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                if first_chunk:
                    chunk.to_csv(output_file, mode="w", index=False, header=True)
                    first_chunk = False
                else:
                    chunk.to_csv(output_file, mode="a", index=False, header=False)
                total_rows += len(chunk)

        verify_df = pd.read_csv(output_file, nrows=5)
        if label_col not in verify_df.columns:
            raise ValueError(f"Combined file missing label column '{label_col}'")
        return output_file

