"""
Exploratory visualizations for combined_dataset.csv.

Generates PNG plots into visualizations/data_exploration/outputs by default:
- Label/class balance (if a label-like column exists)
- Feature distributions for numeric columns
- Correlation heatmap for numeric columns
- Time series overview if a timestamp-like column exists

Usage:
    python visualizations/data_exploration/explore_combined_dataset.py \\
        --input data/combined_dataset.csv \\
        --output visualizations/data_exploration/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_OUTPUT_DIR = Path("visualizations/data_exploration/outputs")
POSSIBLE_TIME_COLS = ["timestamp", "time", "datetime", "date", "TimeStamp"]
POSSIBLE_LABEL_COLS = [
    "label",
    "target",
    "class",
    "pattern",
    "Pattern",
    "Label",
]
POSSIBLE_ID_COLS = ["id", "ID"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Parse timestamp if present
    for col in POSSIBLE_TIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def find_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    exclude_set = set(exclude)
    cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if c not in exclude_set
    ]
    return cols


def plot_label_balance(df: pd.DataFrame, label_col: str, output_dir: Path) -> None:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="crest")
    plt.title(f"Class Distribution: {label_col}")
    plt.xlabel(label_col)
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    output_path = output_dir / "label_balance.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[saved] {output_path}")


def plot_feature_distributions(
    df: pd.DataFrame, numeric_cols: Sequence[str], output_dir: Path
) -> None:
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=40, color="steelblue")
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        output_path = output_dir / f"dist_{col}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"[saved] {output_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame, numeric_cols: Sequence[str], output_dir: Path
) -> None:
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Correlation Heatmap")
    output_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[saved] {output_path}")


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    numeric_cols: Sequence[str],
    output_dir: Path,
    max_points: int = 10_000,
) -> None:
    ts_df = df[[time_col, *numeric_cols]].dropna(subset=[time_col]).copy()
    ts_df = ts_df.sort_values(time_col)
    # Downsample if very large to keep plot readable
    if len(ts_df) > max_points:
        ts_df = ts_df.iloc[:: max(1, len(ts_df) // max_points)]
    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        plt.plot(ts_df[time_col], ts_df[col], label=col, linewidth=1)
    plt.title("Time Series Overview")
    plt.xlabel(time_col)
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "time_series.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[saved] {output_path}")


def run_visualizations(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_path)

    time_col = find_first(df, POSSIBLE_TIME_COLS)
    label_col = find_first(df, POSSIBLE_LABEL_COLS)
    id_col = find_first(df, POSSIBLE_ID_COLS)

    exclude_cols = [c for c in [time_col, label_col, id_col] if c]
    numeric_cols = numeric_feature_columns(df, exclude=exclude_cols)

    print(f"[info] loaded {len(df):,} rows from {input_path}")
    print(f"[info] time column: {time_col or 'none'}")
    print(f"[info] label column: {label_col or 'none'}")
    print(f"[info] numeric columns: {numeric_cols}")

    if label_col:
        plot_label_balance(df, label_col, output_dir)
    else:
        print("[skip] no label column found; label balance plot skipped")

    if numeric_cols:
        plot_feature_distributions(df, numeric_cols, output_dir)
        plot_correlation_heatmap(df, numeric_cols, output_dir)
    else:
        print("[skip] no numeric columns found; distribution/heatmap skipped")

    if time_col and numeric_cols:
        plot_time_series(df, time_col, numeric_cols, output_dir)
    else:
        print("[skip] time series plot skipped (missing time or numeric columns)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG visualizations for combined_dataset.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/combined_dataset.csv"),
        help="Path to combined dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save PNG outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visualizations(args.input, args.output)


if __name__ == "__main__":
    main()

