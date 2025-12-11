import argparse
from typing import Optional, Tuple

import pandas as pd

from src.utils.io import ensure_parent_dir

DEFAULT_INPUT = "data/combined_dataset.csv"
DEFAULT_OUTPUT = "data/test/combined_subset.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"


def _determine_start_date(
    input_path: str, timestamp_col: str, start_date: Optional[str], chunksize: int
) -> pd.Timestamp:
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"Could not parse start_date '{start_date}'")
        return start.normalize()

    min_ts: Optional[pd.Timestamp] = None
    for chunk in pd.read_csv(
        input_path,
        usecols=[timestamp_col],
        parse_dates=[timestamp_col],
        chunksize=chunksize,
    ):
        ts = chunk[timestamp_col].dropna()
        if ts.empty:
            continue
        chunk_min = ts.min().normalize()
        if min_ts is None or chunk_min < min_ts:
            min_ts = chunk_min

    if min_ts is None:
        raise ValueError(f"No valid timestamps found in '{timestamp_col}'")
    return min_ts


def create_subset(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    days: int = 2,
    start_date: Optional[str] = None,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    chunksize: int = 200_000,
) -> Tuple[str, int, pd.Timestamp, pd.Timestamp]:
    """
    Create a small date-bounded subset of a large CSV for testing.
    Returns (output_path, rows_written, start_ts, end_ts).
    """
    if days <= 0:
        raise ValueError("days must be positive")

    start_ts = _determine_start_date(input_path, timestamp_col, start_date, chunksize)
    end_ts = start_ts + pd.Timedelta(days=days)

    ensure_parent_dir(output_path)
    wrote_header = False
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        parse_dates=[timestamp_col],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=[timestamp_col])
        mask = (chunk[timestamp_col] >= start_ts) & (chunk[timestamp_col] < end_ts)
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue

        filtered = filtered.sort_values(timestamp_col)
        mode = "w" if not wrote_header else "a"
        filtered.to_csv(output_path, mode=mode, index=False, header=not wrote_header)
        wrote_header = True
        total_rows += len(filtered)

    if not wrote_header:
        raise ValueError(
            f"No rows found between {start_ts.date()} and {end_ts.date()} "
            f"using column '{timestamp_col}'"
        )

    return output_path, total_rows, start_ts, end_ts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small subset for testing.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source CSV path.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (directories auto-created).",
    )
    parser.add_argument("--days", type=int, default=2, help="Number of days to include.")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD). Defaults to earliest date in file.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=DEFAULT_TIMESTAMP_COL,
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streaming the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path, rows, start_ts, end_ts = create_subset(
        input_path=args.input,
        output_path=args.output,
        days=args.days,
        start_date=args.start_date,
        timestamp_col=args.timestamp_col,
        chunksize=args.chunksize,
    )
    print(
        f"Wrote {rows} rows to {output_path} "
        f"for {start_ts.date()} through {end_ts.date()} (exclusive)."
    )


if __name__ == "__main__":
    main()

