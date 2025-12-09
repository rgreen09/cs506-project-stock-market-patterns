"""
Triangle Pattern Dataset Builder

This script processes intraday stock data to build a labeled dataset for detecting
triangle patterns using sliding windows (e.g., 15 minutes = 300 bars at 3-second intervals).

Each row in the output corresponds to:
    (symbol, window_start_timestamp, window_end_timestamp)
with engineered numeric features and a binary label:
    label_triangle = 1 if the window looks like a triangle pattern, else 0.

Usage examples
--------------

Minimal (uses defaults: all symbols, 300-bar window):
    python build_triangle_dataset.py --input data/combined_dataset.csv

Specify output file:
    python build_triangle_dataset.py --input data/combined_dataset.csv --output triangles_15m_windows.csv

Change window size (e.g., 200 bars instead of 300):
    python build_triangle_dataset.py --input data/combined_dataset.csv --window-bars 200

Limit to specific symbols only:
    python build_triangle_dataset.py --input data/combined_dataset.csv --symbols MSFT,AAPL,NVDA
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ======================================================
# Triangle pattern detection helpers
# ======================================================

@dataclass
class TriangleConfig:
    min_extrema: int = 2          # minimum number of peaks/troughs
    min_slope_mag: float = 0.05   # minimum slope magnitude for converging lines (in normalized space)
    corridor_tolerance: float = 0.05
    min_fraction_inside: float = 0.8  # fraction of points that must lie inside the triangle corridor


def find_local_peaks(values: np.ndarray) -> List[int]:
    """Return indices of simple local maxima."""
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(i)
    return peaks


def find_local_troughs(values: np.ndarray) -> List[int]:
    """Return indices of simple local minima."""
    troughs = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            troughs.append(i)
    return troughs


def is_triangle_in_window(
    close_window: pd.Series,
    config: TriangleConfig
) -> bool:
    """
    Heuristic detection of a triangle pattern in one price window.

    Steps:
      - normalize time to [0, 1] and prices to [0, 1]
      - find local peaks and troughs (on a smoothed series)
      - fit upper trendline across peaks and lower trendline across troughs
      - require converging lines (upper sloping down, lower sloping up)
      - require narrowing corridor
      - require most points to lie within the corridor (± tolerance)
    """
    prices = close_window.values.astype(float)
    n = len(prices)

    if n < 10:
        return False

    # Normalize time as evenly spaced (we assume constant bar spacing)
    t = np.linspace(0.0, 1.0, n)

    # Normalize prices to [0, 1]
    p_min, p_max = prices.min(), prices.max()
    if p_max == p_min:
        # Flat window -> not a triangle
        return False
    p = (prices - p_min) / (p_max - p_min)

    # Smooth for peak/trough detection
    smooth = pd.Series(p).rolling(window=5, center=True, min_periods=1).mean().values

    peaks = find_local_peaks(smooth)
    troughs = find_local_troughs(smooth)

    if len(peaks) < config.min_extrema or len(troughs) < config.min_extrema:
        return False

    # Upper trendline: first and last peak
    p1_idx, p2_idx = peaks[0], peaks[-1]
    t1, t2 = t[p1_idx], t[p2_idx]
    if t2 == t1:
        return False
    upper_slope = (p[p2_idx] - p[p1_idx]) / (t2 - t1)

    # Lower trendline: first and last trough
    l1_idx, l2_idx = troughs[0], troughs[-1]
    lt1, lt2 = t[l1_idx], t[l2_idx]
    if lt2 == lt1:
        return False
    lower_slope = (p[l2_idx] - p[l1_idx]) / (lt2 - lt1)

    # Require converging lines:
    #   upper trendline sloping DOWN
    #   lower trendline sloping UP
    if upper_slope > -config.min_slope_mag:
        return False
    if lower_slope < config.min_slope_mag:
        return False

    # Compute corridor spread at start and end
    t_start, t_end = 0.0, 1.0

    upper_start = p[p1_idx] + upper_slope * (t_start - t1)
    lower_start = p[l1_idx] + lower_slope * (t_start - lt1)

    upper_end = p[p1_idx] + upper_slope * (t_end - t1)
    lower_end = p[l1_idx] + lower_slope * (t_end - lt1)

    spread_start = upper_start - lower_start
    spread_end = upper_end - lower_end

    if spread_start <= spread_end:
        # Corridor is not narrowing
        return False

    # Check fraction of points inside the corridor
    inside = 0
    for ti, pi in zip(t, p):
        upper_line = p[p1_idx] + upper_slope * (ti - t1)
        lower_line = p[l1_idx] + lower_slope * (ti - lt1)
        if (pi <= upper_line + config.corridor_tolerance) and (pi >= lower_line - config.corridor_tolerance):
            inside += 1

    frac_inside = inside / n
    if frac_inside < config.min_fraction_inside:
        return False

    return True


# ======================================================
# Feature engineering
# ======================================================

def linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y ~ x."""
    if len(x) < 2:
        return 0.0
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def compute_window_features(close_window: pd.Series) -> Dict[str, float]:
    """
    Compute engineered features from a window of closing prices.

    The features are generic price/volatility/shape stats that should be
    useful for classifying triangle vs non-triangle windows.
    """
    c = close_window.values.astype(float)
    n = len(c)

    features: Dict[str, float] = {}

    # Basic stats
    features["close_last"] = float(c[-1])
    features["close_mean"] = float(np.mean(c))
    features["close_std"] = float(np.std(c))
    features["price_range_abs"] = float(np.max(c) - np.min(c))
    features["price_range_pct"] = float(
        (features["price_range_abs"] / c[-1]) if c[-1] != 0 else 0.0
    )

    # Returns & momentum
    features["ret_total"] = float((c[-1] / c[0] - 1.0) if c[0] != 0 else 0.0)
    if n > 1 and c[-2] != 0:
        features["ret_1"] = float(c[-1] / c[-2] - 1.0)
    else:
        features["ret_1"] = 0.0

    if n > 5 and c[-6] != 0:
        features["ret_5"] = float(c[-1] / c[-6] - 1.0)
    else:
        features["ret_5"] = 0.0

    if n > 20 and c[-21] != 0:
        features["ret_20"] = float(c[-1] / c[-21] - 1.0)
    else:
        features["ret_20"] = 0.0

    # Simple linear trend features
    x_full = np.arange(n)
    features["slope_full"] = linear_regression_slope(x_full, c)

    mid = n // 2
    features["slope_first_half"] = linear_regression_slope(x_full[:mid], c[:mid])
    features["slope_second_half"] = linear_regression_slope(x_full[mid:], c[mid:])
    features["slope_diff"] = features["slope_second_half"] - features["slope_first_half"]

    # Volatility (over returns)
    if n > 1:
        rets = np.diff(c)
        features["ret_std"] = float(np.std(rets))
        features["ret_abs_mean"] = float(np.mean(np.abs(rets)))
    else:
        features["ret_std"] = 0.0
        features["ret_abs_mean"] = 0.0

    # Smooth series and local extremes
    smooth = pd.Series(c).rolling(window=5, center=True, min_periods=1).mean().values
    peaks = find_local_peaks(smooth)
    troughs = find_local_troughs(smooth)
    features["num_peaks"] = float(len(peaks))
    features["num_troughs"] = float(len(troughs))

    # Peak/trough geometry (simple)
    if len(peaks) >= 2 and len(troughs) >= 2:
        i1, i2 = peaks[-2], peaks[-1]
        features["peak_distance"] = float(i2 - i1)

        h1, h2 = smooth[i1], smooth[i2]
        max_peak = max(h1, h2)
        features["peak_height_diff_pct"] = float(
            abs(h1 - h2) / max_peak if max_peak != 0 else 0.0
        )
    else:
        features["peak_distance"] = 0.0
        features["peak_height_diff_pct"] = 0.0

    # Compression (ratio of min-to-max distance between extremes)
    p_min, p_max = np.min(c), np.max(c)
    features["compression"] = float(
        (p_max - p_min) / np.abs(c[0]) if c[0] != 0 else 0.0
    )

    return features


# ======================================================
# Dataset builder
# ======================================================

def build_triangle_dataset(
    input_csv_path: str,
    output_csv_path: str,
    symbols: Optional[List[str]] = None,
    window_bars: int = 300,
    config: Optional[TriangleConfig] = None,
) -> None:
    """
    Build a labeled dataset for triangle pattern detection for *all* specified symbols.

    Args:
        input_csv_path: path to input CSV file with columns: ID, TimeStamp, and stock symbols
        output_csv_path: path where output CSV will be saved
        symbols: list of symbols to process; if None, use all non-ID, non-TimeStamp columns
        window_bars: size of sliding window in bars (e.g., 300 for ~15 minutes at 3s bars)
        config: TriangleConfig with detection thresholds
    """
    if config is None:
        config = TriangleConfig()

    print(f"Loading data from {input_csv_path} ...")
    df = pd.read_csv(input_csv_path)

    if "TimeStamp" not in df.columns:
        raise ValueError("CSV must contain a 'TimeStamp' column")

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    df = df.sort_values("TimeStamp").reset_index(drop=True)

    # Infer symbols if not provided
    if symbols is None:
        symbols = [c for c in df.columns if c not in ("ID", "TimeStamp")]
        print(f"No symbols specified; using all detected symbols: {symbols}")
    else:
        # Validate symbols
        for s in symbols:
            if s not in df.columns:
                raise ValueError(f"Symbol '{s}' not found in CSV columns.")

    rows = []
    n_rows = len(df)
    print(f"Total rows: {n_rows:,}")
    print(f"Window size (bars): {window_bars}")

    if n_rows < window_bars:
        print("Not enough rows for even one window. Exiting.")
        return

    # Sliding windows over the *index* (time-ordered)
    # We reuse the same window indices for all symbols.
    for end_idx in range(window_bars - 1, n_rows):
        if (end_idx % 10000) == 0:
         print(f"Processed {end_idx} / {n_rows} rows...")

        start_idx = end_idx - window_bars + 1

        start_ts = df.loc[start_idx, "TimeStamp"]
        end_ts = df.loc[end_idx, "TimeStamp"]

        for symbol in symbols:
            price_series = df[symbol].copy()
            if price_series.isna().any():
                price_series = price_series.ffill().bfill()

            close_window = price_series.iloc[start_idx : end_idx + 1]

            if len(close_window) < window_bars:
                continue  # safety check

            # Features
            feats = compute_window_features(close_window)

            # Label
            label = 1 if is_triangle_in_window(close_window, config) else 0

            row = {
                "symbol": symbol,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                **feats,
                "label_triangle": label,
            }
            rows.append(row)

    out_df = pd.DataFrame(rows)
    print(f"Built {len(out_df):,} window rows.")
    print(f"Triangle windows: {int(out_df['label_triangle'].sum())} "
          f"({100 * out_df['label_triangle'].mean() if len(out_df) > 0 else 0:.2f}%)")

    out_df.to_csv(output_csv_path, index=False)
    print(f"Saved dataset to {output_csv_path}")


# ======================================================
# CLI
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build labeled dataset for triangle pattern detection (all stocks)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (must contain 'TimeStamp' and stock columns).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="triangle_windows_dataset.csv",
        help="Path to output CSV file (default: triangle_windows_dataset.csv).",
    )
    parser.add_argument(
        "--window-bars",
        type=int,
        default=300,
        help="Sliding window size in bars (e.g., 300 for ~15 minutes at 3-second bars).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to include (e.g., MSFT,AAPL,NVDA). "
             "Default: all non-ID, non-TimeStamp columns.",
    )

    args = parser.parse_args()

    if args.symbols is not None:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = None

    config = TriangleConfig(
        min_extrema=2,
        min_slope_mag=0.05,
        corridor_tolerance=0.05,
        min_fraction_inside=0.8,
    )

    build_triangle_dataset(
        input_csv_path=args.input,
        output_csv_path=args.output,
        symbols=symbols,
        window_bars=args.window_bars,
        config=config,
    )


if __name__ == "__main__":
    main()
