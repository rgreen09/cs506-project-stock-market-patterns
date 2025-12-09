"""
Build a Triangle dataset for ONE symbol.

Usage (WIDE input like combined_dataset.csv):
    python N_builder_triangles.py \
        --input combined_dataset.csv \
        --output MSFT_triangle_dataset.csv \
        --symbol MSFT

Also works with LONG input of the form:
    columns = ['timestamp', 'symbol', 'close']
"""

import argparse
import pandas as pd

# NOTE: this matches your N_Triangles_detector.py,
# which defines: def is_triangle(close_window: pd.Series) -> bool
from N_Triangles_detector import is_triangle  # ✅ correct name


# Sliding window length (rows). You can change this if you want.
WINDOW = 15   # similar to your friend's H&S example


def detect_triangle(window: pd.DataFrame) -> bool:
    """
    Wrapper around the triangle detector.
    Expects a window with a 'close' column.
    """
    close_series = window["close"]
    return bool(is_triangle(close_series))   # ✅ use is_triangle from detector


def build_dataset_long(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Build dataset when the input is in LONG format:
        timestamp, symbol, close
    """
    df = df[df["symbol"] == symbol].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    results = []
    for i in range(len(df) - WINDOW):
        w = df.iloc[i:i + WINDOW]
        label = int(detect_triangle(w))
        results.append({
            "timestamp": df.loc[i + WINDOW - 1, "timestamp"],
            "symbol": symbol,
            "close": df.loc[i + WINDOW - 1, "close"],
            "label_triangle": label,
        })

    return pd.DataFrame(results)


def build_dataset_wide(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Build dataset when the input is in WIDE format, e.g.:

        ID, TimeStamp, MSFT, AAPL, NVDA, ...

    We:
      - use TimeStamp as 'timestamp'
      - use the chosen symbol column as 'close'
      - add a 'symbol' column with the symbol name
    """
    if "TimeStamp" not in df.columns:
        raise ValueError("Wide-format input must have a 'TimeStamp' column.")
    if symbol not in df.columns:
        raise ValueError(
            f"Symbol '{symbol}' not found in input columns. "
            f"Available price columns: {[c for c in df.columns if c not in ['ID', 'TimeStamp']]}"
        )

    # Create a minimal long-style DataFrame for this symbol
    df_sym = df[["TimeStamp", symbol]].copy()
    df_sym = df_sym.rename(columns={"TimeStamp": "timestamp", symbol: "close"})
    df_sym["symbol"] = symbol

    df_sym = df_sym.sort_values("timestamp").reset_index(drop=True)

    results = []
    n = len(df_sym)
    for i in range(n - WINDOW):
        w = df_sym.iloc[i:i + WINDOW]
        label = int(detect_triangle(w))
        results.append({
            "timestamp": df_sym.loc[i + WINDOW - 1, "timestamp"],
            "symbol": symbol,
            "close": df_sym.loc[i + WINDOW - 1, "close"],
            "label_triangle": label,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output Triangle dataset CSV.")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., MSFT, AAPL, NVDA).")
    args = parser.parse_args()

    print(f"Loading dataset: {args.input}")
    df = pd.read_csv(args.input)

    cols = set(df.columns)
    required_long = {"timestamp", "symbol", "close"}

    # Decide whether input is LONG or WIDE
    if required_long.issubset(cols):
        print("Detected LONG-format input (timestamp, symbol, close).")
        out_df = build_dataset_long(df, args.symbol)
    else:
        print("Detected WIDE-format input (TimeStamp + per-symbol columns).")
        out_df = build_dataset_wide(df, args.symbol)

    print(f"Built Triangle dataset for: {args.symbol}")
    print(f"Rows: {len(out_df):,}")
    if len(out_df) > 0:
        positives = int(out_df["label_triangle"].sum())
        print(f"Triangle windows: {positives} ({100 * positives / len(out_df):.4f}%)")

    print(f"Saving to: {args.output}")
    out_df.to_csv(args.output, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
