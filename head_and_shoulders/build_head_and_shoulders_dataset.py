"""
Build a Head & Shoulders dataset for ONE symbol.

Usage:
    python build_head_and_shoulders_dataset.py \
        --input combined_dataset.csv \
        --output AAPL_HS_dataset.csv \
        --symbol AAPL
"""

import pandas as pd
import argparse

WINDOW = 15   # 15-row sliding window

def detect_head_and_shoulders(window):
    # Simple pattern detection logic
    prices = window["close"].values
    left = prices[:5].mean()
    head = prices[5:10].mean()
    right = prices[10:].mean()

    return head > left * 1.02 and head > right * 1.02


def build_dataset(df, symbol):
    df = df[df["symbol"] == symbol].copy()
    df.reset_index(drop=True, inplace=True)

    results = []
    for i in range(len(df) - WINDOW):
        w = df.iloc[i:i+WINDOW]
        label = int(detect_head_and_shoulders(w))
        results.append({
            "timestamp": df.loc[i+WINDOW-1, "timestamp"],
            "symbol": symbol,
            "close": df.loc[i+WINDOW-1, "close"],
            "label": label
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    print(f"Loading dataset: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Building Head & Shoulders dataset for: {args.symbol}")
    out_df = build_dataset(df, args.symbol)

    print(f"Saving to: {args.output}")
    out_df.to_csv(args.output, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
