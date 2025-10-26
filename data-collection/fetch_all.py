# fetch_all.py
# Fetches the last 10 years of daily OHLCV data for all S&P 500 companies.
# Uses fetch_one.fetch_daily_last_10y for each ticker.

import pathlib
from fetch_one import fetch_daily_last_10y
from sp500_companies import sp500_companies

if __name__ == "__main__":
    out_dir = pathlib.Path("data")
    
    # Get all ticker symbols (values from the dictionary)
    tickers = list(sp500_companies.values())
    total = len(tickers)
    
    print(f"Fetching data for {total} S&P 500 companies...")
    print(f"Output directory: {out_dir}")
    print()
    
    successful = 0
    failed = []
    
    for i, ticker in enumerate(tickers, start=1):
        try:
            print(f"[{i}/{total}] Fetching {ticker}...", end=" ")
            csv_path, num_rows = fetch_daily_last_10y(ticker, out_dir)
            print(f"✅ {num_rows} rows")
            successful += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed.append(ticker)
    
    print()
    print("=" * 60)
    print(f"Complete! Successfully fetched: {successful}/{total}")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")
    print("=" * 60)

