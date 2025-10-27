# fetch_one_minute.py
# Fetches intraday minute-level OHLCV data from Alpha Vantage for the past 365 days.
# Automatically rotates through your API_KEYS list if a key hits its limit.

import csv
import time
import pathlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ================================================================
# 🔑 ALPHA VANTAGE FREE API KEYS HERE
# ================================================================
API_KEYS = [
    "ZE4MEE8PVMBI6IGX",
    "IFFP8QX6DTST8M50",
    "C9BN6PB59APKSG9X",
    "8DYIUM5CRLWSFVW4",
    "UW28D1JUXWMAYQLK",
    "LTNHSI042UKPG0FU",
    "7LZ0QYKJLE41CBZ1",
    "DPXY8VRXCMBN7X29",
    "SM7X8Y0F0GC6OJEQ",
    "2U8Q63KPWGLX9AGK",
    "C9XG5M5KRIPQME94",
    "XXP1R4KZ8CX08LFO",
    "QS8MYECVUA30LL6C",
    "S1X1OCJGYIRF7344",
    "PUCJDE86YP1PXIK8",
    "9J3HOIP30YDQPD6C",
    "VJY88CSPDX56SYLH",
    "VCRJKZ16YGX0BULA",
    "LPYYWKVW8EIUCPE1",
    "GAB6N0I3CFRVAOWS",
    "M170KFBB6DK95648",
    "ABLS5VPNWM4UTNCJ",
    "VX0V0IUH5YNZ38I7",
    "QUFF3LR09BUZCQIH",
    "RKKF9UWK4N7NONJG"
]
# ================================================================

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(Exception):
    pass


def _normalize_row(datetime_str: str, row: Dict[str, str]) -> Dict[str, str]:
    return {
        "datetime": datetime_str,
        "open": row.get("1. open"),
        "high": row.get("2. high"),
        "low": row.get("3. low"),
        "close": row.get("4. close"),
        "volume": row.get("5. volume"),
    }


def _try_fetch(symbol: str, api_key: str, interval: str, timeout: int) -> Dict:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "Error Message" in data:
        raise AlphaVantageError(data["Error Message"])
    if "Note" in data:
        raise AlphaVantageError(data["Note"])
    if "Information" in data:
        raise AlphaVantageError(f"API Information: {data['Information']}")
    if "Meta Data" not in data:
        raise AlphaVantageError(f"Unexpected response: {list(data.keys())}")
    return data


def fetch_intraday_past_365d(
    symbol: str,
    out_dir: pathlib.Path,
    interval: str = "1min",
    timeout: int = 60,
    backoff_sec: float = 5.0,
) -> Tuple[pathlib.Path, int]:
    """
    Fetch the past 365 days of minute-level OHLCV data for one ticker.
    Automatically rotates through API_KEYS if a key fails.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        out_dir: Directory to save the CSV file
        interval: Data interval ("1min", "5min", "15min", "30min", "60min")
        timeout: Request timeout in seconds
        backoff_sec: Seconds to wait between retries
        
    Returns:
        Tuple of (csv_path, row_count)
    """
    if not API_KEYS:
        raise RuntimeError("No API keys in API_KEYS list — add them at the top of this file!")
    
    # Note: Alpha Vantage free tier has limitations on intraday data
    # - For 1min data, only last 30 days
    # - For 5min/15min/30min/60min, up to 30 days
    # This will fetch whatever data is available and filter to last 365 days if possible
    
    data = None
    for key in API_KEYS:
        try:
            data = _try_fetch(symbol, key, interval, timeout)
            break
        except (AlphaVantageError, requests.RequestException) as e:
            print(f"{symbol}: key {key[:5]}... failed -> {e}")
            time.sleep(backoff_sec)
    if data is None:
        raise RuntimeError(f"All API keys failed for {symbol}")
    
    # Extract time series data - key format varies by interval
    series_key = f"Time Series ({interval})"
    if series_key not in data:
        # Try alternative key format
        series_key = None
        for key in data.keys():
            if "Time Series" in key:
                series_key = key
                break
        if series_key is None:
            raise AlphaVantageError(f"No time series data found in response: {list(data.keys())}")
    
    series = data.get(series_key, {})
    
    # Calculate cutoff date (365 days ago)
    cutoff_date = datetime.now() - timedelta(days=365)
    
    rows = []
    for datetime_str, row in series.items():
        try:
            # Parse datetime string (format: "2024-01-15 10:30:00")
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Try alternative format if first one fails
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                print(f"Warning: Could not parse datetime '{datetime_str}', skipping")
                continue
        
        # Only include data from the last 365 days
        if dt >= cutoff_date:
            rows.append(_normalize_row(datetime_str, row))
    
    # Sort by datetime
    rows.sort(key=lambda r: r["datetime"])
    
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{symbol}_minute_{interval}_365d.csv"
    
    fieldnames = ["datetime", "open", "high", "low", "close", "volume"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return csv_path, len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Fetch intraday minute-level OHLCV data for a stock (past 365 days)"
    )
    ap.add_argument("--symbol", required=True, help="Ticker symbol, e.g., NVDA")
    ap.add_argument("--out", default="data", help="Output folder (default: data/)")
    ap.add_argument(
        "--interval",
        default="1min",
        choices=["1min", "5min", "15min", "30min", "60min"],
        help="Data interval (default: 1min)",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    
    print(f"Fetching {args.interval} data for {args.symbol.upper()}...")
    print("Note: Free tier intraday data is limited to last 30 days for 1min data.")
    
    csv_path, n = fetch_intraday_past_365d(
        args.symbol.upper(),
        out_dir,
        interval=args.interval
    )
    print(f"✅ Wrote {csv_path} ({n} rows)")

