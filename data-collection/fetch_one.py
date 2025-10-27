# fetch_one.py
# Fetches a single ticker's last 10 years of DAILY OHLCV (adjusted) from Alpha Vantage.
# Automatically rotates through your API_KEYS list if a key hits its limit.

import csv
import time
import pathlib
import requests
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

# ================================================================
# 🔑 ALPHA VANTAGE FREE API KEYS HERE
# ================================================================
API_KEYS = [
    "6AVQ8COIP2615TOQ",
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


def _normalize_row(datestr: str, row: Dict[str, str], adjusted: bool = False) -> Dict[str, str]:
    if adjusted:
        return {
            "date": datestr,
            "open": row.get("1. open"),
            "high": row.get("2. high"),
            "low": row.get("3. low"),
            "close": row.get("4. close"),
            "volume": row.get("6. volume"),
            "adjusted_close": row.get("5. adjusted close"),
            "dividend_amount": row.get("7. dividend amount"),
            "split_coefficient": row.get("8. split coefficient"),
        }
    else:
        return {
            "date": datestr,
            "open": row.get("1. open"),
            "high": row.get("2. high"),
            "low": row.get("3. low"),
            "close": row.get("4. close"),
            "volume": row.get("5. volume"),
            "adjusted_close": None,  # Not available in free endpoint
            "dividend_amount": None,  # Not available in free endpoint
            "split_coefficient": None,  # Not available in free endpoint
        }


def _try_fetch(symbol: str, api_key: str, adjusted: bool, timeout: int) -> Dict:
    function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    params = {
        "function": function,
        "symbol": symbol,
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


def fetch_daily_last_10y(
    symbol: str,
    out_dir: pathlib.Path,
    adjusted: bool = False,  # Changed default to False since adjusted is now premium
    timeout: int = 60,
    backoff_sec: float = 5.0,
) -> Tuple[pathlib.Path, int]:
    """
    Fetch the last 10 years of daily OHLCV data for one ticker.
    Automatically rotates through API_KEYS if a key fails.
    """
    if not API_KEYS:
        raise RuntimeError("No API keys in API_KEYS list — add them at the top of this file!")

    data = None
    for key in API_KEYS:
        try:
            data = _try_fetch(symbol, key, adjusted, timeout)
            break
        except (AlphaVantageError, requests.RequestException) as e:
            print(f"{symbol}: key {key[:5]}... failed -> {e}")
            time.sleep(backoff_sec)
    if data is None:
        raise RuntimeError(f"All API keys failed for {symbol}")

    series = data.get("Time Series (Daily)", {})
    ten_years_ago = date.today() - timedelta(days=3652)
    rows = []
    for d, row in series.items():
        try:
            y, m, dd = map(int, d.split("-"))
            dt = date(y, m, dd)
        except ValueError:
            continue
        if dt >= ten_years_ago:
            rows.append(_normalize_row(d, row, adjusted))
    rows.sort(key=lambda r: r["date"])

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{symbol}_daily_10y.csv"
    
    # Define fieldnames based on whether we have adjusted data
    fieldnames = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    if adjusted:
        fieldnames.extend([
            "adjusted_close",
            "dividend_amount",
            "split_coefficient",
        ])
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # Only write fields that are defined in fieldnames
            filtered_row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(filtered_row)
    return csv_path, len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Ticker symbol, e.g., NVDA")
    ap.add_argument("--out", default="data", help="Output folder (default: data/)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    csv_path, n = fetch_daily_last_10y(args.symbol.upper(), out_dir)
    print(f"✅ Wrote {csv_path} ({n} rows)")
