# double_top_detector.py
# Detect Double Top chart patterns in daily OHLCV data.
# Author: CS506 Project (Raymond) — 2025-10-26

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# Optional: if available, we can draw candlesticks with mplfinance; otherwise we fallback to a line chart.
try:
    import mplfinance as mpf
    _HAS_MPLFIN = True
except Exception:
    _HAS_MPLFIN = False

import matplotlib.pyplot as plt


# =============================
# 1) Config and return objects
# =============================

@dataclass
class PatternParams:
    # Extrema detection
    lookback_extrema: int = 3          # neighborhood half-width for strict local max/min
    ema_span: Optional[int] = 3        # smoothing span (None to disable)
    # Peak pair rules
    min_sep: int = 5                   # min bars between peaks
    max_sep: int = 40                  # max bars between peaks
    peak_tolerance_pct: float = 2.0    # peaks within ±X% of each other
    min_trough_drop_pct: float = 3.0   # each peak must drop to trough by at least X%
    # Trend (optional but helpful)
    require_uptrend: bool = True
    uptrend_lookback: int = 20
    uptrend_min_return_pct: float = 5.0
    # Confirmation rules
    confirm_lookahead: int = 15
    break_margin_pct: float = 0.5      # need close <= neckline*(1 - margin/100)
    # Volume filter (optional)
    use_volume_filter: bool = True
    vol_ma_window: int = 20
    vol_confirm_mult: float = 1.2      # confirm bar vol >= mult * SMA(volume)
    require_vol_divergence: bool = False  # if True, Volume at peak1 > Volume at peak2
    # De-duplication & scoring
    prefer_earlier_confirm: bool = True


@dataclass
class DoubleTop:
    symbol: Optional[str]
    peak1_idx: int
    peak2_idx: int
    trough_idx: int
    confirm_idx: int
    peak1: float
    peak2: float
    trough: float
    neckline: float
    peak_diff_pct: float
    drop1_pct: float
    drop2_pct: float
    separation_bars: int
    confirm_delay_bars: int
    score: float
    peak1_date: pd.Timestamp
    peak2_date: pd.Timestamp
    trough_date: pd.Timestamp
    confirm_date: pd.Timestamp


# =============================
# 2) Public API
# =============================

def detect_double_tops(df: pd.DataFrame, params: PatternParams, symbol: Optional[str] = None) -> List[DoubleTop]:
    """
    High-level orchestrator that returns a list of confirmed DoubleTop objects.
    Expects df with columns: Open, High, Low, Close, Volume and a DatetimeIndex or a 'Date' column.
    """
    data = ensure_ohlcv(df)
    px_close = data["Close"].astype(float)

    # Smoothing for extrema (improves stability of peak/trough detection)
    px_for_extrema = apply_ema(px_close, params.ema_span) if params.ema_span and params.ema_span > 1 else px_close

    is_peak, is_trough = find_local_extrema(px_for_extrema, window=params.lookback_extrema)
    peak_idx = np.where(is_peak)[0]
    trough_idx_flags = np.where(is_trough)[0]  # not used directly, but available

    # Candidate pairs (consecutive peaks with a trough between)
    candidates = pair_peaks_with_troughs(
        data,
        peak_idx=peak_idx,
        low_vals=data["Low"].values,
        params=params
    )

    # Precompute helpers
    vol_ma = data["Volume"].rolling(params.vol_ma_window, min_periods=1).mean() if "Volume" in data.columns else None

    results: List[DoubleTop] = []
    n = len(data)

    for (p1, p2, t_idx) in candidates:
        # Rule checks
        ok_sim, peak_diff = passes_peak_similarity(data, p1, p2, params.peak_tolerance_pct)
        if not ok_sim:
            continue

        ok_drop, drop1, drop2 = passes_trough_depth(data, p1, p2, t_idx, params.min_trough_drop_pct)
        if not ok_drop:
            continue

        if params.require_uptrend and not passes_uptrend(data, p1, params):
            continue

        neckline = float(data["Low"].iloc[t_idx])

        c_idx = confirm_break(data, p2, neckline, params)
        if c_idx is None:
            continue

        # Optional volume filter
        if params.use_volume_filter and "Volume" in data.columns:
            if not passes_volume(data, p1, p2, c_idx, vol_ma, params):
                continue

        # Build quality score
        delay = c_idx - p2
        score = quality_score(peak_diff, drop1, drop2, delay, params)

        results.append(DoubleTop(
            symbol=symbol,
            peak1_idx=p1,
            peak2_idx=p2,
            trough_idx=t_idx,
            confirm_idx=c_idx,
            peak1=float(data["High"].iloc[p1]),
            peak2=float(data["High"].iloc[p2]),
            trough=float(data["Low"].iloc[t_idx]),
            neckline=neckline,
            peak_diff_pct=float(peak_diff),
            drop1_pct=float(drop1),
            drop2_pct=float(drop2),
            separation_bars=int(p2 - p1),
            confirm_delay_bars=int(delay),
            score=float(score),
            peak1_date=data.index[p1],
            peak2_date=data.index[p2],
            trough_date=data.index[t_idx],
            confirm_date=data.index[c_idx],
        ))

    # De-duplicate overlapping detections and sort
    results = deduplicate_overlaps(results, prefer_earliest=params.prefer_earlier_confirm)
    results.sort(key=lambda r: (r.confirm_date, -r.score))
    return results


def plot_double_tops(df: pd.DataFrame, patterns: List[DoubleTop], start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, title: Optional[str] = None):
    """
    Visualize detections. Uses candlesticks if mplfinance is available, else a Close line chart.
    Draws markers for peaks/trough and a dashed neckline, plus a vertical marker on confirmation.
    """
    data = ensure_ohlcv(df)
    if start is None:
        start = data.index.min()
    if end is None:
        end = data.index.max()

    view = data.loc[(data.index >= start) & (data.index <= end)].copy()
    if view.empty:
        print("No data in the requested plotting window.")
        return

    if _HAS_MPLFIN:
        # Use mplfinance candlesticks; add overlays via make_addplot & annotate with matplotlib
        apds = []
        fig, axlist = mpf.plot(
            view,
            type='candle',
            style='yahoo',
            addplot=apds,
            returnfig=True,
            volume=True,
            title=title or "Double Top Detections",
            figsize=(12, 7)
        )
        price_ax = axlist[0]
    else:
        # Fallback: Close line plot
        fig, (price_ax, vol_ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
        price_ax.plot(view.index, view["Close"], linewidth=1.0)
        price_ax.set_ylabel("Price")
        vol_ax.bar(view.index, view["Volume"], width=0.8)
        vol_ax.set_ylabel("Volume")
        price_ax.set_title(title or "Double Top Detections")

    # Overlay each pattern (only if its indices are in the selected window)
    for pat in patterns:
        # Limit to items within the view
        if pat.peak1_date < start or pat.confirm_date > end:
            continue

        idx = view.index
        # Points
        plot_point(price_ax, pat.peak1_date, pat.peak1, color='tab:red', label="Peak")
        plot_point(price_ax, pat.peak2_date, pat.peak2, color='tab:red')
        plot_point(price_ax, pat.trough_date, pat.trough, color='tab:blue', label="Trough")
        plot_point(price_ax, pat.confirm_date, view.loc[pat.confirm_date, "Close"], color='black', label="Confirm")

        # Neckline (horizontal dashed)
        x_left = pat.peak1_date
        x_right = pat.confirm_date
        price_ax.hlines(pat.neckline, xmin=x_left, xmax=x_right, colors='tab:blue', linestyles='dashed', linewidth=1.5)

        # Optional: connect peaks
        price_ax.plot([pat.peak1_date, pat.peak2_date], [pat.peak1, pat.peak2], color='tab:red', linewidth=1.0, alpha=0.6)

    # Legend handling (avoid duplicates)
    handles, labels = price_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        price_ax.legend(unique.values(), unique.keys(), loc='best')

    plt.tight_layout()
    plt.show()


# =============================
# 3) Helpers (modular steps)
# =============================

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.set_index("Date")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.sort_index()
    # ensure numeric
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(subset=["Open", "High", "Low", "Close"])  # Volume can be NaN
    return data


def apply_ema(series: pd.Series, span: Optional[int]) -> pd.Series:
    if span is None or span <= 1:
        return series
    return series.ewm(span=span, adjust=False).mean()


def find_local_extrema(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict local extrema: a point is a peak if it is strictly greater than all values in its +/- window neighborhood.
    Returns boolean arrays (is_peak, is_trough) aligned to the series index.
    """
    s = series.to_numpy()
    n = len(s)
    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)

    if n == 0 or window < 1:
        return is_peak, is_trough

    for i in range(window, n - window):
        seg = s[i - window:i + window + 1]
        center = s[i]
        left = seg[:window]
        right = seg[window + 1:]

        # Peak
        if center == seg.max():
            if (center > left).all() and (center > right).all():
                is_peak[i] = True

        # Trough
        if center == seg.min():
            if (center < left).all() and (center < right).all():
                is_trough[i] = True

    return is_peak, is_trough


def pair_peaks_with_troughs(
    data: pd.DataFrame,
    peak_idx: np.ndarray,
    low_vals: np.ndarray,
    params: PatternParams
) -> List[Tuple[int, int, int]]:
    """
    Iterate consecutive peaks and locate the lowest Low (trough) between them.
    Enforce separation constraints. Return list of (p1, p2, trough_idx).
    """
    pairs: List[Tuple[int, int, int]] = []
    n = len(data)

    for i in range(len(peak_idx) - 1):
        p1 = int(peak_idx[i])
        p2 = int(peak_idx[i + 1])

        sep = p2 - p1
        if sep < params.min_sep or sep > params.max_sep:
            continue

        # Trough is the index of the minimum low between p1 and p2
        if p1 + 1 >= p2:
            continue
        mid_slice = slice(p1 + 1, p2)
        t_local = np.argmin(low_vals[mid_slice]) + (p1 + 1)
        trough_idx = int(t_local)

        pairs.append((p1, p2, trough_idx))

    return pairs


def passes_peak_similarity(data: pd.DataFrame, p1: int, p2: int, tol_pct: float) -> Tuple[bool, float]:
    """
    Compare peak heights using High (can be more sensitive) and fallback to Close if desired.
    Returns (ok, diff_pct).
    """
    h1 = float(data["High"].iloc[p1])
    h2 = float(data["High"].iloc[p2])
    base = max(h1, h2) if max(h1, h2) != 0 else 1.0
    diff_pct = abs(h2 - h1) / base * 100.0
    return (diff_pct <= tol_pct), diff_pct


def passes_trough_depth(
    data: pd.DataFrame, p1: int, p2: int, t_idx: int, min_drop_pct: float
) -> Tuple[bool, float, float]:
    """
    Each peak must have dropped to the trough by at least min_drop_pct.
    Returns (ok, drop1_pct, drop2_pct)
    """
    peak1 = float(data["High"].iloc[p1])
    peak2 = float(data["High"].iloc[p2])
    trough = float(data["Low"].iloc[t_idx])
    if peak1 == 0 or peak2 == 0:
        return False, 0.0, 0.0
    drop1 = (peak1 - trough) / peak1 * 100.0
    drop2 = (peak2 - trough) / peak2 * 100.0
    ok = (drop1 >= min_drop_pct) and (drop2 >= min_drop_pct)
    return ok, drop1, drop2


def passes_uptrend(data: pd.DataFrame, p1: int, params: PatternParams) -> bool:
    """
    Simple uptrend filter: percentage return over lookback >= threshold (or add MA slope logic).
    """
    i0 = max(0, p1 - params.uptrend_lookback)
    c0 = float(data["Close"].iloc[i0])
    c1 = float(data["Close"].iloc[p1])
    if c0 == 0:
        return False
    ret_pct = (c1 - c0) / c0 * 100.0
    return ret_pct >= params.uptrend_min_return_pct


def confirm_break(data: pd.DataFrame, p2: int, neckline: float, params: PatternParams) -> Optional[int]:
    """
    Look forward up to confirm_lookahead bars for a decisive close below neckline*(1 - margin).
    Returns index of first confirming bar or None.
    """
    n = len(data)
    start = p2 + 1
    end = min(p2 + params.confirm_lookahead, n - 1)
    if start >= n:
        return None
    threshold = neckline * (1.0 - params.break_margin_pct / 100.0)
    closes = data["Close"].values
    for j in range(start, end + 1):
        if closes[j] <= threshold:
            return j
    return None


def passes_volume(
    data: pd.DataFrame,
    p1: int,
    p2: int,
    confirm_idx: int,
    vol_ma: Optional[pd.Series],
    params: PatternParams
) -> bool:
    """
    Volume heuristics:
      - optional: Volume at first peak > Volume at second peak
      - confirmation bar volume >= vol_confirm_mult * SMA(volume)
    """
    if "Volume" not in data.columns:
        return True  # nothing to check
    v = data["Volume"]
    # Divergence: first peak higher volume than second
    if params.require_vol_divergence and not (v.iloc[p1] > v.iloc[p2]):
        return False
    # Confirm spike vs moving average
    if vol_ma is not None:
        if np.isnan(vol_ma.iloc[confirm_idx]):
            return True  # can't check
        return v.iloc[confirm_idx] >= (params.vol_confirm_mult * vol_ma.iloc[confirm_idx])
    # If no vol_ma provided, accept
    return True


def quality_score(peak_diff_pct: float, drop1_pct: float, drop2_pct: float, confirm_delay: int, params: PatternParams) -> float:
    """
    Heuristic quality score in [0,1] to rank candidates.
    """
    # Peak similarity: the smaller the diff, the better
    tol = max(params.peak_tolerance_pct, 1e-6)
    sim = 1.0 - min(peak_diff_pct / tol, 1.0)  # 0..1

    # Trough depth: deeper is better up to 2x threshold
    md = max(params.min_trough_drop_pct, 1e-6)
    depth = min(min(drop1_pct, drop2_pct) / (2.0 * md), 1.0)

    # Early confirmation bonus
    early = 0.0
    if params.confirm_lookahead > 0:
        early = max(0.0, (params.confirm_lookahead - confirm_delay) / params.confirm_lookahead)

    return 0.5 * sim + 0.3 * depth + 0.2 * early


def deduplicate_overlaps(patterns: List[DoubleTop], prefer_earliest: bool = True) -> List[DoubleTop]:
    """
    If multiple detections share the same peak2 or have heavy overlap, keep the best by score (or earliest confirm).
    """
    if not patterns:
        return patterns

    # Group by peak2 index (common to have overlaps there)
    by_p2 = {}
    for p in patterns:
        by_p2.setdefault(p.peak2_idx, []).append(p)

    kept: List[DoubleTop] = []
    for _, group in by_p2.items():
        if prefer_earliest:
            group.sort(key=lambda g: (g.confirm_date, -g.score))
        else:
            group.sort(key=lambda g: (-g.score, g.confirm_date))
        kept.append(group[0])

    # Optionally we could also resolve overlaps by time-span intersection, but this is usually enough.
    kept.sort(key=lambda r: (r.confirm_date, -r.score))
    return kept


def plot_point(ax, x, y, color='tab:red', label=None):
    ax.scatter([x], [y], color=color, s=40, zorder=5, label=label if label else None)


# =============================
# 4) Utility: export to DataFrame
# =============================

def patterns_to_dataframe(patterns: List[DoubleTop]) -> pd.DataFrame:
    """
    Convert detections to a tidy DataFrame for logging, analysis, or ML labeling.
    """
    if not patterns:
        return pd.DataFrame(columns=[
            "symbol", "peak1_date", "peak2_date", "trough_date", "confirm_date",
            "peak1", "peak2", "trough", "neckline", "peak_diff_pct", "drop1_pct", "drop2_pct",
            "separation_bars", "confirm_delay_bars", "score",
            "peak1_idx", "peak2_idx", "trough_idx", "confirm_idx"
        ])
    rows = []
    for p in patterns:
        rows.append({
            "symbol": p.symbol,
            "peak1_date": p.peak1_date,
            "peak2_date": p.peak2_date,
            "trough_date": p.trough_date,
            "confirm_date": p.confirm_date,
            "peak1": p.peak1,
            "peak2": p.peak2,
            "trough": p.trough,
            "neckline": p.neckline,
            "peak_diff_pct": p.peak_diff_pct,
            "drop1_pct": p.drop1_pct,
            "drop2_pct": p.drop2_pct,
            "separation_bars": p.separation_bars,
            "confirm_delay_bars": p.confirm_delay_bars,
            "score": p.score,
            "peak1_idx": p.peak1_idx,
            "peak2_idx": p.peak2_idx,
            "trough_idx": p.trough_idx,
            "confirm_idx": p.confirm_idx,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["confirm_date", "score"], ascending=[True, False]).reset_index(drop=True)


# =============================
# 5) Example usage (only runs if executed as a script)
# =============================

if __name__ == "__main__":
    # Minimal demo with a CSV (expects columns: Date, Open, High, Low, Close, Volume)
    import argparse

    parser = argparse.ArgumentParser(description="Detect Double Tops in daily OHLCV data.")
    parser.add_argument("--csv", type=str, required=False, help="Path to CSV with OHLCV data.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol name.")
    parser.add_argument("--plot", action="store_true", help="Show a visualization.")
    args = parser.parse_args()

    # Example: if no CSV provided, create a tiny synthetic example (for structure sanity, not real detection)
    if not args.csv:
        print("No CSV provided; creating a tiny synthetic dataset just to show the API.")
        dates = pd.date_range("2022-01-01", periods=80, freq="B")
        rng = np.random.default_rng(42)
        base = np.linspace(100, 120, len(dates)) + rng.normal(0, 0.5, len(dates))
        # Add synthetic double-top-ish bump
        base[25] += 8.0
        base[35] += 8.1
        highs = base + rng.normal(0.6, 0.3, len(dates))
        lows = base - rng.normal(0.6, 0.3, len(dates))
        opens = base + rng.normal(0, 0.2, len(dates))
        closes = base + rng.normal(0, 0.2, len(dates))
        vols = rng.integers(1_000_000, 2_000_000, len(dates))
        demo = pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols
        })
        df_in = demo
        symbol = args.symbol or "SYNTH"
    else:
        df_in = pd.read_csv(args.csv, parse_dates=["Date"])
        symbol = args.symbol

    params = PatternParams(
        lookback_extrema=3,
        ema_span=3,
        min_sep=5,
        max_sep=40,
        peak_tolerance_pct=2.0,
        min_trough_drop_pct=3.0,
        confirm_lookahead=15,
        break_margin_pct=0.5,
        use_volume_filter=True,
        vol_ma_window=20,
        vol_confirm_mult=1.2,
        require_uptrend=True,
        uptrend_lookback=20,
        uptrend_min_return_pct=5.0,
        require_vol_divergence=False,   # set True to enforce vol at peak1 > peak2
    )

    detections = detect_double_tops(df_in, params, symbol=symbol)
    out_df = patterns_to_dataframe(detections)
    print(out_df.head(20))
    print(f"Total double tops found: {len(out_df)}")

    if args.plot and len(detections) > 0:
        # Plot entire range with overlays
        plot_double_tops(df_in, detections, title=f"Double Tops — {symbol or ''}")
