# head_and_shoulders_fixed.py
# Improved Head & Shoulders pattern detector with fixed ROC evaluation
# Author: Jigar — 2025-10-27

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import roc_curve, auc

try:
    import mplfinance as mpf
    _HAS_MPLFIN = True
except Exception:
    _HAS_MPLFIN = False


# ======================================
# CONFIG
# ======================================

@dataclass
class PatternParams:
    lookback_extrema: int = 3
    ema_span: Optional[int] = 5
    min_sep: int = 3
    max_sep: int = 120
    shoulder_tolerance_pct: float = 10.0
    head_taller_pct: float = 2.0
    confirm_lookahead: int = 60
    break_margin_pct: float = 0.0
    prefer_earlier_confirm: bool = True


@dataclass
class HeadAndShoulders:
    symbol: str
    direction: str
    left_idx: int
    head_idx: int
    right_idx: int
    confirm_idx: int
    neckline: float
    score: float
    left_date: pd.Timestamp
    head_date: pd.Timestamp
    right_date: pd.Timestamp
    confirm_date: pd.Timestamp


# ======================================
# DETECTION LOGIC
# ======================================

def detect_patterns(df: pd.DataFrame, params: PatternParams, symbol: str) -> List[HeadAndShoulders]:
    data = ensure_ohlcv(df)
    px = data["High"].astype(float)
    px_smooth = apply_ema(px, params.ema_span)
    is_peak, is_trough = find_local_extrema(px_smooth, window=params.lookback_extrema)
    peak_idx = np.where(is_peak)[0]
    trough_idx = np.where(is_trough)[0]
    patterns = []
    # detect both
    patterns += detect_triplets(data, peak_idx, params, symbol, "normal")
    patterns += detect_triplets(data, trough_idx, params, symbol, "inverse")
    return patterns


def detect_triplets(data, idx_array, params, symbol, direction="normal"):
    pats = []
    val_series = data["High"] if direction == "normal" else -data["Low"]
    for i in range(len(idx_array) - 2):
        l, h, r = idx_array[i], idx_array[i + 1], idx_array[i + 2]
        sep1, sep2 = h - l, r - h
        if not (params.min_sep <= sep1 <= params.max_sep and params.min_sep <= sep2 <= params.max_sep):
            continue
        left, head, right = val_series.iloc[[l, h, r]]
        shoulder_diff = abs(left - right) / max(left, right) * 100
        if shoulder_diff > params.shoulder_tolerance_pct:
            continue
        avg_sh = (left + right) / 2
        head_diff = (head - avg_sh) / avg_sh * 100
        if direction == "normal" and head_diff < params.head_taller_pct:
            continue
        if direction == "inverse" and head_diff > -params.head_taller_pct:
            continue

        neckline = calc_neckline(data, h, direction)
        conf = confirm_break(data, r, neckline, params, direction)
        if conf is None:
            continue

        score = compute_score(shoulder_diff, abs(head_diff), conf - r, params)
        pats.append(HeadAndShoulders(
            symbol, direction, l, h, r, conf, neckline, score,
            data.index[l], data.index[h], data.index[r], data.index[conf]
        ))
    return pats


def calc_neckline(data, h, direction):
    if direction == "normal":
        return data["Low"].iloc[h-2:h+5].min()
    else:
        return data["High"].iloc[h-2:h+5].max()


def confirm_break(data, r_idx, neckline, params, direction):
    closes = data["Close"].values
    end = min(len(closes) - 1, r_idx + params.confirm_lookahead)
    for j in range(r_idx + 1, end + 1):
        if direction == "normal" and closes[j] <= neckline:
            return j
        if direction == "inverse" and closes[j] >= neckline:
            return j
    return None


def compute_score(sym, head, delay, params):
    # balanced weighting
    s1 = 1 - min(sym / (params.shoulder_tolerance_pct or 1e-6), 1)
    s2 = min(head / (2 * params.head_taller_pct), 1)
    s3 = max(0, (params.confirm_lookahead - delay) / params.confirm_lookahead)
    return 0.4*s1 + 0.4*s2 + 0.2*s3


# ======================================
# HELPERS
# ======================================

def ensure_ohlcv(df: pd.DataFrame):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def apply_ema(s: pd.Series, span: Optional[int]):
    return s.ewm(span=span, adjust=False).mean() if span and span > 1 else s


def find_local_extrema(s: pd.Series, window: int):
    s = s.to_numpy()
    n = len(s)
    peak, trough = np.zeros(n, bool), np.zeros(n, bool)
    eps = np.finfo(float).eps * 10
    for i in range(window, n - window):
        seg = s[i - window:i + window + 1]
        if s[i] >= seg.max() - eps:
            peak[i] = True
        if s[i] <= seg.min() + eps:
            trough[i] = True
    return peak, trough


# ======================================
# ROC EVALUATION (fixed)
# ======================================

def evaluate_roc(df: pd.DataFrame, patterns: List[HeadAndShoulders],
                 lookahead_days=20, drop_threshold=0.03):
    if not patterns:
        print("⚠️ No patterns detected.")
        return

    y_score, y_true = [], []

    for p in patterns:
        conf_idx = df.index.get_loc(p.confirm_date)
        future_end = min(conf_idx + lookahead_days, len(df) - 1)
        future_close = df["Close"].iloc[conf_idx:future_end].values
        price_now = df["Close"].iloc[conf_idx]
        if len(future_close) < 2:
            continue
        change = (future_close[-1] - price_now) / price_now

        # FIX: label based on direction
        if p.direction == "normal":
            label = 1 if change <= -drop_threshold else 0
        else:
            label = 1 if change >= drop_threshold else 0

        y_score.append(p.score)
        y_true.append(label)

    if len(set(y_true)) < 2:
        print("⚠️ Not enough variety in outcomes for ROC curve.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, color="tab:blue", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Pattern Score vs Real Outcome")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"✅ ROC AUC Score: {roc_auc:.3f}")


# ======================================
# MAIN
# ======================================

if __name__ == "__main__":
    symbol = "AAPL"
    df = yf.download(symbol, start="2010-01-01", end="2025-01-01", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.rename(columns=lambda x: x.strip().title())

    params = PatternParams()
    patterns = detect_patterns(df, params, symbol)
    df_out = pd.DataFrame([vars(p) for p in patterns])
    print(f"✅ Total patterns detected: {len(df_out)}")
    print(df_out.groupby("direction")["direction"].count())
    print(df_out.head(10))
    evaluate_roc(df, patterns)
