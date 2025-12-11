from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureExtractor
from src.features.common import calculate_linear_trend


def find_strongest_move(prices: np.ndarray, max_bars: int = 100) -> Dict[str, float]:
    n = len(prices)
    if n < 3:
        return {"max_move_pct": 0.0, "max_move_bars": 0.0, "max_move_direction": 0.0}

    max_move_pct = 0.0
    max_move_bars = 0
    max_move_direction = 0

    for start_idx in range(n - 2):
        for end_idx in range(start_idx + 2, min(n, start_idx + max_bars)):
            if prices[start_idx] == 0:
                continue
            move_pct = (prices[end_idx] - prices[start_idx]) / prices[start_idx]
            abs_move = abs(move_pct)
            if abs_move > abs(max_move_pct):
                max_move_pct = move_pct
                max_move_bars = end_idx - start_idx
                max_move_direction = 1 if move_pct > 0 else -1

    return {
        "max_move_pct": float(max_move_pct),
        "max_move_bars": float(max_move_bars),
        "max_move_direction": float(max_move_direction),
    }


def compute_consolidation_features(prices: np.ndarray, pole_end: int) -> Dict[str, float]:
    n = len(prices)
    if pole_end >= n - 10 or pole_end < 10:
        return {
            "consolidation_range_pct": 0.0,
            "consolidation_volatility": 0.0,
            "consolidation_slope": 0.0,
            "price_compression_ratio": 0.0,
        }

    after_pole = prices[pole_end:]
    if len(after_pole) < 10:
        return {
            "consolidation_range_pct": 0.0,
            "consolidation_volatility": 0.0,
            "consolidation_slope": 0.0,
            "price_compression_ratio": 0.0,
        }

    segment = after_pole[: min(150, len(after_pole))]
    cons_min = np.min(segment)
    cons_max = np.max(segment)
    pole_price = prices[pole_end]
    range_pct = (cons_max - cons_min) / pole_price if pole_price > 0 else 0.0

    cons_volatility = np.std(segment)
    cons_slope = calculate_linear_trend(segment)

    n_cons = len(segment)
    if n_cons >= 30:
        first_third = segment[: n_cons // 3]
        last_third = segment[-n_cons // 3 :]
        first_range = np.max(first_third) - np.min(first_third)
        last_range = np.max(last_third) - np.min(last_third)
        compression_ratio = last_range / first_range if first_range > 0 else 1.0
    else:
        compression_ratio = 1.0

    return {
        "consolidation_range_pct": float(range_pct),
        "consolidation_volatility": float(cons_volatility),
        "consolidation_slope": float(cons_slope),
        "price_compression_ratio": float(compression_ratio),
    }


class FlagsPennantsFeatures(BaseFeatureExtractor):
    """
    Feature extractor for Flags & Pennants windows.
    Mirrors compute_window_features from the legacy script.
    """

    def __init__(self) -> None:
        self._feature_names = [
            "close_last",
            "close_mean",
            "close_std",
            "price_range_abs",
            "price_range_pct",
            "cumulative_return_window",
            "ret_1",
            "ret_5",
            "ret_20",
            "momentum_last_30",
            "slope_entire_window",
            "slope_last_30",
            "max_move_pct",
            "max_move_bars",
            "max_move_direction",
            "pole_strength_ratio",
            "return_first_third",
            "return_middle_third",
            "return_last_third",
            "estimated_pole_end_idx",
            "estimated_pole_end_rel_pos",
            "pole_velocity",
            "consolidation_range_pct",
            "consolidation_volatility",
            "consolidation_slope",
            "price_compression_ratio",
            "max_deviation_after_pole_pct",
            "avg_deviation_after_pole_pct",
            "slope_after_pole",
            "volatility_ratio_after_pole",
            "sma_20",
            "close_over_sma20",
            "sma_50",
            "rsi_14",
            "macd_line",
        ]

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def compute_features(self, close_window: pd.Series) -> Dict[str, float]:
        c = close_window.values
        n = len(c)

        features: Dict[str, float] = {}
        features["close_last"] = float(c[-1])
        features["close_mean"] = float(np.mean(c))
        features["close_std"] = float(np.std(c))
        features["price_range_abs"] = float(np.max(c) - np.min(c))
        features["price_range_pct"] = float(
            (features["price_range_abs"] / c[-1]) if c[-1] != 0 else 0.0
        )

        features["cumulative_return_window"] = float(
            (c[-1] / c[0] - 1) if c[0] != 0 else 0.0
        )
        features["ret_1"] = float((c[-1] / c[-2] - 1) if n > 1 and c[-2] != 0 else 0.0)
        features["ret_5"] = float((c[-1] / c[-6] - 1) if n > 5 and c[-6] != 0 else 0.0)
        features["ret_20"] = float(
            (c[-1] / c[-21] - 1) if n > 20 and c[-21] != 0 else 0.0
        )
        features["momentum_last_30"] = float(
            (c[-1] - c[-31]) / c[-31] if n > 30 and c[-31] != 0 else 0.0
        )

        features["slope_entire_window"] = float(calculate_linear_trend(c))
        features["slope_last_30"] = float(
            calculate_linear_trend(c[-30:]) if n >= 30 else calculate_linear_trend(c)
        )

        move_stats = find_strongest_move(c, max_bars=100)
        features["max_move_pct"] = move_stats["max_move_pct"]
        features["max_move_bars"] = move_stats["max_move_bars"]
        features["max_move_direction"] = move_stats["max_move_direction"]

        abs_max_move = abs(move_stats["max_move_pct"])
        features["pole_strength_ratio"] = float(
            abs_max_move / features["price_range_pct"] if features["price_range_pct"] > 0 else 0.0
        )

        if n >= 100:
            first_third = c[: n // 3]
            middle_third = c[n // 3 : 2 * n // 3]
            last_third = c[2 * n // 3 :]
            ret_first = (first_third[-1] / first_third[0] - 1) if first_third[0] != 0 else 0.0
            ret_middle = (middle_third[-1] / middle_third[0] - 1) if middle_third[0] != 0 else 0.0
            ret_last = (last_third[-1] / last_third[0] - 1) if last_third[0] != 0 else 0.0
            features["return_first_third"] = float(ret_first)
            features["return_middle_third"] = float(ret_middle)
            features["return_last_third"] = float(ret_last)
        else:
            features["return_first_third"] = 0.0
            features["return_middle_third"] = 0.0
            features["return_last_third"] = 0.0

        if move_stats["max_move_bars"] > 0:
            estimated_pole_end = min(int(move_stats["max_move_bars"]), n // 2)
        else:
            estimated_pole_end = n // 3

        features["estimated_pole_end_idx"] = float(estimated_pole_end)
        features["estimated_pole_end_rel_pos"] = float(estimated_pole_end / n if n > 0 else 0.0)
        features["pole_velocity"] = float(
            move_stats["max_move_pct"] / move_stats["max_move_bars"]
            if move_stats["max_move_bars"] > 0
            else 0.0
        )

        cons_features = compute_consolidation_features(c, estimated_pole_end)
        features.update(cons_features)

        if estimated_pole_end < n - 10:
            after_pole_segment = c[estimated_pole_end:]
            pole_end_price = c[estimated_pole_end]
            max_dev = np.max(np.abs(after_pole_segment - pole_end_price)) / pole_end_price if pole_end_price > 0 else 0.0
            features["max_deviation_after_pole_pct"] = float(max_dev)
            avg_dev = np.mean(np.abs(after_pole_segment - pole_end_price)) / pole_end_price if pole_end_price > 0 else 0.0
            features["avg_deviation_after_pole_pct"] = float(avg_dev)
            after_pole_slope = calculate_linear_trend(after_pole_segment)
            features["slope_after_pole"] = float(after_pole_slope)
            before_pole_vol = np.std(c[:estimated_pole_end]) if estimated_pole_end > 1 else 0.0
            after_pole_vol = np.std(after_pole_segment)
            vol_ratio = after_pole_vol / before_pole_vol if before_pole_vol > 0 else 1.0
            features["volatility_ratio_after_pole"] = float(vol_ratio)
        else:
            features["max_deviation_after_pole_pct"] = 0.0
            features["avg_deviation_after_pole_pct"] = 0.0
            features["slope_after_pole"] = 0.0
            features["volatility_ratio_after_pole"] = 1.0

        sma_20 = float(np.mean(c[-20:])) if n >= 20 else float(np.mean(c))
        features["sma_20"] = sma_20
        features["close_over_sma20"] = float(c[-1] / sma_20 if sma_20 != 0 else 1.0)
        features["sma_50"] = float(np.mean(c[-50:])) if n >= 50 else float(np.mean(c))

        if n >= 15:
            delta = np.diff(c[-15:])
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            features["rsi_14"] = float(rsi)
        else:
            features["rsi_14"] = 50.0

        if n >= 26:
            ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().iloc[-1]
            ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().iloc[-1]
            features["macd_line"] = float(ema_fast - ema_slow)
        else:
            features["macd_line"] = 0.0

        return features

