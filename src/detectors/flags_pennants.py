import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.features.common import calculate_linear_trend, smooth_prices


DEFAULT_CONFIG = {
    "pole_min_move_pct": 0.02,
    "pole_max_bars": 100,
    "consolidation_min_bars": 10,
    "consolidation_max_bars": 150,
    "consolidation_max_range_pct": 0.01,
    "flag_slope_tolerance": 0.3,
    "pennant_converge_ratio": 0.6,
    "breakout_threshold_pct": 0.005,
    "require_breakout": True,
    "min_bars_after_consolidation": 5,
    "volume_increase_ratio": 1.2,
    "smoothing_window": 5,
}


class FlagsPennantsDetector(BaseDetector):
    pattern_name = "flags_pennants"

    def __init__(self, config: dict | None = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def detect(self, price_series: pd.Series, window_bars: int | None = None) -> bool:
        cfg = self.config
        prices = price_series.values
        if window_bars is None:
            window_bars = len(prices)

        smooth = smooth_prices(prices, window=cfg["smoothing_window"])
        n = len(smooth)
        if n < 20:
            return False

        pole = self._find_pole(smooth, cfg["pole_max_bars"])
        if pole is None:
            return False
        pole_start_idx, pole_end_idx, pole_move_pct, pole_direction = pole
        if abs(pole_move_pct) < cfg["pole_min_move_pct"]:
            return False

        consolidation_start = pole_end_idx
        remaining_bars = n - consolidation_start
        if remaining_bars < cfg["consolidation_min_bars"]:
            return False

        if cfg["require_breakout"]:
            max_consolidation_end = min(
                n - cfg["min_bars_after_consolidation"],
                consolidation_start + cfg["consolidation_max_bars"],
            )
        else:
            max_consolidation_end = min(n, consolidation_start + cfg["consolidation_max_bars"])

        if max_consolidation_end <= consolidation_start + cfg["consolidation_min_bars"]:
            return False

        best_end = None
        best_score = float("-inf")
        best_analysis = None

        for consolidation_end in range(
            consolidation_start + cfg["consolidation_min_bars"],
            max_consolidation_end + 1,
        ):
            segment = smooth[consolidation_start:consolidation_end]
            pole_end_price = smooth[pole_end_idx]
            range_pct, slope, pattern_type = self._analyze_consolidation(
                segment, pole_end_price, cfg
            )
            if range_pct > cfg["consolidation_max_range_pct"]:
                continue

            if pattern_type == "flag":
                pole_slope = (
                    (smooth[pole_end_idx] - smooth[pole_start_idx]) / (pole_end_idx - pole_start_idx)
                    if pole_end_idx > pole_start_idx
                    else 0
                )
                if abs(pole_slope) > 0:
                    slope_ratio = abs(slope / pole_slope)
                    if slope_ratio > cfg["flag_slope_tolerance"]:
                        continue

            score = -range_pct
            if score > best_score:
                best_score = score
                best_end = consolidation_end
                best_analysis = (range_pct, slope, pattern_type)

        if best_end is None or best_analysis is None:
            return False

        range_pct, consolidation_slope, pattern_type = best_analysis
        consolidation_end_idx = best_end

        breakout_confirmed = False
        if consolidation_end_idx < n:
            consolidation_high = np.max(smooth[consolidation_start:consolidation_end_idx])
            consolidation_low = np.min(smooth[consolidation_start:consolidation_end_idx])
            if pole_direction == "up":
                breakout_threshold = consolidation_high * (1 + cfg["breakout_threshold_pct"])
                mask = smooth[consolidation_end_idx:] > breakout_threshold
            else:
                breakout_threshold = consolidation_low * (1 - cfg["breakout_threshold_pct"])
                mask = smooth[consolidation_end_idx:] < breakout_threshold

            if len(mask) > 0 and np.any(mask):
                breakout_confirmed = True

            if cfg["require_breakout"]:
                if len(mask) >= cfg["min_bars_after_consolidation"] and not breakout_confirmed:
                    return False
        else:
            if cfg["require_breakout"]:
                return False

        return True

    def _find_pole(self, prices: np.ndarray, max_bars: int):
        n = len(prices)
        if n < 3:
            return None
        best_pole = None
        best_move_pct = 0.0
        for start_idx in range(n - 2):
            for end_idx in range(start_idx + 2, min(n, start_idx + max_bars)):
                start_price = prices[start_idx]
                end_price = prices[end_idx]
                if start_price == 0:
                    continue
                move_pct = (end_price - start_price) / start_price
                abs_move = abs(move_pct)
                if abs_move > best_move_pct:
                    best_move_pct = abs_move
                    direction = "up" if move_pct > 0 else "down"
                    best_pole = (start_idx, end_idx, move_pct, direction)
        return best_pole

    def _analyze_consolidation(self, prices: np.ndarray, pole_end_price: float, cfg: dict):
        n = len(prices)
        if n < 2 or pole_end_price == 0:
            return 0.0, 0.0, "unknown"

        price_min = np.min(prices)
        price_max = np.max(prices)
        range_pct = (price_max - price_min) / pole_end_price

        slope = calculate_linear_trend(prices)

        first_third = prices[: n // 3]
        last_third = prices[-n // 3 :]
        if len(first_third) > 0 and len(last_third) > 0:
            first_range = np.max(first_third) - np.min(first_third)
            last_range = np.max(last_third) - np.min(last_third)
            if first_range > 0:
                convergence_ratio = last_range / first_range
                if convergence_ratio < cfg["pennant_converge_ratio"]:
                    pattern_type = "pennant"
                else:
                    pattern_type = "flag"
            else:
                pattern_type = "flag"
        else:
            pattern_type = "flag"

        return float(range_pct), float(slope), pattern_type

