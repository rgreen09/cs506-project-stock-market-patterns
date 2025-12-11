import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector


DEFAULT_CONFIG = {
    "min_cup_duration": 7,
    "max_cup_duration": 65,
    "min_cup_depth": 12,  # percent
    "max_cup_depth": 33,  # percent
    "peak_similarity_threshold": 5,  # percent
    "min_handle_duration": 5,
    "max_handle_duration": 20,
    "max_handle_depth": 15,  # percent
    "volume_breakout_threshold": 1.2,  # unused without volume
    "extrema_order": 5,
}


def _find_extrema(values: np.ndarray, order: int, mode: str):
    idxs = []
    n = len(values)
    for i in range(order, n - order):
        window = values[i - order : i + order + 1]
        if mode == "max" and values[i] == window.max() and values[i] > window[0] and values[i] > window[-1]:
            idxs.append(i)
        if mode == "min" and values[i] == window.min() and values[i] < window[0] and values[i] < window[-1]:
            idxs.append(i)
    return np.array(idxs, dtype=int)


def _depth_pct(peak_price: float, trough_price: float) -> float:
    if peak_price == 0:
        return 0.0
    return abs((peak_price - trough_price) / peak_price * 100)


class CupAndHandleDetector(BaseDetector):
    pattern_name = "cup_and_handle"

    def __init__(self, config: dict | None = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def detect(self, price_series: pd.Series, **kwargs) -> bool:
        cfg = self.config
        closes = price_series.values
        n = len(closes)
        if n < cfg["min_cup_duration"] + cfg["min_handle_duration"] + 10:
            return False

        peaks = _find_extrema(closes, cfg["extrema_order"], mode="max")
        troughs = _find_extrema(closes, cfg["extrema_order"], mode="min")

        for peak1_idx in peaks:
            valid_troughs = troughs[troughs > peak1_idx]
            for trough_idx in valid_troughs:
                valid_peaks2 = peaks[peaks > trough_idx]
                for peak2_idx in valid_peaks2:
                    if not self._is_valid_cup(closes, peak1_idx, trough_idx, peak2_idx, cfg):
                        continue
                    if not self._has_valid_handle(closes, peak2_idx, cfg):
                        continue
                    if self._has_breakout(closes, peak1_idx, peak2_idx, cfg):
                        return True
        return False

    def _is_valid_cup(self, closes, peak1_idx, trough_idx, peak2_idx, cfg) -> bool:
        peak1_price = closes[peak1_idx]
        trough_price = closes[trough_idx]
        peak2_price = closes[peak2_idx]

        cup_duration = peak2_idx - peak1_idx
        if not (cfg["min_cup_duration"] <= cup_duration <= cfg["max_cup_duration"]):
            return False

        depth_pct = _depth_pct(peak1_price, trough_price)
        if not (cfg["min_cup_depth"] <= depth_pct <= cfg["max_cup_depth"]):
            return False

        peak_diff_pct = abs((peak1_price - peak2_price) / peak1_price * 100)
        if peak_diff_pct > cfg["peak_similarity_threshold"]:
            return False

        trough_position = (trough_idx - peak1_idx) / cup_duration if cup_duration else 0.5
        if 0.4 <= trough_position <= 0.6:
            return False

        return True

    def _has_valid_handle(self, closes, peak2_idx, cfg) -> bool:
        search_end = min(peak2_idx + cfg["max_handle_duration"] + 10, len(closes))
        if search_end - peak2_idx < cfg["min_handle_duration"]:
            return False
        segment = closes[peak2_idx:search_end]
        handle_low_rel_idx = int(np.argmin(segment))
        handle_low_idx = peak2_idx + handle_low_rel_idx
        handle_low_price = closes[handle_low_idx]
        peak2_price = closes[peak2_idx]

        handle_depth = _depth_pct(peak2_price, handle_low_price)
        if handle_depth > cfg["max_handle_depth"]:
            return False

        handle_duration = handle_low_idx - peak2_idx
        if not (cfg["min_handle_duration"] <= handle_duration <= cfg["max_handle_duration"]):
            return False
        return True

    def _has_breakout(self, closes, peak1_idx, peak2_idx, cfg) -> bool:
        resistance = max(closes[peak1_idx], closes[peak2_idx])
        search_start = peak2_idx + cfg["min_handle_duration"]
        search_end = min(peak2_idx + cfg["max_handle_duration"] + 15, len(closes))
        for idx in range(search_start, search_end):
            if closes[idx] > resistance * 1.01:
                return True
        return False

