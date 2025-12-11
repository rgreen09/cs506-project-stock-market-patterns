import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.features.common import find_local_peaks, find_local_troughs, smooth_prices


DEFAULT_CONFIG = {
    "shoulder_tolerance": 0.04,
    "min_head_height_pct": 0.02,
    "min_drop_pct": 0.005,
    "smoothing_window": 5,
    "require_confirmation": False,
    "break_buffer": 0.001,
    "min_bars_after_head": 5,
}


class HeadAndShouldersDetector(BaseDetector):
    pattern_name = "head_and_shoulders"

    def __init__(self, config: dict | None = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def detect(self, price_series: pd.Series, **kwargs) -> bool:
        cfg = self.config
        prices = price_series.values
        smooth = smooth_prices(prices, window=cfg["smoothing_window"])
        peaks = find_local_peaks(smooth)
        if len(peaks) < 3:
            return False

        for i in range(len(peaks) - 2):
            ls, head, rs = peaks[i], peaks[i + 1], peaks[i + 2]
            left, mid, right = smooth[ls], smooth[head], smooth[rs]

            if not (mid > left and mid > right):
                continue

            shoulder_diff = abs(left - right) / ((left + right) / 2)
            if shoulder_diff > cfg["shoulder_tolerance"]:
                continue

            shoulder_avg = (left + right) / 2
            head_rise = (mid - shoulder_avg) / shoulder_avg
            if head_rise < cfg["min_head_height_pct"]:
                continue

            troughs = find_local_troughs(smooth[ls : rs + 1])
            if not troughs:
                continue
            neckline_idx = ls + troughs[np.argmin(smooth[ls : rs + 1][troughs])]
            neckline_price = smooth[neckline_idx]

            drop_pct = (mid - neckline_price) / mid
            if drop_pct < cfg["min_drop_pct"]:
                continue

            if cfg["require_confirmation"]:
                after_rs = smooth[rs + 1 :]
                if len(after_rs) < cfg["min_bars_after_head"]:
                    continue
                if not np.any(after_rs < neckline_price * (1 - cfg["break_buffer"])):
                    continue

            return True

        return False

