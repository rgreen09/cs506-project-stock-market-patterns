import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.features.common import find_local_peaks, find_local_troughs, smooth_prices


DEFAULT_CONFIG = {
    "peak_tolerance": 0.02,
    "min_drop_pct": 0.005,
    "break_buffer": 0.001,
    "min_gap": 10,
    "max_gap_ratio": 0.5,
    "smoothing_window": 5,
    "require_confirmation": True,
    "min_bars_after_peak": 5,
}


class DoubleTopDetector(BaseDetector):
    pattern_name = "double_top"

    def __init__(self, config: dict | None = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def detect(self, price_series: pd.Series, window_bars: int | None = None) -> bool:
        cfg = self.config
        prices = price_series.values
        if window_bars is None:
            window_bars = len(prices)

        smooth = smooth_prices(prices, window=cfg["smoothing_window"])
        peaks = find_local_peaks(smooth)

        if len(peaks) < 2:
            return False

        max_gap = int(window_bars * cfg["max_gap_ratio"])

        for peak_idx in range(len(peaks) - 1):
            i1, i2 = peaks[peak_idx], peaks[peak_idx + 1]
            gap = i2 - i1
            if gap < cfg["min_gap"] or gap > max_gap:
                continue

            h1, h2 = smooth[i1], smooth[i2]
            if max(h1, h2) == 0:
                continue

            height_diff_pct = abs(h1 - h2) / max(h1, h2)
            if height_diff_pct > cfg["peak_tolerance"]:
                continue

            segment_between = smooth[i1 : i2 + 1]
            neckline_idx = i1 + int(np.argmin(segment_between))
            neck_price = smooth[neckline_idx]
            if neck_price <= 0:
                continue

            min_peak = min(h1, h2)
            drop_pct = (min_peak - neck_price) / min_peak
            if drop_pct < cfg["min_drop_pct"]:
                continue

            confirmed = False
            if i2 + 1 < len(smooth):
                after_second = smooth[i2 + 1 :]
                if cfg["require_confirmation"]:
                    break_threshold = neck_price * (1 - cfg["break_buffer"])
                    confirmed = np.any(after_second < break_threshold)
                    if len(after_second) >= cfg["min_bars_after_peak"] and not confirmed:
                        continue
                else:
                    confirmed = True
            else:
                if cfg["require_confirmation"]:
                    continue
                confirmed = False

            if confirmed or not cfg["require_confirmation"]:
                return True

        return False

