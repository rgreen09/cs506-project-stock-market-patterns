import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.features.common import find_local_peaks, find_local_troughs, smooth_prices


DEFAULT_CONFIG = {
    "min_extrema": 2,
    "min_slope_mag": 0.05,
    "corridor_tolerance": 0.05,
    "min_fraction_inside": 0.8,
    "smoothing_window": 5,
}


class TriangleDetector(BaseDetector):
    pattern_name = "triangle"

    def __init__(self, config: dict | None = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def detect(self, price_series: pd.Series, **kwargs) -> bool:
        cfg = self.config
        prices = price_series.values.astype(float)
        n = len(prices)
        if n < 10:
            return False

        t = np.linspace(0.0, 1.0, n)
        p_min, p_max = prices.min(), prices.max()
        if p_max == p_min:
            return False
        p = (prices - p_min) / (p_max - p_min)

        smooth = smooth_prices(p, window=cfg["smoothing_window"])
        peaks = find_local_peaks(smooth)
        troughs = find_local_troughs(smooth)

        if len(peaks) < cfg["min_extrema"] or len(troughs) < cfg["min_extrema"]:
            return False

        p1_idx, p2_idx = peaks[0], peaks[-1]
        l1_idx, l2_idx = troughs[0], troughs[-1]
        t1, t2 = t[p1_idx], t[p2_idx]
        lt1, lt2 = t[l1_idx], t[l2_idx]
        if t2 == t1 or lt2 == lt1:
            return False

        upper_slope = (p[p2_idx] - p[p1_idx]) / (t2 - t1)
        lower_slope = (p[l2_idx] - p[l1_idx]) / (lt2 - lt1)

        if upper_slope > -cfg["min_slope_mag"]:
            return False
        if lower_slope < cfg["min_slope_mag"]:
            return False

        t_start, t_end = 0.0, 1.0
        upper_start = p[p1_idx] + upper_slope * (t_start - t1)
        lower_start = p[l1_idx] + lower_slope * (t_start - lt1)
        upper_end = p[p1_idx] + upper_slope * (t_end - t1)
        lower_end = p[l1_idx] + lower_slope * (t_end - lt1)
        spread_start = upper_start - lower_start
        spread_end = upper_end - lower_end
        if spread_start <= spread_end:
            return False

        inside = 0
        for ti, pi in zip(t, p):
            upper_line = p[p1_idx] + upper_slope * (ti - t1)
            lower_line = p[l1_idx] + lower_slope * (ti - lt1)
            if (pi <= upper_line + cfg["corridor_tolerance"]) and (
                pi >= lower_line - cfg["corridor_tolerance"]
            ):
                inside += 1
        frac_inside = inside / n
        if frac_inside < cfg["min_fraction_inside"]:
            return False
        return True

