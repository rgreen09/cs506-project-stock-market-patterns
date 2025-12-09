"""
Triangle Pattern Detector
-------------------------

This module detects *triangle / wedge* patterns in a price series.

A valid triangle pattern here is defined heuristically as:

    - Price action is consolidating inside converging trendlines.
    - Upper trendline slopes DOWN.
    - Lower trendline slopes UP.
    - The distance between upper & lower lines (the "corridor") narrows over time.
    - Most prices lie within that corridor.

This follows the structure of the HeadAndShouldersDetector module:
    - A dataclass TriangleConfig
    - A TriangleDetector class with:
        * find_local_peaks
        * find_local_troughs
        * smooth
        * detect(price_series)
    - A convenience function: is_triangle(close_window)
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class TriangleConfig:
    # Minimum number of peaks and troughs needed to define trendlines
    min_extrema: int = 2

    # Minimum slope magnitude (in normalized [0,1] price space)
    #   upper trendline must slope DOWN with at least this magnitude
    #   lower trendline must slope UP   with at least this magnitude
    min_slope_mag: float = 0.05

    # Allowed "buffer" around the corridor for points to be considered inside
    corridor_tolerance: float = 0.05

    # Fraction of points that must lie within the corridor
    min_fraction_inside: float = 0.8

    # Rolling window used for smoothing before peak/trough detection
    smoothing_window: int = 5


class TriangleDetector:
    def __init__(self, config: Optional[TriangleConfig] = None):
        self.config = config or TriangleConfig()

    # ---------------------------------------------------------------------
    # Basic helpers for extrema & smoothing
    # ---------------------------------------------------------------------
    @staticmethod
    def find_local_peaks(values: np.ndarray) -> List[int]:
        """Return indices of simple local maxima."""
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def find_local_troughs(values: np.ndarray) -> List[int]:
        """Return indices of simple local minima."""
        troughs = []
        for i in range(1, len(values) - 1):
            if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
                troughs.append(i)
        return troughs

    def smooth(self, prices: np.ndarray) -> np.ndarray:
        """Apply simple rolling-mean smoothing."""
        return (
            pd.Series(prices)
            .rolling(window=self.config.smoothing_window, center=True, min_periods=1)
            .mean()
            .values
        )

    # ---------------------------------------------------------------------
    # Core detection logic
    # ---------------------------------------------------------------------
    def detect(self, price_series: pd.Series) -> bool:
        """
        Detect a triangle / wedge pattern in the given price series.

        Steps:
          1. Convert to numpy, ensure we have enough points.
          2. Normalize prices to [0, 1] to make shape scale-invariant.
          3. Smooth and find local peaks & troughs.
          4. Fit upper trendline using first + last peak.
          5. Fit lower trendline using first + last trough.
          6. Check that upper line slopes down, lower line slopes up,
             and their vertical distance narrows from start to end.
          7. Check that most points lie inside the corridor between them.
        """
        prices = price_series.values.astype(float)
        n = len(prices)

        if n < 10:
            # Too short for meaningful triangle
            return False

        # Normalize time as linearly spaced in [0, 1]
        t = np.linspace(0.0, 1.0, n)

        # Normalize prices to [0, 1]
        p_min, p_max = prices.min(), prices.max()
        if p_max == p_min:
            # Completely flat window
            return False
        p = (prices - p_min) / (p_max - p_min)

        # Smooth for more robust extrema detection
        smooth = self.smooth(p)

        peaks = self.find_local_peaks(smooth)
        troughs = self.find_local_troughs(smooth)

        # Need enough extrema to define meaningful trendlines
        if len(peaks) < self.config.min_extrema or len(troughs) < self.config.min_extrema:
            return False

        # Upper trendline: join first and last peak
        p1_idx, p2_idx = peaks[0], peaks[-1]
        t1, t2 = t[p1_idx], t[p2_idx]
        if t2 == t1:
            return False
        upper_slope = (p[p2_idx] - p[p1_idx]) / (t2 - t1)

        # Lower trendline: join first and last trough
        l1_idx, l2_idx = troughs[0], troughs[-1]
        lt1, lt2 = t[l1_idx], t[l2_idx]
        if lt2 == lt1:
            return False
        lower_slope = (p[l2_idx] - p[l1_idx]) / (lt2 - lt1)

        # Require converging lines:
        #   upper trendline sloping DOWN with enough magnitude
        #   lower trendline sloping UP   with enough magnitude
        if upper_slope > -self.config.min_slope_mag:
            return False
        if lower_slope < self.config.min_slope_mag:
            return False

        # Compute corridor (distance between upper & lower lines) at start and end
        t_start, t_end = 0.0, 1.0

        upper_start = p[p1_idx] + upper_slope * (t_start - t1)
        lower_start = p[l1_idx] + lower_slope * (t_start - lt1)

        upper_end = p[p1_idx] + upper_slope * (t_end - t1)
        lower_end = p[l1_idx] + lower_slope * (t_end - lt1)

        spread_start = upper_start - lower_start
        spread_end = upper_end - lower_end

        # Corridor must be *narrowing*
        if spread_start <= spread_end:
            return False

        # Check fraction of points lying inside the corridor
        inside = 0
        for ti, pi in zip(t, p):
            upper_line = p[p1_idx] + upper_slope * (ti - t1)
            lower_line = p[l1_idx] + lower_slope * (ti - lt1)

            if (pi <= upper_line + self.config.corridor_tolerance) and (
                pi >= lower_line - self.config.corridor_tolerance
            ):
                inside += 1

        frac_inside = inside / n
        if frac_inside < self.config.min_fraction_inside:
            return False

        return True


# Convenience wrapper to mirror is_head_and_shoulders(...)
def is_triangle(close_window: pd.Series) -> bool:
    detector = TriangleDetector()
    return detector.detect(close_window)
