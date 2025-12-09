"""
Head-and-Shoulders Pattern Detector
-----------------------------------

This module detects Head & Shoulders (H&S) patterns in a price series.

A valid pattern looks like:

    left_shoulder  <  head  >  right_shoulder
           \         |         /
            \        |        /
             -------neckline-----

Pattern criteria (simplified for sliding-window detection):
- Three peaks: left shoulder < head > right shoulder
- Head must be higher than BOTH shoulders
- Shoulders should be similar height (within tolerance)
- Neckline is a local minimum between peaks
- Drop to neckline must be meaningful

This follows the structure of Raymond's double_top_detector.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HSConfig:
    shoulder_tolerance: float = 0.04    # shoulders within ±4%
    min_head_height_pct: float = 0.02   # head must be 2% higher
    min_drop_pct: float = 0.005         # drop to neckline at least 0.5%
    smoothing_window: int = 5           # for noise reduction
    require_confirmation: bool = False  # allow pattern even without final breakdown
    break_buffer: float = 0.001         # neckline buffer
    min_bars_after_head: int = 5        # bars needed for confirmation


class HeadAndShouldersDetector:

    def __init__(self, config: Optional[HSConfig] = None):
        self.config = config or HSConfig()

    @staticmethod
    def find_local_peaks(prices: np.ndarray) -> List[int]:
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def find_local_troughs(prices: np.ndarray) -> List[int]:
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(i)
        return troughs

    def smooth(self, prices: np.ndarray) -> np.ndarray:
        return pd.Series(prices).rolling(
            window=self.config.smoothing_window,
            center=True,
            min_periods=1
        ).mean().values

    def detect(self, price_series: pd.Series) -> bool:
        prices = price_series.values
        smooth = self.smooth(prices)

        peaks = self.find_local_peaks(smooth)

        # need at least 3 peaks for H&S
        if len(peaks) < 3:
            return False

        # try every consecutive triple
        for i in range(len(peaks) - 2):
            ls, head, rs = peaks[i], peaks[i+1], peaks[i+2]

            left = smooth[ls]
            mid = smooth[head]
            right = smooth[rs]

            # head must be the highest peak
            if not (mid > left and mid > right):
                continue

            # shoulders roughly equal height
            shoulder_diff = abs(left - right) / ((left + right) / 2)
            if shoulder_diff > self.config.shoulder_tolerance:
                continue

            # head must rise meaningfully above shoulders
            shoulder_avg = (left + right) / 2
            head_rise = (mid - shoulder_avg) / shoulder_avg
            if head_rise < self.config.min_head_height_pct:
                continue

            # neckline: lowest trough between shoulders
            troughs = self.find_local_troughs(smooth[ls:rs+1])
            if not troughs:
                continue

            neckline_idx = ls + troughs[np.argmin(smooth[ls:rs+1][troughs])]
            neckline_price = smooth[neckline_idx]

            # drop check: head → neckline
            drop_pct = (mid - neckline_price) / mid
            if drop_pct < self.config.min_drop_pct:
                continue

            # confirmation after right shoulder
            if self.config.require_confirmation:
                after_rs = smooth[rs+1:]
                if len(after_rs) < self.config.min_bars_after_head:
                    continue
                if not np.any(after_rs < neckline_price * (1 - self.config.break_buffer)):
                    continue

            return True  # valid pattern found

        return False


def is_head_and_shoulders(close_window: pd.Series) -> bool:
    detector = HeadAndShouldersDetector()
    return detector.detect(close_window)
