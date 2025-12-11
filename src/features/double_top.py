from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureExtractor
from src.features.common import calculate_linear_trend, find_local_peaks, find_local_troughs


class DoubleTopFeatures(BaseFeatureExtractor):
    """
    Feature extractor for Double-Top pattern windows.
    Mirrors the original engineered features from B_build_double_top_dataset.py.
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
            "num_peaks_window",
            "num_troughs_window",
            "peak1_rel_pos",
            "peak2_rel_pos",
            "bars_between_last_two_peaks",
            "peak_height_diff_pct",
            "neckline_drop_pct",
            "drawdown_from_last_peak",
            "peak1_sharpness",
            "peak2_sharpness",
            "rolling_std_20",
            "rolling_std_60",
            "true_range_mean_20",
            "sma_20",
            "close_over_sma20",
            "sma_50",
            "rsi_14",
            "percent_b",
        ]

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def compute_features(self, close_window: pd.Series) -> Dict[str, float]:
        c = close_window.values
        n = len(c)

        # Core stats
        features: Dict[str, float] = {}
        features["close_last"] = float(c[-1])
        features["close_mean"] = float(np.mean(c))
        features["close_std"] = float(np.std(c))
        features["price_range_abs"] = float(np.max(c) - np.min(c))
        features["price_range_pct"] = float(
            (features["price_range_abs"] / c[-1]) if c[-1] != 0 else 0.0
        )

        # Returns & momentum
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

        # Slopes
        features["slope_entire_window"] = float(calculate_linear_trend(c))
        if n >= 30:
            features["slope_last_30"] = float(calculate_linear_trend(c[-30:]))
        else:
            features["slope_last_30"] = float(calculate_linear_trend(c) if n > 1 else 0.0)

        # Peaks / troughs
        smooth_series = pd.Series(c).rolling(window=5, center=True, min_periods=1).mean()
        smooth = smooth_series.values
        peaks = find_local_peaks(smooth)
        troughs = find_local_troughs(smooth)

        features["num_peaks_window"] = float(len(peaks))
        features["num_troughs_window"] = float(len(troughs))

        if len(peaks) >= 2:
            i1, i2 = peaks[-2], peaks[-1]
            h1, h2 = smooth[i1], smooth[i2]
            neckline = float(np.min(smooth[i1 : i2 + 1]))

            features["peak1_rel_pos"] = float(i1 / (n - 1) if n > 1 else 0.0)
            features["peak2_rel_pos"] = float(i2 / (n - 1) if n > 1 else 0.0)
            features["bars_between_last_two_peaks"] = float(i2 - i1)

            max_peak = max(h1, h2)
            features["peak_height_diff_pct"] = float(
                abs(h1 - h2) / max_peak if max_peak != 0 else 0.0
            )

            min_peak = min(h1, h2)
            features["neckline_drop_pct"] = float(
                (min_peak - neckline) / min_peak if min_peak != 0 else 0.0
            )

            features["drawdown_from_last_peak"] = float(
                (h2 - c[-1]) / h2 if h2 != 0 else 0.0
            )

            if 1 <= i1 < len(smooth) - 1:
                features["peak1_sharpness"] = float(
                    h1 - (smooth[i1 - 1] + smooth[i1 + 1]) / 2
                )
            else:
                features["peak1_sharpness"] = 0.0

            if 1 <= i2 < len(smooth) - 1:
                features["peak2_sharpness"] = float(
                    h2 - (smooth[i2 - 1] + smooth[i2 + 1]) / 2
                )
            else:
                features["peak2_sharpness"] = 0.0
        else:
            features["peak1_rel_pos"] = 0.0
            features["peak2_rel_pos"] = 0.0
            features["bars_between_last_two_peaks"] = 0.0
            features["peak_height_diff_pct"] = 0.0
            features["neckline_drop_pct"] = 0.0
            features["drawdown_from_last_peak"] = 0.0
            features["peak1_sharpness"] = 0.0
            features["peak2_sharpness"] = 0.0

        # Volatility
        features["rolling_std_20"] = float(np.std(c[-20:]) if n >= 20 else np.std(c))
        features["rolling_std_60"] = float(np.std(c[-60:]) if n >= 60 else np.std(c))

        if n >= 20:
            diffs = np.diff(c[-20:])
            features["true_range_mean_20"] = float(np.mean(np.abs(diffs)) if len(diffs) else 0.0)
        elif n > 1:
            diffs = np.diff(c)
            features["true_range_mean_20"] = float(np.mean(np.abs(diffs)))
        else:
            features["true_range_mean_20"] = 0.0

        # Moving averages & oscillators
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
                rsi = 100.0 - (100.0 / (1 + rs))
            features["rsi_14"] = float(rsi)
        else:
            features["rsi_14"] = 50.0

        if n >= 20:
            ma20 = float(np.mean(c[-20:]))
            std20 = float(np.std(c[-20:]))
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            features["percent_b"] = float((c[-1] - lower) / (upper - lower)) if upper > lower else 0.0
        else:
            features["percent_b"] = 0.0

        return features

