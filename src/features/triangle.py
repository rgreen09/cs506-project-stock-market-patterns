from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureExtractor
from src.features.common import calculate_linear_trend


class TriangleFeatures(BaseFeatureExtractor):
    """
    Lightweight features for triangle windows.
    """

    def __init__(self) -> None:
        self._feature_names = [
            "close_last",
            "close_mean",
            "close_std",
            "price_range_abs",
            "price_range_pct",
            "slope_entire_window",
            "slope_last_half",
        ]

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def compute_features(self, close_window: pd.Series) -> Dict[str, float]:
        c = close_window.values.astype(float)
        n = len(c)

        features: Dict[str, float] = {}
        features["close_last"] = float(c[-1])
        features["close_mean"] = float(np.mean(c))
        features["close_std"] = float(np.std(c))
        features["price_range_abs"] = float(np.max(c) - np.min(c))
        features["price_range_pct"] = float(
            (features["price_range_abs"] / c[-1]) if c[-1] != 0 else 0.0
        )
        features["slope_entire_window"] = float(calculate_linear_trend(c))
        half = max(2, n // 2)
        features["slope_last_half"] = float(calculate_linear_trend(c[-half:]))
        return features

