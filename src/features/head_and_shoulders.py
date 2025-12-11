from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureExtractor


class HeadAndShouldersFeatures(BaseFeatureExtractor):
    """
    Coarse, leakage-safe features used in the HS modeling script.
    """

    def __init__(self) -> None:
        self._feature_names = [
            "close_last",
            "close_mean",
            "close_std",
            "price_range",
            "return_1",
            "return_20",
            "vol_ratio",
            "range_ratio",
            "abs_ret1",
            "abs_ret20",
            "ret20_sign",
        ]

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def compute_features(self, close_window: pd.Series) -> Dict[str, float]:
        c = close_window.values
        n = len(c)

        close_last = float(c[-1])
        close_mean = float(np.mean(c))
        close_std = float(np.std(c))
        price_range = float(np.max(c) - np.min(c))
        return_1 = float((c[-1] / c[-2] - 1) if n > 1 and c[-2] != 0 else 0.0)
        return_20 = float((c[-1] / c[-21] - 1) if n > 20 and c[-21] != 0 else 0.0)

        cm_abs = abs(close_mean) + 1e-8
        vol_ratio = close_std / cm_abs
        range_ratio = price_range / cm_abs
        abs_ret1 = abs(return_1)
        abs_ret20 = abs(return_20)
        ret20_sign = float(np.sign(return_20))

        return {
            "close_last": close_last,
            "close_mean": close_mean,
            "close_std": close_std,
            "price_range": price_range,
            "return_1": return_1,
            "return_20": return_20,
            "vol_ratio": vol_ratio,
            "range_ratio": range_ratio,
            "abs_ret1": abs_ret1,
            "abs_ret20": abs_ret20,
            "ret20_sign": ret20_sign,
        }

