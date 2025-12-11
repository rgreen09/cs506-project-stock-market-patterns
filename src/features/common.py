import numpy as np
import pandas as pd
from typing import List, Tuple


def smooth_prices(
    prices: np.ndarray,
    window: int = 5,
    center: bool = True,
) -> np.ndarray:
    """
    Apply a rolling mean to reduce noise prior to peak/trough detection.
    """
    return (
        pd.Series(prices)
        .rolling(window=window, center=center, min_periods=1)
        .mean()
        .values
    )


def find_local_peaks(prices: np.ndarray) -> List[int]:
    """
    Simple local maxima detection.
    """
    peaks: List[int] = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            peaks.append(i)
    return peaks


def find_local_troughs(prices: np.ndarray) -> List[int]:
    """
    Simple local minima detection.
    """
    troughs: List[int] = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            troughs.append(i)
    return troughs


def calculate_linear_trend(prices: np.ndarray) -> float:
    """
    Compute linear regression slope for a price array.
    """
    n = len(prices)
    if n < 2:
        return 0.0
    x_axis = np.arange(n)
    slope = np.polyfit(x_axis, prices, 1)[0]
    return float(slope)


def slope_and_intercept(prices: np.ndarray) -> Tuple[float, float]:
    """
    Convenience function to return slope and intercept together.
    """
    n = len(prices)
    if n < 2:
        return 0.0, float(prices[0]) if n == 1 else 0.0
    x_axis = np.arange(n)
    slope, intercept = np.polyfit(x_axis, prices, 1)
    return float(slope), float(intercept)

