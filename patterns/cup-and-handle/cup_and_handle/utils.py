"""
Utilities for detecting local extrema in price series.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def find_peaks(data, order=5):
    """
    Finds local maxima (peaks) in a price series.
    
    Args:
        data: Series or array with prices
        order: Window to consider a point as local extremum
        
    Returns:
        Array of indices where peaks occur
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    peaks = argrelextrema(data, np.greater, order=order)[0]
    return peaks


def find_troughs(data, order=5):
    """
    Finds local minima (troughs) in a price series.
    
    Args:
        data: Series or array with prices
        order: Window to consider a point as local extremum
        
    Returns:
        Array of indices where troughs occur
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    troughs = argrelextrema(data, np.less, order=order)[0]
    return troughs


def calculate_depth_percentage(peak_price, trough_price):
    """
    Calculates the depth of a decline as a percentage.
    
    Args:
        peak_price: Price at the peak
        trough_price: Price at the trough
        
    Returns:
        Decline percentage (positive value)
    """
    return ((peak_price - trough_price) / peak_price) * 100


def is_rounded_bottom(prices, trough_idx, window=5):
    """
    Verifies if the trough has a rounded shape (not a sharp V).
    
    Compares the trough position relative to the midpoint of the window.
    A rounded shape will have the minimum closer to the center.
    
    Args:
        prices: Price array
        trough_idx: Index of the trough
        window: Window on each side of the trough to evaluate
        
    Returns:
        True if the shape is rounded
    """
    start = max(0, trough_idx - window)
    end = min(len(prices), trough_idx + window + 1)
    
    segment = prices[start:end]
    min_idx_in_segment = np.argmin(segment)
    expected_center = len(segment) // 2
    
    # If minimum is within 40% of center, consider it rounded
    deviation = abs(min_idx_in_segment - expected_center)
    max_deviation = len(segment) * 0.2
    
    return deviation <= max_deviation


def calculate_volume_ratio(volume, avg_volume):
    """
    Calculates the ratio between current volume and average volume.
    
    Args:
        volume: Volume on a specific day
        avg_volume: Average volume
        
    Returns:
        Volume ratio
    """
    if avg_volume == 0:
        return 0
    return volume / avg_volume


def get_moving_average(data, window=20):
    """
    Calculates the moving average of a series.
    
    Args:
        data: Pandas series or array
        window: Window for moving average
        
    Returns:
        Series with moving average
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=window).mean()
    else:
        return pd.Series(data).rolling(window=window).mean().values
