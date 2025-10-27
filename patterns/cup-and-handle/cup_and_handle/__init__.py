"""
Cup and Handle Pattern Detector Module

This module provides tools for detecting the Cup and Handle pattern
in historical stock data.
"""

from .detector import CupAndHandleDetector, detect_cup_and_handle
from .data_fetcher import get_sp500_tickers, fetch_stock_data, fetch_multiple_stocks
from .utils import find_peaks, find_troughs, calculate_depth_percentage

__version__ = '1.0.0'
__all__ = [
    'CupAndHandleDetector',
    'detect_cup_and_handle',
    'get_sp500_tickers',
    'fetch_stock_data',
    'fetch_multiple_stocks',
    'find_peaks',
    'find_troughs',
    'calculate_depth_percentage'
]
