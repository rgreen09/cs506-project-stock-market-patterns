"""
Cup and Handle Pattern Detector Module

Este módulo proporciona herramientas para detectar el patrón Cup and Handle
en datos históricos de acciones.
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

