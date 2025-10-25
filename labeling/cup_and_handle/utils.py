"""
Utilidades para detección de extremos locales en series de precios.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def find_peaks(data, order=5):
    """
    Encuentra máximos locales (picos) en una serie de precios.
    
    Args:
        data: Series o array con precios
        order: Ventana para considerar un punto como extremo local
        
    Returns:
        Array de índices donde ocurren los picos
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    peaks = argrelextrema(data, np.greater, order=order)[0]
    return peaks


def find_troughs(data, order=5):
    """
    Encuentra mínimos locales (valles) en una serie de precios.
    
    Args:
        data: Series o array con precios
        order: Ventana para considerar un punto como extremo local
        
    Returns:
        Array de índices donde ocurren los valles
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    troughs = argrelextrema(data, np.less, order=order)[0]
    return troughs


def calculate_depth_percentage(peak_price, trough_price):
    """
    Calcula la profundidad de una caída como porcentaje.
    
    Args:
        peak_price: Precio en el pico
        trough_price: Precio en el valle
        
    Returns:
        Porcentaje de caída (valor positivo)
    """
    return ((peak_price - trough_price) / peak_price) * 100


def is_rounded_bottom(prices, trough_idx, window=5):
    """
    Verifica si el valle tiene forma redondeada (no es una V pronunciada).
    
    Compara la posición del valle con respecto al punto medio de la ventana.
    Una forma redondeada tendrá el mínimo más cerca del centro.
    
    Args:
        prices: Array de precios
        trough_idx: Índice del valle
        window: Ventana a cada lado del valle para evaluar
        
    Returns:
        True si la forma es redondeada
    """
    start = max(0, trough_idx - window)
    end = min(len(prices), trough_idx + window + 1)
    
    segment = prices[start:end]
    min_idx_in_segment = np.argmin(segment)
    expected_center = len(segment) // 2
    
    # Si el mínimo está dentro del 40% central, consideramos que es redondeado
    deviation = abs(min_idx_in_segment - expected_center)
    max_deviation = len(segment) * 0.2
    
    return deviation <= max_deviation


def calculate_volume_ratio(volume, avg_volume):
    """
    Calcula el ratio entre volumen actual y volumen promedio.
    
    Args:
        volume: Volumen en un día específico
        avg_volume: Volumen promedio
        
    Returns:
        Ratio de volumen
    """
    if avg_volume == 0:
        return 0
    return volume / avg_volume


def get_moving_average(data, window=20):
    """
    Calcula la media móvil de una serie.
    
    Args:
        data: Series de pandas o array
        window: Ventana para el promedio móvil
        
    Returns:
        Series con el promedio móvil
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=window).mean()
    else:
        return pd.Series(data).rolling(window=window).mean().values

