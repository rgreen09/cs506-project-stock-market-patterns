"""
Detector del patrón Cup and Handle en datos de precios históricos.
"""

import numpy as np
import pandas as pd
from utils import (
    find_peaks, 
    find_troughs, 
    calculate_depth_percentage,
    is_rounded_bottom,
    calculate_volume_ratio,
    get_moving_average
)


class CupAndHandleDetector:
    """
    Detector algorítmico del patrón Cup and Handle.
    """
    
    def __init__(self, 
                 min_cup_duration=7,
                 max_cup_duration=65,
                 min_cup_depth=12,
                 max_cup_depth=33,
                 peak_similarity_threshold=5,
                 min_handle_duration=5,
                 max_handle_duration=20,
                 max_handle_depth=15,
                 volume_breakout_threshold=1.2,
                 extrema_order=5):
        """
        Inicializa el detector con parámetros configurables.
        
        Args:
            min_cup_duration: Duración mínima de la taza en días
            max_cup_duration: Duración máxima de la taza en días
            min_cup_depth: Profundidad mínima de la taza (%)
            max_cup_depth: Profundidad máxima de la taza (%)
            peak_similarity_threshold: Diferencia máxima entre picos (%)
            min_handle_duration: Duración mínima del asa en días
            max_handle_duration: Duración máxima del asa en días
            max_handle_depth: Profundidad máxima del asa (%)
            volume_breakout_threshold: Ratio mínimo de volumen en breakout
            extrema_order: Ventana para detectar extremos locales
        """
        self.min_cup_duration = min_cup_duration
        self.max_cup_duration = max_cup_duration
        self.min_cup_depth = min_cup_depth
        self.max_cup_depth = max_cup_depth
        self.peak_similarity_threshold = peak_similarity_threshold
        self.min_handle_duration = min_handle_duration
        self.max_handle_duration = max_handle_duration
        self.max_handle_depth = max_handle_depth
        self.volume_breakout_threshold = volume_breakout_threshold
        self.extrema_order = extrema_order
    
    def detect_patterns(self, df):
        """
        Detecta todos los patrones Cup and Handle en un DataFrame.
        
        Args:
            df: DataFrame con columnas Date, Open, High, Low, Close, Volume
            
        Returns:
            Lista de diccionarios con información de cada patrón detectado
        """
        if len(df) < self.min_cup_duration + self.min_handle_duration + 10:
            return []
        
        closes = df['Close'].values
        volumes = df['Volume'].values
        dates = df['Date'].values
        
        # Encontrar extremos locales
        peaks = find_peaks(closes, order=self.extrema_order)
        troughs = find_troughs(closes, order=self.extrema_order)
        
        patterns = []
        
        # Iterar sobre picos para buscar formaciones de taza
        for i, peak1_idx in enumerate(peaks):
            # Buscar valles después de este pico
            valid_troughs = troughs[troughs > peak1_idx]
            
            for trough_idx in valid_troughs:
                # Buscar segundo pico después del valle
                valid_peaks2 = peaks[peaks > trough_idx]
                
                for peak2_idx in valid_peaks2:
                    # Validar la formación de la taza
                    cup_result = self._validate_cup(
                        closes, peak1_idx, trough_idx, peak2_idx
                    )
                    
                    if cup_result['valid']:
                        # Buscar el asa después del segundo pico
                        handle_result = self._find_handle(
                            closes, volumes, peak2_idx, dates
                        )
                        
                        if handle_result['valid']:
                            # Buscar confirmación de breakout
                            breakout_result = self._check_breakout(
                                df, peak1_idx, peak2_idx, 
                                handle_result['handle_end_idx']
                            )
                            
                            if breakout_result['valid']:
                                # Patrón completo detectado
                                pattern = {
                                    'ticker': None,  # Se asignará desde fuera
                                    'pattern_start_date': dates[peak1_idx],
                                    'pattern_end_date': breakout_result['breakout_date'],
                                    'cup_start_date': dates[peak1_idx],
                                    'cup_end_date': dates[peak2_idx],
                                    'handle_start_date': dates[peak2_idx],
                                    'handle_end_date': dates[handle_result['handle_end_idx']],
                                    'breakout_date': breakout_result['breakout_date'],
                                    'cup_depth_pct': cup_result['depth'],
                                    'handle_depth_pct': handle_result['depth'],
                                    'breakout_price': breakout_result['breakout_price'],
                                    'confidence_score': self._calculate_confidence(
                                        cup_result, handle_result, breakout_result
                                    )
                                }
                                patterns.append(pattern)
        
        return patterns
    
    def _validate_cup(self, closes, peak1_idx, trough_idx, peak2_idx):
        """
        Valida si tres puntos forman una taza válida.
        """
        result = {'valid': False, 'depth': 0}
        
        peak1_price = closes[peak1_idx]
        trough_price = closes[trough_idx]
        peak2_price = closes[peak2_idx]
        
        # 1. Verificar duración
        cup_duration = peak2_idx - peak1_idx
        if not (self.min_cup_duration <= cup_duration <= self.max_cup_duration):
            return result
        
        # 2. Verificar profundidad
        depth_pct = calculate_depth_percentage(peak1_price, trough_price)
        if not (self.min_cup_depth <= depth_pct <= self.max_cup_depth):
            return result
        
        # 3. Verificar que los picos sean similares
        peak_diff_pct = abs((peak1_price - peak2_price) / peak1_price * 100)
        if peak_diff_pct > self.peak_similarity_threshold:
            return result
        
        # 4. Verificar forma redondeada
        if not is_rounded_bottom(closes, trough_idx):
            return result
        
        # 5. Verificar que el valle no esté exactamente en el medio (sería V)
        trough_position = (trough_idx - peak1_idx) / cup_duration
        if 0.4 <= trough_position <= 0.6:
            # El valle está muy centrado, puede ser una V
            return result
        
        result['valid'] = True
        result['depth'] = depth_pct
        return result
    
    def _find_handle(self, closes, volumes, peak2_idx, dates):
        """
        Encuentra el asa después del segundo pico de la taza.
        """
        result = {'valid': False, 'handle_end_idx': None, 'depth': 0}
        
        # Buscar dentro de una ventana después del pico
        search_end = min(
            peak2_idx + self.max_handle_duration + 10,
            len(closes)
        )
        
        if search_end - peak2_idx < self.min_handle_duration:
            return result
        
        handle_segment = closes[peak2_idx:search_end]
        peak2_price = closes[peak2_idx]
        
        # Encontrar el mínimo en este segmento
        handle_low_rel_idx = np.argmin(handle_segment)
        handle_low_idx = peak2_idx + handle_low_rel_idx
        handle_low_price = closes[handle_low_idx]
        
        # Verificar profundidad del asa
        handle_depth = calculate_depth_percentage(peak2_price, handle_low_price)
        if handle_depth > self.max_handle_depth:
            return result
        
        # Verificar duración del asa
        handle_duration = handle_low_idx - peak2_idx
        if not (self.min_handle_duration <= handle_duration <= self.max_handle_duration):
            return result
        
        result['valid'] = True
        result['handle_end_idx'] = handle_low_idx
        result['depth'] = handle_depth
        return result
    
    def _check_breakout(self, df, peak1_idx, peak2_idx, handle_end_idx):
        """
        Verifica si hay un breakout confirmado después del asa.
        """
        result = {'valid': False, 'breakout_date': None, 'breakout_price': 0}
        
        # Determinar el nivel de resistencia
        resistance_level = max(df['Close'].iloc[peak1_idx], df['Close'].iloc[peak2_idx])
        
        # Buscar breakout en los días siguientes al asa
        search_start = handle_end_idx + 1
        search_end = min(handle_end_idx + 15, len(df))
        
        if search_start >= len(df):
            return result
        
        # Calcular volumen promedio de los últimos 20 días antes del breakout
        vol_start = max(0, handle_end_idx - 20)
        avg_volume = df['Volume'].iloc[vol_start:handle_end_idx].mean()
        
        for idx in range(search_start, search_end):
            close_price = df['Close'].iloc[idx]
            volume = df['Volume'].iloc[idx]
            
            # Verificar si el precio rompe la resistencia
            if close_price > resistance_level * 1.01:  # 1% por encima
                # Verificar volumen
                vol_ratio = calculate_volume_ratio(volume, avg_volume)
                
                if vol_ratio >= self.volume_breakout_threshold:
                    result['valid'] = True
                    result['breakout_date'] = df['Date'].iloc[idx]
                    result['breakout_price'] = close_price
                    return result
        
        return result
    
    def _calculate_confidence(self, cup_result, handle_result, breakout_result):
        """
        Calcula un score de confianza para el patrón detectado.
        
        Basado en qué tan bien se ajusta a los parámetros ideales.
        """
        score = 0.0
        
        # Cup depth ideal: 20-25%
        cup_depth = cup_result['depth']
        if 20 <= cup_depth <= 25:
            score += 0.3
        elif 15 <= cup_depth <= 30:
            score += 0.2
        else:
            score += 0.1
        
        # Handle depth ideal: 5-10%
        handle_depth = handle_result['depth']
        if 5 <= handle_depth <= 10:
            score += 0.3
        elif handle_depth <= 15:
            score += 0.2
        else:
            score += 0.1
        
        # Breakout con buen volumen
        score += 0.4  # Si llegó hasta aquí, el breakout fue válido
        
        return round(score, 2)


def detect_cup_and_handle(ticker, df, **kwargs):
    """
    Función auxiliar para detectar patrones en un ticker específico.
    
    Args:
        ticker: Símbolo de la acción
        df: DataFrame con datos históricos
        **kwargs: Parámetros opcionales para el detector
        
    Returns:
        Lista de patrones detectados
    """
    detector = CupAndHandleDetector(**kwargs)
    patterns = detector.detect_patterns(df)
    
    # Asignar el ticker a cada patrón
    for pattern in patterns:
        pattern['ticker'] = ticker
    
    return patterns

