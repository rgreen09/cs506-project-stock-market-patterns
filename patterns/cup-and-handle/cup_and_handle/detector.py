"""
Cup and Handle pattern detector for historical price data.
"""

import numpy as np
import pandas as pd
try:
    # When imported as package
    from .utils import (
        find_peaks, 
        find_troughs, 
        calculate_depth_percentage,
        is_rounded_bottom,
        calculate_volume_ratio,
        get_moving_average
    )
except ImportError:
    # When run directly
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
    Algorithmic detector for the Cup and Handle pattern.
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
        Initializes the detector with configurable parameters.
        
        Args:
            min_cup_duration: Minimum cup duration in days
            max_cup_duration: Maximum cup duration in days
            min_cup_depth: Minimum cup depth (%)
            max_cup_depth: Maximum cup depth (%)
            peak_similarity_threshold: Maximum difference between peaks (%)
            min_handle_duration: Minimum handle duration in days
            max_handle_duration: Maximum handle duration in days
            max_handle_depth: Maximum handle depth (%)
            volume_breakout_threshold: Minimum volume ratio on breakout
            extrema_order: Window for detecting local extrema
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
        Detects all Cup and Handle patterns in a DataFrame.
        
        Args:
            df: DataFrame with columns Date, Open, High, Low, Close, Volume
            
        Returns:
            List of dictionaries with information for each detected pattern
        """
        if len(df) < self.min_cup_duration + self.min_handle_duration + 10:
            return []
        
        closes = df['Close'].values
        volumes = df['Volume'].values
        dates = df['Date'].values
        
        # Find local extrema
        peaks = find_peaks(closes, order=self.extrema_order)
        troughs = find_troughs(closes, order=self.extrema_order)
        
        patterns = []
        
        # Iterate over peaks to search for cup formations
        for i, peak1_idx in enumerate(peaks):
            # Search for troughs after this peak
            valid_troughs = troughs[troughs > peak1_idx]
            
            for trough_idx in valid_troughs:
                # Search for second peak after the trough
                valid_peaks2 = peaks[peaks > trough_idx]
                
                for peak2_idx in valid_peaks2:
                    # Validate the cup formation
                    cup_result = self._validate_cup(
                        closes, peak1_idx, trough_idx, peak2_idx
                    )
                    
                    if cup_result['valid']:
                        # Search for the handle after the second peak
                        handle_result = self._find_handle(
                            closes, volumes, peak2_idx, dates
                        )
                        
                        if handle_result['valid']:
                            # Search for breakout confirmation
                            breakout_result = self._check_breakout(
                                df, peak1_idx, peak2_idx, 
                                handle_result['handle_end_idx']
                            )
                            
                            if breakout_result['valid']:
                                # Complete pattern detected
                                pattern = {
                                    'ticker': None,  # Will be assigned from outside
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
        Validates if three points form a valid cup.
        """
        result = {'valid': False, 'depth': 0}
        
        peak1_price = closes[peak1_idx]
        trough_price = closes[trough_idx]
        peak2_price = closes[peak2_idx]
        
        # 1. Verify duration
        cup_duration = peak2_idx - peak1_idx
        if not (self.min_cup_duration <= cup_duration <= self.max_cup_duration):
            return result
        
        # 2. Verify depth
        depth_pct = calculate_depth_percentage(peak1_price, trough_price)
        if not (self.min_cup_depth <= depth_pct <= self.max_cup_depth):
            return result
        
        # 3. Verify that peaks are similar
        peak_diff_pct = abs((peak1_price - peak2_price) / peak1_price * 100)
        if peak_diff_pct > self.peak_similarity_threshold:
            return result
        
        # 4. Verify rounded shape
        if not is_rounded_bottom(closes, trough_idx):
            return result
        
        # 5. Verify trough is not exactly in the middle (would be V)
        trough_position = (trough_idx - peak1_idx) / cup_duration
        if 0.4 <= trough_position <= 0.6:
            # Trough is too centered, may be a V
            return result
        
        result['valid'] = True
        result['depth'] = depth_pct
        return result
    
    def _find_handle(self, closes, volumes, peak2_idx, dates):
        """
        Finds the handle after the second cup peak.
        """
        result = {'valid': False, 'handle_end_idx': None, 'depth': 0}
        
        # Search within a window after the peak
        search_end = min(
            peak2_idx + self.max_handle_duration + 10,
            len(closes)
        )
        
        if search_end - peak2_idx < self.min_handle_duration:
            return result
        
        handle_segment = closes[peak2_idx:search_end]
        peak2_price = closes[peak2_idx]
        
        # Find the minimum in this segment
        handle_low_rel_idx = np.argmin(handle_segment)
        handle_low_idx = peak2_idx + handle_low_rel_idx
        handle_low_price = closes[handle_low_idx]
        
        # Verify handle depth
        handle_depth = calculate_depth_percentage(peak2_price, handle_low_price)
        if handle_depth > self.max_handle_depth:
            return result
        
        # Verify handle duration
        handle_duration = handle_low_idx - peak2_idx
        if not (self.min_handle_duration <= handle_duration <= self.max_handle_duration):
            return result
        
        result['valid'] = True
        result['handle_end_idx'] = handle_low_idx
        result['depth'] = handle_depth
        return result
    
    def _check_breakout(self, df, peak1_idx, peak2_idx, handle_end_idx):
        """
        Verifies if there's a confirmed breakout after the handle.
        """
        result = {'valid': False, 'breakout_date': None, 'breakout_price': 0}
        
        # Determine resistance level
        resistance_level = max(df['Close'].iloc[peak1_idx], df['Close'].iloc[peak2_idx])
        
        # Search for breakout in the days following the handle
        search_start = handle_end_idx + 1
        search_end = min(handle_end_idx + 15, len(df))
        
        if search_start >= len(df):
            return result
        
        # Calculate average volume of last 20 days before breakout
        vol_start = max(0, handle_end_idx - 20)
        avg_volume = df['Volume'].iloc[vol_start:handle_end_idx].mean()
        
        for idx in range(search_start, search_end):
            close_price = df['Close'].iloc[idx]
            volume = df['Volume'].iloc[idx]
            
            # Verify if price breaks resistance
            if close_price > resistance_level * 1.01:  # 1% above
                # Verify volume
                vol_ratio = calculate_volume_ratio(volume, avg_volume)
                
                if vol_ratio >= self.volume_breakout_threshold:
                    result['valid'] = True
                    result['breakout_date'] = df['Date'].iloc[idx]
                    result['breakout_price'] = close_price
                    return result
        
        return result
    
    def _calculate_confidence(self, cup_result, handle_result, breakout_result):
        """
        Calculates a confidence score for the detected pattern.
        
        Based on how well it fits the ideal parameters.
        """
        score = 0.0
        
        # Ideal cup depth: 20-25%
        cup_depth = cup_result['depth']
        if 20 <= cup_depth <= 25:
            score += 0.3
        elif 15 <= cup_depth <= 30:
            score += 0.2
        else:
            score += 0.1
        
        # Ideal handle depth: 5-10%
        handle_depth = handle_result['depth']
        if 5 <= handle_depth <= 10:
            score += 0.3
        elif handle_depth <= 15:
            score += 0.2
        else:
            score += 0.1
        
        # Breakout with good volume
        score += 0.4  # If it got here, breakout was valid
        
        return round(score, 2)


def detect_cup_and_handle(ticker, df, **kwargs):
    """
    Helper function to detect patterns for a specific ticker.
    
    Args:
        ticker: Stock symbol
        df: DataFrame with historical data
        **kwargs: Optional parameters for the detector
        
    Returns:
        List of detected patterns
    """
    detector = CupAndHandleDetector(**kwargs)
    patterns = detector.detect_patterns(df)
    
    # Assign ticker to each pattern
    for pattern in patterns:
        pattern['ticker'] = ticker
    
    return patterns
