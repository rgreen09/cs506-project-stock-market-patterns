"""
Double-Top Pattern Detector

A standalone module for detecting double-top patterns in price data.
Can be tested independently and imported into other modules.

Usage:
    from double_top_detector import DoubleTopDetector
    
    detector = DoubleTopDetector()
    is_pattern = detector.detect(price_series)
    
    # Or with custom parameters
    detector = DoubleTopDetector(
        peak_tolerance=0.02,
        min_drop_pct=0.005,
        min_gap=10,
        require_confirmation=True
    )
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DoubleTopConfig:
    """Configuration parameters for double-top detection."""
    peak_tolerance: float = 0.02      # Peaks within 2% in height
    min_drop_pct: float = 0.005       # At least 0.5% drop to neckline
    break_buffer: float = 0.001       # Small extra buffer below neckline
    min_gap: int = 10                 # Minimum bars between peaks
    max_gap_ratio: float = 0.5        # Max gap as ratio of window size
    smoothing_window: int = 5         # Rolling window for smoothing
    require_confirmation: bool = True  # Require break below neckline
    min_bars_after_peak: int = 5      # Min bars needed for confirmation


@dataclass
class DoubleTopResult:
    """Result of double-top detection."""
    is_pattern: bool
    peak1_idx: Optional[int] = None
    peak2_idx: Optional[int] = None
    peak1_price: Optional[float] = None
    peak2_price: Optional[float] = None
    neckline_price: Optional[float] = None
    neckline_idx: Optional[int] = None
    drop_pct: Optional[float] = None
    height_diff_pct: Optional[float] = None
    bars_between_peaks: Optional[int] = None
    confirmed: Optional[bool] = None
    failure_reason: Optional[str] = None


class DoubleTopDetector:
    """
    Detector for double-top patterns in price data.
    """
    
    def __init__(self, config: Optional[DoubleTopConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or DoubleTopConfig()
    
    @staticmethod
    def find_local_peaks(prices: np.ndarray, min_prominence: float = 0.001) -> List[int]:
        """
        Find local maxima in a price array with minimum prominence.
        
        A local peak at index i is defined as:
        prices[i] > prices[i-1] and prices[i] > prices[i+1]
        AND the peak has minimum prominence (height above neighbors).
        
        Args:
            prices: 1D numpy array of prices
            min_prominence: Minimum relative prominence (as fraction of price, default 0.1%)
            
        Returns:
            List of indices where local peaks occur
        """
        peaks = []
        n = len(prices)
        
        if n < 3:
            return peaks
        
        # Calculate minimum absolute prominence based on average price
        avg_price = np.mean(prices)
        min_abs_prominence = avg_price * min_prominence
        
        for i in range(1, n - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                # Check prominence: peak should be significantly higher than neighbors
                left_valley = min(prices[max(0, i-3):i])
                right_valley = min(prices[i+1:min(n, i+4)])
                valley = min(left_valley, right_valley)
                prominence = prices[i] - valley
                
                if prominence >= min_abs_prominence:
                    peaks.append(i)
        
        return peaks
    
    @staticmethod
    def find_local_troughs(prices: np.ndarray) -> List[int]:
        """
        Find local minima in a price array.
        
        A local trough at index i is defined as:
        prices[i] < prices[i-1] and prices[i] < prices[i+1]
        
        Args:
            prices: 1D numpy array of prices
            
        Returns:
            List of indices where local troughs occur
        """
        troughs = []
        n = len(prices)
        
        if n < 3:
            return troughs
        
        for i in range(1, n - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(i)
        
        return troughs
    
    def _smooth_prices(self, prices: np.ndarray) -> np.ndarray:
        """Apply smoothing to price data."""
        smooth_series = pd.Series(prices).rolling(
            window=self.config.smoothing_window,
            center=True,
            min_periods=1
        ).mean()
        return smooth_series.values
    
    def detect(self, price_series: pd.Series, window_bars: Optional[int] = None) -> DoubleTopResult:
        """
        Detect if a double-top pattern exists in the given price series.
        
        Args:
            price_series: pandas Series of closing prices
            window_bars: Optional window size (for max_gap calculation). 
                        If None, uses length of price_series.
        
        Returns:
            DoubleTopResult object with detection results
        """
        if window_bars is None:
            window_bars = len(price_series)
        
        # Convert to numpy array and smooth
        prices = price_series.values
        smooth = self._smooth_prices(prices)
        
        # Find local peaks
        peaks = self.find_local_peaks(smooth)
        
        # Need at least 2 peaks
        if len(peaks) < 2:
            return DoubleTopResult(
                is_pattern=False,
                failure_reason=f"Insufficient peaks (need at least 2, found {len(peaks)})"
            )
        
        # Calculate max gap
        max_gap = int(window_bars * self.config.max_gap_ratio)
        
        # Check ALL pairs of peaks, not just the last two
        # This allows us to find patterns anywhere in the window
        for peak_idx in range(len(peaks) - 1):
            i1, i2 = peaks[peak_idx], peaks[peak_idx + 1]
            
            # Check spacing
            gap = i2 - i1
            if gap < self.config.min_gap or gap > max_gap:
                continue
            
            # Get peak heights
            h1, h2 = smooth[i1], smooth[i2]
            if max(h1, h2) == 0:
                continue
            
            # Check height similarity
            height_diff_pct = abs(h1 - h2) / max(h1, h2)
            if height_diff_pct > self.config.peak_tolerance:
                continue
            
            # Find neckline (minimum between peaks)
            segment_between = smooth[i1:i2+1]
            neckline_idx = i1 + np.argmin(segment_between)
            neck_price = smooth[neckline_idx]
            
            if neck_price <= 0:
                continue
            
            # Check drop from peaks to neckline
            min_peak = min(h1, h2)
            drop_pct = (min_peak - neck_price) / min_peak
            if drop_pct < self.config.min_drop_pct:
                continue
            
            # Confirmation: price should break below neckline after second peak
            confirmed = False
            if i2 + 1 < len(smooth):
                after_second = smooth[i2+1:]
                bars_after = len(after_second)
                
                if self.config.require_confirmation:
                    # Check if price goes below neckline
                    break_threshold = neck_price * (1 - self.config.break_buffer)
                    confirmed = np.any(after_second < break_threshold)
                    
                    # If we have enough data after the peak, require confirmation
                    if bars_after >= self.config.min_bars_after_peak and not confirmed:
                        continue
                    # If we're very close to the end, allow without confirmation
                else:
                    # Don't require confirmation
                    confirmed = True
            else:
                # No data after second peak
                if self.config.require_confirmation:
                    # If we require confirmation but have no data, skip
                    continue
                else:
                    # Allow pattern without confirmation
                    confirmed = False
            
            # All criteria met for this peak pair
            return DoubleTopResult(
                is_pattern=True,
                peak1_idx=i1,
                peak2_idx=i2,
                peak1_price=float(h1),
                peak2_price=float(h2),
                neckline_price=float(neck_price),
                neckline_idx=neckline_idx,
                drop_pct=float(drop_pct),
                height_diff_pct=float(height_diff_pct),
                bars_between_peaks=gap,
                confirmed=confirmed
            )
        
        # No valid double-top found
        return DoubleTopResult(
            is_pattern=False,
            failure_reason="No valid peak pairs found matching criteria"
        )
    
    def detect_simple(self, price_series: pd.Series, window_bars: Optional[int] = None) -> bool:
        """
        Simple boolean detection (for backward compatibility).
        
        Args:
            price_series: pandas Series of closing prices
            window_bars: Optional window size
            
        Returns:
            True if double-top pattern is detected, False otherwise
        """
        result = self.detect(price_series, window_bars)
        return result.is_pattern


# Convenience function for backward compatibility
def is_double_top_in_window(
    close_window: pd.Series,
    window_bars: int = 300,
    peak_tolerance: float = 0.02,
    min_drop_pct: float = 0.005,
    require_confirmation: bool = True
) -> bool:
    """
    Legacy function for detecting double-top patterns.
    
    Args:
        close_window: pandas Series of closing prices
        window_bars: size of the window (default 300)
        peak_tolerance: tolerance for peak height similarity (default 0.02)
        min_drop_pct: minimum drop to neckline (default 0.005)
        require_confirmation: require break below neckline (default True)
        
    Returns:
        True if double-top pattern is detected, False otherwise
    """
    config = DoubleTopConfig(
        peak_tolerance=peak_tolerance,
        min_drop_pct=min_drop_pct,
        require_confirmation=require_confirmation
    )
    detector = DoubleTopDetector(config)
    return detector.detect_simple(close_window, window_bars)


if __name__ == "__main__":
    """
    Test the detector with sample data.
    """
    import matplotlib.pyplot as plt
    
    # Create a synthetic double-top pattern
    np.random.seed(42)
    n = 300
    
    # Create base trend
    base = np.linspace(100, 105, n)
    
    # Add first peak
    peak1_idx = 80
    peak1_height = 110
    peak1_width = 20
    
    # Add valley
    valley_idx = 150
    valley_height = 105
    
    # Add second peak (similar height)
    peak2_idx = 220
    peak2_height = 110.5  # Slightly different but within tolerance
    peak2_width = 20
    
    # Build the pattern
    prices = base.copy()
    
    # First peak
    for i in range(peak1_idx - peak1_width, peak1_idx + peak1_width):
        if 0 <= i < n:
            dist = abs(i - peak1_idx) / peak1_width
            prices[i] = peak1_height - dist * 5
    
    # Valley between peaks
    for i in range(peak1_idx, peak2_idx):
        if 0 <= i < n:
            progress = (i - peak1_idx) / (peak2_idx - peak1_idx)
            prices[i] = peak1_height - (peak1_height - valley_height) * progress
    
    # Second peak
    for i in range(peak2_idx - peak2_width, peak2_idx + peak2_width):
        if 0 <= i < n:
            dist = abs(i - peak2_idx) / peak2_width
            prices[i] = peak2_height - dist * 5
    
    # Add noise
    prices += np.random.normal(0, 0.2, n)
    
    # Add confirmation: price drops below neckline after second peak
    neckline = valley_height
    for i in range(peak2_idx + 10, n):
        prices[i] = neckline - 0.5  # Break below neckline
    
    # Convert to Series
    price_series = pd.Series(prices)
    
    # Test detection
    print("Testing Double-Top Detector")
    print("=" * 60)
    
    detector = DoubleTopDetector()
    result = detector.detect(price_series, window_bars=300)
    
    print(f"Pattern detected: {result.is_pattern}")
    if result.is_pattern:
        print(f"  Peak 1: index {result.peak1_idx}, price {result.peak1_price:.2f}")
        print(f"  Peak 2: index {result.peak2_idx}, price {result.peak2_price:.2f}")
        print(f"  Neckline: index {result.neckline_idx}, price {result.neckline_price:.2f}")
        print(f"  Drop %: {result.drop_pct*100:.2f}%")
        print(f"  Height diff %: {result.height_diff_pct*100:.2f}%")
        print(f"  Bars between peaks: {result.bars_between_peaks}")
        print(f"  Confirmed: {result.confirmed}")
    else:
        print(f"  Failure reason: {result.failure_reason}")
    
    # Test with different configurations
    print("\n" + "=" * 60)
    print("Testing with relaxed confirmation (no confirmation required):")
    config_relaxed = DoubleTopConfig(require_confirmation=False)
    detector_relaxed = DoubleTopDetector(config_relaxed)
    result_relaxed = detector_relaxed.detect(price_series, window_bars=300)
    print(f"Pattern detected: {result_relaxed.is_pattern}")
    
    # Optional: Plot the pattern
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(price_series.values, label='Price')
        if result.is_pattern:
            plt.plot(result.peak1_idx, result.peak1_price, 'ro', markersize=10, label='Peak 1')
            plt.plot(result.peak2_idx, result.peak2_price, 'ro', markersize=10, label='Peak 2')
            plt.axhline(y=result.neckline_price, color='g', linestyle='--', label='Neckline')
        plt.title('Double-Top Pattern Detection Test')
        plt.xlabel('Bar Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('double_top_test.png', dpi=150)
        print("\nPlot saved to 'double_top_test.png'")
    except ImportError:
        print("\nMatplotlib not available, skipping plot")

