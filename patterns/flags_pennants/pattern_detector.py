"""
Flags and Pennants Pattern Detector
 --
A standalone module for detecting flag and pennant continuation patterns in price data.
Can be tested independently and imported into other modules.

Usage:
    from flags_pennants_detector import FlagsPennantsDetector
    
    detector = FlagsPennantsDetector()
    is_pattern = detector.detect(price_series)
    
    # Or with custom parameters
    detector = FlagsPennantsDetector(
        pole_min_move_pct=0.02,
        consolidation_max_range_pct=0.01,
        breakout_threshold_pct=0.005,
        require_breakout=True
    )
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlagsPennantsConfig:
    """Configuration parameters for flags and pennants detection."""
    pole_min_move_pct: float = 0.02       # Minimum 2% move for pole formation
    pole_max_bars: int = 100              # Maximum bars for pole (1/3 of window)
    consolidation_min_bars: int = 10      # Minimum bars in consolidation
    consolidation_max_bars: int = 150     # Maximum bars in consolidation
    consolidation_max_range_pct: float = 0.01  # Max 1% range during consolidation
    flag_slope_tolerance: float = 0.3     # Max slope ratio for flag vs pole
    pennant_converge_ratio: float = 0.6   # Pennant width should narrow by 40%+
    breakout_threshold_pct: float = 0.005  # 0.5% breakout confirmation
    require_breakout: bool = True         # Require breakout confirmation
    min_bars_after_consolidation: int = 5  # Min bars for breakout confirmation
    volume_increase_ratio: float = 1.2    # Volume should increase 20% on breakout
    smoothing_window: int = 5             # Rolling window for smoothing


@dataclass
class FlagsPennantsResult:
    """Result of flags/pennants pattern detection."""
    is_pattern: bool
    pattern_type: Optional[str] = None    # 'flag' or 'pennant'
    pole_start_idx: Optional[int] = None
    pole_end_idx: Optional[int] = None
    pole_start_price: Optional[float] = None
    pole_end_price: Optional[float] = None
    pole_move_pct: Optional[float] = None
    consolidation_start_idx: Optional[int] = None
    consolidation_end_idx: Optional[int] = None
    consolidation_range_pct: Optional[float] = None
    consolidation_slope: Optional[float] = None
    breakout_idx: Optional[int] = None
    breakout_price: Optional[float] = None
    breakout_confirmed: Optional[bool] = None
    pole_direction: Optional[str] = None  # 'up' or 'down'
    failure_reason: Optional[str] = None


class FlagsPennantsDetector:
    """
    Detector for flag and pennant continuation patterns in price data.
    
    Flag: Strong directional move (pole) followed by rectangular consolidation
    Pennant: Strong directional move (pole) followed by converging consolidation
    """
    
    def __init__(self, config: Optional[FlagsPennantsConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or FlagsPennantsConfig()
    
    @staticmethod
    def calculate_linear_trend(prices: np.ndarray) -> Tuple[float, float]:
        """
        Calculate linear regression slope and intercept for price array.
        
        Args:
            prices: 1D numpy array of prices
            
        Returns:
            Tuple of (slope, intercept)
        """
        n = len(prices)
        if n < 2:
            return 0.0, float(prices[0]) if n == 1 else 0.0
        
        # Linear regression using numpy polyfit
        x_axis = np.arange(n)
        slope, intercept = np.polyfit(x_axis, prices, 1)
        return float(slope), float(intercept)
    
    @staticmethod
    def find_pole(prices: np.ndarray, max_bars: int) -> Optional[Tuple[int, int, float, str]]:
        """
        Find the strongest directional move (pole) in the price data.
        
        Args:
            prices: 1D numpy array of prices
            max_bars: Maximum number of bars to consider for pole
            
        Returns:
            Tuple of (start_idx, end_idx, move_pct, direction) or None if no pole found
        """
        n = len(prices)
        if n < 3:
            return None
        
        best_pole = None
        best_move_pct = 0.0
        
        # Search for strongest consecutive move within max_bars
        for start_idx in range(n - 2):
            for end_idx in range(start_idx + 2, min(n, start_idx + max_bars)):
                start_price = prices[start_idx]
                end_price = prices[end_idx]
                
                if start_price == 0:
                    continue
                
                # Calculate move percentage
                move_pct = (end_price - start_price) / start_price
                abs_move = abs(move_pct)
                
                # Keep track of strongest move
                if abs_move > best_move_pct:
                    best_move_pct = abs_move
                    direction = 'up' if move_pct > 0 else 'down'
                    best_pole = (start_idx, end_idx, move_pct, direction)
        
        return best_pole
    
    def _smooth_prices(self, prices: np.ndarray) -> np.ndarray:
        """Apply smoothing to price data to reduce noise."""
        smooth_series = pd.Series(prices).rolling(
            window=self.config.smoothing_window,
            center=True,
            min_periods=1
        ).mean()
        return smooth_series.values
    
    def _analyze_consolidation(
        self, 
        prices: np.ndarray, 
        pole_end_price: float
    ) -> Tuple[float, float, str]:
        """
        Analyze consolidation pattern characteristics.
        
        Args:
            prices: Price array of consolidation period
            pole_end_price: Price at end of pole (for normalization)
            
        Returns:
            Tuple of (range_pct, slope, pattern_type)
        """
        n = len(prices)
        if n < 2 or pole_end_price == 0:
            return 0.0, 0.0, 'unknown'
        
        # Calculate range of consolidation
        price_min = np.min(prices)
        price_max = np.max(prices)
        range_pct = (price_max - price_min) / pole_end_price
        
        # Calculate slope of consolidation
        slope, _ = self.calculate_linear_trend(prices)
        
        # Determine if it's a flag (rectangular) or pennant (converging)
        # For pennant: width should narrow over time
        first_third = prices[:n//3]
        last_third = prices[-n//3:]
        
        if len(first_third) > 0 and len(last_third) > 0:
            first_range = np.max(first_third) - np.min(first_third)
            last_range = np.max(last_third) - np.min(last_third)
            
            # Check if range is converging (pennant) or stable (flag)
            if first_range > 0:
                convergence_ratio = last_range / first_range
                if convergence_ratio < self.config.pennant_converge_ratio:
                    pattern_type = 'pennant'
                else:
                    pattern_type = 'flag'
            else:
                pattern_type = 'flag'
        else:
            pattern_type = 'flag'
        
        return float(range_pct), float(slope), pattern_type
    
    def detect(
        self, 
        price_series: pd.Series, 
        window_bars: Optional[int] = None
    ) -> FlagsPennantsResult:
        """
        Detect if a flag or pennant pattern exists in the given price series.
        
        Args:
            price_series: pandas Series of closing prices
            window_bars: Optional window size. If None, uses length of price_series.
        
        Returns:
            FlagsPennantsResult object with detection results
        """
        if window_bars is None:
            window_bars = len(price_series)
        
        # Convert to numpy and smooth
        prices = price_series.values
        smooth = self._smooth_prices(prices)
        n = len(smooth)
        
        # Need minimum data points
        if n < 20:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason=f"Insufficient data (need at least 20 bars, got {n})"
            )
        
        # Step 1: Find the pole (strongest directional move)
        pole_result = self.find_pole(smooth, self.config.pole_max_bars)
        
        if pole_result is None:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason="No significant directional move found for pole"
            )
        
        pole_start_idx, pole_end_idx, pole_move_pct, pole_direction = pole_result
        
        # Check if pole meets minimum strength requirement
        if abs(pole_move_pct) < self.config.pole_min_move_pct:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason=f"Pole move too weak ({abs(pole_move_pct)*100:.2f}% < {self.config.pole_min_move_pct*100:.2f}%)"
            )
        
        # Step 2: Find consolidation period after pole
        consolidation_start = pole_end_idx
        
        # Need enough bars after pole for consolidation
        remaining_bars = n - consolidation_start
        if remaining_bars < self.config.consolidation_min_bars:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason=f"Insufficient bars after pole for consolidation (need {self.config.consolidation_min_bars}, got {remaining_bars})"
            )
        
        # Define consolidation end (leave room for breakout if required)
        if self.config.require_breakout:
            max_consolidation_end = min(
                n - self.config.min_bars_after_consolidation,
                consolidation_start + self.config.consolidation_max_bars
            )
        else:
            max_consolidation_end = min(n, consolidation_start + self.config.consolidation_max_bars)
        
        if max_consolidation_end <= consolidation_start + self.config.consolidation_min_bars:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason="Not enough room for valid consolidation period"
            )
        
        # Try different consolidation end points to find best pattern
        best_consolidation_end = None
        best_pattern_score = float('-inf')
        best_analysis = None
        
        for consolidation_end in range(
            consolidation_start + self.config.consolidation_min_bars,
            max_consolidation_end + 1
        ):
            consolidation_prices = smooth[consolidation_start:consolidation_end]
            pole_end_price = smooth[pole_end_idx]
            
            # Analyze this consolidation segment
            range_pct, slope, pattern_type = self._analyze_consolidation(
                consolidation_prices, 
                pole_end_price
            )
            
            # Check if consolidation range is acceptable
            if range_pct > self.config.consolidation_max_range_pct:
                continue
            
            # Check if flag slope is acceptable (should be relatively flat or counter-trend)
            if pattern_type == 'flag':
                pole_slope = (smooth[pole_end_idx] - smooth[pole_start_idx]) / (pole_end_idx - pole_start_idx) if pole_end_idx > pole_start_idx else 0
                if abs(pole_slope) > 0:
                    slope_ratio = abs(slope / pole_slope)
                    if slope_ratio > self.config.flag_slope_tolerance:
                        continue
            
            # Score this consolidation (prefer tighter consolidation)
            pattern_score = -range_pct  # Negative because lower range is better
            
            if pattern_score > best_pattern_score:
                best_pattern_score = pattern_score
                best_consolidation_end = consolidation_end
                best_analysis = (range_pct, slope, pattern_type)
        
        if best_consolidation_end is None or best_analysis is None:
            return FlagsPennantsResult(
                is_pattern=False,
                failure_reason="No valid consolidation period found meeting criteria"
            )

        # Extract best consolidation analysis
        range_pct, consolidation_slope, pattern_type = best_analysis
        consolidation_end_idx = best_consolidation_end
        
        # Step 3: Check for breakout confirmation if required
        breakout_confirmed = False
        breakout_idx = None
        breakout_price = None
        
        if consolidation_end_idx < n:
            # Define breakout threshold based on pole direction
            consolidation_high = np.max(smooth[consolidation_start:consolidation_end_idx])
            consolidation_low = np.min(smooth[consolidation_start:consolidation_end_idx])
            
            if pole_direction == 'up':
                # For bullish pattern, breakout should be above consolidation high
                breakout_threshold = consolidation_high * (1 + self.config.breakout_threshold_pct)
            else:
                # For bearish pattern, breakout should be below consolidation low
                breakout_threshold = consolidation_low * (1 - self.config.breakout_threshold_pct)
            
            # Check for breakout in remaining bars
            after_consolidation = smooth[consolidation_end_idx:]
            
            if len(after_consolidation) > 0:
                if pole_direction == 'up':
                    breakout_mask = after_consolidation > breakout_threshold
                else:
                    breakout_mask = after_consolidation < breakout_threshold
                
                if np.any(breakout_mask):
                    breakout_confirmed = True
                    # Find first breakout point
                    breakout_relative_idx = np.argmax(breakout_mask)
                    breakout_idx = consolidation_end_idx + breakout_relative_idx
                    breakout_price = float(smooth[breakout_idx])
            
            # If breakout required but not confirmed, reject pattern
            if self.config.require_breakout:
                bars_after = len(after_consolidation)
                if bars_after >= self.config.min_bars_after_consolidation and not breakout_confirmed:
                    return FlagsPennantsResult(
                        is_pattern=False,
                        failure_reason="Breakout not confirmed after consolidation"
                    )
        else:
            # No data after consolidation
            if self.config.require_breakout:
                return FlagsPennantsResult(
                    is_pattern=False,
                    failure_reason="No data available for breakout confirmation"
                )
        
        # Pattern detected successfully
        return FlagsPennantsResult(
            is_pattern=True,
            pattern_type=pattern_type,
            pole_start_idx=pole_start_idx,
            pole_end_idx=pole_end_idx,
            pole_start_price=float(smooth[pole_start_idx]),
            pole_end_price=float(smooth[pole_end_idx]),
            pole_move_pct=float(pole_move_pct),
            consolidation_start_idx=consolidation_start,
            consolidation_end_idx=consolidation_end_idx,
            consolidation_range_pct=float(range_pct),
            consolidation_slope=float(consolidation_slope),
            breakout_idx=breakout_idx,
            breakout_price=breakout_price,
            breakout_confirmed=breakout_confirmed,
            pole_direction=pole_direction
        )
    
    def detect_simple(
        self, 
        price_series: pd.Series, 
        window_bars: Optional[int] = None
    ) -> bool:
        """
        Simple boolean detection (for backward compatibility).
        
        Args:
            price_series: pandas Series of closing prices
            window_bars: Optional window size
            
        Returns:
            True if flag or pennant pattern is detected, False otherwise
        """
        result = self.detect(price_series, window_bars)
        return result.is_pattern


# Convenience function for backward compatibility
def is_flags_pennants_in_window(
    close_window: pd.Series,
    window_bars: int = 300,
    pole_min_move_pct: float = 0.02,
    consolidation_max_range_pct: float = 0.01,
    require_breakout: bool = True
) -> bool:
    """
    Legacy function for detecting flag/pennant patterns.
    
    Args:
        close_window: pandas Series of closing prices
        window_bars: size of the window (default 300)
        pole_min_move_pct: minimum pole movement percentage (default 0.02)
        consolidation_max_range_pct: max consolidation range (default 0.01)
        require_breakout: require breakout confirmation (default True)
        
    Returns:
        True if flag or pennant pattern is detected, False otherwise
    """
    config = FlagsPennantsConfig(
        pole_min_move_pct=pole_min_move_pct,
        consolidation_max_range_pct=consolidation_max_range_pct,
        require_breakout=require_breakout
    )
    detector = FlagsPennantsDetector(config)
    return detector.detect_simple(close_window, window_bars)


if __name__ == "__main__":
    """
    Test the detector with sample data.
    """
    import matplotlib.pyplot as plt
    
    # Create a synthetic flag pattern
    np.random.seed(42)
    n = 300
    
    # Phase 1: Strong upward pole (bars 0-50)
    pole_start = 0
    pole_end = 50
    pole_prices = np.linspace(100, 115, pole_end - pole_start)  # 15% gain
    
    # Phase 2: Consolidation flag (bars 50-150)
    consolidation_start = pole_end
    consolidation_end = 150
    consolidation_length = consolidation_end - consolidation_start
    # Slight downward drift in flag
    consolidation_prices = np.linspace(115, 113, consolidation_length)
    # Add oscillation within tight range
    oscillation = 0.5 * np.sin(np.linspace(0, 4*np.pi, consolidation_length))
    consolidation_prices += oscillation
    
    # Phase 3: Breakout (bars 150-200)
    breakout_start = consolidation_end
    breakout_end = 200
    breakout_prices = np.linspace(113, 120, breakout_end - breakout_start)  # Continue upward
    
    # Phase 4: Post-breakout continuation
    continuation_start = breakout_end
    continuation_prices = np.linspace(120, 122, n - continuation_start)
    
    # Combine all phases
    prices = np.concatenate([
        pole_prices,
        consolidation_prices,
        breakout_prices,
        continuation_prices
    ])
    
    # Add noise
    prices += np.random.normal(0, 0.2, n)
    
    # Convert to Series
    price_series = pd.Series(prices)
    
    # Test detection
    print("Testing Flags and Pennants Detector")
    print("=" * 60)
    
    detector = FlagsPennantsDetector()
    result = detector.detect(price_series, window_bars=300)
    
    print(f"Pattern detected: {result.is_pattern}")
    if result.is_pattern:
        print(f"  Pattern type: {result.pattern_type}")
        print(f"  Pole direction: {result.pole_direction}")
        print(f"  Pole: index {result.pole_start_idx} to {result.pole_end_idx}")
        print(f"  Pole move: {result.pole_move_pct*100:.2f}%")
        print(f"  Consolidation: index {result.consolidation_start_idx} to {result.consolidation_end_idx}")
        print(f"  Consolidation range: {result.consolidation_range_pct*100:.2f}%")
        print(f"  Consolidation slope: {result.consolidation_slope:.4f}")
        print(f"  Breakout confirmed: {result.breakout_confirmed}")
        if result.breakout_confirmed:
            print(f"  Breakout at index {result.breakout_idx}, price {result.breakout_price:.2f}")
    else:
        print(f"  Failure reason: {result.failure_reason}")
    
    # Test with relaxed parameters
    print("\n" + "=" * 60)
    print("Testing with no breakout requirement:")
    config_relaxed = FlagsPennantsConfig(require_breakout=False)
    detector_relaxed = FlagsPennantsDetector(config_relaxed)
    result_relaxed = detector_relaxed.detect(price_series, window_bars=300)
    print(f"Pattern detected: {result_relaxed.is_pattern}")
    
    # Plot the pattern
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(price_series.values, label='Price', linewidth=1.5)
        
        if result.is_pattern:
            # Mark pole
            plt.axvspan(result.pole_start_idx, result.pole_end_idx, 
                       alpha=0.2, color='blue', label='Pole')
            plt.plot(result.pole_start_idx, result.pole_start_price, 
                    'bo', markersize=8)
            plt.plot(result.pole_end_idx, result.pole_end_price, 
                    'bo', markersize=8)
            
            # Mark consolidation
            plt.axvspan(result.consolidation_start_idx, result.consolidation_end_idx, 
                       alpha=0.2, color='yellow', label='Consolidation')
            
            # Mark breakout if confirmed
            if result.breakout_confirmed:
                plt.plot(result.breakout_idx, result.breakout_price, 
                        'go', markersize=10, label='Breakout')
        
        plt.title(f'Flags and Pennants Pattern Detection Test\n'
                 f'Pattern: {result.pattern_type if result.is_pattern else "None"}')
        plt.xlabel('Bar Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('flags_pennants_test.png', dpi=150)
        print("\nPlot saved to 'flags_pennants_test.png'")
    except ImportError:
        print("\nMatplotlib not available, skipping plot")