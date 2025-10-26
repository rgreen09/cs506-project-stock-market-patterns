"""
Module for visualizing detected Cup and Handle patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def plot_cup_and_handle(df, pattern, save_path=None):
    """
    Generates a candlestick chart showing the detected pattern.
    
    Args:
        df: DataFrame with OHLCV data
        pattern: Dictionary with pattern information
        save_path: Path to save the image (None to display)
    """
    # Extract pattern dates - use cup start for better context
    cup_start = pd.to_datetime(pattern['cup_start_date'])
    pattern_end = pd.to_datetime(pattern['breakout_date'])
    
    # Remove timezone for calculations
    if hasattr(cup_start, 'tz') and cup_start.tz is not None:
        cup_start = cup_start.tz_localize(None)
    if hasattr(pattern_end, 'tz') and pattern_end.tz is not None:
        pattern_end = pattern_end.tz_localize(None)
    
    # Calculate total pattern duration
    total_duration = (pattern_end - cup_start).days
    
    # Optimal range for clear visibility without being too wide
    # 8 days before, pattern, 8 days after
    margin_days = 8
    margin = timedelta(days=margin_days)
    
    start_date = cup_start - margin
    end_date = pattern_end + margin
    
    # Filter data for the range
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remove timezone info for comparison if present
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    plot_df = df.loc[mask].copy()
    
    if plot_df.empty:
        print(f"⚠️  No data to visualize pattern for {pattern['ticker']}")
        return
    
    # Use matplotlib directly for full control
    num_days = len(plot_df)
    # Sweet spot: 0.35 inches per day - visible candles, manageable width
    width = max(18, min(35, num_days * 0.35))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)
    
    # Plot candlesticks manually with slightly wider bodies
    for idx, row in plot_df.iterrows():
        date = mdates.date2num(row['Date'])
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        
        # Color based on up/down day
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw high-low line (wick)
        ax1.plot([date, date], [low_price, high_price], color=color, linewidth=2, alpha=0.8)
        
        # Draw open-close rectangle (body) - slightly wider for better visibility
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        rect = Rectangle((date - 0.35, bottom), 0.7, height if height > 0 else 0.01,
                        facecolor=color, edgecolor=color, alpha=0.8, linewidth=1.5)
        ax1.add_patch(rect)
    
    # Pattern markers
    cup_start_date = pd.to_datetime(pattern['cup_start_date'])
    cup_end_date = pd.to_datetime(pattern['cup_end_date'])
    breakout_date = pd.to_datetime(pattern['breakout_date'])
    
    # Remove timezone
    if hasattr(cup_start_date, 'tz') and cup_start_date.tz is not None:
        cup_start_date = cup_start_date.tz_localize(None)
    if hasattr(cup_end_date, 'tz') and cup_end_date.tz is not None:
        cup_end_date = cup_end_date.tz_localize(None)
    if hasattr(breakout_date, 'tz') and breakout_date.tz is not None:
        breakout_date = breakout_date.tz_localize(None)
    
    # Resistance line
    resistance_price = pattern['breakout_price'] * 0.99
    ax1.axhline(y=resistance_price, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Resistance')
    
    # Vertical markers
    ax1.axvline(x=mdates.date2num(cup_start_date), color='blue', linestyle='--', linewidth=2.5, alpha=0.7)
    ax1.axvline(x=mdates.date2num(cup_end_date), color='purple', linestyle='--', linewidth=2.5, alpha=0.7)
    ax1.axvline(x=mdates.date2num(breakout_date), color='green', linestyle='-', linewidth=3, alpha=0.9)
    
    # Volume bars (wider to match candles)
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in plot_df.iterrows()]
    ax2.bar(mdates.date2num(plot_df['Date']), plot_df['Volume'], color=colors, alpha=0.7, width=0.7)
    
    # Formatting
    ax1.set_title(f"{pattern['ticker']} - Cup and Handle (Confidence: {pattern['confidence_score']})", 
                  fontsize=16, weight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, num_days // 15)))
    
    ax2.set_ylabel('Volume', fontsize=12, weight='bold')
    ax2.set_xlabel('Date', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Info box
    info_text = (
        f"Cup Depth: {pattern['cup_depth_pct']:.1f}%\n"
        f"Handle Depth: {pattern['handle_depth_pct']:.1f}%\n"
        f"Breakout Price: ${pattern['breakout_price']:.2f}"
    )
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    # Add breakout label
    ax1.text(0.98, 0.02, 'Breakout', transform=ax1.transAxes, fontsize=14,
             color='green', weight='bold', ha='right', rotation=45)
    
    plt.xticks(rotation=45, ha='right')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Chart saved to {save_path}")
        plt.close()
    else:
        plt.show()


def generate_visualizations(stock_data, patterns, output_dir, max_plots=10):
    """
    Generates visualizations for multiple detected patterns.
    
    Args:
        stock_data: Dictionary {ticker: DataFrame} with historical data
        patterns: List of detected patterns
        output_dir: Directory to save images
        max_plots: Maximum number of charts to generate
    """
    import os
    
    if not patterns:
        print("⚠️  No patterns to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort patterns by confidence
    sorted_patterns = sorted(patterns, key=lambda x: x['confidence_score'], reverse=True)
    
    # Limit to maximum number
    patterns_to_plot = sorted_patterns[:max_plots]
    
    print(f"\n📊 Generating visualizations ({len(patterns_to_plot)} patterns)...")
    
    for i, pattern in enumerate(patterns_to_plot, 1):
        ticker = pattern['ticker']
        
        if ticker not in stock_data:
            print(f"⚠️  No data for {ticker}")
            continue
        
        df = stock_data[ticker]
        
        # Filename
        date_str = pd.to_datetime(pattern['breakout_date']).strftime('%Y%m%d')
        filename = f"{ticker}_{date_str}_cup_and_handle.png"
        save_path = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(patterns_to_plot)}] Plotting {ticker}...", end=' ')
        
        try:
            plot_cup_and_handle(df, pattern, save_path)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Visualizations completed in {output_dir}")


def create_summary_plot(patterns, save_path=None):
    """
    Creates a summary chart with pattern statistics.
    
    Args:
        patterns: List of patterns
        save_path: Path to save (None to display)
    """
    if not patterns:
        print("No patterns to summarize")
        return
    
    df = pd.DataFrame(patterns)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cup and Handle Pattern Detection Summary', fontsize=16, weight='bold')
    
    # 1. Cup depth distribution
    axes[0, 0].hist(df['cup_depth_pct'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cup Depth (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution: Cup Depth')
    axes[0, 0].axvline(df['cup_depth_pct'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["cup_depth_pct"].mean():.1f}%')
    axes[0, 0].legend()
    
    # 2. Handle depth distribution
    axes[0, 1].hist(df['handle_depth_pct'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Handle Depth (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution: Handle Depth')
    axes[0, 1].axvline(df['handle_depth_pct'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["handle_depth_pct"].mean():.1f}%')
    axes[0, 1].legend()
    
    # 3. Confidence score distribution
    axes[1, 0].hist(df['confidence_score'], bins=10, color='green', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution: Detection Confidence')
    axes[1, 0].axvline(df['confidence_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["confidence_score"].mean():.2f}')
    axes[1, 0].legend()
    
    # 4. Top 10 stocks with most patterns
    ticker_counts = df['ticker'].value_counts().head(10)
    axes[1, 1].barh(ticker_counts.index, ticker_counts.values, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Number of Patterns')
    axes[1, 1].set_ylabel('Ticker')
    axes[1, 1].set_title('Top 10 Stocks with Most Patterns')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Summary saved to {save_path}")
        plt.close()
    else:
        plt.show()
