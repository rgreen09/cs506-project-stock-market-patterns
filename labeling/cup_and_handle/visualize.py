"""
Module for visualizing detected Cup and Handle patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta


def plot_cup_and_handle(df, pattern, save_path=None):
    """
    Generates a candlestick chart showing the detected pattern.
    
    Args:
        df: DataFrame with OHLCV data
        pattern: Dictionary with pattern information
        save_path: Path to save the image (None to display)
    """
    # Extract pattern dates
    pattern_start = pd.to_datetime(pattern['pattern_start_date'])
    pattern_end = pd.to_datetime(pattern['breakout_date'])
    
    # Add margin for visualization
    margin = timedelta(days=10)
    start_date = pattern_start - margin
    end_date = pattern_end + margin
    
    # Filter data for the range
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remove timezone info for comparison if present
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_localize(None)
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    plot_df = df.loc[mask].copy()
    
    if plot_df.empty:
        print(f"⚠️  No data to visualize pattern for {pattern['ticker']}")
        return
    
    # Prepare data for mplfinance
    plot_df.set_index('Date', inplace=True)
    
    # Create markers for pattern phases
    cup_start = pd.to_datetime(pattern['cup_start_date'])
    cup_end = pd.to_datetime(pattern['cup_end_date'])
    handle_start = pd.to_datetime(pattern['handle_start_date'])
    handle_end = pd.to_datetime(pattern['handle_end_date'])
    breakout = pd.to_datetime(pattern['breakout_date'])
    
    # Remove timezone for consistency
    if hasattr(cup_start, 'tz') and cup_start.tz is not None:
        cup_start = cup_start.tz_localize(None)
    if hasattr(cup_end, 'tz') and cup_end.tz is not None:
        cup_end = cup_end.tz_localize(None)
    if hasattr(handle_start, 'tz') and handle_start.tz is not None:
        handle_start = handle_start.tz_localize(None)
    if hasattr(handle_end, 'tz') and handle_end.tz is not None:
        handle_end = handle_end.tz_localize(None)
    if hasattr(breakout, 'tz') and breakout.tz is not None:
        breakout = breakout.tz_localize(None)
    
    # Create annotation lines
    addplot_lines = []
    
    # Horizontal line for resistance level
    resistance_price = pattern['breakout_price'] * 0.99
    resistance_line = [resistance_price] * len(plot_df)
    addplot_lines.append(
        mpf.make_addplot(resistance_line, color='red', linestyle='--', width=1.5)
    )
    
    # Configure style
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
    
    # Create the chart
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=s,
        title=f"{pattern['ticker']} - Cup and Handle (Confidence: {pattern['confidence_score']})",
        ylabel='Price ($)',
        volume=True,
        addplot=addplot_lines if addplot_lines else None,
        returnfig=True,
        figsize=(14, 8)
    )
    
    # Add text annotations
    ax = axes[0]
    
    # Annotate phases
    y_pos = plot_df['High'].max() * 1.05
    
    if cup_start in plot_df.index:
        ax.axvline(x=cup_start, color='blue', linestyle=':', alpha=0.6, linewidth=2)
        ax.text(cup_start, y_pos, 'Cup Start', fontsize=9, color='blue', 
                rotation=45, ha='right')
    
    if cup_end in plot_df.index:
        ax.axvline(x=cup_end, color='purple', linestyle=':', alpha=0.6, linewidth=2)
        ax.text(cup_end, y_pos, 'Handle Start', fontsize=9, color='purple',
                rotation=45, ha='right')
    
    if breakout in plot_df.index:
        ax.axvline(x=breakout, color='green', linestyle='-', alpha=0.8, linewidth=2)
        ax.text(breakout, y_pos, 'Breakout', fontsize=10, color='green',
                rotation=45, ha='right', weight='bold')
    
    # Additional information
    info_text = (
        f"Cup Depth: {pattern['cup_depth_pct']:.1f}%\n"
        f"Handle Depth: {pattern['handle_depth_pct']:.1f}%\n"
        f"Breakout Price: ${pattern['breakout_price']:.2f}"
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
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
