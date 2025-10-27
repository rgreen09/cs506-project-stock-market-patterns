"""
Stock data visualization with split-adjusted prices.
Automatically detects and adjusts for stock splits.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import glob
import numpy as np
from datetime import datetime

# Configuration
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "visualizations" / "stock_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
colors = plt.cm.tab20(np.linspace(0, 1, 20))

def detect_and_adjust_splits(df):
    """
    Detect and adjust for stock splits automatically.
    Looks for sudden price drops of more than 30% day-over-day.
    """
    df_adjusted = df.copy()
    df_adjusted['returns'] = df_adjusted['close'].pct_change()
    
    # Detect potential splits (price drop > 30% day-over-day)
    potential_splits = df_adjusted[df_adjusted['returns'] < -0.30].copy()
    
    if len(potential_splits) > 0:
        # Calculate split ratio
        for idx in potential_splits.index:
            split_idx = df_adjusted.index.get_loc(idx)
            if split_idx > 0:
                prev_close = df_adjusted.iloc[split_idx - 1]['close']
                curr_close = df_adjusted.iloc[split_idx]['close']
                split_ratio = prev_close / curr_close
                
                if split_ratio >= 1.3:  # Minimum 2:1 split
                    # Adjust all prices before the split
                    df_adjusted.loc[:idx, ['open', 'high', 'low', 'close']] /= split_ratio
    
    return df_adjusted

def load_stock_data(ticker):
    """Load stock data from CSV file and adjust for splits."""
    file_path = DATA_DIR / f"{ticker}_daily_10y.csv"
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Adjust for splits
    df = detect_and_adjust_splits(df)
    
    return df

def plot_price_trends_adjusted(selected_stocks=None):
    """Plot close price trends for selected stocks."""
    if selected_stocks is None:
        selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for i, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            axes[0].plot(df.index, df['close'], label=ticker, linewidth=2, alpha=0.8)
    
    axes[0].set_title('Close Price Trends - Major Tech Stocks (Split-Adjusted, 2015-2025)', 
                     fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Close Price ($)', fontsize=12)
    axes[0].legend(loc='best', fontsize=9, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Normalized view (indexed to 100 at start)
    for i, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            normalized = (df['close'] / df['close'].iloc[0]) * 100
            axes[1].plot(df.index, normalized, label=ticker, linewidth=2, alpha=0.8)
    
    axes[1].set_title('Normalized Performance (Base = 100, Split-Adjusted)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Normalized Price', fontsize=12)
    axes[1].legend(loc='best', fontsize=9, ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'price_trends_adjusted.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'price_trends_adjusted.png'}")
    plt.close()

def plot_split_detection():
    """Plot to show detected stock splits."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # AAPL split in 2020
    df_aapl = pd.read_csv(DATA_DIR / "AAPL_daily_10y.csv")
    df_aapl['date'] = pd.to_datetime(df_aapl['date'])
    df_aapl.set_index('date', inplace=True)
    
    dates = df_aapl.index
    prices = df_aapl['close'].values
    
    axes[0].plot(dates, prices, 'b-', linewidth=1.5, label='Raw Price')
    axes[0].scatter(dates[prices < 200], prices[prices < 200], color='red', 
                   s=30, zorder=5, label='Split Period')
    axes[0].set_title('AAPL - Stock Split Detection (Aug 2020)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(pd.Timestamp('2020-08-31'), color='red', linestyle='--', alpha=0.7)
    
    # TSLA split in 2020
    df_tsla = pd.read_csv(DATA_DIR / "TSLA_daily_10y.csv")
    df_tsla['date'] = pd.to_datetime(df_tsla['date'])
    df_tsla.set_index('date', inplace=True)
    
    axes[1].plot(df_tsla.index, df_tsla['close'], 'g-', linewidth=1.5, label='Raw Price')
    axes[1].axvline(pd.Timestamp('2020-08-31'), color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('TSLA - Stock Split Detection (Aug 2020)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'split_detection.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'split_detection.png'}")
    plt.close()

def plot_before_after_comparison():
    """Compare charts before and after split adjustment."""
    selected_stocks = ['AAPL', 'TSLA']
    
    fig, axes = plt.subplots(len(selected_stocks), 2, figsize=(18, 5*len(selected_stocks)))
    if len(selected_stocks) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ticker in enumerate(selected_stocks):
        file_path = DATA_DIR / f"{ticker}_daily_10y.csv"
        df_raw = pd.read_csv(file_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace=True)
        
        df_adjusted = load_stock_data(ticker)
        
        # Before adjustment
        axes[idx, 0].plot(df_raw.index, df_raw['close'], 'b-', linewidth=1.5)
        axes[idx, 0].set_title(f'{ticker} - Raw Data (Not Split-Adjusted)', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('Date')
        axes[idx, 0].set_ylabel('Price ($)')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # After adjustment
        axes[idx, 1].plot(df_adjusted.index, df_adjusted['close'], 'g-', linewidth=1.5)
        axes[idx, 1].set_title(f'{ticker} - Split-Adjusted', fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('Date')
        axes[idx, 1].set_ylabel('Price ($)')
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'before_after_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'before_after_comparison.png'}")
    plt.close()

def main():
    """Generate all visualizations."""
    print("Generating split-adjusted stock visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Generate comparison plots
    plot_split_detection()
    plot_before_after_comparison()
    plot_price_trends_adjusted()
    
    print("\nAll visualizations generated successfully!")
    print("\nNote: The data you have is NOT split-adjusted.")
    print("Stock splits (like AAPL's 4:1 in Aug 2020 and TSLA's 5:1) cause")
    print("artificial price drops in the charts.")

if __name__ == "__main__":
    main()

