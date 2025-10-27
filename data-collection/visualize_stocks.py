"""
Comprehensive stock data visualization script.
Creates multiple types of visualizations from the stock data.
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

def load_stock_data(ticker):
    """Load stock data from CSV file."""
    file_path = DATA_DIR / f"{ticker}_daily_10y.csv"
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def plot_price_trends(selected_stocks=None):
    """Plot close price trends for selected stocks."""
    if selected_stocks is None:
        selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for i, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            axes[0].plot(df.index, df['close'], label=ticker, linewidth=1.5, alpha=0.8)
    
    axes[0].set_title('Close Price Trends - Major Tech Stocks (2015-2025)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Close Price ($)', fontsize=12)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Normalized view (indexed to 100 at start)
    for i, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            normalized = (df['close'] / df['close'].iloc[0]) * 100
            axes[1].plot(df.index, normalized, label=ticker, linewidth=1.5, alpha=0.8)
    
    axes[1].set_title('Normalized Performance (Base = 100)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Normalized Price', fontsize=12)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'price_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'price_trends.png'}")
    plt.close()

def plot_volume_analysis():
    """Plot volume analysis for selected stocks."""
    selected_stocks = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            ax = axes[idx]
            
            # Plot closing price and volume
            ax2 = ax.twinx()
            
            line1 = ax.plot(df.index, df['close'], 'b-', label='Close Price', linewidth=1.5)
            line2 = ax2.bar(df.index, df['volume'] / 1e6, alpha=0.3, color='orange', label='Volume (M)')
            
            ax.set_title(f'{ticker} - Price and Volume', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price ($)', color='b')
            ax2.set_ylabel('Volume (Millions)', color='orange')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            lines = line1 + [line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'volume_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'volume_analysis.png'}")
    plt.close()

def plot_returns_comparison():
    """Plot returns comparison for all stocks."""
    csv_files = glob.glob(str(DATA_DIR / "*.csv"))
    
    returns_data = {}
    for file in csv_files:
        ticker = Path(file).stem.replace('_daily_10y', '')
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate total return
        total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        returns_data[ticker] = total_return
    
    # Sort by returns
    sorted_data = sorted(returns_data.items(), key=lambda x: x[1], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Bar chart
    tickers, returns = zip(*sorted_data)
    colors_bar = ['green' if r > 0 else 'red' for r in returns]
    
    axes[0].barh(range(len(tickers)), returns, color=colors_bar, alpha=0.7)
    axes[0].set_yticks(range(len(tickers)))
    axes[0].set_yticklabels(tickers, fontsize=8)
    axes[0].set_xlabel('Total Return (%)', fontsize=12)
    axes[0].set_title('Stock Returns Over 10 Years (2015-2025)', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Histogram
    axes[1].hist(returns, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Total Return (%)', fontsize=12)
    axes[1].set_ylabel('Number of Stocks', fontsize=12)
    axes[1].set_title('Distribution of Returns', fontsize=14, fontweight='bold')
    axes[1].axvline(x=np.mean(returns), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(returns):.1f}%', linewidth=2)
    axes[1].axvline(x=np.median(returns), color='orange', linestyle='--', 
                    label=f'Median: {np.median(returns):.1f}%', linewidth=2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'returns_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'returns_comparison.png'}")
    plt.close()

def plot_volatility_analysis():
    """Plot volatility analysis."""
    selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    volatility_data = []
    ticker_names = []
    
    for ticker in selected_stocks:
        df = load_stock_data(ticker)
        if df is not None:
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            # Calculate annualized volatility (252 trading days)
            volatility = df['returns'].std() * np.sqrt(252) * 100
            volatility_data.append(volatility)
            ticker_names.append(ticker)
    
    bars = ax.bar(ticker_names, volatility_data, color=plt.cm.viridis(np.linspace(0, 1, len(ticker_names))))
    ax.set_title('Annualized Volatility Comparison (10 Years)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stock Ticker', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, vol) in enumerate(zip(bars, volatility_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{vol:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'volatility_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'volatility_analysis.png'}")
    plt.close()

def plot_candle_volume_multiple():
    """Plot candlestick and volume for multiple stocks."""
    selected_stocks = ['AAPL', 'NVDA', 'TSLA'][:2]
    
    fig, axes = plt.subplots(len(selected_stocks), 1, figsize=(16, 6*len(selected_stocks)))
    if len(selected_stocks) == 1:
        axes = [axes]
    
    for idx, ticker in enumerate(selected_stocks):
        df = load_stock_data(ticker)
        if df is not None:
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            
            # Plot candlesticks (simplified)
            for i in range(0, len(df), 5):  # Sample every 5 days for clarity
                row = df.iloc[i]
                color = 'green' if row['close'] >= row['open'] else 'red'
                ax1.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)
                ax1.plot([i, i], [row['open'], row['close']], color=color, linewidth=2)
            
            ax1.plot(df['close'], 'b-', linewidth=1, alpha=0.7, label='Close Price')
            ax1.set_title(f'{ticker} - Price Chart (Last 10 Years)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Price ($)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot volume
            ax2.bar(range(len(df)), df['volume'] / 1e6, alpha=0.3, color='orange', label='Volume')
            ax2.set_ylabel('Volume (Millions)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'candle_stock_charts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'candle_stock_charts.png'}")
    plt.close()

def plot_sector_comparison():
    """Plot comparison across different sectors."""
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META'],
        'E-commerce': ['AMZN', 'TSLA'],
        'Finance': ['JPM', 'BAC', 'V', 'MA'],
        'Healthcare': ['JNJ', 'LLY', 'ABBV'],
        'Energy': ['XOM'],
        'Retail': ['WMT', 'COST']
    }
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = 0
    color_idx = 0
    
    for sector, tickers in sectors.items():
        positions = []
        values = []
        
        for ticker in tickers:
            df = load_stock_data(ticker)
            if df is not None:
                total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                positions.append(x_pos)
                values.append(total_return)
                x_pos += 1
        
        color = colors[color_idx % len(colors)]
        ax.barh(positions, values, label=sector, color=color, alpha=0.7)
        color_idx += 1
        x_pos += 0.5  # Space between sectors
    
    ax.set_xlabel('Total Return (%)', fontsize=12)
    ax.set_title('Sector Performance Comparison', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sector_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'sector_comparison.png'}")
    plt.close()

def plot_correlation_heatmap():
    """Plot correlation heatmap for selected stocks."""
    selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX']
    
    returns_df = pd.DataFrame()
    
    for ticker in selected_stocks:
        df = load_stock_data(ticker)
        if df is not None:
            df['returns'] = df['close'].pct_change()
            returns_df[ticker] = df['returns']
    
    correlation_matrix = returns_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(len(selected_stocks)):
        for j in range(len(selected_stocks)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(selected_stocks)))
    ax.set_yticks(range(len(selected_stocks)))
    ax.set_xticklabels(selected_stocks, rotation=45, ha='right')
    ax.set_yticklabels(selected_stocks)
    ax.set_title('Stock Returns Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'correlation_heatmap.png'}")
    plt.close()

def main():
    """Generate all visualizations."""
    print("Generating stock data visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Generate all plots
    plot_price_trends()
    plot_volume_analysis()
    plot_returns_comparison()
    plot_volatility_analysis()
    plot_sector_comparison()
    plot_correlation_heatmap()
    # plot_candle_volume_multiple()  # Takes a while to render
    
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()

