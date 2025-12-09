#!/usr/bin/env python3
"""
Main script to detect Cup and Handle patterns in S&P 500 stocks.

Usage:
    python main.py --tickers 50 --output ../../../data/labeled/cup_and_handle_labels.csv
    python main.py --tickers 100 --visualize --max-plots 15
"""

import argparse
import pandas as pd
import os
import sys
from datetime import datetime

from data_fetcher import get_sp500_tickers, fetch_multiple_stocks
from detector import detect_cup_and_handle
from visualize import generate_visualizations, create_summary_plot


def parse_arguments():
    """Processes command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect Cup and Handle patterns in S&P 500 stocks'
    )
    
    parser.add_argument(
        '--tickers',
        type=int,
        default=50,
        help='Number of S&P 500 stocks to analyze (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../../../data/labeled/cup_and_handle_labels.csv',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations of detected patterns'
    )
    
    parser.add_argument(
        '--max-plots',
        type=int,
        default=10,
        help='Maximum number of charts to generate (default: 10)'
    )
    
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='../../../data/visualizations',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='10y',
        help='Historical data period (default: 10y)'
    )
    
    return parser.parse_args()


def save_patterns_to_csv(patterns, output_path):
    """
    Saves detected patterns to a CSV file.
    
    Args:
        patterns: List of detected patterns
        output_path: Output file path
    """
    if not patterns:
        print("⚠️  No patterns to save")
        return
    
    df = pd.DataFrame(patterns)
    
    # Sort columns in specified order
    column_order = [
        'ticker', 'pattern_start_date', 'pattern_end_date',
        'cup_start_date', 'cup_end_date', 
        'handle_start_date', 'handle_end_date',
        'breakout_date', 'cup_depth_pct', 'handle_depth_pct',
        'breakout_price', 'confidence_score'
    ]
    
    df = df[column_order]
    
    # Format dates
    date_columns = [col for col in df.columns if 'date' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
    
    # Format numbers
    df['cup_depth_pct'] = df['cup_depth_pct'].round(2)
    df['handle_depth_pct'] = df['handle_depth_pct'].round(2)
    df['breakout_price'] = df['breakout_price'].round(2)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Patterns saved to: {output_path}")
    print(f"   Total patterns: {len(df)}")


def print_summary(patterns):
    """Prints a summary of detected patterns."""
    if not patterns:
        print("\n❌ No Cup and Handle patterns detected")
        return
    
    df = pd.DataFrame(patterns)
    
    print("\n" + "="*70)
    print("📊 DETECTION SUMMARY")
    print("="*70)
    print(f"Total patterns detected: {len(df)}")
    print(f"Number of stocks with patterns: {df['ticker'].nunique()}")
    print(f"\nCup Depth Statistics:")
    print(f"  Mean: {df['cup_depth_pct'].mean():.2f}%")
    print(f"  Median: {df['cup_depth_pct'].median():.2f}%")
    print(f"  Range: {df['cup_depth_pct'].min():.2f}% - {df['cup_depth_pct'].max():.2f}%")
    print(f"\nHandle Depth Statistics:")
    print(f"  Mean: {df['handle_depth_pct'].mean():.2f}%")
    print(f"  Median: {df['handle_depth_pct'].median():.2f}%")
    print(f"  Range: {df['handle_depth_pct'].min():.2f}% - {df['handle_depth_pct'].max():.2f}%")
    print(f"\nConfidence Score:")
    print(f"  Mean: {df['confidence_score'].mean():.2f}")
    print(f"  High confidence patterns (>0.7): {(df['confidence_score'] > 0.7).sum()}")
    
    print(f"\nTop 5 stocks with most patterns:")
    top_tickers = df['ticker'].value_counts().head(5)
    for ticker, count in top_tickers.items():
        print(f"  {ticker}: {count} patterns")
    
    print("="*70 + "\n")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*70)
    print("🔍 CUP AND HANDLE PATTERN DETECTOR")
    print("="*70)
    print(f"Execution date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of stocks to analyze: {args.tickers}")
    print(f"Data period: {args.period}")
    print("="*70 + "\n")
    
    # 1. Get ticker list
    print("📋 Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers(limit=args.tickers)
    print(f"✅ Will analyze {len(tickers)} stocks\n")
    
    # 2. Download historical data
    stock_data = fetch_multiple_stocks(tickers, period=args.period)
    
    if not stock_data:
        print("❌ Error: Could not obtain data for any stock")
        sys.exit(1)
    
    print(f"\n✅ Data downloaded for {len(stock_data)} stocks")
    
    # 3. Detect patterns in each stock
    print("\n🔎 Detecting Cup and Handle patterns...")
    all_patterns = []
    
    total = len(stock_data)
    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        print(f"[{i}/{total}] Analyzing {ticker}...", end=' ')
        
        try:
            patterns = detect_cup_and_handle(ticker, df)
            
            if patterns:
                all_patterns.extend(patterns)
                print(f"✓ ({len(patterns)} patterns)")
            else:
                print("✓ (0 patterns)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # 4. Show summary
    print_summary(all_patterns)
    
    # 5. Save results
    save_patterns_to_csv(all_patterns, args.output)
    
    # 6. Generate visualizations (optional)
    if args.visualize and all_patterns:
        print("\n📈 Generating visualizations...")
        generate_visualizations(
            stock_data, 
            all_patterns, 
            args.viz_dir, 
            max_plots=args.max_plots
        )
        
        # Create summary plot
        summary_path = os.path.join(args.viz_dir, 'summary_statistics.png')
        create_summary_plot(all_patterns, save_path=summary_path)
    
    print("\n✅ Process completed successfully!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
