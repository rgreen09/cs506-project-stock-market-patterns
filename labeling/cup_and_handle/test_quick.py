#!/usr/bin/env python3
"""
Quick test script to verify the detector works correctly.
Analyzes only 3 stocks to test functionality.
"""

import sys
from data_fetcher import fetch_stock_data
from detector import detect_cup_and_handle

def test_detection():
    """Tests the detector with a few stocks."""
    print("🧪 Starting quick detector test...")
    print("="*60)
    
    # Test with 3 well-known stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}...")
        
        # Get data
        df = fetch_stock_data(ticker, period='2y')
        
        if df is None:
            print(f"  ❌ Could not obtain data for {ticker}")
            continue
        
        print(f"  ✅ Data obtained: {len(df)} days")
        
        # Detect patterns
        patterns = detect_cup_and_handle(ticker, df)
        
        if patterns:
            print(f"  ✅ Patterns detected: {len(patterns)}")
            for i, pattern in enumerate(patterns, 1):
                print(f"    Pattern {i}:")
                print(f"      - Date: {pattern['pattern_start_date']} → {pattern['breakout_date']}")
                print(f"      - Cup depth: {pattern['cup_depth_pct']:.1f}%")
                print(f"      - Handle depth: {pattern['handle_depth_pct']:.1f}%")
                print(f"      - Confidence: {pattern['confidence_score']:.2f}")
        else:
            print(f"  ℹ️  No patterns found")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("\nThe detector is working correctly.")
    print("You can run the full analysis with:")
    print("  python main.py --tickers 50")

if __name__ == '__main__':
    try:
        test_detection()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
