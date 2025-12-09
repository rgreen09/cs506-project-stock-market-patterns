"""
Module for fetching historical stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time


def get_sp500_tickers(limit=None):
    """
    Gets the list of S&P 500 tickers from Wikipedia.
    
    Args:
        limit: Maximum number of tickers to return (None for all)
        
    Returns:
        List of tickers
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers (some have special characters)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        if limit:
            tickers = tickers[:limit]
            
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback: static list of popular tickers
        fallback = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'BAC', 'XOM',
            'ABBV', 'PFE', 'COST', 'TMO', 'AVGO', 'KO', 'CSCO', 'MRK', 'PEP',
            'ACN', 'WMT', 'ADBE', 'NKE', 'LLY', 'CVX', 'NFLX', 'ABT', 'DHR',
            'VZ', 'CMCSA', 'INTC', 'CRM', 'TXN', 'NEE', 'PM', 'AMD', 'UNP',
            'BMY', 'QCOM', 'HON', 'RTX', 'UPS'
        ]
        if limit:
            return fallback[:limit]
        return fallback


def fetch_stock_data(ticker, start_date=None, end_date=None, period='10y'):
    """
    Downloads historical data for a stock.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD format) or None
        end_date: End date (YYYY-MM-DD format) or None
        period: Time period if dates not specified ('10y', '5y', etc.)
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            print(f"⚠️  No data available for {ticker}")
            return None
        
        # Reset index to have Date as a column
        df.reset_index(inplace=True)
        
        # Select only necessary columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None


def fetch_multiple_stocks(tickers, start_date=None, end_date=None, period='10y', delay=0.5):
    """
    Downloads data for multiple stocks with error handling.
    
    Args:
        tickers: List of symbols
        start_date: Start date
        end_date: End date
        period: Period if dates not specified
        delay: Seconds to wait between requests
        
    Returns:
        Dictionary {ticker: DataFrame}
    """
    data_dict = {}
    total = len(tickers)
    
    print(f"📊 Downloading data for {total} stocks...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Processing {ticker}...", end=' ')
        
        df = fetch_stock_data(ticker, start_date, end_date, period)
        
        if df is not None:
            data_dict[ticker] = df
            print("✓")
        else:
            print("✗")
        
        # Small pause to avoid saturating the API
        if i < total:
            time.sleep(delay)
    
    print(f"\n✅ Data obtained for {len(data_dict)}/{total} stocks")
    return data_dict


def get_date_range(years_back=10):
    """
    Calculates the date range for downloading data.
    
    Args:
        years_back: Years back from today
        
    Returns:
        Tuple (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
