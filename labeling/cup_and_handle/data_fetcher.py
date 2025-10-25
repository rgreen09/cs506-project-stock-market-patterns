"""
Módulo para obtener datos históricos de acciones usando yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time


def get_sp500_tickers(limit=None):
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    
    Args:
        limit: Número máximo de tickers a devolver (None para todos)
        
    Returns:
        Lista de tickers
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Limpiar tickers (algunos tienen caracteres especiales)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        if limit:
            tickers = tickers[:limit]
            
        return tickers
    except Exception as e:
        print(f"Error obteniendo tickers del S&P 500: {e}")
        # Fallback: lista estática de tickers populares
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
    Descarga datos históricos de una acción.
    
    Args:
        ticker: Símbolo de la acción
        start_date: Fecha inicial (formato YYYY-MM-DD) o None
        end_date: Fecha final (formato YYYY-MM-DD) o None
        period: Período de tiempo si no se especifican fechas ('10y', '5y', etc.)
        
    Returns:
        DataFrame con columnas: Date, Open, High, Low, Close, Volume
        o None si hay error
    """
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            print(f"⚠️  No hay datos para {ticker}")
            return None
        
        # Resetear índice para tener Date como columna
        df.reset_index(inplace=True)
        
        # Seleccionar solo las columnas necesarias
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
        
    except Exception as e:
        print(f"❌ Error descargando {ticker}: {e}")
        return None


def fetch_multiple_stocks(tickers, start_date=None, end_date=None, period='10y', delay=0.5):
    """
    Descarga datos para múltiples acciones con manejo de errores.
    
    Args:
        tickers: Lista de símbolos
        start_date: Fecha inicial
        end_date: Fecha final
        period: Período si no se especifican fechas
        delay: Segundos de espera entre peticiones
        
    Returns:
        Diccionario {ticker: DataFrame}
    """
    data_dict = {}
    total = len(tickers)
    
    print(f"📊 Descargando datos de {total} acciones...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Procesando {ticker}...", end=' ')
        
        df = fetch_stock_data(ticker, start_date, end_date, period)
        
        if df is not None:
            data_dict[ticker] = df
            print("✓")
        else:
            print("✗")
        
        # Pequeña pausa para no saturar la API
        if i < total:
            time.sleep(delay)
    
    print(f"\n✅ Datos obtenidos para {len(data_dict)}/{total} acciones")
    return data_dict


def get_date_range(years_back=10):
    """
    Calcula el rango de fechas para descargar datos.
    
    Args:
        years_back: Años hacia atrás desde hoy
        
    Returns:
        Tupla (start_date, end_date) en formato YYYY-MM-DD
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

