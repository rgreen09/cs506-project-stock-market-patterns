#!/usr/bin/env python3
"""
Script principal para detectar patrones Cup and Handle en acciones del S&P 500.

Uso:
    python main.py --tickers 50 --output ../../data/labeled/cup_and_handle_labels.csv
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
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Detecta patrones Cup and Handle en acciones del S&P 500'
    )
    
    parser.add_argument(
        '--tickers',
        type=int,
        default=50,
        help='Número de acciones del S&P 500 a analizar (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/labeled/cup_and_handle_labels.csv',
        help='Ruta del archivo CSV de salida'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generar visualizaciones de los patrones detectados'
    )
    
    parser.add_argument(
        '--max-plots',
        type=int,
        default=10,
        help='Número máximo de gráficos a generar (default: 10)'
    )
    
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='../../data/visualizations',
        help='Directorio para guardar visualizaciones'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='10y',
        help='Período de datos históricos (default: 10y)'
    )
    
    return parser.parse_args()


def save_patterns_to_csv(patterns, output_path):
    """
    Guarda los patrones detectados en un archivo CSV.
    
    Args:
        patterns: Lista de patrones detectados
        output_path: Ruta del archivo de salida
    """
    if not patterns:
        print("⚠️  No hay patrones para guardar")
        return
    
    df = pd.DataFrame(patterns)
    
    # Ordenar columnas en el orden especificado
    column_order = [
        'ticker', 'pattern_start_date', 'pattern_end_date',
        'cup_start_date', 'cup_end_date', 
        'handle_start_date', 'handle_end_date',
        'breakout_date', 'cup_depth_pct', 'handle_depth_pct',
        'breakout_price', 'confidence_score'
    ]
    
    df = df[column_order]
    
    # Formatear fechas
    date_columns = [col for col in df.columns if 'date' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
    
    # Formatear números
    df['cup_depth_pct'] = df['cup_depth_pct'].round(2)
    df['handle_depth_pct'] = df['handle_depth_pct'].round(2)
    df['breakout_price'] = df['breakout_price'].round(2)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar
    df.to_csv(output_path, index=False)
    print(f"\n✅ Patrones guardados en: {output_path}")
    print(f"   Total de patrones: {len(df)}")


def print_summary(patterns):
    """Imprime un resumen de los patrones detectados."""
    if not patterns:
        print("\n❌ No se detectaron patrones Cup and Handle")
        return
    
    df = pd.DataFrame(patterns)
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE DETECCIÓN")
    print("="*70)
    print(f"Total de patrones detectados: {len(df)}")
    print(f"Número de acciones con patrones: {df['ticker'].nunique()}")
    print(f"\nEstadísticas de Profundidad de Taza:")
    print(f"  Media: {df['cup_depth_pct'].mean():.2f}%")
    print(f"  Mediana: {df['cup_depth_pct'].median():.2f}%")
    print(f"  Rango: {df['cup_depth_pct'].min():.2f}% - {df['cup_depth_pct'].max():.2f}%")
    print(f"\nEstadísticas de Profundidad de Asa:")
    print(f"  Media: {df['handle_depth_pct'].mean():.2f}%")
    print(f"  Mediana: {df['handle_depth_pct'].median():.2f}%")
    print(f"  Rango: {df['handle_depth_pct'].min():.2f}% - {df['handle_depth_pct'].max():.2f}%")
    print(f"\nScore de Confianza:")
    print(f"  Media: {df['confidence_score'].mean():.2f}")
    print(f"  Patrones de alta confianza (>0.7): {(df['confidence_score'] > 0.7).sum()}")
    
    print(f"\nTop 5 acciones con más patrones:")
    top_tickers = df['ticker'].value_counts().head(5)
    for ticker, count in top_tickers.items():
        print(f"  {ticker}: {count} patrones")
    
    print("="*70 + "\n")


def main():
    """Función principal."""
    args = parse_arguments()
    
    print("="*70)
    print("🔍 DETECTOR DE PATRONES CUP AND HANDLE")
    print("="*70)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Número de acciones a analizar: {args.tickers}")
    print(f"Período de datos: {args.period}")
    print("="*70 + "\n")
    
    # 1. Obtener lista de tickers
    print("📋 Obteniendo lista de tickers del S&P 500...")
    tickers = get_sp500_tickers(limit=args.tickers)
    print(f"✅ Se analizarán {len(tickers)} acciones\n")
    
    # 2. Descargar datos históricos
    stock_data = fetch_multiple_stocks(tickers, period=args.period)
    
    if not stock_data:
        print("❌ Error: No se pudieron obtener datos de ninguna acción")
        sys.exit(1)
    
    print(f"\n✅ Datos descargados para {len(stock_data)} acciones")
    
    # 3. Detectar patrones en cada acción
    print("\n🔎 Detectando patrones Cup and Handle...")
    all_patterns = []
    
    total = len(stock_data)
    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        print(f"[{i}/{total}] Analizando {ticker}...", end=' ')
        
        try:
            patterns = detect_cup_and_handle(ticker, df)
            
            if patterns:
                all_patterns.extend(patterns)
                print(f"✓ ({len(patterns)} patrones)")
            else:
                print("✓ (0 patrones)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # 4. Mostrar resumen
    print_summary(all_patterns)
    
    # 5. Guardar resultados
    save_patterns_to_csv(all_patterns, args.output)
    
    # 6. Generar visualizaciones (opcional)
    if args.visualize and all_patterns:
        print("\n📈 Generando visualizaciones...")
        generate_visualizations(
            stock_data, 
            all_patterns, 
            args.viz_dir, 
            max_plots=args.max_plots
        )
        
        # Crear gráfico de resumen
        summary_path = os.path.join(args.viz_dir, 'summary_statistics.png')
        create_summary_plot(all_patterns, save_path=summary_path)
    
    print("\n✅ Proceso completado exitosamente!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

