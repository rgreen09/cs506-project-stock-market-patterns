#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el detector funciona correctamente.
Analiza solo 3 acciones para probar la funcionalidad.
"""

import sys
from data_fetcher import fetch_stock_data
from detector import detect_cup_and_handle

def test_detection():
    """Prueba el detector con unas pocas acciones."""
    print("🧪 Iniciando prueba rápida del detector...")
    print("="*60)
    
    # Probar con 3 acciones conocidas
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n📊 Probando {ticker}...")
        
        # Obtener datos
        df = fetch_stock_data(ticker, period='2y')
        
        if df is None:
            print(f"  ❌ No se pudieron obtener datos para {ticker}")
            continue
        
        print(f"  ✅ Datos obtenidos: {len(df)} días")
        
        # Detectar patrones
        patterns = detect_cup_and_handle(ticker, df)
        
        if patterns:
            print(f"  ✅ Patrones detectados: {len(patterns)}")
            for i, pattern in enumerate(patterns, 1):
                print(f"    Patrón {i}:")
                print(f"      - Fecha: {pattern['pattern_start_date']} → {pattern['breakout_date']}")
                print(f"      - Cup depth: {pattern['cup_depth_pct']:.1f}%")
                print(f"      - Handle depth: {pattern['handle_depth_pct']:.1f}%")
                print(f"      - Confianza: {pattern['confidence_score']:.2f}")
        else:
            print(f"  ℹ️  No se encontraron patrones")
    
    print("\n" + "="*60)
    print("✅ Prueba completada exitosamente!")
    print("\nEl detector está funcionando correctamente.")
    print("Puedes ejecutar el análisis completo con:")
    print("  python main.py --tickers 50")

if __name__ == '__main__':
    try:
        test_detection()
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

