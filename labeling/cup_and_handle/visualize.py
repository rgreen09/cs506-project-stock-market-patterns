"""
Módulo para visualizar patrones Cup and Handle detectados.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta


def plot_cup_and_handle(df, pattern, save_path=None):
    """
    Genera un gráfico de candlestick mostrando el patrón detectado.
    
    Args:
        df: DataFrame con datos OHLCV
        pattern: Diccionario con información del patrón
        save_path: Ruta donde guardar la imagen (None para mostrar)
    """
    # Extraer fechas del patrón
    pattern_start = pd.to_datetime(pattern['pattern_start_date'])
    pattern_end = pd.to_datetime(pattern['breakout_date'])
    
    # Añadir margen para visualización
    margin = timedelta(days=10)
    start_date = pattern_start - margin
    end_date = pattern_end + margin
    
    # Filtrar datos para el rango
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    plot_df = df.loc[mask].copy()
    
    if plot_df.empty:
        print(f"⚠️  No hay datos para visualizar el patrón de {pattern['ticker']}")
        return
    
    # Preparar datos para mplfinance
    plot_df.set_index('Date', inplace=True)
    
    # Crear marcadores para las fases del patrón
    cup_start = pd.to_datetime(pattern['cup_start_date'])
    cup_end = pd.to_datetime(pattern['cup_end_date'])
    handle_start = pd.to_datetime(pattern['handle_start_date'])
    handle_end = pd.to_datetime(pattern['handle_end_date'])
    breakout = pd.to_datetime(pattern['breakout_date'])
    
    # Crear líneas de anotación
    addplot_lines = []
    
    # Línea horizontal para el nivel de resistencia
    resistance_price = pattern['breakout_price'] * 0.99
    resistance_line = [resistance_price] * len(plot_df)
    addplot_lines.append(
        mpf.make_addplot(resistance_line, color='red', linestyle='--', width=1.5)
    )
    
    # Configurar estilo
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
    
    # Crear el gráfico
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=s,
        title=f"{pattern['ticker']} - Cup and Handle (Confianza: {pattern['confidence_score']})",
        ylabel='Precio ($)',
        volume=True,
        addplot=addplot_lines if addplot_lines else None,
        returnfig=True,
        figsize=(14, 8)
    )
    
    # Añadir anotaciones de texto
    ax = axes[0]
    
    # Anotar fases
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
    
    # Información adicional
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
        print(f"✅ Gráfico guardado en {save_path}")
        plt.close()
    else:
        plt.show()


def generate_visualizations(stock_data, patterns, output_dir, max_plots=10):
    """
    Genera visualizaciones para múltiples patrones detectados.
    
    Args:
        stock_data: Diccionario {ticker: DataFrame} con datos históricos
        patterns: Lista de patrones detectados
        output_dir: Directorio donde guardar las imágenes
        max_plots: Número máximo de gráficos a generar
    """
    import os
    
    if not patterns:
        print("⚠️  No hay patrones para visualizar")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Ordenar patrones por confianza
    sorted_patterns = sorted(patterns, key=lambda x: x['confidence_score'], reverse=True)
    
    # Limitar al número máximo
    patterns_to_plot = sorted_patterns[:max_plots]
    
    print(f"\n📊 Generando visualizaciones ({len(patterns_to_plot)} patrones)...")
    
    for i, pattern in enumerate(patterns_to_plot, 1):
        ticker = pattern['ticker']
        
        if ticker not in stock_data:
            print(f"⚠️  No hay datos para {ticker}")
            continue
        
        df = stock_data[ticker]
        
        # Nombre del archivo
        date_str = pd.to_datetime(pattern['breakout_date']).strftime('%Y%m%d')
        filename = f"{ticker}_{date_str}_cup_and_handle.png"
        save_path = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(patterns_to_plot)}] Graficando {ticker}...", end=' ')
        
        try:
            plot_cup_and_handle(df, pattern, save_path)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Visualizaciones completadas en {output_dir}")


def create_summary_plot(patterns, save_path=None):
    """
    Crea un gráfico de resumen con estadísticas de los patrones detectados.
    
    Args:
        patterns: Lista de patrones
        save_path: Ruta para guardar (None para mostrar)
    """
    if not patterns:
        print("No hay patrones para resumir")
        return
    
    df = pd.DataFrame(patterns)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resumen de Patrones Cup and Handle Detectados', fontsize=16, weight='bold')
    
    # 1. Distribución de profundidades de taza
    axes[0, 0].hist(df['cup_depth_pct'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Profundidad de la Taza (%)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución: Profundidad de Taza')
    axes[0, 0].axvline(df['cup_depth_pct'].mean(), color='red', linestyle='--', 
                       label=f'Media: {df["cup_depth_pct"].mean():.1f}%')
    axes[0, 0].legend()
    
    # 2. Distribución de profundidades de asa
    axes[0, 1].hist(df['handle_depth_pct'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Profundidad del Asa (%)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución: Profundidad de Asa')
    axes[0, 1].axvline(df['handle_depth_pct'].mean(), color='red', linestyle='--',
                       label=f'Media: {df["handle_depth_pct"].mean():.1f}%')
    axes[0, 1].legend()
    
    # 3. Distribución de scores de confianza
    axes[1, 0].hist(df['confidence_score'], bins=10, color='green', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Score de Confianza')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución: Confianza de Detección')
    axes[1, 0].axvline(df['confidence_score'].mean(), color='red', linestyle='--',
                       label=f'Media: {df["confidence_score"].mean():.2f}')
    axes[1, 0].legend()
    
    # 4. Top 10 acciones con más patrones
    ticker_counts = df['ticker'].value_counts().head(10)
    axes[1, 1].barh(ticker_counts.index, ticker_counts.values, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Número de Patrones')
    axes[1, 1].set_ylabel('Ticker')
    axes[1, 1].set_title('Top 10 Acciones con Más Patrones')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Resumen guardado en {save_path}")
        plt.close()
    else:
        plt.show()

