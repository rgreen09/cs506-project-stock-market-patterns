# Cup and Handle Pattern Detector

Detector algorítmico del patrón **Cup and Handle** (Taza con Asa) para acciones del S&P 500.

## 📋 Descripción

Este módulo implementa un sistema de etiquetado algorítmico para identificar el patrón técnico "Cup and Handle" en datos históricos de acciones. El patrón consiste en:

1. **Cup (Taza)**: Una formación en forma de U que representa una consolidación
2. **Handle (Asa)**: Una pequeña corrección descendente después de la taza
3. **Breakout**: Una ruptura alcista confirmada con volumen

## 🔧 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

### Uso Básico

```bash
# Analizar 50 acciones del S&P 500
python main.py --tickers 50

# Analizar 100 acciones
python main.py --tickers 100

# Especificar archivo de salida personalizado
python main.py --tickers 50 --output ../../data/labeled/my_patterns.csv
```

### Con Visualizaciones

```bash
# Generar gráficos de los mejores patrones detectados
python main.py --tickers 50 --visualize --max-plots 15

# Especificar directorio para visualizaciones
python main.py --tickers 100 --visualize --viz-dir ../../data/my_charts
```

### Opciones Disponibles

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--tickers` | Número de acciones a analizar | 50 |
| `--output` | Ruta del archivo CSV de salida | `../../data/labeled/cup_and_handle_labels.csv` |
| `--visualize` | Generar gráficos de los patrones | False |
| `--max-plots` | Número máximo de gráficos | 10 |
| `--viz-dir` | Directorio para visualizaciones | `../../data/visualizations` |
| `--period` | Período de datos históricos | 10y |

## 📊 Formato de Salida

El script genera un archivo CSV con las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| `ticker` | Símbolo de la acción |
| `pattern_start_date` | Inicio del patrón completo |
| `pattern_end_date` | Fecha del breakout |
| `cup_start_date` | Inicio de la taza (primer pico) |
| `cup_end_date` | Fin de la taza (segundo pico) |
| `handle_start_date` | Inicio del asa |
| `handle_end_date` | Fin del asa (mínimo del asa) |
| `breakout_date` | Fecha de confirmación del breakout |
| `cup_depth_pct` | Profundidad de la taza (%) |
| `handle_depth_pct` | Profundidad del asa (%) |
| `breakout_price` | Precio en el breakout |
| `confidence_score` | Score de confianza (0-1) |

### Ejemplo de Salida

```csv
ticker,pattern_start_date,pattern_end_date,cup_start_date,cup_end_date,handle_start_date,handle_end_date,breakout_date,cup_depth_pct,handle_depth_pct,breakout_price,confidence_score
AAPL,2020-03-15,2020-05-10,2020-03-15,2020-04-20,2020-04-21,2020-05-05,2020-05-10,28.5,8.2,305.50,0.92
MSFT,2020-02-20,2020-04-15,2020-02-20,2020-03-25,2020-03-26,2020-04-10,2020-04-15,25.3,6.5,175.80,0.88
```

## 🎯 Reglas de Detección

### Parámetros de la Taza

- **Duración**: 7-65 días
- **Profundidad**: 12-33% desde el pico inicial
- **Forma**: Debe ser redondeada (no una V pronunciada)
- **Picos**: Los dos picos deben ser similares (±5%)

### Parámetros del Asa

- **Duración**: 5-20 días
- **Profundidad**: Máximo 15% desde el segundo pico
- **Posición**: Debe formarse en la mitad superior de la taza

### Confirmación de Breakout

- Precio cierra por encima del nivel de resistencia (+1%)
- Volumen en el breakout > 1.2x el promedio de 20 días

## 📁 Estructura de Archivos

```
cup_and_handle/
├── main.py           # Script principal ejecutable
├── detector.py       # Lógica de detección del patrón
├── data_fetcher.py   # Obtención de datos con yfinance
├── utils.py          # Funciones auxiliares
├── visualize.py      # Generación de gráficos
├── requirements.txt  # Dependencias
└── README.md         # Esta documentación
```

## 🔬 Algoritmo Técnico

1. **Detección de Extremos**: Usa `scipy.signal.argrelextrema` para encontrar picos y valles locales
2. **Validación de Taza**: Verifica duración, profundidad, forma redondeada y similitud de picos
3. **Identificación de Asa**: Busca consolidación descendente después del segundo pico
4. **Confirmación**: Verifica breakout con volumen superior al promedio
5. **Score de Confianza**: Calcula confianza basada en qué tan bien se ajusta a parámetros ideales

## 📈 Visualizaciones

Si se activa la opción `--visualize`, se generan:

1. **Gráficos individuales**: Candlestick charts con anotaciones de cada fase del patrón
2. **Gráfico de resumen**: Estadísticas agregadas de todos los patrones detectados

## 🧪 Ejemplo de Ejecución Completa

```bash
cd labeling/cup_and_handle

# Ejecutar análisis completo con visualizaciones
python main.py --tickers 100 --visualize --max-plots 20

# El script generará:
# - ../../data/labeled/cup_and_handle_labels.csv
# - ../../data/visualizations/*.png (gráficos individuales)
# - ../../data/visualizations/summary_statistics.png
```

## 📝 Notas

- El script maneja automáticamente errores de descarga de datos
- Las acciones sin datos o con errores se omiten sin detener la ejecución
- El tiempo de ejecución depende del número de acciones (aprox. 0.5s por acción)
- Para grandes cantidades de datos, considerar ejecutar en lotes

## 👥 Autor

Desarrollado como parte del proyecto CS506 - Stock Market Pattern Recognition

## 📄 Licencia

Este código es parte de un proyecto académico para la asignatura CS506.

