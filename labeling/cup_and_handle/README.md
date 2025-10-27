# Cup and Handle Pattern Detector

Algorithmic detector for the **Cup and Handle** pattern in S&P 500 stocks.

## 📋 Description

This module implements an algorithmic labeling system to identify the "Cup and Handle" technical pattern in historical stock data. The pattern consists of:

1. **Cup**: A U-shaped formation representing consolidation
2. **Handle**: A small downward correction after the cup
3. **Breakout**: A confirmed upward breakout with volume

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🧪 Quick Test

Before running a full analysis, you can verify the installation with a quick test:

```bash
python test_quick.py
```

This will analyze only 3 stocks (AAPL, MSFT, GOOGL) over 2 years to confirm everything works correctly. Takes ~1-2 minutes.

## 🚀 Usage

### Basic Usage

```bash
# Analyze 50 S&P 500 stocks
python main.py --tickers 50

# Analyze 100 stocks
python main.py --tickers 100

# Specify custom output file
python main.py --tickers 50 --output ../../data/labeled/my_patterns.csv
```

### With Visualizations

```bash
# Generate charts of best detected patterns
python main.py --tickers 50 --visualize --max-plots 15

# Specify visualization directory
python main.py --tickers 100 --visualize --viz-dir ../../data/my_charts
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tickers` | Number of stocks to analyze | 50 |
| `--output` | Output CSV file path | `../../data/labeled/cup_and_handle_labels.csv` |
| `--visualize` | Generate pattern charts | False |
| `--max-plots` | Maximum number of charts | 10 |
| `--viz-dir` | Directory for visualizations | `../../data/visualizations` |
| `--period` | Historical data period | 10y |

## 📊 Output Format

The script generates a CSV file with the following columns:

| Column | Description |
|---------|-------------|
| `ticker` | Stock symbol |
| `pattern_start_date` | Complete pattern start |
| `pattern_end_date` | Breakout date |
| `cup_start_date` | Cup start (first peak) |
| `cup_end_date` | Cup end (second peak) |
| `handle_start_date` | Handle start |
| `handle_end_date` | Handle end (handle low) |
| `breakout_date` | Breakout confirmation date |
| `cup_depth_pct` | Cup depth (%) |
| `handle_depth_pct` | Handle depth (%) |
| `breakout_price` | Breakout price |
| `confidence_score` | Confidence score (0-1) |

### Output Example

```csv
ticker,pattern_start_date,pattern_end_date,cup_start_date,cup_end_date,handle_start_date,handle_end_date,breakout_date,cup_depth_pct,handle_depth_pct,breakout_price,confidence_score
AAPL,2020-03-15,2020-05-10,2020-03-15,2020-04-20,2020-04-21,2020-05-05,2020-05-10,28.5,8.2,305.50,0.92
MSFT,2020-02-20,2020-04-15,2020-02-20,2020-03-25,2020-03-26,2020-04-10,2020-04-15,25.3,6.5,175.80,0.88
```

## 🎯 Detection Rules

### Cup Parameters

- **Duration**: 7-65 days
- **Depth**: 12-33% from initial peak
- **Shape**: Must be rounded (not a sharp V)
- **Peaks**: Two similar peaks (±5%)

### Handle Parameters

- **Duration**: 5-20 days
- **Depth**: Maximum 15% from second peak
- **Position**: Must form in upper half of cup

### Breakout Confirmation

- Price closes above resistance level (+1%)
- Volume on breakout > 1.2x 20-day average

## 📁 File Structure

```
cup_and_handle/
├── main.py           # Main executable script
├── detector.py       # Pattern detection logic
├── data_fetcher.py   # Data acquisition with yfinance
├── utils.py          # Helper functions
├── visualize.py      # Chart generation
├── requirements.txt  # Dependencies
└── README.md         # This documentation
```

## 🔬 Technical Algorithm

1. **Extrema Detection**: Uses `scipy.signal.argrelextrema` to find local peaks and troughs
2. **Cup Validation**: Verifies duration, depth, rounded shape, and peak similarity
3. **Handle Identification**: Searches for downward consolidation after second peak
4. **Confirmation**: Verifies breakout with above-average volume
5. **Confidence Score**: Calculates confidence based on fit to ideal parameters

## 📈 Visualizations

If the `--visualize` option is enabled, the script generates:

1. **Individual charts**: Candlestick charts with annotations for each pattern phase
2. **Summary chart**: Aggregate statistics of all detected patterns

## 📊 Current Dataset

**Generated Data** (already available in repository):

| Metric | Value |
|--------|-------|
| **Output file** | `../../data/labeled/cup_and_handle_labels.csv` |
| **Total patterns** | 178 detected patterns |
| **Stocks analyzed** | 50 S&P 500 companies |
| **Time range** | 2015-2025 (10 years) |
| **Visualizations** | 5 example charts in `../../data/visualizations/` |

**Dataset Statistics**:
- Average confidence score: 0.82
- Cup depth: 12-33% (mean: ~17%)
- Handle depth: 1-15% (mean: ~6%)
- Top stocks: AMD (27 patterns), NFLX (24), NVDA (13)

## 🧪 Complete Execution Example

```bash
cd labeling/cup_and_handle

# Run complete analysis with visualizations
python main.py --tickers 100 --visualize --max-plots 20

# The script will generate:
# - ../../data/labeled/cup_and_handle_labels.csv
# - ../../data/visualizations/*.png (individual charts)
# - ../../data/visualizations/summary_statistics.png
```

## 🔧 Common Issues

**If you get import errors**:
```bash
pip install -r requirements.txt
```

**If no patterns are detected**:
- Try different time period: `--period 5y` or `--period 15y`
- Check internet connection (yfinance requires API access)

**If execution is too slow**:
- Reduce number of tickers: `--tickers 10`
- yfinance has a delay between requests to avoid rate limiting

## ⚙️ Customizing Detection

To modify detection sensitivity, edit parameters in `detector.py` (lines 22-32):

```python
min_cup_depth=12,       # Minimum cup decline %
max_cup_depth=33,       # Maximum cup decline %
min_handle_duration=5,  # Minimum handle days
max_handle_duration=20, # Maximum handle days
extrema_order=5         # Peak detection sensitivity
```

Increase `extrema_order` (e.g., to 7) for smoother peak detection and fewer false positives.

## 📝 Notes

- The script automatically handles data download errors
- Stocks without data or with errors are skipped without stopping execution
- Execution time depends on number of stocks (approx. 0.5s per stock)
- For large datasets, consider running in batches

## 👥 Author

Developed as part of the CS506 project - Stock Market Pattern Recognition

## 📄 License

This code is part of an academic project for CS506 course.
