# 📈 CS506 Project: Stock Market Pattern Recognition

## 1. Project Overview

This project addresses a core challenge in technical analysis: the **subjective and time-intensive nature of recognizing chart patterns**. Traders often rely on manual inspection to identify formations that may indicate either a continuation or a reversal of the current price trend. This process is not only inefficient but also highly inconsistent between analysts.

Our solution is to build a **machine learning model** that automatically detects these patterns from stock market data. By combining programmatic labeling with supervised learning, the model provides a scalable, consistent, and objective tool for technical analysis.

Importantly, this system does not produce direct trading signals. Instead, it generates **flags** or **probabilistic scores** indicating whether a specific chart formation is present. These outputs can then be integrated into larger algorithmic trading frameworks or used as decision-support tools by traders.

---

## 2. Goals and Objectives

The **primary goal** of this project is to create a machine learning model that can accurately identify key stock chart patterns from historical OHLCV data.

To achieve this, we set the following objectives:

1. **Algorithmic Labeling**: Develop a rule-based system to automatically identify and label patterns within raw price data. This step provides the labeled dataset necessary for model training.
2. **Model Development**: Train and evaluate a predictive model capable of detecting the selected chart patterns with high accuracy.
3. **Pattern Recognition in Practice**: Ensure the final model can analyze new, unseen data and output the detected pattern(s).
4. **Proof of Concept**: Demonstrate how detected patterns could inform a simple, rule-based trading strategy using historical backtesting.

---

## 3. Data Collection and Preparation

### Data Requirements

The project relies on **historical stock market data** with the following attributes:

- **Open, High, Low, Close, and Volume (OHLCV)** values  
- Coverage across multiple equities  
- At least **five years** of data to ensure enough examples of the target patterns  

### Data Sources

Data will be collected programmatically from reliable APIs, such as:

- [Alpha Vantage](https://www.alphavantage.co/)  
- [Finnhub](https://finnhub.io/)  
- [`yfinance`](https://pypi.org/project/yfinance/) (Python library for Yahoo Finance data)  

### Algorithmic Labeling

One of the most important parts of this project is **algorithmic labeling**. Since supervised machine learning requires large labeled datasets, we need a way to label chart patterns **without manual annotation**. Our approach is to design a **rule-based labeling pipeline** that systematically scans through OHLCV data and identifies instances of specific patterns based on well-defined criteria.

The process works as follows:

1. **Identify Local Extrema**: The algorithm first detects local maxima (peaks) and minima (troughs) within the price series.
2. **Apply Pattern-Specific Rules**: Each pattern has strict rules for recognition (e.g., relative height of peaks, separation distance, breakout confirmation).
3. **Assign Labels**: When a formation matches all conditions for a pattern, the algorithm labels that sequence accordingly.
4. **Validate and Store**: Confirmed patterns are stored in the dataset along with metadata, such as time range and stock symbol.

This systematic method ensures that the model is trained on data labeled consistently and objectively.

### Example: Double Top (Bearish Reversal)

Below is an illustration of how the algorithmic labeling process would detect a **Double Top** pattern:
```
a) Find Peaks: Identify two recent local maxima of similar height.
The peaks must be separated by a significant trough (a local minimum).

b) Check Peak Heights: Verify that the two peaks are within a small
threshold of each other (e.g., ±2%).

c) Define the Neckline: Mark the trough between the peaks as the neckline.

d) Confirmation: Confirm the Double Top only if the price falls
decisively below the neckline after the second peak.
```

## 4. Modeling the Data
We will experiment with a progression of models, starting with simple baselines and moving to more advanced methods. The goal is to evaluate which type of model best captures the structural properties of chart patterns.
1.**Baseline Models (for interpretability and quick benchmarking):
Logistic Regression using a small set of engineered features (e.g., relative peak heights, neckline slope, moving average slopes).
Decision Trees / Random Forests to capture non-linear relationships in features without requiring deep learning.
2.**Tree-Based Gradient Boosting Models:
XGBoost or LightGBM trained on engineered features such as returns, volume ratios, and structural characteristics of patterns.
These models are powerful for tabular data, handle imbalanced classes well, and provide feature importance for interpretability.
3.**Deep Learning Approaches:
1D CNNs / LSTMs on OHLCV time-series windows to directly learn sequential dependencies and temporal structure in the data.
(Optional) 2D CNNs on candlestick chart images, which allows the model to learn visual representations similar to how human traders interpret charts.

### Model Setup:
We will train models as binary classifiers for each pattern type
Alternatively, we may extend to a multi-label classification setup, since windows could contain overlapping patterns.
1. **Handling Class Imbalance:
Use class weights, oversampling techniques, or focal loss (for deep learning models).
Emphasize precision and recall rather than raw accuracy.
2.**Evaluation:
Compare baseline, boosting, and deep learning models using precision, recall, F1 score, and PR-AUC.
Select the best-performing approach for final deployment.


   
