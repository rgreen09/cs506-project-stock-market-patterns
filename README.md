# CS506 Project: Stock Market Pattern Recognition

## 1. Description of the Project

This project aims to address a fundamental challenge in technical analysis: the **labor-intensive and subjective nature of identifying chart patterns**. By developing a machine learning model, we will create an automated, scalable, and objective solution for recognizing major price formations in stock data. 

### Core Function
The model's core function is to analyze historical and real-time stock data to recognize specific chart formations that have historically signaled either:
- A **continuation** of the current price trend, or
- A **reversal** of the current price trend

### Target Patterns
The project will focus on identifying a core set of reliable and common chart patterns, including:

**Reversal Patterns:**
- Head and Shoulders
- Double Top/Bottom

**Continuation Patterns:**
- Triangles
- Flags/Pennants

### Value Proposition
This automated approach provides a powerful and efficient resource that can be integrated into a larger algorithmic trading framework, democratizing access to sophisticated technical analysis. The output of this model will not be a direct trading signal but a clear indication of a recognized pattern's presence, which can then be used to inform subsequent trading decisions.

---

## 2. Clear Goals

The **primary goal** is to develop a machine learning model that can accurately identify major chart patterns from historical stock data using an algorithmic labeling method.

### Specific Objectives

**Algorithmic Labeling:** Develop a program to algorithmically identify and label chart patterns, creating a large, labeled dataset for model training and validation.

**Model Development:** Design, train, and test a predictive model capable of recognizing the defined set of chart patterns with a high degree of accuracy.

**Pattern Identification:** Create a final model that can take new stock data and output a clear indication of which, if any, of the target patterns are present.

**Proof of Concept:** Demonstrate how the model's pattern identification can be used to inform a simple, rule-based trading strategy through backtesting.

---

## 3. Data Collection and Preparation

### Data Requirements
The data required is comprehensive, historical stock data consisting of time-series records of **Open, High, Low, Close, and Volume (OHLCV)** values.

### Data Sources
The data will be collected programmatically from a reliable financial data API, such as:
- Alpha Vantage
- Finnhub
- Python library like yfinance

This will gather a diverse set of stocks over a period of **at least five years**.

### Processing Pipeline
Once the raw data is collected, a custom Python script will be used to perform algorithmic labeling. This script will systematically analyze the OHLCV data for each stock, applying a series of rules to find key price points and identify instances of the target chart patterns.

### Algorithmic Labeling Examples

To demonstrate how the labeling script would work, here is an example of the rules it would follow:

#### Double Top Pattern

The script would look for a bearish reversal pattern using the following logic:

```
1. Find Peaks: Identify two distinct, recent peaks (local maxima) that are of similar height. 
   The peaks must be separated by a significant trough (a local minimum).

2. Check Peak Heights: The two peaks must be within a predefined percentage (e.g., 2%) 
   of each other's height.

3. Identify the Neckline: The neckline is defined as the lowest point (the trough) 
   between the two peaks.

4. Confirmation: The algorithm will only label the pattern as a Double Top once the price 
   breaks decisively below the neckline, confirming the reversal.
```

