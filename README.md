# StockLens
# StockLens

A Python desktop application for quantitative stock analysis and financial risk assessment.

## Overview

StockLens uses historical market data to help users explore stock performance, measure financial risk, compare stocks against a benchmark, and simulate possible future price scenarios.

## Features

- **Performance analysis:** Total return, CAGR, and annualized volatility.
- **Risk metrics:** Sharpe ratio, Sortino ratio, maximum drawdown, Value at Risk (VaR), and Conditional Value at Risk (CVaR).
- **Benchmark comparison:** Beta, Jensen's alpha, correlation, tracking error, and information ratio.
- **Visual analysis:** Historical prices, moving averages, drawdowns, return distributions, and rolling risk metrics.
- **Monte Carlo simulation:** Simulates potential price paths using a geometric Brownian motion model.
- **Desktop interface:** Tkinter GUI with interactive charts, analysis tabs, and light/dark themes.

## Tech Stack

Python, NumPy, Pandas, Matplotlib, SciPy, Tkinter, and yfinance.

## Installation

1. Install Python 3.10 or later.
2. Clone this repository or download the source code.
3. Open a terminal in the project folder.
4. Create and activate a virtual environment.

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

Run the following command from the project folder, replacing `stocklens.py` if your Python file has a different name.

```bash
python stocklens.py
```

An internet connection is required to retrieve market data through Yahoo Finance. Data availability and access may vary.

## Methodology and Limitations

StockLens calculates historical performance and risk statistics from market data and uses a geometric Brownian motion model for Monte Carlo scenario analysis. Results depend on historical observations, data quality, and modelling assumptions.

Simulated price paths are illustrative scenarios, not reliable predictions of future prices or investment recommendations. Historical performance does not guarantee future results.

## Disclaimer

StockLens is an educational and analytical software project. It is not financial advice and does not execute trades.

## Author

## Author

**Essa Asim**

Student developer interested in Python, quantitative finance, financial risk analysis, and data science.

- **GitHub:** https://github.com/essaasim000-gif
- **LinkedIn:** www.linkedin.com/in/essa-asim-b04a623b9
