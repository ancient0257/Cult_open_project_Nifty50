# 📈 NIFTY-50 Investment Intelligence Platform

An AI-powered investment analytics platform built for the **Data-Driven Investment Intelligence Using NIFTY-50 Market Data** hackathon. The platform transforms 20+ years of historical NIFTY-50 market data into actionable investment insights through machine learning, portfolio optimisation, and risk analytics.

---

## 🚀 Features

### Mandatory
| Module | Description |
|--------|-------------|
| **Stock Predictor Engine** | Random Forest classifier (direction) + regressor (5-day return), walk-forward evaluated |
| **Portfolio Construction** | Markowitz mean-variance optimisation — Conservative, Balanced, Aggressive & Equal-Weight profiles |
| **Risk Assessment** | Per-stock Sharpe, Sortino, VaR 95%, CVaR, Max Drawdown, Calmar, rolling volatility |

### Optional (Implemented)
| Module | Description |
|--------|-------------|
| **Explainable AI** | Feature importance (RF), signal-level explanations, plain-English narratives for every recommendation |
| **Personalized Strategies** | Three distinct investor profiles with sector diversification and quantitative justification |
| **Market Anomaly Detection** | Z-score based detection of return spikes, crashes, and volume anomalies |
| **Forecasting Module** | 5-day return regression + directional probability forecasting |
| **Deployment** | Streamlit web application with 7 interactive pages |

---

## 📁 Project Structure

```
nifty50_platform/
├── app.py                   # Main Streamlit application (7 pages)
├── generate_outputs.py      # Pre-generation script for all charts/models
├── src/
│   ├── data_loader.py       # Data loading + 30+ technical indicators
│   ├── risk_module.py       # Risk metrics + anomaly detection
│   ├── portfolio_module.py  # Portfolio optimisation + backtesting
│   ├── predictor.py         # ML models (RF classifier + regressor)
│   ├── explainer.py         # XAI: signal summaries + prediction narratives
│   └── visualizer.py        # All charts (matplotlib, dark theme)
├── data/
│   ├── stocks/              # 50 NIFTY-50 stock CSVs
│   └── index/               # NIFTY 50 index data
├── models/                  # Saved .pkl models (50 stocks)
├── outputs/                 # Pre-generated charts + CSVs
└── reports/                 # Technical report (PDF)
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9+
- ~500 MB disk space

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data Setup
Place the NIFTY-50 dataset CSVs in:
```
data/stocks/   ← one CSV per symbol (e.g. RELIANCE.csv, TCS.csv)
data/index/    ← NIFTY 50.csv, INDIA VIX.csv
```

---

## ▶️ Running the Application

### Option A — Streamlit web app (recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Option B — Pre-generate all outputs first
This trains all 50 ML models and generates all charts ahead of time (~5–10 minutes):
```bash
python3 generate_outputs.py
streamlit run app.py
```

---

## 🤖 Reproducing Results

### 1. Train all models
```python
from src.data_loader import load_stock, add_technical_indicators
from src.predictor import train_stock_model, save_model

df = load_stock('RELIANCE')
df = add_technical_indicators(df)
result = train_stock_model(df, 'RELIANCE')
print(result['metrics'])
```

### 2. Build a portfolio
```python
from src.data_loader import build_returns_matrix
from src.portfolio_module import build_balanced_portfolio

prices, returns = build_returns_matrix()
portfolio = build_balanced_portfolio(returns)
print(portfolio['Metrics'])
print(portfolio['Weights'])
```

### 3. Assess risk
```python
from src.risk_module import compute_stock_risk
risk = compute_stock_risk(df, 'RELIANCE')
print(risk)
```

### 4. Detect anomalies
```python
from src.risk_module import detect_anomalies
anomalies = detect_anomalies(df, z_threshold=3.0)
print(anomalies.head())
```

---

## 📊 Technical Indicators Implemented

| Category | Indicators |
|----------|-----------|
| Trend | MA(5,10,20,50,100,200), EMA(5,10,20,50,100,200), Golden/Death Cross |
| Momentum | MACD, MACD Signal, MACD Histogram, RSI(14), Stochastic %K/%D |
| Volatility | Bollinger Bands (20,2σ), ATR(14), 20d/60d rolling volatility |
| Volume | OBV (On-Balance Volume) |
| Return | Daily return, log return, momentum(5,10,20,60d), gap-up |

---

## 📈 Model Architecture

### Direction Model (Classification)
- **Algorithm:** Random Forest Classifier (200 trees, max_depth=8)
- **Target:** Next-day price direction (Up/Down)
- **Features:** 30+ technical indicators + 5 lag periods
- **Evaluation:** Accuracy, Precision, Recall, F1 via TimeSeriesSplit

### Return Model (Regression)
- **Algorithm:** Random Forest Regressor
- **Target:** 5-day forward return (%)
- **Evaluation:** MAE, RMSE, R² via TimeSeriesSplit

### Average Performance (50 stocks)
| Metric | Value |
|--------|-------|
| Directional Accuracy | ~51% |
| Best Stock Accuracy | ~56% (SBILIFE) |
| MAE (5-day return) | ~3.2% |

> Note: Stock market direction prediction above 55% consistently is generally considered strong. Random walk hypothesis applies.

---

## 💼 Portfolio Results (2010–2021 backtest)

| Profile | Annual Return | Volatility | Sharpe | Max Drawdown | Total Return |
|---------|--------------|-----------|--------|-------------|-------------|
| Conservative | 13.9% | 13.3% | 0.524 | -23.1% | +223.5% |
| Balanced | 32.8% | 17.9% | 1.253 | -33.2% | +1723.9% |
| Aggressive | 36.2% | 23.3% | 1.068 | -44.1% | +2002.1% |
| Equal Weight | 13.9% | 17.0% | 0.415 | -38.4% | +114.7% |

---

## ⚠️ Disclaimer

This platform is built for **educational and research purposes** as part of a data science competition. The models, predictions, and portfolio recommendations are based solely on historical data and **do not constitute financial advice**. Past performance does not guarantee future results.

---

## 👥 Team

Built for the NIFTY-50 Investment Intelligence Hackathon.

Dataset: [Kaggle — NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
