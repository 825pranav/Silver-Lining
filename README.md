# Silver Lining — Gold/Silver Ratio Trading System

A regime-aware quantitative trading system for Gold vs Silver allocation.
Combines HMM regime detection, three strategy sub-models, an XGBoost ensemble, walk-forward backtesting, and a live Streamlit dashboard.

> **Disclaimer:** For educational and research purposes only. Not financial advice.

---

## What it does

Given live market data, the system outputs one of three signals every day:

| Signal | Meaning |
|---|---|
| `BUY_GOLD` | GSR expected to rise — gold outperforms |
| `HOLD` | No clear edge |
| `BUY_SILVER` | GSR expected to fall — silver outperforms |

Each signal comes with a confidence score and full SHAP feature attribution explaining why the model chose that signal.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/825pranav/Silver-Lining.git
cd Silver-Lining

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501` in your browser. The pipeline fetches live data and builds automatically on first load (takes ~1–2 minutes).

---

## Project structure

```
Silver-Lining/
│
├── data_pipeline/
│   ├── price_fetcher.py        # Gold & Silver prices via yfinance (GC=F, SI=F)
│   └── macro_fetcher.py        # DXY, commodity index, 10Y/2Y yield curve
│
├── features/
│   ├── gsr_features.py         # GSR, rolling z-scores, slope, vol ratio
│   ├── momentum_features.py    # EMA slopes, MACD, breakout, relative strength
│   └── volatility_features.py  # ATR, rolling vol, GARCH(1,1) conditional vol
│
├── models/
│   ├── regime_detection.py     # GaussianHMM → crisis / trending / risk_on / range_bound
│   ├── strategy_models.py      # Mean-reversion, momentum, breakout sub-models
│   └── ensemble.py             # XGBoost classifier + SHAP explanations
│
├── backtesting/
│   ├── simulator.py            # Walk-forward backtest, vol-based position sizing
│   └── metrics.py              # Sharpe, CAGR, max drawdown, hit rate, Calmar
│
├── experiments/
│   └── walk_forward.py         # Rolling windows, IC analysis, alpha decay curve
│
├── dashboard/
│   └── streamlit_app.py        # Live dashboard (dark theme, 4 tabs)
│
├── requirements.txt
└── README.md
```

---

## How the pipeline works

```
yfinance / FRED
      │
      ▼
price_fetcher  ──►  gsr_features
macro_fetcher  ──►  momentum_features
                    volatility_features
                          │
                          ▼
                  regime_detection          ← GaussianHMM (4 states)
                          │
                          ▼
                  strategy_models           ← 3 regime-conditioned sub-models
                  (mean-rev / mom / bo)
                          │
                          ▼
                  ensemble (XGBoost)        ← trained on 80% of data, tested on 20%
                  + SHAP attribution
                          │
                          ▼
                  backtesting/simulator     ← walk-forward, vol sizing, no lookahead
                          │
                          ▼
                  dashboard (Streamlit)     ← live signal + charts
```

### Regime labels

The HMM assigns each day to one of four regimes:

| Regime | Characteristics |
|---|---|
| `crisis` | High volatility, extreme GSR dislocations |
| `trending` | Strong directional moves in gold or silver |
| `risk_on` | Low GSR z-score, risk appetite high |
| `range_bound` | Low volatility, GSR oscillating around mean |

Each strategy sub-model has per-regime confidence multipliers — mean-reversion is amplified in `crisis` and `range_bound`, momentum in `trending` and `risk_on`.

### Ensemble target

5-day forward GSR % change, discretised into 3 classes at ±0.30 σ:
- Class 0 → `BUY_GOLD` (GSR rises)
- Class 1 → `HOLD`
- Class 2 → `BUY_SILVER` (GSR falls)

Training uses the first 80% of the time series. No data from the test period touches the model during training.

---

## Dashboard tabs

| Tab | Contents |
|---|---|
| **Market Overview** | Live Gold / Silver / GSR prices, backtest metrics, equity curve, drawdown |
| **Live Signal** | Ensemble recommendation, SHAP waterfall, sub-model breakdown |
| **Alpha Decay** | Rolling Sharpe, hit rate, IC composite + exponential decay fit + half-life |
| **Regime Analysis** | Regime distribution, statistics, timeline, transition probability heatmap |

Sidebar controls: spread/slippage bps, walk-forward window and step size, refresh button.

---

## Running individual modules

Each module has a `main()` and can be run standalone:

```bash
# Fetch prices only
python data_pipeline/price_fetcher.py

# Fetch macro data
python data_pipeline/macro_fetcher.py

# Run regime detection on saved features
python models/regime_detection.py

# Run the full ensemble
python models/ensemble.py

# Walk-forward + alpha decay analysis
python experiments/walk_forward.py
```

Output CSVs are saved to `data/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Gold, Silver, DXY, commodity prices |
| `pandas-datareader` | FRED (10Y/2Y Treasury yields) |
| `hmmlearn` | Gaussian HMM regime detection |
| `xgboost` | Ensemble classifier |
| `shap` | Feature attribution / explainability |
| `arch` | GARCH(1,1) volatility (optional — falls back to EWMA if missing) |
| `streamlit` | Dashboard |
| `plotly` | Interactive charts |

Install everything:

```bash
pip install -r requirements.txt
```

---

## Porting to another machine

1. Clone the repo and run `pip install -r requirements.txt` — that's it.
2. No API keys required. All data is fetched from public sources (Yahoo Finance, FRED).
3. The dashboard fetches live data on first load; results are cached for 1 hour.
4. On Windows, if running modules directly in a terminal, set UTF-8 mode first (`chcp 65001`). The dashboard handles this internally.

---

## Backtest signal convention

```
signal > 0   →  BUY_SILVER  (GSR falling, silver outperforms)
signal < 0   →  BUY_GOLD    (GSR rising,  gold outperforms)
signal ≈ 0   →  HOLD
```

Position sizing is volatility-adjusted: `size = min(TARGET_VOL / realized_vol, 1.0)`.
No-lookahead enforced: signal at close_t → position entered at open_t+1.
