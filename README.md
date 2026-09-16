# Silver Lining — Gold/Silver Regime Allocation

A regime-aware research system that decides, each trading day, whether to hold **gold**, **silver**, or **nothing**. A Gaussian hidden Markov model labels the market regime, a gold/silver-ratio rule picks the metal, positions are sized by volatility, and the whole strategy is validated with a purged walk-forward backtest over 25 years of data.

> **Disclaimer:** For educational and research purposes only. Not financial advice. Past out-of-sample results do not predict future returns.

---

## Results at a glance

Out-of-sample, 23 annual walk-forward folds, 6,363 daily bars (2001-05-03 → 2026-09-11), costs included. Full per-fold table in [`docs/results.md`](docs/results.md), written by `experiments/validate.py`.

| Strategy | Profitable folds | Pooled Sharpe | Total return | Max drawdown |
|---|---|---|---|---|
| **Regime-gated (served)** | **17 / 23** | **0.708** | +750% | −51.3% |
| Ratio rule alone | 14 / 23 | 0.483 | +323% | −58.8% |
| Buy & hold gold | — | 0.690 | +1,031% | −44.4% |
| Buy & hold silver | — | 0.491 | +1,024% | −75.9% |

The regime layer adds about 0.22 Sharpe and three profitable years over the ratio rule alone. It roughly matches buy-and-hold gold on risk-adjusted return and trails it on total return over what was a long gold bull market.

---

## What it does

Every day the served strategy returns a **decision**: an asset, a position size between 0 and 1, the regime, and a one-line reason.

| Regime (from the HMM) | Position |
|---|---|
| `crisis` | hold gold — haven bid, ratio elevated |
| `risk_on` | hold silver — silver leads |
| `range_bound` | stand aside (size 0) — no edge |
| `trending` or unknown | the ratio rule decides (below) |

**Ratio rule.** If the gold/silver ratio's 30-day z-score is above +1 and its 5-day slope is under 0.5 in magnitude, the ratio is stretched and no longer rising: hold silver. Below −1 under the same condition: hold gold. Otherwise follow the ratio's drift — rising slope holds gold, falling slope holds silver.

**Sizing.** `size = min(0.01 / 20-day volatility of the held metal, 1.0)` — a 1% daily volatility target, never leveraged.

This logic lives in one function, `strategy.decide`, which the validation script, the backtest simulator and the dashboard all call, so the recommendation shown is the strategy that was measured.

---

## Quick start

```bash
git clone https://github.com/825pranav/Silver-Lining.git
cd Silver-Lining

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Regenerate the headline results
python experiments/validate.py

# Launch the dashboard
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501`. The dashboard fetches live data and builds the pipeline on first load (about 1–2 minutes), then caches it for an hour.

---

## Project structure

```
Silver-Lining/
│
├── strategy.py                 # decide(): the single served decision function
├── regime_model.py             # RegimeModel: HMM with a fit/predict split, joblib persistence
├── paths.py                    # repo-root-relative data paths
├── console.py                  # UTF-8-safe stdout for Windows consoles
│
├── data_pipeline/
│   ├── price_fetcher.py        # gold & silver futures via yfinance (GC=F, SI=F), full history
│   └── macro_fetcher.py        # DXY, commodity index, 10Y/2Y yields (fetched; not yet used by any model)
│
├── features/
│   ├── gsr_features.py         # ratio, 30/90-day z-scores, 5-day slope, 20-day vol, vol ratio
│   ├── momentum_features.py    # EMA slopes, MACD, 20-day breakout position, relative strength
│   └── volatility_features.py  # close-only ATR, 5/60-day vol, GARCH(1,1) conditional vol
│
├── models/
│   ├── regime_detection.py     # regime features, state-naming rules, fit_and_label (full-history fit)
│   ├── strategy_models.py      # mean-reversion, momentum, breakout sub-models (second opinion)
│   └── ensemble.py             # XGBoost classifier + SHAP (second opinion, not walk-forward tested)
│
├── backtesting/
│   ├── simulator.py            # backtest loop; delegates signals to strategy.decide
│   └── metrics.py              # Sharpe, Sortino, CAGR, max drawdown, hit rate, Calmar
│
├── experiments/
│   ├── validate.py             # purged walk-forward of the served strategy → docs/results.md
│   └── walk_forward.py         # rolling IC analysis and alpha-decay half-life
│
├── dashboard/
│   └── streamlit_app.py        # live dashboard (4 tabs)
│
├── docs/results.md             # generated results — do not edit by hand
├── requirements.txt
└── README.md
```

---

## How the pipeline works

```
yfinance (gold, silver futures)
      │
      ▼
price_fetcher ──► gsr_features ──► momentum_features ──► volatility_features
                                                               │
                                                               ▼
                                             RegimeModel (GaussianHMM, 4 states)
                                                               │
                                                               ▼
                                             strategy.decide  ← ratio rule + vol sizing
                                                               │
                              ┌────────────────────────────────┼─────────────────────┐
                              ▼                                ▼                     ▼
                  experiments/validate.py            dashboard recommendation   backtesting/simulator
                  (walk-forward, costs)
```

### Regime detection

The HMM is fitted on 11 standardised features: ratio z-score and slope, volatility ratio, gold/silver/ratio EMA slopes, gold MACD histogram, 20-day relative strength, gold ATR %, gold GARCH volatility and the GARCH volatility ratio. Hidden states have arbitrary numbers, so after every fit they are named from their average features:

| Regime | Assigned to the state with |
|---|---|
| `crisis` | the highest combined rank of volatility and ratio z-score |
| `range_bound` | the lowest volatility of the rest |
| `risk_on` | the lowest ratio z-score of the rest |
| `trending` | the remaining state |

`RegimeModel` (diagonal covariance, 200 iterations) fits on training rows only and predicts on unseen rows — this is what validation uses. `models/regime_detection.fit_and_label` fits and labels one frame with full covariance and 10 restarts, and is what the dashboard uses for display.

### Validation method

Expanding-window walk-forward in `experiments/validate.py`:

- at least **750** training bars before the first out-of-sample call
- **20** bars purged between training and test
- **250**-bar (about one year) test folds — 23 in total, the last one partial
- a fresh `RegimeModel` fitted on each fold's training rows
- **2 bp spread + 1 bp slippage** charged when switching away from a held metal

The loop is close-to-close: the position chosen at day *t*'s close earns the return from close *t* to close *t+1*. Only closing prices are fetched.

### Second opinion: sub-models and XGBoost

`models/strategy_models.py` computes three regime-weighted signals (mean reversion, momentum, breakout). `models/ensemble.py` feeds them, 15 raw features and the regime into an XGBoost classifier predicting the 5-day forward ratio move in three classes (split at ±0.30 standard deviations; 300 trees, depth 4, first 80% of rows for training) and explains each prediction with SHAP. The dashboard shows this as a labelled second opinion. It has **not** been walk-forward validated, so no performance is claimed for it.

---

## Dashboard tabs

| Tab | Contents |
|---|---|
| **Market Overview** | Gold, silver and ratio prices, backtest metrics, equity curve, drawdown |
| **Live Signal** | The served decision (asset, size, regime, reason), plus the XGBoost second opinion with its SHAP waterfall and sub-model breakdown |
| **Alpha Decay** | Rolling Sharpe, hit rate, information coefficient, exponential decay fit and half-life |
| **Regime Analysis** | Regime distribution, statistics, timeline and empirical transition heatmap |

Sidebar controls: spread and slippage, walk-forward window and step size, refresh.

---

## Known limitations

- **GARCH is fitted once on the full history**, so the GARCH features that feed the regime model use parameters estimated partly from later data. The HMM itself is refit per fold.
- **The dashboard's equity curve is the ratio rule without regimes**, and its regimes come from a full-history `fit_and_label` fit. `experiments/validate.py` is the source of truth for performance; the persisted `data/regime_model.joblib` is not yet loaded by the dashboard.
- **Macro data is fetched but unused** by any model or feature.
- **XGBoost leaks mildly**: its class threshold uses the standard deviation of the whole series, and its regime input comes from a full-history fit.
- **Simplified market model**: continuous futures with roll jumps, Sharpe with a zero risk-free rate, flat costs, no financing, and a 20-bar purge shorter than the longest 60-bar feature window.
- **Thresholds and the regime-to-metal mapping were set by hand**, not tuned on test folds — but chosen with knowledge of this period.

---

## Running individual modules

Every script resolves data paths from the repository root, so these work from anywhere:

```bash
# Rebuild data and features — run in this order; each stage adds columns
python data_pipeline/price_fetcher.py
python features/gsr_features.py
python features/momentum_features.py
python features/volatility_features.py

# Served strategy: purged walk-forward, writes docs/results.md and data/validation_folds.csv
python experiments/validate.py

# Regime labelling on the full feature set
python models/regime_detection.py

# Second opinion: sub-models and XGBoost + SHAP
python models/strategy_models.py
python models/ensemble.py

# Rolling IC and alpha decay
python experiments/walk_forward.py

# Macro data (not used by any model yet)
python data_pipeline/macro_fetcher.py
```

Output CSVs are saved to `data/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Gold and silver futures; DXY and commodity index |
| `pandas-datareader` | FRED Treasury yields |
| `pandas`, `numpy` | Features and backtests |
| `hmmlearn` | Gaussian HMM regime detection |
| `scikit-learn` | Feature scaling and label encoding |
| `arch` | GARCH(1,1) volatility (falls back to EWMA if missing) |
| `scipy` | Spearman IC and decay curve fitting |
| `xgboost`, `shap` | Second-opinion classifier and attribution |
| `streamlit`, `plotly` | Dashboard |

```bash
pip install -r requirements.txt
```

No API keys are required; all data comes from Yahoo Finance and FRED. On Windows, scripts reconfigure stdout to UTF-8 themselves.

---

## Signal and position conventions

```
decision.asset = "gold"    → hold gold
decision.asset = "silver"  → hold silver
decision.asset = None      → stand aside (size 0)
decision.size  ∈ [0, 1]    → min(0.01 / realised 20-day vol, 1.0)
```

The second-opinion sub-models use a separate signed convention: positive leans silver, negative leans gold, near zero is hold.
