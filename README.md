# Silver Lining — Gold/Silver Regime Allocation

A research system that decides, each trading day, whether to hold **gold**, **silver**, or **nothing**. It combines a Gaussian hidden Markov model of market regimes, a gold/silver-ratio rule, a trend gate and volatility-targeted sizing. Every variant is tested with a purged walk-forward backtest over 25 years of data. Configurations are chosen on early folds only, and each result comes with a significance test.

> **Disclaimer:** For educational and research purposes only. Not financial advice. Past out-of-sample results do not predict future returns.

---

## Results at a glance

Out-of-sample, 23 annual walk-forward folds (2004-06-08 → 2026-09-11), one-day execution lag, 3bp per unit of turnover. Every number is copied from [`docs/results.md`](docs/results.md), which `experiments/validate.py` writes.

| Strategy | Profitable folds | Sharpe | Total return | Ann. vol | Max drawdown |
|---|---|---|---|---|---|
| **Served: vol-targeted gold, 12-month trend gate** | 13 / 23 | **0.721** | +543% | 12.8% | **−29.7%** |
| Best regime-based configuration (HMM, sticky prior, trend gate) | 14 / 23 | 0.553 | +315% | 13.2% | −33.2% |
| Original rules (HMM regimes + ratio rule), lookahead removed | 15 / 23 | 0.462 | +270% | 15.3% | −54.1% |
| Buy & hold gold | 18 / 23 | 0.691 | +1,036% | 18.3% | −44.4% |
| Buy & hold silver | 15 / 23 | 0.494 | +1,047% | 34.7% | −75.8% |
| Buy & hold gold scaled to 12.8% vol (diagnostic) | 18 / 23 | 0.691 | +490% | 12.8% | −32.5% |

What this does and does not show:

- **The earlier headline (Sharpe 0.708) had lookahead bias.** Regimes were decoded with Viterbi over each whole test year, GARCH was fitted on all 25 years, and trades filled at the same close that produced the signal. With those fixed, the original strategy scores 0.462. The audit table in `docs/results.md` reproduces 0.708 and removes the biases one at a time.
- **The HMM does not earn its place.** In a 120-configuration search (`docs/search.md`), the configuration with the best early-fold Sharpe uses no regimes, so that is what is served. The best regime-based configuration trails buy-and-hold gold.
- **Drawdown: yes. Sharpe: not proven.** The served strategy's max drawdown is −29.7% against −44.4% for gold, and its Sharpe is 0.721 against 0.691. The Sharpe gap is not significant: bootstrap 95% CI [−0.188, +0.255]. Most of the drawdown gap comes from holding less. Gold scaled to the same volatility draws down −32.5%.
- **Later years are weaker.** On folds 12–23 the served strategy's Sharpe is 0.599 against 0.766 for gold. Deflated Sharpe Ratio over all 120 trials: 0.983 (probability the Sharpe is above zero after the search, not that it beats gold).

---

## What it does

Every day the served strategy returns a **decision**: an asset, per-metal weights between 0 and 1, the regime (when one is used), and a one-line reason. All variants share one function, `strategy.decide`, configured by a `StrategyConfig`:

| Setting | Options |
|---|---|
| Regime model | none, or a Gaussian HMM (4 states or BIC-chosen from 3/4/5; optional sticky transition prior; optional scale-free features) |
| Regime use | hard (act on the most likely regime) or soft (weight each regime's action by its filtered probability) |
| Universe | `switch`: regimes and the ratio rule choose between metals. `gold_core`: gold unless risk-on. `gold_only`: silver is never held |
| Trend gate | off, 126 or 252 days: a metal is held only while its trailing return is positive |
| Drawdown control | off, or halve exposure while more than 10% below the one-year equity peak |

Regime mapping (for regime-based configurations):

| Regime (from the HMM) | Position |
|---|---|
| `crisis` | hold gold: haven bid, ratio elevated |
| `risk_on` | hold silver: silver leads |
| `range_bound` | stand aside (size 0): no edge |
| `trending` or unknown | the ratio rule decides (below) |

**Ratio rule.** If the gold/silver ratio's 30-day z-score is above +1 and the ratio's 5-day change is under 0.5 in magnitude, the ratio is stretched and no longer rising: hold silver. Below −1 under the same condition: hold gold. Otherwise follow the ratio's drift: a rising ratio holds gold, a falling one holds silver.

**Sizing.** Each metal leg is `min(0.01 / 20-day volatility, 1.0)` times its regime weight: a 1% daily volatility target, never levered.

**Served configuration** (`strategy.SERVED`): no regime model, gold only, 252-day trend gate, no drawdown control. In words: hold gold sized to 1% daily volatility while gold's trailing 12-month return is positive; otherwise stand aside. `experiments/search.py` chose it on folds 1–11 only.

`strategy.simulate` runs `decide` bar by bar. The validation script, the backtest simulator and the dashboard all call it, so the recommendation shown is the strategy that was measured.

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

# Choose the configuration (folds 1-11 only), then report it on all folds
python experiments/search.py
python experiments/validate.py

# Tests
python -m pytest

# Launch the dashboard
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501`. The dashboard fetches live data and builds the pipeline on first load (about 1–2 minutes), then caches it for an hour.

---

## Project structure

```
Silver-Lining/
│
├── strategy.py                 # StrategyConfig, decide(), simulate(): the served strategy
├── regime_model.py             # RegimeModel: causal HMM (forward filter, train-only GARCH)
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
│   ├── metrics.py              # Sharpe, Sortino, CAGR, max drawdown, hit rate, Calmar
│   └── stats.py                # PSR, Deflated Sharpe, block-bootstrap Sharpe difference
│
├── experiments/
│   ├── walkforward_core.py     # folds, per-fold regime fits, out-of-sample paths
│   ├── search.py               # 120-configuration search, chosen on folds 1-11 → docs/search.md
│   ├── validate.py             # served strategy on all folds, stats, audit → docs/results.md
│   └── walk_forward.py         # rolling IC analysis and alpha-decay half-life
│
├── dashboard/
│   └── streamlit_app.py        # live dashboard (4 tabs)
│
├── tests/                      # causality, timing, cost and statistics tests
├── docs/results.md             # generated results — do not edit by hand
├── docs/search.md              # generated search report — do not edit by hand
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
                                   RegimeModel (GaussianHMM, forward-filtered)
                                                               │
                                                               ▼
                           strategy.simulate → strategy.decide ← config: regimes, trend gate, sizing
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

`RegimeModel` (diagonal covariance, 200 iterations) fits on training rows only, including its scaler and its GARCH(1,1) parameters. Validation uses it. Its probabilities come from the **forward filter**: the call on day *t* uses bars up to *t*, and a test checks that appending future rows never changes past output. The earlier version used Viterbi decoding over the whole test year, so each label depended on later bars. `models/regime_detection.fit_and_label` fits one frame with full covariance and 10 restarts. The dashboard uses it only for the regime charts.

### Validation method

Expanding-window walk-forward in `experiments/walkforward_core.py`:

- at least **750** training bars before the first out-of-sample call
- **20** bars purged between training and test
- **250**-bar (about one year) test folds: 23 in total, the last one partial
- a fresh `RegimeModel` fitted on each fold's training rows
- a position decided at the close of day *t* is traded at the close of *t+1*
- **3bp** (2bp spread + 1bp slippage) per unit of turnover, including resizing
- one continuous path across folds, so positions and drawdown state carry over

**Choosing without peeking.** `experiments/search.py` runs every configuration and picks the highest pooled Sharpe on **folds 1–11** (mid-2004 to mid-2015). Folds 12–23 are reported for every configuration but never used for a choice. Round 1 (102 configurations) was fixed in advance. Round 2 (18 gold-core configurations) was added after round-1 results were seen, and is labelled exploratory. The Deflated Sharpe Ratio counts all 120.

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

- **No significant edge over buy-and-hold gold.** The served strategy's Sharpe advantage is within bootstrap noise, and it trails gold on folds 12–23. Its drawdown advantage mostly comes from lower average exposure (64%).
- **The served configuration came from an exploratory round.** Round 2 was added after round-1 full-sample results were seen, so folds 12–23 are a clean holdout only for the round-1 configuration.
- **The dashboard's regime charts come from a full-history fit** (`fit_and_label`), so they are context only. The served decision and equity curve do not use regimes.
- **Macro data is fetched but unused** by any model or feature.
- **XGBoost leaks mildly**: its class threshold uses the standard deviation of the whole series, and its regime input comes from a full-history fit. It is a labelled second opinion and is not used for decisions.
- **Simplified market model**: continuous futures with roll jumps, Sharpe with a zero risk-free rate (futures returns are already excess of cash), flat costs, closing prices only.
- **Hand-set rules**: the ratio-rule thresholds, the regime-to-metal mapping and the 1% vol target were set by hand with knowledge of this period, not searched.

---

## Running individual modules

Every script resolves data paths from the repository root, so these work from anywhere:

```bash
# Rebuild data and features — run in this order; each stage adds columns
python data_pipeline/price_fetcher.py
python features/gsr_features.py
python features/momentum_features.py
python features/volatility_features.py

# Configuration search on folds 1-11: writes docs/search.md and data/search_trials.csv
python experiments/search.py

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
| `pytest` | Tests |

```bash
pip install -r requirements.txt
```

No API keys are required; all data comes from Yahoo Finance and FRED. On Windows, scripts reconfigure stdout to UTF-8 themselves.

---

## Signal and position conventions

```
decision.asset         = "gold" | "silver" → the larger holding
decision.asset         = None              → stand aside (size 0)
decision.gold_weight   ∈ [0, 1]            → regime weight × min(0.01 / gold 20-day vol, 1)
decision.silver_weight ∈ [0, 1]            → regime weight × min(0.01 / silver 20-day vol, 1)
decision.size          = gold_weight + silver_weight ≤ 1
```

The second-opinion sub-models use a separate signed convention: positive leans silver, negative leans gold, near zero is hold.
