# dashboard/streamlit_app.py
#
# Silver Lining — Gold/Silver Ratio Trading Dashboard
#
# Ties together every existing module without reimplementing any logic:
#   data_pipeline/price_fetcher      fetch_prices()
#   features/gsr_features            compute_zscore()
#   features/momentum_features       compute_ema()
#   features/volatility_features     compute_atr(), compute_garch_vol()
#   models/regime_detection          fit_and_label()          [needs hmmlearn]
#   models/strategy_models           build_strategy_signals()
#   models/ensemble                  fit_and_predict()        [needs xgboost + shap]
#   backtesting/simulator            _run_fold()
#   backtesting/metrics              compute_all()
#   experiments/walk_forward         rolling_walk_forward(), fit_exponential_decay(),
#                                    aggregate_summary(), _exp_decay()
#
# Run from repo root:
#   streamlit run dashboard/streamlit_app.py
#
# Required:  pip install streamlit plotly
# Optional:  pip install hmmlearn xgboost shap
#            (dashboard degrades gracefully without them)

import os
import sys
import contextlib
import warnings
import numpy as np
import pandas as pd


@contextlib.contextmanager
def _silence():
    """Redirect stdout to /dev/null to suppress emoji print() calls in
    existing modules that crash on Windows cp1252 encoding."""
    with open(os.devnull, "w", encoding="utf-8") as null:
        with contextlib.redirect_stdout(null):
            yield

# ── path setup ───────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── hard dependencies ─────────────────────────────────────────────────────────
try:
    import streamlit as st
except ImportError:
    raise ImportError("streamlit is not installed.  Run:  pip install streamlit")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("plotly is not installed.  Run:  pip install plotly")

# ── project imports ───────────────────────────────────────────────────────────
from data_pipeline.price_fetcher        import fetch_prices
from features.gsr_features              import compute_zscore
from features.momentum_features         import compute_ema
from features.volatility_features       import compute_atr, compute_garch_vol
from models.strategy_models             import build_strategy_signals
from backtesting.simulator              import _run_fold
from backtesting.metrics                import compute_all
from experiments.walk_forward           import (
    rolling_walk_forward,
    fit_exponential_decay,
    aggregate_summary,
    _exp_decay,
    _available_signals,
    _forward_gsr,
    _composite_ic,
)

# ── optional imports (graceful degradation) ───────────────────────────────────
try:
    from models.regime_detection import fit_and_label
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

try:
    from models.ensemble import fit_and_predict, LABEL_MAP
    _ENSEMBLE_OK = True
except ImportError:
    _ENSEMBLE_OK = False
    LABEL_MAP = {0: "BUY_GOLD", 1: "HOLD", 2: "BUY_SILVER"}

# ── colour palette ────────────────────────────────────────────────────────────
_REGIME_COLORS = {
    "crisis":      "#e74c3c",
    "trending":    "#3498db",
    "risk_on":     "#2ecc71",
    "range_bound": "#95a5a6",
}
_SIGNAL_COLORS = {
    "BUY_GOLD":   "#f1c40f",
    "HOLD":       "#95a5a6",
    "BUY_SILVER": "#bdc3c7",
}

# =============================================================================
# Pipeline builder  (cached — rebuilds at most once per hour)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="⚙️  Building pipeline from live data…")
def _build_pipeline(spread: float, slippage: float):
    """
    Fetch live prices, compute all features using imported functions,
    run regime detection + strategy signals + ensemble (where available),
    and execute a single-pass backtest.

    Returns
    -------
    df      : Full feature DataFrame (DatetimeIndex)
    capital : pd.Series — portfolio value from _run_fold
    rets    : pd.Series — daily net returns
    metrics : dict      — compute_all() output
    warns   : list[str] — degradation warnings to surface in the UI
    """
    warns = []

    # ── 1. live prices ────────────────────────────────────────────────────────
    with _silence():
        df = fetch_prices()                      # gold_close, silver_close, gsr

    # ── 2. GSR features  (compute_zscore from gsr_features) ──────────────────
    df["gsr_zscore_30"] = compute_zscore(df["gsr"], 30)
    df["gsr_zscore_90"] = compute_zscore(df["gsr"], 90)
    df["gsr_slope"]     = df["gsr"].diff(5)
    df["gold_ret_5"]    = df["gold_close"].pct_change(5)
    df["gold_ret_20"]   = df["gold_close"].pct_change(20)
    df["silver_ret_5"]  = df["silver_close"].pct_change(5)
    df["silver_ret_20"] = df["silver_close"].pct_change(20)
    df["gold_vol_20"]   = df["gold_close"].pct_change().rolling(20).std()
    df["silver_vol_20"] = df["silver_close"].pct_change().rolling(20).std()
    df["vol_ratio"]     = df["gold_vol_20"] / df["silver_vol_20"]

    # ── 3. momentum features  (compute_ema from momentum_features) ────────────
    df["gold_ema_10"]    = compute_ema(df["gold_close"], 10)
    df["gold_ema_30"]    = compute_ema(df["gold_close"], 30)
    df["gold_ema_slope"] = df["gold_ema_10"] - df["gold_ema_30"]

    df["silver_ema_10"]    = compute_ema(df["silver_close"], 10)
    df["silver_ema_30"]    = compute_ema(df["silver_close"], 30)
    df["silver_ema_slope"] = df["silver_ema_10"] - df["silver_ema_30"]

    df["gsr_ema_10"]    = compute_ema(df["gsr"], 10)
    df["gsr_ema_30"]    = compute_ema(df["gsr"], 30)
    df["gsr_ema_slope"] = df["gsr_ema_10"] - df["gsr_ema_30"]

    _g12 = compute_ema(df["gold_close"], 12)
    _g26 = compute_ema(df["gold_close"], 26)
    df["gold_macd"]        = _g12 - _g26
    df["gold_macd_signal"] = compute_ema(df["gold_macd"], 9)
    df["gold_macd_hist"]   = df["gold_macd"] - df["gold_macd_signal"]

    _s12 = compute_ema(df["silver_close"], 12)
    _s26 = compute_ema(df["silver_close"], 26)
    df["silver_macd"]        = _s12 - _s26
    df["silver_macd_signal"] = compute_ema(df["silver_macd"], 9)
    df["silver_macd_hist"]   = df["silver_macd"] - df["silver_macd_signal"]

    _w = 20
    _gh = df["gold_close"].rolling(_w).max()
    _gl = df["gold_close"].rolling(_w).min()
    df["gold_breakout"] = (df["gold_close"] - _gl) / (_gh - _gl).clip(lower=1e-9)

    _sh = df["silver_close"].rolling(_w).max()
    _sl = df["silver_close"].rolling(_w).min()
    df["silver_breakout"] = (df["silver_close"] - _sl) / (_sh - _sl).clip(lower=1e-9)

    df["rel_strength_5"]  = df["gold_ret_5"]  - df["silver_ret_5"]
    df["rel_strength_20"] = df["gold_ret_20"] - df["silver_ret_20"]

    # ── 4. volatility features  (compute_atr, compute_garch_vol) ─────────────
    gold_ret   = df["gold_close"].pct_change()
    silver_ret = df["silver_close"].pct_change()

    df["gold_atr_14"]   = compute_atr(df["gold_close"], 14)
    df["silver_atr_14"] = compute_atr(df["silver_close"], 14)
    df["gold_atr_pct"]   = df["gold_atr_14"]   / df["gold_close"]
    df["silver_atr_pct"] = df["silver_atr_14"] / df["silver_close"]

    df["gold_vol_5"]    = gold_ret.rolling(5).std()
    df["gold_vol_60"]   = gold_ret.rolling(60).std()
    df["silver_vol_5"]  = silver_ret.rolling(5).std()
    df["silver_vol_60"] = silver_ret.rolling(60).std()
    df["vol_ratio_5"]   = df["silver_vol_5"]  / df["gold_vol_5"]
    df["vol_ratio_60"]  = df["silver_vol_60"] / df["gold_vol_60"]

    with _silence(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gold_garch   = compute_garch_vol(gold_ret,   label="gold")
        silver_garch = compute_garch_vol(silver_ret, label="silver")

    df["gold_garch_vol"]   = gold_garch.reindex(df.index)
    df["silver_garch_vol"] = silver_garch.reindex(df.index)
    df["garch_vol_ratio"]  = df["silver_garch_vol"] / df["gold_garch_vol"]

    df.dropna(inplace=True)

    # ── 5. regime detection  (fit_and_label from regime_detection) ───────────
    if _HMM_OK:
        with _silence(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = fit_and_label(df, n_states=4, n_restarts=5)
    else:
        warns.append("hmmlearn not installed — regime set to 'range_bound'. "
                     "Run:  pip install hmmlearn")
        df["regime"]       = "range_bound"
        df["regime_state"] = 0

    # ── 6. strategy signals  (build_strategy_signals from strategy_models) ───
    with _silence():
        df = build_strategy_signals(df)

    # ── 7. ensemble  (fit_and_predict from ensemble) ──────────────────────────
    if _ENSEMBLE_OK:
        with _silence(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = fit_and_predict(df)
    else:
        warns.append("xgboost / shap not installed — ensemble signal unavailable. "
                     "Run:  pip install xgboost shap")
        df["ensemble_signal"]     = None
        df["ensemble_confidence"] = None

    # ── 8. single-pass backtest  (_run_fold from simulator) ──────────────────
    capital, rets, _ = _run_fold(df, spread=spread, slippage=slippage)
    metrics = compute_all(capital, rets)

    return df, capital, rets, metrics, warns


@st.cache_data(ttl=86400, show_spinner="🔄 Running walk-forward analysis…")
def _build_walk_forward(df_json: str, window_size: int, step_size: int,
                        spread: float, slippage: float):
    """
    Cached wrapper for the walk-forward + alpha decay analysis.
    Accepts df serialised as JSON to satisfy Streamlit's cache hash requirements.
    Uses rolling_walk_forward + fit_exponential_decay from experiments/walk_forward.
    """
    df = pd.read_json(df_json)
    df.index = pd.to_datetime(df.index)

    with _silence():
        fold_summary = rolling_walk_forward(
            df,
            window_size=window_size,
            step_size=step_size,
            purge_days=5,
            spread=spread,
            slippage=slippage,
        )

    decay_params = fit_exponential_decay(
        fold_summary["ic_composite"],
        step_size_days=step_size,
    )

    agg = aggregate_summary(fold_summary)
    return fold_summary, decay_params, agg


# =============================================================================
# Chart builders  (all return plotly Figure objects)
# =============================================================================

def _chart_equity(capital: pd.Series, df: pd.DataFrame) -> go.Figure:
    """Equity curve with drawdown subplot and regime color bands."""
    rolling_peak = capital.cummax()
    drawdown     = (capital - rolling_peak) / rolling_peak

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    # Regime background bands (if available)
    if "regime" in df.columns:
        _add_regime_bands(fig, df["regime"].reindex(capital.index), rows=[1, 2])

    fig.add_trace(go.Scatter(
        x=capital.index, y=capital.values,
        name="Portfolio NAV", line=dict(color="#3498db", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>NAV: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        name="Drawdown %", line=dict(color="#e74c3c", width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title="Equity Curve & Drawdown",
        height=480, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=50, r=20, t=50, b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="NAV (×1)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def _chart_prices(df: pd.DataFrame) -> go.Figure:
    """Gold & Silver price chart with regime background bands."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])

    if "regime" in df.columns:
        _add_regime_bands(fig, df["regime"], rows=[1, 2])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["gold_close"],
        name="Gold", line=dict(color="#f39c12", width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["silver_close"],
        name="Silver", line=dict(color="#95a5a6", width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["gsr"],
        name="GSR", line=dict(color="#9b59b6", width=1.8),
    ), row=2, col=1)

    # GSR z-score bands at ±1 and ±2
    if "gsr_zscore_30" in df.columns:
        mean30 = df["gsr"].rolling(30).mean()
        std30  = df["gsr"].rolling(30).std()
        fig.add_trace(go.Scatter(
            x=df.index, y=(mean30 + std30).values,
            name="+1σ", line=dict(color="rgba(231,76,60,0.4)", dash="dot", width=1),
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=(mean30 - std30).values,
            name="−1σ", line=dict(color="rgba(46,204,113,0.4)", dash="dot", width=1),
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        title="Gold & Silver Prices + GSR",
        height=480, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=50, r=20, t=50, b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="GSR", row=2, col=1)
    return fig


def _chart_shap_waterfall(last_row: pd.Series) -> go.Figure:
    """Horizontal waterfall chart for SHAP feature attributions (today's signal)."""
    shap_cols = sorted(
        [c for c in last_row.index if c.startswith("shap_")],
        key=lambda c: abs(last_row[c]),
        reverse=True,
    )[:15]                            # top 15 by |SHAP|

    labels = [c.replace("shap_", "") for c in shap_cols]
    values = [last_row[c] for c in shap_cols]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#8b949e")
    fig.update_layout(
        title="SHAP Feature Attribution — Today's Signal",
        xaxis_title="SHAP value  (+ = BUY_SILVER, − = BUY_GOLD)",
        height=420,
        margin=dict(l=160, r=80, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _chart_regime_timeline(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing time spent in each regime over the dataset."""
    if "regime" not in df.columns:
        return go.Figure().add_annotation(text="Regime data unavailable",
                                          showarrow=False)

    dummies = pd.get_dummies(df["regime"])
    # Rolling 30-day proportion of each regime
    proportions = dummies.rolling(30, min_periods=5).mean()

    fig = go.Figure()
    for regime in _REGIME_COLORS:
        if regime in proportions.columns:
            fig.add_trace(go.Scatter(
                x=proportions.index,
                y=proportions[regime],
                name=regime,
                stackgroup="one",
                mode="none",
                fillcolor=_REGIME_COLORS[regime],
                line=dict(color=_REGIME_COLORS[regime]),
                hovertemplate=f"{regime}: %{{y:.1%}}<extra></extra>",
            ))

    fig.update_layout(
        title="Regime Presence (30-day rolling proportion)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=50, r=20, t=50, b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _chart_transition_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Empirical regime transition probability heatmap.
    Computed directly from the 'regime' column (not the HMM model matrix,
    which is printed but not returned by fit_and_label).
    """
    if "regime" not in df.columns or df["regime"].isna().all():
        return go.Figure().add_annotation(text="Regime data unavailable",
                                          showarrow=False)

    reg = df["regime"].dropna()
    regimes = sorted(reg.unique())
    matrix  = pd.DataFrame(0.0, index=regimes, columns=regimes)

    for i in range(1, len(reg)):
        fr, to = reg.iloc[i - 1], reg.iloc[i]
        matrix.loc[fr, to] += 1

    row_sums = matrix.sum(axis=1).clip(lower=1e-9)
    matrix   = matrix.div(row_sums, axis=0)

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[f"{v:.2%}" for v in row] for row in matrix.values],
        texttemplate="%{text}",
        hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.2%}<extra></extra>",
        colorbar=dict(title="Probability", tickformat=".0%"),
    ))
    fig.update_layout(
        title="Regime Transition Probability Heatmap (Empirical)",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=380,
        margin=dict(l=120, r=20, t=50, b=60),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _chart_alpha_decay(fold_summary: pd.DataFrame, decay_params: dict) -> go.Figure:
    """
    Three-panel alpha decay chart:
      Top    — Rolling Sharpe per window
      Middle — Rolling Hit Rate per window
      Bottom — IC composite with exponential decay fit overlay
    Uses _exp_decay from experiments/walk_forward.
    """
    if fold_summary.empty:
        return go.Figure().add_annotation(text="No walk-forward data", showarrow=False)

    x = pd.to_datetime(fold_summary["window_start"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Rolling Sharpe", "Rolling Hit Rate", "IC Composite + Decay Fit"],
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34],
    )

    # Panel 1: Sharpe bars
    sharpe_vals = fold_summary["sharpe"].tolist()
    fig.add_trace(go.Bar(
        x=x, y=sharpe_vals,
        name="Sharpe",
        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in sharpe_vals],
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="#8b949e", line_width=0.8, row=1, col=1)

    # Panel 2: Hit Rate
    fig.add_trace(go.Scatter(
        x=x, y=fold_summary["hit_rate"],
        name="Hit Rate", mode="lines+markers",
        line=dict(color="#3498db", width=1.8),
        marker=dict(size=5),
        hovertemplate="%{x|%Y-%m-%d}<br>Hit Rate: %{y:.1%}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_color="#8b949e", line_dash="dot", line_width=1, row=2, col=1)

    # Panel 3: IC + decay fit
    ic_vals = fold_summary["ic_composite"].tolist()
    fig.add_trace(go.Scatter(
        x=x, y=ic_vals,
        name="IC composite", mode="lines+markers",
        line=dict(color="#9b59b6", width=1.8),
        marker=dict(size=5, symbol="square"),
        hovertemplate="%{x|%Y-%m-%d}<br>IC: %{y:.4f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="#8b949e", line_width=0.8, row=3, col=1)

    # Exponential decay fit overlay (uses _exp_decay from walk_forward)
    if decay_params.get("fit_success"):
        t_arr  = np.arange(len(ic_vals), dtype=float)
        a      = decay_params["alpha_initial"] - decay_params["alpha_floor"]
        b      = decay_params["decay_rate"]
        c      = decay_params["alpha_floor"]
        y_fit  = _exp_decay(t_arr, a, b, c)
        hl     = decay_params["half_life_days"]
        hl_lbl = f"Fit  (HL={hl:.0f}d)" if np.isfinite(hl) else "Fit (HL=∞)"

        fig.add_trace(go.Scatter(
            x=x, y=y_fit,
            name=hl_lbl, mode="lines",
            line=dict(color="#e67e22", width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Fit: %{y:.4f}<extra></extra>",
        ), row=3, col=1)

    fig.update_layout(
        title="Strategy Alpha Decay Analysis",
        height=580,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=80, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig


# =============================================================================
# Small helpers
# =============================================================================

def _add_regime_bands(fig: go.Figure, regime_series: pd.Series,
                      rows: list[int]) -> None:
    """Shade background with semi-transparent regime colour bands."""
    if regime_series.isna().all():
        return

    prev_regime = None
    band_start  = None

    for date, regime in regime_series.items():
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                color = _REGIME_COLORS.get(prev_regime, "#cccccc")
                for row in rows:
                    fig.add_vrect(
                        x0=str(band_start), x1=str(date),
                        fillcolor=color, opacity=0.08,
                        layer="below", line_width=0,
                        row=row, col=1,
                    )
            band_start  = date
            prev_regime = regime

    # close the final band
    if prev_regime is not None and band_start is not None:
        color = _REGIME_COLORS.get(prev_regime, "#cccccc")
        for row in rows:
            fig.add_vrect(
                x0=str(band_start), x1=str(regime_series.index[-1]),
                fillcolor=color, opacity=0.08,
                layer="below", line_width=0,
                row=row, col=1,
            )


def _signal_badge(signal: str | None, confidence: float | None) -> None:
    """Render a glowing terminal recommendation card with confidence gauge."""
    if signal is None:
        st.info("Ensemble unavailable — install xgboost + shap")
        return

    _GLOW = {
        "BUY_GOLD":   ("#f1c40f", "0 0 18px rgba(241,196,15,0.55)"),
        "HOLD":       ("#8b949e", "0 0 18px rgba(139,148,158,0.35)"),
        "BUY_SILVER": ("#58a6ff", "0 0 18px rgba(88,166,255,0.55)"),
    }
    color, shadow = _GLOW.get(signal, ("#8b949e", "none"))
    label = {"BUY_GOLD": "BUY GOLD", "HOLD": "HOLD", "BUY_SILVER": "BUY SILVER"}.get(signal, signal)
    conf_pct = round((confidence or 0) * 100, 1)

    st.markdown(
        f"""
        <div style="
            background:#0d1117;
            border:2px solid {color};
            box-shadow:{shadow};
            padding:24px 32px;
            border-radius:12px;
            text-align:center;
            margin-bottom:14px;
        ">
            <div style="
                font-family:'Courier New',monospace;
                font-size:2rem;
                font-weight:800;
                color:{color};
                letter-spacing:3px;
            ">
                {label}
            </div>
            <div style="font-size:0.88rem; color:#8b949e; margin-top:6px; font-family:'Courier New',monospace;">
                confidence: <span style="color:{color}; font-weight:600;">{conf_pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(confidence or 0.0))


def _sub_signal_table(last_row: pd.Series) -> pd.DataFrame:
    """Build a small DataFrame showing sub-signal breakdown for the latest bar."""
    rows = []
    for prefix, label in [("mr", "Mean Reversion"), ("mom", "Momentum"), ("bo", "Breakout")]:
        sig_col  = f"{prefix}_signal"
        conf_col = f"{prefix}_confidence"
        if sig_col in last_row.index:
            sig  = float(last_row[sig_col])
            conf = float(last_row[conf_col])
            direction = "BUY_SILVER" if sig > 0.1 else ("BUY_GOLD" if sig < -0.1 else "HOLD")
            rows.append({
                "Model"      : label,
                "Signal"     : f"{sig:+.3f}",
                "Direction"  : direction,
                "Confidence" : f"{conf:.1%}",
            })
    return pd.DataFrame(rows)


# =============================================================================
# Page layout
# =============================================================================

def main():
    st.set_page_config(
        page_title="Silver Lining — GSR Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Dark theme CSS ────────────────────────────────────────────────────────
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif !important;
                background-color: #0d1117 !important;
                color: #c9d1d9 !important;
            }
            .main .block-container { background-color: #0d1117; padding-top: 0.5rem; }
            #MainMenu, footer, header { visibility: hidden; }
            [data-testid="metric-container"] {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 14px 18px;
            }
            [data-testid="stMetricValue"] { color: #e6edf3 !important; }
            [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
            [data-baseweb="tab-list"] {
                background-color: #161b22 !important;
                border-radius: 8px;
                padding: 4px;
                gap: 4px;
                border: 1px solid #30363d;
            }
            [data-baseweb="tab"] {
                background-color: transparent !important;
                color: #8b949e !important;
                border-radius: 6px !important;
                font-family: 'Courier New', monospace !important;
                font-size: 0.82rem;
                border: none !important;
            }
            [aria-selected="true"][data-baseweb="tab"] {
                background-color: #21262d !important;
                color: #58a6ff !important;
                border: 1px solid #30363d !important;
            }
            [data-testid="stSidebar"] {
                background-color: #0d1117;
                border-right: 1px solid #21262d;
            }
            [data-testid="baseButton-secondary"] {
                background: #21262d !important;
                color: #c9d1d9 !important;
                border: 1px solid #30363d !important;
            }
            hr { border-color: #21262d !important; }
            [data-testid="stDataFrame"] { background: #161b22 !important; }
        </style>
    """, unsafe_allow_html=True)

    # ── Header bar ────────────────────────────────────────────────────────────
    from datetime import date as _date
    _hmm_dot   = "color:#3fb950;" if _HMM_OK      else "color:#f85149;"
    _ens_dot   = "color:#3fb950;" if _ENSEMBLE_OK  else "color:#f85149;"
    _hmm_sym   = "●" if _HMM_OK      else "○"
    _ens_sym   = "●" if _ENSEMBLE_OK  else "○"
    st.markdown(f"""
        <div style="
            display:flex; align-items:center; justify-content:space-between;
            background:#161b22; border:1px solid #30363d; border-radius:10px;
            padding:14px 24px; margin-bottom:18px;
        ">
            <div style="font-family:'Courier New',monospace; font-size:1.45rem;
                        font-weight:700; color:#58a6ff; letter-spacing:3px;">
                ◆ SILVER LINING
            </div>
            <div style="font-size:0.85rem; color:#8b949e; font-family:'Inter',sans-serif;">
                {_date.today().strftime('%A, %d %B %Y')}
            </div>
            <div style="font-size:0.82rem; font-family:'Courier New',monospace; display:flex; gap:18px;">
                <span style="{_hmm_dot}">{_hmm_sym} HMM</span>
                <span style="{_ens_dot}">{_ens_sym} Ensemble</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Controls")

        st.subheader("Live Data")
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.subheader("Backtest Parameters")
        spread   = st.slider("Spread (bps)",   0.0, 10.0, 2.0,  0.5) / 10_000
        slippage = st.slider("Slippage (bps)", 0.0,  5.0, 1.0,  0.5) / 10_000

        st.subheader("Walk-Forward Parameters")
        wf_window = st.slider("Window size (days)", 30, 120, 60, 10)
        wf_step   = st.slider("Step size (days)",    5,  40, 20,  5)
        run_wf    = st.button("▶ Run Walk-Forward", use_container_width=True)

        st.divider()
        st.caption("Pipeline status")
        st.write(f"HMM regime detection : {'✅' if _HMM_OK else '❌ hmmlearn missing'}")
        st.write(f"Ensemble (XGBoost)   : {'✅' if _ENSEMBLE_OK else '❌ xgboost/shap missing'}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    with st.spinner("Loading…"):
        df, capital, rets, metrics, warns = _build_pipeline(spread, slippage)

    for w in warns:
        st.warning(w)

    last = df.iloc[-1]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_signal, tab_alpha, tab_regime = st.tabs([
        "📈 Market Overview",
        "🎯 Live Signal",
        "📊 Alpha Decay",
        "🔄 Regime Analysis",
    ])

    # =========================================================================
    # Tab 1 — Market Overview
    # =========================================================================
    with tab_overview:
        st.subheader("Live Market Snapshot")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Gold (USD)",  f"${last['gold_close']:,.0f}")
        c2.metric("Silver (USD)", f"${last['silver_close']:.2f}")
        c3.metric("GSR",          f"{last['gsr']:.2f}")
        c4.metric("GSR z-score (30d)", f"{last['gsr_zscore_30']:+.2f}")
        _regime_val = last.get("regime")
        current_regime = _regime_val if pd.notna(_regime_val) else "—"
        c5.metric("Current Regime", str(current_regime).replace("_", " ").title())

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Backtest Metrics")
            m_cols = st.columns(5)
            for i, (k, v) in enumerate(metrics.items()):
                m_cols[i].metric(
                    k.replace("_", " ").title(),
                    f"{v:.2%}" if k in ("cagr", "max_drawdown", "hit_rate") else f"{v:.2f}",
                )

        st.plotly_chart(_chart_equity(capital, df), use_container_width=True)
        st.plotly_chart(_chart_prices(df),          use_container_width=True)

    # =========================================================================
    # Tab 2 — Live Signal
    # =========================================================================
    with tab_signal:
        st.subheader(f"Recommendation as of {df.index[-1].date()}")

        col_rec, col_sub = st.columns([1, 2])

        with col_rec:
            _signal_badge(
                last.get("ensemble_signal"),
                last.get("ensemble_confidence"),
            )

            st.subheader("Current Regime")
            regime_color = _REGIME_COLORS.get(str(current_regime), "#cccccc")
            st.markdown(
                f'<div style="background:{regime_color};padding:12px 20px;'
                f'border-radius:8px;text-align:center;font-weight:700;'
                f'font-size:1.1rem;">{str(current_regime).replace("_"," ").upper()}</div>',
                unsafe_allow_html=True,
            )

        with col_sub:
            st.subheader("Sub-Signal Breakdown")
            sub_df = _sub_signal_table(last)
            if not sub_df.empty:
                st.dataframe(sub_df, use_container_width=True, hide_index=True)
            else:
                st.info("Strategy signals not available.")

        st.divider()

        # SHAP waterfall
        shap_cols = [c for c in last.index if c.startswith("shap_")]
        if shap_cols and pd.notna(last[shap_cols[0]]):
            st.plotly_chart(_chart_shap_waterfall(last), use_container_width=True)
            st.caption(
                "Each bar shows how much that feature pushed the model toward "
                "BUY_SILVER (+) or BUY_GOLD (−) for today's prediction."
            )
        else:
            st.info("SHAP explanations unavailable — install xgboost + shap.")

    # =========================================================================
    # Tab 3 — Alpha Decay
    # =========================================================================
    with tab_alpha:
        st.subheader("Strategy Alpha Decay Analysis")

        wf_state = st.session_state.get("wf_result", None)

        if run_wf or wf_state is not None:
            if run_wf:
                with st.spinner("Running walk-forward (this may take a minute)…"):
                    fold_summary, decay_params, agg = _build_walk_forward(
                        df.to_json(),
                        window_size=wf_window,
                        step_size=wf_step,
                        spread=spread,
                        slippage=slippage,
                    )
                st.session_state["wf_result"] = (fold_summary, decay_params, agg)
            else:
                fold_summary, decay_params, agg = wf_state

            # Alpha decay chart
            st.plotly_chart(
                _chart_alpha_decay(fold_summary, decay_params),
                use_container_width=True,
            )

            # Half-life callout
            hl = decay_params.get("half_life_days", None)
            if hl is not None:
                hl_str = f"{hl:.0f} days" if np.isfinite(hl) else "∞ (no decay detected)"
                st.info(f"**Alpha half-life:** {hl_str}  |  "
                        f"Initial IC: {decay_params.get('alpha_initial', 0):+.4f}  |  "
                        f"Alpha floor: {decay_params.get('alpha_floor', 0):+.4f}")

            st.divider()

            col_folds, col_agg = st.columns([2, 1])
            with col_folds:
                st.subheader("Walk-Forward Window Results")
                display_cols = [c for c in
                    ["window_idx", "window_start", "window_end", "sharpe",
                     "cagr", "max_drawdown", "hit_rate", "calmar", "ic_composite"]
                    if c in fold_summary.columns]
                st.dataframe(
                    fold_summary[display_cols].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_agg:
                st.subheader("Aggregate Summary")
                st.dataframe(agg, use_container_width=True)
        else:
            st.info("Click **▶ Run Walk-Forward** in the sidebar to compute "
                    "rolling performance and alpha decay.")

    # =========================================================================
    # Tab 4 — Regime Analysis
    # =========================================================================
    with tab_regime:
        st.subheader("Regime Analysis")

        if "regime" not in df.columns or df["regime"].isna().all():
            st.warning("Regime data unavailable — install hmmlearn and refresh.")
        else:
            # Distribution
            dist = df["regime"].value_counts()
            col_dist, col_stats = st.columns([1, 1])

            with col_dist:
                fig_dist = px.bar(
                    x=dist.index,
                    y=dist.values,
                    color=dist.index,
                    color_discrete_map=_REGIME_COLORS,
                    labels={"x": "Regime", "y": "Days"},
                    title="Regime Distribution (total days)",
                )
                fig_dist.update_layout(
                    showlegend=False, height=300,
                    margin=dict(l=40, r=20, t=50, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_stats:
                st.subheader("Regime Statistics")
                regime_stats = []
                for r in sorted(df["regime"].dropna().unique()):
                    mask = df["regime"] == r
                    days = mask.sum()
                    pct  = days / len(df)
                    regime_stats.append({
                        "Regime"    : r.replace("_", " ").title(),
                        "Days"      : days,
                        "% of time" : f"{pct:.1%}",
                        "Avg GSR"   : f"{df.loc[mask, 'gsr'].mean():.2f}",
                        "Avg Vol"   : f"{df.loc[mask, 'gold_vol_20'].mean():.4f}",
                    })
                st.dataframe(pd.DataFrame(regime_stats),
                             use_container_width=True, hide_index=True)

            st.divider()

            # Regime timeline
            st.plotly_chart(_chart_regime_timeline(df), use_container_width=True)

            st.divider()

            # Transition heatmap
            st.plotly_chart(_chart_transition_heatmap(df), use_container_width=True)
            st.caption(
                "Empirical transition probabilities computed from regime labels. "
                "Each cell = P(column regime | row regime) over the full history."
            )


# =============================================================================
if __name__ == "__main__":
    main()
