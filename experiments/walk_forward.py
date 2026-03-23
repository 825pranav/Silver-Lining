# experiments/walk_forward.py
#
# Rolling walk-forward validation + strategy alpha decay analysis.
#
# Uses existing project primitives directly:
#   backtesting/simulator._run_fold()   P&L engine, no-lookahead guaranteed
#   backtesting/metrics.compute_all()   Sharpe / CAGR / drawdown / hit-rate / Calmar
#   scipy.stats.spearmanr               Information Coefficient (IC) per window
#   scipy.optimize.curve_fit            Exponential decay fit for half-life
#
# Alpha Decay Methodology
# -----------------------
# Information Coefficient (IC) = Spearman rank correlation between
#   signal(t)  and  -(5-day forward GSR pct change)(t)
# Sign convention: high signal → expect GSR to FALL (silver outperforms)
#   → IC > 0 means the signal has positive predictive power.
#
# Across successive rolling windows we fit:
#   IC(t) = a · exp(−b · t) + c       t = window index
#   half_life = ln(2) / b  [windows] × step_size  [days/window]
#
# Output DataFrames
# -----------------
# fold_summary  : one row per rolling window  — all simulator metrics + IC columns
# decay_curve   : time-indexed — rolling Sharpe, rolling hit-rate, IC, normalized alpha
# (scalar)      : alpha_half_life_days
#
# Dependencies: numpy, pandas, scipy  (all installed)
#               matplotlib            (installed, used for optional plot)

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit, OptimizeWarning

# ── Project imports (works whether run from repo root or experiments/) ──────
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtesting.simulator import _run_fold
from backtesting.metrics   import compute_all, sharpe as _sharpe, hit_rate as _hit_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FWD_WINDOW   = 5     # days for forward GSR return (matches ensemble.py)
PERIODS_PER_YEAR = 252

# Signals available from strategy_models (used for IC if present)
_OPTIONAL_SIGNAL_COLS = ["mr_signal", "mom_signal", "bo_signal"]
# Always-present signal proxy from gsr_features
_FALLBACK_SIGNAL_COL  = "gsr_zscore_30"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward_gsr(df: pd.DataFrame, fwd_window: int = FWD_WINDOW) -> pd.Series:
    """5-day forward GSR pct change, shifted so index aligns with signal date."""
    return df["gsr"].pct_change(fwd_window).shift(-fwd_window)


def _window_ic(signal: pd.Series, fwd_gsr: pd.Series) -> float:
    """
    Spearman IC between signal and *negative* forward GSR return.

    Negative sign: high signal → expect GSR to fall (silver outperforms).
    Returns NaN if fewer than 10 valid pairs.
    """
    aligned = pd.concat([signal, -fwd_gsr], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    corr, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(corr)


def _available_signals(df: pd.DataFrame) -> list[str]:
    """Return signal columns present in df, falling back to gsr_zscore_30."""
    present = [c for c in _OPTIONAL_SIGNAL_COLS if c in df.columns]
    if not present:
        if _FALLBACK_SIGNAL_COL in df.columns:
            present = [_FALLBACK_SIGNAL_COL]
        else:
            raise ValueError(
                f"No signal columns found. Need at least one of "
                f"{_OPTIONAL_SIGNAL_COLS} or '{_FALLBACK_SIGNAL_COL}'."
            )
    return present


def _composite_ic(df_window: pd.DataFrame, signal_cols: list[str],
                  fwd_gsr: pd.Series) -> dict:
    """
    Compute IC for each signal column within *df_window*.
    Returns dict with ic_{col} keys and composite_ic (unweighted mean).
    """
    result = {}
    fwd_window_slice = fwd_gsr.reindex(df_window.index)
    ics = []
    for col in signal_cols:
        ic = _window_ic(df_window[col], fwd_window_slice)
        result[f"ic_{col}"] = round(ic, 6) if not np.isnan(ic) else np.nan
        if not np.isnan(ic):
            ics.append(ic)
    result["ic_composite"] = float(np.mean(ics)) if ics else np.nan
    return result


# ---------------------------------------------------------------------------
# Exponential decay fitting
# ---------------------------------------------------------------------------

def _exp_decay(t, a, b, c):
    """Exponential decay model: f(t) = a·exp(−b·t) + c."""
    return a * np.exp(-b * t) + c


def fit_exponential_decay(
    values: pd.Series,
    step_size_days: int,
) -> dict:
    """
    Fit IC(t) = a·exp(−b·t) + c to a window-indexed series of IC values.

    Parameters
    ----------
    values         : Series of IC (or Sharpe) values, one per rolling window.
    step_size_days : Days between successive windows (converts half-life to days).

    Returns
    -------
    dict with keys:
        alpha_initial      float   a + c  (IC at t=0)
        alpha_floor        float   c      (asymptotic IC)
        decay_rate         float   b      (per-window decay rate)
        half_life_windows  float   ln(2)/b  (NaN if b ≤ 0)
        half_life_days     float   half_life_windows × step_size_days
        fit_success        bool
    """
    t = np.arange(len(values), dtype=float)
    y = values.values.astype(float)
    finite_mask = np.isfinite(y)

    if finite_mask.sum() < 4:
        return dict(alpha_initial=np.nan, alpha_floor=np.nan,
                    decay_rate=np.nan, half_life_windows=np.nan,
                    half_life_days=np.nan, fit_success=False)

    t_fit, y_fit = t[finite_mask], y[finite_mask]

    # Initial guess: a = spread of y, b = 0.1, c = minimum y
    a0  = float(y_fit.max() - y_fit.min())
    c0  = float(y_fit.min())
    p0  = [a0, 0.1, c0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _exp_decay, t_fit, y_fit,
                p0=p0,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),  # b ≥ 0
                maxfev=5000,
            )
        a, b, c = popt
        if b > 1e-9:
            hl_windows = float(np.log(2) / b)
            hl_days    = hl_windows * step_size_days
        else:
            hl_windows = np.inf
            hl_days    = np.inf

        return dict(
            alpha_initial     = round(a + c, 6),
            alpha_floor       = round(c, 6),
            decay_rate        = round(b, 6),
            half_life_windows = round(hl_windows, 2),
            half_life_days    = round(hl_days, 1),
            fit_success       = True,
        )

    except Exception:
        return dict(alpha_initial=np.nan, alpha_floor=np.nan,
                    decay_rate=np.nan, half_life_windows=np.nan,
                    half_life_days=np.nan, fit_success=False)


# ---------------------------------------------------------------------------
# Rolling walk-forward
# ---------------------------------------------------------------------------

def rolling_walk_forward(
    df: pd.DataFrame,
    window_size: int = 60,
    step_size: int   = 20,
    purge_days: int  = 5,
    spread: float    = 0.0002,
    slippage: float  = 0.0001,
) -> pd.DataFrame:
    """
    Slide a fixed-size window across *df*, run the simulator on each slice,
    and return performance metrics + IC for every window.

    Window layout
    -------------
    [────── window_size bars ──────]
                                    [purge_days gap]
                                                    [next window starts]

    The purge gap between adjacent windows prevents leakage from any rolling
    features with a lookback ≤ purge_days (gsr_slope uses 5-day diff).

    Parameters
    ----------
    df          : Full feature DataFrame with DatetimeIndex.
    window_size : Bars per test window (default 60 ≈ 3 months).
    step_size   : Bars between window starts (default 20 ≈ 1 month).
    purge_days  : Gap between consecutive windows (default 5).
    spread      : One-way bid-ask spread.
    slippage    : One-way slippage.

    Returns
    -------
    fold_summary : DataFrame, one row per window with columns:
                   window_idx, window_start, window_end, n_bars,
                   sharpe, cagr, max_drawdown, hit_rate, calmar,
                   ic_{signal}, ic_composite.
    """
    n = len(df)
    signal_cols = _available_signals(df)
    fwd_gsr     = _forward_gsr(df)

    records = []
    window_idx = 0
    start = 0

    print(f"\n🔄 Rolling Walk-Forward | window={window_size}d | step={step_size}d | "
          f"purge={purge_days}d | signals={signal_cols}")
    print("-" * 72)

    while start + window_size <= n:
        end      = start + window_size
        df_slice = df.iloc[start:end]

        # ── Simulator metrics ────────────────────────────────────────────────
        cap, rets, _ = _run_fold(df_slice, spread=spread, slippage=slippage)
        m = compute_all(cap, rets)

        # ── IC metrics ───────────────────────────────────────────────────────
        ic_dict = _composite_ic(df_slice, signal_cols, fwd_gsr)

        record = {
            "window_idx"   : window_idx,
            "window_start" : df_slice.index[0].date(),
            "window_end"   : df_slice.index[-1].date(),
            "n_bars"       : len(df_slice),
            **m,
            **ic_dict,
        }
        records.append(record)

        print(
            f"  [{window_idx:>2}] {df_slice.index[0].date()} → {df_slice.index[-1].date()} | "
            f"Sharpe={m['sharpe']:+.2f}  HitRate={m['hit_rate']:.2%}  "
            f"IC={ic_dict['ic_composite']:+.3f}"
        )

        start      += step_size + purge_days
        window_idx += 1

    print("-" * 72)
    fold_summary = pd.DataFrame(records)
    print(f"  Total windows: {len(fold_summary)}")
    return fold_summary


# ---------------------------------------------------------------------------
# Alpha decay analysis
# ---------------------------------------------------------------------------

def alpha_decay_analysis(
    df: pd.DataFrame,
    window_size: int = 60,
    step_size: int   = 20,
    purge_days: int  = 5,
    spread: float    = 0.0002,
    slippage: float  = 0.0001,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute rolling Sharpe, rolling hit rate, and IC per window, then fit an
    exponential decay model to quantify how quickly the strategy's alpha erodes.

    Parameters
    ----------
    (same as rolling_walk_forward)

    Returns
    -------
    decay_curve : DataFrame indexed by window_start with columns:
                    window_idx, rolling_sharpe, rolling_hit_rate,
                    ic_composite, ic_{signal...},
                    normalized_alpha   (IC / IC[0], rescaled to [0,1] for plotting)
    decay_params : dict from fit_exponential_decay — includes half_life_days.
    """
    fold_summary = rolling_walk_forward(
        df, window_size=window_size, step_size=step_size,
        purge_days=purge_days, spread=spread, slippage=slippage,
    )

    if fold_summary.empty:
        raise ValueError("No windows produced — reduce window_size or step_size.")

    # ── Build decay curve ────────────────────────────────────────────────────
    ic_cols = [c for c in fold_summary.columns if c.startswith("ic_")]

    decay_curve = fold_summary[
        ["window_idx", "window_start", "window_end", "sharpe", "hit_rate"] + ic_cols
    ].copy()
    decay_curve = decay_curve.rename(columns={
        "sharpe"   : "rolling_sharpe",
        "hit_rate" : "rolling_hit_rate",
    })
    decay_curve = decay_curve.set_index("window_start")
    decay_curve.index = pd.to_datetime(decay_curve.index)

    # Normalized alpha: IC relative to window 0 (first window's IC = 1.0)
    ic0 = decay_curve["ic_composite"].iloc[0]
    if pd.notna(ic0) and ic0 != 0:
        decay_curve["normalized_alpha"] = decay_curve["ic_composite"] / abs(ic0)
    else:
        decay_curve["normalized_alpha"] = np.nan

    # ── Fit exponential decay to IC series ──────────────────────────────────
    decay_params = fit_exponential_decay(
        decay_curve["ic_composite"],
        step_size_days=step_size,
    )

    # ── Print decay summary ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  ALPHA DECAY ANALYSIS")
    print("=" * 64)
    print(f"\n  IC series ({len(decay_curve)} windows):")
    print(f"    Mean IC        : {decay_curve['ic_composite'].mean():+.4f}")
    print(f"    Std IC         : {decay_curve['ic_composite'].std():.4f}")
    print(f"    Min IC         : {decay_curve['ic_composite'].min():+.4f}")
    print(f"    Max IC         : {decay_curve['ic_composite'].max():+.4f}")

    print(f"\n  Exponential decay fit  (IC = a·exp(−b·t) + c):")
    if decay_params["fit_success"]:
        print(f"    Initial alpha  : {decay_params['alpha_initial']:+.4f}")
        print(f"    Alpha floor    : {decay_params['alpha_floor']:+.4f}")
        print(f"    Decay rate (b) : {decay_params['decay_rate']:.6f} per window")
        hl = decay_params['half_life_days']
        hl_str = f"{hl:.1f} days" if np.isfinite(hl) else "∞  (no decay detected)"
        print(f"    Half-life      : {hl_str}")
    else:
        print("    Fit failed — insufficient data or non-decaying IC series.")

    print(f"\n  Rolling performance summary:")
    print(f"    Mean Sharpe    : {decay_curve['rolling_sharpe'].mean():+.4f}")
    print(f"    Mean Hit Rate  : {decay_curve['rolling_hit_rate'].mean():.2%}")
    print("=" * 64)

    return decay_curve, decay_params


# ---------------------------------------------------------------------------
# Aggregated summary
# ---------------------------------------------------------------------------

def aggregate_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all rolling windows into a single summary DataFrame with
    mean, std, min, max, and consistency ratio for each metric.

    Consistency ratio = fraction of windows with Sharpe > 0.
    A ratio > 0.6 suggests alpha is stable across time.

    Parameters
    ----------
    fold_summary : Output of rolling_walk_forward().

    Returns
    -------
    summary : DataFrame with one row per metric and columns:
              mean, std, min, max, consistency (where applicable).
    """
    numeric_cols = ["sharpe", "cagr", "max_drawdown", "hit_rate", "calmar", "ic_composite"]
    numeric_cols = [c for c in numeric_cols if c in fold_summary.columns]

    agg = fold_summary[numeric_cols].agg(["mean", "std", "min", "max"]).T
    agg.index.name = "metric"

    # Consistency: % windows where Sharpe > 0
    if "sharpe" in fold_summary.columns:
        agg.loc["sharpe", "pct_positive_windows"] = (
            (fold_summary["sharpe"] > 0).mean()
        )
    if "ic_composite" in fold_summary.columns:
        agg.loc["ic_composite", "pct_positive_windows"] = (
            (fold_summary["ic_composite"] > 0).mean()
        )

    return agg.round(4)


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_full_analysis(
    df: pd.DataFrame  = None,
    data_path: str    = None,
    window_size: int  = 60,
    step_size: int    = 20,
    purge_days: int   = 5,
    spread: float     = 0.0002,
    slippage: float   = 0.0001,
    save_outputs: bool = True,
    plot: bool         = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Full walk-forward + alpha decay analysis pipeline.

    Parameters
    ----------
    df          : Feature DataFrame (DatetimeIndex). Loaded from disk if None.
    data_path   : Explicit path to load df from (overrides default search).
    window_size : Bars per rolling test window.
    step_size   : Bars between window starts.
    purge_days  : Gap between consecutive windows.
    spread      : One-way spread cost.
    slippage    : One-way slippage.
    save_outputs: Write CSVs to data/.
    plot        : Save a decay curve PNG to experiments/.

    Returns
    -------
    fold_summary  : Per-window metrics + IC.
    decay_curve   : Rolling Sharpe / hit-rate / IC per window (time-indexed).
    agg_summary   : Aggregated statistics across all windows.
    decay_params  : Exponential decay fit parameters including half_life_days.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    if df is None:
        candidates = [
            data_path,
            os.path.join(_PROJECT_ROOT, "data", "features_with_signals.csv"),
            os.path.join(_PROJECT_ROOT, "data", "features_with_regimes.csv"),
            os.path.join(_PROJECT_ROOT, "data", "features.csv"),
        ]
        loaded = False
        for path in candidates:
            if path and os.path.exists(path):
                print(f"🚀 Loading {path}...")
                df = pd.read_csv(path, parse_dates=["Date"])
                df.set_index("Date", inplace=True)
                print(f"   {len(df)} rows | {len(df.columns)} columns")
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(
                "No feature file found. Run the full pipeline first:\n"
                "  price_fetcher → gsr_features → momentum_features → "
                "volatility_features → (regime_detection → strategy_models)"
            )

    # ── Run analysis ─────────────────────────────────────────────────────────
    decay_curve, decay_params = alpha_decay_analysis(
        df,
        window_size=window_size,
        step_size=step_size,
        purge_days=purge_days,
        spread=spread,
        slippage=slippage,
    )

    # Reconstruct fold_summary from decay_curve for aggregate step
    fold_summary = rolling_walk_forward(
        df,
        window_size=window_size,
        step_size=step_size,
        purge_days=purge_days,
        spread=spread,
        slippage=slippage,
    )

    agg_summary = aggregate_summary(fold_summary)

    print("\n📊 Aggregate Summary Across All Windows:")
    print(agg_summary.to_string())

    # ── Save outputs ─────────────────────────────────────────────────────────
    if save_outputs:
        out_dir = os.path.join(_PROJECT_ROOT, "data")
        os.makedirs(out_dir, exist_ok=True)

        fs_path = os.path.join(out_dir, "walk_forward_folds.csv")
        dc_path = os.path.join(out_dir, "alpha_decay_curve.csv")
        as_path = os.path.join(out_dir, "walk_forward_summary.csv")

        fold_summary.to_csv(fs_path, index=False)
        decay_curve.to_csv(dc_path)
        agg_summary.to_csv(as_path)

        print(f"\n💾 Saved:")
        print(f"   {fs_path}")
        print(f"   {dc_path}")
        print(f"   {as_path}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if plot:
        _plot_decay_curve(decay_curve, decay_params)

    return fold_summary, decay_curve, agg_summary, decay_params


# ---------------------------------------------------------------------------
# Optional matplotlib plot
# ---------------------------------------------------------------------------

def _plot_decay_curve(decay_curve: pd.DataFrame, decay_params: dict) -> None:
    """Save a 3-panel decay curve PNG to experiments/."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("Strategy Alpha Decay Analysis", fontsize=13, fontweight="bold")

        x = decay_curve.index

        # Panel 1: Rolling Sharpe
        axes[0].bar(x, decay_curve["rolling_sharpe"],
                    width=15, color=["#2ecc71" if v > 0 else "#e74c3c"
                                     for v in decay_curve["rolling_sharpe"]],
                    alpha=0.8)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_ylabel("Rolling Sharpe")
        axes[0].set_title("Rolling Sharpe per Window")

        # Panel 2: Rolling Hit Rate
        axes[1].plot(x, decay_curve["rolling_hit_rate"], marker="o",
                     markersize=4, color="#3498db", linewidth=1.5)
        axes[1].axhline(0.5, color="grey", linewidth=0.8, linestyle="--",
                        label="50% baseline")
        axes[1].set_ylabel("Hit Rate")
        axes[1].set_title("Rolling Hit Rate per Window")
        axes[1].legend(fontsize=8)

        # Panel 3: IC + exponential decay fit
        ic_vals = decay_curve["ic_composite"].values
        t       = np.arange(len(ic_vals), dtype=float)

        axes[2].plot(x, ic_vals, marker="s", markersize=4,
                     color="#9b59b6", linewidth=1.5, label="IC (composite)")
        axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")

        if decay_params.get("fit_success"):
            a  = decay_params["alpha_initial"] - decay_params["alpha_floor"]
            b  = decay_params["decay_rate"]
            c  = decay_params["alpha_floor"]
            y_fit = _exp_decay(t, a, b, c)
            axes[2].plot(x, y_fit, color="#e67e22", linewidth=2,
                         linestyle="--", label="Exp decay fit")
            hl = decay_params["half_life_days"]
            hl_label = f"Half-life: {hl:.0f}d" if np.isfinite(hl) else "Half-life: ∞"
            axes[2].set_title(f"IC Decay Curve  ({hl_label})")
        else:
            axes[2].set_title("IC Decay Curve  (fit failed)")

        axes[2].set_ylabel("Information Coefficient")
        axes[2].legend(fontsize=8)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_dir = os.path.join(_PROJECT_ROOT, "experiments")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "alpha_decay_curve.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Plot saved to {out_path}")

    except Exception as e:
        print(f"⚠️  Plot skipped: {e}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fold_summary, decay_curve, agg_summary, decay_params = run_full_analysis(
        window_size=60,
        step_size=20,
        purge_days=5,
        plot=True,
    )

    print("\n--- Decay Curve (all windows) ---")
    print(decay_curve[["window_idx", "rolling_sharpe",
                        "rolling_hit_rate", "ic_composite",
                        "normalized_alpha"]].to_string())

    print(f"\n⏱️  Alpha Half-Life: {decay_params.get('half_life_days', 'N/A')} days")
