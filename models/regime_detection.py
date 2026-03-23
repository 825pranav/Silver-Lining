# models/regime_detection.py
#
# Fits a GaussianHMM on the combined feature set produced by:
#   features/gsr_features.py → features/momentum_features.py → features/volatility_features.py
#
# Run order: price_fetcher → gsr_features → momentum_features → volatility_features → regime_detection
#
# Requires: pip install hmmlearn scikit-learn

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Exact column names consumed from each upstream feature module
# ---------------------------------------------------------------------------

# features/gsr_features.py
_GSR_COLS = [
    "gsr_zscore_30",   # GSR deviation from 30-day mean (crisis detector)
    "gsr_slope",       # 5-day rate of change in GSR (trend direction)
    "vol_ratio",       # gold_vol_20 / silver_vol_20 (relative vol)
]

# features/momentum_features.py
_MOMENTUM_COLS = [
    "gold_ema_slope",    # EMA(10) - EMA(30) for gold
    "silver_ema_slope",  # EMA(10) - EMA(30) for silver
    "gsr_ema_slope",     # EMA(10) - EMA(30) for GSR ratio
    "gold_macd_hist",    # MACD histogram for gold (momentum conviction)
    "rel_strength_20",   # gold_ret_20 - silver_ret_20 (relative performance)
]

# features/volatility_features.py
_VOLATILITY_COLS = [
    "gold_atr_pct",     # ATR normalised by price (regime-invariant vol level)
    "gold_garch_vol",   # GARCH(1,1) conditional vol for gold
    "garch_vol_ratio",  # silver_garch_vol / gold_garch_vol
]

REGIME_FEATURES = _GSR_COLS + _MOMENTUM_COLS + _VOLATILITY_COLS

# ---------------------------------------------------------------------------
# Regime labelling logic
# ---------------------------------------------------------------------------

def _assign_regimes(state_means: pd.DataFrame) -> dict:
    """
    Greedy priority assignment of regime labels to HMM state indices.

    Economic intuition:
      crisis      – gold flight-to-safety: high vol + elevated GSR z-score
      range_bound – quiet market: lowest vol, weak momentum
      risk_on     – metals broadly bid, silver outperforming: low GSR z-score
      trending    – remainder (directional momentum, moderate vol)

    Returns dict {state_int: regime_str}.
    """
    assignment = {}
    used = set()

    vol_col = "gold_atr_pct"
    gsr_col = "gsr_zscore_30"

    # 1. crisis: highest combined vol + GSR z-score rank
    crisis_score = state_means[vol_col].rank() + state_means[gsr_col].rank()
    crisis_state = int(crisis_score.idxmax())
    assignment[crisis_state] = "crisis"
    used.add(crisis_state)

    # 2. range_bound: lowest vol among remaining states
    remaining = [s for s in state_means.index if s not in used]
    range_state = int(state_means.loc[remaining, vol_col].idxmin())
    assignment[range_state] = "range_bound"
    used.add(range_state)

    # 3. risk_on: lowest GSR z-score among remaining (silver outperforming gold)
    remaining = [s for s in state_means.index if s not in used]
    risk_state = int(state_means.loc[remaining, gsr_col].idxmin())
    assignment[risk_state] = "risk_on"
    used.add(risk_state)

    # 4. trending: the last unassigned state
    remaining = [s for s in state_means.index if s not in used]
    assignment[remaining[0]] = "trending"

    return assignment


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def fit_and_label(
    df: pd.DataFrame,
    n_states: int = 4,
    n_restarts: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit a GaussianHMM on the combined feature set and append a regime column.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex containing all columns produced by
         gsr_features.py, momentum_features.py, and volatility_features.py.
    n_states : Number of hidden states (3 or 4). Default 4.
    n_restarts : Random restarts; the run with the best log-likelihood is kept.
    random_state : Base random seed for reproducibility.

    Returns
    -------
    df_out : Copy of df with two new columns appended:
               - 'regime'       (str)  : human-readable label
               - 'regime_state' (int)  : raw HMM state index
             Rows that were NaN-dropped before fitting will have NaN in these columns.

    Side effects
    ------------
    Prints the regime transition probability matrix and per-state feature means
    to stdout so downstream callers can audit the model without needing to
    handle extra return values.
    """
    # Validate columns
    missing = [c for c in REGIME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Ensure you have run gsr_features, momentum_features, and "
            f"volatility_features in order before calling fit_and_label()."
        )

    X_raw = df[REGIME_FEATURES].copy().dropna()
    if len(X_raw) < n_states * 10:
        raise ValueError(
            f"Only {len(X_raw)} complete rows — need at least {n_states * 10} "
            f"to fit a {n_states}-state HMM reliably."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Multiple restarts; keep the model with the best log-likelihood
    best_model = None
    best_score = -np.inf

    print(f"🔍 Fitting GaussianHMM ({n_states} states, {n_restarts} restarts)...")
    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            tol=1e-4,
            random_state=random_state + i,
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed. Check your data for degenerate columns.")

    hidden_states = best_model.predict(X)

    # Unscaled means for interpretable labelling and display
    state_means = pd.DataFrame(
        scaler.inverse_transform(best_model.means_),
        columns=REGIME_FEATURES,
    )

    state_to_regime = _assign_regimes(state_means)

    # Build output columns aligned to the original df index (NaN where dropped)
    regime_state_series = pd.Series(hidden_states, index=X_raw.index, dtype=int)
    regime_label_series = regime_state_series.map(state_to_regime)

    df_out = df.copy()
    df_out["regime_state"] = regime_state_series
    df_out["regime"] = regime_label_series

    # ----- Print summary -----
    regime_labels = [state_to_regime[i] for i in range(n_states)]

    print("\n" + "=" * 64)
    print("  REGIME DETECTION SUMMARY")
    print("=" * 64)

    print(f"\n  Best log-likelihood : {best_score:.2f}")
    print(f"  Observations used   : {len(X_raw)}")

    print("\n📊 Transition Probability Matrix (row = from, col = to):")
    trans_df = pd.DataFrame(
        best_model.transmat_,
        index=regime_labels,
        columns=regime_labels,
    ).round(4)
    print(trans_df.to_string())

    print("\n📈 Per-State Feature Means (original scale):")
    means_display = state_means.copy()
    means_display.index = regime_labels
    means_display.index.name = "regime"
    print(means_display.round(6).to_string())

    print("\n🏷️  Regime Distribution:")
    dist = df_out["regime"].value_counts()
    for regime, count in dist.items():
        pct = 100 * count / dist.sum()
        print(f"    {regime:<15} {count:>4} days  ({pct:.1f}%)")

    print("=" * 64)

    return df_out


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    try:
        feature_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        feature_path = os.path.normpath(feature_path)

        print(f"🚀 Loading features from {feature_path}...")
        df = pd.read_csv(feature_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        print(f"   {len(df)} rows, {len(df.columns)} columns loaded.")

        df_out = fit_and_label(df, n_states=4, n_restarts=10)

        print("\n--- Latest Regime Snapshot (last 10 rows) ---")
        snap_cols = ["regime"] + _GSR_COLS[:2] + _MOMENTUM_COLS[:2] + _VOLATILITY_COLS[:1]
        print(df_out[snap_cols].tail(10).to_string())

        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "features_with_regimes.csv")
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path)
        print(f"\n💾 Saved to {out_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
