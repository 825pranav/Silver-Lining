# models/strategy_models.py
#
# Three regime-conditioned sub-models, each emitting a continuous signal
# (-1 = strong BUY_GOLD, 0 = HOLD, +1 = strong BUY_SILVER) and a
# confidence score [0, 1].
#
# Signal convention (used throughout the project):
#   signal > 0  →  lean BUY_SILVER  (GSR falling, silver outperforming)
#   signal < 0  →  lean BUY_GOLD    (GSR rising,  gold outperforming)
#   signal ≈ 0  →  HOLD
#
# Inputs (exact column names from upstream feature modules)
# ---------------------------------------------------------
#   gsr_features.py     : gsr, gsr_zscore_30, gsr_zscore_90, gsr_slope, vol_ratio
#   momentum_features.py: gsr_ema_slope, gold_ema_slope, silver_ema_slope,
#                         gold_macd_hist, rel_strength_20,
#                         gold_breakout, silver_breakout
#   volatility_features : gold_atr_pct, gold_vol_5, gold_vol_20
#   regime_detection.py : regime
#
# Requires: numpy, pandas (both already installed)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regime-conditioned signal multipliers
# Weights above 1.0 amplify a model's confidence in that regime;
# below 1.0 attenuate it.
# ---------------------------------------------------------------------------
_REGIME_WEIGHTS = {
    "mean_reversion": {
        "crisis":      1.5,   # extreme GSR dislocations → reversion bets pay off
        "range_bound": 1.3,   # quiet market → mean-reversion is reliable
        "risk_on":     0.8,
        "trending":    0.4,   # trend fights reversion — dampen hard
    },
    "momentum": {
        "trending":    1.5,
        "risk_on":     1.2,
        "crisis":      0.7,   # crisis moves are sharp but can reverse
        "range_bound": 0.4,
    },
    "breakout": {
        "trending":    1.4,
        "risk_on":     1.1,
        "crisis":      0.8,
        "range_bound": 0.3,   # false breakouts dominate in quiet markets
    },
}

_FALLBACK_WEIGHT = 1.0   # used if regime column is missing or unknown


def _regime_weight(regime_series: pd.Series, model_key: str) -> pd.Series:
    """Map each row's regime label to its multiplier for *model_key*."""
    weights = _REGIME_WEIGHTS[model_key]
    return regime_series.map(weights).fillna(_FALLBACK_WEIGHT)


def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score with *window*-day lookback.
    Using a rolling window instead of full-series normalisation prevents
    any forward-looking data leaking into the signal.
    """
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std().clip(lower=1e-9)
    return (series - m) / s


def _clip_confidence(raw: pd.Series) -> pd.Series:
    """Squash any value to [0, 1]."""
    return raw.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Sub-model 1 — Mean Reversion
# ---------------------------------------------------------------------------

def mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime-conditioned mean-reversion signal and confidence.

    Logic
    -----
    GSR stretched above its 30-day mean  →  silver undervalued  →  BUY_SILVER (+)
    GSR stretched below its 30-day mean  →  gold  undervalued  →  BUY_GOLD   (-)
    Confidence scales with z-score magnitude and is amplified when the
    slower 90-day z-score agrees with the 30-day z-score (both extremes).

    New columns
    -----------
    mr_signal      float  [-1, 1]
    mr_confidence  float  [ 0, 1]
    """
    required = ["gsr_zscore_30", "gsr_zscore_90", "gsr_slope", "regime"]
    _check_cols(df, required, "mean_reversion_signals")

    out = df.copy()

    z30 = out["gsr_zscore_30"]
    z90 = out["gsr_zscore_90"]

    # Raw signal: positive z-score → silver cheap → positive signal
    raw_signal = np.tanh(z30 * 0.8)

    # Agreement bonus: when 30d and 90d z-scores agree in direction,
    # magnitude of confidence gets an extra boost (up to +30%).
    agree = np.sign(z30) == np.sign(z90)
    agree_bonus = np.where(agree, 1.3, 0.85)

    # Confidence: normalised absolute z-score, boosted by agreement
    raw_conf = (z30.abs() / 3.0) * agree_bonus

    # Slope filter: if GSR is still accelerating in the stretched direction,
    # reversion hasn't started — reduce confidence
    slope_penalty = np.where(
        (np.sign(z30) == np.sign(out["gsr_slope"])),   # same direction
        0.7,
        1.0,
    )

    regime_mult = _regime_weight(out["regime"], "mean_reversion")

    out["mr_signal"]     = raw_signal.astype(float)
    out["mr_confidence"] = _clip_confidence(raw_conf * slope_penalty * regime_mult)

    return out


# ---------------------------------------------------------------------------
# Sub-model 2 — Momentum
# ---------------------------------------------------------------------------

def momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime-conditioned momentum signal and confidence.

    Logic
    -----
    Composite of rolling-normalised GSR EMA slope, relative strength, and
    gold MACD histogram.  A rising GSR (gold outperforming) → BUY_GOLD (−).
    Confidence is the absolute magnitude of the composite score.

    New columns
    -----------
    mom_signal      float  [-1, 1]
    mom_confidence  float  [ 0, 1]
    """
    required = [
        "gsr_ema_slope", "rel_strength_20",
        "gold_macd_hist", "silver_ema_slope",
        "gold_ema_slope", "regime",
    ]
    _check_cols(df, required, "momentum_signals")

    out = df.copy()

    # Normalise each component to a common scale (rolling 60-day z-score)
    z_gsr_ema  = _rolling_zscore(out["gsr_ema_slope"])
    z_rel      = _rolling_zscore(out["rel_strength_20"])
    z_macd     = _rolling_zscore(out["gold_macd_hist"])
    z_silver   = _rolling_zscore(out["silver_ema_slope"])

    # Sign convention: GSR up → gold outperforming → BUY_GOLD → negative signal
    composite = (
        -z_gsr_ema  * 0.35
        - z_rel     * 0.30
        - z_macd    * 0.20
        + z_silver  * 0.15   # silver trend up → BUY_SILVER → positive
    )

    regime_mult = _regime_weight(out["regime"], "momentum")

    out["mom_signal"]     = np.tanh(composite).astype(float)
    out["mom_confidence"] = _clip_confidence(composite.abs() / 3.0 * regime_mult)

    return out


# ---------------------------------------------------------------------------
# Sub-model 3 — Breakout
# ---------------------------------------------------------------------------

def breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime-conditioned breakout signal and confidence.

    Logic
    -----
    gold_breakout / silver_breakout ∈ [0, 1]:
      0 = price at 20-day low, 1 = price at 20-day high.
    Relative breakout: silver near high while gold lags → BUY_SILVER (+).
    Confidence is amplified when short-term vol is above its recent average
    (confirms the breakout is supported by activity).

    New columns
    -----------
    bo_signal      float  [-1, 1]
    bo_confidence  float  [ 0, 1]
    """
    required = [
        "gold_breakout", "silver_breakout",
        "gold_vol_5", "gold_vol_20", "gold_atr_pct",
        "regime",
    ]
    _check_cols(df, required, "breakout_signals")

    out = df.copy()

    # Centre breakout scores at 0: +1 = at high, -1 = at low
    gold_bo   = (out["gold_breakout"]   - 0.5) * 2.0
    silver_bo = (out["silver_breakout"] - 0.5) * 2.0

    # Relative breakout: silver breaking out vs gold → positive signal
    rel_bo = silver_bo - gold_bo

    # Vol confirmation: ratio of 5-day to 20-day vol; >1 means vol is expanding
    vol_ratio = (out["gold_vol_5"] / out["gold_vol_20"].clip(lower=1e-9)).clip(0.5, 2.5)

    regime_mult = _regime_weight(out["regime"], "breakout")

    raw_signal = np.tanh(rel_bo * vol_ratio)
    raw_conf   = (rel_bo.abs() * vol_ratio * out["gold_atr_pct"].clip(lower=1e-9))

    # Normalise confidence by its rolling 60-day max so it stays in [0, 1]
    rolling_max = raw_conf.rolling(60, min_periods=10).max().clip(lower=1e-9)
    norm_conf   = (raw_conf / rolling_max).clip(0.0, 1.0)

    out["bo_signal"]     = raw_signal.astype(float)
    out["bo_confidence"] = _clip_confidence(norm_conf * regime_mult)

    return out


# ---------------------------------------------------------------------------
# Combined convenience function
# ---------------------------------------------------------------------------

def build_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three sub-models in sequence.

    Expects a DataFrame that already has the *regime* column (from
    regime_detection.fit_and_label).  Returns the same DataFrame with
    six new columns appended:
      mr_signal, mr_confidence,
      mom_signal, mom_confidence,
      bo_signal, bo_confidence
    """
    df = mean_reversion_signals(df)
    df = momentum_signals(df)
    df = breakout_signals(df)

    new_cols = [
        "mr_signal",  "mr_confidence",
        "mom_signal", "mom_confidence",
        "bo_signal",  "bo_confidence",
    ]
    print("✅ Strategy signals built:", new_cols)
    print(f"   Non-null rows: {df[new_cols].dropna().shape[0]} / {len(df)}")

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_cols(df: pd.DataFrame, required: list, caller: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{caller}(): missing required columns {missing}.\n"
            f"Ensure the full pipeline has been run before calling this function."
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import os
    try:
        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "features_with_regimes.csv")
        )
        print(f"🚀 Loading {path}...")
        df = pd.read_csv(path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        print(f"   {len(df)} rows, {len(df.columns)} columns")

        df = build_strategy_signals(df)

        print("\n--- Signal Snapshot (last 5 rows) ---")
        snap = ["regime", "mr_signal", "mr_confidence",
                "mom_signal", "mom_confidence", "bo_signal", "bo_confidence"]
        print(df[snap].tail().to_string())

        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "features_with_signals.csv")
        )
        df.to_csv(out_path)
        print(f"\n💾 Saved to {out_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
