"""
The strategy. One definition, used by the backtest and by the live dashboard.

Why this module exists
----------------------
The signal used to be defined twice. `backtesting/simulator._signal` drove the
equity curve, while the dashboard's recommendation card rendered
`ensemble_signal` from a separate XGBoost model that curve never tested. A user
saw a Sharpe ratio produced by one strategy sitting next to a "BUY SILVER" from
another. Whatever is advised today has to be the thing that was measured, so
both paths now call `decide()` here.

The decision
------------
Two layers, in order:

1. Regime, from the Gaussian HMM. Its economic reading is fixed in
   `models/regime_detection`:
       crisis      gold bid as a haven, ratio elevated
       risk_on     silver outperforming, ratio compressed
       range_bound quiet, weak momentum, nothing to trade
       trending    directional, defer to the ratio rule
2. The gold/silver ratio rule, for the trending case and whenever no regime is
   available: a stretched ratio reverts, a drifting one is followed.

Position size is inverse to realised volatility, so a signal fired during a
turbulent stretch is taken smaller. Standing aside in `range_bound` is a real
position (zero), not a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Kept here so the backtest and the dashboard cannot drift apart on thresholds.
ZSCORE_THRESHOLD = 1.0     # |z| above which the ratio counts as stretched
SLOPE_THRESHOLD = 0.5      # max |slope| for the ratio to count as decelerating
TARGET_DAILY_VOL = 0.01    # 1% daily vol target
MAX_POSITION_SIZE = 1.0    # no leverage

GOLD, SILVER, FLAT = "gold", "silver", None


@dataclass(frozen=True)
class Decision:
    """What to hold, how much of it, and why."""

    asset: str | None          # "gold" | "silver" | None (stand aside)
    size: float                # 0.0 - 1.0 of capital
    regime: str                # regime that produced it, or "unknown"
    rationale: str             # one line, for the dashboard to show

    @property
    def label(self) -> str:
        if self.asset is None:
            return "STAND ASIDE"
        return f"HOLD {self.asset.upper()}"


def ratio_rule(row: pd.Series) -> str:
    """
    The gold/silver ratio rule, unchanged from the original strategy.

    A ratio stretched far from its mean while no longer accelerating tends to
    revert, so buy the leg that is cheap on the ratio. Otherwise follow the
    ratio's drift.
    """
    z = row.get("gsr_zscore_30", np.nan)
    slope = row.get("gsr_slope", np.nan)

    if pd.notna(z) and pd.notna(slope) and abs(slope) < SLOPE_THRESHOLD:
        if z > ZSCORE_THRESHOLD:
            return SILVER          # ratio high: silver cheap against gold
        if z < -ZSCORE_THRESHOLD:
            return GOLD            # ratio low: gold cheap against silver

    if pd.notna(slope):
        return GOLD if slope > 0 else SILVER
    return GOLD


def position_size(row: pd.Series, asset: str | None) -> float:
    """Inverse-volatility sizing, capped at fully invested."""
    if asset is None:
        return 0.0
    vol = row.get("gold_vol_20" if asset == GOLD else "silver_vol_20", np.nan)
    if pd.isna(vol) or vol <= 0:
        return MAX_POSITION_SIZE
    return float(min(TARGET_DAILY_VOL / vol, MAX_POSITION_SIZE))


def decide(row: pd.Series, regime: str | None = None) -> Decision:
    """
    The single decision function.

    `row` is one bar of features; `regime` is the HMM's label for that bar, or
    None when no regime model is available — in which case the ratio rule
    stands on its own rather than the call being skipped.
    """
    reg = regime if isinstance(regime, str) and regime else "unknown"

    if reg == "crisis":
        asset = GOLD
        why = "crisis regime: gold bid as a haven"
    elif reg == "risk_on":
        asset = SILVER
        why = "risk-on regime: silver leads"
    elif reg == "range_bound":
        asset = FLAT
        why = "range-bound regime: no edge, standing aside"
    else:
        asset = ratio_rule(row)
        z = row.get("gsr_zscore_30", float("nan"))
        stretched = pd.notna(z) and abs(z) > ZSCORE_THRESHOLD
        why = (
            f"ratio {'stretched' if stretched else 'drifting'} "
            f"(z={z:+.2f}); {reg} regime"
            if pd.notna(z) else f"ratio rule; {reg} regime"
        )

    return Decision(asset=asset, size=position_size(row, asset), regime=reg, rationale=why)


def decide_series(df: pd.DataFrame, regimes: pd.Series | None = None) -> pd.DataFrame:
    """Vectorised-ish helper: a Decision per row, returned as columns."""
    out = []
    for ts, row in df.iterrows():
        reg = None
        if regimes is not None and ts in regimes.index:
            reg = regimes.loc[ts]
        d = decide(row, reg)
        out.append({"asset": d.asset, "size": d.size,
                    "regime": d.regime, "rationale": d.rationale})
    return pd.DataFrame(out, index=df.index)
