"""
The strategy. One definition, used by validation, the backtest and the dashboard.

Why this module exists
----------------------
The signal used to be defined twice. `backtesting/simulator._signal` drove the
equity curve, while the dashboard's recommendation card rendered
`ensemble_signal` from a separate XGBoost model that curve never tested. A user
saw a Sharpe ratio produced by one strategy sitting next to a "BUY SILVER" from
another. Whatever is advised today has to be the thing that was measured, so
every path calls `decide()` here, and every path-dependent run goes through
`simulate()`, which calls `decide()` once per bar.

The decision
------------
1. Regime, from the Gaussian HMM (`regime_model.RegimeModel`). Its economic
   reading is fixed in `models/regime_detection`:
       crisis      gold bid as a haven, ratio elevated     -> gold
       risk_on     silver outperforming, ratio compressed  -> silver
       range_bound quiet, weak momentum, nothing to trade  -> stand aside
       trending    directional, defer to the ratio rule
   With ``soft=True`` each regime's action is weighted by its filtered
   probability instead of acting on the single most likely regime.
2. The gold/silver ratio rule, for the trending case and whenever no regime is
   available: a stretched ratio reverts, a drifting one is followed.
3. Risk controls, each optional in `StrategyConfig`:
   * a time-series momentum gate — a metal is only held while its own
     trailing return over `trend_lookback` days is positive;
   * drawdown de-risking — exposure is halved while the strategy is more than
     10% below its trailing one-year equity peak.
   * the universe: "switch" (original — trending defers to the ratio rule),
     "gold_core" (trending holds gold; silver only in risk_on) or
     "gold_only" (silver never held; the regime can only stand aside).

Each metal leg is sized to a 1% daily volatility target from its 20-day
realised volatility, never levered. Standing aside is a real position (zero).

`LEGACY` reproduces the original rule set exactly. `SERVED` is the
configuration chosen by `experiments/search.py` on the early validation folds
only; `experiments/validate.py` reports it on every fold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

# Kept here so the backtest and the dashboard cannot drift apart on thresholds.
ZSCORE_THRESHOLD = 1.0     # |z| above which the ratio counts as stretched
SLOPE_THRESHOLD = 0.5      # max |slope| for the ratio to count as decelerating
TARGET_DAILY_VOL = 0.01    # 1% daily vol target per leg
MAX_POSITION_SIZE = 1.0    # no leverage
DD_THRESHOLD = 0.10        # de-risk below this drawdown from the 1-year peak
DD_SCALE = 0.5             # ...by halving exposure
DD_PEAK_WINDOW = 252
COST_PER_TURN = 0.0002 + 0.0001   # spread + slippage, per unit of turnover
EXECUTION_LAG = 1          # decide at close t, trade at close t+1

GOLD, SILVER, FLAT = "gold", "silver", None
REGIME_ACTION = {"crisis": GOLD, "risk_on": SILVER, "range_bound": FLAT}


@dataclass(frozen=True)
class StrategyConfig:
    """Everything that distinguishes one tested variant from another."""

    n_states: int | str | None = 4   # HMM states, "bic", or None for no regimes
    sticky: float = 0.0              # transition-matrix diagonal prior
    scale_free: bool = False         # price-denominated features divided by price
    soft: bool = False               # posterior-weighted vs argmax regime
    trend_lookback: int | None = None
    dd_derisk: bool = False
    universe: str = "switch"         # "switch" | "gold_core" | "gold_only"

    @property
    def uses_regimes(self) -> bool:
        return self.n_states is not None

    @property
    def regime_kwargs(self) -> dict:
        return {"n_states": self.n_states, "sticky": self.sticky,
                "scale_free": self.scale_free}

    @property
    def name(self) -> str:
        if not self.uses_regimes:
            parts = ["no-regime"]
        else:
            parts = [f"k={self.n_states}", f"sticky={self.sticky:g}",
                     "scalefree" if self.scale_free else "raw",
                     "soft" if self.soft else "hard"]
        if self.universe != "switch":
            parts.append(self.universe)
        parts += [f"trend={self.trend_lookback or 'off'}",
                  "dd" if self.dd_derisk else "nodd"]
        return " ".join(parts)

    def as_dict(self) -> dict:
        return asdict(self)


LEGACY = StrategyConfig()
# Chosen by experiments/search.py on folds 1-11 only (see docs/search.md):
# the highest early-fold Sharpe of all configurations was vol-targeted gold
# with a 12-month trend gate. No regime-based configuration beat it, so the
# served strategy does not use the HMM; experiments/validate.py still reports
# the best regime-based configuration next to it.
SERVED = StrategyConfig(n_states=None, trend_lookback=252, dd_derisk=False,
                        universe="gold_only")


@dataclass(frozen=True)
class Decision:
    """What to hold, how much of it, and why."""

    asset: str | None          # dominant holding: "gold" | "silver" | None
    size: float                # total fraction of capital, 0.0 - 1.0
    regime: str                # most likely regime, or "unknown"
    rationale: str             # one line, for the dashboard to show
    gold_weight: float = 0.0
    silver_weight: float = 0.0

    @property
    def label(self) -> str:
        if self.asset is None:
            return "STAND ASIDE"
        if self.gold_weight > 0 and self.silver_weight > 0:
            return f"HOLD {self.asset.upper()} (+{'SILVER' if self.asset == GOLD else 'GOLD'})"
        return f"HOLD {self.asset.upper()}"


# ── features the strategy needs beyond features.csv ─────────────────────────

def momentum_col(asset: str, lookback: int) -> str:
    return f"{asset}_mom_{lookback}"


def add_trend_features(df: pd.DataFrame, lookbacks=(126, 252)) -> pd.DataFrame:
    """Trailing total return of each metal. Uses closes up to t only."""
    out = df.copy()
    for L in lookbacks:
        out[momentum_col(GOLD, L)] = out["gold_close"].pct_change(L)
        out[momentum_col(SILVER, L)] = out["silver_close"].pct_change(L)
    return out


# ── building blocks ──────────────────────────────────────────────────────────

def ratio_rule(row: Mapping) -> str:
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


def position_size(row: Mapping, asset: str | None) -> float:
    """Inverse-volatility sizing, capped at fully invested."""
    if asset is None:
        return 0.0
    vol = row.get("gold_vol_20" if asset == GOLD else "silver_vol_20", np.nan)
    if pd.isna(vol) or vol <= 0:
        return MAX_POSITION_SIZE
    return float(min(TARGET_DAILY_VOL / vol, MAX_POSITION_SIZE))


def _regime_action(reg: str, row: Mapping, universe: str = "switch") -> str | None:
    """
    Which metal a regime calls for.

    switch     the original mapping; trending/unknown defer to the ratio rule
    gold_core  gold unless the regime is risk_on (silver) or range_bound (flat)
    gold_only  gold, or flat in range_bound; silver is never held
    """
    if reg == "range_bound":
        return FLAT
    if universe == "gold_only":
        return GOLD
    if reg in REGIME_ACTION:
        return REGIME_ACTION[reg]
    return GOLD if universe == "gold_core" else ratio_rule(row)


def _clean_probs(probs) -> dict | None:
    if probs is None:
        return None
    p = {k: float(v) for k, v in dict(probs).items() if pd.notna(v)}
    return p if p and sum(p.values()) > 0 else None


# ── the decision ─────────────────────────────────────────────────────────────

def decide(row: Mapping, regime: str | None = None, probs: Mapping | None = None,
           config: StrategyConfig = LEGACY, drawdown: float = 0.0) -> Decision:
    """
    The single decision function.

    `row` is one bar of features (a Series or dict). The regime comes either as
    a label (`regime`) or as filtered probabilities per regime name (`probs`,
    preferred); with neither, the ratio rule stands on its own rather than the
    call being skipped. `drawdown` is the strategy's current drawdown from its
    trailing peak (<= 0), used only when `config.dd_derisk` is on.
    """
    p = _clean_probs(probs)
    if p is not None:
        reg = max(p, key=p.get)
    else:
        reg = regime if isinstance(regime, str) and regime else "unknown"

    # 1-2. regime -> which metal(s), before sizing
    exposure = {GOLD: 0.0, SILVER: 0.0}
    if config.soft and p is not None:
        total = sum(p.values())
        for name, mass in p.items():
            a = _regime_action(name, row, config.universe)
            if a is not None:
                exposure[a] += mass / total
    else:
        a = _regime_action(reg, row, config.universe)
        if a is not None:
            exposure[a] = 1.0

    if not config.uses_regimes and config.universe != "switch":
        why = "gold, sized to a 1% daily vol target"
    elif config.universe != "switch" and reg != "range_bound":
        held = "gold" if exposure[GOLD] >= exposure[SILVER] else "silver"
        why = f"{reg} regime: hold {held} ({config.universe.replace('_', '-')})"
    elif reg in REGIME_ACTION:
        why = {"crisis": "crisis regime: gold bid as a haven",
               "risk_on": "risk-on regime: silver leads",
               "range_bound": "range-bound regime: no edge, standing aside"}[reg]
    else:
        z = row.get("gsr_zscore_30", float("nan"))
        stretched = pd.notna(z) and abs(z) > ZSCORE_THRESHOLD
        why = (f"ratio {'stretched' if stretched else 'drifting'} (z={z:+.2f}); {reg} regime"
               if pd.notna(z) else f"ratio rule; {reg} regime")

    # 3. risk controls
    if config.trend_lookback:
        for a in (GOLD, SILVER):
            mom = row.get(momentum_col(a, config.trend_lookback), np.nan)
            if exposure[a] > 0 and pd.notna(mom):
                if mom <= 0:
                    exposure[a] = 0.0
                    why += f"; {a} {config.trend_lookback}d return {mom:+.0%}, standing aside"
                else:
                    why += f"; {a} {config.trend_lookback}d return {mom:+.0%}"
    scale = 1.0
    if config.dd_derisk and drawdown <= -DD_THRESHOLD:
        scale = DD_SCALE
        why += f"; drawdown {drawdown:.0%}, exposure halved"

    wg = exposure[GOLD] * position_size(row, GOLD) * scale
    ws = exposure[SILVER] * position_size(row, SILVER) * scale
    if wg <= 0 and ws <= 0:
        asset = FLAT
    else:
        asset = GOLD if wg >= ws else SILVER
    return Decision(asset=asset, size=float(wg + ws), regime=reg, rationale=why,
                    gold_weight=float(wg), silver_weight=float(ws))


# ── running it through time ──────────────────────────────────────────────────

def simulate(df: pd.DataFrame, probs: pd.DataFrame | None = None,
             regimes: pd.Series | None = None, config: StrategyConfig = LEGACY,
             cost: float = COST_PER_TURN, lag: int = EXECUTION_LAG) -> pd.DataFrame:
    """
    Walk `df` bar by bar, calling `decide` at each close.

    The weights decided at close t are traded at close t+lag and earn the
    return from t+lag to t+lag+1. Costs are `cost` per unit of turnover
    (switching fully from gold to silver is two units). Returns one row per bar
    with the decision and the net return realised on that bar.
    """
    rg = df["gold_close"].pct_change().fillna(0.0).to_numpy()
    rs = df["silver_close"].pct_change().fillna(0.0).to_numpy()
    rows = df.to_dict("records")
    prob_rows = probs.reindex(df.index).to_dict("records") if probs is not None else None
    reg_list = regimes.reindex(df.index).tolist() if regimes is not None else None

    n = len(df)
    wg, ws = np.zeros(n), np.zeros(n)
    ret, turn = np.zeros(n), np.zeros(n)
    eq, peaks = 1.0, deque(maxlen=DD_PEAK_WINDOW)
    decisions = []
    for i in range(n):
        j = i - 1 - lag                      # decision whose position earns bar i
        if j >= 0:
            prev = (wg[j - 1], ws[j - 1]) if j >= 1 else (0.0, 0.0)
            turn[i] = abs(wg[j] - prev[0]) + abs(ws[j] - prev[1])
            ret[i] = wg[j] * rg[i] + ws[j] * rs[i] - cost * turn[i]
            eq *= 1 + ret[i]
        peaks.append(eq)
        dd = eq / max(peaks) - 1
        d = decide(rows[i],
                   regime=reg_list[i] if reg_list is not None else None,
                   probs=prob_rows[i] if prob_rows is not None else None,
                   config=config, drawdown=dd)
        wg[i], ws[i] = d.gold_weight, d.silver_weight
        decisions.append(d)

    return pd.DataFrame({
        "gold_weight": wg, "silver_weight": ws,
        "asset": [d.asset for d in decisions],
        "regime": [d.regime for d in decisions],
        "rationale": [d.rationale for d in decisions],
        "turnover": turn, "ret": ret,
    }, index=df.index)


def decide_series(df: pd.DataFrame, regimes: pd.Series | None = None) -> pd.DataFrame:
    """A Decision per row from hard regime labels, returned as columns."""
    out = []
    for ts, row in df.iterrows():
        reg = regimes.loc[ts] if regimes is not None and ts in regimes.index else None
        d = decide(row, reg)
        out.append({"asset": d.asset, "size": d.size,
                    "regime": d.regime, "rationale": d.rationale})
    return pd.DataFrame(out, index=df.index)
