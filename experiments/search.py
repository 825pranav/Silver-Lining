"""
Configuration search, chosen on early folds only.

    python experiments/search.py

Two rounds, every configuration run and reported — nothing is dropped after
the fact. Choices use only folds 1-11 (test years mid-2004 to mid-2015).

Round 1 (`grid()`, 102 configurations) was fixed before any result was seen.

Round 2 was added after round 1 had been run and its full-sample results
looked at, which showed that a simple vol-targeted gold position beat every
round-1 configuration. It adds a "universe" choice — gold as the core holding
instead of the ratio rule deciding between metals — and so is exploratory,
not pre-registered. To keep it small and mechanical, each round-1 dimension
is pruned to the values whose mean Sharpe on folds 1-11 is within
PRUNE_MARGIN of the best (`prune()`); a no-regime control is always kept.

The served configuration is the highest folds 1-11 Sharpe over both rounds.
Folds 12-23 are reported for every configuration but play no part in any
choice. The total number of configurations and the spread of their Sharpe
ratios feed the Deflated Sharpe Ratio in `experiments/validate.py`.

Writes data/search_trials.csv and docs/search.md.
"""

from __future__ import annotations

import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import console as _console  # noqa: F401,E402

import walkforward_core as W  # noqa: E402
from paths import DATA, ROOT  # noqa: E402
from strategy import SERVED, StrategyConfig  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

TRIALS_PATH = DATA / "search_trials.csv"
PRUNE_MARGIN = 0.02
DIMENSIONS = ("n_states", "sticky", "scale_free", "soft", "trend_lookback", "dd_derisk")
HMM_ONLY = ("sticky", "scale_free", "soft")


# ── the search space ─────────────────────────────────────────────────────────

def grid() -> list[StrategyConfig]:
    """Round 1, fixed in advance: 6 no-regime + 96 regime configurations."""
    out = []
    for trend, dd in itertools.product((None, 126, 252), (False, True)):
        out.append(StrategyConfig(n_states=None, trend_lookback=trend, dd_derisk=dd))
    for k, sticky, sf, soft, trend, dd in itertools.product(
            (4, "bic"), (0.0, 0.5), (False, True), (False, True),
            (None, 126, 252), (False, True)):
        out.append(StrategyConfig(n_states=k, sticky=sticky, scale_free=sf, soft=soft,
                                  trend_lookback=trend, dd_derisk=dd))
    return out


def _as_key(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "none"
    if isinstance(v, float) and v.is_integer() and v > 1:
        return str(int(v))
    return str(v)


def marginal_effects(trials: pd.DataFrame, col: str = "sel_sharpe") -> dict:
    """Mean Sharpe per value of each dimension (HMM-only ones over HMM configs)."""
    out = {}
    for d in DIMENSIONS:
        g = trials[trials["n_states"].map(_as_key) != "none"] if d in HMM_ONLY else trials
        out[d] = g.groupby(g[d].map(_as_key))[col].mean().to_dict()
    return out


def prune(trials: pd.DataFrame) -> dict[str, list[str]]:
    """Per dimension, the values within PRUNE_MARGIN of the best folds 1-11 mean."""
    keep = {}
    for d, means in marginal_effects(trials).items():
        best = max(means.values())
        keep[d] = sorted(v for v, m in means.items() if m >= best - PRUNE_MARGIN)
    return keep


def _parse(d: str, v: str):
    if v == "none":
        return None
    if d == "n_states":
        return v if v == "bic" else int(v)
    if d == "sticky":
        return float(v)
    if d == "trend_lookback":
        return int(float(v))
    return v == "True"


def grid_round2(keep: dict[str, list[str]]) -> list[StrategyConfig]:
    """Gold-core universes over the pruned round-1 values, plus a no-regime control."""
    vals = {d: [_parse(d, v) for v in vs] for d, vs in keep.items()}
    out = []
    for trend, dd in itertools.product(vals["trend_lookback"], vals["dd_derisk"]):
        out.append(StrategyConfig(n_states=None, trend_lookback=trend, dd_derisk=dd,
                                  universe="gold_only"))
    ks = [k for k in vals["n_states"] if k is not None]
    for uni, k, sticky, sf, soft, trend, dd in itertools.product(
            ("gold_core", "gold_only"), ks, vals["sticky"], vals["scale_free"],
            vals["soft"], vals["trend_lookback"], vals["dd_derisk"]):
        out.append(StrategyConfig(n_states=k, sticky=sticky, scale_free=sf, soft=soft,
                                  trend_lookback=trend, dd_derisk=dd, universe=uni))
    return out


# ── running it ───────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, config: StrategyConfig, probs) -> dict:
    sim = W.run_config(df, config, probs=probs)
    r = W.oos_returns(sim)
    fid = W.fold_of(r.index, df)
    sel = r[fid.isin(W.SELECTION_FOLDS)]
    conf = r[fid.isin(W.CONFIRMATION_FOLDS)]
    pf = W.per_fold(r, df)
    full, s_sel, s_conf = W.summarise(r), W.summarise(sel), W.summarise(conf)
    return {
        "config": config.name, **config.as_dict(),
        "sel_sharpe": s_sel["sharpe"], "sel_max_dd": s_sel["max_drawdown"],
        "sel_daily_sr": sel.mean() / sel.std(),
        "conf_sharpe": s_conf["sharpe"], "conf_max_dd": s_conf["max_drawdown"],
        "full_sharpe": full["sharpe"], "full_total_return": full["total_return"],
        "full_max_dd": full["max_drawdown"], "full_ann_vol": full["ann_vol"],
        "full_daily_sr": r.mean() / r.std(),
        "profitable_folds": int((pf["sharpe"] > 0).sum()),
    }


def run_round(df: pd.DataFrame, configs: list[StrategyConfig], rnd: int,
              probs_by_model: dict) -> pd.DataFrame:
    print(f"[round {rnd}] {len(configs)} configurations", flush=True)
    rows = []
    for i, c in enumerate(configs, 1):
        probs = None
        if c.uses_regimes:
            key = tuple(sorted(c.regime_kwargs.items()))
            if key not in probs_by_model:
                print(f"  [fit] regime model {dict(key)}", flush=True)
                probs_by_model[key], _ = W.oos_regime_probs(df, c.regime_kwargs)
            probs = probs_by_model[key]
        rows.append({"round": rnd, **evaluate(df, c, probs)})
        if i % 12 == 0:
            print(f"  {i}/{len(configs)} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    df = W.load_features()
    probs_by_model: dict[tuple, pd.DataFrame] = {}
    r1 = run_round(df, grid(), 1, probs_by_model)
    keep = prune(r1)
    print(f"[prune] kept on folds 1-11: {keep}")
    r2 = run_round(df, grid_round2(keep), 2, probs_by_model)

    trials = pd.concat([r1, r2], ignore_index=True)
    trials["selected"] = False
    best = int(trials["sel_sharpe"].idxmax())
    trials.loc[best, "selected"] = True
    trials.to_csv(TRIALS_PATH, index=False)

    chosen = trials.loc[best]
    r1_best = r1.loc[r1["sel_sharpe"].idxmax()]
    print(f"\n[round 1 best on folds 1-11] {r1_best['config']}  ({r1_best['sel_sharpe']:.3f})")
    print(f"[chosen on folds 1-11]        {chosen['config']}  ({chosen['sel_sharpe']:.3f})")
    if chosen["config"] != SERVED.name:
        print(f"[note] strategy.SERVED is '{SERVED.name}'. Update it to the chosen "
              "configuration so validation and the dashboard serve it.")
    _write_doc(trials, df, keep)
    print("[saved] data/search_trials.csv and docs/search.md")


def _write_doc(trials: pd.DataFrame, df: pd.DataFrame, keep: dict) -> None:
    fs = W.folds(df)
    sel_span = f"{fs[0]['test_start'].date()} to {fs[10]['test_end'].date()}"
    conf_span = f"{fs[11]['test_start'].date()} to {fs[-1]['test_end'].date()}"
    show = trials.sort_values("sel_sharpe", ascending=False)[[
        "round", "config", "sel_sharpe", "sel_max_dd", "conf_sharpe", "conf_max_dd",
        "full_sharpe", "full_max_dd", "profitable_folds", "selected"]].round(3)
    rank_corr = float(trials["sel_sharpe"].rank().corr(trials["conf_sharpe"].rank()))

    r1 = trials[trials["round"] == 1]
    sel_fx, conf_fx = marginal_effects(r1), marginal_effects(r1, "conf_sharpe")
    effects = pd.DataFrame([
        {"dimension": d, "value": v, "mean_sel_sharpe": round(m, 3),
         "mean_conf_sharpe": round(conf_fx[d][v], 3), "kept_for_round_2": v in keep[d]}
        for d, means in sel_fx.items() for v, m in means.items()])
    n1, n2 = int((trials["round"] == 1).sum()), int((trials["round"] == 2).sum())

    lines = [
        "# Configuration search",
        "",
        "Regenerate with `python experiments/search.py`. Every number below is",
        "written by that script; nothing here is typed by hand.",
        "",
        f"**{len(trials)} configurations** were run ({n1} in round 1, {n2} in",
        "round 2), all listed below. Every choice used only folds 1-11",
        f"({sel_span}): highest pooled Sharpe wins. Folds 12-23 ({conf_span})",
        "are shown for every configuration but did not influence any choice.",
        "",
        "Round 1 was fixed before any result was seen. Round 2 is exploratory:",
        "it was added after round 1's full-sample results showed that plain",
        "vol-targeted gold beat every round-1 configuration, and it adds a",
        "gold-core universe (`gold_core`: trending holds gold, silver only in",
        "risk_on; `gold_only`: silver never held). Its other dimensions were",
        f"pruned mechanically to values within {PRUNE_MARGIN} of the best mean",
        "folds 1-11 Sharpe in round 1, plus a no-regime control. Because round 2",
        "was motivated by results that include folds 12-23, those folds are not",
        "a clean holdout for a round-2 winner; the Deflated Sharpe Ratio in",
        "`docs/results.md` counts every configuration from both rounds.",
        "",
        "Round 1 grid: regime model none, or a Gaussian HMM with 4 states or",
        "BIC-chosen from 3/4/5; sticky transition prior 0 or 0.5; raw or",
        "scale-free price features; hard (argmax) or soft (probability-weighted)",
        "regime actions; time-series momentum gate off, 126 or 252 days;",
        "drawdown de-risking off or on. Fixed, not searched: 1% daily vol target",
        "per leg, no leverage, 3bp per unit of turnover, one-day execution lag,",
        "the ratio-rule thresholds and the regime-to-metal mapping.",
        "",
        "Rank correlation between folds 1-11 and folds 12-23 Sharpe across all",
        f"configurations: {rank_corr:.2f}.",
        "",
        "## Average effect of each round-1 choice",
        "",
        effects.to_markdown(index=False),
        "",
        "## All configurations, by folds 1-11 Sharpe",
        "",
        show.to_markdown(index=False),
        "",
    ]
    (ROOT / "docs" / "search.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
