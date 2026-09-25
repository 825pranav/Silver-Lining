"""
Walk-forward validation of the strategy that is actually served.

    python experiments/search.py     # first: chooses strategy.SERVED on folds 1-11
    python experiments/validate.py   # then: reports it on every fold

Every regime model is fitted on its fold's training rows only, and its
probabilities come from the forward filter, so no decision sees a later bar.
Positions come from `strategy.simulate`, which calls `strategy.decide` — the
same function the dashboard calls for today's recommendation.

Also reported, all regenerated here:
* the original rule set (regimes + ratio rule), measured the corrected way;
* the best regime-based configuration of the pre-registered round-1 search;
* buy-and-hold gold and silver on the same days, and gold scaled to the
  served strategy's volatility;
* Probabilistic and Deflated Sharpe Ratios (DSR counts every configuration in
  data/search_trials.csv) and block-bootstrap intervals on the Sharpe
  difference against buy-and-hold;
* a lookahead audit showing how the previously published Sharpe of 0.708 was
  produced and what each correction does to it.

Writes docs/results.md and data/validation_folds.csv.
"""

from __future__ import annotations

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
from backtesting.metrics import sharpe  # noqa: E402
from backtesting.stats import (bootstrap_sharpe_diff, deflated_sharpe,  # noqa: E402
                               probabilistic_sharpe)
from paths import ROOT  # noqa: E402
from regime_model import RegimeModel  # noqa: E402
from strategy import LEGACY, SERVED, StrategyConfig, decide, simulate  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

TRIALS_PATH = ROOT / "data" / "search_trials.csv"
OLD_SPREAD, OLD_SLIPPAGE = 0.0002, 0.0001


# ── helpers ──────────────────────────────────────────────────────────────────

def _config_from_row(row: pd.Series) -> StrategyConfig:
    def clean(v):
        return None if pd.isna(v) else v
    k = clean(row["n_states"])
    if k is not None and k != "bic":
        k = int(float(k))
    trend = clean(row["trend_lookback"])
    return StrategyConfig(n_states=k, sticky=float(row["sticky"]),
                          scale_free=bool(row["scale_free"]), soft=bool(row["soft"]),
                          trend_lookback=int(trend) if trend is not None else None,
                          dd_derisk=bool(row["dd_derisk"]),
                          universe=row.get("universe", "switch") or "switch")


def _row(name: str, r: pd.Series, df: pd.DataFrame) -> dict:
    fid = W.fold_of(r.index, df)
    s = W.summarise(r)
    pf = W.per_fold(r, df)
    return {
        "strategy": name,
        "profitable_folds": f"{int((pf['sharpe'] > 0).sum())}/{len(pf)}",
        "sharpe": round(s["sharpe"], 3),
        "sharpe_folds_1_11": round(sharpe(r[fid.isin(W.SELECTION_FOLDS)]), 3),
        "sharpe_folds_12_23": round(sharpe(r[fid.isin(W.CONFIRMATION_FOLDS)]), 3),
        "total_return": f"{s['total_return']:+.0%}",
        "cagr": f"{s['cagr']:.1%}",
        "ann_vol": f"{s['ann_vol']:.1%}",
        "max_drawdown": f"{s['max_drawdown']:.1%}",
    }


# ── the lookahead audit ──────────────────────────────────────────────────────

def _old_run_period(test: pd.DataFrame, regimes: pd.Series | None) -> pd.Series:
    """The original validation loop: same-close execution, cost only on switches."""
    capital, held, size, curve = 1.0, None, 0.0, []
    for i in range(1, len(test)):
        today, yday = test.iloc[i], test.iloc[i - 1]
        gross = (today[f"{held}_close"] / yday[f"{held}_close"] - 1) if held else 0.0
        d = decide(today, regimes.iloc[i] if regimes is not None else None)
        switched = int(d.asset != held and held is not None)
        capital *= 1 + size * gross - switched * (OLD_SPREAD + OLD_SLIPPAGE)
        curve.append(capital)
        held, size = d.asset, d.size
    return pd.Series(curve, index=test.index[1:])


def _old_pooled(df: pd.DataFrame, labeller) -> pd.Series:
    curves = []
    for f in W.folds(df):
        test = df.iloc[f["t0"]: f["t1"]]
        cap = _old_run_period(test, labeller(f, test))
        curves.append((cap / cap.iloc[0]).pct_change().dropna())
    return pd.concat(curves)


def lookahead_audit(df: pd.DataFrame, corrected: pd.Series) -> pd.DataFrame:
    def viterbi(f, test):
        m = RegimeModel(refit_garch=False).fit(df.iloc[: f["train_end"]])
        return m.predict_viterbi(test)

    def filtered(refit_garch):
        def lab(f, test):
            m = RegimeModel(refit_garch=refit_garch).fit(df.iloc[: f["train_end"]])
            return m.predict(df.iloc[: f["t1"]]).iloc[f["t0"]: f["t1"]]
        return lab

    steps = [
        ("as published: Viterbi over each test year, GARCH fitted on all data, "
         "trade at the signal's own close", _old_pooled(df, viterbi)),
        ("+ forward-filtered regimes (no future bars)", _old_pooled(df, filtered(False))),
        ("+ GARCH refitted on each training fold", _old_pooled(df, filtered(True))),
        ("+ trade one day after the signal, cost on all turnover, one continuous path",
         corrected),
    ]
    rows = []
    for name, r in steps:
        eq = (1 + r).cumprod()
        rows.append({"step": name, "sharpe": round(sharpe(r), 3),
                     "total_return": f"{eq.iloc[-1] - 1:+.0%}",
                     "max_drawdown": f"{(eq / eq.cummax() - 1).min():.1%}"})
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = W.load_features()
    print(f"[data] {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    trials = pd.read_csv(TRIALS_PATH)
    chosen = trials[trials["selected"]].iloc[0]
    if chosen["config"] != SERVED.name:
        raise SystemExit(f"strategy.SERVED ({SERVED.name}) is not the configuration "
                         f"experiments/search.py chose ({chosen['config']}).")
    r1 = trials[trials["round"] == 1]
    prereg = _config_from_row(r1.loc[r1["sel_sharpe"].idxmax()])

    runs = {}
    print(f"[run] served: {SERVED.name}")
    served_sim = W.run_config(df, SERVED)
    runs["served"] = W.oos_returns(served_sim)
    print(f"[run] pre-registered regime winner: {prereg.name}")
    runs["prereg"] = W.oos_returns(W.run_config(df, prereg))
    print("[run] original rule set, corrected")
    runs["legacy"] = W.oos_returns(W.run_config(df, LEGACY))
    runs["ratio"] = W.oos_returns(W.run_config(df, StrategyConfig(n_states=None)))
    idx = runs["served"].index
    bh = W.buy_and_hold_returns(df, idx)
    vol_ratio = runs["served"].std() / bh["gold"].std()
    runs["gold_volmatched"] = bh["gold"] * vol_ratio

    summary = pd.DataFrame([
        _row(f"served: {SERVED.name}", runs["served"], df),
        _row(f"best regime config, round 1: {prereg.name}", runs["prereg"], df),
        _row("original rules (regimes + ratio rule), corrected", runs["legacy"], df),
        _row("ratio rule alone", runs["ratio"], df),
        _row("buy & hold gold", bh["gold"], df),
        _row("buy & hold silver", bh["silver"], df),
        _row(f"buy & hold gold at {vol_ratio:.2f}x (served strategy's vol)",
             runs["gold_volmatched"], df),
    ])
    print("\n" + summary.to_string(index=False))

    # statistics
    n_trials = len(trials)
    stats_rows = []
    for key, label in (("served", "served"), ("prereg", "best regime config, round 1")):
        r = runs[key]
        d = deflated_sharpe(r, trials["full_daily_sr"], n_trials)
        g = bootstrap_sharpe_diff(r, bh["gold"])
        s = bootstrap_sharpe_diff(r, bh["silver"])
        stats_rows.append({
            "strategy": label, "PSR(SR>0)": round(probabilistic_sharpe(r), 3),
            "DSR": round(d["dsr"], 3), "trials": d["n_trials"],
            "E[max SR] of trials (ann.)": round(d["sr0_annual"], 3),
            "Sharpe - gold [95% CI]": f"{g['diff']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]",
            "P(not > gold)": round(g["p_not_better"], 3),
            "Sharpe - silver [95% CI]": f"{s['diff']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
        })
    stats = pd.DataFrame(stats_rows)
    print("\n" + stats.to_string(index=False))

    print("\n[audit] reproducing the published figure and correcting it")
    audit = lookahead_audit(df, runs["legacy"])
    print(audit.to_string(index=False))

    folds = W.per_fold(runs["served"], df)
    gold_folds = W.per_fold(bh["gold"], df)[["fold", "sharpe", "max_drawdown"]].rename(
        columns={"sharpe": "gold_sharpe", "max_drawdown": "gold_max_dd"})
    folds = folds.merge(gold_folds, on="fold")
    folds.to_csv(ROOT / "data" / "validation_folds.csv", index=False)
    exposure = served_sim[["gold_weight", "silver_weight"]].loc[idx].sum(axis=1)
    _write_results(df, summary, stats, audit, folds, trials, prereg, exposure)
    print("\n[saved] docs/results.md and data/validation_folds.csv")


def _write_results(df, summary, stats, audit, folds, trials, prereg, exposure) -> None:
    fs = W.folds(df)
    n1 = int((trials["round"] == 1).sum())
    lines = [
        "# Results",
        "",
        "Regenerate with `python experiments/search.py` then",
        "`python experiments/validate.py`. Every number below is written by those",
        "scripts; nothing here is typed by hand.",
        "",
        f"Data: {len(df)} daily bars, {df.index[0].date()} to {df.index[-1].date()}.",
        f"Out-of-sample: {len(fs)} folds, {fs[0]['test_start'].date()} to "
        f"{fs[-1]['test_end'].date()}.",
        "",
        "## Method",
        "",
        f"Expanding-window walk-forward: at least {W.MIN_TRAIN} training bars, "
        f"{W.PURGE_DAYS} bars purged",
        f"between train and test, {W.TEST_SIZE}-bar (about one year) test folds. Each fold's",
        "regime model (HMM, scaler and GARCH parameters) is fitted on its training",
        "rows only, and regime probabilities come from the forward filter, so the",
        "call on day t uses bars up to t. A position decided at the close of day t",
        "is traded at the close of t+1. Costs are 3bp (2bp spread + 1bp slippage)",
        "per unit of turnover, including resizing. The strategy runs as one",
        "continuous path across folds.",
        "",
        f"`strategy.SERVED` was chosen by `experiments/search.py` from {len(trials)}",
        "configurations using folds 1-11 only (highest pooled Sharpe); see",
        "`docs/search.md` for every configuration. Round 2 of that search was",
        "exploratory (added after seeing round-1 full-sample results), so folds",
        "12-23 are a clean holdout only for the round-1 configuration shown below,",
        "not for the served one.",
        "",
        f"Served: `{SERVED.name}`. Average exposure {exposure.mean():.0%} of capital; "
        f"flat {(exposure == 0).mean():.0%} of days.",
        "",
        "## Out-of-sample summary",
        "",
        summary.to_markdown(index=False),
        "",
        "The volatility-matched row scales buy-and-hold gold by a constant chosen",
        "after the fact to match the served strategy's realised volatility. It is",
        "not a tradable benchmark; it shows how much of a drawdown gap comes from",
        "holding less rather than from timing.",
        "",
        "## Is the Sharpe ratio real?",
        "",
        stats.to_markdown(index=False),
        "",
        "PSR: probability the true Sharpe is above zero given the sample's length,",
        "skew and kurtosis. DSR: the same probability against the best Sharpe",
        f"expected from {len(trials)} unskilled trials with the dispersion observed",
        f"across the search ({n1} pre-registered + {len(trials) - n1} exploratory).",
        "Intervals: paired stationary block bootstrap (mean block 20 days,",
        "2,000 resamples) of the annualised Sharpe difference on the same days.",
        "",
        "## Lookahead audit",
        "",
        "The previously published regime-gated Sharpe was 0.708. The first row",
        "below reproduces it; each later row adds one correction to the original",
        "rule set. The last row is the corrected original strategy used above.",
        "",
        audit.to_markdown(index=False),
        "",
        "## Per fold, served strategy (with buy & hold gold)",
        "",
        folds.to_markdown(index=False),
        "",
    ]
    (ROOT / "docs" / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
