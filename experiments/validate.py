"""
Walk-forward validation of the strategy that is actually served.

Every fold fits the regime model on training rows only, predicts regimes on the
held-out year, and asks `strategy.decide` for a position on each bar — the same
function the dashboard calls for today's recommendation. Nothing here is tuned
on a test fold; the regime-to-position mapping comes from the economic reading
documented in `models/regime_detection`.

    python experiments/validate.py

Writes docs/results.md and data/validation_folds.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import console as _console  # noqa: F401,E402

from backtesting.metrics import cagr, hit_rate, max_drawdown, sharpe  # noqa: E402
from paths import FEATURE_PATH, ROOT  # noqa: E402
from regime_model import RegimeModel  # noqa: E402
from strategy import decide  # noqa: E402

warnings.filterwarnings("ignore")

PURGE_DAYS = 20      # dropped between train and test so rolling features cannot leak
MIN_TRAIN = 750      # ~3 years before the first out-of-sample call
TEST_SIZE = 250      # ~1 trading year per fold
SPREAD, SLIPPAGE = 0.0002, 0.0001


def run_period(test: pd.DataFrame, regimes: pd.Series | None) -> pd.Series:
    """Equity curve over one fold, using the served decision function."""
    capital, held, size = 1.0, None, 0.0
    curve = []
    for i in range(1, len(test)):
        today, yday = test.iloc[i], test.iloc[i - 1]

        if held == "gold":
            gross = today["gold_close"] / yday["gold_close"] - 1
        elif held == "silver":
            gross = today["silver_close"] / yday["silver_close"] - 1
        else:
            gross = 0.0

        reg = regimes.iloc[i] if regimes is not None else None
        d = decide(today, reg)
        switched = int(d.asset != held and held is not None)
        capital *= (1 + size * gross - switched * (SPREAD + SLIPPAGE))
        curve.append(capital)
        held, size = d.asset, d.size

    return pd.Series(curve, index=test.index[1:], name="capital")


def walk_forward(df: pd.DataFrame, use_regimes: bool) -> tuple[pd.DataFrame, pd.Series]:
    rows, curves = [], []
    t0, fold = MIN_TRAIN + PURGE_DAYS, 0

    while t0 + 60 <= len(df):
        fold += 1
        t1 = min(t0 + TEST_SIZE, len(df))
        train, test = df.iloc[: t0 - PURGE_DAYS], df.iloc[t0:t1]

        regimes = None
        if use_regimes:
            model = RegimeModel().fit(train)
            regimes = model.predict(test)

        cap = run_period(test, regimes)
        r = cap.pct_change().dropna()
        rows.append({
            "fold": fold,
            "test_start": test.index[0].date(),
            "test_end": test.index[-1].date(),
            "sharpe": round(sharpe(r), 3),
            "cagr": round(cagr(cap), 4),
            "max_drawdown": round(max_drawdown(cap), 4),
            "hit_rate": round(hit_rate(r), 4),
            "days": len(cap),
        })
        curves.append(cap / cap.iloc[0])
        t0 = t1

    pooled = pd.concat([c.pct_change().dropna() for c in curves])
    return pd.DataFrame(rows), pooled


def summarise(name: str, folds: pd.DataFrame, pooled: pd.Series) -> dict:
    eq = (1 + pooled).cumprod()
    return {
        "strategy": name,
        "folds": len(folds),
        "profitable_folds": int((folds["sharpe"] > 0).sum()),
        "mean_fold_sharpe": round(folds["sharpe"].mean(), 3),
        "pooled_sharpe": round(sharpe(pooled), 3),
        "total_return": round(float(eq.iloc[-1] - 1), 4),
        "max_drawdown": round(max_drawdown(eq), 4),
    }


def buy_and_hold(df: pd.DataFrame, start, end) -> list[dict]:
    span = df.loc[start:end]
    out = []
    for col, label in (("gold_close", "buy & hold gold"), ("silver_close", "buy & hold silver")):
        curve = span[col] / span[col].iloc[0]
        r = curve.pct_change().dropna()
        out.append({"strategy": label, "folds": "-", "profitable_folds": "-",
                    "mean_fold_sharpe": "-", "pooled_sharpe": round(sharpe(r), 3),
                    "total_return": round(float(curve.iloc[-1] - 1), 4),
                    "max_drawdown": round(max_drawdown(curve), 4)})
    return out


def main() -> None:
    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"]).set_index("Date")
    df = df.dropna(subset=["gold_close", "silver_close", "gsr_zscore_30", "gsr_slope"])
    print(f"[data] {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")

    print("[run] ratio rule only")
    base_folds, base_pooled = walk_forward(df, use_regimes=False)
    print("[run] regime-gated (HMM fitted per fold on train only)")
    reg_folds, reg_pooled = walk_forward(df, use_regimes=True)

    rows = [summarise("ratio rule", base_folds, base_pooled),
            summarise("regime-gated", reg_folds, reg_pooled)]
    span = (1 + reg_pooled).cumprod()
    rows += buy_and_hold(df, span.index[0], span.index[-1])
    summary = pd.DataFrame(rows)

    print("\n" + summary.to_string(index=False))

    reg_folds.to_csv(ROOT / "data" / "validation_folds.csv", index=False)
    _write_results(summary, reg_folds, df)
    print(f"\n[saved] docs/results.md and data/validation_folds.csv")


def _write_results(summary: pd.DataFrame, folds: pd.DataFrame, df: pd.DataFrame) -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    lines = [
        "# Results",
        "",
        "Regenerate with `python experiments/validate.py`. Every number below is",
        "written by that script; nothing here is typed by hand.",
        "",
        f"Data: {len(df)} daily bars, {df.index[0].date()} to {df.index[-1].date()}.",
        "",
        "## Method",
        "",
        f"Expanding-window walk-forward. The first {MIN_TRAIN} bars (~3 years) train",
        f"before any out-of-sample call, each test fold is {TEST_SIZE} bars (~1 year),",
        f"and {PURGE_DAYS} bars are purged between train and test so rolling features",
        "cannot leak across the boundary. The regime model is fitted on training rows",
        "only and never sees the fold it scores. Costs are 2bp spread plus 1bp",
        "slippage on every position change.",
        "",
        "The positions come from `strategy.decide`, which is also what the dashboard",
        "calls for its live recommendation, so the figures below describe the",
        "strategy that is actually served.",
        "",
        "## Out-of-sample summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per fold, regime-gated",
        "",
        folds.to_markdown(index=False),
        "",
    ]
    (docs / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
