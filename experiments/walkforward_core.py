"""
Walk-forward plumbing shared by `experiments/search.py` and `experiments/validate.py`.

Folds are expanding-window: at least MIN_TRAIN bars of training, PURGE_DAYS
dropped between train and test, TEST_SIZE-bar test folds. For every fold a
fresh `RegimeModel` is fitted on that fold's training rows only, and its
filtered (forward-only) regime probabilities are taken for the test rows. The
strategy then runs as one continuous out-of-sample path through
`strategy.simulate`, so positions and the drawdown state carry across fold
boundaries the way they would live.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import cagr, hit_rate, max_drawdown, sharpe  # noqa: E402
from paths import DATA, FEATURE_PATH  # noqa: E402
from regime_model import RegimeModel  # noqa: E402
from strategy import StrategyConfig, add_trend_features, simulate  # noqa: E402

PURGE_DAYS = 20      # dropped between train and test
MIN_TRAIN = 750      # ~3 years before the first out-of-sample call
TEST_SIZE = 250      # ~1 trading year per fold
SELECTION_FOLDS = range(1, 12)     # folds 1-11: configuration is chosen here
CONFIRMATION_FOLDS = range(12, 24) # folds 12-23: never used for choosing
CACHE = DATA / "regime_probs_cache.pkl"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"]).set_index("Date")
    df = df.dropna(subset=["gold_close", "silver_close", "gsr_zscore_30", "gsr_slope"])
    return add_trend_features(df)


def folds(df: pd.DataFrame) -> list[dict]:
    """Fold boundaries as integer positions: train [0, t0-purge), test [t0, t1)."""
    out, t0, k = [], MIN_TRAIN + PURGE_DAYS, 0
    while t0 + 60 <= len(df):
        k += 1
        t1 = min(t0 + TEST_SIZE, len(df))
        out.append({"fold": k, "train_end": t0 - PURGE_DAYS, "t0": t0, "t1": t1,
                    "test_start": df.index[t0], "test_end": df.index[t1 - 1]})
        t0 = t1
    return out


def _key(df: pd.DataFrame, regime_kwargs: dict) -> str:
    sig = f"{len(df)}|{df.index[-1]}|{float(df['gold_close'].sum()):.6f}|{sorted(regime_kwargs.items())}"
    return hashlib.sha1(sig.encode()).hexdigest()


def oos_regime_probs(df: pd.DataFrame, regime_kwargs: dict,
                     use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtered regime probabilities on every test row, each fold's from a model
    fitted on that fold's training rows. Also returns per-fold model facts.
    """
    cache = {}
    if use_cache and CACHE.exists():
        try:
            cache = pickle.loads(CACHE.read_bytes())
        except Exception:
            cache = {}
    key = _key(df, regime_kwargs)
    if key in cache:
        return cache[key]

    parts, info = [], []
    for f in folds(df):
        model = RegimeModel(**regime_kwargs).fit(df.iloc[: f["train_end"]])
        # The filter runs forward from the start of history, so the test rows'
        # probabilities condition only on bars up to each row.
        probs = model.predict_proba(df.iloc[: f["t1"]]).iloc[f["t0"]: f["t1"]]
        parts.append(probs)
        info.append({"fold": f["fold"], "states": model.chosen_states,
                     "mapping": dict(model.mapping)})
    out = (pd.concat(parts), pd.DataFrame(info))
    cache[key] = out
    if use_cache:
        CACHE.write_bytes(pickle.dumps(cache))
    return out


def oos_span(df: pd.DataFrame) -> pd.DataFrame:
    fs = folds(df)
    return df.iloc[fs[0]["t0"]: fs[-1]["t1"]]


def run_config(df: pd.DataFrame, config: StrategyConfig,
               probs: pd.DataFrame | None = None, use_regimes: bool = True) -> pd.DataFrame:
    """One continuous out-of-sample path for `config`."""
    span = oos_span(df)
    use_regimes = use_regimes and config.uses_regimes
    if use_regimes and probs is None:
        probs, _ = oos_regime_probs(df, config.regime_kwargs)
    return simulate(span, probs=probs if use_regimes else None, config=config)


def fold_of(index: pd.DatetimeIndex, df: pd.DataFrame) -> pd.Series:
    fid = pd.Series(0, index=index)
    for f in folds(df):
        fid[(index >= f["test_start"]) & (index <= f["test_end"])] = f["fold"]
    return fid


def summarise(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    return {"sharpe": sharpe(ret), "total_return": float(eq.iloc[-1] - 1),
            "cagr": cagr(eq), "max_drawdown": max_drawdown(pd.concat([pd.Series([1.0]), eq])),
            "ann_vol": float(ret.std() * np.sqrt(252))}


def per_fold(ret: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    fid = fold_of(ret.index, df)
    rows = []
    for f in folds(df):
        r = ret[fid == f["fold"]]
        eq = (1 + r).cumprod()
        rows.append({"fold": f["fold"], "test_start": f["test_start"].date(),
                     "test_end": f["test_end"].date(), "sharpe": round(sharpe(r), 3),
                     "cagr": round(cagr(eq), 4),
                     "max_drawdown": round(max_drawdown(pd.concat([pd.Series([1.0]), eq])), 4),
                     "hit_rate": round(hit_rate(r), 4), "days": len(r)})
    return pd.DataFrame(rows)


def buy_and_hold_returns(df: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    span = oos_span(df)
    out = {}
    for col, name in (("gold_close", "gold"), ("silver_close", "silver")):
        r = span[col].pct_change().fillna(0.0)
        out[name] = r.reindex(index)
    return out


def oos_returns(sim: pd.DataFrame) -> pd.Series:
    """Net returns from the first bar a decision can have earned anything."""
    from strategy import EXECUTION_LAG
    return sim["ret"].iloc[1 + EXECUTION_LAG:]
