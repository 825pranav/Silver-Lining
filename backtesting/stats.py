"""
Statistical checks on a backtest: is the Sharpe ratio distinguishable from
luck, and from the benchmark?

* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): probability that
  the true Sharpe exceeds a threshold, allowing for skew and fat tails.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the PSR against the
  Sharpe one would expect from the best of N unskilled trials, where N is the
  number of configurations tried. This is the honest number after a search.
* Stationary block bootstrap (Politis & Romano, 1994) of the difference in
  annualised Sharpe between two return series on the same days. Blocks keep
  the volatility clustering that an i.i.d. bootstrap would destroy.

All Sharpe inputs here are per-period (daily), not annualised, unless a
function says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _moments(returns) -> tuple[float, float, float, int]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    return sr, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), len(r)


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """P(true daily Sharpe > sr_benchmark), given sample skew and kurtosis."""
    sr, skew, kurt, n = _moments(returns)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """
    Expected maximum daily Sharpe among N trials with no skill, given the
    dispersion of the Sharpe ratios actually observed across the trials.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    n = int(n_trials or len(s))
    if n < 2:
        return 0.0
    sd = s.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(returns, trial_sharpes, n_trials: int | None = None) -> dict:
    """DSR = PSR against the expected best-of-N Sharpe under no skill."""
    sr0 = expected_max_sharpe(trial_sharpes, n_trials)
    return {"dsr": probabilistic_sharpe(returns, sr0),
            "sr0_daily": sr0, "sr0_annual": sr0 * np.sqrt(252),
            "n_trials": int(n_trials or len(trial_sharpes))}


def _stationary_indices(n: int, mean_block: float, reps: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Index matrix (reps x n) for the stationary bootstrap."""
    p = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new_block = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _ann_sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=-1, ddof=1)
    return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, reps: int = 2000,
                          mean_block: float = 20.0, seed: int = 0) -> dict:
    """
    Paired stationary block bootstrap of Sharpe(a) - Sharpe(b), annualised.

    Returns the point estimate, a 95% percentile interval and the share of
    resamples in which a did not beat b (a one-sided bootstrap p-value).
    """
    both = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(both)
    rng = np.random.default_rng(seed)
    diffs = []
    for chunk in range(0, reps, 250):           # bounded memory
        k = min(250, reps - chunk)
        idx = _stationary_indices(n, mean_block, k, rng)
        diffs.append(_ann_sharpe(both[idx, 0]) - _ann_sharpe(both[idx, 1]))
    d = np.concatenate(diffs)
    point = float(_ann_sharpe(both[:, 0]) - _ann_sharpe(both[:, 1]))
    return {"diff": point, "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_not_better": float((d <= 0).mean()), "reps": reps,
            "mean_block": mean_block}
