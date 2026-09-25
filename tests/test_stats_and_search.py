"""Significance statistics, and that the served strategy is the one the search chose."""

import numpy as np
import pandas as pd
import pytest

from backtesting.stats import (bootstrap_sharpe_diff, deflated_sharpe,
                               expected_max_sharpe, probabilistic_sharpe)
from paths import DATA
from strategy import SERVED


def test_psr_orders_sensibly():
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 2500)
    bad = rng.normal(-0.001, 0.01, 2500)
    assert probabilistic_sharpe(good) > 0.99
    assert probabilistic_sharpe(bad) < 0.01
    assert probabilistic_sharpe(good, sr_benchmark=0.5) < probabilistic_sharpe(good)


def test_expected_max_sharpe_grows_with_trials():
    sr = np.random.default_rng(1).normal(0, 0.01, 200)
    assert expected_max_sharpe(sr, 10) < expected_max_sharpe(sr, 100) < expected_max_sharpe(sr, 1000)


def test_dsr_below_psr():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, 2500)
    trials = rng.normal(0.03, 0.01, 100)
    assert deflated_sharpe(r, trials)["dsr"] < probabilistic_sharpe(r)


def test_bootstrap_of_identical_series_is_zero():
    r = pd.Series(np.random.default_rng(3).normal(0.0004, 0.01, 1000))
    out = bootstrap_sharpe_diff(r, r, reps=200)
    assert out["diff"] == pytest.approx(0.0)
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(0.0)


def test_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0.0, 0.01, 3000))
    a = b + 0.002
    out = bootstrap_sharpe_diff(a, b, reps=300)
    assert out["ci_low"] > 0


def test_served_config_is_the_search_choice():
    path = DATA / "search_trials.csv"
    if not path.exists():
        pytest.skip("run experiments/search.py first")
    trials = pd.read_csv(path)
    assert trials["selected"].sum() == 1
    assert trials.loc[trials["selected"], "config"].iloc[0] == SERVED.name
    # the choice used folds 1-11 only
    assert trials.loc[trials["selected"], "sel_sharpe"].iloc[0] == trials["sel_sharpe"].max()
