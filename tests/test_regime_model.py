"""The regime model must not use data from after the day it labels."""

import numpy as np
import pandas as pd
import pytest

from models.regime_detection import _assign_regimes
from paths import FEATURE_PATH
from regime_model import RegimeModel, fit_garch, garch_vol


@pytest.fixture(scope="module")
def features():
    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"]).set_index("Date")
    return df.iloc[:1800]


@pytest.fixture(scope="module")
def model(features):
    return RegimeModel().fit(features.iloc[:1000])


def test_filtered_probabilities_ignore_future_rows(model, features):
    short = model.predict_proba(features.iloc[:1500])
    long = model.predict_proba(features)
    pd.testing.assert_frame_equal(short, long.iloc[:1500])


def test_probabilities_are_distributions(model, features):
    p = model.predict_proba(features).dropna()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert set(p.columns) == {"crisis", "range_bound", "risk_on", "trending"}


def test_predict_is_argmax_of_filter(model, features):
    p = model.predict_proba(features)
    labels = model.predict(features)
    ok = p.notna().all(axis=1)
    assert (labels[ok] == p[ok].idxmax(axis=1)).all()


def test_garch_parameters_come_from_training_rows(features):
    m = RegimeModel().fit(features.iloc[:1000])
    ref = fit_garch(features["gold_close"].iloc[:1000].pct_change())
    assert m.garch["gold"] == pytest.approx(ref)


def test_garch_recursion_is_causal():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 500))
    params = {"mu": 0.0, "omega": 0.02, "alpha": 0.05, "beta": 0.9, "var0": 1.0}
    a = garch_vol(r, params)
    r2 = r.copy()
    r2.iloc[300:] *= 10
    b = garch_vol(r2, params)
    # value at t uses returns up to t-1, so t=300 is still unchanged
    pd.testing.assert_series_equal(a.iloc[:301], b.iloc[:301])


def test_sticky_prior_raises_persistence(features):
    plain = RegimeModel().fit(features.iloc[:1000])
    sticky = RegimeModel(sticky=0.5).fit(features.iloc[:1000])
    assert np.diag(sticky._model.transmat_).mean() > np.diag(plain._model.transmat_).mean()


def test_bic_picks_from_candidates(features):
    m = RegimeModel(n_states="bic").fit(features.iloc[:1000])
    assert m.chosen_states in (3, 4, 5)
    assert set(m.bic_scores) == {3, 4, 5}


@pytest.mark.parametrize("k", [3, 4, 5])
def test_every_state_gets_a_name(k):
    rng = np.random.default_rng(k)
    means = pd.DataFrame({"gold_atr_pct": rng.random(k), "gsr_zscore_30": rng.random(k)})
    mapping = _assign_regimes(means)
    assert set(mapping) == set(range(k))
    assert {"crisis", "range_bound", "risk_on"} <= set(mapping.values())


def test_save_load_round_trip(model, features, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    loaded = RegimeModel.load(path)
    pd.testing.assert_frame_equal(model.predict_proba(features), loaded.predict_proba(features))
