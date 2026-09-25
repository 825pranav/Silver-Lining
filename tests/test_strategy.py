"""Decision rules, the path simulator and its timing."""

import numpy as np
import pandas as pd
import pytest

import strategy as S
from strategy import GOLD, LEGACY, SILVER, StrategyConfig, decide, simulate


def _row(**kw):
    base = {"gsr_zscore_30": 0.0, "gsr_slope": 0.1, "gold_vol_20": 0.02,
            "silver_vol_20": 0.04, "gold_mom_252": 0.1, "silver_mom_252": 0.1}
    base.update(kw)
    return base


@pytest.mark.parametrize("regime,asset", [("crisis", GOLD), ("risk_on", SILVER),
                                          ("range_bound", None)])
def test_legacy_regime_mapping(regime, asset):
    d = decide(_row(), regime)
    assert d.asset == asset
    assert d.size == pytest.approx(S.position_size(_row(), asset))


def test_legacy_trending_defers_to_ratio_rule():
    assert decide(_row(gsr_zscore_30=1.5, gsr_slope=0.1), "trending").asset == SILVER
    assert decide(_row(gsr_zscore_30=-1.5, gsr_slope=0.1), "trending").asset == GOLD
    assert decide(_row(gsr_zscore_30=0.0, gsr_slope=0.8), None).asset == GOLD
    assert decide(_row(gsr_zscore_30=0.0, gsr_slope=-0.8), None).asset == SILVER


def test_size_is_inverse_vol_and_capped():
    assert decide(_row(gold_vol_20=0.02), "crisis").size == pytest.approx(0.5)
    assert decide(_row(gold_vol_20=0.005), "crisis").size == pytest.approx(1.0)


def test_probs_argmax_matches_label_when_hard():
    probs = {"crisis": 0.1, "range_bound": 0.2, "risk_on": 0.6, "trending": 0.1}
    assert decide(_row(), probs=probs).asset == SILVER


def test_soft_weights_follow_probabilities():
    cfg = StrategyConfig(soft=True)
    probs = {"crisis": 0.5, "range_bound": 0.25, "risk_on": 0.25, "trending": 0.0}
    d = decide(_row(), probs=probs, config=cfg)
    assert d.gold_weight == pytest.approx(0.5 * 0.5)      # 50% x (0.01/0.02)
    assert d.silver_weight == pytest.approx(0.25 * 0.25)  # 25% x (0.01/0.04)


def test_trend_gate_drops_metal_with_negative_momentum():
    cfg = StrategyConfig(trend_lookback=252)
    assert decide(_row(gold_mom_252=-0.01), "crisis", config=cfg).asset is None
    assert decide(_row(gold_mom_252=0.01), "crisis", config=cfg).asset == GOLD
    # no history yet: the gate does not act
    assert decide(_row(gold_mom_252=np.nan), "crisis", config=cfg).asset == GOLD


def test_drawdown_derisk_halves_exposure():
    cfg = StrategyConfig(dd_derisk=True)
    full = decide(_row(), "crisis", config=cfg, drawdown=-0.05).size
    cut = decide(_row(), "crisis", config=cfg, drawdown=-0.15).size
    assert cut == pytest.approx(full * S.DD_SCALE)


def test_universes():
    core = StrategyConfig(universe="gold_core")
    only = StrategyConfig(universe="gold_only")
    trending = _row(gsr_zscore_30=1.5)                    # ratio rule would say silver
    assert decide(trending, "trending", config=core).asset == GOLD
    assert decide(_row(), "risk_on", config=core).asset == SILVER
    assert decide(_row(), "risk_on", config=only).asset == GOLD
    assert decide(_row(), "range_bound", config=only).asset is None
    assert decide(_row(), None, config=only).asset == GOLD


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame({
        "gold_close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        "silver_close": 20 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))),
    }, index=idx)
    df["gold_vol_20"] = df["gold_close"].pct_change().rolling(20).std()
    df["silver_vol_20"] = df["silver_close"].pct_change().rolling(20).std()
    df["gsr_zscore_30"] = 0.0
    df["gsr_slope"] = 0.1
    return S.add_trend_features(df, lookbacks=(252,))


def test_simulate_trades_one_day_after_the_signal():
    df = _prices()
    sim = simulate(df, config=S.SERVED, cost=0.0)
    rg = df["gold_close"].pct_change().to_numpy()
    w = sim["gold_weight"].to_numpy()
    for i in range(300, 320):
        assert sim["ret"].iloc[i] == pytest.approx(w[i - 2] * rg[i])


def test_simulate_charges_turnover():
    df = _prices()
    free = simulate(df, config=LEGACY, cost=0.0)
    paid = simulate(df, config=LEGACY, cost=0.001)
    assert np.allclose(free["ret"] - paid["ret"], 0.001 * paid["turnover"])
    assert paid["turnover"].sum() > 0


def test_simulate_does_not_look_ahead():
    df = _prices()
    k = 350
    shocked = df.copy()
    shocked.iloc[k:, shocked.columns.get_loc("gold_close")] *= 3.0
    shocked = S.add_trend_features(shocked[["gold_close", "silver_close", "gold_vol_20",
                                            "silver_vol_20", "gsr_zscore_30", "gsr_slope"]],
                                   lookbacks=(252,))
    for cfg in (S.SERVED, LEGACY, StrategyConfig(trend_lookback=252, dd_derisk=True)):
        a, b = simulate(df, config=cfg), simulate(shocked, config=cfg)
        pd.testing.assert_series_equal(a["gold_weight"].iloc[:k], b["gold_weight"].iloc[:k])
        pd.testing.assert_series_equal(a["ret"].iloc[:k], b["ret"].iloc[:k])


def test_trend_features_are_causal():
    df = _prices()
    shocked = df.copy()
    shocked.iloc[300:, 0] *= 2
    a = S.add_trend_features(df, (252,))
    b = S.add_trend_features(shocked, (252,))
    pd.testing.assert_series_equal(a["gold_mom_252"].iloc[:300], b["gold_mom_252"].iloc[:300])
