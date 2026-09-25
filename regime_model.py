"""
Regime model with a fit/predict split, so training and serving are separable.

`models.regime_detection.fit_and_label` fits and labels one frame in a single
call. That is fine for a chart, but it cannot express "fit on history, apply to
today" — and fitting on a frame that includes the rows being scored leaks the
test period into the state definitions. This wrapper keeps the project's
feature list and its economic labelling, and adds:

    model = RegimeModel().fit(train_df)      # states learned from train only
    probs = model.predict_proba(df)          # filtered P(regime_t | data <= t)
    labels = model.predict(df)               # argmax of the filtered posterior
    model.save(path) / RegimeModel.load(path)

Causality
---------
Two things an earlier version got wrong, both fixed here:

* Decoding. `GaussianHMM.predict` runs Viterbi over the whole frame it is
  given, so the label on day t depended on bars after t. Every label and
  probability now comes from the forward filter: the posterior on day t uses
  observations up to and including day t only. Appending future rows never
  changes an earlier row's output (tested).
* GARCH. `features/volatility_features.py` fits GARCH(1,1) once on the full
  history, so its parameters had seen the test years. The model refits
  GARCH(1,1) on training returns only and rebuilds `gold_garch_vol` and
  `garch_vol_ratio` with those parameters, running the recursion forward.

Options (all fitted on training rows only):

* ``n_states`` — an int, or ``"bic"`` to pick from ``BIC_CANDIDATES`` by the
  Bayesian information criterion on the training fold.
* ``sticky`` — Dirichlet pseudo-counts added to the transition matrix
  diagonal (0 = off). Encourages persistent regimes.
* ``sticky`` is scaled by training length: the diagonal receives
  ``sticky * rows / states`` pseudo-counts, so 0.5 means "as if every state
  had been seen staying put half again as often".
* ``scale_free`` — divide the price-denominated features (gold/silver EMA
  slopes, gold MACD histogram) by the metal's price. In dollars they grow with
  the price level, so a scaler fitted on 2001-2010 sees 2025 values as extreme
  outliers.
* ``n_init`` — random restarts; the best training log-likelihood is kept.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.regime_detection import REGIME_FEATURES, _assign_regimes

warnings.filterwarnings("ignore")

DEFAULT_STATES = 4
BIC_CANDIDATES = (3, 4, 5)
MIN_ROWS_PER_STATE = 25          # below this the fit is not worth trusting
REGIMES = ("crisis", "range_bound", "risk_on", "trending")
GARCH_COLS = ("gold_garch_vol", "garch_vol_ratio")
PRICE_SCALED = {"gold_ema_slope": "gold_close", "silver_ema_slope": "silver_close",
                "gold_macd_hist": "gold_close"}


# ── train-only GARCH ─────────────────────────────────────────────────────────

def fit_garch(returns: pd.Series) -> dict:
    """GARCH(1,1) parameters fitted on the given (training) returns, in %."""
    r = returns.dropna() * 100
    try:
        from arch import arch_model

        res = arch_model(r, vol="Garch", p=1, q=1, rescale=False).fit(disp="off")
        p = res.params
        return {"mu": float(p["mu"]), "omega": float(p["omega"]),
                "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
                "var0": float(r.var())}
    except Exception:
        # No arch, or the optimiser failed: an EWMA-like recursion (lambda 0.94)
        # keeps the feature defined and still causal.
        return {"mu": float(r.mean()), "omega": 0.0, "alpha": 0.06,
                "beta": 0.94, "var0": float(r.var())}


def garch_vol(returns: pd.Series, params: dict) -> pd.Series:
    """
    Conditional volatility from fixed parameters, run forward over `returns`.

    sigma^2_t = omega + alpha * e_{t-1}^2 + beta * sigma^2_{t-1}, so the value
    on day t uses returns up to t-1 only, matching arch's convention.
    """
    e = (returns.fillna(0.0).to_numpy() * 100) - params["mu"]
    out = np.empty(len(e))
    s2 = params["var0"]
    for i in range(len(e)):
        out[i] = s2
        s2 = params["omega"] + params["alpha"] * e[i] ** 2 + params["beta"] * s2
    return pd.Series(np.sqrt(out) / 100, index=returns.index)


# ── the model ───────────────────────────────────────────────────────────────

class RegimeModel:
    """Gaussian HMM over the project's regime features, fit and predict split."""

    def __init__(self, n_states: int | str = DEFAULT_STATES, random_state: int = 42,
                 sticky: float = 0.0, n_init: int = 1, refit_garch: bool = True,
                 scale_free: bool = False) -> None:
        self.n_states = n_states
        self.scale_free = scale_free
        self.random_state = random_state
        self.sticky = sticky
        self.n_init = n_init
        self.refit_garch = refit_garch
        self.columns: list[str] = []
        self.mapping: dict[int, str] = {}
        self.garch: dict[str, dict] = {}
        self.chosen_states: int | None = None
        self.bic_scores: dict[int, float] = {}
        self._model = None
        self._scaler = None
        self.fitted_rows = 0
        self.fitted_through: pd.Timestamp | None = None

    # ── features ─────────────────────────────────────────────────

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime features, with GARCH columns rebuilt from train-fit parameters."""
        out = df[self.columns].copy()
        if self.scale_free:
            for col, px in PRICE_SCALED.items():
                if col in out and px in df:
                    out[col] = out[col] / df[px]
        if self.garch and {"gold_close", "silver_close"} <= set(df.columns):
            g = garch_vol(df["gold_close"].pct_change(), self.garch["gold"])
            s = garch_vol(df["silver_close"].pct_change(), self.garch["silver"])
            if "gold_garch_vol" in out:
                out["gold_garch_vol"] = g
            if "garch_vol_ratio" in out:
                out["garch_vol_ratio"] = s / g
            out = out.iloc[1:]          # first row has no prior return
        return out.dropna()

    # ── training ─────────────────────────────────────────────────

    def _fit_hmm(self, X: np.ndarray, k: int):
        from hmmlearn.hmm import GaussianHMM

        prior = 1.0 + self.sticky * len(X) / k * np.eye(k) if self.sticky > 0 else 1.0
        best, best_ll = None, -np.inf
        for i in range(self.n_init):
            m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=200,
                            random_state=self.random_state + i, transmat_prior=prior)
            try:
                m.fit(X)
                ll = m.score(X)
            except Exception:
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        return best

    def fit(self, df: pd.DataFrame) -> RegimeModel:
        from sklearn.preprocessing import StandardScaler

        self.columns = [c for c in REGIME_FEATURES if c in df.columns]
        self.garch = {}
        if self.refit_garch and {"gold_close", "silver_close"} <= set(df.columns):
            self.garch = {"gold": fit_garch(df["gold_close"].pct_change()),
                          "silver": fit_garch(df["silver_close"].pct_change())}
        train = self._features(df)

        candidates = BIC_CANDIDATES if self.n_states == "bic" else (int(self.n_states),)
        if len(train) < max(candidates) * MIN_ROWS_PER_STATE:
            # Too little history to distinguish states; predict() will fall
            # back to "unknown" and the ratio rule carries the decision.
            self._model = None
            return self

        self._scaler = StandardScaler().fit(train)
        X = self._scaler.transform(train)

        self._model, self.bic_scores = None, {}
        best_bic = np.inf
        for k in candidates:
            m = self._fit_hmm(X, k)
            if m is None:
                continue
            bic = float(m.bic(X)) if len(candidates) > 1 else 0.0
            self.bic_scores[k] = bic
            if bic < best_bic:
                best_bic, self._model, self.chosen_states = bic, m, k
        if self._model is None:
            return self

        # Label states from training means only, using the project's own rules.
        states = self._model.predict(X)
        means = pd.DataFrame(train.values, columns=self.columns).groupby(states).mean()
        self.mapping = _assign_regimes(means)
        self.fitted_rows = len(train)
        self.fitted_through = train.index[-1] if isinstance(train.index, pd.DatetimeIndex) else None
        return self

    # ── inference ────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _filter(self, X: np.ndarray) -> np.ndarray:
        """Forward filter: row t is P(state_t | x_1..x_t). No smoothing."""
        m = self._model
        var = np.asarray(m._covars_) if m.covariance_type == "diag" else \
            np.diagonal(m.covars_, axis1=1, axis2=2)
        mu = m.means_
        # log N(x | mu_k, diag(var_k)) for every row and state
        logb = -0.5 * (np.log(2 * np.pi * var).sum(1)[None, :]
                       + (((X[:, None, :] - mu[None, :, :]) ** 2) / var[None, :, :]).sum(2))
        b = np.exp(logb - logb.max(1, keepdims=True))
        out = np.empty_like(b)
        a = m.startprob_ * b[0]
        out[0] = a / a.sum()
        T = m.transmat_
        for t in range(1, len(b)):
            a = (out[t - 1] @ T) * b[t]
            s = a.sum()
            out[t] = a / s if s > 0 else np.full(len(a), 1 / len(a))
        return out

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtered regime probabilities, one column per regime name.

        Rows without usable features are NaN. States that share a name (for
        example two "trending" states when BIC picks five) are summed.
        """
        cols = list(REGIMES)
        if not self.is_fitted:
            return pd.DataFrame(np.nan, index=df.index, columns=cols)
        usable = self._features(df)
        if usable.empty:
            return pd.DataFrame(np.nan, index=df.index, columns=cols)
        post = self._filter(self._scaler.transform(usable))
        out = pd.DataFrame(0.0, index=usable.index, columns=cols)
        for s, name in self.mapping.items():
            out[name] += post[:, s]
        return out.reindex(df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Regime label per row (argmax of the filtered posterior)."""
        probs = self.predict_proba(df)
        ok = probs.notna().all(axis=1)
        labels = pd.Series("unknown", index=df.index, dtype=object)
        if ok.any():
            labels[ok] = probs[ok].idxmax(axis=1)
        return labels.astype(str)

    def predict_viterbi(self, df: pd.DataFrame) -> pd.Series:
        """
        Viterbi labels over the whole frame. NOT causal: a row's label depends
        on later rows. Kept only so experiments/validate.py can reproduce the
        original, biased measurement in its lookahead audit.
        """
        if not self.is_fitted:
            return pd.Series("unknown", index=df.index)
        usable = self._features(df)
        states = self._model.predict(self._scaler.transform(usable))
        labels = pd.Series([self.mapping.get(int(s), "unknown") for s in states],
                           index=usable.index)
        return labels.reindex(df.index, fill_value="unknown")

    def predict_latest(self, df: pd.DataFrame) -> str:
        """Regime for the most recent usable row — what the dashboard shows."""
        labels = self.predict(df)
        known = labels[labels != "unknown"]
        return str(known.iloc[-1]) if len(known) else "unknown"

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({k: v for k, v in self.__dict__.items()}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeModel:
        import joblib

        blob = joblib.load(Path(path))
        obj = cls()
        obj.__dict__.update(blob)
        return obj

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "RegimeModel(unfitted)"
        through = self.fitted_through.date() if self.fitted_through is not None else "?"
        return (f"RegimeModel({self.chosen_states} states, sticky={self.sticky}, "
                f"{self.fitted_rows} rows, through {through})")
