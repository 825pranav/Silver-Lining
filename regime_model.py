"""
Regime model with a fit/predict split, so training and serving are separable.

`models.regime_detection.fit_and_label` fits and labels one frame in a single
call. That is fine for a chart, but it cannot express "fit on history, apply to
today" — and fitting on a frame that includes the rows being scored leaks the
test period into the state definitions. This wrapper keeps the project's
feature list and its economic labelling, and adds:

    model = RegimeModel().fit(train_df)      # states learned from train only
    labels = model.predict(today_df)         # applied to unseen rows
    model.save(path) / RegimeModel.load(path)

Serving loads a model fitted on everything up to the last training run, so the
dashboard does not refit a Gaussian HMM on every page view.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.regime_detection import REGIME_FEATURES, _assign_regimes

warnings.filterwarnings("ignore")

DEFAULT_STATES = 4
MIN_ROWS_PER_STATE = 25          # below this the fit is not worth trusting


class RegimeModel:
    """Gaussian HMM over the project's regime features, fit and predict split."""

    def __init__(self, n_states: int = DEFAULT_STATES, random_state: int = 42) -> None:
        self.n_states = n_states
        self.random_state = random_state
        self.columns: list[str] = []
        self.mapping: dict[int, str] = {}
        self._model = None
        self._scaler = None
        self.fitted_rows = 0
        self.fitted_through: pd.Timestamp | None = None

    # ── training ─────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> RegimeModel:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        self.columns = [c for c in REGIME_FEATURES if c in df.columns]
        train = df[self.columns].dropna()
        if len(train) < self.n_states * MIN_ROWS_PER_STATE:
            # Too little history to distinguish states; predict() will fall
            # back to "unknown" and the ratio rule carries the decision.
            self._model = None
            return self

        self._scaler = StandardScaler().fit(train)
        self._model = GaussianHMM(
            n_components=self.n_states, covariance_type="diag",
            n_iter=200, random_state=self.random_state,
        )
        self._model.fit(self._scaler.transform(train))

        # Label states from training means only, using the project's own rules.
        states = self._model.predict(self._scaler.transform(train))
        means = pd.DataFrame(train.values, columns=self.columns).groupby(states).mean()
        self.mapping = _assign_regimes(means)
        self.fitted_rows = len(train)
        self.fitted_through = train.index[-1] if isinstance(train.index, pd.DatetimeIndex) else None
        return self

    # ── inference ────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Regime label per row. 'unknown' where the model cannot speak."""
        if not self.is_fitted:
            return pd.Series("unknown", index=df.index)

        usable = df[self.columns].dropna()
        if usable.empty:
            return pd.Series("unknown", index=df.index)

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
        joblib.dump({
            "n_states": self.n_states, "random_state": self.random_state,
            "columns": self.columns, "mapping": self.mapping,
            "model": self._model, "scaler": self._scaler,
            "fitted_rows": self.fitted_rows, "fitted_through": self.fitted_through,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeModel:
        import joblib

        blob = joblib.load(Path(path))
        obj = cls(n_states=blob["n_states"], random_state=blob["random_state"])
        obj.columns = blob["columns"]
        obj.mapping = blob["mapping"]
        obj._model = blob["model"]
        obj._scaler = blob["scaler"]
        obj.fitted_rows = blob.get("fitted_rows", 0)
        obj.fitted_through = blob.get("fitted_through")
        return obj

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "RegimeModel(unfitted)"
        through = self.fitted_through.date() if self.fitted_through is not None else "?"
        return (f"RegimeModel({self.n_states} states, {self.fitted_rows} rows, "
                f"through {through})")
