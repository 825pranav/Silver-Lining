# models/ensemble.py
#
# XGBoost ensemble that combines all strategy signals, regime labels, and
# raw features into a final prediction:
#   BUY_GOLD | HOLD | BUY_SILVER   + confidence score
#
# SHAP explainability: every prediction row gets a shap_{feature} column
# showing how much each input feature pushed the prediction in either direction.
#
# Target construction (no lookahead)
# ------------------------------------
# Label = direction of 5-day-forward GSR change, discretised into 3 bins:
#   BUY_GOLD   (class 0) → GSR rises   > +threshold  (gold outperforms)
#   HOLD       (class 1) → GSR moves within ±threshold
#   BUY_SILVER (class 2) → GSR falls   < -threshold  (silver outperforms)
# Forward-return labels are only used during training.  At inference time the
# model predicts from current features with zero future information.
#
# DataFrame contract
# ------------------
# Input  : df produced by strategy_models.build_strategy_signals()
#          (which itself requires regime_detection.fit_and_label() output)
# Output : same df with columns appended:
#            ensemble_signal      str   BUY_GOLD / HOLD / BUY_SILVER
#            ensemble_confidence  float [0, 1]
#            shap_{feature}       float one column per XGB input feature
#
# Requires: pip install xgboost shap
#           (numpy, pandas, scikit-learn already installed)

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is required but not installed.  Run:  pip install xgboost"
    ) from exc

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "shap is required but not installed.  Run:  pip install shap"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP  = {0: "BUY_GOLD", 1: "HOLD", 2: "BUY_SILVER"}
FWD_WINDOW = 5          # trading days for forward GSR return
TRAIN_FRAC = 0.80       # first 80 % of data used for training
THRESHOLD_SIGMA = 0.30  # class boundary at ±0.30 std of fwd GSR returns

# Exact column names consumed from upstream modules
_STRATEGY_SIGNAL_COLS = [
    "mr_signal",  "mr_confidence",
    "mom_signal", "mom_confidence",
    "bo_signal",  "bo_confidence",
]

# gsr_features / momentum_features / volatility_features columns
_RAW_FEATURE_COLS = [
    "gsr_zscore_30",    # gsr_features
    "gsr_zscore_90",    # gsr_features
    "gsr_slope",        # gsr_features
    "vol_ratio",        # gsr_features
    "gsr_ema_slope",    # momentum_features
    "gold_ema_slope",   # momentum_features
    "silver_ema_slope", # momentum_features
    "gold_macd_hist",   # momentum_features
    "rel_strength_20",  # momentum_features
    "gold_breakout",    # momentum_features
    "silver_breakout",  # momentum_features
    "gold_atr_pct",     # volatility_features
    "gold_vol_5",       # volatility_features
    "gold_garch_vol",   # volatility_features
    "garch_vol_ratio",  # volatility_features
]

ALL_FEATURE_COLS = _STRATEGY_SIGNAL_COLS + _RAW_FEATURE_COLS + ["regime_encoded"]


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def _build_targets(df: pd.DataFrame) -> pd.Series:
    """
    5-day forward GSR return → 3-class label (no lookahead in production).

    This is computed once during fit() and never used at inference time.
    """
    fwd_gsr = df["gsr"].pct_change(FWD_WINDOW).shift(-FWD_WINDOW)
    threshold = fwd_gsr.std() * THRESHOLD_SIGMA

    labels = np.where(
        fwd_gsr >  threshold, 0,      # BUY_GOLD   (GSR rises → gold outperforms)
        np.where(
            fwd_gsr < -threshold, 2,  # BUY_SILVER (GSR falls → silver outperforms)
            1,                        # HOLD
        )
    )
    return pd.Series(labels, index=df.index, name="target")


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def _build_feature_matrix(df: pd.DataFrame, regime_encoder: LabelEncoder) -> pd.DataFrame:
    """
    Assemble the XGBoost feature matrix from the full pipeline DataFrame.
    Encodes the regime column and returns a DataFrame with ALL_FEATURE_COLS.
    """
    X = df[_STRATEGY_SIGNAL_COLS + _RAW_FEATURE_COLS].copy()
    X["regime_encoded"] = regime_encoder.transform(df["regime"].fillna("range_bound"))
    return X[ALL_FEATURE_COLS]


# ---------------------------------------------------------------------------
# SHAP extraction
# ---------------------------------------------------------------------------

def _compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-prediction SHAP values.

    For a 3-class model, shap_values has shape (n_samples, n_features, n_classes).
    We return the SHAP slice for the *predicted* class so each row reflects
    "why the model chose the label it did".

    Returns a DataFrame with columns shap_{feature_name}.
    """
    explainer  = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X)

    # Normalise to (n_samples, n_features, n_classes)
    if isinstance(sv, list):
        sv_array = np.stack(sv, axis=2)          # list of (n, f) → (n, f, c)
    else:
        sv_array = sv                             # already (n, f, c) in newer shap

    pred_classes = model.predict(X)              # integer class per row

    shap_for_pred = np.array([
        sv_array[i, :, c] for i, c in enumerate(pred_classes)
    ])

    shap_df = pd.DataFrame(
        shap_for_pred,
        index=X.index,
        columns=[f"shap_{c}" for c in ALL_FEATURE_COLS],
    )
    return shap_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_predict(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train an XGBoost ensemble on the first *train_frac* of data and generate
    predictions + SHAP explanations for the entire DataFrame.

    Parameters
    ----------
    df           : Full pipeline DataFrame (must include regime + strategy signals).
    train_frac   : Fraction of rows used for training (time-ordered, no shuffle).
    n_estimators : XGBoost trees.
    max_depth    : Tree depth.
    learning_rate: XGBoost learning rate (eta).
    random_state : Reproducibility seed.

    Returns
    -------
    df_out : df with the following columns appended (NaN for non-predicting rows):
               ensemble_signal      str
               ensemble_confidence  float
               shap_{feature}       float  (one column per feature in ALL_FEATURE_COLS)
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    required = _STRATEGY_SIGNAL_COLS + _RAW_FEATURE_COLS + ["regime", "gsr"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"fit_and_predict(): DataFrame missing columns: {missing}\n"
            f"Run the full pipeline: gsr_features → momentum_features → "
            f"volatility_features → regime_detection → strategy_models → ensemble"
        )

    # ── Encode regime ────────────────────────────────────────────────────────
    regime_encoder = LabelEncoder()
    known_regimes  = ["crisis", "range_bound", "risk_on", "trending"]
    regime_encoder.fit(known_regimes)

    # ── Build feature matrix and targets ─────────────────────────────────────
    X_all = _build_feature_matrix(df, regime_encoder)
    y_all = _build_targets(df)

    # Drop rows with any NaN in features or targets
    valid_mask = X_all.notna().all(axis=1) & y_all.notna()
    X_valid    = X_all[valid_mask].astype(float)
    y_valid    = y_all[valid_mask].astype(int)

    # Time-ordered train / test split (no shuffle — strict no-lookahead)
    split_idx  = int(len(X_valid) * train_frac)
    X_train, y_train = X_valid.iloc[:split_idx], y_valid.iloc[:split_idx]
    X_test             = X_valid.iloc[split_idx:]

    print(f"🚀 Training XGBoost ensemble...")
    print(f"   Train: {len(X_train)} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test : {len(X_test)}  rows  ({X_test.index[0].date() if len(X_test) else 'n/a'} → "
          f"{X_test.index[-1].date() if len(X_test) else 'n/a'})")
    print(f"   Class distribution — "
          f"BUY_GOLD: {(y_train==0).sum()}  "
          f"HOLD: {(y_train==1).sum()}  "
          f"BUY_SILVER: {(y_train==2).sum()}")

    # ── Fit XGBoost ──────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_valid.iloc[split_idx:])] if len(X_test) > 0 else None,
        verbose=False,
    )

    # ── Predictions on all valid rows ────────────────────────────────────────
    pred_classes = model.predict(X_valid)
    pred_proba   = model.predict_proba(X_valid)

    signals      = pd.Series(pred_classes, index=X_valid.index).map(LABEL_MAP)
    confidence   = pd.Series(
        pred_proba[np.arange(len(pred_classes)), pred_classes],
        index=X_valid.index,
    )

    # ── SHAP values ──────────────────────────────────────────────────────────
    print("🔍 Computing SHAP explanations...")
    shap_df = _compute_shap(model, X_valid)

    # ── Evaluate on test set ─────────────────────────────────────────────────
    if len(X_test) > 0:
        test_preds  = pred_classes[split_idx:]
        test_labels = y_valid.iloc[split_idx:].values
        accuracy    = (test_preds == test_labels).mean()
        print(f"\n📊 Out-of-sample accuracy: {accuracy:.2%}")

        # Per-class accuracy
        for cls, name in LABEL_MAP.items():
            mask = test_labels == cls
            if mask.sum() > 0:
                cls_acc = (test_preds[mask] == cls).mean()
                print(f"   {name:<15} acc={cls_acc:.2%}  n={mask.sum()}")

    # ── Assemble output ──────────────────────────────────────────────────────
    df_out = df.copy()
    df_out["ensemble_signal"]     = signals
    df_out["ensemble_confidence"] = confidence

    for col in shap_df.columns:
        df_out[col] = shap_df[col]

    print(f"\n✅ Ensemble complete. New columns: ensemble_signal, ensemble_confidence, "
          f"+ {len(shap_df.columns)} shap_* columns")

    return df_out


# ---------------------------------------------------------------------------
# SHAP summary helper (optional post-hoc analysis)
# ---------------------------------------------------------------------------

def shap_feature_importance(df_out: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute mean |SHAP| across all rows and return a ranked importance table.

    Parameters
    ----------
    df_out : Output of fit_and_predict() (must have shap_* columns).
    top_n  : Number of top features to return.

    Returns
    -------
    DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    shap_cols = [c for c in df_out.columns if c.startswith("shap_")]
    if not shap_cols:
        raise ValueError("No shap_* columns found. Run fit_and_predict() first.")

    importance = (
        df_out[shap_cols]
        .abs()
        .mean()
        .rename(index=lambda c: c.replace("shap_", ""))
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    importance.columns = ["feature", "mean_abs_shap"]
    return importance


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    try:
        signals_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "features_with_signals.csv")
        )
        print(f"🚀 Loading {signals_path}...")
        df = pd.read_csv(signals_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        print(f"   {len(df)} rows, {len(df.columns)} columns")

        df_out = fit_and_predict(df)

        print("\n--- Ensemble Snapshot (last 10 rows) ---")
        snap = ["regime", "ensemble_signal", "ensemble_confidence",
                "mr_signal", "mom_signal", "bo_signal"]
        print(df_out[snap].tail(10).to_string())

        print("\n📊 Top-10 SHAP Feature Importances:")
        print(shap_feature_importance(df_out, top_n=10).to_string(index=False))

        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "ensemble_predictions.csv")
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path)
        print(f"\n💾 Saved to {out_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
