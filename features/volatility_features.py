# features/volatility_features.py
#
# Reads features.csv (output of momentum_features.py), adds volatility features,
# and writes back to features.csv.
#
# Run order: price_fetcher.py → gsr_features.py → momentum_features.py → volatility_features.py
#
# ATR note: price_fetcher.py only stores Close prices (no High/Low), so ATR is
# approximated as rolling mean of |close_t - close_{t-1}| (true range, close-only).
#
# GARCH note: requires the `arch` package (pip install arch).
# Falls back to EWMA volatility if arch is not installed.

import pandas as pd
import numpy as np

FEATURE_PATH = "../data/features.csv"


def compute_atr(close, window=14):
    """Close-only ATR: rolling mean of absolute daily price change."""
    tr = close.diff().abs()
    return tr.rolling(window).mean()


def compute_garch_vol(returns_series, label=""):
    """
    Fit GARCH(1,1) on a returns series and return conditional volatility.
    Returns daily vol in the same scale as pct_change() (i.e., 0.01 = 1%).
    Falls back to EWMA if arch is not installed.
    """
    try:
        from arch import arch_model
        scaled = returns_series.dropna() * 100  # scale to percent for numerical stability
        model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
        res = model.fit(disp="off")
        vol = res.conditional_volatility / 100  # rescale back
        print(f"  GARCH(1,1) fitted for {label} ({len(scaled)} observations)")
        return vol
    except ImportError:
        print(f"  arch library not found — using EWMA vol for {label}")
        return returns_series.ewm(span=20).std()


def build_volatility_features():

    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    gold_ret = df["gold_close"].pct_change()
    silver_ret = df["silver_close"].pct_change()

    # ---- ATR (close-only approximation) ----
    df["gold_atr_14"] = compute_atr(df["gold_close"], 14)
    df["silver_atr_14"] = compute_atr(df["silver_close"], 14)

    # Normalised ATR: ATR as % of price (comparable across time)
    df["gold_atr_pct"] = df["gold_atr_14"] / df["gold_close"]
    df["silver_atr_pct"] = df["silver_atr_14"] / df["silver_close"]

    # ---- Rolling Volatility (complement gsr_features' 20-day vol) ----
    # gsr_features.py already computes gold_vol_20 / silver_vol_20 / vol_ratio
    df["gold_vol_5"] = gold_ret.rolling(5).std()
    df["gold_vol_60"] = gold_ret.rolling(60).std()
    df["silver_vol_5"] = silver_ret.rolling(5).std()
    df["silver_vol_60"] = silver_ret.rolling(60).std()

    # ---- Silver / Gold Vol Ratio (short and long window) ----
    # vol_ratio in gsr_features uses 20-day; add 5-day and 60-day here
    df["vol_ratio_5"] = df["silver_vol_5"] / df["gold_vol_5"]
    df["vol_ratio_60"] = df["silver_vol_60"] / df["gold_vol_60"]

    # ---- GARCH Volatility ----
    print("Fitting GARCH models...")
    gold_garch = compute_garch_vol(gold_ret, label="gold")
    silver_garch = compute_garch_vol(silver_ret, label="silver")

    df["gold_garch_vol"] = gold_garch.reindex(df.index)
    df["silver_garch_vol"] = silver_garch.reindex(df.index)

    # GARCH vol ratio: silver vs gold modelled vol
    df["garch_vol_ratio"] = df["silver_garch_vol"] / df["gold_garch_vol"]

    df.dropna(inplace=True)

    new_cols = [
        "gold_atr_14", "silver_atr_14", "gold_atr_pct", "silver_atr_pct",
        "gold_vol_5", "gold_vol_60", "silver_vol_5", "silver_vol_60",
        "vol_ratio_5", "vol_ratio_60",
        "gold_garch_vol", "silver_garch_vol", "garch_vol_ratio",
    ]
    print("Volatility features added:", new_cols)
    print("Total columns:", len(df.columns), "| Final rows:", len(df))

    df.to_csv(FEATURE_PATH)
    print(f"Saved features to {FEATURE_PATH}")


if __name__ == "__main__":
    build_volatility_features()
