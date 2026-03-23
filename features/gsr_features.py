# features/gsr_features.py

import pandas as pd
import numpy as np

RAW_DATA_PATH = "../data/raw_metals_data.csv"
FEATURE_PATH = "../data/features.csv"


def compute_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def build_features():

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # ---- Core GSR Features ----
    df["gsr"] = df["gold_close"] / df["silver_close"]

    df["gsr_zscore_30"] = compute_zscore(df["gsr"], 30)
    df["gsr_zscore_90"] = compute_zscore(df["gsr"], 90)

    df["gsr_slope"] = df["gsr"].diff(5)

    # ---- Momentum Features ----
    df["gold_ret_5"] = df["gold_close"].pct_change(5)
    df["gold_ret_20"] = df["gold_close"].pct_change(20)

    df["silver_ret_5"] = df["silver_close"].pct_change(5)
    df["silver_ret_20"] = df["silver_close"].pct_change(20)

    # ---- Volatility Features ----
    df["gold_vol_20"] = df["gold_close"].pct_change().rolling(20).std()
    df["silver_vol_20"] = df["silver_close"].pct_change().rolling(20).std()

    df["vol_ratio"] = df["gold_vol_20"] / df["silver_vol_20"]

    df.dropna(inplace=True)

    print("Feature set created with columns:")
    print(df.columns.tolist())
    print("Final rows:", len(df))

    df.to_csv(FEATURE_PATH)
    print(f"Saved features to {FEATURE_PATH}")


if __name__ == "__main__":
    build_features()
