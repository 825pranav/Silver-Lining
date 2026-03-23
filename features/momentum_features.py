# features/momentum_features.py
#
# Reads features.csv (output of gsr_features.py), adds momentum features,
# and writes back to features.csv.
#
# Run order: price_fetcher.py → gsr_features.py → momentum_features.py

import pandas as pd
import numpy as np

FEATURE_PATH = "../data/features.csv"


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def build_momentum_features():

    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # ---- EMA Slopes ----
    # Short EMA above long EMA = uptrend (positive slope)
    df["gold_ema_10"] = compute_ema(df["gold_close"], 10)
    df["gold_ema_30"] = compute_ema(df["gold_close"], 30)
    df["gold_ema_slope"] = df["gold_ema_10"] - df["gold_ema_30"]

    df["silver_ema_10"] = compute_ema(df["silver_close"], 10)
    df["silver_ema_30"] = compute_ema(df["silver_close"], 30)
    df["silver_ema_slope"] = df["silver_ema_10"] - df["silver_ema_30"]

    df["gsr_ema_10"] = compute_ema(df["gsr"], 10)
    df["gsr_ema_30"] = compute_ema(df["gsr"], 30)
    df["gsr_ema_slope"] = df["gsr_ema_10"] - df["gsr_ema_30"]

    # ---- MACD (12 / 26 / 9) ----
    gold_ema_12 = compute_ema(df["gold_close"], 12)
    gold_ema_26 = compute_ema(df["gold_close"], 26)
    df["gold_macd"] = gold_ema_12 - gold_ema_26
    df["gold_macd_signal"] = compute_ema(df["gold_macd"], 9)
    df["gold_macd_hist"] = df["gold_macd"] - df["gold_macd_signal"]

    silver_ema_12 = compute_ema(df["silver_close"], 12)
    silver_ema_26 = compute_ema(df["silver_close"], 26)
    df["silver_macd"] = silver_ema_12 - silver_ema_26
    df["silver_macd_signal"] = compute_ema(df["silver_macd"], 9)
    df["silver_macd_hist"] = df["silver_macd"] - df["silver_macd_signal"]

    # ---- Breakout Strength ----
    # Position within 20-day range: 0.0 = at recent low, 1.0 = at recent high
    window = 20
    gold_high = df["gold_close"].rolling(window).max()
    gold_low = df["gold_close"].rolling(window).min()
    df["gold_breakout"] = (df["gold_close"] - gold_low) / (gold_high - gold_low)

    silver_high = df["silver_close"].rolling(window).max()
    silver_low = df["silver_close"].rolling(window).min()
    df["silver_breakout"] = (df["silver_close"] - silver_low) / (silver_high - silver_low)

    # ---- Relative Strength (Gold vs Silver) ----
    # Positive = gold outperforming silver over that window
    # Reuses gold_ret_5/20 and silver_ret_5/20 already in features.csv
    df["rel_strength_5"] = df["gold_ret_5"] - df["silver_ret_5"]
    df["rel_strength_20"] = df["gold_ret_20"] - df["silver_ret_20"]

    df.dropna(inplace=True)

    new_cols = [
        "gold_ema_10", "gold_ema_30", "gold_ema_slope",
        "silver_ema_10", "silver_ema_30", "silver_ema_slope",
        "gsr_ema_10", "gsr_ema_30", "gsr_ema_slope",
        "gold_macd", "gold_macd_signal", "gold_macd_hist",
        "silver_macd", "silver_macd_signal", "silver_macd_hist",
        "gold_breakout", "silver_breakout",
        "rel_strength_5", "rel_strength_20",
    ]
    print("Momentum features added:", new_cols)
    print("Total columns:", len(df.columns), "| Final rows:", len(df))

    df.to_csv(FEATURE_PATH)
    print(f"Saved features to {FEATURE_PATH}")


if __name__ == "__main__":
    build_momentum_features()
