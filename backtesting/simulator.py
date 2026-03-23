# backtesting/simulator.py

import pandas as pd
import numpy as np

FEATURE_PATH = "../data/features.csv"


def run_backtest():

    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    capital = 1.0
    capital_curve = []

    position = None  # "gold" or "silver"

    for i in range(1, len(df)):

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        signal = None

        # ---------- MEAN REVERSION SIGNAL ----------
        if row["gsr_zscore_30"] > 1 and abs(row["gsr_slope"]) < 0.5:
            signal = "silver"   # ratio high -> silver undervalued
        elif row["gsr_zscore_30"] < -1 and abs(row["gsr_slope"]) < 0.5:
            signal = "gold"

        # ---------- TREND FOLLOWING FALLBACK ----------
        if signal is None:
            if row["gsr_slope"] > 0:
                signal = "gold"
            else:
                signal = "silver"

        # ---------- DAILY RETURN ----------
        if position == "gold":
            ret = row["gold_close"] / prev["gold_close"] - 1
        elif position == "silver":
            ret = row["silver_close"] / prev["silver_close"] - 1
        else:
            ret = 0

        capital *= (1 + ret)
        capital_curve.append(capital)

        position = signal

    df = df.iloc[1:]
    df["capital"] = capital_curve

    total_return = (capital - 1) * 100

    print("Final Capital:", round(capital, 3))
    print("Total Return %:", round(total_return, 2))

    df.to_csv("../data/backtest_results.csv")
    print("Saved results to data/backtest_results.csv")


if __name__ == "__main__":
    run_backtest()

    