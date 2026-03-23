import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import os
from datetime import datetime, timedelta


def fetch_macro():
    print("🚀 Downloading macro data (DXY, treasury yields, commodity index)...")

    end = datetime.today()
    start = end - timedelta(days=2 * 365)

    # --- yfinance: USD index (DXY) and broad commodity index ---
    # DX-Y.NYB = ICE US Dollar Index (DXY)
    # ^BCOM    = Bloomberg Commodity Index
    yf_tickers = ["DX-Y.NYB", "^BCOM"]
    yf_raw = yf.download(yf_tickers, start=start, end=end, interval="1d")

    if "Close" in yf_raw.columns:
        yf_df = yf_raw["Close"].copy()
    else:
        yf_df = yf_raw.xs("Close", axis=1, level=0)

    yf_df = yf_df.rename(columns={"DX-Y.NYB": "dxy", "^BCOM": "commodity_index"})
    yf_df.index.name = "Date"

    # --- FRED: 10-year and 2-year constant-maturity treasury yields ---
    # DGS10 = 10-Year Treasury Constant Maturity Rate (%)
    # DGS2  = 2-Year Treasury Constant Maturity Rate (%)
    print("📡 Fetching treasury yield data from FRED...")
    treasury_10y = web.DataReader("DGS10", "fred", start, end)
    treasury_2y = web.DataReader("DGS2", "fred", start, end)

    treasury_10y.columns = ["yield_10y"]
    treasury_2y.columns = ["yield_2y"]

    fred_df = treasury_10y.join(treasury_2y, how="outer")
    fred_df.index.name = "Date"
    # FRED publishes on business days; forward-fill up to 3 days to align with
    # trading-day gaps (weekends/holidays where FRED has NaN but markets are open)
    fred_df = fred_df.ffill(limit=3)

    # --- Merge yfinance and FRED on shared trading days ---
    df = yf_df.join(fred_df, how="inner")

    # Drop any remaining rows where any column is missing
    df = df.dropna()

    # --- Derived feature: yield curve slope (10Y - 2Y) ---
    df["yield_curve_slope"] = df["yield_10y"] - df["yield_2y"]

    # Reorder columns for clarity
    df = df[["dxy", "yield_10y", "yield_2y", "yield_curve_slope", "commodity_index"]]

    return df


def main():
    try:
        df = fetch_macro()

        print("\n✅ Successfully synced macro data!")
        print(f"Total trading days captured: {len(df)}")
        print("\n--- Latest Macro Snapshot ---")
        print(df.tail())

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/macro_data.csv")
        print("\n💾 Saved to data/macro_data.csv")

    except Exception as e:
        print(f"❌ Error occurred: {e}")


if __name__ == "__main__":
    main()
