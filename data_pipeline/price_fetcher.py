import yfinance as yf
import pandas as pd
import os
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import console as _console  # noqa: F401,E402  UTF-8 safe stdout

def fetch_prices():
    print("[run] Downloading gold and silver futures data...")
    
    # GC=F is Gold, SI=F is Silver
    # We download both at once to let yfinance align the dates for us
    tickers = ["GC=F", "SI=F"]
    
    # Full available history. Two years covers a single market regime - the
    # 2024-25 metals rally - and a model fitted there learns "be long metals".
    # The full series reaches back to 2000 and includes the 2013 gold crash,
    # the 2015 decline and the 2021-22 flat stretch, which is what makes an
    # out-of-sample test mean anything.
    period = os.getenv("SL_HISTORY_PERIOD", "max")
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True)
    
    # In recent yfinance versions, this returns a MultiIndex. 
    # We just want the 'Close' prices.
    if 'Close' in data.columns:
        df = data['Close'].copy()
    else:
        # Fallback for different yf versions
        df = data.xs('Close', axis=1, level=0)

    # Clean up column names
    df = df.rename(columns={"GC=F": "gold_close", "SI=F": "silver_close"})
    
    # Drop any days where one market was closed but the other was open
    df = df.dropna()
    
    # Calculate our target ratio
    df['gsr'] = df['gold_close'] / df['silver_close']
    
    return df

def main():
    try:
        df = fetch_prices()
        
        print("\n[ok] Successfully synced Gold and Silver data!")
        print(f"Total trading days captured: {len(df)}")
        print("\n--- Latest Market Snapshot ---")
        print(df.tail())

        # Ensure directory exists before saving
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_metals_data.csv")
        print("\n[saved] Saved to data/raw_metals_data.csv")

    except Exception as e:
        print(f"[error] Error occurred: {e}")

if __name__ == "__main__":
    main()