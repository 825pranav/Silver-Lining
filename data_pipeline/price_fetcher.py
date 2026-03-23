import yfinance as yf
import pandas as pd
import os

def fetch_prices():
    print("🚀 Downloading gold and silver futures data...")
    
    # GC=F is Gold, SI=F is Silver
    # We download both at once to let yfinance align the dates for us
    tickers = ["GC=F", "SI=F"]
    
    # Fetching 2 years of daily data
    data = yf.download(tickers, period="2y", interval="1d")
    
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
        
        print("\n✅ Successfully synced Gold and Silver data!")
        print(f"Total trading days captured: {len(df)}")
        print("\n--- Latest Market Snapshot ---")
        print(df.tail())

        # Ensure directory exists before saving
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_metals_data.csv")
        print("\n💾 Saved to data/raw_metals_data.csv")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()