import pandas as pd
import numpy as np

meta_val = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test = pd.read_csv("data/training/test/metadata_test.csv")

symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in symbols}

for split_name, meta in [("Validation", meta_val), ("Test", meta_test)]:
    print(f"\n--- {split_name} Split ({meta['Target_Date'].min()} to {meta['Target_Date'].max()}) ---")
    bnh_returns = []
    for sym in symbols:
        sub_meta = meta[meta["Symbol"] == sym].copy().reset_index(drop=True)
        # Dates
        d0 = sub_meta["Target_Date"].iloc[0]
        d_end = sub_meta["Target_Date"].iloc[-1]
        
        df_sym = raw_prices[sym]
        row0 = df_sym[df_sym["Date"] == d0]
        row_end = df_sym[df_sym["Date"] == d_end]
        
        day0_open = row0["Open"].values[0]
        dayend_close = row_end["Close"].values[0]
        
        shares = (100000.0 * (1 - 0.001)) / day0_open
        final_val = shares * dayend_close
        bnh_ret = (final_val - 100000.0) / 100000.0 * 100.0
        bnh_returns.append(bnh_ret)
        print(f"  {sym:10s} ({d0} to {d_end}) Buy & Hold Return: {bnh_ret:+6.2f}%")
        
    print(f"  --> Average Buy & Hold Return: {np.mean(bnh_returns):+6.2f}%")
