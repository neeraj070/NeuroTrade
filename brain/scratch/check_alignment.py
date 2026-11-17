import pandas as pd
import numpy as np

for sym in ["RELIANCE", "HDFCBANK"]:
    meta = pd.read_csv("data/training/train/metadata_train.csv")
    sub_meta = meta[meta["Symbol"] == sym].head(5)
    proc = pd.read_csv(f"data/processed/{sym}_processed.csv")
    print(f"\n=== {sym} Alignment Check ===")
    print("Metadata head:")
    print(sub_meta[["Symbol", "Window_Start_Date", "Window_End_Date", "Target_Date", "Target_Return"]])
    
    first_end = sub_meta.iloc[0]["Window_End_Date"]
    first_target = sub_meta.iloc[0]["Target_Date"]
    
    end_row = proc[proc["Date"] == first_end]
    tgt_row = proc[proc["Date"] == first_target]
    
    print(f"End row ({first_end}): Close={end_row['Close'].values[0]}")
    print(f"Target row ({first_target}): Open={tgt_row['Open'].values[0]}, Close={tgt_row['Close'].values[0]}")
    calc_ret = (tgt_row['Close'].values[0] - end_row['Close'].values[0]) / end_row['Close'].values[0]
    print(f"Calculated return: {calc_ret:.6f}, Meta Target_Return: {sub_meta.iloc[0]['Target_Return']:.6f}")

