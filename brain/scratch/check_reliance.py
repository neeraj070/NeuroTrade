import pandas as pd

meta = pd.read_csv("data/training/test/metadata_test.csv")
proc = pd.read_csv("data/processed/RELIANCE_processed.csv")
print("Proc last 5 dates:\n", proc[["Date", "Open", "Close", "Target_Return_Next", "Target_Date_Next"]].tail(5))
print("\nMeta last 5 rows for RELIANCE:\n", meta[meta["Symbol"] == "RELIANCE"].tail(5))

