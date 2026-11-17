import pandas as pd

for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
    raw = pd.read_csv(f"data/raw/{sym}.csv")
    print(f"{sym}: raw dates {raw['Date'].min()} to {raw['Date'].max()} (len={len(raw)})")
    
for split, mp in [("train", "data/training/train/metadata_train.csv"),
                  ("val", "data/training/validation/metadata_val.csv"),
                  ("test", "data/training/test/metadata_test.csv")]:
    meta = pd.read_csv(mp)
    for sym in meta["Symbol"].unique():
        sub = meta[meta["Symbol"] == sym]
        raw = pd.read_csv(f"data/raw/{sym}.csv")
        raw_dates = set(raw["Date"])
        missing_end = set(sub["Window_End_Date"]) - raw_dates
        missing_tgt = set(sub["Target_Date"]) - raw_dates
        assert len(missing_end) == 0, f"Missing Window_End_Date in {split} {sym}: {missing_end}"
        assert len(missing_tgt) == 0, f"Missing Target_Date in {split} {sym}: {missing_tgt}"
print("All Window_End_Date and Target_Date exist in raw CSVs for all symbols and splits [VERIFIED!]")

