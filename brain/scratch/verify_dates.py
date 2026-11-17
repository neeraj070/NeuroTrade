import pandas as pd

for split, mp in [("train", "data/training/train/metadata_train.csv"),
                  ("val", "data/training/validation/metadata_val.csv"),
                  ("test", "data/training/test/metadata_test.csv")]:
    meta = pd.read_csv(mp)
    for sym in meta["Symbol"].unique():
        sub = meta[meta["Symbol"] == sym]
        proc = pd.read_csv(f"data/processed/{sym}_processed.csv")
        proc_dates = set(proc["Date"])
        missing_end = set(sub["Window_End_Date"]) - proc_dates
        missing_tgt = set(sub["Target_Date"]) - proc_dates
        assert len(missing_end) == 0, f"Missing Window_End_Date in {split} {sym}: {missing_end}"
        assert len(missing_tgt) == 0, f"Missing Target_Date in {split} {sym}: {missing_tgt}"
print("All Window_End_Date and Target_Date exist in processed CSVs [VERIFIED!]")

