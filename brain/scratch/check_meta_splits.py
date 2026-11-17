import pandas as pd

splits = [
    ("train", "data/training/train/metadata_train.csv"),
    ("val", "data/training/validation/metadata_val.csv"),
    ("test", "data/training/test/metadata_test.csv")
]

for name, mp in splits:
    meta = pd.read_csv(mp)
    print(f"\n=== {name.upper()} METADATA ===")
    print(meta.groupby("Symbol", sort=False).size())
    print("Total:", len(meta))
