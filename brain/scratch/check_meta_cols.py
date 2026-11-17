import pandas as pd

meta = pd.read_csv("data/training/train/metadata_train.csv")
print("Meta cols:", meta.columns.tolist())
print(meta.head(3))

