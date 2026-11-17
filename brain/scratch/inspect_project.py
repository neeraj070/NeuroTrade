import os
import json
import torch
import pandas as pd
import numpy as np

print("=== CHECKPOINT AUDIT ===")
ckpt_path = "backend/checkpoints/improved_rnn_encoder.pt"
ckpt = torch.load(ckpt_path, weights_only=True)
print("Config:", ckpt.get("config"))
print("Epoch:", ckpt.get("epoch"))
print("Best val loss:", ckpt.get("best_val_loss"))
print("Experiment:", ckpt.get("experiment_name"))

print("\n=== EMBEDDINGS AUDIT ===")
for split in ["train", "val", "test"]:
    p = f"backend/embeddings/{split}_embeddings_improved.pt"
    t = torch.load(p, weights_only=True)
    print(f"{split} embeddings shape: {t.shape}, dtype: {t.dtype}")

print("\n=== DATA TRAINING TENSORS & METADATA ===")
splits = [
    ("train", "data/training/train/X_train.pt", "data/training/train/y_train.pt", "data/training/train/metadata_train.csv"),
    ("val", "data/training/validation/X_val.pt", "data/training/validation/y_val.pt", "data/training/validation/metadata_val.csv"),
    ("test", "data/training/test/X_test.pt", "data/training/test/y_test.pt", "data/training/test/metadata_test.csv")
]

for name, xp, yp, mp in splits:
    x = torch.load(xp, weights_only=True)
    y = torch.load(yp, weights_only=True)
    meta = pd.read_csv(mp)
    print(f"{name}: X={x.shape}, y={y.shape}, meta={meta.shape}")
    print(f"  Meta columns: {list(meta.columns)}")
    print(f"  Unique symbols: {meta['Symbol'].unique()}")
    print(f"  Date range: {meta['Target_Date'].min()} to {meta['Target_Date'].max()}")
    print(f"  Head(2):\n{meta.head(2)}")

print("\n=== SCALER METADATA ===")
if os.path.exists("data/training/scalers/scaler_metadata.json"):
    with open("data/training/scalers/scaler_metadata.json") as f:
        sm = json.load(f)
    print("Features in scaler:", sm.get("features"))
    print("Feature count:", len(sm.get("features", [])))

print("\n=== PROCESSED DATA SAMPLE ===")
for sym in ["HDFCBANK", "INFY"]:
    df = pd.read_csv(f"data/processed/{sym}_processed.csv")
    print(f"{sym} columns: {list(df.columns)}")
    print(f"{sym} shape: {df.shape}, date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"{sym} head(2):\n{df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head(2)}")
