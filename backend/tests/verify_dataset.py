"""
Dataset Verification Script
===========================
Academic Project: Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading

Phase 2: Comprehensive Dataset Integrity Verification
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "training", "train")
DATA_VAL_DIR = os.path.join(PROJECT_ROOT, "data", "training", "validation")
DATA_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "training", "test")


def verify_phase2_dataset():
    print("=" * 70)
    print("PHASE 2: RIGOROUS DATASET AUDIT & INTEGRITY VERIFICATION")
    print("=" * 70)

    splits = {
        "train": (DATA_TRAIN_DIR, "X_train.pt", "y_train.pt", "metadata_train.csv"),
        "val": (DATA_VAL_DIR, "X_val.pt", "y_val.pt", "metadata_val.csv"),
        "test": (DATA_TEST_DIR, "X_test.pt", "y_test.pt", "metadata_test.csv"),
    }

    loaded_data = {}
    target_date_sets = {}

    for name, (dir_path, x_file, y_file, meta_file) in splits.items():
        x_path = os.path.join(dir_path, x_file)
        y_path = os.path.join(dir_path, y_file)
        m_path = os.path.join(dir_path, meta_file)

        assert os.path.exists(x_path), f"Missing {x_path}"
        assert os.path.exists(y_path), f"Missing {y_path}"
        assert os.path.exists(m_path), f"Missing {m_path}"

        X = torch.load(x_path, weights_only=True)
        y = torch.load(y_path, weights_only=True)
        meta_df = pd.read_csv(m_path)

        # 1. Verify shapes and dimensions
        assert X.ndim == 3, f"Expected 3D tensor for X, got {X.ndim}D"
        assert X.shape[1] == 30, f"Expected sequence length 30, got {X.shape[1]}"
        assert X.shape[2] == 14, f"Expected 14 features, got {X.shape[2]}"
        assert y.ndim == 2, f"Expected 2D tensor for y, got {y.ndim}D"
        assert y.shape[1] == 1, f"Expected target dimension 1, got {y.shape[1]}"
        assert X.shape[0] == y.shape[0] == len(meta_df), (
            f"Sample count mismatch in {name}: X={X.shape[0]}, y={y.shape[0]}, meta={len(meta_df)}"
        )

        # 2. Verify Data Types
        assert X.dtype == torch.float32, f"Expected float32 for X, got {X.dtype}"
        assert y.dtype == torch.float32, f"Expected float32 for y, got {y.dtype}"

        # 3. Check NaNs and Infs
        nan_x = torch.isnan(X).sum().item()
        inf_x = torch.isinf(X).sum().item()
        nan_y = torch.isnan(y).sum().item()
        inf_y = torch.isinf(y).sum().item()
        assert nan_x == 0 and inf_x == 0, f"{name} X contains NaNs ({nan_x}) or Infs ({inf_x})"
        assert nan_y == 0 and inf_y == 0, f"{name} y contains NaNs ({nan_y}) or Infs ({inf_y})"

        # 4. Check targets match metadata
        meta_targets = torch.tensor(meta_df["Target_Return"].values, dtype=torch.float32).unsqueeze(1)
        assert torch.allclose(y, meta_targets, atol=1e-5), f"Targets in {name} do not match metadata!"

        # 5. Check metadata columns and stocks
        required_cols = {"Sample_Index", "Symbol", "Window_Start_Date", "Window_End_Date", "Target_Date", "Target_Return"}
        assert required_cols.issubset(set(meta_df.columns)), f"Missing metadata columns in {name}"
        unique_stocks = sorted(meta_df["Symbol"].unique())
        expected_stocks = ["HDFCBANK", "INFY", "ITC", "RELIANCE", "TCS"]
        assert unique_stocks == expected_stocks, f"Stock mismatch in {name}: {unique_stocks}"

        target_date_sets[name] = set(meta_df["Target_Date"].unique())
        loaded_data[name] = {
            "N": X.shape[0],
            "L": X.shape[1],
            "D": X.shape[2],
            "min_target_date": meta_df["Target_Date"].min(),
            "max_target_date": meta_df["Target_Date"].max(),
            "stocks_breakdown": meta_df["Symbol"].value_counts().to_dict(),
        }

        print(f"[{name.upper()}] Verified {X.shape[0]} windows | Shape: {tuple(X.shape)} | Targets: {tuple(y.shape)}")
        print(f"  Target Dates: {loaded_data[name]['min_target_date']} -> {loaded_data[name]['max_target_date']}")
        print(f"  Distribution per stock: {loaded_data[name]['stocks_breakdown']}")

    # 6. Check Chronological Separation & Zero Target Date Overlap
    tr_val_overlap = target_date_sets["train"].intersection(target_date_sets["val"])
    val_test_overlap = target_date_sets["val"].intersection(target_date_sets["test"])
    tr_test_overlap = target_date_sets["train"].intersection(target_date_sets["test"])

    assert len(tr_val_overlap) == 0, f"Leakage! Train & Val overlap on {tr_val_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage! Val & Test overlap on {val_test_overlap}"
    assert len(tr_test_overlap) == 0, f"Leakage! Train & Test overlap on {tr_test_overlap}"

    assert loaded_data["train"]["max_target_date"] < loaded_data["val"]["min_target_date"], "Train/Val date ordering violated!"
    assert loaded_data["val"]["max_target_date"] < loaded_data["test"]["min_target_date"], "Val/Test date ordering violated!"

    print("-" * 70)
    print("ALL PHASE 2 DATASET VERIFICATIONS PASSED:")
    print("  - 14 input features and 30 time steps strictly confirmed.")
    print("  - Forward 1-day return targets perfectly verified against metadata.")
    print("  - Zero NaNs, zero Infs, strictly float32 across all tensors.")
    print("  - Complete absence of target-date overlap (100% disjoint).")
    print("  - Monotonic chronological sequence: Train < Val < Test.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_phase2_dataset()

