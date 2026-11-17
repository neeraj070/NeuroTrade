import os
import json
import torch
import pandas as pd
import numpy as np

audit_results = {}

# 1. Datasets
raw_files = [f"data/raw/{s}.csv" for s in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]]
audit_results["raw_files_exist"] = all(os.path.exists(f) for f in raw_files)

proc_files = [f"data/processed/{s}_processed.csv" for s in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]]
audit_results["proc_files_exist"] = all(os.path.exists(f) for f in proc_files)

train_tensors = ["X_train.pt", "y_train.pt", "metadata_train.csv"]
val_tensors = ["X_val.pt", "y_val.pt", "metadata_val.csv"]
test_tensors = ["X_test.pt", "y_test.pt", "metadata_test.csv"]

audit_results["train_tensors_exist"] = all(os.path.exists(f"data/training/train/{f}") for f in train_tensors)
audit_results["val_tensors_exist"] = all(os.path.exists(f"data/training/validation/{f}") for f in val_tensors)
audit_results["test_tensors_exist"] = all(os.path.exists(f"data/training/test/{f}") for f in test_tensors)

# Check shapes
X_tr = torch.load("data/training/train/X_train.pt", weights_only=True)
y_tr = torch.load("data/training/train/y_train.pt", weights_only=True)
X_va = torch.load("data/training/validation/X_val.pt", weights_only=True)
y_va = torch.load("data/training/validation/y_val.pt", weights_only=True)
X_te = torch.load("data/training/test/X_test.pt", weights_only=True)
y_te = torch.load("data/training/test/y_test.pt", weights_only=True)

audit_results["train_shape"] = list(X_tr.shape) == [4000, 30, 14]
audit_results["val_shape"] = list(X_va.shape) == [885, 30, 14]
audit_results["test_shape"] = list(X_te.shape) == [895, 30, 14]

# 2. Checkpoints
ckpts = ["improved_rnn_encoder.pt", "best_rnn_encoder.pt", "bce_directional_encoder.pt", "focal_directional_encoder.pt"]
audit_results["ckpts_exist"] = {c: os.path.exists(f"backend/checkpoints/{c}") for c in ckpts}

# 3. Embeddings
embs = [
    "train_embeddings_improved.pt", "val_embeddings_improved.pt", "test_embeddings_improved.pt",
    "train_embeddings_bce.pt", "val_embeddings_bce.pt", "test_embeddings_bce.pt",
    "train_embeddings_focal.pt", "val_embeddings_focal.pt", "test_embeddings_focal.pt"
]
audit_results["embeddings_exist"] = {e: os.path.exists(f"backend/embeddings/{e}") for e in embs}

# 4. Reports
reports = [
    "reports/data_preparation/data_quality_report.md",
    "reports/data_preparation/feature_documentation.md",
    "reports/data_preparation/final_data_preparation_report.md",
    "reports/model_training/corrected_evaluation_report.md",
    "reports/model_training/lstm_improvement_report.md",
    "reports/model_training/final_test_evaluation.json",
    "reports/model_training/embedding_analysis/embedding_quality_report.md",
    "reports/model_training/final_lstm_improvement_audit.md",
    "reports/model_training/directional_classification_report.md",
    "reports/reinforcement_learning/trading_environment_report.md"
]
audit_results["reports_exist"] = {r: os.path.exists(r) for r in reports}

# 5. Notebook execution outputs
notebooks = [
    "data/notebooks/01_data_collection.ipynb",
    "data/notebooks/02_data_cleaning.ipynb",
    "data/notebooks/03_eda.ipynb",
    "data/notebooks/04_feature_engineering.ipynb",
    "data/notebooks/05_preprocessing.ipynb",
    "data/notebooks/06_data_validation.ipynb",
    "backend/notebooks/01_lstm_training.ipynb",
    "backend/notebooks/02_lstm_evaluation.ipynb",
    "backend/notebooks/03_embedding_analysis.ipynb",
    "backend/notebooks/04_rl_development.ipynb",
    "backend/notebooks/05_directional_classification.ipynb"
]
nb_output_status = {}
for nb_path in notebooks:
    if os.path.exists(nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        nb_output_status[nb_path] = {
            "exists": True,
            "total_cells": len(nb["cells"]),
            "code_cells": len(code_cells),
            "executed_cells": len(executed_cells),
            "has_outputs": len(executed_cells) > 0
        }
    else:
        nb_output_status[nb_path] = {"exists": False}

audit_results["notebook_status"] = nb_output_status

with open("brain/scratch/audit_summary.json", "w") as f:
    json.dump(audit_results, f, indent=2)

print("Audit verification completed successfully. Saved to brain/scratch/audit_summary.json")
