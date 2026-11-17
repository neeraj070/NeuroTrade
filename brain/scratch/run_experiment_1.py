import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.spatial.distance import pdist

# Seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load data
X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
y_train_raw = torch.load("data/training/train/y_train.pt", weights_only=True).squeeze()
X_val = torch.load("data/training/validation/X_val.pt", weights_only=True)
y_val_raw = torch.load("data/training/validation/y_val.pt", weights_only=True).squeeze()
X_test = torch.load("data/training/test/X_test.pt", weights_only=True)
y_test_raw = torch.load("data/training/test/y_test.pt", weights_only=True).squeeze()

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test = pd.read_csv("data/training/test/metadata_test.csv")

# 2. Binary target construction: UP = return > 0 (1.0), DOWN = return <= 0 (0.0)
y_train_bin = (y_train_raw > 0).float()
y_val_bin = (y_val_raw > 0).float()
y_test_bin = (y_test_raw > 0).float()

print("Class distribution:")
for name, y in [("Train", y_train_bin), ("Val", y_val_bin), ("Test", y_test_bin)]:
    n_up = (y == 1.0).sum().item()
    n_down = (y == 0.0).sum().item()
    n_tot = len(y)
    print(f"  {name:<6}: Total={n_tot}, UP={n_up} ({n_up/n_tot*100:.2f}%), DOWN={n_down} ({n_down/n_tot*100:.2f}%)")

# Majority direction baseline from Train distribution:
# Train majority is UP (51.00%). Predicting UP for all test gives 44.13%.
maj_train_class = 1.0 if (y_train_bin == 1.0).sum() >= (y_train_bin == 0.0).sum() else 0.0
maj_test_acc = (y_test_bin == maj_train_class).float().mean().item()
print(f"Majority-direction baseline on test: {maj_test_acc*100:.2f}% (predicted class: {maj_train_class})")

# 3. Model Architecture
class MarketStateLSTMClassifier(nn.Module):
    def __init__(self, input_dim=14, sequence_length=30, hidden_dim=64, num_layers=2,
                 embedding_dim=32, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, 1)
        )

    def encode(self, x):
        out, _ = self.lstm(x)
        return self.embedding_layer(out[:, -1, :])

    def forward(self, x, return_embedding=False):
        emb = self.encode(x)
        logits = self.classifier(emb).squeeze(-1)
        return (logits, emb) if return_embedding else logits

# 4. Focal Loss implementation
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        else:
            alpha_t = 1.0
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# Training helper
def train_model(criterion, model_name="model", epochs=60, lr=0.001, weight_decay=1e-4, patience=15, seed=42):
    set_seed(seed)
    model = MarketStateLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    train_dataset = TensorDataset(X_train, y_train_bin)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": [], "val_auc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * len(by)
        train_loss = train_loss_total / len(y_train_bin)
        
        # Validation
        model.eval()
        with torch.no_grad():
            vx, vy = X_val.to(device), y_val_bin.to(device)
            val_logits = model(vx)
            val_loss = criterion(val_logits, vy).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            vy_np = y_val_bin.numpy()
            val_preds = (val_probs >= 0.5).astype(float)
            val_acc = accuracy_score(vy_np, val_preds)
            val_bacc = balanced_accuracy_score(vy_np, val_preds)
            val_auc = roc_auc_score(vy_np, val_probs)
            
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bacc"].append(val_bacc)
        history["val_auc"].append(val_auc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # Early stop
                break
                
    print(f"[{model_name}] Finished at epoch {epoch}. Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}, Val Acc: {history['val_acc'][best_epoch-1]*100:.2f}%, Val BAcc: {history['val_bacc'][best_epoch-1]*100:.2f}%, Val AUC: {history['val_auc'][best_epoch-1]:.4f}")
    
    # Reload best state
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_loss

print("\n--- Training Model A: BCE With Logits ---")
bce_criterion = nn.BCEWithLogitsLoss()
model_bce, hist_bce, best_ep_bce, best_loss_bce = train_model(bce_criterion, model_name="BCE", epochs=60)

print("\n--- Tuning Model B: Focal Loss (Validation Grid Search) ---")
focal_results = []
gamma_candidates = [0.5, 1.0, 2.0, 3.0]
alpha_candidates = [0.25, 0.5, 0.75]

best_focal_cfg = None
best_focal_val_bacc = -1.0

for g in gamma_candidates:
    for a in alpha_candidates:
        fl_crit = BinaryFocalLoss(alpha=a, gamma=g)
        m, h, bep, bl = train_model(fl_crit, model_name=f"Focal(g={g},a={a})", epochs=50, patience=12)
        val_acc = h["val_acc"][bep-1]
        val_bacc = h["val_bacc"][bep-1]
        val_auc = h["val_auc"][bep-1]
        res_entry = {
            "gamma": g, "alpha": a, "best_epoch": bep, "val_loss": bl,
            "val_acc": val_acc, "val_bacc": val_bacc, "val_auc": val_auc
        }
        focal_results.append(res_entry)
        if val_bacc > best_focal_val_bacc:
            best_focal_val_bacc = val_bacc
            best_focal_cfg = (g, a)

print("\nFocal Loss Grid Search Results (Sorted by Val Balanced Acc):")
focal_df = pd.DataFrame(focal_results).sort_values("val_bacc", ascending=False)
print(focal_df.to_string(index=False))
print(f"\nBest Focal Configuration selected: gamma={best_focal_cfg[0]}, alpha={best_focal_cfg[1]}")

# Train the selected Focal Loss model with full patience
best_fl_crit = BinaryFocalLoss(alpha=best_focal_cfg[1], gamma=best_focal_cfg[0])
model_focal, hist_focal, best_ep_focal, best_loss_focal = train_model(
    best_fl_crit, model_name=f"Focal_Best(g={best_focal_cfg[0]},a={best_focal_cfg[1]})", epochs=60, patience=15
)

# Threshold Tuning on Validation Set ONLY
def find_best_threshold(model, X_v, y_v, metric="bacc"):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_v.to(device))).cpu().numpy()
    y_v_np = y_v.numpy()
    
    thresholds = np.linspace(0.35, 0.65, 31)
    best_th = 0.5
    best_score = -1.0
    for th in thresholds:
        preds = (probs >= th).astype(float)
        score = balanced_accuracy_score(y_v_np, preds) if metric == "bacc" else accuracy_score(y_v_np, preds)
        if score > best_score:
            best_score = score
            best_th = th
    return float(best_th), float(best_score)

th_bce, val_bacc_bce = find_best_threshold(model_bce, X_val, y_val_bin, "bacc")
th_focal, val_bacc_focal = find_best_threshold(model_focal, X_val, y_val_bin, "bacc")
print(f"\nTuned Thresholds on Val: BCE={th_bce:.3f} (Val BAcc: {val_bacc_bce*100:.2f}%), Focal={th_focal:.3f} (Val BAcc: {val_bacc_focal*100:.2f}%)")

# Evaluation on Test Data
def evaluate_model_test(model, X_t, y_t_bin, y_t_raw, meta_t, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(X_t.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_t_np = y_t_bin.numpy()
    preds = (probs >= threshold).astype(float)
    
    acc = accuracy_score(y_t_np, preds)
    bacc = balanced_accuracy_score(y_t_np, preds)
    prec = precision_score(y_t_np, preds, zero_division=0)
    rec = recall_score(y_t_np, preds, zero_division=0)
    f1 = f1_score(y_t_np, preds, zero_division=0)
    cm = confusion_matrix(y_t_np, preds)
    roc_auc = roc_auc_score(y_t_np, probs)
    pr_auc = average_precision_score(y_t_np, probs)
    
    # Per-stock accuracy
    per_stock = {}
    for sym in meta_t["Symbol"].unique():
        idx = (meta_t["Symbol"] == sym).values
        sub_preds = preds[idx]
        sub_true = y_t_np[idx]
        per_stock[sym] = {
            "Accuracy": float(accuracy_score(sub_true, sub_preds)),
            "Balanced_Acc": float(balanced_accuracy_score(sub_true, sub_preds)),
            "Count": int(len(sub_true))
        }
        
    return {
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bacc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Confusion_Matrix": cm.tolist(),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Threshold": threshold,
        "Per_Stock": per_stock,
        "Predictions": preds.tolist(),
        "Probabilities": probs.tolist()
    }

eval_bce_default = evaluate_model_test(model_bce, X_test, y_test_bin, y_test_raw, meta_test, threshold=0.5)
eval_bce_tuned = evaluate_model_test(model_bce, X_test, y_test_bin, y_test_raw, meta_test, threshold=th_bce)

eval_focal_default = evaluate_model_test(model_focal, X_test, y_test_bin, y_test_raw, meta_test, threshold=0.5)
eval_focal_tuned = evaluate_model_test(model_focal, X_test, y_test_bin, y_test_raw, meta_test, threshold=th_focal)

print("\n================== TEST SET EVALUATION ==================")
print(f"BCE (th=0.50): Acc={eval_bce_default['Accuracy']*100:.2f}%, BAcc={eval_bce_default['Balanced_Accuracy']*100:.2f}%, F1={eval_bce_default['F1']:.4f}, AUC={eval_bce_default['ROC_AUC']:.4f}")
print(f"BCE (th={th_bce:.2f}): Acc={eval_bce_tuned['Accuracy']*100:.2f}%, BAcc={eval_bce_tuned['Balanced_Accuracy']*100:.2f}%, F1={eval_bce_tuned['F1']:.4f}, AUC={eval_bce_tuned['ROC_AUC']:.4f}")
print(f"Focal (th=0.50): Acc={eval_focal_default['Accuracy']*100:.2f}%, BAcc={eval_focal_default['Balanced_Accuracy']*100:.2f}%, F1={eval_focal_default['F1']:.4f}, AUC={eval_focal_default['ROC_AUC']:.4f}")
print(f"Focal (th={th_focal:.2f}): Acc={eval_focal_tuned['Accuracy']*100:.2f}%, BAcc={eval_focal_tuned['Balanced_Accuracy']*100:.2f}%, F1={eval_focal_tuned['F1']:.4f}, AUC={eval_focal_tuned['ROC_AUC']:.4f}")

# Save detailed experiment 1 results
with open("brain/scratch/exp1_results.json", "w") as f:
    json.dump({
        "bce_default": eval_bce_default,
        "bce_tuned": eval_bce_tuned,
        "focal_default": eval_focal_default,
        "focal_tuned": eval_focal_tuned,
        "focal_grid": focal_results,
        "best_focal_cfg": best_focal_cfg,
        "th_bce": th_bce,
        "th_focal": th_focal
    }, f, indent=2)

print("\nSaved intermediate results to brain/scratch/exp1_results.json")

