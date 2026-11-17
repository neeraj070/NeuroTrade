import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.spatial.distance import pdist

# Ensure directories exist
os.makedirs("backend/checkpoints", exist_ok=True)
os.makedirs("backend/embeddings", exist_ok=True)
os.makedirs("reports/model_training/figures", exist_ok=True)

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

# Load data
X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
y_train_raw = torch.load("data/training/train/y_train.pt", weights_only=True).squeeze()
X_val = torch.load("data/training/validation/X_val.pt", weights_only=True)
y_val_raw = torch.load("data/training/validation/y_val.pt", weights_only=True).squeeze()
X_test = torch.load("data/training/test/X_test.pt", weights_only=True)
y_test_raw = torch.load("data/training/test/y_test.pt", weights_only=True).squeeze()

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test = pd.read_csv("data/training/test/metadata_test.csv")

y_train_bin = (y_train_raw > 0).float()
y_val_bin = (y_val_raw > 0).float()
y_test_bin = (y_test_raw > 0).float()

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

def train_classifier(criterion, name="BCE", epochs=60, lr=0.001, weight_decay=1e-4, patience=15, seed=42):
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
                break
                
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_loss

print("1. Training BCE Classifier...")
crit_bce = nn.BCEWithLogitsLoss()
model_bce, hist_bce, ep_bce, loss_bce = train_classifier(crit_bce, name="BCE", epochs=60, patience=15)

# Save BCE checkpoint
bce_ckpt = {
    "model_state_dict": model_bce.state_dict(),
    "config": {
        "input_dim": 14, "sequence_length": 30, "hidden_dim": 64,
        "num_layers": 2, "embedding_dim": 32, "dropout": 0.2
    },
    "epoch": ep_bce,
    "best_val_loss": loss_bce,
    "loss_type": "BCEWithLogitsLoss",
    "seed": 42
}
torch.save(bce_ckpt, "backend/checkpoints/bce_directional_encoder.pt")
print(f"Saved BCE checkpoint: epoch {ep_bce}, val_loss {loss_bce:.6f}")

print("\n2. Focal Loss tuning & training...")
# Compare symmetric alpha=0.5 with varying gamma, plus other alphas
best_fl_cfg = None
best_fl_val_loss = float("inf")
fl_tuning_records = []

# Systematic sweep over gamma and alpha
for g in [0.5, 1.0, 2.0, 3.0]:
    for a in [0.25, 0.5, 0.75]:
        crit = BinaryFocalLoss(alpha=a, gamma=g)
        m, h, ep, vl = train_classifier(crit, name=f"FL_g{g}_a{a}", epochs=50, patience=12)
        val_acc = h["val_acc"][ep-1]
        val_bacc = h["val_bacc"][ep-1]
        val_auc = h["val_auc"][ep-1]
        rec = {
            "gamma": g, "alpha": a, "best_epoch": ep, "val_loss": vl,
            "val_acc": val_acc, "val_bacc": val_bacc, "val_auc": val_auc
        }
        fl_tuning_records.append(rec)
        # Select best based on val_loss
        if vl < best_fl_val_loss and a == 0.5: # Balanced alpha prevents trivial collapse
            best_fl_val_loss = vl
            best_fl_cfg = (g, a)

print(f"Best balanced Focal Loss config: gamma={best_fl_cfg[0]}, alpha={best_fl_cfg[1]}")
crit_focal = BinaryFocalLoss(alpha=best_fl_cfg[1], gamma=best_fl_cfg[0])
model_focal, hist_focal, ep_focal, loss_focal = train_classifier(crit_focal, name="Focal_Best", epochs=60, patience=15)

focal_ckpt = {
    "model_state_dict": model_focal.state_dict(),
    "config": {
        "input_dim": 14, "sequence_length": 30, "hidden_dim": 64,
        "num_layers": 2, "embedding_dim": 32, "dropout": 0.2
    },
    "epoch": ep_focal,
    "best_val_loss": loss_focal,
    "loss_type": "BinaryFocalLoss",
    "gamma": best_fl_cfg[0],
    "alpha": best_fl_cfg[1],
    "seed": 42
}
torch.save(focal_ckpt, "backend/checkpoints/focal_directional_encoder.pt")
print(f"Saved Focal checkpoint: epoch {ep_focal}, val_loss {loss_focal:.6f}")

# Threshold evaluation on VALIDATION data only
def tune_threshold(model, X_v, y_v):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_v.to(device))).cpu().numpy()
    y_true = y_v.numpy()
    thresholds = np.linspace(0.35, 0.65, 31)
    best_th = 0.5
    best_bacc = -1.0
    for th in thresholds:
        preds = (probs >= th).astype(float)
        bacc = balanced_accuracy_score(y_true, preds)
        if bacc > best_bacc:
            best_bacc = bacc
            best_th = float(th)
    return best_th, best_bacc

th_bce, val_bacc_bce = tune_threshold(model_bce, X_val, y_val_bin)
th_focal, val_bacc_focal = tune_threshold(model_focal, X_val, y_val_bin)
print(f"\nValidation Selected Thresholds: BCE={th_bce:.3f} (BAcc={val_bacc_bce*100:.2f}%), Focal={th_focal:.3f} (BAcc={val_bacc_focal*100:.2f}%)")

# Extract and save embeddings
print("\n3. Extracting and saving embeddings...")
def extract_all_embeddings(model, prefix):
    model.eval()
    with torch.no_grad():
        emb_tr = model.encode(X_train.to(device)).cpu()
        emb_va = model.encode(X_val.to(device)).cpu()
        emb_te = model.encode(X_test.to(device)).cpu()
    torch.save(emb_tr, f"backend/embeddings/train_embeddings_{prefix}.pt")
    torch.save(emb_va, f"backend/embeddings/val_embeddings_{prefix}.pt")
    torch.save(emb_te, f"backend/embeddings/test_embeddings_{prefix}.pt")
    return emb_tr.numpy(), emb_va.numpy(), emb_te.numpy()

emb_tr_bce, emb_va_bce, emb_te_bce = extract_all_embeddings(model_bce, "bce")
emb_tr_focal, emb_va_focal, emb_te_focal = extract_all_embeddings(model_focal, "focal")
print("Saved embeddings for BCE and Focal models.")

# Load regression embeddings for baseline comparison
emb_tr_reg = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True).numpy()
emb_va_reg = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True).numpy()
emb_te_reg = torch.load("backend/embeddings/test_embeddings_improved.pt", weights_only=True).numpy()

# 4. Statistical Embedding Analysis
def analyze_embedding_space(emb):
    vars_ = emb.var(axis=0)
    cov = np.cov(emb, rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigs_pos = np.maximum(eigs, 1e-15)
    p = eigs_pos / np.sum(eigs_pos)
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    sub_n = min(1000, len(emb))
    idx = np.random.RandomState(42).choice(len(emb), sub_n, replace=False)
    pw = pdist(emb[idx])
    return {
        "eff_rank": eff_rank,
        "mean_var": float(vars_.mean()),
        "min_var": float(vars_.min()),
        "max_var": float(vars_.max()),
        "active_dims_1e4": int((vars_ > 1e-4).sum()),
        "active_dims_1e3": int((vars_ > 1e-3).sum()),
        "top3_ratio": float(np.sum(eigs[:3]) / np.sum(eigs_pos)),
        "pw_mean": float(pw.mean()),
        "vars": vars_.tolist()
    }

emb_stats = {
    "Regression": analyze_embedding_space(emb_te_reg),
    "BCE": analyze_embedding_space(emb_te_bce),
    "Focal": analyze_embedding_space(emb_te_focal)
}

# Linear Probes on embeddings
X_tr_last = X_train[:, -1, :].numpy()
X_te_last = X_test[:, -1, :].numpy()
y_tr_np = y_train_bin.numpy()
y_te_np = y_test_bin.numpy()

def run_probes(emb_tr, emb_te):
    # Logistic Regression Probe
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(emb_tr, y_tr_np)
    preds = clf.predict(emb_te)
    probs = clf.predict_proba(emb_te)[:, 1]
    acc = accuracy_score(y_te_np, preds)
    bacc = balanced_accuracy_score(y_te_np, preds)
    auc = roc_auc_score(y_te_np, probs)
    return {"acc": float(acc), "bacc": float(bacc), "auc": float(auc)}

probe_raw = run_probes(X_tr_last, X_te_last)
probe_reg = run_probes(emb_tr_reg, emb_te_reg)
probe_bce = run_probes(emb_tr_bce, emb_te_bce)
probe_focal = run_probes(emb_tr_focal, emb_te_focal)

# 5. Full Evaluation on Untouched Test Data
def full_test_eval(model, threshold, name="Model"):
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_test_bin.numpy()
    preds = (probs >= threshold).astype(float)
    
    acc = float(accuracy_score(y_true, preds))
    bacc = float(balanced_accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds).tolist()
    auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    
    # Per-stock
    per_stock = {}
    for sym in ["HDFCBANK", "INFY", "ITC", "RELIANCE", "TCS"]:
        idx = (meta_test["Symbol"] == sym).values
        sub_true = y_true[idx]
        sub_preds = preds[idx]
        per_stock[sym] = {
            "Accuracy": float(accuracy_score(sub_true, sub_preds)),
            "Balanced_Acc": float(balanced_accuracy_score(sub_true, sub_preds)),
            "Precision": float(precision_score(sub_true, sub_preds, zero_division=0)),
            "Recall": float(recall_score(sub_true, sub_preds, zero_division=0)),
            "F1": float(f1_score(sub_true, sub_preds, zero_division=0)),
            "Count": int(len(sub_true))
        }
    return {
        "Name": name,
        "Threshold": threshold,
        "Accuracy": acc,
        "Balanced_Accuracy": bacc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Confusion_Matrix": cm,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Per_Stock": per_stock,
        "Probs": probs.tolist(),
        "Preds": preds.tolist()
    }

eval_bce = full_test_eval(model_bce, th_bce, "BCE_Classifier (th_val)")
eval_bce_default = full_test_eval(model_bce, 0.5, "BCE_Classifier (th=0.5)")
eval_focal = full_test_eval(model_focal, th_focal, "Focal_Classifier (th_val)")
eval_focal_default = full_test_eval(model_focal, 0.5, "Focal_Classifier (th=0.5)")

# Baselines on test set
# 1. Majority baseline from train: predict UP (1.0)
maj_preds = np.ones_like(y_te_np)
maj_eval = {
    "Name": "Majority-Direction (Train Prior=UP)",
    "Accuracy": float(accuracy_score(y_te_np, maj_preds)),
    "Balanced_Accuracy": float(balanced_accuracy_score(y_te_np, maj_preds)),
    "Precision": float(precision_score(y_te_np, maj_preds, zero_division=0)),
    "Recall": float(recall_score(y_te_np, maj_preds, zero_division=0)),
    "F1": float(f1_score(y_te_np, maj_preds, zero_division=0)),
    "Confusion_Matrix": confusion_matrix(y_te_np, maj_preds).tolist()
}

# 2. Regression baseline (55.53% conventional dir acc)
# Load regression evaluation
with open("reports/model_training/final_test_evaluation.json", "r") as f:
    reg_eval_data = json.load(f)

# 6. Plotting Loss Curves & Evaluation Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BCE curves
axes[0].plot(hist_bce["train_loss"], label="Train Loss", color="tab:blue")
axes[0].plot(hist_bce["val_loss"], label="Val Loss", color="tab:orange")
axes[0].axvline(ep_bce - 1, color="red", linestyle="--", label=f"Best Ep {ep_bce}")
axes[0].set_title(f"BCE Loss Curve (Early Stopped @ Ep {ep_bce})", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Focal curves
axes[1].plot(hist_focal["train_loss"], label="Train Loss", color="tab:blue")
axes[1].plot(hist_focal["val_loss"], label="Val Loss", color="tab:orange")
axes[1].axvline(ep_focal - 1, color="red", linestyle="--", label=f"Best Ep {ep_focal}")
axes[1].set_title(f"Focal Loss Curve (gamma={best_fl_cfg[0]}, alpha={best_fl_cfg[1]})", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Focal Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_loss_path = "reports/model_training/figures/exp1_loss_curves.png"
plt.savefig(fig_loss_path, dpi=200)
plt.close()

# ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr_bce, tpr_bce, _ = roc_curve(y_te_np, eval_bce["Probs"])
fpr_fl, tpr_fl, _ = roc_curve(y_te_np, eval_focal["Probs"])

axes[0].plot(fpr_bce, tpr_bce, label=f"BCE (AUC = {eval_bce['ROC_AUC']:.3f})", color="tab:blue")
axes[0].plot(fpr_fl, tpr_fl, label=f"Focal (AUC = {eval_focal['ROC_AUC']:.3f})", color="tab:orange")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.500)")
axes[0].set_title("Test ROC Curves", fontsize=12, fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

p_bce, r_bce, _ = precision_recall_curve(y_te_np, eval_bce["Probs"])
p_fl, r_fl, _ = precision_recall_curve(y_te_np, eval_focal["Probs"])

axes[1].plot(r_bce, p_bce, label=f"BCE (PR-AUC = {eval_bce['PR_AUC']:.3f})", color="tab:blue")
axes[1].plot(r_fl, p_fl, label=f"Focal (PR-AUC = {eval_focal['PR_AUC']:.3f})", color="tab:orange")
axes[1].axhline(y_te_np.mean(), color="k", linestyle="--", alpha=0.6, label=f"Base Rate ({y_te_np.mean():.3f})")
axes[1].set_title("Test Precision-Recall Curves", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_roc_path = "reports/model_training/figures/exp1_roc_pr_curves.png"
plt.savefig(fig_roc_path, dpi=200)
plt.close()

# Save complete results dictionary
all_results = {
    "majority_baseline": maj_eval,
    "regression_baseline": {
        "Conventional_Dir_Acc": reg_eval_data["improved_lstm"]["Conv_Dir_Acc"],
        "Filtered_Dir_Acc": reg_eval_data["improved_lstm"]["Filt_Dir_Acc"],
        "MAE": reg_eval_data["improved_lstm"]["MAE"],
        "RMSE": reg_eval_data["improved_lstm"]["RMSE"],
        "Per_Stock": reg_eval_data["per_company"]
    },
    "bce_eval": eval_bce,
    "bce_eval_default": eval_bce_default,
    "focal_eval": eval_focal,
    "focal_eval_default": eval_focal_default,
    "focal_tuning_records": fl_tuning_records,
    "best_focal_cfg": best_fl_cfg,
    "thresholds": {"bce": th_bce, "focal": th_focal},
    "embedding_stats": emb_stats,
    "probes": {
        "raw_features": probe_raw,
        "regression_embeddings": probe_reg,
        "bce_embeddings": probe_bce,
        "focal_embeddings": probe_focal
    }
}

with open("reports/model_training/exp1_complete_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n--- EXPERIMENT 1 COMPREHENSIVE SUMMARY ---")
print(f"Majority Baseline Acc: {maj_eval['Accuracy']*100:.2f}% | BAcc: {maj_eval['Balanced_Accuracy']*100:.2f}%")
print(f"Regression LSTM Dir Acc: {reg_eval_data['improved_lstm']['Conv_Dir_Acc']*100:.2f}% (MAE: {reg_eval_data['improved_lstm']['MAE']:.6f})")
print(f"BCE Classifier (th=0.50): Acc={eval_bce_default['Accuracy']*100:.2f}%, BAcc={eval_bce_default['Balanced_Accuracy']*100:.2f}%, F1={eval_bce_default['F1']:.4f}, ROC-AUC={eval_bce_default['ROC_AUC']:.4f}")
print(f"BCE Classifier (th={th_bce:.3f}): Acc={eval_bce['Accuracy']*100:.2f}%, BAcc={eval_bce['Balanced_Accuracy']*100:.2f}%, F1={eval_bce['F1']:.4f}, ROC-AUC={eval_bce['ROC_AUC']:.4f}")
print(f"Focal Classifier (th=0.50): Acc={eval_focal_default['Accuracy']*100:.2f}%, BAcc={eval_focal_default['Balanced_Accuracy']*100:.2f}%, F1={eval_focal_default['F1']:.4f}, ROC-AUC={eval_focal_default['ROC_AUC']:.4f}")
print(f"Focal Classifier (th={th_focal:.3f}): Acc={eval_focal['Accuracy']*100:.2f}%, BAcc={eval_focal['Balanced_Accuracy']*100:.2f}%, F1={eval_focal['F1']:.4f}, ROC-AUC={eval_focal['ROC_AUC']:.4f}")
print("\nLinear Probes on Test Embeddings:")
print(f"  Raw Features Probe       : Acc={probe_raw['acc']*100:.2f}%, BAcc={probe_raw['bacc']*100:.2f}%, AUC={probe_raw['auc']:.4f}")
print(f"  Regression Embeddings    : Acc={probe_reg['acc']*100:.2f}%, BAcc={probe_reg['bacc']*100:.2f}%, AUC={probe_reg['auc']:.4f}")
print(f"  BCE Embeddings           : Acc={probe_bce['acc']*100:.2f}%, BAcc={probe_bce['bacc']*100:.2f}%, AUC={probe_bce['auc']:.4f}")
print(f"  Focal Embeddings         : Acc={probe_focal['acc']*100:.2f}%, BAcc={probe_focal['bacc']*100:.2f}%, AUC={probe_focal['auc']:.4f}")
print("\nEmbedding Geometry:")
for k, s in emb_stats.items():
    print(f"  {k:<12}: EffRank={s['eff_rank']:.2f}/32, MeanVar={s['mean_var']:.6f}, Active(>1e-4)={s['active_dims_1e4']}, Top3Eig={s['top3_ratio']*100:.1f}%, PWDist={s['pw_mean']:.4f}")

