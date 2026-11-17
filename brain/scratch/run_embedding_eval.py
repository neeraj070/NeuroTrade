import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.spatial.distance import pdist

def compute_rank_and_stats(emb):
    vars_ = emb.var(axis=0)
    cov = np.cov(emb, rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigs_pos = np.maximum(eigs, 1e-15)
    p = eigs_pos / np.sum(eigs_pos)
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    sub_n = min(1000, len(emb))
    idx = np.random.RandomState(42).choice(len(emb), sub_n, replace=False)
    pw = pdist(emb[idx])
    
    # Active dimensions thresholded at 1e-4 and 1e-3
    active_1e4 = int((vars_ > 1e-4).sum())
    active_1e3 = int((vars_ > 1e-3).sum())
    
    return {
        "eff_rank": eff_rank,
        "mean_var": float(vars_.mean()),
        "min_var": float(vars_.min()),
        "max_var": float(vars_.max()),
        "active_dims_1e4": active_1e4,
        "active_dims_1e3": active_1e3,
        "top3_ratio": float(np.sum(eigs[:3]) / np.sum(eigs_pos)),
        "pw_mean": float(pw.mean()),
        "pw_median": float(np.median(pw)),
        "pw_max": float(pw.max()),
        "eigs": eigs.tolist(),
        "vars": vars_.tolist()
    }

def evaluate_embeddings(emb_train, emb_val, emb_test, y_train_bin, y_val_bin, y_test_bin, X_tr_last, X_te_last):
    stats_tr = compute_rank_and_stats(emb_train)
    stats_va = compute_rank_and_stats(emb_val)
    stats_te = compute_rank_and_stats(emb_test)
    
    # Linear Probes on embeddings
    # 1. Logistic Regression Probe (Directional)
    lr_emb = LogisticRegression(max_iter=1000, random_state=42).fit(emb_train, y_train_bin)
    p_lr_emb = lr_emb.predict(emb_test)
    acc_lr_emb = float(accuracy_score(y_test_bin, p_lr_emb))
    bacc_lr_emb = float(balanced_accuracy_score(y_test_bin, p_lr_emb))
    
    # 2. Ridge Probe (Continuous / Directional via sign)
    ridge_emb = Ridge(alpha=1.0, random_state=42).fit(emb_train, y_train_bin)
    p_ridge_emb = (ridge_emb.predict(emb_test) >= 0.5).astype(float)
    acc_ridge_emb = float(accuracy_score(y_test_bin, p_ridge_emb))
    
    # 3. Logistic Regression Probe on Raw Features at t (for baseline comparison)
    lr_raw = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_last, y_train_bin)
    p_lr_raw = lr_raw.predict(X_te_last)
    acc_lr_raw = float(accuracy_score(y_test_bin, p_lr_raw))
    bacc_lr_raw = float(balanced_accuracy_score(y_test_bin, p_lr_raw))
    
    return {
        "stats_train": stats_tr,
        "stats_val": stats_va,
        "stats_test": stats_te,
        "probe_lr_emb": {"acc": acc_lr_emb, "bacc": bacc_lr_emb},
        "probe_ridge_emb": {"acc": acc_ridge_emb},
        "probe_lr_raw": {"acc": acc_lr_raw, "bacc": bacc_lr_raw}
    }

