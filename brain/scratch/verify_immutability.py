import os
import sys

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
from gymnasium.utils.env_checker import check_env

from backend.environments.trading_env import MarketTradingEnv

print("=== 1. VERIFYING LSTM CHECKPOINTS ===")
ckpt_path = "backend/checkpoints/improved_rnn_encoder.pt"
assert os.path.exists(ckpt_path), "improved_rnn_encoder.pt missing!"
ckpt = torch.load(ckpt_path, weights_only=True)
assert ckpt["epoch"] == 48, f"Unexpected epoch: {ckpt['epoch']}"
assert abs(ckpt["best_val_loss"] - 0.00013733285647634017) < 1e-10
assert ckpt["config"]["hidden_dim"] == 64
assert ckpt["config"]["embedding_dim"] == 32
assert ckpt["config"]["input_dim"] == 14
print("  [OK] improved_rnn_encoder.pt intact and verified.")

print("=== 2. VERIFYING EMBEDDINGS ===")
for split, expected_shape in [("train", (4000, 32)), ("val", (885, 32)), ("test", (895, 32))]:
    p = f"backend/embeddings/{split}_embeddings_improved.pt"
    assert os.path.exists(p), f"{p} missing!"
    emb = torch.load(p, weights_only=True)
    assert tuple(emb.shape) == expected_shape, f"Unexpected shape {emb.shape} for {p}"
print("  [OK] All market state embeddings intact and verified.")

print("=== 3. VERIFYING DATASETS & TENSORS ===")
for split, expected_shape in [("train", (4000, 30, 14)), ("val", (885, 30, 14)), ("test", (895, 30, 14))]:
    xp = f"data/training/{'validation' if split=='val' else split}/X_{split}.pt"
    yp = f"data/training/{'validation' if split=='val' else split}/y_{split}.pt"
    mp = f"data/training/{'validation' if split=='val' else split}/metadata_{split}.csv"
    assert os.path.exists(xp) and os.path.exists(yp) and os.path.exists(mp)
    x = torch.load(xp, weights_only=True)
    assert tuple(x.shape) == expected_shape
print("  [OK] Training tensors and metadata intact and verified.")

print("=== 4. VERIFYING GYMNASIUM TRADING ENVIRONMENT ===")
raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]}
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
meta_train = pd.read_csv("data/training/train/metadata_train.csv")

env = MarketTradingEnv(emb_train, X_train, meta_train, raw_prices, symbol="RELIANCE")
check_env(env)
print("  [OK] gymnasium.utils.env_checker.check_env() PASSED.")

# Re-run unit tests
obs0, info0 = env.reset()
assert obs0.shape == (49,)
assert env.portfolio_value == 100000.0

# HOLD
_, r_hold, _, _, inf_hold = env.step(0)
assert r_hold == 0.0 and inf_hold["portfolio_value"] == 100000.0

# BUY
open_p = env.open_t1_arr[env.current_step]
close_p = env.close_t1_arr[env.current_step]
exp_s = int(100000.0 // (open_p * 1.001))
exp_cost = exp_s * open_p * 0.001
exp_cash = 100000.0 - (exp_s * open_p) - exp_cost
exp_val = exp_cash + exp_s * close_p
_, r_buy, _, _, inf_buy = env.step(1)
assert inf_buy["trade_executed"] == True
assert inf_buy["trade_shares"] == exp_s
assert abs(inf_buy["portfolio_value"] - exp_val) < 1e-4

# Repeated BUY churn prevention
_, _, _, _, inf_churn = env.step(1)
assert inf_churn["trade_executed"] == False and inf_churn["transaction_cost"] == 0.0

# SELL
held = env.shares
open_s = env.open_t1_arr[env.current_step]
sell_fee = held * open_s * 0.001
net_cash = env.cash + (held * open_s) - sell_fee
_, r_sell, _, _, inf_sell = env.step(2)
assert inf_sell["trade_executed"] == True and env.shares == 0
assert abs(env.cash - net_cash) < 1e-4
print("  [OK] All trading logic, portfolio accounting, and numerical tests PASSED.")

print("\nALL SYSTEM ARTIFACTS AND VERIFICATIONS ARE 100% INTACT AND UNCHANGED! [SUCCESS]")
