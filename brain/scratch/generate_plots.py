import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from test_env_prototype import MarketTradingEnv

os.makedirs("reports/reinforcement_learning", exist_ok=True)

# Load data
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
emb_val   = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True)
emb_test  = torch.load("backend/embeddings/test_embeddings_improved.pt", weights_only=True)

X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
X_val   = torch.load("data/training/validation/X_val.pt", weights_only=True)
X_test  = torch.load("data/training/test/X_test.pt", weights_only=True)

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val   = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test  = pd.read_csv("data/training/test/metadata_test.csv")

raw_prices = {}
for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
    raw_prices[sym] = pd.read_csv(f"data/raw/{sym}.csv")

# 1. Run baselines on Test set and record equity curves
equity_curves = {}
symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

for sym in symbols:
    equity_curves[sym] = {}
    proc_df = pd.read_csv(f"data/processed/{sym}_processed.csv")
    for strat in ["hold", "buy_and_hold", "ma_crossover"]:
        env = MarketTradingEnv(emb_test, X_test, meta_test, raw_prices, symbol=sym)
        obs, info = env.reset()
        vals = [100000.0]
        dates = [env.end_dates[0]]
        terminated = False
        step_num = 0
        while not terminated:
            if strat == "hold":
                act = 0
            elif strat == "buy_and_hold":
                act = 1 if step_num == 0 else 0
            elif strat == "ma_crossover":
                d = env.end_dates[env.current_step]
                row = proc_df[proc_df["Date"] == d]
                if len(row) > 0 and not pd.isna(row["SMA_10"].values[0]) and not pd.isna(row["SMA_50"].values[0]):
                    act = 1 if (row["SMA_10"].values[0] > row["SMA_50"].values[0] and env.shares == 0) else (2 if (row["SMA_10"].values[0] <= row["SMA_50"].values[0] and env.shares > 0) else 0)
                else:
                    act = 0
            obs, reward, terminated, truncated, info = env.step(act)
            vals.append(info["portfolio_value"])
            dates.append(info["date"])
            step_num += 1
        equity_curves[sym][strat] = (dates, vals)

# Plot Test Equity Curves
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, sym in enumerate(symbols):
    ax = axes[idx]
    for strat, label, col, ls in [
        ("hold", "Hold (Cash)", "gray", "--"),
        ("buy_and_hold", "Buy & Hold", "tab:blue", "-"),
        ("ma_crossover", "MA Crossover", "tab:orange", "-.")
    ]:
        dates, vals = equity_curves[sym][strat]
        ax.plot(range(len(vals)), np.array(vals)/1000.0, label=label, color=col, linestyle=ls, linewidth=1.8)
    ax.set_title(f"{sym} (Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value (₹ thousands)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

# Average summary in 6th subplot
ax6 = axes[5]
splits = ["Train", "Val", "Test"]
bnh_rets = [32.72, -9.49, -15.74]
ma_rets = [14.30, -2.23, -9.63]
hold_rets = [0.0, 0.0, 0.0]

x = np.arange(len(splits))
w = 0.25
ax6.bar(x - w, bnh_rets, width=w, label="Buy & Hold", color="tab:blue")
ax6.bar(x, ma_rets, width=w, label="MA Crossover", color="tab:orange")
ax6.bar(x + w, hold_rets, width=w, label="Hold (Cash)", color="gray")
ax6.axhline(0, color="black", linewidth=0.8)
ax6.set_title("Average Return by Partition (%)", fontsize=12, fontweight="bold")
ax6.set_xticks(x)
ax6.set_xticklabels(splits)
ax6.set_ylabel("Average Total Return (%)")
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)

plt.tight_layout()
plt.savefig("reports/reinforcement_learning/baseline_performance.png", dpi=200)
plt.close()
print("Saved reports/reinforcement_learning/baseline_performance.png [OK]")

