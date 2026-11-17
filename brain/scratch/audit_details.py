import json
import torch
import hashlib
import numpy as np
import pandas as pd

print("=== CHECKING AUDIT ITEMS 5, 9, 10 & SEED DETAILS ===")

# 1. Hashes of improved_rnn_encoder.pt and embeddings
def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

encoder_path = "backend/checkpoints/improved_rnn_encoder.pt"
print(f"improved_rnn_encoder.pt SHA256: {get_file_hash(encoder_path)}")
print(f"train_embeddings_improved.pt SHA256: {get_file_hash('backend/embeddings/train_embeddings_improved.pt')}")
print(f"val_embeddings_improved.pt SHA256: {get_file_hash('backend/embeddings/val_embeddings_improved.pt')}")
print(f"test_embeddings_improved.pt SHA256: {get_file_hash('backend/embeddings/test_embeddings_improved.pt')}")

# 2. Check exact numbers in ppo_training_results.json
with open("reports/reinforcement_learning/ppo_training_results.json", "r") as f:
    res = json.load(f)

test_stocks = res["test_per_stock"]
returns = [s["Total_Return%"] for s in test_stocks]
sharpes = [s["Sharpe"] for s in test_stocks]
drawdowns = [s["Max_Drawdown%"] for s in test_stocks]
trades = [s["Trades"] for s in test_stocks]
costs = [s["Costs"] for s in test_stocks]

print("\n--- Per Stock Test Statistics ---")
for s in test_stocks:
    print(f"{s['Symbol']:10s}: Ret={s['Total_Return%']:+7.2f}%, Sharpe={s['Sharpe']:+5.2f}, DD={s['Max_Drawdown%']:6.2f}%, Trades={s['Trades']:2d}, Costs=Rs.{s['Costs']:8.2f}")

avg_ret = np.mean(returns)
avg_sharpe = np.mean(sharpes)
avg_dd = np.mean(drawdowns)
avg_trades = np.mean(trades)
total_costs = np.sum(costs)
avg_cost = np.mean(costs)

print(f"\nCalculated Averages:")
print(f"Avg Return%      : {avg_ret:.6f}% (Reported: +0.34%)")
print(f"Avg Sharpe       : {avg_sharpe:.6f} (Reported: +0.13)")
print(f"Avg Max Drawdown%: {avg_dd:.6f}% (Reported: -13.03%)")
print(f"Avg Trades/Stock : {avg_trades:.1f} (Reported: 16.8)")
print(f"Total Costs (Sum): Rs.{total_costs:.2f}")
print(f"Avg Costs/Stock  : Rs.{avg_cost:.2f}")

# 3. Check batch update math
buffer_size = 4000
batch_size = 128
epochs = 4
num_batches = int(np.ceil(buffer_size / batch_size)) # 32
updates_per_iter = num_batches * epochs # 32 * 4 = 128
total_iterations = 35
total_updates = updates_per_iter * total_iterations
total_transitions = buffer_size * total_iterations

print(f"\n--- Batch and Update Math ---")
print(f"Buffer size: {buffer_size}")
print(f"Batch size : {batch_size}")
print(f"Batches per epoch: ceil({buffer_size}/{batch_size}) = {num_batches}")
print(f"Epochs per iter  : {epochs}")
print(f"Updates per iter : {updates_per_iter}")
print(f"Total iterations : {total_iterations}")
print(f"Total updates    : {total_updates} (Reported: 4,480)")
print(f"Total transitions: {total_transitions} (Reported: 140,000)")
assert total_updates == 4480, f"Expected 4480, got {total_updates}"
assert total_transitions == 140000, f"Expected 140000, got {total_transitions}"
print("Math is 100% mathematically and internally consistent!")
