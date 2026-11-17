import os
import sys
import json
import torch
import numpy as np
import pandas as pd

print("=== FINAL TECHNICAL & METHODOLOGICAL AUDIT ===")

# 1. Inspect ppo_training_results.json
results_path = "reports/reinforcement_learning/ppo_training_results.json"
with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {results_path}")
print(f"Algorithm: {results.get('algorithm')}")
print(f"Best Iteration: {results.get('best_iteration')}")

# Check consistency of iterations, transitions, updates
history = results.get("training_history", [])
print(f"Training History length: {len(history)}")

# Check math of transitions & updates:
# In PPO:
# Training batch size: 5 assets * 800 steps = 4000 transitions per iteration
# 35 iterations * 4000 transitions = 140,000 transitions.
# Mini-batch size: 128
# Mini-batches per iteration: 4000 / 128 = 31.25 -> wait, let's see how many updates per epoch!
# Let's inspect ppo update loop in backend/agents/ppo_agent.py or run_ppo_pipeline.py.

# Check checkpoints
best_ckpt_path = "backend/checkpoints/ppo_best_agent.pt"
best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
print("\n--- Checkpoint: ppo_best_agent.pt ---")
print("Keys in checkpoint:", list(best_ckpt.keys()))
print("Iteration in checkpoint:", best_ckpt.get("iteration"))
print("Val summary in checkpoint:", best_ckpt.get("val_summary"))

latest_ckpt_path = "backend/checkpoints/ppo_trading_agent.pt"
latest_ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=True)
print("\n--- Checkpoint: ppo_trading_agent.pt ---")
print("Iteration in latest checkpoint:", latest_ckpt.get("iteration"))

# Check test summary in json
print("\n--- Test Summary in JSON ---")
for k, v in results.get("test_summary", {}).items():
    print(f"  {k}: {v}")

print("\n--- Baseline Comparison in JSON ---")
for strat, metrics in results.get("baseline_comparison", {}).items():
    print(f"  {strat}: {metrics}")

print("\n--- Validation Selected Iteration 4 details ---")
iter_4 = next((h for h in history if h["iteration"] == 4), None)
print("Iteration 4 in history:", iter_4)
