import json
import torch
import numpy as np
import random
import os

with open("backend/notebooks/05_ppo_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells in notebook: {len(code_cells)}")

for idx, c in enumerate(code_cells):
    src = "".join(c["source"])
    print(f"\n--- Code Cell {idx} (Exec Count: {c.get('execution_count')}) ---")
    first_lines = "\n".join(src.strip().splitlines()[:5])
    print(first_lines)
    # Check if there are torch, numpy, or random calls in this cell
    calls = []
    if "torch.manual_seed" in src or "set_seed" in src: calls.append("seed")
    if "torch.randn" in src: calls.append("torch.randn")
    if "ActorCriticTradingNetwork" in src: calls.append("ActorCriticTradingNetwork")
    if "np.random" in src: calls.append("np.random")
    if "Categorical" in src: calls.append("Categorical")
    print(f"Identified RNG/Network elements: {calls}")
