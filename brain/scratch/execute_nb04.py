import os
import sys
import json
import nbformat
from nbclient import NotebookClient

nb_path = "backend/notebooks/04_rl_development.ipynb"

# 1. Read notebook
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# 2. Adjust cell 2 to robustly find project root
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "PROJECT_ROOT" in cell.source:
        cell.source = """import os
import sys
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Ensure working directory is project root
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "backend")):
    if os.path.exists(os.path.join(cwd, "..", "backend")):
        os.chdir(os.path.abspath(os.path.join(cwd, "..")))
    elif os.path.exists(os.path.join(cwd, "..", "..", "backend")):
        os.chdir(os.path.abspath(os.path.join(cwd, "..", "..")))
PROJECT_ROOT = os.getcwd()
print(f"Project root: {PROJECT_ROOT}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)"""
        print(f"Updated Cell {idx} with robust root resolution.")

# 3. Execute notebook using NotebookClient
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "backend/notebooks"}})
print("Executing notebook...")
client.execute()
print("Execution complete! Saving notebook...")

# 4. Save executed notebook
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Saved executed notebook to {nb_path} [SUCCESS]")

# 5. Verification of outputs
code_cells = [c for c in nb.cells if c.cell_type == "code"]
cells_with_out = [c for c in code_cells if len(c.get("outputs", [])) > 0]
print(f"Total code cells: {len(code_cells)}, Executed cells with outputs: {len(cells_with_out)}")
assert len(code_cells) == len(cells_with_out), "Not all code cells produced outputs!"

for i, cell in enumerate(code_cells):
    print(f"\n--- Code Cell {i+1} Output Preview ---")
    for out in cell.get("outputs", []):
        if "text" in out:
            print(out["text"].strip()[:200])
        elif "data" in out and "text/plain" in out["data"]:
            print(out["data"]["text/plain"].strip()[:200])
        elif "data" in out and "image/png" in out["data"]:
            print(f"[Image Output: {len(out['data']['image/png'])} base64 chars]")
