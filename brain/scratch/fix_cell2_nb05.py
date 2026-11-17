import json
import os

nb_path = "backend/notebooks/05_ppo_training.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 2
cell2_src = """import os
import sys
import time
import json
import random
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Robust root discovery
curr = os.path.abspath(os.getcwd())
while curr and not os.path.exists(os.path.join(curr, "backend")):
    parent = os.path.dirname(curr)
    if parent == curr:
        break
    curr = parent
PROJECT_ROOT = curr
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"Project root confirmed: {PROJECT_ROOT}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)"""

for c in nb["cells"]:
    if c["cell_type"] == "code" and "PROJECT_ROOT" in "".join(c["source"]):
        c["source"] = [l + "\n" for l in cell2_src.split("\n")]
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 2 with sys.path.insert(0, PROJECT_ROOT)")
