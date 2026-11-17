import json
import torch

with open("backend/notebooks/05_ppo_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Inspect outputs of Cell 10 (code cell 5)
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
train_cell = code_cells[4] # 0-indexed: cell 0 is root, cell 1 is load data, cell 2 is network, cell 3 is eval func, cell 4 is training loop
print("Cell 5 (Training loop) outputs:")
for o in train_cell.get("outputs", []):
    print("Output type:", o.get("output_type"))
    if o.get("name") == "stdout":
        text = "".join(o.get("text", []))
        print(text)

# Inspect best checkpoint
best_ckpt = torch.load("backend/checkpoints/ppo_best_agent.pt", weights_only=True)
print("\nBest checkpoint iteration:", best_ckpt.get("iteration"))
print("Best checkpoint val summary:", best_ckpt.get("val_summary"))
