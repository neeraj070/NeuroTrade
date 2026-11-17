import json

with open("backend/notebooks/05_ppo_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print("=== Code Cell 2 (Network Definition and Test) ===")
print("".join(code_cells[2]["source"]))
