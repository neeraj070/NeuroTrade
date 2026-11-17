import json

with open("backend/notebooks/05_ppo_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print("=== Cell 5 output (Training loop) ===")
print("".join(code_cells[4]["outputs"][0]["text"]))

print("\n=== Cell 6 output (Evaluation) ===")
print("".join(code_cells[5]["outputs"][0]["text"]))
