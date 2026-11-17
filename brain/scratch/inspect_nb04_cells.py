import json

with open("backend/notebooks/04_rl_development.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        print(f"=== Cell {idx} ===")
        print("".join(cell["source"])[:250])
