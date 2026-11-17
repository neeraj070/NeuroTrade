import json

with open("backend/notebooks/archive/03_rl_development_legacy.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "ActorCritic" in "".join(cell["source"]):
        print(f"=== Cell {idx} ===")
        print("".join(cell["source"]))
