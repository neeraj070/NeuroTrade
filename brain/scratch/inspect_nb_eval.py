import json

with open("backend/notebooks/05_ppo_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
eval_cell = code_cells[5] # cell 6
print("Cell 6 (Evaluation) outputs:")
for o in eval_cell.get("outputs", []):
    print("Output type:", o.get("output_type"))
    if o.get("name") == "stdout":
        text = "".join(o.get("text", []))
        print(text)
