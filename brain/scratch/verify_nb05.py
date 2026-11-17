import json

nb_path = "backend/notebooks/05_ppo_training.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"=== VERIFYING {nb_path} ===")
print("nbformat:", nb.get("nbformat"), "nbformat_minor:", nb.get("nbformat_minor"))
print("Total cells:", len(nb["cells"]))

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells: {len(code_cells)}")

for i, c in enumerate(code_cells):
    ec = c.get("execution_count")
    outs = c.get("outputs", [])
    out_types = [o.get("output_type") for o in outs]
    has_error = any(ot == "error" for ot in out_types)
    has_image = any("image/png" in o.get("data", {}) for o in outs)
    print(f"Cell {i+1}: execution_count={ec}, outputs_count={len(outs)}, types={out_types}, has_error={has_error}, has_image={has_image}")
    assert ec is not None, f"Cell {i+1} has None execution_count!"
    assert not has_error, f"Cell {i+1} contains an execution error!"
    
print("\nNotebook 05 reopens cleanly with outputs intact and ZERO execution errors! [VERIFIED]")
