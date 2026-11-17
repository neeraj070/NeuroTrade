import os
import json
import glob
import pandas as pd
import numpy as np

print("=== CHECKING ALL NOTEBOOKS ===")
notebooks = glob.glob("data/notebooks/*.ipynb") + glob.glob("backend/notebooks/*.ipynb")
for nb_path in sorted(notebooks):
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        cells_with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        print(f"{nb_path}: total_cells={len(nb['cells'])}, code_cells={len(code_cells)}, executed_cells={len(cells_with_output)}")
    except Exception as e:
        print(f"{nb_path}: ERROR {e}")
