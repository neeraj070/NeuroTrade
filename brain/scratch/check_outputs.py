import json
import glob

for p in glob.glob("backend/notebooks/*.ipynb") + glob.glob("data/notebooks/*.ipynb"):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    has_out = any(len(c.get("outputs", [])) > 0 for c in d["cells"])
    print(f"{p}: has_outputs={has_out}")
