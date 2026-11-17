import json

with open('backend/notebooks/03_rl_development.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    print(f"=== Cell {idx} ({cell['cell_type']}) ===")
    src = cell['source'] if isinstance(cell['source'], str) else "".join(cell['source'])
    print(src[:400])

