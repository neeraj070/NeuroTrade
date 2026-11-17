import json, os, hashlib
from pathlib import Path
from PIL import Image
from docx import Document
from pptx import Presentation

BASE = Path(r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade")

print("=== CHECK 1: Notebook inventory ===")
all_nbs = sorted(list(BASE.glob("**/*.ipynb")))
for nb in all_nbs:
    print(nb.relative_to(BASE))
print(f"Total notebooks found: {len(all_nbs)}")

print("\n=== CHECK 2: Transaction Friction ===")
with open(BASE / "reports/reinforcement_learning/ppo_reproducibility_results.json") as f:
    rep = json.load(f)
test_stocks = rep['run1']['test_per_stock']
tot_cost = rep['run1']['test_summary']['Total_Costs']
avg_cost = rep['run1']['test_summary']['Avg_Costs']
print(f"Total Costs across all 5 assets: {tot_cost:.2f}")
print(f"Avg Cost per asset: {avg_cost:.2f}")
for s in test_stocks:
    print(f"  {s['Symbol']:<10}: Trades={s['Trades']:2d}, Costs=Rs.{s['Costs']:.2f}")

print("\n=== CHECK 3: Buy-and-Hold Trade & Cost Accounting ===")
with open(BASE / "reports/reinforcement_learning/ppo_training_results.json") as f:
    tr = json.load(f)
bh = tr['baselines_comparison']['Buy_and_Hold']
print(f"B&H stats in ppo_training_results.json: {bh}")

# Let's inspect how B&H was simulated in 05_ppo_training.ipynb
with open(BASE / "backend/notebooks/05_ppo_training.ipynb", encoding='utf-8') as f:
    nb05 = json.load(f)
for cell in nb05['cells']:
    src = "".join(cell.get('source', ''))
    if "Buy_and_Hold" in src or "buy_and_hold" in src or "baseline" in src.lower():
        print("Found B&H reference in notebook cell:")
        for line in src.split("\n"):
            if "buy" in line.lower() or "hold" in line.lower() or "cost" in line.lower() or "trade" in line.lower():
                print("  ", line[:100])

print("\n=== CHECK 4: Recalculate LSTM Directional Accuracy ===")
# Let's check evaluation_metrics.json or LSTM notebooks/reports
for p in BASE.glob("**/evaluation_metrics.json"):
    print(f"Found {p}:")
    print(p.read_text()[:300])

# Check test predictions / tensors
y_test_path = BASE / "data/training/test/y_test.pt"
print(f"y_test exists: {y_test_path.exists()}")
if y_test_path.exists():
    import torch
    y_test = torch.load(y_test_path)
    print(f"y_test shape: {y_test.shape}")

# Let's search where 55.53% or 497 came from
for p in BASE.glob("reports/**/*.json"):
    txt = p.read_text(encoding='utf-8', errors='ignore')
    if "55.53" in txt or "497" in txt:
        print(f"Found 55.53/497 in {p}")
for p in BASE.glob("reports/**/*.md"):
    txt = p.read_text(encoding='utf-8', errors='ignore')
    if "55.53" in txt or "497" in txt:
        print(f"Found 55.53/497 in {p}")

print("\n=== CHECK 5: Strategy comparison metrics ===")
print("Baselines from ppo_training_results.json:")
for k, v in tr['baselines_comparison'].items():
    print(f"  {k}: {v}")

print("\n=== CHECK 6: Diagrams and Visualizations ===")
diag_dir = BASE / "reports/final_submission/Architecture_Diagrams"
vis_dir = BASE / "reports/final_submission/Result_Visualizations"
for d in sorted(diag_dir.glob("*.png")):
    im = Image.open(d)
    print(f"  Diagram: {d.name:<45} | Size: {im.size} | Mode: {im.mode}")
for v in sorted(vis_dir.glob("*.png")):
    im = Image.open(v)
    print(f"  Vis:     {v.name:<45} | Size: {im.size} | Mode: {im.mode}")

# Check DOCX and PPTX image parts
doc = Document(str(BASE / "reports/final_submission/Final_Project_Report.docx"))
img_count_docx = len(doc.inline_shapes)
print(f"DOCX inline shapes (images): {img_count_docx}")

prs = Presentation(str(BASE / "reports/final_submission/Project_Presentation.pptx"))
img_count_pptx = 0
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.shape_type == 13: # picture
            img_count_pptx += 1
print(f"PPTX picture shapes: {img_count_pptx}")

print("\n=== CHECK 8: Dependencies in requirements.txt ===")
reqs = (BASE / "requirements.txt").read_text().splitlines()
print(f"Total lines in requirements.txt: {len(reqs)}")
import importlib
for r in reqs:
    r = r.strip()
    if r and not r.startswith("#"):
        pkg = r.split(">=")[0].split("==")[0].split("<")[0].strip()
        mod_name = pkg.lower().replace("-", "_")
        try:
            m = importlib.import_module(mod_name)
            ver = getattr(m, "__version__", "unknown")
            print(f"  {pkg:<20}: IMPORT OK (v{ver})")
        except Exception as e:
            # try alternate name (e.g. scikit-learn -> sklearn, pyyaml -> yaml)
            alt = {"scikit_learn": "sklearn", "python_docx": "docx", "python_pptx": "pptx"}.get(mod_name, mod_name)
            try:
                m = importlib.import_module(alt)
                ver = getattr(m, "__version__", "unknown")
                print(f"  {pkg:<20}: IMPORT OK via {alt} (v{ver})")
            except Exception as e2:
                print(f"  {pkg:<20}: IMPORT FAILED ({e2})")

print("\n=== CHECK 10: Deliverables Corruption Check ===")
delivs = [
    "reports/final_submission/Final_Project_Report.docx",
    "reports/final_submission/Project_Presentation.pptx",
    "reports/final_submission/Reproducibility_Guide.md",
    "reports/final_submission/Notebook_Audit_Report.md",
    "reports/final_submission/Final_Submission_Checklist.md",
]
for d in delivs:
    p = BASE / d
    try:
        if d.endswith(".docx"):
            d_obj = Document(str(p))
            print(f"  {d}: OK, {len(d_obj.paragraphs)} paragraphs, {len(d_obj.tables)} tables")
        elif d.endswith(".pptx"):
            p_obj = Presentation(str(p))
            print(f"  {d}: OK, {len(p_obj.slides)} slides")
        else:
            txt = p.read_text(encoding='utf-8')
            print(f"  {d}: OK, {len(txt):,} chars, {len(txt.splitlines())} lines")
    except Exception as e:
        print(f"  {d}: CORRUPT ({e})")
