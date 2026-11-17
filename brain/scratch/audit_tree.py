import os
import glob

print("=== PROJECT TREE RECURSIVE ===")
for root, dirs, files in os.walk("."):
    if any(p in root for p in [".git", "neuro_env", "__pycache__", ".ipynb_checkpoints"]):
        continue
    rel_root = os.path.relpath(root, ".")
    print(f"\n[{rel_root}]")
    for f in sorted(files):
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"  {f:<45} ({size:>10,} bytes)")
