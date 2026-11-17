import os
import sys

curr = os.path.abspath(os.getcwd())
while curr and not os.path.exists(os.path.join(curr, "backend")):
    parent = os.path.dirname(curr)
    if parent == curr:
        break
    curr = parent
PROJECT_ROOT = curr
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

mpl_dir = os.path.join(PROJECT_ROOT, "backend", ".mpl_cache")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

import io
import json
import base64
import contextlib
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = os.path.join(PROJECT_ROOT, "backend", "notebooks", "05_ppo_training.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded {nb_path}, cell count: {len(nb['cells'])}")

exec_globals = {
    "__name__": "__main__",
    "__file__": nb_path
}

execution_counter = 1
has_errors = False

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_src = "".join(cell["source"])
        print(f"\n[Executing Cell {idx}] Exec Count {execution_counter}...")
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        plt.close("all")
        cell_outputs = []
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code_src, exec_globals)
        except Exception as e:
            has_errors = True
            tb_str = traceback.format_exc()
            print(f"  ERROR in Cell {idx}: {e}")
            cell_outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb_str.splitlines(keepends=True)
            })
            
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        
        if stdout_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True)
            })
            print(f"  Captured stdout: {len(stdout_text.splitlines())} lines")
            print("  Stdout preview:", stdout_text.strip()[:160].replace("\n", " "))
            
        filtered_stderr = "\n".join([
            l for l in stderr_text.splitlines()
            if not any(k in l for k in ["Matplotlib is building the font cache", "MPLCONFIGDIR", "not a writable directory", "UserWarning"])
        ]).strip()
        
        if filtered_stderr:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": filtered_stderr.splitlines(keepends=True)
            })
            
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": "<Figure size ...>"
                },
                "metadata": {}
            })
            print(f"  Captured inline figure: {len(img_b64)} base64 bytes")
        plt.close("all")
        
        cell["execution_count"] = execution_counter
        cell["outputs"] = cell_outputs
        execution_counter += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"\n==========================================")
print(f"Notebook execution finished: {nb_path}")
print(f"Total Code Cells Executed: {execution_counter - 1}")
print(f"Errors Encountered: {has_errors}")
print(f"==========================================")
assert not has_errors, "Execution encountered errors!"
