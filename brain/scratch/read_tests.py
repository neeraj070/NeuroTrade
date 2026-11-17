import os

for f in ["backend/tests/test_model.py", "backend/tests/verify_dataset.py"]:
    if os.path.exists(f):
        print(f"=== {f} ===")
        with open(f, "r") as fp:
            print(fp.read()[:600])
