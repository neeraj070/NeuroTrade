import json
with open("brain/scratch/audit_summary.json") as f:
    d = json.load(f)
print("Raw files exist:", d["raw_files_exist"])
print("Proc files exist:", d["proc_files_exist"])
print("Train tensors exist:", d["train_tensors_exist"], d["train_shape"])
print("Val tensors exist:", d["val_tensors_exist"], d["val_shape"])
print("Test tensors exist:", d["test_tensors_exist"], d["test_shape"])
print("\nCheckpoints:")
for k, v in d["ckpts_exist"].items():
    print(f"  {k}: {v}")
print("\nEmbeddings:")
for k, v in d["embeddings_exist"].items():
    print(f"  {k}: {v}")
print("\nReports:")
for k, v in d["reports_exist"].items():
    print(f"  {k}: {v}")
print("\nNotebooks:")
for k, v in d["notebook_status"].items():
    print(f"  {k}: exists={v.get('exists')}, executed_cells={v.get('executed_cells')}/{v.get('code_cells')}")
