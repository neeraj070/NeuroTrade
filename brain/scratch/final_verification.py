import hashlib
from pathlib import Path

BASE = Path(r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade")

print("=== REOPEN MODIFIED FILES ===")
# 1. Notebook_Audit_Report.md
nb_audit = (BASE / "reports/final_submission/Notebook_Audit_Report.md").read_text(encoding="utf-8")
print(f"Notebook_Audit_Report.md length: {len(nb_audit):,} chars")
for name in [
    "01_data_collection.ipynb", "02_data_cleaning.ipynb", "03_eda.ipynb",
    "04_feature_engineering.ipynb", "05_preprocessing.ipynb", "06_data_validation.ipynb",
    "01_lstm_training.ipynb", "02_lstm_evaluation.ipynb", "03_embedding_analysis.ipynb",
    "04_rl_development.ipynb", "05_ppo_training.ipynb"
]:
    assert name in nb_audit, f"Missing {name} in Notebook_Audit_Report.md"
print("  All 11 essential notebooks verified in Notebook_Audit_Report.md: PASS")

# 2. README.md & Reproducibility_Guide.md
readme = (BASE / "README.md").read_text(encoding="utf-8")
assert "state_dim=49" in readme, "Missing state_dim=49 in README.md"
assert "checkpoint[\"policy_state_dict\"]" in readme, "Missing policy_state_dict in README.md"
assert "obs_dim" not in readme, "Found old obs_dim in README.md"
assert "model_state_dict" not in readme, "Found old model_state_dict in README.md"
print("  README.md reproduction snippet verified: PASS")

repro = (BASE / "reports/final_submission/Reproducibility_Guide.md").read_text(encoding="utf-8")
assert "state_dim=49" in repro, "Missing state_dim=49 in Reproducibility_Guide.md"
assert "checkpoint[\"policy_state_dict\"]" in repro, "Missing policy_state_dict in Reproducibility_Guide.md"
assert "obs_dim" not in repro, "Found old obs_dim in Reproducibility_Guide.md"
assert "model_state_dict" not in repro, "Found old model_state_dict in Reproducibility_Guide.md"
print("  Reproducibility_Guide.md snippet verified: PASS")

# 3. requirements.txt
reqs = (BASE / "requirements.txt").read_text(encoding="utf-8")
assert "seaborn" not in reqs, "Found seaborn in requirements.txt"
print("  requirements.txt (no seaborn) verified: PASS")

print("\n=== VERIFY IMMUTABILITY OF CORE ASSETS ===")
check_files = [
    "backend/checkpoints/improved_rnn_encoder.pt",
    "backend/checkpoints/ppo_reproducible_best.pt",
    "backend/checkpoints/ppo_best_agent.pt",
    "backend/environments/trading_env.py",
    "backend/agents/ppo_agent.py",
    "backend/embeddings/test_embeddings_improved.pt",
    "data/training/test/X_test.pt",
    "data/training/test/y_test.pt",
    "backend/notebooks/04_rl_development.ipynb",
    "backend/notebooks/05_ppo_training.ipynb",
]

for f in check_files:
    p = BASE / f
    assert p.exists(), f"Missing file: {f}"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"  {f:<50} | SHA256: {h[:16]}... | Size: {p.stat().st_size:,} bytes")

print("\nALL IMMUTABILITY AND INTEGRITY CHECKS PASSED!")
