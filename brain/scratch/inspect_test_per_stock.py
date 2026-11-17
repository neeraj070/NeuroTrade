import json

with open("reports/reinforcement_learning/ppo_training_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("test_per_stock:", res.get("test_per_stock"))
print("baselines_comparison:", res.get("baselines_comparison"))
