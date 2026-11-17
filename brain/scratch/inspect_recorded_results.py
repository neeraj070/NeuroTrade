import json

with open("reports/reinforcement_learning/ppo_training_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("Keys in ppo_training_results.json:", list(res.keys()))
print("Best iteration recorded:", res.get("best_iteration"))
print("Best val summary recorded:", res.get("best_val_summary"))
print("Test summary recorded:", res.get("test_summary"))
print("Per asset test results recorded:")
for sym, r in res.get("test_results_per_asset", {}).items():
    print(f"  {sym}: Return={r['Total_Return%']:.2f}%, Sharpe={r['Sharpe']:.2f}, DD={r['Max_Drawdown%']:.2f}%, Trades={r['Trades']}, Costs={r['Costs']:.2f}")

print("\nBaseline comparison recorded:")
for b, r in res.get("baseline_comparison", {}).items():
    print(f"  {b}: {r}")
