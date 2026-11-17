import json

with open("reports/reinforcement_learning/ppo_training_results.json") as f:
    d = json.load(f)

print("=== Test Summary in ppo_training_results.json ===")
print(json.dumps(d['test_summary'], indent=2))
print("\n=== Baselines Comparison in ppo_training_results.json ===")
print(json.dumps(d['baselines_comparison'], indent=2))
print("\n=== Test Per Stock in ppo_training_results.json ===")
for s in d['test_per_stock']:
    sym = s['Symbol']
    ret = s['Total_Return%']
    shp = s['Sharpe']
    mdd = s['Max_Drawdown%']
    trd = s['Trades']
    cst = s['Costs']
    print(f"{sym:<10} | Return: {ret:+.2f}% | Sharpe: {shp:+.4f} | MaxDD: {mdd:.2f}% | Trades: {trd} | Cost: Rs.{cst:.2f}")
