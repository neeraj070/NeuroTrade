import json

with open("reports/reinforcement_learning/ppo_training_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

hist = res["training_history"]
print(f"Total iterations in json: {len(hist)}")
for h in hist:
    is_best_mark = " ★ BEST" if h.get("is_best") else ""
    print(f"Iter {h['iteration']:2d} | Val Ret: {h['val_return%']:+6.2f}% | Val Sharpe: {h['val_sharpe']:+6.4f} | Val DD: {h['val_max_dd%']:6.2f}% | Train Trades: {h['train_trades']:4d}{is_best_mark}")
