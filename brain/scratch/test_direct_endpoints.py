import sys
sys.path.insert(0, "neurotrader-api")
import main
from main import (
    load_resources, get_health, get_stocks, get_market_data,
    get_lstm_prediction, get_lstm_embedding, post_ppo_decision,
    post_run_simulation, get_performance, get_artifacts, PPODecisionRequest, SimulationRequest
)

print("Loading resources...")
load_resources()

print("1. get_health():")
h = get_health()
print("  Status:", h["status"], "LSTM:", h["models"]["lstm_encoder"]["status"])

print("2. get_stocks():")
s = get_stocks()
print("  Stocks count:", s["count"])

print("3. get_market_data():")
md = get_market_data(stock="RELIANCE", split="test", limit=10)
print("  Candles returned:", len(md["candles"]), "Latest:", md["candles"][-1]["date"])

print("4. get_lstm_prediction():")
lp = get_lstm_prediction(stock="RELIANCE", split="test", step=0)
print("  Pred:", lp["prediction"])

print("5. get_lstm_embedding():")
le = get_lstm_embedding(stock="RELIANCE", split="test", step=0)
print("  Embedding dim:", le["embedding_dim"], "Norm:", le["statistics"]["norm"])

print("6. post_ppo_decision():")
req_ppo = PPODecisionRequest(stock="RELIANCE", split="test", step=0, in_position=False, portfolio_value=100000.0, initial_capital=100000.0)
pd_res = post_ppo_decision(req_ppo)
print("  Action:", pd_res["action_name"], "Probs:", pd_res["probabilities"], "Critic V(s):", pd_res["critic_value"])

print("7. post_run_simulation():")
req_sim = SimulationRequest(stock="RELIANCE", split="test", initial_capital=100000.0, transaction_fee=0.001)
sim_res = post_run_simulation(req_sim)
print("  Return%:", sim_res["total_return_pct"], "Max DD%:", sim_res["max_drawdown_pct"], "Trades:", sim_res["total_trades"])

print("8. get_performance():")
perf = get_performance()
print("  Overall Return%:", perf["overall_summary"]["Avg_Return%"])

print("9. get_artifacts():")
art = get_artifacts()
print("  Diagrams:", len(art["diagrams"]), "Visualizations:", len(art["visualizations"]))

print("\n=== ALL DIRECT API ENDPOINTS TESTED SUCCESSFULLY! ===")
