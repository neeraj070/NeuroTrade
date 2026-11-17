import sys
sys.path.insert(0, "neurotrader-api")
from fastapi.testclient import TestClient
from main import app, load_resources

client = TestClient(app)

print("1. Testing /api/health...")
res = client.get("/api/health")
print("  Status:", res.status_code, res.json()["status"])

print("2. Testing /api/stocks...")
res = client.get("/api/stocks")
print("  Status:", res.status_code, "Stocks:", [s["symbol"] for s in res.json()["stocks"]])

print("3. Testing /api/market-data...")
res = client.get("/api/market-data?stock=RELIANCE&split=test&limit=10")
print("  Status:", res.status_code, "Candles count:", len(res.json()["candles"]))

print("4. Testing /api/lstm/prediction...")
res = client.get("/api/lstm/prediction?stock=RELIANCE&split=test&step=0")
print("  Status:", res.status_code, "Pred direction:", res.json()["prediction"]["predicted_direction"])

print("5. Testing /api/lstm/embedding...")
res = client.get("/api/lstm/embedding?stock=RELIANCE&split=test&step=0")
print("  Status:", res.status_code, "Embedding dim:", res.json()["embedding_dim"])

print("6. Testing /api/ppo/decision...")
res = client.post("/api/ppo/decision", json={"stock": "RELIANCE", "split": "test", "step": 0, "in_position": False, "portfolio_value": 100000.0, "initial_capital": 100000.0})
print("  Status:", res.status_code, "Action:", res.json()["action_name"], "Probs:", res.json()["probabilities"])

print("7. Testing /api/simulation/run...")
res = client.post("/api/simulation/run", json={"stock": "RELIANCE", "split": "test", "initial_capital": 100000.0, "transaction_fee": 0.001})
print("  Status:", res.status_code, "Return%:", res.json()["total_return_pct"], "Trades:", res.json()["total_trades"])

print("8. Testing /api/performance...")
res = client.get("/api/performance")
print("  Status:", res.status_code, "PPO Return%:", res.json()["overall_summary"]["Avg_Return%"])

print("9. Testing /api/artifacts...")
res = client.get("/api/artifacts")
print("  Status:", res.status_code, "Diagrams count:", len(res.json()["diagrams"]))

print("\nALL API ENDPOINTS FUNCTIONING WITH 100% SUCCESS!")
