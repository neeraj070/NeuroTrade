import os
import sys
import torch
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from test_env_prototype import MarketTradingEnv

# Load data
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
emb_val   = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True)
emb_test  = torch.load("backend/embeddings/test_embeddings_improved.pt", weights_only=True)

X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
X_val   = torch.load("data/training/validation/X_val.pt", weights_only=True)
X_test  = torch.load("data/training/test/X_test.pt", weights_only=True)

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val   = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test  = pd.read_csv("data/training/test/metadata_test.csv")

raw_prices = {}
for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
    raw_prices[sym] = pd.read_csv(f"data/raw/{sym}.csv")

print("=== 1. GYMNASIUM ENV_CHECKER TEST ===")
env_test = MarketTradingEnv(
    embeddings=emb_train,
    features=X_train,
    metadata_df=meta_train,
    raw_prices_dict=raw_prices,
    symbol="RELIANCE"
)

# Run official Gymnasium env checker
check_env(env_test)
print("gymnasium.utils.env_checker.check_env(env): PASSED! [OK]")

print("\n=== 2. OBSERVATION SPACE AND SHAPE AUDIT ===")
obs, info = env_test.reset()
print(f"Observation shape: {obs.shape}")
assert obs.shape == (49,), f"Expected (49,), got {obs.shape}"
assert not np.isnan(obs).any(), "NaN in observation!"
assert not np.isinf(obs).any(), "Inf in observation!"
print("Observation shape (49,) and finite values: PASSED! [OK]")

print("Components:")
print("  Embedding slice (0:32) norm:", np.linalg.norm(obs[:32]))
print("  Features slice (32:46):", obs[32:46])
print("  Portfolio slice (46:49) [pos, cash_ratio, norm_val]:", obs[46:49])
assert obs[46] == 0.0, "Initial position ratio should be 0.0"
assert obs[47] == 1.0, "Initial cash ratio should be 1.0"
assert obs[48] == 1.0, "Initial normalized portfolio value should be 1.0"

print("\n=== 3. NUMERICAL LOGIC & PORTFOLIO ACCOUNTING TESTS ===")
# Test 1: HOLD on step 0
obs1, r1, term1, trunc1, info1 = env_test.step(0)
assert info1["action_taken"] == 0
assert info1["trade_executed"] == False
assert info1["transaction_cost"] == 0.0
assert info1["portfolio_value"] == 100000.0
assert r1 == 0.0
print("  Test 3.1: HOLD action accounting [PASSED]")

# Test 2: Selling without shares
obs2, r2, term2, trunc2, info2 = env_test.step(2)
assert info2["trade_executed"] == False
assert info2["transaction_cost"] == 0.0
assert info2["portfolio_value"] == 100000.0
assert r2 == 0.0
print("  Test 3.2: Selling without shares rejected safely [PASSED]")

# Test 3: BUY action
open_p = env_test.open_t1_arr[env_test.current_step]
close_p = env_test.close_t1_arr[env_test.current_step]
expected_shares = int(100000.0 // (open_p * 1.001))
expected_trade_val = expected_shares * open_p
expected_cost = expected_trade_val * 0.001
expected_cash = 100000.0 - expected_trade_val - expected_cost
expected_val = expected_cash + expected_shares * close_p
expected_reward = float(np.log(expected_val / 100000.0))

obs3, r3, term3, trunc3, info3 = env_test.step(1)
assert info3["trade_executed"] == True
assert info3["trade_shares"] == expected_shares
assert abs(info3["transaction_cost"] - expected_cost) < 1e-4
assert abs(info3["cash"] - expected_cash) < 1e-4
assert abs(info3["portfolio_value"] - expected_val) < 1e-4
assert abs(r3 - expected_reward) < 1e-6
print(f"  Test 3.3: BUY execution: {expected_shares} shares @ Open ₹{open_p:.2f}, Fee ₹{expected_cost:.2f}, Val ₹{expected_val:.2f}, Reward {r3:.6f} [PASSED]")

# Test 4: Insufficient cash repeated BUY
obs4, r4, term4, trunc4, info4 = env_test.step(1)
assert info4["trade_executed"] == False
assert info4["transaction_cost"] == 0.0
print("  Test 3.4: Repeated BUY with insufficient cash rejected with 0 fee [PASSED]")

# Test 5: SELL action
held_shares = env_test.shares
open_p_sell = env_test.open_t1_arr[env_test.current_step]
close_p_prev = env_test.close_t_arr[env_test.current_step]
prev_v = env_test.cash + held_shares * close_p_prev
sell_val = held_shares * open_p_sell
sell_fee = sell_val * 0.001
new_cash = env_test.cash + sell_val - sell_fee
expected_sell_reward = float(np.log(new_cash / prev_v))

obs5, r5, term5, trunc5, info5 = env_test.step(2)
assert info5["trade_executed"] == True
assert info5["trade_shares"] == held_shares
assert env_test.shares == 0
assert abs(env_test.cash - new_cash) < 1e-4
assert abs(r5 - expected_sell_reward) < 1e-6
print(f"  Test 3.5: SELL execution: {held_shares} shares @ Open ₹{open_p_sell:.2f}, Fee ₹{sell_fee:.2f}, New Cash ₹{new_cash:.2f}, Reward {r5:.6f} [PASSED]")

print("\n=== 4. BASELINE EVALUATION ACROSS PARTITIONS ===")
def run_baseline(env, strategy="hold", proc_df=None):
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    initial_val = env.initial_capital
    rewards = []
    values = [initial_val]
    actions = []
    trades = 0
    
    step_num = 0
    while not (terminated or truncated):
        if strategy == "hold":
            act = 0
        elif strategy == "buy_and_hold":
            act = 1 if step_num == 0 else 0
        elif strategy == "ma_crossover":
            # SMA 10 vs SMA 50
            d = env.end_dates[env.current_step]
            row = proc_df[proc_df["Date"] == d]
            if len(row) > 0 and not pd.isna(row["SMA_10"].values[0]) and not pd.isna(row["SMA_50"].values[0]):
                sma10 = row["SMA_10"].values[0]
                sma50 = row["SMA_50"].values[0]
                if sma10 > sma50:
                    act = 1 if env.shares == 0 else 0
                else:
                    act = 2 if env.shares > 0 else 0
            else:
                act = 0
        else:
            act = 0
            
        obs, reward, terminated, truncated, info = env.step(act)
        rewards.append(reward)
        values.append(info["portfolio_value"])
        actions.append(act)
        if info["trade_executed"]:
            trades += 1
        step_num += 1
        
    final_val = values[-1]
    total_ret = (final_val - initial_val) / initial_val
    ann_factor = 252.0 / len(rewards)
    # Daily returns
    v_arr = np.array(values)
    d_rets = np.diff(v_arr) / v_arr[:-1]
    mean_d = np.mean(d_rets)
    std_d = np.std(d_rets) + 1e-9
    sharpe = float((mean_d / std_d) * np.sqrt(252.0))
    # Max drawdown
    peaks = np.maximum.accumulate(v_arr)
    drawdowns = (v_arr - peaks) / peaks
    max_dd = float(np.min(drawdowns))
    
    return {
        "Strategy": strategy,
        "Initial_Val": initial_val,
        "Final_Val": float(final_val),
        "Total_Return%": float(total_ret * 100),
        "Sharpe": sharpe,
        "Max_Drawdown%": float(max_dd * 100),
        "Trades": trades,
        "Steps": len(rewards)
    }

results = []
for split_name, emb, feats, meta in [
    ("Train", emb_train, X_train, meta_train),
    ("Val",   emb_val,   X_val,   meta_val),
    ("Test",  emb_test,  X_test,  meta_test)
]:
    for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
        proc_df = pd.read_csv(f"data/processed/{sym}_processed.csv")
        env = MarketTradingEnv(emb, feats, meta, raw_prices, symbol=sym)
        for strat in ["hold", "buy_and_hold", "ma_crossover"]:
            res = run_baseline(env, strategy=strat, proc_df=proc_df)
            res["Split"] = split_name
            res["Symbol"] = sym
            results.append(res)

res_df = pd.DataFrame(results)
print("\nBaseline Results Summary (Averaged across 5 Symbols per Split):")
avg_summary = res_df.groupby(["Split", "Strategy"])[["Total_Return%", "Sharpe", "Max_Drawdown%", "Trades"]].mean()
print(avg_summary.round(2))

# Test set specific table
print("\nTest Set Detailed Performance per Stock:")
test_table = res_df[res_df["Split"] == "Test"][["Symbol", "Strategy", "Final_Val", "Total_Return%", "Sharpe", "Max_Drawdown%", "Trades"]]
print(test_table.to_string(index=False))

