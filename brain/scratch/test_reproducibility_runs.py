import os
import sys
import random
import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.ppo_agent import ActorCriticTradingNetwork
from backend.environments.trading_env import MarketTradingEnv

# Centralized seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load data
symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
emb_val   = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True)
X_train   = torch.load("data/training/train/X_train.pt", weights_only=True)
X_val     = torch.load("data/training/validation/X_val.pt", weights_only=True)
meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val   = pd.read_csv("data/training/validation/metadata_val.csv")
raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in symbols}

def evaluate_policy(policy, emb_data, feat_data, meta_data, split_name="Val", deterministic=True):
    policy.eval()
    results = []
    for sym in symbols:
        env = MarketTradingEnv(emb_data, feat_data, meta_data, raw_prices, symbol=sym)
        obs, info = env.reset()
        initial_val = env.initial_capital
        values = [initial_val]
        actions = []
        trades = 0
        costs = 0.0
        terminated = False
        while not terminated:
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                act, _, _, _ = policy.get_action(t_obs, deterministic=deterministic)
                action = act.item()
            obs, reward, terminated, truncated, info = env.step(action)
            values.append(info["portfolio_value"])
            actions.append(action)
            if info["trade_executed"]:
                trades += 1
                costs += info["transaction_cost"]
        final_val = values[-1]
        total_ret = (final_val - initial_val) / initial_val
        v_arr = np.array(values)
        d_rets = np.diff(v_arr) / v_arr[:-1]
        std_d = np.std(d_rets) + 1e-9
        sharpe = float((np.mean(d_rets) / std_d) * np.sqrt(252.0))
        peaks = np.maximum.accumulate(v_arr)
        drawdowns = (v_arr - peaks) / peaks
        max_dd = float(np.min(drawdowns))
        results.append({
            "Symbol": sym, "Split": split_name, "Initial_Val": initial_val,
            "Final_Val": float(final_val), "Total_Return%": float(total_ret * 100),
            "Sharpe": sharpe, "Max_Drawdown%": float(max_dd * 100),
            "Trades": trades, "Costs": float(costs)
        })
    res_df = pd.DataFrame(results)
    return {
        "summary": {
            "Avg_Return%": float(res_df["Total_Return%"].mean()),
            "Avg_Sharpe": float(res_df["Sharpe"].mean()),
            "Avg_Max_Drawdown%": float(res_df["Max_Drawdown%"].mean()),
            "Avg_Trades": float(res_df["Trades"].mean())
        }
    }

def run_test_iterations(n_iters=5, reseed_before_init=True):
    set_seed(42)
    if not reseed_before_init:
        # Simulate notebook offset:
        dummy = ActorCriticTradingNetwork().float()
        _ = dummy.get_action(torch.randn(2, 49))
    else:
        set_seed(42)
        
    policy = ActorCriticTradingNetwork().float()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    VALUE_COEF = 0.5
    BATCH_SIZE = 128
    PPO_EPOCHS = 4
    
    records = []
    best_score = -float("inf")
    
    for iteration in range(1, n_iters + 1):
        policy.train()
        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, b_returns, b_advantages = [], [], [], [], [], [], [], []
        iter_trades = 0
        
        for sym in symbols:
            env = MarketTradingEnv(emb_train, X_train, meta_train, raw_prices, symbol=sym)
            obs, info = env.reset()
            s_obs, s_actions, s_logprobs, s_rewards, s_dones, s_values = [], [], [], [], [], []
            terminated = False
            while not terminated:
                with torch.no_grad():
                    t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act, logp, _, val = policy.get_action(t_obs, deterministic=False)
                action = act.item()
                next_obs, reward, terminated, truncated, info = env.step(action)
                s_obs.append(obs)
                s_actions.append(action)
                s_logprobs.append(logp.item())
                s_rewards.append(reward)
                s_dones.append(float(terminated))
                s_values.append(val.item())
                if info["trade_executed"]:
                    iter_trades += 1
                obs = next_obs
                
            s_adv = np.zeros(len(s_rewards), dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(len(s_rewards))):
                nextnonterminal = 0.0 if t == len(s_rewards) - 1 else 1.0 - s_dones[t]
                nextvalues = 0.0 if t == len(s_rewards) - 1 else s_values[t + 1]
                delta = s_rewards[t] + GAMMA * nextvalues * nextnonterminal - s_values[t]
                s_adv[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            s_ret = s_adv + np.array(s_values)
            
            b_obs.extend(s_obs); b_actions.extend(s_actions); b_logprobs.extend(s_logprobs)
            b_rewards.extend(s_rewards); b_dones.extend(s_dones); b_values.extend(s_values)
            b_returns.extend(s_ret.tolist()); b_advantages.extend(s_adv.tolist())
            
        t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
        t_act = torch.tensor(np.array(b_actions), dtype=torch.long)
        t_old_logp = torch.tensor(np.array(b_logprobs), dtype=torch.float32)
        t_ret = torch.tensor(np.array(b_returns), dtype=torch.float32)
        t_adv = torch.tensor(np.array(b_advantages), dtype=torch.float32)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
        
        indices = np.arange(len(t_obs))
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(t_obs), BATCH_SIZE):
                b_idx = indices[start:start+BATCH_SIZE]
                _, new_logp, entropy, new_val = policy.get_action(t_obs[b_idx], t_act[b_idx])
                ratio = torch.exp(new_logp - t_old_logp[b_idx])
                surr1 = ratio * t_adv[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * t_adv[b_idx]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_vf = 0.5 * ((new_val - t_ret[b_idx]) ** 2).mean()
                loss = loss_pi + VALUE_COEF * loss_vf - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
        val_eval = evaluate_policy(policy, emb_val, X_val, meta_val, split_name="Val", deterministic=True)
        val_s = val_eval["summary"]["Avg_Sharpe"]
        val_r = val_eval["summary"]["Avg_Return%"]
        val_trades = val_eval["summary"]["Avg_Trades"]
        is_best = val_s > best_score
        if is_best: best_score = val_s
        star = " ★ BEST" if is_best else ""
        print(f"Iter {iteration} | Val Ret: {val_r:+.2f}% | Val Sharpe: {val_s:+.4f} | Val Trades: {val_trades:.1f}{star}")
        records.append((val_r, val_s, val_trades, is_best))
    return records

print("=== RUN 1: Clean Seed 42 directly before policy (Standalone replication) ===")
rec1 = run_test_iterations(5, reseed_before_init=True)

print("\n=== RUN 2: Re-run Run 1 to verify 100% exact reproducibility ===")
rec2 = run_test_iterations(5, reseed_before_init=True)

diffs = [abs(r1[1] - r2[1]) for r1, r2 in zip(rec1, rec2)]
print(f"\nMax Sharpe difference between Run 1 and Run 2: {max(diffs):.8f}")
assert max(diffs) == 0.0, "Runs must be 100% identical!"
print("CONTROLLED RUNS DEMONSTRATE 100% BITWISE IDENTICAL RESULTS!")

print("\n=== RUN 3: Old Notebook Offset (without reseed before policy) ===")
rec3 = run_test_iterations(5, reseed_before_init=False)
