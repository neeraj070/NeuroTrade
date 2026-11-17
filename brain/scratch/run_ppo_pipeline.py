import os
import sys
import time
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

from backend.environments.trading_env import MarketTradingEnv

# Reproducibility seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Ensure directories exist
os.makedirs("backend/checkpoints", exist_ok=True)
os.makedirs("backend/agents", exist_ok=True)
os.makedirs("reports/reinforcement_learning", exist_ok=True)

# 1. Define Actor-Critic Architecture
class ActorCriticTradingNetwork(nn.Module):
    """
    Decoupled Actor-Critic Architecture for 49-D Market States.
    
    Actor:
      Input (49,) -> Linear(49, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 3)
      Outputs logits for Categorical distribution over {0: HOLD, 1: BUY, 2: SELL}.
      
    Critic:
      Input (49,) -> Linear(49, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 1)
      Outputs scalar expected return value V(s).
    """
    def __init__(self, state_dim: int = 49, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        # Policy (Actor) stream
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value (Critic) stream
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Orthogonal Initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Final actor layer uses small gain 0.01 for near-uniform initial exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Final critic layer uses gain 1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def get_action(self, state: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
                
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)
        
        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor):
        return self.critic(state).squeeze(-1)

# 2. Load Datasets and Models
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
emb_val   = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True)
emb_test  = torch.load("backend/embeddings/test_embeddings_improved.pt", weights_only=True)

X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
X_val   = torch.load("data/training/validation/X_val.pt", weights_only=True)
X_test  = torch.load("data/training/test/X_test.pt", weights_only=True)

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val   = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test  = pd.read_csv("data/training/test/metadata_test.csv")

symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in symbols}

# 3. Policy Evaluator Function
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
            "Symbol": sym,
            "Split": split_name,
            "Initial_Val": initial_val,
            "Final_Val": float(final_val),
            "Total_Return%": float(total_ret * 100),
            "Sharpe": sharpe,
            "Max_Drawdown%": float(max_dd * 100),
            "Trades": trades,
            "Costs": float(costs),
            "Steps": len(actions),
            "Values": values,
            "Actions": actions
        })
        
    res_df = pd.DataFrame(results)
    avg_ret = res_df["Total_Return%"].mean()
    avg_sharpe = res_df["Sharpe"].mean()
    avg_max_dd = res_df["Max_Drawdown%"].mean()
    avg_trades = res_df["Trades"].mean()
    
    return {
        "per_stock": results,
        "summary": {
            "Split": split_name,
            "Avg_Return%": float(avg_ret),
            "Avg_Sharpe": float(avg_sharpe),
            "Avg_Max_Drawdown%": float(avg_max_dd),
            "Avg_Trades": float(avg_trades)
        }
    }

# 4. PPO Training Hyperparameters
TOTAL_ITERATIONS = 35
LR_INITIAL = 3e-4
LR_FINAL = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF_INIT = 0.015
ENTROPY_COEF_FINAL = 0.005
PPO_EPOCHS = 4
BATCH_SIZE = 128
MAX_GRAD_NORM = 0.5

policy = ActorCriticTradingNetwork().float()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR_INITIAL, eps=1e-5)

print("=== STARTING PPO AGENT TRAINING ===")
print(f"Iterations: {TOTAL_ITERATIONS} | Batch Size: {BATCH_SIZE} | Epochs per Iter: {PPO_EPOCHS}")
print(f"Discount Gamma: {GAMMA} | GAE Lambda: {GAE_LAMBDA} | Clip Eps: {CLIP_EPS}")

training_history = []
best_val_score = -float("inf")
best_checkpoint_path = "backend/checkpoints/ppo_best_agent.pt"
latest_checkpoint_path = "backend/checkpoints/ppo_trading_agent.pt"

start_train_time = time.time()

for iteration in range(1, TOTAL_ITERATIONS + 1):
    policy.train()
    
    # Anneal learning rate and entropy coefficient linearly
    progress = (iteration - 1) / max(1, TOTAL_ITERATIONS - 1)
    current_lr = LR_INITIAL + progress * (LR_FINAL - LR_INITIAL)
    current_entropy_coef = ENTROPY_COEF_INIT + progress * (ENTROPY_COEF_FINAL - ENTROPY_COEF_INIT)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
        
    # 1. Rollout Collection across all 5 Training Stocks
    b_obs = []
    b_actions = []
    b_logprobs = []
    b_rewards = []
    b_dones = []
    b_values = []
    b_returns = []
    b_advantages = []
    
    iter_trades = 0
    iter_reward_sum = 0.0
    
    for sym in symbols:
        env = MarketTradingEnv(emb_train, X_train, meta_train, raw_prices, symbol=sym)
        obs, info = env.reset()
        
        s_obs = []
        s_actions = []
        s_logprobs = []
        s_rewards = []
        s_dones = []
        s_values = []
        
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
            iter_reward_sum += reward
            obs = next_obs
            
        # GAE per stock (isolated boundary)
        s_adv = np.zeros(len(s_rewards), dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(s_rewards))):
            if t == len(s_rewards) - 1:
                nextnonterminal = 0.0
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0 - s_dones[t]
                nextvalues = s_values[t + 1]
            delta = s_rewards[t] + GAMMA * nextvalues * nextnonterminal - s_values[t]
            s_adv[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        s_ret = s_adv + np.array(s_values)
        
        b_obs.extend(s_obs)
        b_actions.extend(s_actions)
        b_logprobs.extend(s_logprobs)
        b_rewards.extend(s_rewards)
        b_dones.extend(s_dones)
        b_values.extend(s_values)
        b_returns.extend(s_ret.tolist())
        b_advantages.extend(s_adv.tolist())
        
    # Convert buffer to PyTorch tensors
    t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
    t_act = torch.tensor(np.array(b_actions), dtype=torch.long)
    t_old_logp = torch.tensor(np.array(b_logprobs), dtype=torch.float32)
    t_ret = torch.tensor(np.array(b_returns), dtype=torch.float32)
    t_adv = torch.tensor(np.array(b_advantages), dtype=torch.float32)
    
    # Normalize advantages across rollout
    t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
    
    # 2. PPO Optimization Epochs
    dataset_size = len(t_obs)
    indices = np.arange(dataset_size)
    
    clip_fracs = []
    loss_pi_list = []
    loss_vf_list = []
    entropy_list = []
    
    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            b_idx = indices[start:end]
            
            _, new_logp, entropy, new_val = policy.get_action(t_obs[b_idx], t_act[b_idx])
            
            # Policy objective
            ratio = torch.exp(new_logp - t_old_logp[b_idx])
            mb_adv = t_adv[b_idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value objective
            val_loss = 0.5 * ((new_val - t_ret[b_idx]) ** 2).mean()
            
            # Entropy objective
            ent_loss = -current_entropy_coef * entropy.mean()
            
            total_loss = policy_loss + VALUE_COEF * val_loss + ent_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            
            loss_pi_list.append(policy_loss.item())
            loss_vf_list.append(val_loss.item())
            entropy_list.append(entropy.mean().item())
            clip_fracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())
            
    # 3. Periodic Validation Evaluation (Every 2 iterations)
    val_eval = evaluate_policy(policy, emb_val, X_val, meta_val, split_name="Val", deterministic=True)
    val_ret = val_eval["summary"]["Avg_Return%"]
    val_sharpe = val_eval["summary"]["Avg_Sharpe"]
    val_dd = val_eval["summary"]["Avg_Max_Drawdown%"]
    
    # Validation selection criterion: Sharpe ratio (with return tiebreaker)
    val_score = val_sharpe if not np.isnan(val_sharpe) else -10.0
    
    is_best = False
    if val_score > best_val_score:
        best_val_score = val_score
        is_best = True
        torch.save({
            "iteration": iteration,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_summary": val_eval["summary"],
            "config": {"state_dim": 49, "action_dim": 3, "hidden_dim": 64}
        }, best_checkpoint_path)
        
    # Save latest checkpoint
    torch.save({
        "iteration": iteration,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_summary": val_eval["summary"],
        "config": {"state_dim": 49, "action_dim": 3, "hidden_dim": 64}
    }, latest_checkpoint_path)
    
    iter_record = {
        "iteration": iteration,
        "loss_pi": float(np.mean(loss_pi_list)),
        "loss_vf": float(np.mean(loss_vf_list)),
        "entropy": float(np.mean(entropy_list)),
        "clip_frac": float(np.mean(clip_fracs)),
        "train_reward_sum": float(iter_reward_sum),
        "train_trades": iter_trades,
        "val_return%": val_ret,
        "val_sharpe": val_sharpe,
        "val_max_dd%": val_dd,
        "is_best": is_best
    }
    training_history.append(iter_record)
    
    if iteration % 5 == 0 or is_best or iteration == 1:
        star = " ★ BEST" if is_best else ""
        print(f"Iter {iteration:2d}/{TOTAL_ITERATIONS} | PiLoss: {iter_record['loss_pi']:+.4f} | VfLoss: {iter_record['loss_vf']:.4f} | Ent: {iter_record['entropy']:.3f} | Val Ret: {val_ret:+6.2f}% | Val Sharpe: {val_sharpe:+5.2f} | Val DD: {val_dd:5.2f}%{star}")

print(f"\nTraining completed in {time.time() - start_train_time:.1f}s.")
print(f"Best Validation Sharpe: {best_val_score:.4f} (Saved to {best_checkpoint_path})")

# 5. Out-of-Sample Final Backtest on Untouched Test Set
print("\n=== PERFORMING UNTOUCHED TEST SET EVALUATION ===")
best_ckpt = torch.load(best_checkpoint_path, weights_only=True)
best_policy = ActorCriticTradingNetwork(state_dim=49, action_dim=3, hidden_dim=64)
best_policy.load_state_dict(best_ckpt["policy_state_dict"])
best_policy.eval()

test_eval = evaluate_policy(best_policy, emb_test, X_test, meta_test, split_name="Test", deterministic=True)
train_eval = evaluate_policy(best_policy, emb_train, X_train, meta_train, split_name="Train", deterministic=True)
val_eval_best = evaluate_policy(best_policy, emb_val, X_val, meta_val, split_name="Val", deterministic=True)

# 6. Benchmark Comparisons across 3 Baselines
# Baseline benchmarks on Test Set (from verified Phase 10 evaluation)
baselines_test = {
    "Always HOLD": {"Return%": 0.00, "Sharpe": 0.00, "Max_DD%": 0.00, "Trades": 0.0},
    "Buy-and-Hold": {"Return%": -15.74, "Sharpe": -0.75, "Max_DD%": -26.34, "Trades": 1.0},
    "MA Crossover": {"Return%": -9.63, "Sharpe": -1.54, "Max_DD%": -13.48, "Trades": 3.6}
}

ppo_test_summary = test_eval["summary"]
print("\n--- TEST SET BENCHMARK COMPARISON (AVERAGED OVER 5 ASSETS) ---")
print(f"{'Strategy':<16} | {'Total Return%':>14} | {'Sharpe Ratio':>12} | {'Max Drawdown%':>14} | {'Trades':>8}")
print("-" * 72)
print(f"{'Always HOLD':<16} | {0.0:>13.2f}% | {0.0:>12.2f} | {0.0:>13.2f}% | {0:>8}")
print(f"{'Buy-and-Hold':<16} | {-15.74:>13.2f}% | {-0.75:>12.2f} | {-26.34:>13.2f}% | {1.0:>8.1f}")
print(f"{'MA Crossover':<16} | {-9.63:>13.2f}% | {-1.54:>12.2f} | {-13.48:>13.2f}% | {3.6:>8.1f}")
print(f"{'PPO Agent':<16} | {ppo_test_summary['Avg_Return%']:>13.2f}% | {ppo_test_summary['Avg_Sharpe']:>12.2f} | {ppo_test_summary['Avg_Max_Drawdown%']:>13.2f}% | {ppo_test_summary['Avg_Trades']:>8.1f}")

print("\n--- PPO TEST SET PERFORMANCE PER ASSET ---")
test_df = pd.DataFrame(test_eval["per_stock"])[["Symbol", "Final_Val", "Total_Return%", "Sharpe", "Max_Drawdown%", "Trades", "Costs"]]
print(test_df.to_string(index=False))

# 7. Generate Evaluation Figures
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, sym in enumerate(symbols):
    ax = axes[idx]
    # PPO equity curve
    stock_res = [r for r in test_eval["per_stock"] if r["Symbol"] == sym][0]
    vals_ppo = stock_res["Values"]
    ax.plot(range(len(vals_ppo)), np.array(vals_ppo) / 1000.0, label="PPO Agent", color="tab:green", linewidth=2.0)
    
    # Re-run Buy & Hold for this stock
    env_bh = MarketTradingEnv(emb_test, X_test, meta_test, raw_prices, symbol=sym)
    env_bh.reset()
    vals_bh = [100000.0]
    for s in range(len(vals_ppo) - 1):
        act = 1 if s == 0 else 0
        _, _, _, _, info_bh = env_bh.step(act)
        vals_bh.append(info_bh["portfolio_value"])
    ax.plot(range(len(vals_bh)), np.array(vals_bh) / 1000.0, label="Buy & Hold", color="tab:blue", linestyle="--", linewidth=1.5)
    
    # Cash baseline
    ax.axhline(100.0, color="gray", linestyle=":", label="Hold (Cash)")
    ax.set_title(f"{sym} Test Performance", fontsize=11, fontweight="bold")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value (₹k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# 6th Subplot: Training & Validation Curves
ax6 = axes[5]
hist_df = pd.DataFrame(training_history)
ax6.plot(hist_df["iteration"], hist_df["val_return%"], label="Val Return %", color="tab:purple", linewidth=1.8)
ax6.axhline(0, color="black", linestyle="--", alpha=0.5)
ax6.set_title("Validation Return Convergence", fontsize=11, fontweight="bold")
ax6.set_xlabel("PPO Iteration")
ax6.set_ylabel("Return (%)")
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)

plt.tight_layout()
fig_path = "reports/reinforcement_learning/ppo_evaluation_curves.png"
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"\nSaved evaluation curves to {fig_path} [OK]")

# Save training and evaluation results JSON
full_results = {
    "hyperparameters": {
        "iterations": TOTAL_ITERATIONS,
        "lr_initial": LR_INITIAL,
        "lr_final": LR_FINAL,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": CLIP_EPS,
        "value_coef": VALUE_COEF,
        "entropy_coef_initial": ENTROPY_COEF_INIT,
        "entropy_coef_final": ENTROPY_COEF_FINAL,
        "epochs_per_iter": PPO_EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_grad_norm": MAX_GRAD_NORM,
        "state_dim": 49,
        "action_dim": 3,
        "hidden_dim": 64
    },
    "best_iteration": int(best_ckpt["iteration"]),
    "training_history": training_history,
    "train_summary": train_eval["summary"],
    "val_summary": val_eval_best["summary"],
    "test_summary": test_eval["summary"],
    "test_per_stock": [{k: v for k, v in r.items() if k not in ["Values", "Actions"]} for r in test_eval["per_stock"]],
    "baselines_comparison": {
        "Always_HOLD": baselines_test["Always HOLD"],
        "Buy_and_Hold": baselines_test["Buy-and-Hold"],
        "MA_Crossover": baselines_test["MA Crossover"],
        "PPO_Agent": {
            "Return%": ppo_test_summary["Avg_Return%"],
            "Sharpe": ppo_test_summary["Avg_Sharpe"],
            "Max_DD%": ppo_test_summary["Avg_Max_Drawdown%"],
            "Trades": ppo_test_summary["Avg_Trades"]
        }
    }
}

with open("reports/reinforcement_learning/ppo_training_results.json", "w") as f:
    json.dump(full_results, f, indent=2)

print("Saved full numerical results to reports/reinforcement_learning/ppo_training_results.json [OK]")

# Save reusable agent class to backend/agents/ppo_agent.py
agent_code = '''"""
PPO Actor-Critic Trading Agent for NeuroTrade.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class ActorCriticTradingNetwork(nn.Module):
    """
    Decoupled Actor-Critic Architecture for 49-D Market States.
    
    Actor:
      Input (49,) -> Linear(49, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 3)
      Outputs logits for Categorical distribution over {0: HOLD, 1: BUY, 2: SELL}.
      
    Critic:
      Input (49,) -> Linear(49, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 64) -> LayerNorm(64) -> Tanh -> Linear(64, 1)
      Outputs scalar expected return value V(s).
    """
    def __init__(self, state_dim: int = 49, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        # Policy (Actor) stream
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value (Critic) stream
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Orthogonal Initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def get_action(self, state: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
                
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)
        
        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor):
        return self.critic(state).squeeze(-1)
'''

with open("backend/agents/ppo_agent.py", "w") as f:
    f.write(agent_code)

with open("backend/agents/__init__.py", "w") as f:
    f.write('from backend.agents.ppo_agent import ActorCriticTradingNetwork\n__all__ = ["ActorCriticTradingNetwork"]\n')

print("Created backend/agents/ppo_agent.py and backend/agents/__init__.py [OK]")
