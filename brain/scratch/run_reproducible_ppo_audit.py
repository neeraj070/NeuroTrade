import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.ppo_agent import ActorCriticTradingNetwork, set_seed
from backend.environments.trading_env import MarketTradingEnv

print("=== PPO REPRODUCIBILITY AUDIT: CONTROLLED DUAL EXECUTION ===")

# 1. Load Data
symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
emb_val   = torch.load("backend/embeddings/val_embeddings_improved.pt", weights_only=True)
emb_test  = torch.load("backend/embeddings/test_embeddings_improved.pt", weights_only=True)

X_train   = torch.load("data/training/train/X_train.pt", weights_only=True)
X_val     = torch.load("data/training/validation/X_val.pt", weights_only=True)
X_test    = torch.load("data/training/test/X_test.pt", weights_only=True)

meta_train = pd.read_csv("data/training/train/metadata_train.csv")
meta_val   = pd.read_csv("data/training/validation/metadata_val.csv")
meta_test  = pd.read_csv("data/training/test/metadata_test.csv")

raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in symbols}

def evaluate_policy(policy, emb_data, feat_data, meta_data, split_name="Val", deterministic=True):
    policy.eval()
    results = []
    all_actions = {}
    
    for sym in symbols:
        env = MarketTradingEnv(emb_data, feat_data, meta_data, raw_prices, symbol=sym)
        obs, info = env.reset(seed=42)
        
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
                
        all_actions[sym] = actions
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
    return {
        "per_stock": results,
        "actions_by_symbol": all_actions,
        "summary": {
            "Split": split_name,
            "Avg_Return%": float(res_df["Total_Return%"].mean()),
            "Avg_Sharpe": float(res_df["Sharpe"].mean()),
            "Avg_Max_Drawdown%": float(res_df["Max_Drawdown%"].mean()),
            "Avg_Trades": float(res_df["Trades"].mean()),
            "Total_Costs": float(res_df["Costs"].sum()),
            "Avg_Costs": float(res_df["Costs"].mean())
        }
    }

def run_single_ppo_pipeline(run_id: int, master_seed: int = 42):
    print(f"\n========================================================")
    print(f"=== STARTING REPRODUCIBLE PPO PIPELINE: RUN {run_id} ===")
    print(f"========================================================")
    
    # 1. Centralized Seed Setting immediately prior to model instantiation
    set_seed(master_seed)
    
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
    
    policy = ActorCriticTradingNetwork().float()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR_INITIAL, eps=1e-5)
    
    # Record initial weights for bitwise identity checking
    initial_weights = {k: v.clone().cpu() for k, v in policy.state_dict().items()}
    initial_norm = sum(p.norm().item() for p in policy.parameters())
    print(f"Run {run_id} Initial Weight Norm: {initial_norm:.6f}")
    
    training_history = []
    best_active_score = -float("inf")
    best_active_checkpoint = None
    best_active_iteration = None
    best_val_summary = None
    
    start_time = time.time()
    
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        policy.train()
        
        # Anneal LR and entropy coefficient
        progress = (iteration - 1) / max(1, TOTAL_ITERATIONS - 1)
        current_lr = LR_INITIAL + progress * (LR_FINAL - LR_INITIAL)
        current_entropy_coef = ENTROPY_COEF_INIT + progress * (ENTROPY_COEF_FINAL - ENTROPY_COEF_INIT)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
            
        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, b_returns, b_advantages = [], [], [], [], [], [], [], []
        iter_trades = 0
        iter_reward_sum = 0.0
        
        for sym_idx, sym in enumerate(symbols):
            env = MarketTradingEnv(emb_train, X_train, meta_train, raw_prices, symbol=sym)
            obs, info = env.reset(seed=master_seed + sym_idx)
            
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
                iter_reward_sum += reward
                obs = next_obs
                
            # Boundary-isolated GAE per stock
            s_adv = np.zeros(len(s_rewards), dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(len(s_rewards))):
                nextnonterminal = 0.0 if t == len(s_rewards) - 1 else 1.0 - s_dones[t]
                nextvalues = 0.0 if t == len(s_rewards) - 1 else s_values[t + 1]
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
            
        t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
        t_act = torch.tensor(np.array(b_actions), dtype=torch.long)
        t_old_logp = torch.tensor(np.array(b_logprobs), dtype=torch.float32)
        t_ret = torch.tensor(np.array(b_returns), dtype=torch.float32)
        t_adv = torch.tensor(np.array(b_advantages), dtype=torch.float32)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
        
        dataset_size = len(t_obs)
        indices = np.arange(dataset_size)
        clip_fracs = []
        loss_pi_list = []
        loss_vf_list = []
        entropy_list = []
        
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                batch_indices = indices[start:start + BATCH_SIZE]
                mb_obs = t_obs[batch_indices]
                mb_actions = t_act[batch_indices]
                mb_old_logprobs = t_old_logp[batch_indices]
                mb_returns = t_ret[batch_indices]
                mb_advantages = t_adv[batch_indices]
                
                _, new_logprobs, entropy, new_values = policy.get_action(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_vf = 0.5 * ((new_values - mb_returns) ** 2).mean()
                loss_entropy = entropy.mean()
                
                total_loss = loss_pi + VALUE_COEF * loss_vf - current_entropy_coef * loss_entropy
                
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
                clip_fracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())
                loss_pi_list.append(loss_pi.item())
                loss_vf_list.append(loss_vf.item())
                entropy_list.append(loss_entropy.item())
                
        # Validation Evaluation
        val_eval = evaluate_policy(policy, emb_val, X_val, meta_val, split_name="Val", deterministic=True)
        val_ret = val_eval["summary"]["Avg_Return%"]
        val_sharpe = val_eval["summary"]["Avg_Sharpe"]
        val_dd = val_eval["summary"]["Avg_Max_Drawdown%"]
        val_trades = val_eval["summary"]["Avg_Trades"]
        
        # Refined Model Selection Rule:
        # Exclude zero-trade policies (avg_trades < 1.0). Rank eligible active policies by validation Sharpe.
        is_active_eligible = val_trades >= 1.0
        is_best = False
        
        if is_active_eligible:
            score = val_sharpe if not np.isnan(val_sharpe) else -10.0
            if score > best_active_score:
                best_active_score = score
                best_active_iteration = iteration
                best_val_summary = val_eval["summary"]
                best_active_checkpoint = {k: v.clone().cpu() for k, v in policy.state_dict().items()}
                is_best = True
                
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
            "val_trades": val_trades,
            "is_active_eligible": is_active_eligible,
            "is_best": is_best
        }
        training_history.append(iter_record)
        
        if iteration % 5 == 0 or is_best or iteration == 1:
            star = " ★ BEST" if is_best else ""
            elig = "Eligible" if is_active_eligible else "Ineligible (0 Trades)"
            print(f"Iter {iteration:2d}/{TOTAL_ITERATIONS} | PiLoss: {iter_record['loss_pi']:+.4f} | VfLoss: {iter_record['loss_vf']:.4f} | Ent: {iter_record['entropy']:.3f} | Val Ret: {val_ret:+6.2f}% | Val Sharpe: {val_sharpe:+5.2f} | Trades: {val_trades:4.1f} [{elig}]{star}")
            
    elapsed = time.time() - start_time
    print(f"\nRun {run_id} Training Finished in {elapsed:.1f}s.")
    print(f"Best Active Checkpoint selected at Iteration {best_active_iteration} (Val Sharpe: {best_active_score:+.4f})")
    
    assert best_active_checkpoint is not None, "No active policy qualified!"
    
    # Save Run Checkpoint
    ckpt_save_path = f"backend/checkpoints/ppo_reproducible_run{run_id}.pt"
    torch.save({
        "run_id": run_id,
        "iteration": best_active_iteration,
        "policy_state_dict": best_active_checkpoint,
        "val_summary": best_val_summary,
        "config": {"state_dim": 49, "action_dim": 3, "hidden_dim": 64, "master_seed": master_seed}
    }, ckpt_save_path)
    print(f"Saved Run {run_id} checkpoint to {ckpt_save_path}")
    
    # Evaluate best active checkpoint on Untouched Test Set
    best_policy = ActorCriticTradingNetwork().float()
    best_policy.load_state_dict(best_active_checkpoint)
    best_policy.eval()
    
    test_eval = evaluate_policy(best_policy, emb_test, X_test, meta_test, split_name="Test", deterministic=True)
    test_summary = test_eval["summary"]
    
    print(f"\n--- Run {run_id} Out-of-Sample Test Summary ---")
    print(f"  Return: {test_summary['Avg_Return%']:+.2f}% | Sharpe: {test_summary['Avg_Sharpe']:+.2f} | Max DD: {test_summary['Avg_Max_Drawdown%']:.2f}% | Trades: {test_summary['Avg_Trades']:.1f}")
    
    return {
        "run_id": run_id,
        "initial_weights": initial_weights,
        "initial_norm": initial_norm,
        "best_iteration": best_active_iteration,
        "best_val_score": best_active_score,
        "best_val_summary": best_val_summary,
        "training_history": training_history,
        "test_eval": test_eval,
        "best_state_dict": best_active_checkpoint
    }

# Execute Dual Controlled Runs
res_run1 = run_single_ppo_pipeline(run_id=1, master_seed=42)
res_run2 = run_single_ppo_pipeline(run_id=2, master_seed=42)

print("\n========================================================")
print("=== COMPARING RUN 1 vs RUN 2 REPRODUCIBILITY ===")
print("========================================================")

# 1. Compare Initial Weights
initial_weight_diffs = []
for k in res_run1["initial_weights"]:
    d = (res_run1["initial_weights"][k] - res_run2["initial_weights"][k]).abs().max().item()
    initial_weight_diffs.append(d)
max_init_diff = max(initial_weight_diffs)
print(f"1. Initial Weight Max Absolute Difference: {max_init_diff:.8f}")

# 2. Compare Iteration Losses & Metrics
hist1 = res_run1["training_history"]
hist2 = res_run2["training_history"]

loss_pi_diffs = [abs(h1["loss_pi"] - h2["loss_pi"]) for h1, h2 in zip(hist1, hist2)]
loss_vf_diffs = [abs(h1["loss_vf"] - h2["loss_vf"]) for h1, h2 in zip(hist1, hist2)]
entropy_diffs = [abs(h1["entropy"] - h2["entropy"]) for h1, h2 in zip(hist1, hist2)]
val_sharpe_diffs = [abs(h1["val_sharpe"] - h2["val_sharpe"]) for h1, h2 in zip(hist1, hist2)]
val_return_diffs = [abs(h1["val_return%"] - h2["val_return%"]) for h1, h2 in zip(hist1, hist2)]
val_trades_diffs = [abs(h1["val_trades"] - h2["val_trades"]) for h1, h2 in zip(hist1, hist2)]

print(f"2. Policy Loss Max Diff Across 35 Iters  : {max(loss_pi_diffs):.8f}")
print(f"3. Value Loss Max Diff Across 35 Iters   : {max(loss_vf_diffs):.8f}")
print(f"4. Entropy Max Diff Across 35 Iters      : {max(entropy_diffs):.8f}")
print(f"5. Val Sharpe Max Diff Across 35 Iters   : {max(val_sharpe_diffs):.8f}")
print(f"6. Val Return% Max Diff Across 35 Iters  : {max(val_return_diffs):.8f}")
print(f"7. Val Trades Max Diff Across 35 Iters   : {max(val_trades_diffs):.8f}")

# 3. Compare Best Selected Iteration
print(f"\nRun 1 Selected Iteration: {res_run1['best_iteration']} (Val Sharpe: {res_run1['best_val_score']:+.4f})")
print(f"Run 2 Selected Iteration: {res_run2['best_iteration']} (Val Sharpe: {res_run2['best_val_score']:+.4f})")
same_iteration = res_run1["best_iteration"] == res_run2["best_iteration"]
print(f"Selected Iteration Identical: {same_iteration}")

# 4. Compare Test Evaluation Metrics
test1 = res_run1["test_eval"]["summary"]
test2 = res_run2["test_eval"]["summary"]
print(f"\nRun 1 Test: Return={test1['Avg_Return%']:+.4f}%, Sharpe={test1['Avg_Sharpe']:+.4f}, MaxDD={test1['Avg_Max_Drawdown%']:+.4f}%, Trades={test1['Avg_Trades']:.1f}")
print(f"Run 2 Test: Return={test2['Avg_Return%']:+.4f}%, Sharpe={test2['Avg_Sharpe']:+.4f}, MaxDD={test2['Avg_Max_Drawdown%']:+.4f}%, Trades={test2['Avg_Trades']:.1f}")

test_ret_diff = abs(test1['Avg_Return%'] - test2['Avg_Return%'])
test_sharpe_diff = abs(test1['Avg_Sharpe'] - test2['Avg_Sharpe'])
test_dd_diff = abs(test1['Avg_Max_Drawdown%'] - test2['Avg_Max_Drawdown%'])
test_trades_diff = abs(test1['Avg_Trades'] - test2['Avg_Trades'])

# 5. Compare Bitwise Action Sequences
actions1 = res_run1["test_eval"]["actions_by_symbol"]
actions2 = res_run2["test_eval"]["actions_by_symbol"]
total_actions = sum(len(a) for a in actions1.values())
identical_actions = sum(sum(1 for a1, a2 in zip(actions1[s], actions2[s]) if a1 == a2) for s in symbols)
action_match_pct = (identical_actions / total_actions) * 100.0

print(f"\nAction Sequence Bitwise Agreement: {action_match_pct:.2f}% ({identical_actions}/{total_actions} identical action steps)")

# Assert Determinism / Reproducibility
is_reproducible = (max_init_diff == 0.0 and same_iteration and test_ret_diff == 0.0 and action_match_pct == 100.0)
print(f"\n>>> OVERALL REPRODUCIBILITY VERDICT: {'100% BITWISE REPRODUCIBLE [PASSED]' if is_reproducible else 'DIVERGENCE DETECTED [FAILED]'} <<<")

# Save primary reproducible checkpoint
torch.save({
    "iteration": res_run1["best_iteration"],
    "policy_state_dict": res_run1["best_state_dict"],
    "val_summary": res_run1["best_val_summary"],
    "test_summary": test1,
    "config": {"state_dim": 49, "action_dim": 3, "hidden_dim": 64, "master_seed": 42}
}, "backend/checkpoints/ppo_reproducible_best.pt")

# Save results JSON
results_payload = {
    "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "master_seed": 42,
    "reproducibility_status": "PASSED" if is_reproducible else "FAILED",
    "metrics_comparison": {
        "initial_weights_max_diff": max_init_diff,
        "policy_loss_max_diff": max(loss_pi_diffs),
        "value_loss_max_diff": max(loss_vf_diffs),
        "val_sharpe_max_diff": max(val_sharpe_diffs),
        "val_return_max_diff": max(val_return_diffs),
        "test_return_diff": test_ret_diff,
        "test_sharpe_diff": test_sharpe_diff,
        "action_match_pct": action_match_pct
    },
    "run1": {
        "best_iteration": res_run1["best_iteration"],
        "val_summary": res_run1["best_val_summary"],
        "test_summary": test1,
        "test_per_stock": res_run1["test_eval"]["per_stock"]
    },
    "run2": {
        "best_iteration": res_run2["best_iteration"],
        "val_summary": res_run2["best_val_summary"],
        "test_summary": test2,
        "test_per_stock": res_run2["test_eval"]["per_stock"]
    }
}

with open("reports/reinforcement_learning/ppo_reproducibility_results.json", "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)
print("Saved reports/reinforcement_learning/ppo_reproducibility_results.json successfully!")

# Generate Comparison Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Validation Sharpe Trajectory
ax1 = axes[0]
val_sharpes_1 = [h["val_sharpe"] for h in hist1]
val_sharpes_2 = [h["val_sharpe"] for h in hist2]
iters = [h["iteration"] for h in hist1]
ax1.plot(iters, val_sharpes_1, label="Run 1 Val Sharpe", color="tab:blue", linewidth=2.5)
ax1.plot(iters, val_sharpes_2, label="Run 2 Val Sharpe", color="tab:orange", linestyle="--", linewidth=2.0)
ax1.axhline(0, color="gray", linestyle=":", alpha=0.7)
ax1.set_title("Validation Sharpe Trajectory Across Dual Controlled Runs", fontsize=11, fontweight="bold")
ax1.set_xlabel("PPO Iteration")
ax1.set_ylabel("Sharpe Ratio")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Per-Stock Out-of-Sample Return Comparison
ax2 = axes[1]
stocks = [s["Symbol"] for s in res_run1["test_eval"]["per_stock"]]
rets1 = [s["Total_Return%"] for s in res_run1["test_eval"]["per_stock"]]
rets2 = [s["Total_Return%"] for s in res_run2["test_eval"]["per_stock"]]
x = np.arange(len(stocks))
width = 0.35
ax2.bar(x - width/2, rets1, width, label="Run 1 Test Return %", color="tab:blue")
ax2.bar(x + width/2, rets2, width, label="Run 2 Test Return %", color="tab:orange", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(stocks)
ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
ax2.set_title("Out-of-Sample Test Return by Asset (Dual Controlled Runs)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Total Return %")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("reports/reinforcement_learning/ppo_reproducibility_curves.png", dpi=150)
plt.close(fig)
print("Saved reports/reinforcement_learning/ppo_reproducibility_curves.png successfully!")
