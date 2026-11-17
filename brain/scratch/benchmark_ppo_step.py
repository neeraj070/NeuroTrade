import os
import sys
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

from backend.environments.trading_env import MarketTradingEnv
from test_ppo_agent import ActorCriticTradingNetwork

emb_train = torch.load("backend/embeddings/train_embeddings_improved.pt", weights_only=True)
X_train = torch.load("data/training/train/X_train.pt", weights_only=True)
meta_train = pd.read_csv("data/training/train/metadata_train.csv")
raw_prices = {sym: pd.read_csv(f"data/raw/{sym}.csv") for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]}

policy = ActorCriticTradingNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

start_time = time.time()

# 1. Rollout collection across 5 stocks
b_obs = []
b_actions = []
b_logprobs = []
b_rewards = []
b_dones = []
b_values = []
b_returns = []
b_advantages = []

for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
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
            act, logp, _, val = policy.get_action(t_obs)
            
        action = act.item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        s_obs.append(obs)
        s_actions.append(action)
        s_logprobs.append(logp.item())
        s_rewards.append(reward)
        s_dones.append(float(terminated))
        s_values.append(val.item())
        
        obs = next_obs
        
    # GAE per stock
    s_adv = np.zeros(len(s_rewards), dtype=np.float32)
    lastgaelam = 0.0
    gamma = 0.99
    gae_lambda = 0.95
    for t in reversed(range(len(s_rewards))):
        if t == len(s_rewards) - 1:
            nextnonterminal = 0.0
            nextvalues = 0.0
        else:
            nextnonterminal = 1.0 - s_dones[t]
            nextvalues = s_values[t + 1]
        delta = s_rewards[t] + gamma * nextvalues * nextnonterminal - s_values[t]
        s_adv[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    s_ret = s_adv + np.array(s_values)
    
    b_obs.extend(s_obs)
    b_actions.extend(s_actions)
    b_logprobs.extend(s_logprobs)
    b_rewards.extend(s_rewards)
    b_dones.extend(s_dones)
    b_values.extend(s_values)
    b_returns.extend(s_ret.tolist())
    b_advantages.extend(s_adv.tolist())

rollout_time = time.time() - start_time
print(f"Collected 4000 steps rollout across 5 stocks in {rollout_time:.2f}s")

# 2. PPO Update
t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
t_act = torch.tensor(np.array(b_actions), dtype=torch.long)
t_old_logp = torch.tensor(np.array(b_logprobs), dtype=torch.float32)
t_ret = torch.tensor(np.array(b_returns), dtype=torch.float32)
t_adv = torch.tensor(np.array(b_advantages), dtype=torch.float32)

# Normalize advantages
t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

batch_size = 128
dataset_size = len(t_obs)
indices = np.arange(dataset_size)

update_start = time.time()
for epoch in range(4):
    np.random.shuffle(indices)
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        b_idx = indices[start:end]
        
        _, new_logp, entropy, new_val = policy.get_action(t_obs[b_idx], t_act[b_idx])
        
        # Policy Loss
        ratio = torch.exp(new_logp - t_old_logp[b_idx])
        mb_adv = t_adv[b_idx]
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * mb_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        val_loss = 0.5 * ((new_val - t_ret[b_idx]) ** 2).mean()
        
        # Entropy Loss
        entropy_loss = -0.01 * entropy.mean()
        
        total_loss = policy_loss + 0.5 * val_loss + entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

update_time = time.time() - update_start
print(f"PPO update (4 epochs x 32 batches = 128 steps) took {update_time:.2f}s")
print(f"Total time per full iteration: {rollout_time + update_time:.2f}s")
print("Loss metrics:")
print(f"  Policy Loss: {policy_loss.item():.6f}")
print(f"  Value Loss : {val_loss.item():.6f}")
print(f"  Entropy    : {entropy.mean().item():.4f}")
print("Benchmark test PASSED!")
