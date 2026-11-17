import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

from backend.environments.trading_env import MarketTradingEnv

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

# Smoke test
policy = ActorCriticTradingNetwork()
dummy_obs = torch.randn(4, 49)
act, logp, ent, val = policy.get_action(dummy_obs)
print("Policy smoke test passed:")
print(f"  Obs shape: {dummy_obs.shape}")
print(f"  Actions: {act.tolist()}")
print(f"  Log probs: {logp.shape}")
print(f"  Entropy: {ent.shape}, Mean Entropy: {ent.mean().item():.4f} (Expected near ln(3) ~ 1.098)")
print(f"  Values: {val.shape}")
assert abs(ent.mean().item() - np.log(3)) < 0.1, "Initial entropy not near uniform!"
print("Actor-Critic network architecture and orthogonal initialization [VERIFIED!]")
