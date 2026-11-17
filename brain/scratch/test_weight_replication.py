import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.ppo_agent import ActorCriticTradingNetwork

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Test 1: simulate run_ppo_pipeline.py
set_seed(42)
policy_standalone = ActorCriticTradingNetwork().float()
w_standalone = policy_standalone.actor[0].weight.clone()

# Test 2: simulate notebook as it was (with dummy network in between)
set_seed(42)
dummy_policy = ActorCriticTradingNetwork().float()
dummy_obs = torch.randn(2, 49)
_ = dummy_policy.get_action(dummy_obs)

policy_nb_old = ActorCriticTradingNetwork().float()
w_nb_old = policy_nb_old.actor[0].weight.clone()

print("Difference between standalone and old notebook weights:")
diff_old = (w_standalone - w_nb_old).abs().max().item()
print(f"Max absolute diff: {diff_old:.6f} (Should be > 0 due to RNG shift)")

# Test 3: simulate notebook with set_seed(42) right before training policy initialization
set_seed(42)
dummy_policy = ActorCriticTradingNetwork().float()
dummy_obs = torch.randn(2, 49)
_ = dummy_policy.get_action(dummy_obs)

set_seed(42) # reseed right before training policy
policy_nb_fixed = ActorCriticTradingNetwork().float()
w_nb_fixed = policy_nb_fixed.actor[0].weight.clone()

print("\nDifference between standalone and fixed notebook weights (with reseed):")
diff_fixed = (w_standalone - w_nb_fixed).abs().max().item()
print(f"Max absolute diff: {diff_fixed:.6f} (Should be 0.000000)")
assert diff_fixed == 0.0, "Weights must match exactly!"
print("EXACT INITIAL WEIGHT REPLICATION VERIFIED!")
