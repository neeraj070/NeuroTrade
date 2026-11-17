import sys
sys.path.insert(0, ".")
import torch
from backend.agents.ppo_agent import set_seed, ActorCriticTradingNetwork

set_seed(42)

checkpoint = torch.load("backend/checkpoints/ppo_reproducible_best.pt", map_location="cpu")
model = ActorCriticTradingNetwork(state_dim=49, action_dim=3, hidden_dim=64)
model.load_state_dict(checkpoint["policy_state_dict"])
model.eval()
print(f"SUCCESS! Loaded checkpoint from Iteration {checkpoint['iteration']} with Val Sharpe {checkpoint['val_summary']['Avg_Sharpe']:.4f}")
