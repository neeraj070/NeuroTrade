# Archived Notebooks

This directory contains legacy and superseded notebooks preserved for historical continuity and reference.

## 1. `03_rl_development_legacy.ipynb`
- **Archival Date**: November 15, 2026
- **Status**: Superseded by `backend/notebooks/04_rl_development.ipynb`.
- **Reason for Archival**: Utilized an early, obsolete 34-dimensional state representation (32-D embeddings + 2-D portfolio) before the official 49-dimensional hybrid state architecture (32-D embeddings + 14-D explicit technical features + 3-D internal portfolio state) was established in Step 1.
- **Preserved Useful Work**:
  - Contains preliminary PyTorch architecture scaffolding for `ActorCriticTradingNetwork` (shared layers with LayerNorm + Tanh, discrete actor head Linear(64, action_dim), and scalar critic head Linear(64, 1)).
  - This architecture template will be adapted to 49-dimensional state inputs in Phase 11 (Step 2: PPO Agent Development).
