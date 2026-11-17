# NeuroTrader

## Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading

**B.Tech Final-Year Capstone Project**  
*Evaluation Period: 2021-11-04 to 2026-11-04 (5 NSE Equities)*  
*Status: Completed & 100% Bitwise Reproducible*

---

## 1. Executive Summary

NeuroTrader is an algorithmic trading system that integrates a pre-trained, frozen Recurrent Neural Network (LSTM) market-state encoder with a Proximal Policy Optimization (PPO) reinforcement learning agent. The framework transforms noisy historical market data into stationary, continuous feature representations, enabling an RL agent to make risk-controlled trading decisions.

Evaluated on 5 major National Stock Exchange (NSE) equities over a 5-year window in a genuine bear-market out-of-sample test period, the PPO policy achieves positive risk-adjusted returns while passive benchmarks experience double-digit capital losses.

### Key Performance Highlights (Out-of-Sample Test Set)
- **PPO Trading Agent**: **+0.34% Return**, **+0.13 Sharpe Ratio**, **-13.03% Max Drawdown**, 16.8 trades/asset, ₹1,646 friction
- **Buy-and-Hold Baseline**: **-15.74% Return**, **-0.75 Sharpe Ratio**, **-26.34% Max Drawdown**
- **Moving Average Crossover**: **-9.63% Return**, **-1.54 Sharpe Ratio**, **-13.48% Max Drawdown**
- **LSTM Directional Accuracy**: **55.53%** (497/895 correct on held-out test split, vs 50% random)
- **Reproducibility**: **100.00%** action match across independent runs with master seed 42 (0.00000000 metric delta)

---

## 2. End-to-End System Architecture

```
Raw OHLCV (5 NSE Equities: RELIANCE, TCS, INFY, HDFCBANK, ITC)
  │
  ▼
Data Cleaning & Feature Engineering (14 Stationary Indicators: RSI, MACD, BB, ATR, etc.)
  │
  ▼
Chronological Splitting (70% Train [4000] / 15% Val [885] / 15% Test [895])
  │
  ▼
2-Layer LSTM Market Encoder (14 features × 30 timesteps → 32-D continuous embeddings)
  │ (Frozen Checkpoint: backend/checkpoints/improved_rnn_encoder.pt)
  ▼
Gymnasium Trading Environment (MarketTradingEnv)
  ├── 49-D Hybrid State: [32-D LSTM Embedding + 14-D Technical Features + 3-D Portfolio State]
  ├── Execution Model: Next-Day Open Execution, Close Portfolio Valuation
  └── Trading Frictions: 10 bps (0.10%) transaction costs, Long-Only Constraint
  │
  ▼
Decoupled PPO Actor-Critic Network (Orthogonal Init, LayerNorm, Tanh)
  ├── Actor: 49-D → 64 → 64 → Discrete(3) [HOLD, BUY, SELL]
  └── Critic: 49-D → 64 → 64 → Scalar Value V(s)
  │
  ▼
Model Selection via Validation Sharpe Filter (Excluding Zero-Trade Policies)
  │ (Canonical Checkpoint: backend/checkpoints/ppo_reproducible_best.pt @ Iteration 4)
  ▼
Strict Out-of-Sample Test Evaluation & Baseline Comparison
```

---

## 3. Project Directory Structure

```
NeuroTrade/
├── data/
│   ├── raw/                                # Raw historical CSVs (RELIANCE, TCS, INFY, HDFCBANK, ITC)
│   ├── processed/                          # Cleaned CSVs with 14 engineered indicators
│   ├── training/                           # Chronological tensor splits (train/, validation/, test/)
│   └── notebooks/                          # Data preparation & validation pipeline (01–06)
│       ├── 01_data_collection.ipynb
│       ├── 02_data_cleaning.ipynb
│       ├── 03_eda.ipynb
│       ├── 04_feature_engineering.ipynb
│       ├── 05_preprocessing.ipynb
│       └── 06_data_validation.ipynb
│
├── backend/
│   ├── agents/
│   │   └── ppo_agent.py                    # ActorCriticTradingNetwork, PPOAgent, set_seed(42)
│   ├── environments/
│   │   └── trading_env.py                  # Gymnasium-compliant MarketTradingEnv
│   ├── checkpoints/
│   │   ├── improved_rnn_encoder.pt         # Frozen canonical LSTM encoder (55.53% dir acc)
│   │   ├── ppo_reproducible_best.pt        # Canonical verified PPO model (Iteration 4)
│   │   ├── ppo_reproducible_run1.pt        # Dual-run verification run 1
│   │   ├── ppo_reproducible_run2.pt        # Dual-run verification run 2
│   │   └── ppo_best_agent.pt               # Preserved historical run
│   ├── embeddings/                         # Pre-extracted 32-D continuous tensors
│   └── notebooks/
│       ├── 04_rl_development.ipynb        # Validated Gymnasium env & 6 accounting tests
│       └── 05_ppo_training.ipynb          # Deterministic PPO training & model selection
│
├── reports/
│   ├── final_submission/                   # Complete B.Tech Submission Package
│   │   ├── Final_Project_Report.docx       # Full 30-section academic thesis document
│   │   ├── Project_Presentation.pptx       # 16-slide academic defense presentation
│   │   ├── Reproducibility_Guide.md        # Deterministic reproduction protocol
│   │   ├── Notebook_Audit_Report.md        # 8-notebook code & execution audit
│   │   ├── Final_Submission_Checklist.md   # Deliverables sign-off checklist
│   │   ├── Architecture_Diagrams/          # 7 high-resolution system diagrams
│   │   └── Result_Visualizations/          # 6 verified experimental performance plots
│   └── reinforcement_learning/             # JSON logs, training curves, reproducibility audit
│
├── neuro_env/                              # Python 3.12 virtual environment
├── requirements.txt                        # Pinned dependencies
└── README.md
```

---

## 4. Quickstart & Reproducibility

### Setup Environment
```bash
# Clone or navigate to the repository
cd NeuroTrade

# Create and activate virtual environment
python -m venv neuro_env
.\neuro_env\Scripts\activate      # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Deterministic Seed & Policy Evaluation
All stochastic processes are controlled via centralized seed management:

```python
import sys
sys.path.insert(0, ".")
import torch
from backend.agents.ppo_agent import set_seed, ActorCriticTradingNetwork

# Set master seed
set_seed(42)

# Load canonical checkpoint
checkpoint = torch.load("backend/checkpoints/ppo_reproducible_best.pt", map_location="cpu")
model = ActorCriticTradingNetwork(state_dim=49, action_dim=3, hidden_dim=64)
model.load_state_dict(checkpoint["policy_state_dict"])
model.eval()
print(f"Loaded checkpoint from Iteration {checkpoint['iteration']} with Val Sharpe {checkpoint['val_summary']['Avg_Sharpe']:.4f}")
```

### Notebook Execution Sequence
To inspect or re-run the pipeline in order:
1. `data/notebooks/01_data_collection.ipynb` to `06_data_validation.ipynb`
2. `backend/notebooks/04_rl_development.ipynb` (Gymnasium verification & unit tests)
3. `backend/notebooks/05_ppo_training.ipynb` (Deterministic PPO training & evaluation)

---

## 5. Verified Experimental Results

### Strategy Comparison on Held-Out Test Set (895 Timesteps)
| Strategy | Avg Return (%) | Avg Sharpe | Max Drawdown (%) | Avg Trades / Stock | Avg Friction (₹) |
|---|---|---|---|---|---|
| **PPO Agent (Ours)** | **+0.34%** | **+0.13** | **-13.03%** | **16.8** | **₹1,646** |
| **Buy & Hold Baseline** | -15.74% | -0.75 | -26.34% | 1.0 | ₹100 |
| **MA Crossover (10/30)**| -9.63% | -1.54 | -13.48% | 3.6 | ₹352 |
| **Always Cash (Hold)** | 0.00% | 0.00 | 0.00% | 0.0 | ₹0 |

### Per-Stock Out-of-Sample Performance
| Asset | PPO Return (%) | PPO Sharpe | PPO Max DD (%) | Trades | Friction (₹) |
|---|---|---|---|---|---|
| **RELIANCE** | +3.72% | +0.58 | -5.08% | 20 | ₹2,025 |
| **TCS** | -9.27% | -0.48 | -20.12% | 10 | ₹896 |
| **INFY** | +15.72% | +1.23 | -12.18% | 16 | ₹1,646 |
| **HDFCBANK** | -2.99% | -0.17 | -14.96% | 16 | ₹1,530 |
| **ITC** | -5.49% | -0.53 | -12.83% | 22 | ₹2,133 |

---

## 6. Academic Deliverables & Artifacts

All final academic deliverables have been placed in `reports/final_submission/`:
- **Academic Project Thesis**: [Final_Project_Report.docx](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/final_submission/Final_Project_Report.docx)
- **Slide Presentation**: [Project_Presentation.pptx](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/final_submission/Project_Presentation.pptx)
- **Reproducibility Protocol**: [Reproducibility_Guide.md](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/final_submission/Reproducibility_Guide.md)
- **Notebook Verification Audit**: [Notebook_Audit_Report.md](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/final_submission/Notebook_Audit_Report.md)
- **Submission Sign-Off**: [Final_Submission_Checklist.md](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/final_submission/Final_Submission_Checklist.md)

---

## 7. License & Citation

This codebase is submitted in partial fulfillment of the requirements for the Degree of Bachelor of Technology in Computer Science & Engineering.
