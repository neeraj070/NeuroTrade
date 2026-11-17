# NeuroTrader Reproducibility Guide

**Project**: Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading  
**Author**: B.Tech Final-Year Project  
**Verified**: 2026-11-15 22:02:18  
**Status**: ✅ PASSED — 100% bitwise reproducible across two independent runs

---

## 1. Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Operating System | Windows 11 (verified) |
| Python Version | 3.12 (virtual env: `neuro_env/`) |
| CUDA | Not required — CPU-only execution |
| PyTorch | ≥ 2.2.0 (CPU build) |
| Gymnasium | ≥ 1.0.0 |
| RAM | ≥ 8 GB recommended |
| Storage | ≈ 2 GB (checkpoints + datasets + notebooks) |

> **Note:** Bitwise reproducibility is verified on CPU. GPU execution may introduce additional floating-point non-determinism due to CUDA non-deterministic kernels.

---

## 2. Installation

```bash
# Clone or extract the project
cd path/to/NeuroTrade

# Create virtual environment
python -m venv neuro_env

# Activate (Windows)
neuro_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Master Seed

All stochastic components are seeded via a single call:

```python
from backend.agents.ppo_agent import set_seed
set_seed(42)   # master_seed = 42
```

**What `set_seed(42)` controls:**
- `random.seed(42)` — Python built-in RNG
- `numpy.random.seed(42)` — NumPy RNG  
- `torch.manual_seed(42)` — PyTorch CPU RNG
- `torch.cuda.manual_seed_all(42)` — PyTorch CUDA RNG (if applicable)
- `torch.backends.cudnn.deterministic = True` — CuDNN determinism
- `torch.backends.cudnn.benchmark = False` — Disable auto-tuning
- Environment reset seeds (via `Gymnasium` seed parameter)

---

## 4. Data Acquisition

```python
import yfinance as yf

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
start_date = "2021-11-04"
end_date   = "2026-11-04"

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    # Saves to data/raw/<TICKER>.csv
```

**Dataset Split (Chronological — no shuffling):**

| Split | Records | Period |
|-------|---------|--------|
| Train | 4,000 transitions | 2021-11-04 → ~2024-03 |
| Validation | 885 transitions | ~2024-03 → ~2025-05 |
| Test | 895 transitions | ~2025-05 → 2026-11-04 |

---

## 5. Notebook Execution Order

Execute notebooks **in strict sequence** — later notebooks depend on outputs of earlier ones:

### Stage 1 — Data Pipeline (`data/notebooks/`)

| Order | Notebook | Purpose | Output |
|-------|----------|---------|--------|
| 1 | `01_data_collection.ipynb` | Download NSE OHLCV data via yfinance | `data/raw/*.csv` |
| 2 | `02_data_cleaning.ipynb` | OHLC integrity, zero-price/volume audit | Cleaned CSVs |
| 3 | `03_eda.ipynb` | Price, volume, return distributions | EDA plots |
| 4 | `04_feature_engineering.ipynb` | 14 technical indicators (SMA, EMA, RSI, MACD, BB, ATR, etc.) | `data/processed/*.csv` |
| 5 | `05_preprocessing.ipynb` | Chronological split, MinMax scaling, 30-step sliding windows | `data/training/` tensors |
| 6 | `06_data_validation.ipynb` | Leakage audit, shape verification | Validation report |

### Stage 2 — LSTM Training (requires `notebooks/` in backend)

> These notebooks are in `backend/notebooks/` or project root — check the LSTM training notebook for the exact path.

### Stage 3 — RL Development

| Order | Notebook | Purpose | Output |
|-------|----------|---------|--------|
| 7 | `backend/notebooks/04_rl_development.ipynb` | Gymnasium `MarketTradingEnv`, 6 unit tests | Validated env |
| 8 | `backend/notebooks/05_ppo_training.ipynb` | PPO training (35 iterations), evaluation | Checkpoints + results |

---

## 6. Model Checkpoints

### LSTM Encoder (Frozen)
| Field | Value |
|-------|-------|
| Path | `backend/checkpoints/improved_rnn_encoder.pt` |
| SHA-256 | `8bab84a4ee5448e0…` |
| Architecture | 2-layer LSTM, input=14, hidden=64, output=32 |
| Status | Frozen during PPO training — do NOT retrain |

### PPO Agent (Canonical)
| Field | Value |
|-------|-------|
| Path | `backend/checkpoints/ppo_reproducible_best.pt` |
| SHA-256 | `11b09d4855dfacc7…` |
| Selected at | Iteration 4 (highest val Sharpe among active policies) |
| Validation Sharpe | -0.0079 |
| Test Return | +0.34% |
| Status | Canonical verified checkpoint |

> `backend/checkpoints/ppo_best_agent.pt` is preserved as a historical artifact (Iteration 1 from an earlier non-deterministic run) — do NOT use for final evaluation.

---

## 7. Reproducing LSTM Evaluation

```python
import torch
from backend.agents.ppo_agent import set_seed
set_seed(42)

# Load embeddings (pre-extracted, frozen)
test_emb = torch.load("backend/embeddings/test_embeddings_improved.pt")

# LSTM evaluation metrics (from verified notebook output):
# MAE:  0.012997
# RMSE: 0.017870
# Directional Accuracy: 55.53% (497/895 correct)
# Zero-baseline MAE:    0.013050  → LSTM beats naive baseline
```

---

## 8. Reproducing PPO Evaluation

```python
import torch, sys
sys.path.insert(0, ".")
from backend.agents.ppo_agent import set_seed, ActorCriticTradingNetwork

set_seed(42)  # Must be called BEFORE model instantiation

# Load canonical checkpoint
checkpoint = torch.load("backend/checkpoints/ppo_reproducible_best.pt", map_location="cpu")
model = ActorCriticTradingNetwork(state_dim=49, action_dim=3, hidden_dim=64)
model.load_state_dict(checkpoint["policy_state_dict"])
model.eval()

# Expected test results (verified, 895 steps × 5 assets):
# Avg Return:         +0.34%
# Avg Sharpe:         +0.13
# Avg Max Drawdown:   -13.03%
# Avg Trades/Asset:    16.8
# Avg Transaction Cost: ₹1,646
```

---

## 9. Dual-Run Verification Results

Two fully independent runs with `master_seed=42` produced identical results:

| Metric | Run 1 | Run 2 | Max Difference |
|--------|-------|-------|----------------|
| Initial weights max diff | — | — | **0.00000000** |
| Policy loss max diff | — | — | **0.00000000** |
| Value loss max diff | — | — | **0.00000000** |
| Val Sharpe max diff | — | — | **0.00000000** |
| Val Return max diff | — | — | **0.00000000** |
| Test Return diff | — | — | **0.00000000** |
| Test Sharpe diff | — | — | **0.00000000** |
| Action match % | — | — | **100.00%** (895/895) |
| Best iteration | Iter 4 | Iter 4 | ✅ |

---

## 10. Known Limitations and Caveats

1. **GPU Non-Determinism**: Reproducibility is verified on CPU only. If a GPU is available and PyTorch selects CUDA kernels, additional floating-point non-determinism may occur despite `torch.backends.cudnn.deterministic = True`.

2. **Data Notebook Re-Execution**: Re-downloading data from yfinance may produce slightly different values if the provider updates historical data. Use the archived tensors in `data/training/` for exact reproducibility.

3. **Validation Sharpe is Negative**: Iteration 4's validation Sharpe of -0.0079 is negative because the validation period (avg B&H: -9.58%) was a severe bear market. The PPO agent's -0.60% return significantly outperformed the market, hence a *relatively* best policy despite absolute negative Sharpe.

4. **Bear Market Regime**: Both validation and test periods are bear markets. Results may not generalize to bull markets or sideways conditions without retraining.

5. **Single-Asset Independent Evaluation**: Each of the 5 assets is traded independently with ₹1,00,000 initial capital. The agent does not perform cross-asset portfolio optimization.

---

## 11. File Integrity Check

Run to verify core assets are unchanged:

```python
import hashlib, json
from pathlib import Path

files = {
    "backend/checkpoints/improved_rnn_encoder.pt": "8bab84a4ee5448e01f0422262e4e0abe78c1957f80481d1dd12a9b8e03286c52",
    "backend/checkpoints/ppo_reproducible_best.pt": "11b09d4855dfacc78d69fef2ecee760c4e72d817598ffd20fc2eb578fe39a22e",
}

for path, expected_sha in files.items():
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    status = "✅ OK" if actual == expected_sha else "❌ MISMATCH"
    print(f"{status} {path}")
```

---

*Guide generated from verified experimental results. All metrics sourced from `reports/reinforcement_learning/ppo_reproducibility_results.json` and `ppo_training_results.json`.*
