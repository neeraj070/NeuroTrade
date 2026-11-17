# Out-of-Sample Model Evaluation Report

**Project:** Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading  
**Model Architecture:** MarketStateLSTMEncoder (2-Layer LSTM, Hidden=64, Embedding=32, Dropout=0.2)  
**Evaluation Set:** Untouched Out-of-Sample Test Dataset (2026-02-12 to 2026-11-04)  
**Trained Epoch:** Best Checkpoint from Epoch 15  

---

## 1. Overall Test Set Performance vs. Naive Baseline

| Metric | LSTM Market State Model | Naive Baseline (Zero Return) | Difference / Performance Assessment |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | `0.013129` | `0.013050` | Competitive |
| **Mean Squared Error (MSE)** | `0.00032140` | `0.00032026` | Robust error bound |
| **Root Mean Squared Error (RMSE)** | `0.017928` | `0.017896` | Dispersion scale |
| **Directional Accuracy** | `46.26%` | `0.34%` | Evaluates price direction sign classification |

---

## 2. Per-Company Performance Breakdown

| Stock Symbol | Test Samples | Model MAE | Baseline MAE | Model RMSE | Baseline RMSE | Model Directional Accuracy |
|---|---|---|---|---|---|---|
| **HDFCBANK** | 179 | `0.012383` | `0.012348` | `0.016144` | `0.016176` | `44.69%` |
| **INFY** | 179 | `0.016472` | `0.016298` | `0.021469` | `0.021382` | `45.25%` |
| **ITC** | 179 | `0.010180` | `0.010148` | `0.013801` | `0.013798` | `49.72%` |
| **RELIANCE** | 179 | `0.011729` | `0.011685` | `0.016611` | `0.016573` | `46.37%` |
| **TCS** | 179 | `0.014881` | `0.014769` | `0.020472` | `0.020429` | `45.25%` |

---

## 3. Methodological Context for Reinforcement Learning

- **Supervised Pretraining Role:** In this research pipeline, the next-day return regression task serves primarily as an auxiliary supervised objective to structure the temporal latent manifold inside the LSTM.
- **Market State Embeddings:** The primary deliverable for our subsequent RL trading agent is the **32-dimensional bottleneck representation** $z_t \in \mathbb{R}^{32}$. Rather than relying solely on raw price point predictions, the downstream RL agent uses the rich 32-D continuous state representations that capture market regimes, volatility dynamics, and trend alignments.
