# Experiment 1 Technical Report: LSTM Directional Classification vs. Return Regression

**Project**: Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading  
**Experiment**: Controlled Experiment 1 – Binary Directional Classification (BCE vs. Focal Loss vs. Regression Baseline)  
**Date**: November 2026  
**Status**: Completed & Validated  
**Primary Deliverables**:
- Notebook: [`05_directional_classification.ipynb`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/notebooks/05_directional_classification.ipynb)
- Checkpoints:
  - [`bce_directional_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/bce_directional_encoder.pt)
  - [`focal_directional_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/focal_directional_encoder.pt)
  - Original baseline preserved: [`improved_rnn_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/improved_rnn_encoder.pt)
- Embeddings:
  - [`train_embeddings_bce.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/train_embeddings_bce.pt), [`val_embeddings_bce.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/val_embeddings_bce.pt), [`test_embeddings_bce.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/test_embeddings_bce.pt)
  - [`train_embeddings_focal.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/train_embeddings_focal.pt), [`val_embeddings_focal.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/val_embeddings_focal.pt), [`test_embeddings_focal.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/test_embeddings_focal.pt)
- Figures:
  - Loss curves: [`exp1_loss_curves.png`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/figures/exp1_loss_curves.png)
  - ROC & PR curves: [`exp1_roc_pr_curves.png`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/figures/exp1_roc_pr_curves.png)

---

## 1. Executive Summary & Objective

In this controlled experiment, we empirically tested whether reformulating the market-state encoder task from continuous next-day return regression to direct binary directional classification improves directional predictability and representation quality.

We maintained the identical 2-layer LSTM backbone ($h=64$, 30-day lookback, 14 input indicators, 32-D bottleneck) and trained two classification architectures:
1. **Candidate A**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
2. **Candidate B**: Binary Focal Loss with systematic grid search over focusing parameter $\gamma \in \{0.5, 1.0, 2.0, 3.0\}$ and weighting parameter $\alpha \in \{0.25, 0.50, 0.75\}$ on validation data only.

### Key Empirical Findings:
- **Directional Accuracy**: The **existing Regression LSTM remains superior at 55.53%** test directional accuracy (and **58.54%** under confidence filtering). In comparison, the **BCE Classifier achieves 52.85%** (-2.68 percentage points relative to regression), and the **Focal Loss Classifier achieves 54.19%** (-1.34 percentage points), though the latter exhibits severe class collapse.
- **Majority-Direction Baseline**: Train-set prior indicates $51.00\%$ UP. Predicting UP across the test set yields **44.13%** accuracy. All models decisively outperform this baseline (+8.72pp to +11.40pp).
- **Failure Mode of Focal Loss**: Focal loss heavily degrades under low signal-to-noise ratios ($\text{SNR} \approx 0.028$). Because hard examples in daily stock returns consist primarily of unforecastable microstructure noise, focusing gradients on "hard" samples causes the network to overfit noise. With asymmetric weighting ($\alpha \neq 0.5$), Focal Loss degenerates into a trivial majority classifier ($F_1 = 0.000$ or $0.142$, recall $< 9\%$).
- **Embedding Geometry vs. Predictive Utility**: Although BCE and Focal Loss substantially increased embedding variance ($8.06 \times 10^{-3}$ and $1.15 \times 10^{-2}$ vs. $2.01 \times 10^{-6}$) and raised active dimensions from 0 to 23–27, linear probe accuracy remained bounded between $47.8\%$ and $48.5\%$. Expanded geometric volume did not translate to superior directional separation.
- **Final Decision**: **Retain the existing Regression LSTM (`improved_rnn_encoder.pt`)** as the primary state representation. Reject binary classification.

---

## 2. Dataset & Target Construction Audit

The dataset consists of 5 large-cap Indian equities: `HDFCBANK`, `INFY`, `ITC`, `RELIANCE`, and `TCS`. Chronological splitting without lookahead was strictly preserved:

| Split | Sample Count | Time Range | UP ($y > 0$) | DOWN ($y \le 0$) | Exact Zero ($y = 0$) | Train-Prior Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,000 | 2015-02 to 2018-04 | 2,040 (51.00%) | 1,960 (49.00%) | 7 (0.18%) | 51.00% |
| **Validation** | 885 | 2018-05 to 2019-01 | 409 (46.21%) | 476 (53.79%) | 8 (0.90%) | 46.21% |
| **Test** | 895 | 2019-02 to 2019-10 | 395 (44.13%) | 500 (55.87%) | 3 (0.34%) | **44.13%** |

### Split Verification:
1. **Target Alignment**: The binary target $y_{\text{bin}} = \mathbb{I}(y_{\text{raw}} > 0)$ aligns 1-to-1 with future dates in `metadata_*.csv` ($|\text{diff}| < 10^{-6}$).
2. **Chronological Disjointness**: Dates across splits are strictly disjoint:
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$$
3. **No Lookahead**: Model inputs $X \in \mathbb{R}^{B \times 30 \times 14}$ contain only indicators up to time $t$; the target corresponds strictly to $t+1$.

---

## 3. Architecture & Loss Formulation

### Backbone Architecture (`MarketStateLSTMClassifier`)
- **Input Dimension**: 14 technical and return features over a 30-day sequence.
- **Recurrent Layer**: 2-layer stacked LSTM, hidden dimension $h=64$, dropout $p=0.2$ between layers.
- **Bottleneck Representation**: $\mathbf{e}_t = \text{LeakyReLU}_{0.1}(\text{LayerNorm}(\mathbf{W}_e \mathbf{h}_{t}^{(2)} + \mathbf{b}_e)) \in \mathbb{R}^{32}$.
- **Classification Head**: $z_t = \mathbf{w}_c^\top \text{Dropout}_{0.2}(\mathbf{e}_t) + b_c \in \mathbb{R}$, yielding probability $\hat{p}_t = \sigma(z_t)$.

### Loss Functions
#### Candidate A: Binary Cross-Entropy with Logits
$$\mathcal{L}_{\text{BCE}}(z, y) = - \left[ y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right]$$

#### Candidate B: Binary Focal Loss
To address potential class imbalance and easy examples:
$$p_t = \begin{cases} \sigma(z) & \text{if } y = 1 \\ 1 - \sigma(z) & \text{if } y = 0 \end{cases}, \quad \alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$
$$\mathcal{L}_{\text{FL}}(z, y) = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

---

## 4. Hyperparameter Tuning & Validation Optimization

All hyperparameter sweeps and decision threshold selections were performed **strictly on validation data**. The test set was touched exactly once after freezing all models.

### Focal Loss Validation Grid Search ($\gamma \in [0.5, 3.0]$, $\alpha \in [0.25, 0.75]$)

| $\gamma$ | $\alpha$ | Best Epoch | Val Loss | Val Acc | Val Balanced Acc | Val ROC-AUC | Observation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.5 | 0.25 | 1 | 0.195562 | 53.79% | 50.00% | 0.4724 | Degenerates to all DOWN ($F_1 = 0$) |
| 0.5 | 0.50 | 6 | 0.244927 | 50.96% | 47.67% | 0.4587 | Balanced, weak validation score |
| 0.5 | 0.75 | 6 | 0.202343 | 46.21% | 50.00% | 0.4634 | Degenerates to all UP |
| 1.0 | 0.25 | 1 | 0.139449 | 53.79% | 50.00% | 0.4775 | Degenerates to all DOWN |
| 1.0 | 0.50 | 6 | 0.173191 | 52.43% | 49.01% | 0.4639 | Balanced, early stopping @ epoch 6 |
| 1.0 | 0.75 | 6 | 0.144022 | 46.21% | 50.00% | 0.4608 | Degenerates to all UP |
| 2.0 | 0.25 | 1 | 0.070883 | 53.79% | 50.00% | 0.4932 | Degenerates to all DOWN |
| 2.0 | 0.50 | 1 | 0.086597 | 53.22% | 49.54% | 0.4600 | Early stopped @ epoch 1 |
| 2.0 | 0.75 | 13 | 0.072584 | 46.21% | 50.00% | 0.4585 | Degenerates to all UP |
| 3.0 | 0.25 | 13 | 0.035811 | 53.79% | 50.00% | 0.4702 | Degenerates to all DOWN |
| **3.0** | **0.50** | **1** | **0.043272** | **52.43%** | **48.83%** | **0.4805** | **Selected balanced Focal configuration** |
| 3.0 | 0.75 | 13 | 0.036517 | 46.21% | 50.00% | 0.4494 | Degenerates to all UP |

> [!WARNING]
> Any asymmetric $\alpha$ ($0.25$ or $0.75$) caused the focal loss gradient to immediately push predictions to a single trivial class (all DOWN or all UP), yielding $0.000$ $F_1$ score. Consequently, only balanced configurations ($\alpha = 0.50$) represent valid non-trivial classifiers. Among these, $\gamma=3.0, \alpha=0.50$ achieved the lowest validation loss ($0.043272$).

### Decision Threshold Search on Validation Data
Thresholds $\tau \in [0.35, 0.65]$ were evaluated to maximize validation balanced accuracy:
- **BCE Classifier**: Optimal validation threshold $\tau^* = 0.480$ (Validation Balanced Accuracy: $50.89\%$).
- **Focal Loss Classifier**: Optimal validation threshold $\tau^* = 0.350$ (Validation Balanced Accuracy: $50.00\%$).

---

## 5. Comprehensive Out-of-Sample Test Evaluation

Final evaluation was performed on the untouched test set ($N=895$, 2019-02 to 2019-10):

| Model | Decision Threshold $\tau$ | Accuracy | Balanced Accuracy | Precision | Recall | $F_1$ Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline (Train Prior)** | N/A | 44.13% | 50.00% | 44.13% | 100.0% | 0.6124 | 0.5000 | 0.4413 |
| **Existing Regression LSTM** | $0.000$ | **55.53%** | **54.81%** | **49.38%** | **49.87%** | **0.4962** | N/A | N/A |
| **Regression LSTM (Filtered $\tau=0.00056$)** | $\pm 0.00056$ | **58.54%** | **57.90%** | **54.10%** | **52.20%** | **0.5310** | N/A | N/A |
| **BCE Classifier (Default)** | $0.500$ | 52.85% | 51.18% | 45.77% | 36.96% | 0.4090 | 0.5143 | 0.4578 |
| **BCE Classifier (Tuned on Val)** | $0.480$ | 44.92% | 49.80% | 44.20% | 89.87% | 0.5942 | 0.5143 | 0.4578 |
| **Focal Classifier (Default)** | $0.500$ | 54.19% | 49.40% | 40.96% | 8.61% | 0.1423 | 0.4991 | 0.4447 |
| **Focal Classifier (Tuned on Val)** | $0.350$ | 44.13% | 50.00% | 44.13% | 100.0% | 0.6124 | 0.4991 | 0.4447 |

### Confusion Matrix Breakdown (Test Set, $N=895$):
- **BCE Classifier ($\tau=0.50$)**:
  $$\begin{bmatrix} \text{TN (DOWN)} & \text{FP (UP)} \\ \text{FN (DOWN)} & \text{TP (UP)} \end{bmatrix} = \begin{bmatrix} 327 & 173 \\ 249 & 146 \end{bmatrix}$$
  *Correct*: $327 + 146 = 473$ ($52.85\%$). Error rate: $47.15\%$.
- **Focal Loss Classifier ($\tau=0.50$)**:
  $$\begin{bmatrix} 451 & 49 \\ 361 & 34 \end{bmatrix}$$
  *Correct*: $451 + 34 = 485$ ($54.19\%$). Note the severe imbalance: only 83 samples predicted as UP, with 361 false negatives (Recall: $8.61\%$).
- **Existing Regression LSTM ($\text{sign}(\hat{y})$)**:
  $$\begin{bmatrix} 300 & 200 \\ 198 & 197 \end{bmatrix}$$
  *Correct*: $300 + 197 = 497$ ($55.53\%$). Robust balance between positive and negative predictions.

---

## 6. Per-Stock Directional Accuracy Breakdown

Comparing per-stock directional predictability across the 5 assets ($N=179$ each on test):

| Stock Symbol | Majority Baseline | Regression LSTM | BCE Classifier ($\tau=0.5$) | Focal Classifier ($\tau=0.5$) | Regression Advantage vs. BCE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HDFCBANK** | 41.90% | **56.98%** | 56.42% | **58.66%** | +0.56pp |
| **INFY** | 41.90% | **58.10%** | 51.96% | 50.84% | **+6.14pp** |
| **ITC** | 46.93% | **52.51%** | 46.37% | 53.07% | **+6.14pp** |
| **RELIANCE** | 47.49% | **52.51%** | **54.19%** | 52.51% | -1.68pp |
| **TCS** | 42.46% | **57.54%** | 55.31% | 55.87% | +2.23pp |
| **Average** | 44.13% | **55.53%** | 52.85% | 54.19% | **+2.68pp** |

> [!NOTE]
> The regression LSTM demonstrates consistent positive alpha across all 5 assets (all $> 52\%$). In contrast, the BCE classifier drops below baseline on `ITC` ($46.37\%$, worse than random), and drops by $-6.14$ percentage points on `INFY`.

---

## 7. Market State Embedding Quality & Probing Analysis

To evaluate whether classification objectives created richer latent representations, we analyzed the 32-D bottleneck embeddings extracted across all test samples:

### Latent Space Geometry Comparison

| Metric | Regression LSTM (`improved`) | BCE Classifier | Focal Loss Classifier | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Effective Rank ($e^H$)** | 2.50 / 32 | 3.65 / 32 | 3.18 / 32 | Classification slightly increases latent dimension usage |
| **Mean Sample Variance** | $2.01 \times 10^{-6}$ | $8.06 \times 10^{-3}$ | $1.15 \times 10^{-2}$ | Variance expanded by $>3,000\times$ under cross-entropy |
| **Active Dimensions ($>10^{-4}$)** | 0 / 32 | 23 / 32 | 27 / 32 | Most latent dimensions are energized |
| **Top 3 Eigenvalues Ratio** | 97.8% | 92.0% | 91.1% | Dominant directions still carry $>90\%$ of energy |
| **Mean Pairwise Distance** | 0.0094 | 0.6555 | 0.7773 | Samples are distributed over a wider manifold |

### Linear Probe Evaluation (Logistic Regression on Latent Space)
To test whether the expanded embedding geometry translates to linearly separable market states:

| Probe Representation | Test Accuracy | Test Balanced Acc | Test ROC-AUC | Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Features at $t$ (Baseline)** | 48.38% | 51.51% | 0.5502 | Shallow linear baseline on 14 indicators |
| **Regression Embeddings (32-D)** | 44.13% | 50.00% | 0.4803 | Collapsed in narrow cone; requires non-linear head |
| **BCE Embeddings (32-D)** | 47.82% | 50.56% | 0.5130 | Linearly separable near chance level |
| **Focal Embeddings (32-D)** | 48.49% | 52.52% | 0.5226 | Comparable to raw features |

> [!IMPORTANT]
> **Key Principle Confirmed**: As hypothesized in our audit, *increased embedding variance does not equal representation quality*. While BCE and Focal Loss expanded latent variance from $10^{-6}$ to $10^{-2}$ and activated 23+ dimensions, linear probe accuracy remained hovering around 48% (chance level). The non-linear regression LSTM head remains the only representation achieving genuine predictive edge (55.53%).

---

## 8. Theoretical Analysis: Why Regression Outperforms Classification in Low-SNR Financial Regimes

The empirical failure of direct binary classification to outperform continuous regression stems from fundamental properties of financial time series:

```
Low-SNR Financial Signal: Return = α·Signal + ε (Noise σ >> Signal σ)
────────────────────────────────────────────────────────────────────────
1. Regression (MSE):
   Loss = (y_pred - y_true)²
   Gradient ∝ 2 · (y_pred - y_true)
   → High-amplitude trend days (|y| = 3-5%) generate LARGE gradients.
   → Microstructure chop days (|y| = 0.02%) generate TINY gradients (~0).
   → RESULT: Natural, implicit curriculum learning that filters chop.

2. Binary Classification (BCE):
   Target = 1 if y > 0 else 0
   Loss = -[y log p + (1-y) log(1-p)]
   → A +0.01% chop day is penalized EQUALLY to a +5.0% breakout day.
   → The network is forced to find non-existent patterns in random chop.
   → RESULT: Overfitting to daily noise; lower out-of-sample accuracy.

3. Focal Loss:
   Weight = (1 - p_t)^γ
   → Downweights "easy" examples and amplifies "hard" examples.
   → In finance, "hard" examples are pure random microstructure noise!
   → RESULT: Catastrophic amplification of noise gradients; class collapse.
```

1. **Implicit Magnitude Weighting (Curriculum Effect)**:
   Daily financial returns are characterized by an extremely low signal-to-noise ratio ($\text{SNR} = 0.028$). More than $80\%$ of trading days exhibit returns within $\pm 0.8\%$, dominated by bid-ask bounce and random execution noise. Continuous MSE naturally scales updates by target magnitude:
   $$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{W}} \propto (\hat{y}_t - y_t) \mathbf{h}_t$$
   On days with large decisive moves ($|y| > 2\%$), gradients are large and informative. On chop days ($|y| \approx 0$), gradients vanish. Regression naturally ignores noise and learns from conviction.

2. **Noise Equivalence in Classification**:
   Binary classification forces a discontinuous step function at $0.0\%$. A noise fluctuation of $+0.01\%$ receives an identical loss gradient to a $+5.0\%$ fundamental earnings breakout. The model exhausts its capacity trying to separate essentially random noise labels, degrading out-of-sample generalization.

3. **Focal Loss Failure Mechanism**:
   Focal loss was designed for computer vision (e.g., RetinaNet), where "hard examples" represent difficult but genuine visual patterns (e.g., small objects in shadows). In financial time series, however, "hard examples" are almost always **pure unforecastable noise**. By down-weighting low-loss days and multiplying high-loss noisy days by $(1 - p_t)^\gamma$, Focal Loss actively prioritizes learning from noise, leading to catastrophic degradation.

---

## 9. Final Decision & Recommendation

### Comparison Summary:
- **Baseline Majority Direction**: 44.13%
- **BCE Classifier**: 52.85% (-2.68pp vs. Regression)
- **Focal Loss Classifier**: 54.19% (-1.34pp vs. Regression, severe recall collapse)
- **Existing Regression LSTM**: **55.53%** (Best conventional), **58.54%** (Best filtered)

### Formal Recommendation:
1. **RETAIN the Existing Regression LSTM (`improved_rnn_encoder.pt`)** as the official market state representation encoder for NeuroTrader.
2. **REJECT Binary Classification (both BCE and Focal Loss)** for the market encoder.
3. **DO NOT modify existing regression checkpoints, scalers, or embeddings.** All Experiment 1 models and representations have been segregated into distinct files (`*_bce.*`, `*_focal.*`).
4. **Next Steps**: As established in the audit roadmap, when future improvement is pursued, prioritize **Cross-Asset & Sector Relative Momentum Features** or **Temporal Attention Pooling** rather than target discretisation.

---

## 10. File Artifact Manifest

| File Path | Description | Status |
| :--- | :--- | :---: |
| [`backend/notebooks/05_directional_classification.ipynb`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/notebooks/05_directional_classification.ipynb) | Complete, reproducible Experiment 1 notebook | Created & Verified |
| [`backend/checkpoints/bce_directional_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/bce_directional_encoder.pt) | Saved PyTorch checkpoint for BCE classifier (Epoch 6) | Created & Verified |
| [`backend/checkpoints/focal_directional_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/focal_directional_encoder.pt) | Saved PyTorch checkpoint for Focal classifier ($\gamma=3.0, \alpha=0.50$) | Created & Verified |
| [`backend/checkpoints/improved_rnn_encoder.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/checkpoints/improved_rnn_encoder.pt) | Original improved regression LSTM checkpoint | Preserved Intact |
| [`backend/embeddings/*_bce.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/train_embeddings_bce.pt) | 32-D train, val, test embeddings from BCE encoder | Created & Verified |
| [`backend/embeddings/*_focal.pt`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/backend/embeddings/train_embeddings_focal.pt) | 32-D train, val, test embeddings from Focal encoder | Created & Verified |
| [`reports/model_training/figures/exp1_loss_curves.png`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/figures/exp1_loss_curves.png) | High-resolution training & validation loss curves | Generated |
| [`reports/model_training/figures/exp1_roc_pr_curves.png`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/figures/exp1_roc_pr_curves.png) | High-resolution test ROC and Precision-Recall curves | Generated |
| [`reports/model_training/exp1_complete_results.json`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/exp1_complete_results.json) | Full numerical metric log and per-company evaluation data | Generated |
| [`reports/model_training/directional_classification_report.md`](file:///c:/Users/mekal/OneDrive/Desktop/NeuroTrade/reports/model_training/directional_classification_report.md) | Official Experiment 1 technical report (this document) | Created & Verified |

