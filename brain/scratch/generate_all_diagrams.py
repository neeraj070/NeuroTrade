import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

mpl_dir = os.path.join(PROJECT_ROOT, "backend", ".mpl_cache")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

arch_dir = os.path.join(PROJECT_ROOT, "reports", "final_submission", "Architecture_Diagrams")
res_dir = os.path.join(PROJECT_ROOT, "reports", "final_submission", "Result_Visualizations")
os.makedirs(arch_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

print("=== GENERATING PROFESSIONAL ACADEMIC DIAGRAMS AND VISUALIZATIONS ===")

# Color palette
NAVY = "#1B365D"
TEAL = "#008080"
SLATE = "#4A5568"
LIGHT_BG = "#F7FAFC"
ACCENT_BLUE = "#3182CE"
ACCENT_GREEN = "#38A169"
ACCENT_RED = "#E53E3E"
BOX_BG = "#EDF2F7"
BORDER_COL = "#CBD5E0"

# -------------------------------------------------------------
# 1. Overall System Architecture Diagram
# -------------------------------------------------------------
def make_overall_architecture():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    # Title
    ax.text(7.5, 7.6, "NeuroTrader: Overall System Architecture", fontsize=16, fontweight="bold", ha="center", color=NAVY)
    ax.text(7.5, 7.2, "Decoupled RNN Market Encoder + Gymnasium MarketTradingEnv + Actor-Critic PPO", fontsize=11, fontstyle="italic", ha="center", color=SLATE)
    
    # Block 1: Market Data & Preprocessing
    rect1 = patches.FancyBboxPatch((0.5, 3.8), 2.8, 3.0, boxstyle="round,pad=0.2", facecolor="#EBF8FF", edgecolor=ACCENT_BLUE, linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(1.9, 6.5, "1. Market Data Pipeline", fontsize=11, fontweight="bold", ha="center", color=NAVY)
    ax.text(1.9, 5.8, "• 5 NSE Equities (5 Yrs)\n  RELIANCE, TCS, INFY,\n  HDFCBANK, ITC\n• 1,236 Daily Bars\n• 14 Stationary Features\n• Train / Val / Test (70/15/15)", fontsize=9, ha="center", color=SLATE)
    
    # Arrow 1 -> 2
    ax.annotate("", xy=(3.8, 5.3), xytext=(3.3, 5.3), arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
    
    # Block 2: Frozen LSTM Market Encoder
    rect2 = patches.FancyBboxPatch((3.9, 3.8), 3.0, 3.0, boxstyle="round,pad=0.2", facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(5.4, 6.5, "2. LSTM Market Encoder", fontsize=11, fontweight="bold", ha="center", color="#744210")
    ax.text(5.4, 5.8, "• 2-Layer LSTM (Hidden: 64)\n• 30-Day Sequence Input\n• Trained on Train Set (MSE)\n• 55.53% Directional Acc.\n• Frozen Weights\n• 32-D Latent State h_t", fontsize=9, ha="center", color=SLATE)
    
    # Arrow 2 -> 3
    ax.annotate("", xy=(7.4, 5.3), xytext=(6.9, 5.3), arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
    
    # Block 3: Hybrid 49-D State Space
    rect3 = patches.FancyBboxPatch((7.5, 3.8), 3.1, 3.0, boxstyle="round,pad=0.2", facecolor="#F0FFF4", edgecolor=ACCENT_GREEN, linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(9.05, 6.5, "3. 49-D Observation", fontsize=11, fontweight="bold", ha="center", color="#22543D")
    ax.text(9.05, 5.8, "• 32-D Latent Embedding\n• 14-D Current Indicators\n• 3-D Portfolio Context:\n  - Cash Ratio\n  - Position Ratio\n  - Unrealized PnL\n• Time-Aligned at Close t", fontsize=9, ha="center", color=SLATE)
    
    # Arrow 3 -> 4
    ax.annotate("", xy=(11.1, 5.3), xytext=(10.6, 5.3), arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
    
    # Block 4: PPO Agent
    rect4 = patches.FancyBboxPatch((11.2, 3.8), 3.3, 3.0, boxstyle="round,pad=0.2", facecolor="#FAF5FF", edgecolor="#805AD5", linewidth=1.5)
    ax.add_patch(rect4)
    ax.text(12.85, 6.5, "4. Decoupled PPO Agent", fontsize=11, fontweight="bold", ha="center", color="#44337A")
    ax.text(12.85, 5.8, "• Decoupled Actor-Critic\n• Actor: Categorical(3)\n  {HOLD, BUY, SELL}\n• Critic: Scalar V(s)\n• Clipped Surrogate (eps=0.2)\n• GAE (gamma=0.99, lam=0.95)", fontsize=9, ha="center", color=SLATE)
    
    # Lower Execution Loop: Gymnasium Environment
    rect_env = patches.FancyBboxPatch((3.9, 0.5), 6.7, 2.5, boxstyle="round,pad=0.2", facecolor="#FFF5F5", edgecolor=ACCENT_RED, linewidth=1.5)
    ax.add_patch(rect_env)
    ax.text(7.25, 2.6, "Gymnasium MarketTradingEnv (Execution & Portfolio Engine)", fontsize=11, fontweight="bold", ha="center", color="#742A2A")
    ax.text(7.25, 1.6, "• Long-Only Single Position Execution | Starting Capital: ₹100,000\n• Decision Epoch: Close t | Execution Epoch: Next-Day Open t+1 | Valuation: Close t+1\n• Strict Transaction Friction: 0.10% (10 bps) deducted on execution\n• Reward Signal: Logarithmic Return R_t = ln(V_{t+1} / V_t)", fontsize=9, ha="center", color=SLATE)
    
    # Action Arrow: Agent -> Env
    ax.annotate("Action a_t", xy=(10.6, 2.0), xytext=(12.85, 3.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="#805AD5", lw=2))
    
    # Feedback Arrow: Env -> State / Reward
    ax.annotate("s_{t+1}, R_t", xy=(7.5, 4.3), xytext=(5.5, 3.0),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=ACCENT_GREEN, lw=2))
                
    plt.tight_layout()
    path = os.path.join(arch_dir, "01_overall_system_architecture.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 2. End-to-End Project Workflow Diagram
# -------------------------------------------------------------
def make_workflow_diagram():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(7.5, 5.6, "NeuroTrader End-to-End Research & Engineering Workflow", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    
    steps = [
        ("Phase 1-5\nData Pipeline", "• Raw OHLCV (5 Equities)\n• Data Cleaning & Alignment\n• 14 Rolling Indicators\n• 70/15/15 Chrono Splits\n• Strict Train-only Scaler", "#EBF8FF", ACCENT_BLUE),
        ("Phase 6-9\nRepresentation", "• 2-Layer LSTM Encoder\n• 30-Day Window Input\n• MSE Loss Optimization\n• 55.53% Directional Acc\n• 32-D Embeddings Saved", "#FEFCBF", "#D69E2E"),
        ("Phase 10\nEnvironment", "• Gymnasium Interface\n• Next-Day Open Execution\n• 10 bps Transaction Cost\n• 6 Accounting Unit Tests\n• Baselines Established", "#F0FFF4", ACCENT_GREEN),
        ("Phase 11\nPPO Training", "• Decoupled Actor-Critic\n• 35 Iterations (140k steps)\n• GAE + Clipped Objective\n• Val Checkpoint Selection\n• Best Model @ Iter 4", "#FAF5FF", "#805AD5"),
        ("Phase 12\nOut-of-Sample", "• Untouched Test Data\n• Benchmark Comparison\n• PPO: +0.34% Net Return\n• Buy-and-Hold: -15.74%\n• 100% Bitwise Verified", "#FFF5F5", ACCENT_RED),
    ]
    
    box_w = 2.4
    for i, (title, desc, bg, edge) in enumerate(steps):
        x = 0.5 + i * 2.9
        rect = patches.FancyBboxPatch((x, 0.8), box_w, 4.2, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, 4.6, title, fontsize=11, fontweight="bold", ha="center", color=NAVY)
        ax.text(x + box_w/2, 2.7, desc, fontsize=8.5, ha="center", color=SLATE)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + 0.45, 2.9), xytext=(x + box_w + 0.05, 2.9),
                        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
                        
    plt.tight_layout()
    path = os.path.join(arch_dir, "02_end_to_end_project_workflow.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 3. LSTM Encoder Architecture Diagram
# -------------------------------------------------------------
def make_lstm_architecture():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(6, 6.6, "Frozen 2-Layer LSTM Market-State Encoder", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    ax.text(6, 6.2, "Input: Sequence of 30 Timesteps x 14 Technical Indicators -> Output: 32-D Market Latent Vector", fontsize=10, fontstyle="italic", ha="center", color=SLATE)
    
    # Sequence Boxes
    ax.text(1.5, 4.8, "Input Sequence (T=30)", fontsize=10, fontweight="bold", ha="center", color=NAVY)
    for i, t in enumerate(["t-29", "t-28", "...", "t-1", "t"]):
        rect = patches.Rectangle((0.5 + i * 0.9, 3.8), 0.7, 0.7, facecolor="#E2E8F0", edgecolor=SLATE)
        ax.add_patch(rect)
        ax.text(0.85 + i * 0.9, 4.15, f"x_{{{t}}}", fontsize=9, ha="center", va="center")
        ax.annotate("", xy=(0.85 + i * 0.9, 3.1), xytext=(0.85 + i * 0.9, 3.8),
                    arrowprops=dict(arrowstyle="->", color=SLATE))
                    
    # LSTM Layer 1
    rect_l1 = patches.FancyBboxPatch((0.4, 2.3), 4.3, 0.7, boxstyle="round,pad=0.1", facecolor="#EBF8FF", edgecolor=ACCENT_BLUE, lw=1.5)
    ax.add_patch(rect_l1)
    ax.text(2.55, 2.65, "LSTM Layer 1 (Input: 14 -> Hidden: 64, Dropout: 0.2)", fontsize=9, fontweight="bold", ha="center", color=NAVY)
    
    # Arrow L1 -> L2
    ax.annotate("", xy=(2.55, 1.5), xytext=(2.55, 2.3), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
    
    # LSTM Layer 2
    rect_l2 = patches.FancyBboxPatch((0.4, 0.8), 4.3, 0.7, boxstyle="round,pad=0.1", facecolor="#EBF8FF", edgecolor=ACCENT_BLUE, lw=1.5)
    ax.add_patch(rect_l2)
    ax.text(2.55, 1.15, "LSTM Layer 2 (Hidden: 64 -> Hidden: 64, Dropout: 0.2)", fontsize=9, fontweight="bold", ha="center", color=NAVY)
    
    # Extraction Arrow
    ax.annotate("", xy=(6.0, 1.15), xytext=(4.7, 1.15), arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=2, headwidth=8))
    ax.text(5.35, 1.4, "h_t (Step 30)", fontsize=9, fontweight="bold", ha="center", color=NAVY)
    
    # Projection Layer (Dense 64 -> 32)
    rect_proj = patches.FancyBboxPatch((6.0, 0.6), 2.5, 1.1, boxstyle="round,pad=0.1", facecolor="#FEFCBF", edgecolor="#D69E2E", lw=1.5)
    ax.add_patch(rect_proj)
    ax.text(7.25, 1.3, "Linear Projection", fontsize=10, fontweight="bold", ha="center", color="#744210")
    ax.text(7.25, 0.9, "Dense(64 -> 32) + Tanh", fontsize=9, ha="center", color=SLATE)
    
    # Arrow Projection -> Embedding
    ax.annotate("", xy=(9.2, 1.15), xytext=(8.5, 1.15), arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=2, headwidth=8))
    
    # Latent Vector Box
    rect_emb = patches.FancyBboxPatch((9.2, 0.4), 2.4, 2.6, boxstyle="round,pad=0.15", facecolor="#F0FFF4", edgecolor=ACCENT_GREEN, lw=2)
    ax.add_patch(rect_emb)
    ax.text(10.4, 2.6, "32-D Latent State", fontsize=11, fontweight="bold", ha="center", color="#22543D")
    ax.text(10.4, 2.2, "(Market Embedding)", fontsize=9, fontstyle="italic", ha="center", color=SLATE)
    ax.text(10.4, 1.3, "• Precomputed & Frozen\n• Dense Representation\n• Directional Acc: 55.53%\n• Passed to PPO Env\n  as Core Market State", fontsize=8.5, ha="center", color=SLATE)
    
    # Regression Head (Historical)
    ax.annotate("Historical Training Head\nLinear(32 -> 1) MSE Loss", xy=(7.25, 3.8), xytext=(7.25, 1.7),
                arrowprops=dict(arrowstyle="->", linestyle="--", color="gray"))
    rect_reg = patches.FancyBboxPatch((6.0, 3.8), 2.5, 0.8, boxstyle="round,pad=0.1", facecolor="#FFF5F5", edgecolor="#E53E3E", linestyle="--")
    ax.add_patch(rect_reg)
    ax.text(7.25, 4.2, "Next-Day Return r_{t+1}\n(MAE: 0.012997)", fontsize=8.5, ha="center", color="#742A2A")
    
    plt.tight_layout()
    path = os.path.join(arch_dir, "03_lstm_encoder_architecture.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 4. 49-D Hybrid Observation Design Diagram
# -------------------------------------------------------------
def make_observation_diagram():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(6.5, 5.5, "Composition of the 49-Dimensional Hybrid Observation Vector", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    ax.text(6.5, 5.1, "Combining Temporal Embeddings, Explicit Technical Indicators, and Dynamic Portfolio Context", fontsize=10, fontstyle="italic", ha="center", color=SLATE)
    
    # 3 Main Component Blocks
    components = [
        ("32-D Market Embedding", "Indices 0 - 31", "#FEFCBF", "#D69E2E", [
            "• Extracted from frozen LSTM layer 2",
            "• Encapsulates 30-day temporal dependencies",
            "• Represents multi-day price momentum and regime",
            "• Fixed representation (gradient-isolated)"
        ], 3.8),
        ("14-D Technical Indicators", "Indices 32 - 45", "#EBF8FF", ACCENT_BLUE, [
            "• Price to Moving Average: SMA5, SMA10, SMA20, SMA50",
            "• Rolling Return Ratios: Ret1, Ret5, Ret20",
            "• Volatility & Range: RollingStd10, High/Low Ratio",
            "• Volume Signals: Vol_SMA5, Vol_SMA20, Volume_Ratio",
            "• Momentum & Mean Reversion: RSI14, MACD Histogram"
        ], 4.2),
        ("3-D Portfolio Context", "Indices 46 - 48", "#F0FFF4", ACCENT_GREEN, [
            "• Cash Ratio = Cash / Total Portfolio Value",
            "• Position Ratio = (Shares * Close) / Total Value",
            "• Unrealized PnL = (Close - Entry) / Entry",
            "• Informs agent of current exposure and risk"
        ], 3.8)
    ]
    
    x_positions = [0.5, 4.6, 9.0]
    widths = [3.8, 4.1, 3.5]
    
    for i, (title, sub, bg, border, bullets, w) in enumerate(components):
        x = x_positions[i]
        rect = patches.FancyBboxPatch((x, 0.8), widths[i], 3.8, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=border, lw=2)
        ax.add_patch(rect)
        ax.text(x + widths[i]/2, 4.2, title, fontsize=11, fontweight="bold", ha="center", color=NAVY)
        ax.text(x + widths[i]/2, 3.8, sub, fontsize=9, fontstyle="italic", ha="center", color=SLATE)
        text_str = "\n".join(bullets)
        ax.text(x + 0.2, 2.2, text_str, fontsize=8.5, ha="left", va="center", color=SLATE)
        
    plt.tight_layout()
    path = os.path.join(arch_dir, "04_49d_hybrid_observation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 5. PPO Actor-Critic Architecture Diagram
# -------------------------------------------------------------
def make_ppo_architecture():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(7, 6.6, "Decoupled Actor-Critic Trading Architecture (ActorCriticTradingNetwork)", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    ax.text(7, 6.2, "Orthogonal Initialization | LayerNorm Regularization | Tanh Non-Linearities", fontsize=10, fontstyle="italic", ha="center", color=SLATE)
    
    # Input State
    rect_in = patches.FancyBboxPatch((5.5, 4.9), 3.0, 0.8, boxstyle="round,pad=0.1", facecolor="#EDF2F7", edgecolor=NAVY, lw=2)
    ax.add_patch(rect_in)
    ax.text(7.0, 5.3, "State Vector s_t (49-D)", fontsize=11, fontweight="bold", ha="center", color=NAVY)
    
    # Split arrows
    ax.annotate("", xy=(3.5, 4.2), xytext=(6.2, 4.9), arrowprops=dict(facecolor=ACCENT_BLUE, edgecolor=ACCENT_BLUE, width=1.5, headwidth=6))
    ax.annotate("", xy=(10.5, 4.2), xytext=(7.8, 4.9), arrowprops=dict(facecolor="#805AD5", edgecolor="#805AD5", width=1.5, headwidth=6))
    
    # ACTOR STREAM
    rect_act = patches.FancyBboxPatch((1.2, 0.6), 4.6, 3.6, boxstyle="round,pad=0.15", facecolor="#EBF8FF", edgecolor=ACCENT_BLUE, lw=2)
    ax.add_patch(rect_act)
    ax.text(3.5, 3.9, "Actor Stream (Policy Head π_θ)", fontsize=11, fontweight="bold", ha="center", color=NAVY)
    ax.text(3.5, 2.7, "• Linear(49 -> 64) + LayerNorm + Tanh\n  [Init Gain = sqrt(2), Bias = 0.0]\n• Linear(64 -> 64) + LayerNorm + Tanh\n  [Init Gain = sqrt(2), Bias = 0.0]\n• Linear(64 -> 3) Policy Logits\n  [Init Gain = 0.01 for near-uniform start]\n• Categorical Distribution:\n  {0: HOLD, 1: BUY, 2: SELL}", fontsize=8.5, ha="center", color=SLATE)
    
    # CRITIC STREAM
    rect_crit = patches.FancyBboxPatch((8.2, 0.6), 4.6, 3.6, boxstyle="round,pad=0.15", facecolor="#FAF5FF", edgecolor="#805AD5", lw=2)
    ax.add_patch(rect_crit)
    ax.text(10.5, 3.9, "Critic Stream (Value Function V_φ)", fontsize=11, fontweight="bold", ha="center", color="#44337A")
    ax.text(10.5, 2.7, "• Linear(49 -> 64) + LayerNorm + Tanh\n  [Init Gain = sqrt(2), Bias = 0.0]\n• Linear(64 -> 64) + LayerNorm + Tanh\n  [Init Gain = sqrt(2), Bias = 0.0]\n• Linear(64 -> 1) Scalar State-Value\n  [Init Gain = 1.0, Bias = 0.0]\n• Computes Baseline V(s_t)\n  for GAE Advantage Estimation", fontsize=8.5, ha="center", color=SLATE)
    
    plt.tight_layout()
    path = os.path.join(arch_dir, "05_ppo_actor_critic_architecture.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 6. Trading Environment Execution Sequence Diagram
# -------------------------------------------------------------
def make_execution_sequence():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(7, 5.5, "MarketTradingEnv Execution & Accounting Sequence", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    ax.text(7, 5.1, "Eliminating Lookahead Bias: Decision at Close t -> Execution at Open t+1 -> Valuation at Close t+1", fontsize=10, fontstyle="italic", ha="center", color=SLATE)
    
    steps = [
        ("Step 1: Observation", "At Day t Close:\n\n• Agent observes state s_t\n• Consists of h_t (LSTM),\n  features x_t, and port_t\n• Evaluates policy logits\n• Selects action a_t in {0,1,2}", "#EBF8FF", ACCENT_BLUE),
        ("Step 2: Execution", "At Day t+1 Open:\n\n• Market opens at Open_{t+1}\n• If BUY (a=1) & cash > 0:\n  Buy shares, pay 0.1% fee\n• If SELL (a=2) & shares > 0:\n  Sell shares, pay 0.1% fee\n• If HOLD (a=0): No trade", "#FEFCBF", "#D69E2E"),
        ("Step 3: Valuation", "At Day t+1 Close:\n\n• Market closes at Close_{t+1}\n• Mark-to-market valuation:\n  V_{t+1} = Cash + Shares * Close_{t+1}\n• Reward computed:\n  R_t = ln(V_{t+1} / V_t)\n• Next state s_{t+1} generated", "#F0FFF4", ACCENT_GREEN)
    ]
    
    for i, (title, desc, bg, edge) in enumerate(steps):
        x = 0.8 + i * 4.3
        rect = patches.FancyBboxPatch((x, 0.8), 3.8, 3.8, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=edge, lw=2)
        ax.add_patch(rect)
        ax.text(x + 1.9, 4.2, title, fontsize=11, fontweight="bold", ha="center", color=NAVY)
        ax.text(x + 1.9, 2.5, desc, fontsize=8.5, ha="center", color=SLATE)
        if i < 2:
            ax.annotate("", xy=(x + 4.15, 2.7), xytext=(x + 3.85, 2.7),
                        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
                        
    plt.tight_layout()
    path = os.path.join(arch_dir, "06_trading_environment_execution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# -------------------------------------------------------------
# 7. Training, Validation & Testing Workflow
# -------------------------------------------------------------
def make_data_partitions_flow():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    
    ax.text(7, 5.5, "Strict Chronological Partitioning & Pipeline Containment", fontsize=15, fontweight="bold", ha="center", color=NAVY)
    ax.text(7, 5.1, "Zero Data Leakage | Scaler Fit Strictly on Train | Test Partition Evaluated Only Once Post-Training", fontsize=10, fontstyle="italic", ha="center", color=SLATE)
    
    splits = [
        ("TRAIN SET (70%)", "2022-03-02 to 2025-05-29\n(800 Days/Stock = 4,000 Trans.)", "• StandardScaler Fit\n• LSTM Supervised Training\n• 32-D Embedding Extraction\n• PPO Rollout Collection\n• 4,480 Mini-Batch Updates", "#EBF8FF", ACCENT_BLUE),
        ("VALIDATION SET (15%)", "2025-05-30 to 2026-02-11\n(177 Days/Stock = 885 Trans.)", "• Frozen LSTM Inference\n• Periodic Policy Evaluation\n• Active Policy Ranking\n• Model Selection (Sharpe)\n• Iteration 4 Selected", "#FEFCBF", "#D69E2E"),
        ("TEST SET (15%)", "2026-02-12 to 2026-11-04\n(179 Days/Stock = 895 Trans.)", "• Strictly Untouched\n• Evaluated ONLY Once\n• Benchmark Comparison:\n  PPO: +0.34% vs B&H: -15.74%\n• 100% Bitwise Verified", "#FFF5F5", ACCENT_RED),
    ]
    
    for i, (title, dates, desc, bg, edge) in enumerate(splits):
        x = 0.8 + i * 4.3
        rect = patches.FancyBboxPatch((x, 0.8), 3.8, 3.8, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=edge, lw=2)
        ax.add_patch(rect)
        ax.text(x + 1.9, 4.2, title, fontsize=11, fontweight="bold", ha="center", color=NAVY)
        ax.text(x + 1.9, 3.7, dates, fontsize=8.5, fontstyle="italic", ha="center", color=SLATE)
        ax.text(x + 1.9, 2.2, desc, fontsize=8.5, ha="center", color=SLATE)
        if i < 2:
            ax.annotate("", xy=(x + 4.15, 2.7), xytext=(x + 3.85, 2.7),
                        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6))
                        
    plt.tight_layout()
    path = os.path.join(arch_dir, "07_training_validation_testing_flow.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# Run architecture diagram generators
make_overall_architecture()
make_workflow_diagram()
make_lstm_architecture()
make_observation_diagram()
make_ppo_architecture()
make_execution_sequence()
make_data_partitions_flow()

# -------------------------------------------------------------
# RESULT VISUALIZATIONS
# -------------------------------------------------------------
# 1. LSTM Performance Comparison
def make_lstm_results_plot():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    
    # Left: Directional Accuracy Comparison
    ax1 = axes[0]
    models = ["Majority Direction\nBaseline", "Initial LSTM\n(Huber Loss)", "Directional BCE\n(Classification)", "Focal Loss\n(Classification)", "Improved LSTM\n(MSE Loss, Ours)"]
    accs = [44.13, 44.13, 52.85, 54.19, 55.53]
    colors = ["#CBD5E0", "#CBD5E0", "#90CDF4", "#63B3ED", "#3182CE"]
    
    bars = ax1.bar(models, accs, color=colors, width=0.55, edgecolor=NAVY, lw=1)
    ax1.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Random Guess (50.0%)")
    ax1.set_ylim(35, 60)
    ax1.set_ylabel("Directional Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Test Directional Accuracy Comparison", fontsize=11, fontweight="bold", color=NAVY)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="lower right")
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.6, f"{h:.2f}%", ha="center", fontsize=9, fontweight="bold")
        
    # Right: Filtered Confidence Directional Accuracy
    ax2 = axes[1]
    tiers = ["Full Test Set\n(No Filter)", "Confidence Filtered\n(tau = 0.000562)"]
    f_accs = [55.53, 58.54]
    f_covs = ["Coverage: 100.0% (895/895)", "Coverage: 36.65% (328/895)"]
    f_colors = ["#3182CE", "#38A169"]
    bars2 = ax2.bar(tiers, f_accs, color=f_colors, width=0.45, edgecolor=NAVY, lw=1)
    ax2.set_ylim(45, 65)
    ax2.set_ylabel("Directional Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Filtered Directional Accuracy at High Confidence", fontsize=11, fontweight="bold", color=NAVY)
    ax2.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars2):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.6, f"{h:.2f}%\n{f_covs[i]}", ha="center", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    path = os.path.join(res_dir, "01_lstm_performance_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# 2. PPO Strategy Performance Comparison
def make_strategy_comparison_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    
    strats = ["Always HOLD\n(Cash)", "Buy-and-Hold\n(Passive)", "MA Crossover\n(10/50 SMA)", "PPO Agent\n(Ours)"]
    colors = ["#A0AEC0", "#E53E3E", "#ED8936", "#38A169"]
    
    # 1. Total Return
    ax1 = axes[0]
    rets = [0.0, -15.74, -9.63, 0.34]
    b1 = ax1.bar(strats, rets, color=colors, width=0.55, edgecolor=NAVY, lw=1)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Total Return (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Out-of-Sample Return % (Test Set)", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_ylim(-20, 5)
    ax1.grid(axis="y", alpha=0.3)
    for b in b1:
        h = b.get_height()
        y_pos = h + 0.6 if h >= 0 else h - 1.8
        ax1.text(b.get_x() + b.get_width()/2, y_pos, f"{h:+.2f}%", ha="center", fontsize=9, fontweight="bold")
        
    # 2. Sharpe Ratio
    ax2 = axes[1]
    sharpes = [0.0, -0.75, -1.54, 0.13]
    b2 = ax2.bar(strats, sharpes, color=colors, width=0.55, edgecolor=NAVY, lw=1)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Annualized Sharpe Ratio", fontsize=10, fontweight="bold")
    ax2.set_title("Risk-Adjusted Performance (Sharpe)", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_ylim(-2.0, 0.6)
    ax2.grid(axis="y", alpha=0.3)
    for b in b2:
        h = b.get_height()
        y_pos = h + 0.06 if h >= 0 else h - 0.18
        ax2.text(b.get_x() + b.get_width()/2, y_pos, f"{h:+.2f}", ha="center", fontsize=9, fontweight="bold")
        
    # 3. Maximum Drawdown
    ax3 = axes[2]
    dds = [0.0, -26.34, -13.48, -13.03]
    b3 = ax3.bar(strats, dds, color=colors, width=0.55, edgecolor=NAVY, lw=1)
    ax3.set_ylabel("Maximum Drawdown (%)", fontsize=10, fontweight="bold")
    ax3.set_title("Downside Risk (Max Drawdown)", fontsize=11, fontweight="bold", color=NAVY)
    ax3.set_ylim(-30, 2)
    ax3.grid(axis="y", alpha=0.3)
    for b in b3:
        h = b.get_height()
        y_pos = h + 0.6 if h >= 0 else h - 2.5
        ax3.text(b.get_x() + b.get_width()/2, y_pos, f"{h:.2f}%", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    path = os.path.join(res_dir, "04_strategy_performance_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# 3. Per Asset Out-of-Sample Return Comparison
def make_per_asset_plot():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
    bnh = [-10.91, -15.20, -21.12, -20.61, -11.84]
    ppo = [3.72, -9.27, 15.72, -2.99, -5.49]
    
    x = np.arange(len(symbols))
    width = 0.35
    
    b_bnh = ax.bar(x - width/2, bnh, width, label="Passive Buy & Hold", color="#E53E3E", alpha=0.8, edgecolor=NAVY)
    b_ppo = ax.bar(x + width/2, ppo, width, label="PPO Trading Agent (Ours)", color="#38A169", edgecolor=NAVY)
    
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Return % (Net of Fees)", fontsize=10, fontweight="bold")
    ax.set_title("Per-Asset Out-of-Sample Test Return: PPO vs. Buy-and-Hold", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylim(-26, 20)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    for b in b_bnh:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h - 2.0, f"{h:+.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#742A2A")
    for b in b_ppo:
        h = b.get_height()
        y_pos = h + 0.8 if h >= 0 else h - 2.0
        ax.text(b.get_x() + b.get_width()/2, y_pos, f"{h:+.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#22543D")
        
    plt.tight_layout()
    path = os.path.join(res_dir, "02_ppo_equity_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# 4. Training Convergence and Validation Trajectory Plot
def make_training_validation_plot():
    with open("reports/reinforcement_learning/ppo_reproducibility_results.json", "r") as f:
        res = json.load(f)
        
    # Load training history from ppo_training_results.json
    with open("reports/reinforcement_learning/ppo_training_results.json", "r") as f:
        orig = json.load(f)
    hist = orig["training_history"]
    
    iters = [h["iteration"] for h in hist]
    pi_losses = [h["loss_pi"] for h in hist]
    vf_losses = [h["loss_vf"] for h in hist]
    val_sharpes = [h["val_sharpe"] for h in hist]
    val_rets = [h["val_return%"] for h in hist]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    
    # 1. Policy Loss
    ax1 = axes[0, 0]
    ax1.plot(iters, pi_losses, color="#3182CE", lw=2)
    ax1.set_title("PPO Policy Surrogate Loss (L_CLIP)", fontweight="bold", color=NAVY)
    ax1.set_xlabel("Iteration")
    ax1.grid(True, alpha=0.3)
    
    # 2. Value Function Loss
    ax2 = axes[0, 1]
    ax2.plot(iters, vf_losses, color="#805AD5", lw=2)
    ax2.set_title("Critic Value Loss (L_VF)", fontweight="bold", color=NAVY)
    ax2.set_xlabel("Iteration")
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Sharpe
    ax3 = axes[1, 0]
    ax3.plot(iters, val_sharpes, color="#D69E2E", lw=2)
    ax3.scatter([4], [val_sharpes[3]], color="red", s=80, zorder=5, label="Best Checkpoint (Iter 4)")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title("Validation Sharpe Ratio Trajectory", fontweight="bold", color=NAVY)
    ax3.set_xlabel("Iteration")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Return
    ax4 = axes[1, 1]
    ax4.plot(iters, val_rets, color="#38A169", lw=2)
    ax4.scatter([4], [val_rets[3]], color="red", s=80, zorder=5, label="Best Checkpoint (Iter 4: -0.60%)")
    ax4.axhline(-9.58, color="black", linestyle="--", alpha=0.6, label="Val Market Benchmark (-9.58%)")
    ax4.set_title("Validation Return % Trajectory", fontweight="bold", color=NAVY)
    ax4.set_xlabel("Iteration")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(res_dir, "05_training_validation_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

# 5. Reproducibility Comparison Plot
def make_reproducibility_plot():
    # Copy or re-save ppo_reproducibility_curves.png
    src = "reports/reinforcement_learning/ppo_reproducibility_curves.png"
    dst = os.path.join(res_dir, "06_reproducibility_comparison.png")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")

# 6. Portfolio Drawdown Comparison
def make_drawdown_plot():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    
    strats = ["Always HOLD (Cash)", "PPO Agent (Ours)", "MA Crossover", "Buy & Hold (Passive)"]
    dds = [0.0, -13.03, -13.48, -26.34]
    colors = ["#A0AEC0", "#38A169", "#ED8936", "#E53E3E"]
    
    bars = ax.barh(strats, dds, color=colors, height=0.5, edgecolor=NAVY)
    ax.axvline(0, color="black", linestyle="-", lw=1)
    ax.set_xlabel("Maximum Drawdown %", fontsize=10, fontweight="bold")
    ax.set_title("Out-of-Sample Maximum Portfolio Drawdown Comparison", fontsize=11, fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.3)
    for b in bars:
        w = b.get_width()
        x_pos = w - 1.8 if w < 0 else 0.5
        ax.text(x_pos, b.get_y() + b.get_height()/2, f"{w:.2f}%", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    path = os.path.join(res_dir, "03_portfolio_drawdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {path}")

make_lstm_results_plot()
make_strategy_comparison_plot()
make_per_asset_plot()
make_training_validation_plot()
make_reproducibility_plot()
make_drawdown_plot()

print("ALL ARCHITECTURE DIAGRAMS AND RESULT VISUALIZATIONS GENERATED SUCCESSFULLY!")
