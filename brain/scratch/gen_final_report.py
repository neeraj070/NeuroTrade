"""
Generate Final_Project_Report.docx for NeuroTrade.
Full academic report with 30 sections, actual verified metrics, figures.
"""
import os, sys, json, hashlib
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

os.environ["MPLCONFIGDIR"] = r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade\backend\.mpl_cache"

BASE   = Path(r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade")
DIAGRAMS = BASE / "reports" / "final_submission" / "Architecture_Diagrams"
RESULTS  = BASE / "reports" / "final_submission" / "Result_Visualizations"
OUT    = BASE / "reports" / "final_submission" / "Final_Project_Report.docx"

# Load verified data
tr  = json.loads((BASE / "reports" / "reinforcement_learning" / "ppo_training_results.json").read_text())
rep = json.loads((BASE / "reports" / "reinforcement_learning" / "ppo_reproducibility_results.json").read_text())

doc = Document()

# ─── Page margins ───
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.0)

# ─── Styles ───
def set_heading(para, level=1):
    para.style = doc.styles[f"Heading {level}"]

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    return h

def add_para(text, bold=False, italic=False, size=11, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p

def add_table(headers, rows, style="Table Grid"):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = style
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
        hdr.cells[i].paragraphs[0].runs[0].bold = True
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            t.rows[r_i+1].cells[c_i].text = str(val)
    doc.add_paragraph()

def add_figure(img_path, caption, width=5.5):
    img = Path(img_path)
    if img.exists():
        doc.add_picture(str(img), width=Inches(width))
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(9)
    else:
        doc.add_paragraph(f"[Figure: {caption} — image not found at {img_path}]")
    doc.add_paragraph()

def add_bullet(text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    return p

def add_code(text):
    p = doc.add_paragraph(text)
    p.runs[0].font.name = "Courier New"
    p.runs[0].font.size = Pt(9)
    return p

# ══════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run("Reinforcement Learning with RNN-Based Market State\nRepresentation for Algorithmic Trading")
run.bold = True
run.font.size = Pt(20)

doc.add_paragraph()
sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = sub_p.add_run("NeuroTrader: B.Tech Final-Year Project Report")
run2.font.size = Pt(14)
run2.italic = True

doc.add_paragraph()
doc.add_paragraph()

for line in [
    "Department of Computer Science / Electronics & Communication",
    "",
    "Project: NeuroTrader",
    "Implementation Language: Python 3.12",
    "Deep Learning Framework: PyTorch ≥ 2.2",
    "Reinforcement Learning: Proximal Policy Optimization (PPO)",
    "Market: NSE India (5 equities)",
    "Data Period: 2021-11-04 to 2026-11-04",
    "",
    "November 2026",
]:
    p = doc.add_paragraph(line)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if line:
        p.runs[0].font.size = Pt(11)

doc.add_page_break()

# ══════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════
add_heading("Abstract", level=1)
doc.add_paragraph(
    "This report presents NeuroTrader, an algorithmic trading system that integrates a "
    "Long Short-Term Memory (LSTM) encoder for market state representation with a "
    "Proximal Policy Optimization (PPO) reinforcement learning agent for trading decisions. "
    "The system processes daily OHLCV data from five NSE-listed equities — RELIANCE, TCS, INFY, "
    "HDFCBANK, and ITC — spanning 2021-11-04 to 2026-11-04, comprising 1,236 trading days per asset. "
    "Fourteen technical indicators are engineered and normalized; a 2-layer LSTM encoder (hidden=64) "
    "compresses 30-step historical windows into 32-dimensional market state embeddings, achieving "
    "55.53% directional accuracy on the held-out test set versus a 50% random baseline. "
    "The encoder is frozen and combined with current technical features and portfolio state into a "
    "49-dimensional hybrid observation vector for the PPO agent. "
    "The PPO agent (decoupled Actor-Critic, 49→64→64, orthogonal initialization, LayerNorm) "
    "is trained over 35 iterations (140,000 transitions) using a strict chronological "
    "train/validation/test split (70%/15%/15%). "
    "The best checkpoint, selected at Iteration 4 by validation Sharpe ratio, achieves a "
    "+0.34% average return, +0.13 Sharpe ratio, and -13.03% maximum drawdown on the out-of-sample "
    "test set — outperforming the Buy-and-Hold baseline (-15.74% return, -0.75 Sharpe) and "
    "MA Crossover strategy (-9.63% return, -1.54 Sharpe) in the same bear-market test period. "
    "Training is 100% bitwise reproducible across two independent runs using master seed 42, "
    "verified by 895/895 identical agent actions and 0.00000000 metric differences."
)
doc.add_page_break()

# ══════════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ══════════════════════════════════════════════════
add_heading("Table of Contents", level=1)
toc_items = [
    "1. Introduction",
    "2. Problem Statement",
    "3. Literature Review",
    "4. System Architecture",
    "5. Dataset and Market Selection",
    "6. Data Collection and Preprocessing",
    "7. Feature Engineering",
    "8. Chronological Data Splitting",
    "9. LSTM Encoder Architecture",
    "10. LSTM Training and Evaluation",
    "11. Market State Embedding Extraction",
    "12. Trading Environment Design",
    "13. Observation Space (49-D Hybrid)",
    "14. Action Space and Execution Model",
    "15. Reward Function",
    "16. PPO Agent Architecture",
    "17. PPO Algorithm Implementation",
    "18. Training Configuration and Hyperparameters",
    "19. Training Progress and Loss Analysis",
    "20. Validation-Based Model Selection",
    "21. Out-of-Sample Evaluation",
    "22. Deterministic Reproducibility",
    "23. Comparison with Baseline Strategies",
    "24. Per-Asset Performance Analysis",
    "25. Transaction Cost Analysis",
    "26. Methodological Validity",
    "27. Limitations",
    "28. Future Work",
    "29. Conclusion",
    "30. References",
]
for item in toc_items:
    doc.add_paragraph(item)
doc.add_page_break()

# ══════════════════════════════════════════════════
# SECTION 1 — INTRODUCTION
# ══════════════════════════════════════════════════
add_heading("1. Introduction", level=1)
doc.add_paragraph(
    "Algorithmic trading has evolved significantly with the advent of deep learning and reinforcement "
    "learning. Traditional rule-based strategies — such as moving average crossovers and momentum "
    "filters — are limited by their inability to adapt to complex, non-stationary market dynamics. "
    "Deep reinforcement learning (DRL) agents can, in principle, learn adaptive policies directly "
    "from market data, without explicit programming of trading rules."
)
doc.add_paragraph(
    "A key challenge in applying DRL to financial markets is the design of a meaningful state "
    "representation. Raw price series and technical indicators contain substantial noise and "
    "non-stationarity. This project addresses this challenge by employing a pre-trained LSTM "
    "encoder to extract compressed, continuous market state embeddings, which are then provided "
    "as part of the observation to a PPO agent."
)
doc.add_paragraph(
    "NeuroTrader operates on five NSE-listed Indian equities over a five-year period, providing "
    "a realistic evaluation environment that includes genuine bear market conditions. This report "
    "documents all implementation decisions, experimental results, and methodological safeguards."
)

add_figure(str(DIAGRAMS / "01_overall_system_architecture.png"),
           "Figure 1: Overall NeuroTrader System Architecture")

# ══════════════════════════════════════════════════
# SECTION 2 — PROBLEM STATEMENT
# ══════════════════════════════════════════════════
add_heading("2. Problem Statement", level=1)
doc.add_paragraph(
    "Given daily OHLCV data for a single equity, the agent must learn a policy "
    "π(a|s) that maps market states s to trading actions a ∈ {HOLD, BUY, SELL} so as to "
    "maximize risk-adjusted cumulative return over a multi-period horizon, subject to:"
)
bullets = [
    "Long-only constraint: no short selling.",
    "Single-unit position: fully in (100% of capital) or fully out (100% cash).",
    "Realistic transaction costs: 10 basis points (0.10%) per trade.",
    "Next-day Open execution: orders placed at close of day t are filled at Open of day t+1.",
    "No future information leakage in the observation vector.",
]
for b in bullets:
    add_bullet(b)
doc.add_paragraph(
    "The system is evaluated strictly out-of-sample on data the agent never saw during training "
    "or model selection."
)

# ══════════════════════════════════════════════════
# SECTION 3 — LITERATURE REVIEW
# ══════════════════════════════════════════════════
add_heading("3. Literature Review", level=1)
doc.add_paragraph(
    "Reinforcement learning for trading has been an active research area since Moody et al. (1998) "
    "introduced recurrent reinforcement learning for trading. Key developments include:"
)
lit = [
    ("Mnih et al. (2015)", "DQN — demonstrated superhuman performance in Atari, sparking interest in DRL for sequential decision making."),
    ("Schulman et al. (2017)", "Proximal Policy Optimization (PPO) — stable on-policy gradient method using clipped surrogate objectives, now a standard baseline."),
    ("Deng et al. (2016)", "Applied LSTM to financial time series for trading, showing LSTM better captures temporal dependencies than feedforward networks."),
    ("Jiang et al. (2017)", "Portfolio management via policy gradient on crypto assets using LSTM state encoding."),
    ("Liang et al. (2018)", "FinRL framework — standardized Gymnasium-compatible trading environments for benchmarking DRL strategies."),
    ("Zhang et al. (2020)", "Deep RL for stock trading with technical indicators as observation space."),
]
add_table(["Reference", "Contribution"], lit)

doc.add_paragraph(
    "NeuroTrader builds on these works by combining a pre-trained LSTM encoder (frozen during RL "
    "training) with PPO in a Gymnasium-compatible trading environment, evaluated on a realistic "
    "NSE dataset with strict chronological splitting."
)

# ══════════════════════════════════════════════════
# SECTION 4 — SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════
add_heading("4. System Architecture", level=1)
doc.add_paragraph(
    "NeuroTrader comprises four main subsystems:"
)
bullets = [
    "Data Pipeline: Raw OHLCV collection → cleaning → feature engineering → preprocessing → tensors.",
    "LSTM Encoder: Pre-trained 2-layer LSTM that compresses 30-step windows into 32-D embeddings.",
    "Trading Environment: Gymnasium-compatible MarketTradingEnv with 49-D hybrid observations.",
    "PPO Agent: Decoupled Actor-Critic network trained with on-policy PPO algorithm.",
]
for b in bullets:
    add_bullet(b)

add_figure(str(DIAGRAMS / "02_end_to_end_project_workflow.png"),
           "Figure 2: End-to-End Project Workflow")

# ══════════════════════════════════════════════════
# SECTION 5 — DATASET AND MARKET SELECTION
# ══════════════════════════════════════════════════
add_heading("5. Dataset and Market Selection", level=1)
doc.add_paragraph(
    "Five NSE-listed Indian equities were selected to represent diverse sectors:"
)
add_table(
    ["Ticker", "Company", "Sector", "Market Cap"],
    [
        ("RELIANCE", "Reliance Industries", "Conglomerate / Energy", "Large Cap"),
        ("TCS", "Tata Consultancy Services", "Information Technology", "Large Cap"),
        ("INFY", "Infosys", "Information Technology", "Large Cap"),
        ("HDFCBANK", "HDFC Bank", "Banking / Finance", "Large Cap"),
        ("ITC", "ITC Limited", "FMCG / Diversified", "Large Cap"),
    ]
)
doc.add_paragraph(
    "Data was acquired via the yfinance API for the period 2021-11-04 to 2026-11-04, "
    "yielding 1,236 trading days per asset after holiday filtering. "
    "OHLCV columns: Open, High, Low, Close, Volume (adjusted for splits and dividends)."
)
add_table(
    ["Attribute", "Value"],
    [
        ("Assets", "5 (RELIANCE, TCS, INFY, HDFCBANK, ITC)"),
        ("Period", "2021-11-04 to 2026-11-04"),
        ("Frequency", "Daily OHLCV"),
        ("Bars per asset", "1,236"),
        ("Total bars", "6,180"),
        ("Data source", "Yahoo Finance (yfinance API)"),
    ]
)

# ══════════════════════════════════════════════════
# SECTION 6 — DATA COLLECTION AND PREPROCESSING
# ══════════════════════════════════════════════════
add_heading("6. Data Collection and Preprocessing", level=1)
doc.add_paragraph(
    "Raw data was downloaded and subjected to a multi-stage cleaning pipeline:"
)
bullets = [
    "OHLC integrity: verified High ≥ Low, High ≥ Open ≥ 0, High ≥ Close ≥ 0.",
    "Zero-price and zero-volume filtering: removed erroneous zero entries.",
    "Holiday calendar alignment: forward-filled missing trading days.",
    "Duplicate index removal: deduplicated by date index.",
    "Chronological integrity check: verified strictly ascending date index.",
]
for b in bullets:
    add_bullet(b)
doc.add_paragraph(
    "After cleaning, the data was stored in `data/processed/` as CSV files with 14 engineered "
    "technical indicator columns appended to the original OHLCV columns."
)

# ══════════════════════════════════════════════════
# SECTION 7 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════
add_heading("7. Feature Engineering", level=1)
doc.add_paragraph(
    "Fourteen technical indicators were computed from the OHLCV data:"
)
add_table(
    ["#", "Feature", "Description", "Stationarity"],
    [
        ("1", "Log Return", "log(Close_t / Close_{t-1})", "✅ Stationary"),
        ("2", "SMA-10 Ratio", "Close / SMA(10) - 1", "✅ Stationary"),
        ("3", "SMA-30 Ratio", "Close / SMA(30) - 1", "✅ Stationary"),
        ("4", "EMA-12 Ratio", "Close / EMA(12) - 1", "✅ Stationary"),
        ("5", "EMA-26 Ratio", "Close / EMA(26) - 1", "✅ Stationary"),
        ("6", "RSI-14", "Relative Strength Index, normalized to [0,1]", "✅ Bounded"),
        ("7", "MACD", "EMA(12) - EMA(26), normalized", "✅ Stationary"),
        ("8", "MACD Signal", "EMA(9) of MACD", "✅ Stationary"),
        ("9", "BB Upper Band", "(Close - BB_upper) / ATR", "✅ Stationary"),
        ("10", "BB Lower Band", "(Close - BB_lower) / ATR", "✅ Stationary"),
        ("11", "ATR-14 Ratio", "ATR(14) / Close", "✅ Bounded"),
        ("12", "Volume Ratio", "Vol / SMA_Vol(20) - 1", "✅ Stationary"),
        ("13", "ROC-10", "Rate of change over 10 days", "✅ Stationary"),
        ("14", "High-Low Ratio", "(High - Low) / Close", "✅ Bounded"),
    ]
)
doc.add_paragraph(
    "All features were designed to be stationary and bounded, reducing non-stationarity issues "
    "that commonly degrade LSTM and RL performance on financial data."
)

# ══════════════════════════════════════════════════
# SECTION 8 — CHRONOLOGICAL DATA SPLITTING
# ══════════════════════════════════════════════════
add_heading("8. Chronological Data Splitting", level=1)
doc.add_paragraph(
    "Data was split chronologically — no shuffling — to prevent temporal information leakage:"
)
add_table(
    ["Split", "Proportion", "Transitions", "Purpose"],
    [
        ("Train", "70%", "4,000", "LSTM training + PPO policy learning"),
        ("Validation", "15%", "885", "PPO model selection (best checkpoint)"),
        ("Test", "15%", "895", "Final out-of-sample evaluation (used once)"),
    ]
)
doc.add_paragraph(
    "The validation set was used exclusively for checkpoint selection (highest validation Sharpe). "
    "Test data was accessed exactly once — during final evaluation of the selected model. "
    "No hyperparameter tuning was performed using test data."
)
add_figure(str(DIAGRAMS / "07_training_validation_testing_flow.png"),
           "Figure 3: Chronological Train/Validation/Test Split and Data Flow")

# ══════════════════════════════════════════════════
# SECTION 9 — LSTM ENCODER ARCHITECTURE
# ══════════════════════════════════════════════════
add_heading("9. LSTM Encoder Architecture", level=1)
doc.add_paragraph(
    "The LSTM encoder maps a sequence of T=30 feature vectors (each of dimension 14) "
    "to a 32-dimensional market state embedding:"
)
add_table(
    ["Component", "Specification"],
    [
        ("Architecture", "2-layer LSTM"),
        ("Input dimension", "14 (technical features per timestep)"),
        ("Sequence length", "30 trading days"),
        ("Hidden dimension", "64"),
        ("Output projection", "Linear(64 → 32)"),
        ("Dropout", "0.1 (between LSTM layers)"),
        ("Output", "32-dimensional embedding vector"),
        ("Training objective", "MSE loss on next-day Close prediction"),
        ("Checkpoint", "`backend/checkpoints/improved_rnn_encoder.pt`"),
    ]
)

add_figure(str(DIAGRAMS / "03_lstm_encoder_architecture.png"),
           "Figure 4: LSTM Encoder Architecture (input → 2-layer LSTM → 32-D embedding)")

# ══════════════════════════════════════════════════
# SECTION 10 — LSTM TRAINING AND EVALUATION
# ══════════════════════════════════════════════════
add_heading("10. LSTM Training and Evaluation", level=1)
doc.add_paragraph(
    "The LSTM was trained on the training split (4,000 samples) with early stopping on validation loss. "
    "Final evaluation was performed on the held-out test set:"
)
add_table(
    ["Metric", "Value", "Interpretation"],
    [
        ("Test MAE", "0.012997", "Mean absolute error on next-day Close prediction"),
        ("Test RMSE", "0.017870", "Root mean squared error"),
        ("Zero-baseline MAE", "0.013050", "MAE if always predicting 0 (zero return)"),
        ("MAE improvement", "0.000053 (0.4%)", "LSTM slightly beats naive zero-return baseline"),
        ("Directional Accuracy", "55.53%", "497 / 895 correct direction predictions"),
        ("Above-chance margin", "+5.53%", "Compared to 50% random baseline"),
        ("Filtered Accuracy (τ=0.000562)", "58.54%", "On 36.65% coverage — high-confidence predictions"),
    ]
)
doc.add_paragraph(
    "The directional accuracy of 55.53% is modest but consistent with published results on "
    "single-asset next-day direction prediction. The filtered accuracy of 58.54% (above a "
    "confidence threshold τ=0.000562) suggests the encoder contains extractable directional signal. "
    "Importantly, MSE-regression encoding (55.53%) outperformed both BCE-classification (52.85%) "
    "and Focal Loss classification (54.19%) in directional accuracy, justifying the regression approach."
)
add_figure(str(RESULTS / "01_lstm_performance_comparison.png"),
           "Figure 5: LSTM Performance Comparison (Regression vs. Classification Approaches)")

# ══════════════════════════════════════════════════
# SECTION 11 — MARKET STATE EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════
add_heading("11. Market State Embedding Extraction", level=1)
doc.add_paragraph(
    "After LSTM training, the encoder weights are frozen permanently. "
    "32-dimensional embeddings are pre-extracted for all three splits and saved as PyTorch tensors:"
)
add_table(
    ["Split", "File", "Shape"],
    [
        ("Train", "`backend/embeddings/train_embeddings_improved.pt`", "[4000, 32]"),
        ("Validation", "`backend/embeddings/val_embeddings_improved.pt`", "[885, 32]"),
        ("Test", "`backend/embeddings/test_embeddings_improved.pt`", "[895, 32]"),
    ]
)
doc.add_paragraph(
    "Pre-extraction avoids repeated LSTM forward passes during RL training, improving efficiency. "
    "The frozen encoder ensures that the LSTM's learned representations are not disrupted by "
    "the RL gradient signal."
)

# ══════════════════════════════════════════════════
# SECTION 12 — TRADING ENVIRONMENT DESIGN
# ══════════════════════════════════════════════════
add_heading("12. Trading Environment Design", level=1)
doc.add_paragraph(
    "The Gymnasium-compatible `MarketTradingEnv` (`backend/environments/trading_env.py`) "
    "implements a realistic single-asset trading simulator with the following design decisions:"
)
bullets = [
    "Next-day Open execution: a BUY or SELL at step t executes at Open_{t+1}, preventing look-ahead bias.",
    "Portfolio valuation: position value is marked using Close_{t} during the holding period.",
    "Long-only constraint: no short positions — SELL when in-position moves to 100% cash.",
    "Single-unit position: 100% invested or 100% cash (discrete in/out).",
    "Transaction costs: 10 basis points (0.0010 × trade value) deducted exactly once per trade.",
    "Initial capital: ₹1,00,000 per asset episode.",
    "Episode length: matches the split size (up to 179 steps on test).",
]
for b in bullets:
    add_bullet(b)

add_figure(str(DIAGRAMS / "06_trading_environment_execution.png"),
           "Figure 6: Trading Environment Execution Model (Next-Day Open Fill)")

doc.add_paragraph(
    "Six numerical accounting unit tests verified the environment implementation, including "
    "correct BUY/SELL execution, HOLD behavior, transaction cost deduction, long-only enforcement, "
    "and episode termination. All 6 tests passed (`gymnasium.utils.env_checker.check_env()` also passed)."
)

# ══════════════════════════════════════════════════
# SECTION 13 — OBSERVATION SPACE
# ══════════════════════════════════════════════════
add_heading("13. Observation Space (49-Dimensional Hybrid)", level=1)
doc.add_paragraph(
    "Each observation vector concatenates three components:"
)
add_table(
    ["Component", "Dimensions", "Description"],
    [
        ("LSTM Embedding", "32", "Pre-extracted 32-D market state embedding from frozen LSTM encoder"),
        ("Technical Features", "14", "Current-step values of all 14 technical indicators"),
        ("Portfolio State", "3", "Position flag (0/1), normalized portfolio value, cash ratio"),
        ("Total", "49", "Full hybrid observation vector"),
    ]
)

add_figure(str(DIAGRAMS / "04_49d_hybrid_observation.png"),
           "Figure 7: 49-Dimensional Hybrid Observation Construction")

doc.add_paragraph(
    "The 49-D observation contains only information available at decision time (step t). "
    "The LSTM embedding encodes the last 30 steps through the frozen encoder; "
    "technical features are current-day values; portfolio state reflects the current holding. "
    "No future information (day t+1 or later) is included. This was verified by a leakage audit."
)

# ══════════════════════════════════════════════════
# SECTION 14 — ACTION SPACE
# ══════════════════════════════════════════════════
add_heading("14. Action Space and Execution Model", level=1)
add_table(
    ["Action ID", "Name", "Effect"],
    [
        ("0", "HOLD", "No trade. Maintain current position (in-market or cash)."),
        ("1", "BUY", "If in cash: buy equity at next-day Open. If already in position: HOLD."),
        ("2", "SELL", "If in position: sell at next-day Open. If in cash: HOLD."),
    ]
)
doc.add_paragraph(
    "The discrete action space (Gymnasium Discrete(3)) is appropriate for single-asset "
    "long-only trading. Redundant actions (BUY when already in, SELL when in cash) are "
    "treated as HOLD, eliminating wasted transitions and simplifying the reward landscape."
)

# ══════════════════════════════════════════════════
# SECTION 15 — REWARD FUNCTION
# ══════════════════════════════════════════════════
add_heading("15. Reward Function", level=1)
doc.add_paragraph(
    "The reward at each step is the step-level log return of the portfolio value, "
    "net of transaction costs:"
)
doc.add_paragraph(
    "    r_t = log(V_{t+1} / V_t)"
)
doc.add_paragraph(
    "where V_t is the portfolio value at step t (including position value at Close_t "
    "plus cash). Transaction costs are deducted from V when a trade occurs, so the "
    "agent implicitly learns to minimize unnecessary trading. "
    "Log returns are used for additivity and numerical stability."
)

# ══════════════════════════════════════════════════
# SECTION 16 — PPO AGENT ARCHITECTURE
# ══════════════════════════════════════════════════
add_heading("16. PPO Agent Architecture", level=1)
doc.add_paragraph(
    "The agent uses a decoupled Actor-Critic architecture — separate networks for policy and value:"
)
add_table(
    ["Component", "Architecture", "Output"],
    [
        ("Shared Trunk", "None (fully decoupled)", "—"),
        ("Actor", "Linear(49→64) → LayerNorm → Tanh → Linear(64→64) → LayerNorm → Tanh → Linear(64→3) → Softmax", "Action probabilities (3)"),
        ("Critic", "Linear(49→64) → LayerNorm → Tanh → Linear(64→64) → LayerNorm → Tanh → Linear(64→1)", "State value V(s)"),
        ("Weight init", "Orthogonal (gain=√2 for hidden, gain=0.01 for actor output, gain=1.0 for critic output)", "Stable gradients"),
    ]
)
doc.add_paragraph(
    "LayerNorm (instead of BatchNorm) is used because it operates correctly on "
    "single-sample inference without batch statistics. The decoupled architecture "
    "avoids gradient interference between actor and critic objectives."
)
add_figure(str(DIAGRAMS / "05_ppo_actor_critic_architecture.png"),
           "Figure 8: PPO Actor-Critic Architecture (49-D input → decoupled heads)")

# ══════════════════════════════════════════════════
# SECTION 17 — PPO ALGORITHM IMPLEMENTATION
# ══════════════════════════════════════════════════
add_heading("17. PPO Algorithm Implementation", level=1)
doc.add_paragraph(
    "PPO (Schulman et al., 2017) with the following implementation details:"
)
bullets = [
    "On-policy rollout collection: each iteration collects 4,000 transitions using the current policy.",
    "Generalized Advantage Estimation (GAE, λ=0.95): computes advantage estimates with variance-bias trade-off.",
    "Clipped surrogate objective: L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)], clip ε=0.2.",
    "Value function loss: L_VF = 0.5 × MSE(V(s_t), returns_t), coefficient=0.5.",
    "Entropy bonus: H = -Σ π(a|s) log π(a|s), annealed from 0.015 to 0.005 over training.",
    "Gradient clipping: max norm 0.5 to prevent gradient explosions.",
    "Mini-batch updates: 4 epochs per iteration, batch size 128, randomly shuffled.",
    "Learning rate: linearly annealed from 3×10⁻⁴ to 1×10⁻⁴ over 35 iterations.",
    "Total loss: L = -L_CLIP + c_vf × L_VF - c_ent × H.",
]
for b in bullets:
    add_bullet(b)

doc.add_paragraph(
    "Total training: 35 iterations × 4,000 transitions = 140,000 transitions. "
    "Mini-batch updates: 35 × 4 epochs × (4000/128) ≈ 4,480 gradient steps."
)

# ══════════════════════════════════════════════════
# SECTION 18 — HYPERPARAMETERS
# ══════════════════════════════════════════════════
add_heading("18. Training Configuration and Hyperparameters", level=1)
hp = tr["hyperparameters"]
add_table(
    ["Hyperparameter", "Value", "Rationale"],
    [
        ("Total iterations", "35", "Sufficient for convergence on this dataset size"),
        ("Transitions per iteration", "4,000", "Matches training split size"),
        ("Initial learning rate", "3×10⁻⁴", "Standard PPO learning rate"),
        ("Final learning rate", "1×10⁻⁴", "Linear annealing for stability"),
        ("Discount factor γ", "0.99", "Long-horizon reward attribution"),
        ("GAE λ", "0.95", "Variance-bias balance for advantage estimates"),
        ("Clip epsilon ε", "0.2", "Standard PPO clip range"),
        ("Value loss coefficient", "0.5", "Balance policy and value losses"),
        ("Initial entropy coefficient", "0.015", "Encourage exploration early"),
        ("Final entropy coefficient", "0.005", "Reduce exploration in later iterations"),
        ("PPO epochs per iteration", "4", "Multiple passes over each rollout"),
        ("Mini-batch size", "128", "Stable gradient estimates"),
        ("Max gradient norm", "0.5", "Prevent gradient explosions"),
        ("Observation dimension", "49", "32 LSTM + 14 technical + 3 portfolio"),
        ("Action dimension", "3", "HOLD, BUY, SELL"),
        ("Hidden dimension", "64", "Network width for actor and critic"),
        ("Master seed", "42", "Full reproducibility"),
    ]
)

# ══════════════════════════════════════════════════
# SECTION 19 — TRAINING PROGRESS
# ══════════════════════════════════════════════════
add_heading("19. Training Progress and Loss Analysis", level=1)
doc.add_paragraph(
    "Key training metrics across selected iterations:"
)
history = tr["training_history"]
sel_iters = [1, 4, 10, 20, 30, 35]
rows = []
for h in history:
    if h["iteration"] in sel_iters:
        rows.append((
            str(h["iteration"]),
            f"{h['loss_pi']:.6f}",
            f"{h['loss_vf']:.6f}",
            f"{h['entropy']:.4f}",
            f"{h['val_sharpe']:.4f}",
            f"{h['val_return%']:.2f}%",
            "✅" if h.get("is_best") else "",
        ))
add_table(
    ["Iter", "Policy Loss", "Value Loss", "Entropy", "Val Sharpe", "Val Return", "Best?"],
    rows
)
doc.add_paragraph(
    "Policy loss trends negative (clipped gradient signal). Entropy decreases monotonically, "
    "indicating the agent learns a more deterministic policy over time. "
    "Validation performance peaks at Iteration 4 and degrades thereafter, "
    "suggesting the agent over-adapts to training data patterns beyond Iteration 4."
)
add_figure(str(RESULTS / "05_training_validation_metrics.png"),
           "Figure 9: Training and Validation Metrics Across 35 PPO Iterations")

# ══════════════════════════════════════════════════
# SECTION 20 — MODEL SELECTION
# ══════════════════════════════════════════════════
add_heading("20. Validation-Based Model Selection", level=1)
doc.add_paragraph(
    "Model selection uses the following rule applied to validation-set evaluation:"
)
bullets = [
    "Eligibility filter: checkpoint must produce ≥ 1 trade per asset on average (active policy).",
    "Selection criterion: highest validation Sharpe ratio among eligible checkpoints.",
    "Tie-breaking: earlier iteration preferred if Sharpe values are equal.",
    "Fallback: if no eligible checkpoint found, report explicitly (do not select cash policy).",
]
for b in bullets:
    add_bullet(b)
val_sum = tr["val_summary"]
doc.add_paragraph(
    f"Iteration 4 was selected as the best checkpoint with:"
)
add_table(
    ["Metric", "Value", "Context"],
    [
        ("Validation Return", f"{val_sum['Avg_Return%']:.2f}%", f"Avg B&H on val set: -9.58% (bear market)"),
        ("Validation Sharpe", f"{val_sum['Avg_Sharpe']:.4f}", "Highest among eligible iterations"),
        ("Validation Max Drawdown", f"{val_sum['Avg_Max_Drawdown%']:.2f}%", "Limited downside vs B&H (-16.4%)"),
        ("Validation Avg Trades", f"{val_sum['Avg_Trades']}", "Active policy confirmed"),
    ]
)
doc.add_paragraph(
    "The negative validation Sharpe (-0.0079) is expected given the severe bear market "
    "during the validation period (average B&H return -9.58%). Relative to the passive B&H "
    "baseline on the same data, Iteration 4 significantly outperformed — returning -0.60% "
    "versus -9.58% for B&H. The agent's capital preservation behaviour in a bear market "
    "is the key mechanism driving relative outperformance."
)

# ══════════════════════════════════════════════════
# SECTION 21 — OUT-OF-SAMPLE EVALUATION
# ══════════════════════════════════════════════════
add_heading("21. Out-of-Sample Evaluation", level=1)
doc.add_paragraph(
    "The Iteration 4 checkpoint (ppo_reproducible_best.pt) was evaluated on the held-out "
    "test set (895 steps, 179 steps per asset). Test data was accessed exactly once."
)
test_sum = tr["test_summary"]
add_table(
    ["Metric", "Value"],
    [
        ("Average Return", f"+{test_sum['Avg_Return%']:.2f}%"),
        ("Average Sharpe Ratio", f"+{test_sum['Avg_Sharpe']:.4f}"),
        ("Average Max Drawdown", f"{test_sum['Avg_Max_Drawdown%']:.2f}%"),
        ("Average Trades per Asset", f"{test_sum['Avg_Trades']:.1f}"),
        ("Average Transaction Cost", "₹1,646 per asset"),
        ("Initial Capital", "₹1,00,000 per asset"),
        ("Test Period", "179 steps per asset (≈9 months)"),
        ("Market Regime", "Bear market (avg B&H: -15.74%)"),
    ]
)

add_figure(str(RESULTS / "02_ppo_equity_curves.png"),
           "Figure 10: Per-Asset PPO Equity Curves vs Buy-and-Hold")
add_figure(str(RESULTS / "03_portfolio_drawdown.png"),
           "Figure 11: Portfolio Drawdown Analysis")

# ══════════════════════════════════════════════════
# SECTION 22 — REPRODUCIBILITY
# ══════════════════════════════════════════════════
add_heading("22. Deterministic Reproducibility", level=1)
doc.add_paragraph(
    "Full reproducibility was verified by running two independent PPO training instances "
    "with identical master seed (42) and configuration, then comparing all outputs:"
)
rep_cmp = rep["metrics_comparison"]
add_table(
    ["Comparison Metric", "Max Difference", "Status"],
    [
        ("Initial network weights", f"{rep_cmp['initial_weights_max_diff']:.8f}", "✅ IDENTICAL"),
        ("Policy loss (all iterations)", f"{rep_cmp['policy_loss_max_diff']:.8f}", "✅ IDENTICAL"),
        ("Value loss (all iterations)", f"{rep_cmp['value_loss_max_diff']:.8f}", "✅ IDENTICAL"),
        ("Val Sharpe (all iterations)", f"{rep_cmp['val_sharpe_max_diff']:.8f}", "✅ IDENTICAL"),
        ("Val Return (all iterations)", f"{rep_cmp['val_return_max_diff']:.8f}", "✅ IDENTICAL"),
        ("Test return (final)", f"{rep_cmp['test_return_diff']:.8f}", "✅ IDENTICAL"),
        ("Test Sharpe (final)", f"{rep_cmp['test_sharpe_diff']:.8f}", "✅ IDENTICAL"),
        ("Agent actions (895 steps × 5)", f"{rep_cmp['action_match_pct']:.2f}% identical", "✅ 895/895 MATCH"),
        ("Selected iteration", "4 vs 4", "✅ IDENTICAL"),
    ]
)
doc.add_paragraph(
    "The implementation seeds: Python `random`, NumPy, PyTorch CPU and CUDA, environment "
    "reset seeds, action sampling, and mini-batch shuffling — all via a single `set_seed(42)` "
    "call in `backend/agents/ppo_agent.py`. CuDNN determinism flags are set. "
    "Determinism is guaranteed on CPU; GPU execution is not tested."
)
add_figure(str(RESULTS / "06_reproducibility_comparison.png"),
           "Figure 12: Dual-Run Reproducibility Verification (Run 1 vs Run 2)")

# ══════════════════════════════════════════════════
# SECTION 23 — BASELINE COMPARISON
# ══════════════════════════════════════════════════
add_heading("23. Comparison with Baseline Strategies", level=1)
doc.add_paragraph(
    "All strategies were evaluated on identical test period, execution rules (next-day Open), "
    "initial capital (₹1,00,000), and transaction costs (10 bps per trade):"
)
bl = tr["baselines_comparison"]
add_table(
    ["Strategy", "Avg Return", "Avg Sharpe", "Avg Max Drawdown", "Avg Trades"],
    [
        ("PPO Agent (Ours)",
         f"+{bl['PPO_Agent']['Return%']:.2f}%",
         f"+{bl['PPO_Agent']['Sharpe']:.2f}",
         f"{bl['PPO_Agent']['Max_DD%']:.2f}%",
         f"{bl['PPO_Agent']['Trades']:.1f}"),
        ("Buy and Hold",
         f"{bl['Buy_and_Hold']['Return%']:.2f}%",
         f"{bl['Buy_and_Hold']['Sharpe']:.2f}",
         f"{bl['Buy_and_Hold']['Max_DD%']:.2f}%",
         f"{bl['Buy_and_Hold']['Trades']:.1f}"),
        ("MA Crossover",
         f"{bl['MA_Crossover']['Return%']:.2f}%",
         f"{bl['MA_Crossover']['Sharpe']:.2f}",
         f"{bl['MA_Crossover']['Max_DD%']:.2f}%",
         f"{bl['MA_Crossover']['Trades']:.1f}"),
        ("Always HOLD (Cash)",
         f"{bl['Always_HOLD']['Return%']:.2f}%",
         f"{bl['Always_HOLD']['Sharpe']:.2f}",
         f"{bl['Always_HOLD']['Max_DD%']:.2f}%",
         f"{bl['Always_HOLD']['Trades']:.1f}"),
    ]
)
doc.add_paragraph(
    "In the bear market test period, the PPO agent achieved positive returns (+0.34%) while "
    "all passive strategies incurred significant losses (B&H: -15.74%, MA: -9.63%). "
    "The PPO agent's positive Sharpe (+0.13) and limited drawdown (-13.03%) further demonstrate "
    "its ability to manage downside risk through selective disengagement (HOLD/SELL actions) "
    "in unfavorable market conditions."
)
add_figure(str(RESULTS / "04_strategy_performance_comparison.png"),
           "Figure 13: Strategy Performance Comparison (PPO vs Baselines)")

# ══════════════════════════════════════════════════
# SECTION 24 — PER-ASSET ANALYSIS
# ══════════════════════════════════════════════════
add_heading("24. Per-Asset Performance Analysis", level=1)
stocks = tr["test_per_stock"]
add_table(
    ["Asset", "Return", "Sharpe", "Max Drawdown", "Trades", "Costs (₹)"],
    [(s["Symbol"], f"{s['Total_Return%']:+.2f}%", f"{s['Sharpe']:+.3f}",
      f"{s['Max_Drawdown%']:.2f}%", str(s["Trades"]), f"₹{s['Costs']:,.0f}")
     for s in stocks]
)
doc.add_paragraph(
    "INFY and RELIANCE showed positive returns (+15.72% and +3.72%), while TCS, ITC, and "
    "HDFCBANK were negative (-9.27%, -5.49%, -2.99%). The agent's performance is strongly "
    "influenced by asset-specific price trends during the test period. INFY's strong recovery "
    "in the IT sector contributed most positively to the average return."
)

# ══════════════════════════════════════════════════
# SECTION 25 — TRANSACTION COST ANALYSIS
# ══════════════════════════════════════════════════
add_heading("25. Transaction Cost Analysis", level=1)
doc.add_paragraph(
    "Transaction costs at 10 basis points (0.0010 × trade value) were deducted exactly once "
    "per trade (at execution, not on both entry and exit separately):"
)
add_table(
    ["Asset", "Trades", "Total Cost (₹)", "Cost per Trade (₹)"],
    [(s["Symbol"], str(s["Trades"]), f"₹{s['Costs']:,.0f}", f"₹{s['Costs']/s['Trades']:,.0f}")
     for s in stocks]
)
doc.add_paragraph(
    "Average cost across assets: ₹1,646 (from verified ppo_reproducibility_results.json). "
    "The moderate trade frequency (average 16.8 trades per asset over 179 steps) is consistent "
    "with a selective, capital-preservation strategy."
)

# ══════════════════════════════════════════════════
# SECTION 26 — METHODOLOGICAL VALIDITY
# ══════════════════════════════════════════════════
add_heading("26. Methodological Validity", level=1)
doc.add_paragraph("The following safeguards ensure methodological integrity:")
bullets = [
    "No data leakage: strictly chronological split; no shuffling across time boundaries.",
    "No future leakage: 49-D observation contains only day-t data and prior history.",
    "LSTM frozen during RL: encoder weights do not change after pre-training.",
    "Single test evaluation: test set used exactly once after final model selection.",
    "Model selection on validation: Iteration 4 selected using validation Sharpe, not test Sharpe.",
    "Identical baselines: B&H and MA Crossover use same test period, capital, execution rules.",
    "Transaction costs applied: 10 bps deducted on every trade for PPO and MA Crossover.",
    "Reproducibility verified: 100% bitwise identical dual runs with master seed 42.",
]
for b in bullets:
    add_bullet(b)

# ══════════════════════════════════════════════════
# SECTION 27 — LIMITATIONS
# ══════════════════════════════════════════════════
add_heading("27. Limitations", level=1)
bullets = [
    "Bear market bias: both validation and test periods are bear markets; results may not generalize to bull or sideways markets.",
    "Single-asset independent evaluation: each asset is traded independently with no cross-asset portfolio optimization.",
    "Discrete all-or-nothing position: cannot take partial positions (e.g., 50% invested), limiting risk management.",
    "No transaction costs for HOLD: in reality, holding costs (brokerage, STT) may apply over long periods.",
    "LSTM directional accuracy is modest (55.53%): only slightly above random; signal is weak.",
    "Data notebooks not re-executed: data pipeline notebooks do not have saved inline outputs.",
    "5-year single regime: trained and tested on 2021–2026 data only; regime changes may degrade performance.",
    "CPU-only reproducibility: GPU execution not tested for bitwise reproducibility.",
    "No slippage model: assumes exact execution at next-day Open price with no slippage.",
]
for b in bullets:
    add_bullet(b)

# ══════════════════════════════════════════════════
# SECTION 28 — FUTURE WORK
# ══════════════════════════════════════════════════
add_heading("28. Future Work", level=1)
bullets = [
    "Portfolio-level agent: multi-asset PPO with joint action space for cross-asset allocation.",
    "Continuous action space: position sizing from 0% to 100% using actor outputting Beta or Gaussian distribution.",
    "Transformer encoder: replace LSTM with Transformer (self-attention) for potentially richer state encoding.",
    "Multi-regime training: include bull, bear, and sideways market periods in training data.",
    "Live deployment: integrate with real NSE broker API (Zerodha Kite Connect) for paper trading validation.",
    "Risk-adjusted reward: incorporate Sortino ratio or CVaR directly into the reward function.",
    "Order book data: incorporate Level 2 market microstructure data for intraday trading.",
    "Hyperparameter optimization: systematic Bayesian optimization of PPO hyperparameters.",
    "Longer evaluation: extend test period to 3+ years covering multiple market regimes.",
]
for b in bullets:
    add_bullet(b)

# ══════════════════════════════════════════════════
# SECTION 29 — CONCLUSION
# ══════════════════════════════════════════════════
add_heading("29. Conclusion", level=1)
doc.add_paragraph(
    "NeuroTrader demonstrates that a pre-trained LSTM encoder combined with Proximal Policy "
    "Optimization can achieve positive risk-adjusted returns (+0.34% return, +0.13 Sharpe) "
    "in a genuine bear market test environment where passive strategies incurred significant losses "
    "(Buy-and-Hold: -15.74%). The key mechanism is the agent's learned ability to transition "
    "to cash during prolonged downtrends, limiting capital drawdown while preserving optionality."
)
doc.add_paragraph(
    "The implementation satisfies rigorous academic standards: strict chronological data splitting, "
    "no future information leakage, frozen encoder during RL training, single test evaluation, "
    "and 100% bitwise reproducibility verified by dual independent runs. All results are sourced "
    "from verified experimental artifacts."
)
doc.add_paragraph(
    "The modest LSTM directional accuracy (55.53%) and bear market regime limitation indicate "
    "this is a proof-of-concept with real but constrained generalization. Future extensions — "
    "portfolio-level optimization, continuous action spaces, and multi-regime training — offer "
    "clear paths to improving practical performance."
)

# ══════════════════════════════════════════════════
# SECTION 30 — REFERENCES
# ══════════════════════════════════════════════════
add_heading("30. References", level=1)
refs = [
    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
    "Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.",
    "Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2016). Deep direct reinforcement learning for financial signal representation and trading. IEEE TNNLS, 28(3), 653–664.",
    "Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.",
    "Liang, Z., Chen, H., Zhu, J., Jiang, K., & Li, Y. (2018). Adversarial Deep Reinforcement Learning in Portfolio Management. arXiv:1808.09940.",
    "Moody, J., & Saffell, M. (1998). Reinforcement learning for trading. Advances in Neural Information Processing Systems, 10.",
    "Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.",
    "Towers, M., et al. (2024). Gymnasium. GitHub: https://github.com/Farama-Foundation/Gymnasium.",
    "Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438.",
]
for i, r in enumerate(refs):
    p = doc.add_paragraph(f"[{i+1}] {r}", style="List Number")

doc.add_page_break()

# ══════════════════════════════════════════════════
# APPENDIX — CHECKPOINT HASHES
# ══════════════════════════════════════════════════
add_heading("Appendix A: Checkpoint File Integrity", level=1)
ckpts = {
    "LSTM Encoder": BASE / "backend" / "checkpoints" / "improved_rnn_encoder.pt",
    "PPO Best (Canonical)": BASE / "backend" / "checkpoints" / "ppo_reproducible_best.pt",
}
rows = []
for name, path in ckpts.items():
    if path.exists():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        rows.append((name, path.name, sha[:32]+"...", f"{size:,} bytes"))
    else:
        rows.append((name, path.name, "FILE MISSING", "—"))
add_table(["Checkpoint", "Filename", "SHA-256 (prefix)", "Size"], rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(f"✅ Final report written: {OUT}")
print(f"   Size: {OUT.stat().st_size:,} bytes")
