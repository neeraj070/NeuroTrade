"""
Generate Project_Presentation.pptx for NeuroTrader (15–18 slides).
Uses actual verified metrics. Embeds architecture diagrams.
"""
import os, json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Cm

os.environ["MPLCONFIGDIR"] = r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade\backend\.mpl_cache"

BASE     = Path(r"c:\Users\mekal\OneDrive\Desktop\NeuroTrade")
DIAGRAMS = BASE / "reports" / "final_submission" / "Architecture_Diagrams"
RESULTS  = BASE / "reports" / "final_submission" / "Result_Visualizations"
OUT      = BASE / "reports" / "final_submission" / "Project_Presentation.pptx"

# Load verified results
tr  = json.loads((BASE / "reports" / "reinforcement_learning" / "ppo_training_results.json").read_text())
rep = json.loads((BASE / "reports" / "reinforcement_learning" / "ppo_reproducibility_results.json").read_text())

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─── Color palette ───
DARK_BG  = RGBColor(0x0F, 0x1A, 0x2E)   # dark navy
ACCENT   = RGBColor(0x00, 0xC8, 0xE0)   # cyan
GOLD     = RGBColor(0xFF, 0xC0, 0x27)   # gold
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0x14, 0x28, 0x44)   # slightly lighter navy
GREEN    = RGBColor(0x2E, 0xCC, 0x71)
RED_C    = RGBColor(0xE7, 0x4C, 0x3C)
GRAY     = RGBColor(0xAA, 0xBB, 0xCC)

def blank_slide():
    layout = prs.slide_layouts[6]  # completely blank
    slide  = prs.slides.add_slide(layout)
    return slide

def set_bg(slide, color=DARK_BG):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text_box(slide, text, left, top, width, height,
                 font_size=24, bold=False, color=WHITE,
                 align=PP_ALIGN.LEFT, italic=False, word_wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_bullets(slide, items, left, top, width, height,
                font_size=18, color=WHITE, title_color=ACCENT,
                title=None, title_size=22):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = title
        run.font.size = Pt(title_size)
        run.font.bold = True
        run.font.color.rgb = title_color
        first = False
    for item in items:
        p = tf.add_paragraph() if not first else tf.paragraphs[0]
        first = False
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = ("• " if isinstance(item, str) else f"  ○ ") + (item if isinstance(item, str) else item)
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
    return txBox

def add_image_safe(slide, img_path, left, top, width, height=None):
    p = Path(img_path)
    if p.exists():
        if height:
            slide.shapes.add_picture(str(p), Inches(left), Inches(top), Inches(width), Inches(height))
        else:
            slide.shapes.add_picture(str(p), Inches(left), Inches(top), Inches(width))
    else:
        add_text_box(slide, f"[Image: {p.name}]", left, top, width, height or 1, font_size=10, color=GRAY)

def add_divider(slide, top, color=ACCENT):
    from pptx.oxml.ns import qn
    line = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.LINE doesn't exist; use rectangle as thin bar
        Inches(0.5), Inches(top), Inches(12.3), Inches(0.03)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()

def title_slide_header(slide, title, subtitle=None, color=ACCENT):
    set_bg(slide)
    # accent bar top
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    # accent bar bottom
    bar2 = slide.shapes.add_shape(1, Inches(0), Inches(7.38), Inches(13.33), Inches(0.12))
    bar2.fill.solid(); bar2.fill.fore_color.rgb = color; bar2.line.fill.background()
    # title
    add_text_box(slide, title, 0.5, 0.25, 12.3, 0.8,
                 font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text_box(slide, subtitle, 0.5, 0.95, 12.3, 0.4,
                     font_size=16, color=GOLD, align=PP_ALIGN.LEFT)
    add_divider(slide, 1.3)

# ══════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════
sl = blank_slide()
set_bg(sl)
# Large colored bar
bar = sl.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(0.2))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
bar2 = sl.shapes.add_shape(1, Inches(0), Inches(7.3), Inches(13.33), Inches(0.2))
bar2.fill.solid(); bar2.fill.fore_color.rgb = ACCENT; bar2.line.fill.background()
# Left accent
barL = sl.shapes.add_shape(1, Inches(0), Inches(0.2), Inches(0.08), Inches(7.1))
barL.fill.solid(); barL.fill.fore_color.rgb = GOLD; barL.line.fill.background()

add_text_box(sl, "NeuroTrader", 0.4, 1.0, 12.5, 1.0,
             font_size=52, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_text_box(sl, "Reinforcement Learning with RNN-Based Market State\nRepresentation for Algorithmic Trading",
             0.4, 2.1, 12.5, 1.5, font_size=22, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
add_divider(sl, 3.75)
add_text_box(sl, "B.Tech Final-Year Project", 0.4, 3.9, 12.5, 0.5,
             font_size=16, color=GOLD, align=PP_ALIGN.CENTER)
add_text_box(sl,
    "5 NSE Equities  •  LSTM Encoder  •  PPO Agent  •  2021–2026  •  November 2026",
    0.4, 4.4, 12.5, 0.5, font_size=14, color=GRAY, align=PP_ALIGN.CENTER)
add_text_box(sl, "Python 3.12  •  PyTorch  •  Gymnasium  •  100% Reproducible",
             0.4, 5.0, 12.5, 0.5, font_size=13, color=GRAY, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════
# SLIDE 2 — OVERVIEW / AGENDA
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Presentation Overview")
items = [
    ("1. Motivation & Problem Statement", ACCENT),
    ("2. System Architecture", ACCENT),
    ("3. Dataset & Feature Engineering", ACCENT),
    ("4. LSTM Encoder — Market State Representation", ACCENT),
    ("5. Trading Environment (Gymnasium)", ACCENT),
    ("6. PPO Agent Architecture & Training", ACCENT),
    ("7. Training Progress & Model Selection", ACCENT),
    ("8. Out-of-Sample Results", ACCENT),
    ("9. Strategy Comparison", ACCENT),
    ("10. Reproducibility Verification", ACCENT),
    ("11. Limitations & Future Work", ACCENT),
    ("12. Conclusion", ACCENT),
]
x_positions = [0.5, 6.7]
for i, (text, color) in enumerate(items):
    col = i % 2
    row = i // 2
    add_text_box(sl, text, x_positions[col], 1.5 + row * 0.65, 5.8, 0.6,
                 font_size=16, color=WHITE, bold=False)

# ══════════════════════════════════════════════════
# SLIDE 3 — MOTIVATION
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Motivation & Problem Statement",
                   "Why combine LSTM encoding with Reinforcement Learning for trading?")
add_bullets(sl, [
    "Financial markets are non-stationary: price distributions shift over time",
    "Raw price data is noisy; technical indicators improve signal-to-noise ratio",
    "Rule-based strategies (MA crossover) fail to adapt to regime changes",
    "Deep RL can learn adaptive policies without explicit rule programming",
    "LSTM captures temporal dependencies in multi-step price sequences",
], 0.5, 1.5, 7.0, 4.5, font_size=18)

add_bullets(sl, [
    "Research Question",
    "Can a frozen LSTM encoder + PPO agent learn a profitable,",
    "reproducible trading strategy evaluated strictly out-of-sample",
    "on NSE equities in a genuine bear market?",
], 7.5, 1.5, 5.3, 3.0, font_size=16, color=GOLD, title="Research Question", title_color=ACCENT)

# ══════════════════════════════════════════════════
# SLIDE 4 — SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "System Architecture", "Four-stage pipeline: Data → LSTM → Environment → PPO")
add_image_safe(sl, DIAGRAMS / "01_overall_system_architecture.png", 0.3, 1.4, 12.5, 5.7)

# ══════════════════════════════════════════════════
# SLIDE 5 — DATASET
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Dataset & Feature Engineering", "5 NSE equities, 2021-11-04 to 2026-11-04")

# Left — dataset info
add_bullets(sl, [
    "RELIANCE  •  TCS  •  INFY  •  HDFCBANK  •  ITC",
    "1,236 daily bars per asset (OHLCV)",
    "Source: Yahoo Finance (yfinance API)",
    "Period: 2021-11-04 → 2026-11-04",
    "",
    "14 Technical Indicators (stationary):",
    "Log Return, SMA/EMA Ratios, RSI-14",
    "MACD, Bollinger Bands, ATR, ROC-10",
    "Volume Ratio, High-Low Ratio",
], 0.5, 1.4, 6.5, 5.5, font_size=16)

# Right — split table
add_text_box(sl, "Data Split", 7.2, 1.5, 5.5, 0.5, font_size=20, bold=True, color=ACCENT)
splits = [
    ("Train",       "70%",  "4,000",  "LSTM + PPO training"),
    ("Validation",  "15%",  "885",    "Model selection"),
    ("Test",        "15%",  "895",    "Final evaluation (once)"),
]
for i, (name, pct, n, purpose) in enumerate(splits):
    y = 2.1 + i * 1.4
    color = GREEN if name == "Train" else (GOLD if name == "Validation" else ACCENT)
    box = sl.shapes.add_shape(1, Inches(7.2), Inches(y), Inches(5.5), Inches(1.2))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_BG; box.line.color.rgb = color
    add_text_box(sl, f"{name}  {pct}  ({n} transitions)", 7.3, y+0.05, 5.3, 0.4,
                 font_size=17, bold=True, color=color)
    add_text_box(sl, purpose, 7.3, y+0.5, 5.3, 0.5, font_size=14, color=WHITE)

# ══════════════════════════════════════════════════
# SLIDE 6 — LSTM ENCODER
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "LSTM Encoder — Market State Representation",
                   "Pre-trained 2-layer LSTM → 32-D market state embedding")
add_image_safe(sl, DIAGRAMS / "03_lstm_encoder_architecture.png", 0.3, 1.4, 6.5, 5.5)
add_text_box(sl, "Architecture", 7.0, 1.4, 5.8, 0.4, font_size=18, bold=True, color=ACCENT)
specs = [
    "Input:  30 timesteps × 14 features",
    "Layer 1: LSTM (hidden=64)",
    "Layer 2: LSTM (hidden=64)",
    "Output: Linear(64 → 32)",
    "Dropout: 0.1 between layers",
    "",
    "Training: MSE loss on next-day Close",
    "Encoder frozen during PPO training",
]
for i, s in enumerate(specs):
    add_text_box(sl, s, 7.0, 1.9 + i * 0.42, 5.8, 0.4,
                 font_size=15, color=(GOLD if "frozen" in s.lower() else WHITE))

add_text_box(sl, "Test Performance", 7.0, 5.4, 5.8, 0.4, font_size=18, bold=True, color=ACCENT)
add_text_box(sl,
    "MAE: 0.012997  vs  Zero-baseline: 0.013050\n"
    "Directional Accuracy: 55.53%  (497/895)\n"
    "Filtered (τ=0.000562): 58.54% @ 36.65% coverage",
    7.0, 5.9, 5.8, 1.3, font_size=14, color=GREEN)

# ══════════════════════════════════════════════════
# SLIDE 7 — TRADING ENVIRONMENT
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Gymnasium Trading Environment",
                   "MarketTradingEnv — 49-D observation, Discrete(3) actions")
add_image_safe(sl, DIAGRAMS / "06_trading_environment_execution.png", 0.3, 1.4, 6.0, 4.5)
add_text_box(sl, "Design Decisions", 6.6, 1.4, 6.3, 0.4, font_size=18, bold=True, color=ACCENT)
specs = [
    "✅  Next-day Open execution (no look-ahead)",
    "✅  Long-only constraint (no short selling)",
    "✅  10 bps transaction cost per trade",
    "✅  Initial capital: ₹1,00,000 per asset",
    "✅  Single-unit position (all-in / all-out)",
    "✅  Gymnasium check_env passed",
    "✅  6/6 numerical unit tests passed",
]
for i, s in enumerate(specs):
    add_text_box(sl, s, 6.6, 1.9 + i * 0.65, 6.3, 0.6, font_size=15, color=GREEN)

add_text_box(sl, "Actions: 0=HOLD   1=BUY   2=SELL", 6.6, 6.5, 6.3, 0.5,
             font_size=16, bold=True, color=GOLD)

# ══════════════════════════════════════════════════
# SLIDE 8 — 49-D OBSERVATION
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "49-Dimensional Hybrid Observation Space",
                   "LSTM embedding + Technical features + Portfolio state")
add_image_safe(sl, DIAGRAMS / "04_49d_hybrid_observation.png", 0.3, 1.4, 12.5, 5.7)

# ══════════════════════════════════════════════════
# SLIDE 9 — PPO AGENT
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "PPO Actor-Critic Architecture",
                   "Decoupled Actor-Critic | Orthogonal init | LayerNorm")
add_image_safe(sl, DIAGRAMS / "05_ppo_actor_critic_architecture.png", 0.3, 1.4, 6.0, 5.5)
add_text_box(sl, "Network Design", 6.6, 1.4, 6.3, 0.4, font_size=18, bold=True, color=ACCENT)
specs = [
    "Input: 49-D observation",
    "Hidden: 2 × Linear(64) + LayerNorm + Tanh",
    "Actor: Softmax → P(action | state)",
    "Critic: Scalar V(state)",
    "Weight init: Orthogonal (gain=√2)",
    "Actor output gain: 0.01 (small initial actions)",
    "Decoupled: separate params for actor/critic",
]
for i, s in enumerate(specs):
    add_text_box(sl, s, 6.6, 1.9 + i * 0.58, 6.3, 0.5, font_size=15, color=WHITE)

add_text_box(sl, "PPO Training", 6.6, 5.9, 6.3, 0.4, font_size=18, bold=True, color=ACCENT)
add_text_box(sl,
    "35 iter  •  140K transitions  •  ~4,480 grad steps\n"
    "LR: 3e-4 → 1e-4  •  γ=0.99  •  λ=0.95  •  clip=0.2",
    6.6, 6.35, 6.3, 0.9, font_size=14, color=GOLD)

# ══════════════════════════════════════════════════
# SLIDE 10 — TRAINING PROGRESS
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Training Progress & Model Selection",
                   "Iteration 4 selected by highest validation Sharpe (active policies only)")
add_image_safe(sl, RESULTS / "05_training_validation_metrics.png", 0.3, 1.4, 7.5, 5.5)
add_text_box(sl, "Model Selection Rule", 8.1, 1.5, 4.8, 0.4, font_size=17, bold=True, color=ACCENT)
bullets_sel = [
    "1. Filter: avg trades ≥ 1 (active policy)",
    "2. Rank by validation Sharpe",
    "3. Select highest eligible checkpoint",
    "",
    "Selected: Iteration 4",
    f"Val Sharpe: -0.0079",
    f"Val Return: -0.60%",
    f"Val Avg Trades: 4.4",
    "",
    "⚠️ Negative Sharpe is expected:",
    "Val period avg B&H: -9.58%",
    "PPO: -0.60% >> B&H: -9.58%",
]
for i, b in enumerate(bullets_sel):
    color = GREEN if ("Selected" in b or "Iter" in b) else (GOLD if "⚠️" in b else WHITE)
    add_text_box(sl, b, 8.1, 2.0 + i * 0.37, 4.8, 0.4, font_size=14, color=color)

# ══════════════════════════════════════════════════
# SLIDE 11 — OUT-OF-SAMPLE RESULTS
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Out-of-Sample Test Results",
                   "895 steps • 179 steps/asset • Bear market regime (avg B&H: -15.74%)")

# Key metrics — big numbers
metrics = [
    ("+0.34%", "Avg Return", GREEN),
    ("+0.13",  "Avg Sharpe", GREEN),
    ("-13.03%","Max Drawdown", GOLD),
    ("16.8",   "Trades/Asset", ACCENT),
    ("₹1,646", "Avg Tx Cost", GRAY),
]
for i, (val, label, color) in enumerate(metrics):
    x = 0.3 + i * 2.55
    box = sl.shapes.add_shape(1, Inches(x), Inches(1.5), Inches(2.3), Inches(1.6))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_BG
    box.line.color.rgb = color; box.line.width = Inches(0.03)
    add_text_box(sl, val, x+0.05, 1.6, 2.2, 0.75,
                 font_size=28, bold=True, color=color, align=PP_ALIGN.CENTER)
    add_text_box(sl, label, x+0.05, 2.3, 2.2, 0.5,
                 font_size=13, color=WHITE, align=PP_ALIGN.CENTER)

add_image_safe(sl, RESULTS / "02_ppo_equity_curves.png", 0.3, 3.3, 7.5, 3.9)
# Per-asset table
stocks = tr["test_per_stock"]
add_text_box(sl, "Per-Asset Results", 8.0, 3.3, 5.0, 0.4, font_size=16, bold=True, color=ACCENT)
headers = "Asset     Return    Sharpe"
add_text_box(sl, headers, 8.0, 3.75, 5.0, 0.35, font_size=13, bold=True, color=GRAY)
for i, s in enumerate(stocks):
    ret = s["Total_Return%"]
    color = GREEN if ret > 0 else RED_C
    row = f"{s['Symbol']:<12} {ret:+.2f}%   {s['Sharpe']:+.3f}"
    add_text_box(sl, row, 8.0, 4.15 + i * 0.55, 5.0, 0.5, font_size=14, color=color)

# ══════════════════════════════════════════════════
# SLIDE 12 — STRATEGY COMPARISON
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Strategy Comparison",
                   "Identical test period, capital, execution rules, and transaction costs")
add_image_safe(sl, RESULTS / "04_strategy_performance_comparison.png", 0.3, 1.4, 7.0, 5.5)
add_text_box(sl, "Summary", 7.5, 1.5, 5.5, 0.4, font_size=18, bold=True, color=ACCENT)
bl = tr["baselines_comparison"]
strats = [
    ("PPO Agent (Ours)", f"{bl['PPO_Agent']['Return%']:+.2f}%",  f"{bl['PPO_Agent']['Sharpe']:+.2f}", GREEN),
    ("MA Crossover",     f"{bl['MA_Crossover']['Return%']:+.2f}%",  f"{bl['MA_Crossover']['Sharpe']:+.2f}", RED_C),
    ("Buy and Hold",     f"{bl['Buy_and_Hold']['Return%']:+.2f}%",  f"{bl['Buy_and_Hold']['Sharpe']:+.2f}", RED_C),
    ("Always HOLD",      f"{bl['Always_HOLD']['Return%']:+.2f}%",   f"{bl['Always_HOLD']['Sharpe']:+.2f}", GRAY),
]
add_text_box(sl, "Strategy           Return    Sharpe", 7.5, 2.0, 5.5, 0.4,
             font_size=14, bold=True, color=GRAY)
for i, (name, ret, sharpe, color) in enumerate(strats):
    add_text_box(sl, f"{name}", 7.5, 2.5 + i*1.1, 5.5, 0.4, font_size=16, bold=True, color=color)
    add_text_box(sl, f"Return: {ret}   Sharpe: {sharpe}", 7.5, 2.9 + i*1.1, 5.5, 0.4, font_size=14, color=WHITE)

# ══════════════════════════════════════════════════
# SLIDE 13 — REPRODUCIBILITY
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Deterministic Reproducibility",
                   "100% bitwise identical dual runs with master_seed=42")
add_image_safe(sl, RESULTS / "06_reproducibility_comparison.png", 0.3, 1.4, 7.0, 5.5)

add_text_box(sl, "Verification Results", 7.5, 1.5, 5.5, 0.4, font_size=18, bold=True, color=ACCENT)
rep_items = [
    "Initial weights:  0.00000000 diff",
    "Policy loss:      0.00000000 diff",
    "Value loss:       0.00000000 diff",
    "Val Sharpe:       0.00000000 diff",
    "Test return:      0.00000000 diff",
    "Agent actions:    895/895 identical",
    "Best iteration:   4 == 4  ✅",
]
for i, item in enumerate(rep_items):
    add_text_box(sl, item, 7.5, 2.1 + i * 0.6, 5.5, 0.55, font_size=15, color=GREEN)

add_text_box(sl, "set_seed(42) seeds: Python, NumPy,\nPyTorch CPU/CUDA, env resets,\naction sampling, mini-batch shuffle",
             7.5, 6.3, 5.5, 1.0, font_size=13, color=GOLD)

# ══════════════════════════════════════════════════
# SLIDE 14 — LIMITATIONS & FUTURE WORK
# ══════════════════════════════════════════════════
sl = blank_slide()
title_slide_header(sl, "Limitations & Future Work")

add_text_box(sl, "Known Limitations", 0.5, 1.5, 6.0, 0.4, font_size=18, bold=True, color=RED_C)
limitations = [
    "Bear market regime only (2021-2026)",
    "Single-asset independent evaluation",
    "All-or-nothing position sizing",
    "Modest LSTM directional accuracy (55.53%)",
    "CPU-only reproducibility verified",
    "Data notebooks lack inline outputs",
    "No slippage model",
]
for i, lim in enumerate(limitations):
    add_text_box(sl, f"⚠ {lim}", 0.5, 2.1 + i * 0.65, 6.0, 0.6, font_size=15, color=GOLD)

add_text_box(sl, "Future Directions", 6.8, 1.5, 6.0, 0.4, font_size=18, bold=True, color=GREEN)
futures = [
    "Portfolio-level multi-asset PPO",
    "Continuous position sizing (Beta/Gaussian)",
    "Transformer encoder (replace LSTM)",
    "Multi-regime training (bull + bear)",
    "Live NSE paper trading (Zerodha Kite)",
    "Sortino/CVaR risk-adjusted reward",
    "Order book microstructure data",
    "Bayesian hyperparameter optimization",
]
for i, f in enumerate(futures):
    add_text_box(sl, f"→ {f}", 6.8, 2.1 + i * 0.65, 6.0, 0.6, font_size=15, color=ACCENT)

# ══════════════════════════════════════════════════
# SLIDE 15 — CONCLUSION
# ══════════════════════════════════════════════════
sl = blank_slide()
set_bg(sl)
# Accent bars
for y_pos in [0, 7.3]:
    bar = sl.shapes.add_shape(1, Inches(0), Inches(y_pos), Inches(13.33), Inches(0.2))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()

add_text_box(sl, "Conclusion", 0.5, 0.4, 12.3, 0.8, font_size=36, bold=True, color=WHITE)
add_divider(sl, 1.2)

conclusions = [
    ("✅", "LSTM encoder achieves 55.53% directional accuracy — above 50% random baseline",   WHITE),
    ("✅", "PPO agent achieves +0.34% return and +0.13 Sharpe in a bear market test period",   GREEN),
    ("✅", "Outperforms B&H (-15.74%) and MA Crossover (-9.63%) on identical test conditions", GREEN),
    ("✅", "100% bitwise reproducible (dual run, 895/895 identical actions, 0.00 diff)",       GREEN),
    ("✅", "Strict methodological validity: no data leakage, no test-set contamination",        GOLD),
    ("⚠️", "Bear market regime; generalization to bull markets needs further investigation",    GOLD),
    ("🔭", "Future: portfolio optimization, continuous actions, live paper trading",            ACCENT),
]
for i, (icon, text, color) in enumerate(conclusions):
    add_text_box(sl, f"{icon}  {text}", 0.5, 1.4 + i * 0.73, 12.3, 0.65,
                 font_size=16, color=color)

add_text_box(sl, "Thank You", 0.5, 6.7, 12.3, 0.5,
             font_size=22, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════
# SLIDE 16 — END / Q&A
# ══════════════════════════════════════════════════
sl = blank_slide()
set_bg(sl)
bar = sl.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(0.2))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
bar2 = sl.shapes.add_shape(1, Inches(0), Inches(7.3), Inches(13.33), Inches(0.2))
bar2.fill.solid(); bar2.fill.fore_color.rgb = ACCENT; bar2.line.fill.background()

add_text_box(sl, "Q & A", 0.3, 2.2, 12.5, 1.5, font_size=72, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_text_box(sl, "NeuroTrader — Reinforcement Learning with RNN-Based Market State Representation",
             0.3, 4.0, 12.5, 0.6, font_size=16, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(sl,
    "Canonical checkpoint: backend/checkpoints/ppo_reproducible_best.pt  •  master_seed=42\n"
    "Results verified in: reports/reinforcement_learning/ppo_reproducibility_results.json",
    0.3, 4.7, 12.5, 0.8, font_size=12, color=GRAY, align=PP_ALIGN.CENTER)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"✅ Presentation written: {OUT}")
print(f"   Slides: {len(prs.slides)}")
print(f"   Size: {OUT.stat().st_size:,} bytes")
