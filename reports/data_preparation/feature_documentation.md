# Feature Engineering & Market State Representation Documentation

**Project:** Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading  
**Author:** NeuroTrade Research Pipeline  
**Version:** 1.0.0  

---

## 1. Mathematical Specifications of Engineered Features

The state representation vector $s_t$ must satisfy two fundamental machine learning properties:
1. **Stationarity:** The statistical properties (mean, variance) of the input features should remain stable across different historical price regimes.
2. **Causality (No Look-Ahead Bias):** Every indicator at time $t$ depends strictly on information set $\mathcal{F}_t = \\{O_i, H_i, L_i, C_i, V_i\\}_{i=0}^t$.

| Feature Name | Category | Mathematical Definition | Financial Rationale & Role in RL State |
|---|---|---|---|
| `Daily_Return` | Return Metric | $R_t = \\frac{C_t - C_{t-1}}{C_{t-1}}$ | Standard daily arithmetic price return; primary stationary measure of momentum. |
| `Log_Return` | Return Metric | $r_t = \\ln\\left(\\frac{C_t}{C_{t-1}}\\right)$ | Continuously compounded return; invariant to compounding frequency. |
| `HL_Pct` | Volatility Metric | $HL\\_Pct_t = \\frac{H_t - L_t}{C_t}$ | Intraday price dispersion / spread normalized by closing price. Invariant to price level. |
| `Rolling_Vol_20` | Volatility Metric | $\\sigma_{20, t} = \\sqrt{\\frac{1}{19}\\sum_{i=0}^{19} (R_{t-i} - \\bar{R})^2}$ | 20-day historical realized return volatility; informs agent of prevailing risk regime. |
| `SMA_10`, `SMA_20`, `SMA_50` | Trend Indicator | $SMA_{n, t} = \\frac{1}{n}\\sum_{i=0}^{n-1} C_{t-i}$ | Short-, intermediate-, and long-term moving baseline levels. |
| `SMA_Ratio_10` | Normalized Trend | $\\frac{C_t - SMA_{10, t}}{SMA_{10, t}}$ | Percentage distance of current price to 10-day moving average. Stationary. |
| `SMA_Ratio_20` | Normalized Trend | $\\frac{C_t - SMA_{20, t}}{SMA_{20, t}}$ | Percentage distance of current price to 20-day moving average. Stationary. |
| `SMA_Ratio_50` | Normalized Trend | $\\frac{C_t - SMA_{50, t}}{SMA_{50, t}}$ | Percentage distance of current price to 50-day moving average. Stationary. |
| `EMA_12`, `EMA_26` | Trend Indicator | $EMA_{n, t} = \\alpha C_t + (1-\\alpha)EMA_{n, t-1}, \\; \\alpha=\\frac{2}{n+1}$ | Exponentially weighted moving averages; responsive to recent trend shifts. |
| `EMA_Ratio_12` | Normalized Trend | $\\frac{C_t - EMA_{12, t}}{EMA_{12, t}}$ | Scale-invariant fast trend deviation. |
| `EMA_Ratio_26` | Normalized Trend | $\\frac{C_t - EMA_{26, t}}{EMA_{26, t}}$ | Scale-invariant slow trend deviation. |
| `RSI_14` | Momentum Oscillator | $100 - \\frac{100}{1 + RS_{14}}$ (Wilder smoothing) | Quantifies overbought (>70) and oversold (<30) momentum conditions. |
| `RSI_14_Norm` | Normalized Oscillator | $\\frac{RSI_{14, t}}{100.0} \\in [0, 1]$ | Rescaled bounded oscillator ideal for neural network gradient flow. |
| `MACD_Line` | Trend / Momentum | $EMA_{12}(C) - EMA_{26}(C)$ | Classic momentum convergence/divergence metric. |
| `MACD_Signal` | Trend / Momentum | $EMA_{9}(MACD\\_Line)$ | 9-day exponential moving average signal line. |
| `MACD_Hist` | Trend / Momentum | $MACD\\_Line - MACD\\_Signal$ | Divergence histogram indicating acceleration of price momentum. |
| `MACD_Line_Norm` | Normalized Momentum | $\\frac{MACD\\_Line_t}{C_t}$ | Stationary MACD line normalized by current closing price. |
| `MACD_Signal_Norm`| Normalized Momentum | $\\frac{MACD\\_Signal_t}{C_t}$ | Stationary MACD signal line normalized by current closing price. |
| `MACD_Hist_Norm`  | Normalized Momentum | $\\frac{MACD\\_Hist_t}{C_t}$ | Stationary MACD histogram normalized by current closing price. |
| `Volume_Ratio_20` | Liquidity Metric | $\\frac{V_t}{SMA_{20}(V)}$ | Ratio of current volume to 20-day average volume; distinguishes high-conviction moves. |
| `Target_Return_Next`| RL Target / Label | $\\frac{C_{t+1} - C_t}{C_t}$ | Next-period forward return. Used exclusively for supervised training / reward calculation. **Excluded from RNN input state**. |

---

## 2. Stationary Market State Feature Subset for RNN

For feeding into the RNN encoder $f_\\theta(x_{t-L+1:t})$, the following stationary feature subset is standardized by the training scaler:

```python
STATE_FEATURE_COLS = [
    "Daily_Return",
    "Log_Return",
    "HL_Pct",
    "Rolling_Vol_20",
    "SMA_Ratio_10",
    "SMA_Ratio_20",
    "SMA_Ratio_50",
    "EMA_Ratio_12",
    "EMA_Ratio_26",
    "RSI_14_Norm",
    "MACD_Line_Norm",
    "MACD_Signal_Norm",
    "MACD_Hist_Norm",
    "Volume_Ratio_20",
]
```

This 14-dimensional feature vector gives the recurrent neural network complete visibility into:
1. Short-term and logarithmic price momentum
2. Intraday and 20-day rolling market volatility
3. Multi-horizon moving average trend distances (10, 20, 50 days)
4. Fast and slow exponential trend alignments (12, 26 days)
5. Bounded bounded momentum oscillator state (RSI)
6. Stationary MACD convergence and momentum acceleration
7. Relative liquidity and institutional trading volume pressure

---

## 3. Warmup Burn-in & Integrity Guarantees

- **50-Day Burn-in Truncation:** Because the 50-day Simple Moving Average requires 50 historical trading sessions, the initial 50 bars are discarded. No heuristic zero-filling or synthetic imputation is performed on initial indicators.
- **Strict Boundary Check:** The final bar where $C_{t+1}$ does not yet exist is removed to prevent target null values.
- **Absence of Look-Ahead Bias:** Every feature at row $t$ is mathematically independent of row $t+1, t+2, \\dots$.
