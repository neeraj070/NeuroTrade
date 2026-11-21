"""
Feature Engineering & Preprocessing Pipeline for Live Data.

Strictly adheres to the specifications from:
- data/notebooks/04_feature_engineering.ipynb
- data/notebooks/05_preprocessing.ipynb

14 Technical State Indicators:
  1. Daily_Return       = (C_t - C_{t-1}) / C_{t-1}
  2. Log_Return         = ln(C_t / C_{t-1})
  3. HL_Pct             = (H_t - L_t) / C_t
  4. Rolling_Vol_20     = 20-day rolling std of Daily_Return
  5. SMA_Ratio_10       = (C_t - SMA_10) / SMA_10
  6. SMA_Ratio_20       = (C_t - SMA_20) / SMA_20
  7. SMA_Ratio_50       = (C_t - SMA_50) / SMA_50
  8. EMA_Ratio_12       = (C_t - EMA_12) / EMA_12
  9. EMA_Ratio_26       = (C_t - EMA_26) / EMA_26
 10. RSI_14_Norm        = Wilder RSI_14 / 100.0
 11. MACD_Line_Norm     = (EMA_12 - EMA_26) / C_t
 12. MACD_Signal_Norm   = EMA_9(MACD_Line) / C_t
 13. MACD_Hist_Norm     = (MACD_Line - MACD_Signal) / C_t
 14. Volume_Ratio_20    = Volume / SMA_20(Volume)

Scales features using the frozen StandardScaler from data/training/scalers/feature_scaler.joblib.
Constructs 30-timestep sequence for the LSTM encoder: shape (1, 30, 14).
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALER_PATH = PROJECT_ROOT / "data" / "training" / "scalers" / "feature_scaler.joblib"

WINDOW_LENGTH = 30

STATE_FEATURE_COLS: List[str] = [
    "Daily_Return", "Log_Return", "HL_Pct", "Rolling_Vol_20",
    "SMA_Ratio_10", "SMA_Ratio_20", "SMA_Ratio_50",
    "EMA_Ratio_12", "EMA_Ratio_26", "RSI_14_Norm",
    "MACD_Line_Norm", "MACD_Signal_Norm", "MACD_Hist_Norm", "Volume_Ratio_20",
]


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's Relative Strength Index (RSI-14)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_features(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes all 14 stationary technical indicators from a list of OHLCV bars.
    
    Args:
        bars: Chronologically sorted list of dicts with 'date', 'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        DataFrame containing Date, OHLCV, and all 14 computed features.
    """
    df = pd.DataFrame(bars)
    if "date" in df.columns and "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    close = df["close"].astype(float) if "close" in df.columns else df["Close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else df["High"].astype(float)
    low = df["low"].astype(float) if "low" in df.columns else df["Low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else df["Volume"].astype(float)

    feat_df = pd.DataFrame()
    feat_df["Date"] = df["Date"]
    feat_df["Close"] = close
    feat_df["Open"] = df["open"].astype(float) if "open" in df.columns else df["Open"].astype(float)
    feat_df["High"] = high
    feat_df["Low"] = low
    feat_df["Volume"] = volume

    # Returns
    feat_df["Daily_Return"] = close.pct_change(fill_method=None)
    feat_df["Log_Return"] = np.log(close / close.shift(1))

    # Volatility
    feat_df["HL_Pct"] = (high - low) / close
    feat_df["Rolling_Vol_20"] = feat_df["Daily_Return"].rolling(window=20).std()

    # Simple Moving Averages
    sma_10 = close.rolling(window=10).mean()
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()

    feat_df["SMA_Ratio_10"] = (close - sma_10) / sma_10
    feat_df["SMA_Ratio_20"] = (close - sma_20) / sma_20
    feat_df["SMA_Ratio_50"] = (close - sma_50) / sma_50

    # Exponential Moving Averages
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    feat_df["EMA_Ratio_12"] = (close - ema_12) / ema_12
    feat_df["EMA_Ratio_26"] = (close - ema_26) / ema_26

    # Momentum: RSI-14
    feat_df["RSI_14"] = calculate_rsi(close, period=14)
    feat_df["RSI_14_Norm"] = feat_df["RSI_14"] / 100.0

    # Momentum: MACD (12, 26, 9)
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    feat_df["MACD_Line_Norm"] = macd_line / close
    feat_df["MACD_Signal_Norm"] = macd_signal / close
    feat_df["MACD_Hist_Norm"] = macd_hist / close

    # Liquidity
    vol_sma_20 = volume.rolling(window=20).mean()
    feat_df["Volume_Ratio_20"] = (volume / vol_sma_20.replace(0.0, np.nan)).fillna(1.0)

    return feat_df


def prepare_inference_sequence(
    bars: List[Dict[str, Any]],
    scaler=None
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    Transforms historical bars into ready-to-infer tensors.
    
    1. Computes all 14 features across the bar history (~120 bars).
    2. Verifies the latest 30 bars have ZERO NaNs.
    3. Normalizes using the trained StandardScaler.
    4. Extracts the 30-timestep sequence for LSTM: (1, 30, 14).
    5. Extracts the last timestep's explicit indicators: (14,).
    
    Returns:
        (sequence_tensor, last_explicit_features, latest_bar_info)
    """
    if scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler missing at: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)

    feat_df = compute_features(bars)

    if len(feat_df) < (50 + WINDOW_LENGTH):
        raise ValueError(
            f"Insufficient historical bars: {len(feat_df)} found. "
            f"Need at least 80 bars (50 warmup + 30 sequence)."
        )

    # Take the latest 30 bars
    last_30_df = feat_df.iloc[-WINDOW_LENGTH:].copy()
    feature_matrix = last_30_df[STATE_FEATURE_COLS].values

    # Strict NaN check
    if np.isnan(feature_matrix).any():
        nan_cols = [col for col in STATE_FEATURE_COLS if last_30_df[col].isna().any()]
        raise ValueError(f"NaN values detected in feature columns: {nan_cols}")

    # Scale with frozen scaler (DO NOT FIT on live data)
    scaled_matrix = scaler.transform(feature_matrix)

    # Tensor for LSTM input: (Batch=1, SeqLen=30, Dim=14)
    sequence_tensor = torch.tensor(scaled_matrix, dtype=torch.float32).unsqueeze(0)

    # Latest explicit indicators (scaled at timestep t)
    last_explicit_14 = scaled_matrix[-1]

    latest_row = feat_df.iloc[-1]
    latest_info = {
        "date": str(latest_row["Date"])[:10],
        "open": float(latest_row["Open"]),
        "high": float(latest_row["High"]),
        "low": float(latest_row["Low"]),
        "close": float(latest_row["Close"]),
        "volume": float(latest_row["Volume"]),
        "rsi_14": float(latest_row["RSI_14"]),
        "daily_return_pct": float(latest_row["Daily_Return"] * 100) if pd.notna(latest_row["Daily_Return"]) else 0.0,
    }

    return sequence_tensor, last_explicit_14, latest_info
