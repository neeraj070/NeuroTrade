"""
Inference Engine for NeuroTrade Live Pipeline.

Loads:
- Frozen LSTM encoder: backend/checkpoints/improved_rnn_encoder.pt
- Frozen PPO agent:    backend/checkpoints/ppo_reproducible_best.pt
- Frozen scaler:       data/training/scalers/feature_scaler.joblib

Executes:
1. Feature extraction & scaling -> (1, 30, 14) sequence
2. LSTM encoding -> 32-D market state representation embedding
3. State synthesis -> 49-D hybrid state:
   - [0:32]  : 32-D LSTM latent embedding
   - [32:46] : 14-D explicit normalized technical features at close of bar t
   - [46:49] : 3-D portfolio context [position_fraction, cash_ratio, norm_value]
4. PPO evaluation -> Action in {HOLD, BUY, SELL} and categorical action distribution
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn

from backend.models.rnn_encoder import MarketStateLSTMEncoder
from backend.agents.ppo_agent import ActorCriticTradingNetwork
from .feature_pipeline import prepare_inference_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LSTM_CHECKPOINT = PROJECT_ROOT / "backend" / "checkpoints" / "improved_rnn_encoder.pt"
PPO_CHECKPOINT = PROJECT_ROOT / "backend" / "checkpoints" / "ppo_reproducible_best.pt"
SCALER_PATH = PROJECT_ROOT / "data" / "training" / "scalers" / "feature_scaler.joblib"

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


class LiveInferenceEngine:
    """Singleton inference engine managing model loading and execution."""

    def __init__(self):
        self.scaler = None
        self.lstm = None
        self.ppo = None
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Scaler
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler missing at: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)

        # 2. LSTM Encoder
        if not LSTM_CHECKPOINT.exists():
            raise FileNotFoundError(f"LSTM checkpoint missing at: {LSTM_CHECKPOINT}")
        lstm_ck = torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=True)
        self.lstm = MarketStateLSTMEncoder(
            input_dim=14, sequence_length=30, hidden_dim=64, num_layers=2, embedding_dim=32
        )
        self.lstm.load_state_dict(lstm_ck["model_state_dict"])
        self.lstm.eval()

        # 3. PPO Agent
        if not PPO_CHECKPOINT.exists():
            raise FileNotFoundError(f"PPO checkpoint missing at: {PPO_CHECKPOINT}")
        ppo_ck = torch.load(PPO_CHECKPOINT, map_location="cpu", weights_only=True)
        self.ppo = ActorCriticTradingNetwork(state_dim=49, action_dim=3, hidden_dim=64)
        self.ppo.load_state_dict(ppo_ck["policy_state_dict"])
        self.ppo.eval()

    def predict_stock(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        quote: Optional[Dict[str, Any]] = None,
        portfolio_context: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Runs full inference pipeline for a single stock.

        Args:
            symbol: Ticker symbol (e.g. 'RELIANCE')
            bars: Completed daily OHLCV bars (need >= 80)
            quote: Optional live snapshot quote from CM Market
            portfolio_context: [equity_fraction, cash_ratio, norm_value].
                               Defaults to flat [0.0, 1.0, 1.0].

        Returns:
            Structured prediction dictionary.
        """
        if portfolio_context is None:
            # Default flat entry evaluation: 0 equity, 100% cash, portfolio=1.0
            portfolio_context = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        # 1. Feature preparation & scaling
        seq_tensor, last_14, latest_bar = prepare_inference_sequence(bars, self.scaler)

        # 2. LSTM Forward Pass
        with torch.no_grad():
            embedding = self.lstm.encode(seq_tensor)  # (1, 32)
            predicted_return = self.lstm(seq_tensor).item()  # (1, 1)

        emb_np = embedding.squeeze(0).numpy()

        # 3. 49-D State Construction
        state_49 = np.concatenate([emb_np, last_14, portfolio_context])
        state_tensor = torch.tensor(state_49, dtype=torch.float32).unsqueeze(0)

        # 4. PPO Policy & Value Evaluation
        with torch.no_grad():
            logits = self.ppo.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            action_idx = int(torch.argmax(logits, dim=-1).item())
            critic_val = self.ppo.critic(state_tensor).item()

        action_name = ACTION_NAMES[action_idx]

        # Extract quote prices if available, otherwise use latest daily candle
        current_price = latest_bar["close"]
        price_change_pct = latest_bar["daily_return_pct"]
        timestamp = latest_bar["date"]
        fifty_two_w_high = None
        fifty_two_w_low = None

        if quote:
            if quote.get("lastTradedPrice"):
                current_price = float(quote["lastTradedPrice"])
            if quote.get("perChange") is not None:
                price_change_pct = float(quote["perChange"])
            if quote.get("latestTimestamp"):
                timestamp = str(quote["latestTimestamp"])
            fifty_two_w_high = quote.get("fiftyTwoWeekHigh")
            fifty_two_w_low = quote.get("fiftyTwoWeekLow")

        return {
            "symbol": symbol,
            "current_price": current_price,
            "daily_open": latest_bar["open"],
            "daily_high": latest_bar["high"],
            "daily_low": latest_bar["low"],
            "daily_close": latest_bar["close"],
            "daily_volume": latest_bar["volume"],
            "price_change_pct": price_change_pct,
            "candle_date": latest_bar["date"],
            "data_timestamp": timestamp,
            "52w_high": fifty_two_w_high,
            "52w_low": fifty_two_w_low,
            "rsi_14": round(latest_bar["rsi_14"], 2),
            "recommendation": action_name,
            "action_code": action_idx,
            "action_probabilities": {
                "HOLD": round(float(probs[0]), 4),
                "BUY": round(float(probs[1]), 4),
                "SELL": round(float(probs[2]), 4),
            },
            "critic_state_value": round(critic_val, 4),
            "lstm_predicted_return": round(predicted_return * 100, 3),
            "embedding_l2_norm": round(float(np.linalg.norm(emb_np)), 4),
        }
