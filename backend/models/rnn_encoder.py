"""
Market State Representation RNN/LSTM Encoder Model
==================================================
Project: Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading

Architecture:
    Input (B, 30, 14) -> 2-Layer LSTM (Hidden=64, Dropout=0.2) -> Last Time-Step Hidden Representation (B, 64)
    -> Embedding Projection: Linear(64 -> 32) + LayerNorm + LeakyReLU -> 32-D Market State Embedding (B, 32)
    -> Regression Head: Dropout(0.2) + Linear(32 -> 1) -> Predicted Forward 1-Day Return (B, 1)

Modularity:
    The encoder method `encode(x)` can be decoupled and plugged directly into any Reinforcement Learning
    policy/value network as the continuous market-state feature extractor.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class MarketStateLSTMEncoder(nn.Module):
    """
    LSTM-based market state representation encoder with a 32-dimensional
    state embedding bottleneck and forward-return regression prediction head.
    """

    def __init__(
        self,
        input_dim: int = 14,
        sequence_length: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # 2-Layer LSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 32-Dimensional Market State Embedding Projection
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Supervised Pretraining Regression Head (Next-day return prediction)
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 32-dimensional market state representation embedding.

        Args:
            x (torch.Tensor): Input batch of shape (Batch_Size, Sequence_Length, Input_Dim).

        Returns:
            torch.Tensor: Market state embedding of shape (Batch_Size, 32).
        """
        # x shape: (B, L, D)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last time step output: lstm_out[:, -1, :] of shape (B, hidden_dim)
        last_hidden = lstm_out[:, -1, :]

        # Project to 32-dimensional market state embedding
        embedding = self.embedding_layer(last_hidden)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and regression head.

        Args:
            x (torch.Tensor): Input batch of shape (B, 30, 14).
            return_embedding (bool): If True, returns tuple (prediction, embedding).

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
                Predicted return of shape (B, 1) and optionally embedding (B, 32).
        """
        embedding = self.encode(x)
        prediction = self.regressor(embedding)

        if return_embedding:
            return prediction, embedding
        return prediction


if __name__ == "__main__":
    # Smoke test model instantiation and forward pass
    model = MarketStateLSTMEncoder()
    dummy_input = torch.randn(8, 30, 14)
    pred, emb = model(dummy_input, return_embedding=True)
    print("MarketStateLSTMEncoder test successful:")
    print(f"  Input: {tuple(dummy_input.shape)}")
    print(f"  Embedding: {tuple(emb.shape)} (dim={model.embedding_dim})")
    print(f"  Prediction: {tuple(pred.shape)} (dim={model.output_dim})")

