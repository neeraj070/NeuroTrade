"""
Unit Test Suite for Model & Backend Pipeline
============================================
Project: Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading
"""

import os
import sys
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.models.rnn_encoder import MarketStateLSTMEncoder


class TestMarketStateModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 16
        self.seq_len = 30
        self.input_dim = 14
        self.model = MarketStateLSTMEncoder(
            input_dim=self.input_dim,
            sequence_length=self.seq_len,
            hidden_dim=64,
            num_layers=2,
            embedding_dim=32,
            output_dim=1,
            dropout=0.2,
        )

    def test_forward_pass_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        pred = self.model(x)
        self.assertEqual(pred.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(pred).any())

    def test_encode_embedding_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        emb = self.model.encode(x)
        self.assertEqual(emb.shape, (self.batch_size, 32))
        self.assertFalse(torch.isnan(emb).any())

    def test_gradient_backprop(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim, requires_grad=True)
        pred = self.model(x)
        loss = pred.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()

