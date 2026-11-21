"""
Automated Verification Suite for NeuroTrade Live Pipeline (Phase 2).

Tests:
1. Checkpoint and scaler artifact existence and integrity
2. Cache TTL and invalidation logic
3. Feature engineering mathematics & zero-NaN constraint
4. 49-D state vector construction
5. PPO model inference output validity
6. Dashboard file generation
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.market_cache import (
    get_cache_status, load_from_cache, save_to_cache, CACHE_DIR, CACHE_TTL_SECONDS
)
from backend.live_data.feature_pipeline import (
    compute_features, prepare_inference_sequence, STATE_FEATURE_COLS, SCALER_PATH
)
from backend.live_data.inference_engine import LiveInferenceEngine
from backend.live_data.dashboard import export_html_dashboard, HTML_OUTPUT_PATH


class TestLivePipeline(unittest.TestCase):

    def test_01_artifacts_exist(self):
        """Verify model checkpoints and scaler artifacts exist."""
        scaler_file = PROJECT_ROOT / "data" / "training" / "scalers" / "feature_scaler.joblib"
        lstm_file = PROJECT_ROOT / "backend" / "checkpoints" / "improved_rnn_encoder.pt"
        ppo_file = PROJECT_ROOT / "backend" / "checkpoints" / "ppo_reproducible_best.pt"

        self.assertTrue(scaler_file.exists(), "feature_scaler.joblib missing")
        self.assertTrue(lstm_file.exists(), "improved_rnn_encoder.pt missing")
        self.assertTrue(ppo_file.exists(), "ppo_reproducible_best.pt missing")

    def test_02_cache_mechanism(self):
        """Verify 3-minute cache manager returns correct structure."""
        status = get_cache_status()
        self.assertIn("exists", status)
        self.assertIn("is_valid", status)
        self.assertEqual(status["ttl_seconds"], 180)

        # Check cached data can be read
        cached = load_from_cache(["RELIANCE"])
        if cached:
            self.assertIn("historical", cached)
            self.assertIn("RELIANCE", cached["historical"])
            self.assertGreater(len(cached["historical"]["RELIANCE"]), 60)

    def test_03_feature_computation(self):
        """Verify 14 features can be computed without NaNs on cached data."""
        cached = load_from_cache(["RELIANCE"])
        self.assertIsNotNone(cached, "Cache required for feature test. Run run_live.py first.")
        bars = cached["historical"]["RELIANCE"]

        feat_df = compute_features(bars)
        self.assertEqual(len(feat_df), len(bars))
        for col in STATE_FEATURE_COLS:
            self.assertIn(col, feat_df.columns)

        # Verify last 30 bars have zero NaNs
        last_30 = feat_df.iloc[-30:][STATE_FEATURE_COLS]
        self.assertEqual(last_30.isna().sum().sum(), 0, "NaNs found in features!")

    def test_04_sequence_preparation(self):
        """Verify prepare_inference_sequence produces correct shapes."""
        cached = load_from_cache(["RELIANCE"])
        bars = cached["historical"]["RELIANCE"]
        seq_tensor, last_14, latest_bar = prepare_inference_sequence(bars)

        self.assertEqual(seq_tensor.shape, (1, 30, 14))
        self.assertEqual(last_14.shape, (14,))
        self.assertIn("close", latest_bar)
        self.assertIn("date", latest_bar)

    def test_05_inference_engine(self):
        """Verify full inference produces valid HOLD/BUY/SELL and probabilities."""
        engine = LiveInferenceEngine()
        cached = load_from_cache(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"])
        self.assertIsNotNone(cached)

        for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]:
            bars = cached["historical"][sym]
            quote = cached["quotes"].get(sym, {})
            pred = engine.predict_stock(sym, bars, quote)

            self.assertIn(pred["recommendation"], ["HOLD", "BUY", "SELL"])
            self.assertIn(pred["action_code"], [0, 1, 2])
            probs = pred["action_probabilities"]
            total_prob = sum(probs.values())
            self.assertAlmostEqual(total_prob, 1.0, places=3)
            self.assertGreater(pred["current_price"], 0)

    def test_06_dashboard_generation(self):
        """Verify HTML dashboard file is written and valid."""
        self.assertTrue(HTML_OUTPUT_PATH.exists())
        content = HTML_OUTPUT_PATH.read_text(encoding="utf-8")
        self.assertIn("NeuroTrade", content)
        self.assertIn("RELIANCE", content)
        self.assertIn("SEBI Regulatory", content)


if __name__ == "__main__":
    unittest.main()
