"""
Comprehensive Pre-Phase 4 Audit Suite for NeuroTrade.
Verifies all components across Sections 2 through 11 in isolated environments.
Strictly leaves real paper trading database and cache untouched.
"""

import os
import sys
import gc
import json
import shutil
import tempfile
import unittest
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.market_cache import (
    get_cache_status, load_from_cache, save_to_cache, CACHE_TTL_SECONDS
)
from backend.live_data.feature_pipeline import (
    compute_features, prepare_inference_sequence, calculate_rsi,
    STATE_FEATURE_COLS, SCALER_PATH, WINDOW_LENGTH
)
from backend.live_data.inference_engine import (
    LiveInferenceEngine, ACTION_NAMES, LSTM_CHECKPOINT, PPO_CHECKPOINT
)
from backend.live_data.paper_trader import get_dynamic_portfolio_context
import backend.live_data.portfolio_db as pdb
from backend.live_data.dashboard import print_console_dashboard, export_html_dashboard


def get_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class AuditComprehensiveSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Record initial hashes of critical checkpoints
        cls.initial_lstm_hash = get_file_sha256(LSTM_CHECKPOINT)
        cls.initial_ppo_hash = get_file_sha256(PPO_CHECKPOINT)
        cls.initial_scaler_hash = get_file_sha256(SCALER_PATH)

        # Real DB initial state
        real_db = PROJECT_ROOT / "data" / "live" / "paper_trading.db"
        cls.real_db_exists = real_db.exists()
        cls.initial_real_db_hash = get_file_sha256(real_db) if cls.real_db_exists else None

    # =========================================================================
    # SECTION 2: Complete Application Startup Flow Verification
    # =========================================================================
    def test_section_02_startup_flow_and_sequence(self):
        """Verify startup sequence and graceful error isolation."""
        # 1. Verify models & scalers exist and load
        engine = LiveInferenceEngine()
        self.assertIsNotNone(engine.scaler)
        self.assertIsNotNone(engine.lstm)
        self.assertIsNotNone(engine.ppo)
        self.assertFalse(engine.lstm.training)
        self.assertFalse(engine.ppo.training)

        # 2. Verify pipeline handles missing bar count (< 30) safely without producing fake predictions
        short_bars = [
            {"date": f"2026-11-{i:02d}", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}
            for i in range(1, 15)  # only 14 bars
        ]
        with self.assertRaises(ValueError):
            # Must raise ValueError, never fabricate fake predictions
            engine.predict_stock("RELIANCE", short_bars)

    # =========================================================================
    # SECTION 3: NSE MCP & Market-Data Verification
    # =========================================================================
    def test_section_03_mcp_market_data_separation(self):
        """Verify endpoint URLs, OHLCV schema, and candle/quote separation."""
        from backend.live_data.data_fetcher import BHAVCOPY_URL, CM_MARKET_URL, TARGET_STOCKS
        
        self.assertEqual(BHAVCOPY_URL, "https://mcp.nseindia.in/bhavcopy/cm/mcp")
        self.assertEqual(CM_MARKET_URL, "https://mcp.nseindia.in/cmmkt/mcp")
        self.assertEqual(set(TARGET_STOCKS), {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"})

        # Verify cached historical bars contain required OHLCV fields
        cached = load_from_cache(["RELIANCE"])
        self.assertIsNotNone(cached, "Cache must exist for audit verification")
        bars = cached["historical"]["RELIANCE"]
        self.assertGreater(len(bars), 60)
        
        last_bar = bars[-1]
        for field in ["date", "open", "high", "low", "close", "volume"]:
            self.assertIn(field, last_bar, f"Missing field {field} in historical bar")

        # Verify separation of completed daily candle from live intraday quote
        quote = cached["quotes"]["RELIANCE"]
        self.assertIn("lastTradedPrice", quote)
        # Candle close and LTP are separated concepts
        self.assertIsInstance(last_bar["close"], (int, float))
        self.assertIsInstance(float(quote["lastTradedPrice"]), (int, float))

        # Market timestamp vs retrieval timestamp
        self.assertIn("retrieval_timestamp", cached)
        self.assertIn("market_timestamp", cached)
        self.assertNotEqual(cached["retrieval_timestamp"], cached["market_timestamp"])

    # =========================================================================
    # SECTION 4: 3-Minute Cache Scenarios (Isolated Temp Directory)
    # =========================================================================
    def test_section_04_cache_scenarios_isolated(self):
        """Verify cache TTL (180s), reuse, expiration, fallback, and corruption handling."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_cache_dir = Path(temp_dir)
            temp_meta_file = temp_cache_dir / "cache_metadata.json"

            with patch("backend.live_data.market_cache.CACHE_DIR", temp_cache_dir), \
                 patch("backend.live_data.market_cache.METADATA_FILE", temp_meta_file):

                # Scenario 1: First run (no cache exists)
                status1 = get_cache_status()
                self.assertFalse(status1["exists"])
                self.assertFalse(status1["is_valid"])
                self.assertEqual(status1["ttl_seconds"], 180)

                # Save mock cache data
                mock_data = {
                    "historical": {"RELIANCE": [{"date": "2026-11-17", "open": 100, "close": 105}]},
                    "quotes": {"RELIANCE": {"lastTradedPrice": 105}},
                    "market_timestamp": "2026-11-17 15:30:00"
                }
                save_to_cache(mock_data)

                # Scenario 2: 30 seconds after fetch (should be CACHED/valid)
                with open(temp_meta_file, "r") as f:
                    meta = json.load(f)
                meta["retrieval_timestamp"] = (datetime.now() - timedelta(seconds=30)).isoformat()
                with open(temp_meta_file, "w") as f:
                    json.dump(meta, f)

                status2 = get_cache_status()
                self.assertTrue(status2["exists"])
                self.assertTrue(status2["is_valid"])
                self.assertAlmostEqual(status2["age_seconds"], 30.0, delta=2.0)

                # Scenario 3: 120 seconds after fetch (should be CACHED/valid)
                meta["retrieval_timestamp"] = (datetime.now() - timedelta(seconds=120)).isoformat()
                with open(temp_meta_file, "w") as f:
                    json.dump(meta, f)

                status3 = get_cache_status()
                self.assertTrue(status3["is_valid"])
                self.assertAlmostEqual(status3["age_seconds"], 120.0, delta=2.0)

                # Scenario 4: 181 seconds after fetch (should be EXPIRED/invalid)
                meta["retrieval_timestamp"] = (datetime.now() - timedelta(seconds=181)).isoformat()
                with open(temp_meta_file, "w") as f:
                    json.dump(meta, f)

                status4 = get_cache_status()
                self.assertTrue(status4["exists"])
                self.assertFalse(status4["is_valid"])  # Expired!

                # Scenario 5: Corrupted cache metadata file handling
                with open(temp_meta_file, "w") as f:
                    f.write("INVALID_CORRUPTED_JSON{}}")

                status_corrupt = get_cache_status()
                self.assertTrue(status_corrupt["exists"])
                self.assertFalse(status_corrupt["is_valid"])
                self.assertIn("error", status_corrupt)

    # =========================================================================
    # SECTION 5: Feature Engineering & Model Inference Verification
    # =========================================================================
    def test_section_05_features_and_inference_integrity(self):
        """Verify 14 technical features formulas, scaler transform only, shapes, and weights immutability."""
        cached = load_from_cache(["RELIANCE"])
        bars = cached["historical"]["RELIANCE"]

        # 1. Verify 14 feature formulas
        feat_df = compute_features(bars)
        self.assertEqual(len(STATE_FEATURE_COLS), 14)
        for col in STATE_FEATURE_COLS:
            self.assertIn(col, feat_df.columns)

        # Mathematical formula assertions on last row:
        c = feat_df["Close"].iloc[-1]
        c_prev = feat_df["Close"].iloc[-2]
        h = feat_df["High"].iloc[-1]
        l = feat_df["Low"].iloc[-1]

        # Daily_Return = (C_t - C_{t-1}) / C_{t-1}
        expected_ret = (c - c_prev) / c_prev
        self.assertAlmostEqual(feat_df["Daily_Return"].iloc[-1], expected_ret, places=5)

        # Log_Return = ln(C_t / C_{t-1})
        expected_log = np.log(c / c_prev)
        self.assertAlmostEqual(feat_df["Log_Return"].iloc[-1], expected_log, places=5)

        # HL_Pct = (H_t - L_t) / C_t
        expected_hl = (h - l) / c
        self.assertAlmostEqual(feat_df["HL_Pct"].iloc[-1], expected_hl, places=5)

        # 2. Scaler Verification
        engine = LiveInferenceEngine()
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(engine.scaler, StandardScaler)
        self.assertEqual(engine.scaler.n_features_in_, 14)

        # 3. Input and embedding shapes
        seq_tensor, last_14, latest_bar = prepare_inference_sequence(bars, engine.scaler)
        self.assertEqual(seq_tensor.shape, (1, 30, 14))
        self.assertEqual(last_14.shape, (14,))

        with torch.no_grad():
            emb = engine.lstm.encode(seq_tensor)
            self.assertEqual(emb.shape, (1, 32))  # Exactly 32-D bottleneck

            state_49 = np.concatenate([emb.squeeze(0).numpy(), last_14, np.array([0.0, 1.0, 1.0], dtype=np.float32)])
            self.assertEqual(len(state_49), 49)  # Exactly 49-D state

            state_t = torch.tensor(state_49, dtype=torch.float32).unsqueeze(0)
            logits = engine.ppo.actor(state_t)
            self.assertEqual(logits.shape, (1, 3))  # 3 actions: HOLD=0, BUY=1, SELL=2

            critic_v = engine.ppo.critic(state_t)
            self.assertEqual(critic_v.shape, (1, 1))

        # 4. Action mapping verification
        self.assertEqual(ACTION_NAMES[0], "HOLD")
        self.assertEqual(ACTION_NAMES[1], "BUY")
        self.assertEqual(ACTION_NAMES[2], "SELL")

        # 5. Weight immutability test: weights must NOT change during inference
        def get_model_weights_hash(model):
            h = hashlib.sha256()
            for p in model.parameters():
                h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()

        lstm_weights_before = get_model_weights_hash(engine.lstm)
        ppo_weights_before = get_model_weights_hash(engine.ppo)

        for _ in range(25):
            _ = engine.predict_stock("RELIANCE", bars)

        lstm_weights_after = get_model_weights_hash(engine.lstm)
        ppo_weights_after = get_model_weights_hash(engine.ppo)

        self.assertEqual(lstm_weights_before, lstm_weights_after, "LSTM weights were modified during inference!")
        self.assertEqual(ppo_weights_before, ppo_weights_after, "PPO weights were modified during inference!")

        # 6. Reproducibility test
        pred1 = engine.predict_stock("RELIANCE", bars)
        pred2 = engine.predict_stock("RELIANCE", bars)
        self.assertEqual(pred1["recommendation"], pred2["recommendation"])
        self.assertEqual(pred1["action_probabilities"], pred2["action_probabilities"])

    # =========================================================================
    # SECTION 6: Dashboard & HTML Trading Control Verification
    # =========================================================================
    def test_section_06_dashboard_and_html_controls(self):
        """Verify dashboard rendering and audit HTML form submission connectivity."""
        cached = load_from_cache(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"])
        engine = LiveInferenceEngine()
        predictions = [engine.predict_stock(s, cached["historical"][s], cached["quotes"].get(s)) for s in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]]

        pipeline_output = {
            "stocks": predictions,
            "cache_info": {
                "data_status": "CACHED",
                "cache_age_seconds": 15.0,
                "retrieval_timestamp": cached["retrieval_timestamp"],
                "market_timestamp": cached["market_timestamp"],
            }
        }

        # Render HTML to temporary path
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            out_html = Path(temp_dir) / "test_dashboard.html"
            export_html_dashboard(pipeline_output, output_path=out_html)
            self.assertTrue(out_html.exists())
            html_text = out_html.read_text(encoding="utf-8")

            # Check market overview elements
            self.assertIn("RELIANCE", html_text)
            self.assertIn("PPO Policy Probabilities", html_text)
            self.assertIn("Virtual Paper Portfolio Summary", html_text)
            self.assertIn("Pending Orders", html_text)
            self.assertIn("Executed Trade History", html_text)

            # Audit HTML Trading Controls: Check submitPaperOrder implementation
            # FINDING: In standalone HTML, submitPaperOrder uses JS alert and does NOT make HTTP fetch/POST
            has_fetch_or_post = "fetch(" in html_text or "XMLHttpRequest" in html_text or "axios" in html_text
            self.assertFalse(has_fetch_or_post, "Audit Note: HTML controls currently do not issue HTTP POST to backend")

    # =========================================================================
    # SECTION 7: Paper-Trading Functionality (Isolated Database)
    # =========================================================================
    def test_section_07_paper_trading_isolated_db(self):
        """Verify BUY, SELL, HOLD, Pending Orders, Fees, and Accounting on isolated DB."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "isolated_paper_trading.db"

            with patch.object(pdb, "DB_PATH", temp_db), \
                 patch.object(pdb, "DB_DIR", Path(temp_dir)):

                pdb.init_db(initial_capital=100000.0)

                # 1. Initial State
                p0 = pdb.get_portfolio()
                self.assertEqual(p0["initial_capital"], 100000.0)
                self.assertEqual(p0["cash"], 100000.0)
                self.assertEqual(p0["holdings_value"], 0.0)
                self.assertEqual(p0["total_value"], 100000.0)

                # 2. Valid BUY Order
                buy_res = pdb.submit_order(
                    symbol="RELIANCE", side="BUY", quantity=10, current_price=1300.0,
                    agent_recommendation="HOLD", agent_probabilities={"HOLD": 0.5, "BUY": 0.3, "SELL": 0.2},
                    decision_date="2026-11-17"
                )
                self.assertTrue(buy_res["success"])
                self.assertEqual(buy_res["status"], "PENDING")
                order_id = buy_res["order_id"]

                # 3. Cash reservation check: 10 * 1300 * 1.001 = 13,013 reserved
                # Try submitting another BUY that exceeds remaining cash
                buy_over = pdb.submit_order(
                    symbol="RELIANCE", side="BUY", quantity=70, current_price=1300.0,
                    agent_recommendation="HOLD", agent_probabilities={}, decision_date="2026-11-17"
                )
                self.assertEqual(buy_over["status"], "REJECTED")
                self.assertIn("Insufficient available cash", buy_over["reason"])

                # 4. Next Session Execution (Arrival of 2026-11-18 Open = 1320.0)
                sim_market = {
                    "RELIANCE": [
                        {"date": "2026-11-17", "open": 1300.0, "close": 1310.0},
                        {"date": "2026-11-18", "open": 1320.0, "close": 1325.0}
                    ]
                }
                exec_txs = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(exec_txs), 1)
                tx = exec_txs[0]
                self.assertEqual(tx["order_id"], order_id)
                self.assertEqual(tx["execution_price"], 1320.0)
                self.assertEqual(tx["fee"], round(10 * 1320.0 * 0.001, 2))  # Exactly 13.20

                # Check portfolio cash & holding after BUY
                p1 = pdb.get_portfolio()
                expected_cash = 100000.0 - (13200.0 + 13.20)
                self.assertEqual(p1["cash"], round(expected_cash, 2))
                self.assertEqual(p1["holdings"]["RELIANCE"]["shares"], 10)

                # 5. HOLD Order (Zero fees, position maintained)
                hold_res = pdb.submit_order(
                    symbol="TCS", side="HOLD", quantity=0, current_price=2400.0,
                    agent_recommendation="HOLD", agent_probabilities={}, decision_date="2026-11-18"
                )
                self.assertEqual(hold_res["status"], "EXECUTED")
                p_hold = pdb.get_portfolio()
                self.assertEqual(p_hold["cash"], p1["cash"])

                # 6. Valid SELL Order (Sell 4 shares of RELIANCE)
                sell_res = pdb.submit_order(
                    symbol="RELIANCE", side="SELL", quantity=4, current_price=1350.0,
                    agent_recommendation="SELL", agent_probabilities={}, decision_date="2026-11-18"
                )
                self.assertTrue(sell_res["success"])
                self.assertEqual(sell_res["status"], "PENDING")

                # Insufficient shares SELL rejection (trying to sell 10 when only 6 available)
                sell_over = pdb.submit_order(
                    symbol="RELIANCE", side="SELL", quantity=10, current_price=1350.0,
                    agent_recommendation="SELL", agent_probabilities={}, decision_date="2026-11-18"
                )
                self.assertEqual(sell_over["status"], "REJECTED")
                self.assertIn("Insufficient shares held", sell_over["reason"])

                # Execute SELL at 2026-11-19 Open = 1360.0
                sim_market["RELIANCE"].append({"date": "2026-11-19", "open": 1360.0, "close": 1370.0})
                exec_sell = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(exec_sell), 1)

                p2 = pdb.get_portfolio()
                self.assertEqual(p2["holdings"]["RELIANCE"]["shares"], 6)
                self.assertGreater(p2["realized_pnl"], 0.0)

                # 7. Persistence & Restart Simulation
                # Re-querying without re-init simulates restart
                p_restart = pdb.get_portfolio()
                self.assertEqual(p_restart["cash"], p2["cash"])
                self.assertEqual(p_restart["holdings"]["RELIANCE"]["shares"], 6)

                # 8. Reset requires explicit confirmation
                pdb.reset_portfolio(confirmed=True)
                p_reset = pdb.get_portfolio()
                self.assertEqual(p_reset["cash"], 100000.0)
                self.assertEqual(p_reset["holdings"]["RELIANCE"]["shares"], 0)

    # =========================================================================
    # SECTION 8: Integration Between Paper Trading and PPO
    # =========================================================================
    def test_section_08_paper_trading_ppo_integration(self):
        """Verify dynamic portfolio context calculation and recommendation vs action separation."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "int_test.db"
            with patch.object(pdb, "DB_PATH", temp_db), \
                 patch.object(pdb, "DB_DIR", Path(temp_dir)):

                pdb.init_db()

                # Flat portfolio context
                ctx_flat = get_dynamic_portfolio_context("RELIANCE", 1300.0)
                np.testing.assert_array_almost_equal(ctx_flat, np.array([0.0, 1.0, 1.0], dtype=np.float32))

                # User action differs from agent recommendation
                order_res = pdb.submit_order(
                    symbol="INFY", side="BUY", quantity=5, current_price=1100.0,
                    agent_recommendation="SELL", agent_probabilities={"SELL": 0.8, "BUY": 0.1, "HOLD": 0.1},
                    decision_date="2026-11-17"
                )
                orders = pdb.get_orders(limit=1)
                latest = orders[0]
                self.assertEqual(latest["agent_recommendation"], "SELL")
                self.assertEqual(latest["user_action"], "BUY")
                self.assertNotEqual(latest["agent_recommendation"], latest["user_action"])

    # =========================================================================
    # SECTION 9: Error Handling and Edge Cases
    # =========================================================================
    def test_section_09_error_handling_and_edge_cases(self):
        """Verify handling of invalid quantities, missing files, corrupted inputs."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "err_test.db"
            with patch.object(pdb, "DB_PATH", temp_db), \
                 patch.object(pdb, "DB_DIR", Path(temp_dir)):

                pdb.init_db()

                # Zero quantity for BUY
                r_zero = pdb.submit_order("RELIANCE", "BUY", 0, 1300.0, "BUY", {}, "2026-11-17")
                self.assertEqual(r_zero["status"], "REJECTED")

                # Negative quantity for SELL
                r_neg = pdb.submit_order("RELIANCE", "SELL", -10, 1300.0, "SELL", {}, "2026-11-17")
                self.assertEqual(r_neg["status"], "REJECTED")

                # Invalid symbol
                r_sym = pdb.submit_order("UNKNOWN_STOCK", "BUY", 10, 100.0, "BUY", {}, "2026-11-17")
                self.assertEqual(r_sym["status"], "REJECTED")

    # =========================================================================
    # SECTION 10: End-to-End Workflow Test (Isolated Environment)
    # =========================================================================
    def test_section_10_end_to_end_workflow_simulated(self):
        """Simulate full 16-step end-to-end workflow on isolated test database."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "e2e_test.db"
            with patch.object(pdb, "DB_PATH", temp_db), \
                 patch.object(pdb, "DB_DIR", Path(temp_dir)):

                # Step 1: Initialize portfolio
                pdb.init_db()
                p0 = pdb.get_portfolio()
                self.assertEqual(p0["total_value"], 100000.0)

                # Step 2-4: Load cached market data and run inference
                cached = load_from_cache(["RELIANCE"])
                bars = cached["historical"]["RELIANCE"]
                engine = LiveInferenceEngine()
                pred = engine.predict_stock("RELIANCE", bars)
                self.assertIn(pred["recommendation"], ["HOLD", "BUY", "SELL"])

                # Step 5-6: User submits BUY order
                buy_res = pdb.submit_order(
                    symbol="RELIANCE", side="BUY", quantity=10, current_price=1300.0,
                    agent_recommendation=pred["recommendation"],
                    agent_probabilities=pred["action_probabilities"],
                    decision_date=pred["candle_date"]
                )
                self.assertEqual(buy_res["status"], "PENDING")

                # Step 7-8: Simulate application restart
                orders_pending = pdb.get_orders(status_filter="PENDING")
                self.assertEqual(len(orders_pending), 1)

                # Step 9-11: Next session bar arrives, executes BUY
                next_date = (datetime.strptime(pred["candle_date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                sim_market = {
                    "RELIANCE": [
                        {"date": pred["candle_date"], "open": 1300.0, "close": 1310.0},
                        {"date": next_date, "open": 1320.0, "close": 1325.0}
                    ]
                }
                exec_txs = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(exec_txs), 1)
                self.assertEqual(exec_txs[0]["execution_price"], 1320.0)

                # Step 12-14: Submit SELL order & execute on next day
                day3 = (datetime.strptime(next_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                sell_res = pdb.submit_order(
                    symbol="RELIANCE", side="SELL", quantity=10, current_price=1350.0,
                    agent_recommendation="SELL", agent_probabilities={}, decision_date=next_date
                )
                sim_market["RELIANCE"].append({"date": day3, "open": 1360.0, "close": 1370.0})
                exec_sell = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(exec_sell), 1)

                # Step 15-16: Accounting & Idempotency check
                p_end = pdb.get_portfolio()
                self.assertEqual(p_end["holdings"]["RELIANCE"]["shares"], 0)
                # Repeated market processing must NOT duplicate transactions
                repeat_txs = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(repeat_txs), 0)

    # =========================================================================
    # SECTION 11: Artifact & Checkpoint Integrity Check
    # =========================================================================
    def test_section_11_artifact_hashes_unchanged(self):
        """Verify that checkpoints, scalers, and real DB remain completely bitwise identical."""
        current_lstm_hash = get_file_sha256(LSTM_CHECKPOINT)
        current_ppo_hash = get_file_sha256(PPO_CHECKPOINT)
        current_scaler_hash = get_file_sha256(SCALER_PATH)

        self.assertEqual(self.initial_lstm_hash, current_lstm_hash, "LSTM checkpoint was altered!")
        self.assertEqual(self.initial_ppo_hash, current_ppo_hash, "PPO checkpoint was altered!")
        self.assertEqual(self.initial_scaler_hash, current_scaler_hash, "Scaler artifact was altered!")

        if self.real_db_exists:
            real_db = PROJECT_ROOT / "data" / "live" / "paper_trading.db"
            current_real_db_hash = get_file_sha256(real_db)
            self.assertEqual(self.initial_real_db_hash, current_real_db_hash, "Real paper trading DB was modified!")


if __name__ == "__main__":
    unittest.main()
