"""
Phase 3.5 Comprehensive System Reliability & Integration Audit.
Executes detailed edge cases and integration scenarios across Sections 4 to 13.
Uses isolated temporary databases and test environments to guarantee zero impact on real user data.
"""

import os
import sys
import gc
import json
import sqlite3
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

import backend.live_data.portfolio_db as pdb
from backend.live_data.paper_trader import get_dynamic_portfolio_context
from backend.live_data.inference_engine import LiveInferenceEngine, ACTION_NAMES
from backend.live_data.feature_pipeline import (
    compute_features, prepare_inference_sequence, STATE_FEATURE_COLS
)
from backend.live_data.market_cache import (
    get_cache_status, load_from_cache, save_to_cache, CACHE_TTL_SECONDS
)


class Phase35ReliabilityAudit(unittest.TestCase):

    # =========================================================================
    # 1. FRONTEND AND BACKEND INTEGRATION AUDIT (Section 4)
    # =========================================================================
    def test_frontend_backend_disconnection_root_cause(self):
        """
        Investigate dashboard-to-CLI order persistence failure:
        Confirm that HTML dashboard submitPaperOrder() is client-side only and never sends an HTTP request.
        """
        dashboard_html_path = PROJECT_ROOT / "reports" / "live_trading_dashboard.html"
        self.assertTrue(dashboard_html_path.exists(), "Dashboard HTML must exist")
        html_content = dashboard_html_path.read_text(encoding="utf-8")

        # Check for the exact user-reported message
        reported_str = "Order submitted successfully! Order is logged in SQLite database as PENDING execution"
        self.assertIn(reported_str, html_content, "The user alert string must be in the HTML file")

        # Check if there is ANY network call in submitPaperOrder
        has_network_call = any(keyword in html_content for keyword in ["fetch(", "XMLHttpRequest", "axios", "$.ajax", "navigator.sendBeacon"])
        self.assertFalse(has_network_call, "ROOT CAUSE CONFIRMED: HTML dashboard contains NO network communication to backend.")

    # =========================================================================
    # 2. DATABASE AND PERSISTENCE EDGE CASES (Section 5)
    # =========================================================================
    def test_database_edge_cases(self):
        """Test missing DB auto-creation, schema idempotency, and transaction rollback."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "test_edge.db"
            with patch.object(pdb, "DB_PATH", temp_db), patch.object(pdb, "DB_DIR", Path(temp_dir)):
                # A. Missing DB file -> auto-creates schema and seeds ₹100,000
                self.assertFalse(temp_db.exists())
                pdb.init_db()
                self.assertTrue(temp_db.exists())
                p = pdb.get_portfolio()
                self.assertEqual(p["initial_capital"], 100000.0)

                # B. Schema re-initialization does NOT reset existing records
                # Add an order
                res = pdb.submit_order("RELIANCE", "BUY", 5, 1300.0, "BUY", {}, "2026-11-17")
                self.assertTrue(res["success"])
                # Re-run init_db
                pdb.init_db()
                orders = pdb.get_orders()
                self.assertEqual(len(orders), 1, "Schema init must not wipe existing orders!")

                # C. Atomicity on failure during execution (rollback test)
                # Attempt to execute an invalid order ID
                res_bad = pdb.execute_order_at_open(order_id=99999, open_price=1300.0, open_date="2026-11-18")
                self.assertFalse(res_bad["success"])
                p_after_bad = pdb.get_portfolio()
                self.assertEqual(p_after_bad["cash"], 100000.0, "Failed execution must not alter cash balance")

    # =========================================================================
    # 3. MARKET-DATA AND CACHING EDGE CASES (Section 6)
    # =========================================================================
    def test_market_data_partial_and_malformed(self):
        """Test behavior when one stock data is missing or dates are out of order."""
        # Out-of-order dates & duplicate candles
        shuffled_bars = [
            {"date": "2026-11-17", "open": 100.0, "high": 105.0, "low": 98.0, "close": 104.0, "volume": 1000},
            {"date": "2026-11-15", "open": 98.0, "high": 102.0, "low": 97.0, "close": 99.0, "volume": 1100},
            {"date": "2026-11-16", "open": 99.0, "high": 101.0, "low": 96.0, "close": 100.0, "volume": 1050},
            {"date": "2026-11-16", "open": 99.0, "high": 101.0, "low": 96.0, "close": 100.0, "volume": 1050}, # Duplicate
        ]
        df = compute_features(shuffled_bars)
        # Should be deduplicated and sorted ascending
        self.assertEqual(len(df), 3)
        self.assertEqual(df["Date"].iloc[0], pd.Timestamp("2026-11-15"))
        self.assertEqual(df["Date"].iloc[-1], pd.Timestamp("2026-11-17"))

    # =========================================================================
    # 4. FEATURE ENGINEERING AND MODEL INFERENCE EDGE CASES (Section 7)
    # =========================================================================
    def test_feature_warmup_and_zero_volume_handling(self):
        """Verify handling of zero volume and rolling window warmup."""
        # 1. Warmup constraint: Need at least 80 bars for 50-day SMA + 30-day window
        short_bars = [
            {"date": (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"), "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}
            for i in range(39) # Only 39 valid date bars
        ]
        engine = LiveInferenceEngine()
        with self.assertRaises(ValueError) as ctx:
            engine.predict_stock("RELIANCE", short_bars)
        self.assertIn("Insufficient historical bars", str(ctx.exception))

        # 2. Dynamic portfolio context edge cases
        # Fully invested (cash = 0)
        p_full = {"initial_capital": 100000.0, "cash": 0.0, "total_value": 120000.0, "holdings": {"RELIANCE": {"shares": 100}}}
        ctx_full = get_dynamic_portfolio_context("RELIANCE", 1200.0, portfolio=p_full)
        self.assertAlmostEqual(ctx_full[0], 1.0) # 100% position ratio
        self.assertAlmostEqual(ctx_full[1], 0.0) # 0% cash ratio
        self.assertAlmostEqual(ctx_full[2], 1.2) # norm value = 1.2

        # Massive loss (portfolio drops to 20,000)
        p_loss = {"initial_capital": 100000.0, "cash": 10000.0, "total_value": 20000.0, "holdings": {"RELIANCE": {"shares": 10}}}
        ctx_loss = get_dynamic_portfolio_context("RELIANCE", 1000.0, portfolio=p_loss)
        self.assertAlmostEqual(ctx_loss[0], 0.5) # 10000 / 20000
        self.assertAlmostEqual(ctx_loss[1], 0.5) # 10000 / 20000
        self.assertAlmostEqual(ctx_loss[2], 0.2) # 20000 / 100000

    # =========================================================================
    # 5. PAPER-TRADING ACCOUNTING & ORDER LIFECYCLE (Section 8: Scenarios A-G)
    # =========================================================================
    def test_accounting_scenarios_a_through_g(self):
        """Test Scenarios A through G on isolated test database."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_db = Path(temp_dir) / "test_accounting.db"
            with patch.object(pdb, "DB_PATH", temp_db), patch.object(pdb, "DB_DIR", Path(temp_dir)):
                pdb.init_db(initial_capital=100000.0)

                # Scenario A: Basic BUY
                # Submit BUY 10 RELIANCE @ 1300.0
                r_buy = pdb.submit_order("RELIANCE", "BUY", 10, 1300.0, "HOLD", {}, "2026-11-17")
                self.assertTrue(r_buy["success"])
                # Execute at Open 1310.0 on 2026-11-18
                sim_market = {"RELIANCE": [{"date": "2026-11-17", "open": 1300.0, "close": 1305.0}, {"date": "2026-11-18", "open": 1310.0, "close": 1315.0}]}
                execs = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(execs), 1)
                tx_buy = execs[0]
                expected_fee_buy = round(10 * 1310.0 * 0.001, 2)  # 13.10
                self.assertEqual(tx_buy["fee"], expected_fee_buy)

                p = pdb.get_portfolio()
                expected_cash_after_buy = round(100000.0 - (13100.0 + expected_fee_buy), 2)
                self.assertEqual(p["cash"], expected_cash_after_buy)
                self.assertEqual(p["holdings"]["RELIANCE"]["shares"], 10)

                # Scenario B: Multiple BUY orders for different stocks (Shared Cash Reservation)
                r_tcs = pdb.submit_order("TCS", "BUY", 10, 2400.0, "BUY", {}, "2026-11-18")  # 24,000 + 24 fee = 24,024
                r_infy = pdb.submit_order("INFY", "BUY", 20, 1100.0, "BUY", {}, "2026-11-18") # 22,000 + 22 fee = 22,022
                self.assertTrue(r_tcs["success"])
                self.assertTrue(r_infy["success"])

                # Scenario D: Competing pending order that exceeds remaining available cash
                # Total cash is ~86,886.90. Reserved is ~46,046. Available is ~40,840.
                # Try submitting BUY 50 TCS @ 2400 (requires 120,120 > 40,840) -> Must be REJECTED!
                r_over = pdb.submit_order("TCS", "BUY", 50, 2400.0, "BUY", {}, "2026-11-18")
                self.assertEqual(r_over["status"], "REJECTED")
                self.assertIn("Insufficient available cash", r_over["reason"])

                # Execute TCS & INFY at next session Open
                sim_market["TCS"] = [{"date": "2026-11-18", "open": 2400.0, "close": 2405.0}, {"date": "2026-11-19", "open": 2420.0, "close": 2430.0}]
                sim_market["INFY"] = [{"date": "2026-11-18", "open": 1100.0, "close": 1105.0}, {"date": "2026-11-19", "open": 1110.0, "close": 1115.0}]
                execs_multi = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(execs_multi), 2)

                # Scenario C: SELL shares
                # Sell 5 shares of RELIANCE (bought 10 earlier at avg cost 1311.31)
                r_sell = pdb.submit_order("RELIANCE", "SELL", 5, 1350.0, "SELL", {}, "2026-11-19")
                self.assertTrue(r_sell["success"])
                # Execute SELL at 2026-11-20 Open = 1380.0
                sim_market["RELIANCE"].append({"date": "2026-11-20", "open": 1380.0, "close": 1385.0})
                execs_sell = pdb.process_pending_orders_against_market(sim_market)
                self.assertEqual(len(execs_sell), 1)
                tx_sell = execs_sell[0]
                expected_fee_sell = round(5 * 1380.0 * 0.001, 2)  # 6.90
                expected_gross_sell = 5 * 1380.0                   # 6900.00
                expected_net_sell = expected_gross_sell - expected_fee_sell  # 6893.10
                self.assertEqual(tx_sell["fee"], expected_fee_sell)

                p_after_sell = pdb.get_portfolio()
                self.assertEqual(p_after_sell["holdings"]["RELIANCE"]["shares"], 5)
                self.assertGreater(p_after_sell["realized_pnl"], 0.0)

                # Scenario E: Order Cancellation Limitation Check
                self.assertFalse(hasattr(pdb, "cancel_order"), "Order cancellation is not implemented in portfolio_db.")

                # Scenario F: Execution failure (same-day candle cannot execute)
                r_same = pdb.submit_order("ITC", "BUY", 10, 290.0, "BUY", {}, "2026-11-20")
                sim_same = {"ITC": [{"date": "2026-11-20", "open": 290.0, "close": 292.0}]}
                execs_same = pdb.process_pending_orders_against_market(sim_same)
                self.assertEqual(len(execs_same), 0, "Order must NOT execute on same-day candle!")

                # Scenario G: Portfolio Accounting Reconciliation
                # Reconcile mark-to-market prices
                pdb.update_holdings_prices({"RELIANCE": 1380.0, "TCS": 2420.0, "INFY": 1110.0, "ITC": 290.0, "HDFCBANK": 810.0})
                p_final = pdb.get_portfolio()
                
                # Check fundamental accounting balance:
                # Total Portfolio Value = Cash + Holdings Market Value
                computed_holdings = sum(h["current_value"] for h in p_final["holdings"].values())
                self.assertAlmostEqual(p_final["total_value"], p_final["cash"] + computed_holdings, places=2)
                # Total P&L = Realized P&L + Unrealized P&L
                self.assertAlmostEqual(p_final["total_pnl"], p_final["realized_pnl"] + p_final["unrealized_pnl"], places=2)
                # Overall Return % = (Total Value - Initial Capital) / Initial Capital * 100
                expected_return = (p_final["total_value"] - 100000.0) / 100000.0 * 100.0
                self.assertAlmostEqual(p_final["total_return_pct"], expected_return, places=2)

    # =========================================================================
    # 6. APPLICATION STARTUP FROM DIFFERENT CWD (Section 11)
    # =========================================================================
    def test_different_working_directory(self):
        """Verify that DB_PATH resolves to the absolute root regardless of current working directory."""
        from backend.live_data.portfolio_db import DB_PATH
        self.assertTrue(DB_PATH.is_absolute())
        self.assertEqual(DB_PATH, PROJECT_ROOT / "data" / "live" / "paper_trading.db")

    # =========================================================================
    # 7. SECURITY & SQL PARAMETERIZATION (Section 12)
    # =========================================================================
    def test_sql_parameterization(self):
        """Audit SQL queries in portfolio_db.py for parameterization (zero string formatting)."""
        pdb_source = (PROJECT_ROOT / "backend" / "live_data" / "portfolio_db.py").read_text(encoding="utf-8")
        
        # Ensure no query formatting like: f"SELECT ... WHERE symbol = '{symbol}'"
        has_fstring_sql = "f\"SELECT" in pdb_source or "f\"INSERT" in pdb_source or "f\"UPDATE" in pdb_source
        self.assertFalse(has_fstring_sql, "All SQL queries in portfolio_db must use ? parameterization")


if __name__ == "__main__":
    unittest.main()
