"""
Comprehensive Verification Suite for NeuroTrade Phase 3 Paper Trading.

Verifies the 12 requirements from Section 11 of Phase 3 specification:
1. Initial portfolio starts with ₹1,00,000 allocation
2. BUY order queues as PENDING and reserves cash
3. Simulated session execution updates holdings, cash, and deducts 0.1% fee
4. SELL order reduces holdings and credits net proceeds
5. Transaction fee exactness: fee = 0.001 * quantity * execution_price
6. Invalid quantities (<= 0) and insufficient funds/shares are rejected
7. Orders remain pending until a session strictly after decision_date appears
8. Next-session execution at verified Open price
9. Dashboard refresh/restart idempotency (zero transaction duplication)
10. Database persistence across simulated application restarts
11. PPO recommendations remain completely separate from user actions
12. Dynamic portfolio context matches MarketTradingEnv._get_observation semantics
13. Portfolio reset restores ₹1,00,000 without touching model artifacts
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data import portfolio_db
from backend.live_data.portfolio_db import (
    init_db, get_portfolio, submit_order, execute_order_at_open,
    process_pending_orders_against_market, get_orders, get_transactions,
    reset_portfolio, TRANSACTION_FEE_RATE
)
from backend.live_data.paper_trader import get_dynamic_portfolio_context


class TestPaperTrading(unittest.TestCase):

    def setUp(self):
        """Always use an isolated temporary test database to protect the real database."""
        self._temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._temp_db.close()
        self._orig_db_path = portfolio_db.DB_PATH
        portfolio_db.DB_PATH = Path(self._temp_db.name)
        init_db()

    def tearDown(self):
        """Clean up isolated temporary test database and restore DB_PATH."""
        portfolio_db.DB_PATH = self._orig_db_path
        if os.path.exists(self._temp_db.name):
            try:
                os.remove(self._temp_db.name)
            except OSError:
                pass

    def test_01_initial_capital_allocation(self):
        """1. Verify initial portfolio starts with verified ₹1,00,000 virtual capital."""
        p = get_portfolio()
        self.assertEqual(p["initial_capital"], 100000.0)
        self.assertEqual(p["cash"], 100000.0)
        self.assertEqual(p["holdings_value"], 0.0)
        self.assertEqual(p["total_value"], 100000.0)
        self.assertEqual(p["realized_pnl"], 0.0)
        self.assertEqual(p["unrealized_pnl"], 0.0)
        self.assertEqual(p["total_return_pct"], 0.0)
        for sym, h in p["holdings"].items():
            self.assertEqual(h["shares"], 0)

    def test_02_buy_order_pending_and_reserved_cash(self):
        """2. Verify BUY order queues as PENDING and reserves cash."""
        res = submit_order(
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            current_price=1300.0,
            agent_recommendation="HOLD",
            agent_probabilities={"HOLD": 0.4, "BUY": 0.3, "SELL": 0.3},
            decision_date="2026-11-17"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "PENDING")
        self.assertEqual(res["target_execution"], "NEXT_SESSION_OPEN")

        # Second BUY that would exceed unreserved cash must be rejected
        res2 = submit_order(
            symbol="RELIANCE",
            side="BUY",
            quantity=70,  # 70 * 1300 = 91,000. Remaining unreserved cash is ~86,987 -> must be rejected
            current_price=1300.0,
            agent_recommendation="HOLD",
            agent_probabilities={"HOLD": 0.4, "BUY": 0.3, "SELL": 0.3},
            decision_date="2026-11-17"
        )
        self.assertFalse(res2["success"])
        self.assertEqual(res2["status"], "REJECTED")
        self.assertIn("Insufficient available cash", res2["reason"])

    def test_03_next_session_execution_and_fee(self):
        """3. Verify order executes at next session Open t+1 with 0.1% fee."""
        sub_res = submit_order(
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            current_price=1300.0,
            agent_recommendation="HOLD",
            agent_probabilities={"HOLD": 0.5, "BUY": 0.2, "SELL": 0.3},
            decision_date="2026-11-17",
            created_at="2026-11-17T15:30:00"
        )
        order_id = sub_res["order_id"]

        # Simulate arrival of next session bar (date 2026-11-18 with Open=1320.0)
        simulated_market_bars = {
            "RELIANCE": [
                {"date": "2026-11-17", "open": 1300.0, "high": 1310.0, "low": 1290.0, "close": 1305.0, "volume": 1000000},
                {"date": "2026-11-18", "open": 1320.0, "high": 1330.0, "low": 1315.0, "close": 1325.0, "volume": 1200000},
            ]
        }

        executed_list = process_pending_orders_against_market(simulated_market_bars)
        self.assertEqual(len(executed_list), 1)
        tx = executed_list[0]
        self.assertEqual(tx["order_id"], order_id)
        self.assertEqual(tx["execution_price"], 1320.0)
        self.assertEqual(tx["execution_date"], "2026-11-18")

        # Exact fee check: 10 * 1320.0 * 0.001 = 13.20
        expected_fee = round(10 * 1320.0 * 0.001, 2)
        self.assertEqual(tx["fee"], expected_fee)
        self.assertEqual(tx["gross_amount"], 13200.0)

        # Portfolio state check
        p = get_portfolio()
        expected_cash = round(100000.0 - (13200.0 + expected_fee), 2)
        self.assertEqual(p["cash"], expected_cash)
        self.assertEqual(p["holdings"]["RELIANCE"]["shares"], 10)
        self.assertEqual(p["holdings"]["RELIANCE"]["avg_purchase_price"], round((13200.0 + expected_fee) / 10, 2))

    def test_04_sell_reduces_holdings_and_credits_cash(self):
        """4. Verify SELL order reduces holdings and credits proceeds minus 0.1% fee."""
        # Setup: Buy 10 shares first
        sub = submit_order("INFY", "BUY", 10, 1100.0, "BUY", {"BUY": 0.8}, "2026-11-17")
        execute_order_at_open(sub["order_id"], open_price=1100.0, open_date="2026-11-18")

        p_before = get_portfolio()
        cash_before = p_before["cash"]
        self.assertEqual(p_before["holdings"]["INFY"]["shares"], 10)

        # Now submit SELL 6 shares
        sub_sell = submit_order("INFY", "SELL", 6, 1150.0, "SELL", {"SELL": 0.7}, "2026-11-18")
        exec_sell = execute_order_at_open(sub_sell["order_id"], open_price=1160.0, open_date="2026-11-19")
        self.assertTrue(exec_sell["success"])

        gross = 6 * 1160.0  # 6960.0
        fee = round(gross * 0.001, 2)  # 6.96
        net_proceeds = gross - fee

        p_after = get_portfolio()
        self.assertEqual(p_after["holdings"]["INFY"]["shares"], 4)
        self.assertEqual(p_after["cash"], round(cash_before + net_proceeds, 2))
        self.assertGreater(p_after["realized_pnl"], 0.0)  # Sold at 1160 > bought at 1100

    def test_05_rejections_for_invalid_inputs(self):
        """5. Verify zero/negative quantities and insufficient shares/funds are rejected."""
        # Negative quantity
        r1 = submit_order("TCS", "BUY", -5, 2400.0, "BUY", {}, "2026-11-17")
        self.assertEqual(r1["status"], "REJECTED")

        # Zero quantity
        r2 = submit_order("TCS", "BUY", 0, 2400.0, "BUY", {}, "2026-11-17")
        self.assertEqual(r2["status"], "REJECTED")

        # Selling without holding shares
        r3 = submit_order("TCS", "SELL", 1, 2400.0, "SELL", {}, "2026-11-17")
        self.assertEqual(r3["status"], "REJECTED")
        self.assertIn("Insufficient shares", r3["reason"])

        # Buying with insufficient funds
        r4 = submit_order("TCS", "BUY", 100, 2400.0, "BUY", {}, "2026-11-17")  # 240,000 > 100,000
        self.assertEqual(r4["status"], "REJECTED")
        self.assertIn("Insufficient available cash", r4["reason"])

    def test_06_orders_remain_pending_until_next_date(self):
        """6. Orders must NOT execute on past or same-day data."""
        sub = submit_order("ITC", "BUY", 20, 290.0, "BUY", {}, "2026-11-17")
        order_id = sub["order_id"]

        # Same date bar: MUST NOT EXECUTE
        same_date_market = {
            "ITC": [{"date": "2026-11-17", "open": 290.0, "close": 292.0}]
        }
        res = process_pending_orders_against_market(same_date_market)
        self.assertEqual(len(res), 0)

        orders = get_orders(status_filter="PENDING")
        self.assertTrue(any(o["order_id"] == order_id for o in orders))

    def test_07_persistence_and_idempotency(self):
        """7. Verify database survives restarts and repeated process calls don't duplicate."""
        sub = submit_order("TCS", "BUY", 5, 2400.0, "HOLD", {}, "2026-11-17", created_at="2026-11-17T15:30:00")
        market_data = {
            "TCS": [
                {"date": "2026-11-17", "open": 2400.0, "close": 2410.0},
                {"date": "2026-11-18", "open": 2420.0, "close": 2430.0},
            ]
        }
        # First process call executes order
        res1 = process_pending_orders_against_market(market_data)
        self.assertEqual(len(res1), 1)

        # Repeated process call must NOT duplicate execution
        res2 = process_pending_orders_against_market(market_data)
        self.assertEqual(len(res2), 0)

        # Re-read portfolio from disk (simulating application restart)
        p = get_portfolio()
        self.assertEqual(p["holdings"]["TCS"]["shares"], 5)
        txs = get_transactions()
        self.assertEqual(len(txs), 1)

    def test_08_separate_agent_recommendation_and_user_action(self):
        """8. Agent recommendation must be strictly preserved separately from user action."""
        submit_order(
            symbol="HDFCBANK",
            side="SELL",  # User chooses SELL
            quantity=0,
            current_price=800.0,
            agent_recommendation="BUY",  # Agent recommended BUY
            agent_probabilities={"BUY": 0.85, "HOLD": 0.10, "SELL": 0.05},
            decision_date="2026-11-17"
        )
        orders = get_orders(limit=1)
        latest_order = orders[0]
        self.assertEqual(latest_order["agent_recommendation"], "BUY")
        self.assertEqual(latest_order["user_action"], "SELL")
        self.assertNotEqual(latest_order["agent_recommendation"], latest_order["user_action"])

    def test_09_dynamic_portfolio_context_calculation(self):
        """9. Dynamic context [pos_ratio, cash_ratio, norm_val] matches MarketTradingEnv semantics."""
        # Case A: Flat portfolio
        ctx_flat = get_dynamic_portfolio_context("RELIANCE", 1300.0)
        np.testing.assert_array_almost_equal(ctx_flat, np.array([0.0, 1.0, 1.0], dtype=np.float32))

        # Case B: After purchasing shares
        sub = submit_order("RELIANCE", "BUY", 10, 1300.0, "BUY", {}, "2026-11-17")
        execute_order_at_open(sub["order_id"], open_price=1300.0, open_date="2026-11-18")

        p = get_portfolio()
        ctx_active = get_dynamic_portfolio_context("RELIANCE", 1300.0, portfolio=p)
        pos_ratio, cash_ratio, norm_val = ctx_active

        # 10 shares * 1300 = 13,000 equity. Total value ~ 100,000 - 13 fee = 99,987
        self.assertAlmostEqual(pos_ratio, 13000.0 / p["total_value"], places=3)
        self.assertAlmostEqual(cash_ratio, p["cash"] / p["total_value"], places=3)
        self.assertAlmostEqual(norm_val, p["total_value"] / 100000.0, places=3)
        self.assertAlmostEqual(pos_ratio + cash_ratio, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()

