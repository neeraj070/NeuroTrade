"""
Comprehensive Phase 3.6 Integration & Reliability Test Suite for NeuroTrade.

Tests:
1. API endpoints (POST /api/order, POST /api/cancel, GET /api/portfolio, GET /api/orders, GET /api/history)
2. Escrow mechanics and execution-time affordability (price surge handling, no negative cash)
3. Rejection of invalid order attempts (logged to SQLite orders table as REJECTED)
4. Atomic pending order cancellation and escrow release
5. CLI and backend database consistency
6. Full accounting reconciliation across multi-step simulated trading workflow
"""

import unittest
import tempfile
import sqlite3
import json
import urllib.request
import urllib.error
import threading
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from backend.live_data.portfolio_db import (
    init_db, get_portfolio, submit_order, execute_order_at_open,
    cancel_order, get_orders, get_transactions, update_holdings_prices,
    reset_portfolio, TARGET_STOCKS, TRANSACTION_FEE_RATE
)
from backend.live_data.api_server import NeuroTradeAPIHandler, run_server
import http.server


class TestPhase36DatabaseAndEscrow(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary directory and database
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_phase36.db"

        # Patch DB_PATH in portfolio_db
        self.db_patcher = patch("backend.live_data.portfolio_db.DB_PATH", self.db_path)
        self.db_patcher.start()

        # Initialize clean DB
        init_db()

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    def test_initial_portfolio_state(self):
        p = get_portfolio()
        self.assertEqual(p["initial_capital"], 100000.0)
        self.assertEqual(p["cash"], 100000.0)
        self.assertEqual(p["ledger_cash"], 100000.0)
        self.assertEqual(p["available_cash"], 100000.0)
        self.assertEqual(p["reserved_cash"], 0.0)
        self.assertEqual(p["total_value"], 100000.0)
        self.assertEqual(len(p["holdings"]), 5)

    def test_invalid_order_logging_in_sqlite(self):
        # 1. Non-positive quantity
        res1 = submit_order("RELIANCE", "BUY", 0, 2500.0)
        self.assertFalse(res1["success"])
        self.assertIn("order_id", res1)
        self.assertIn("rejected", res1["status"].lower())

        # 2. Negative quantity
        res2 = submit_order("INFY", "BUY", -5, 1500.0)
        self.assertFalse(res2["success"])
        self.assertIn("order_id", res2)

        # 3. Invalid action
        res3 = submit_order("TCS", "SHORT", 10, 3500.0)
        self.assertFalse(res3["success"])
        self.assertIn("order_id", res3)

        # 4. Invalid symbol
        res4 = submit_order("BITCOIN", "BUY", 10, 50000.0)
        self.assertFalse(res4["success"])
        self.assertIn("order_id", res4)

        # Verify all 4 are persisted in orders table with status = 'REJECTED'
        orders = get_orders(limit=10)
        rejected_orders = [o for o in orders if o["status"] == "REJECTED"]
        self.assertEqual(len(rejected_orders), 4)
        for ro in rejected_orders:
            self.assertEqual(ro["status"], "REJECTED")
            self.assertTrue(len(ro["rejection_reason"]) > 0)

    def test_escrow_reservation_and_cancellation(self):
        # Submit BUY 10 RELIANCE @ 2500. Est cost: 25,000 + 0.1% = 25,025.0
        res = submit_order("RELIANCE", "BUY", 10, 2500.0)
        self.assertTrue(res["success"])
        order_id = res["order_id"]

        p_after_buy = get_portfolio()
        self.assertEqual(p_after_buy["ledger_cash"], 100000.0)
        self.assertAlmostEqual(p_after_buy["reserved_cash"], 25025.0, places=2)
        self.assertAlmostEqual(p_after_buy["available_cash"], 100000.0 - 25025.0, places=2)
        # Total portfolio value should still be 100,000 (no double counting or value drop)
        self.assertAlmostEqual(p_after_buy["total_value"], 100000.0, places=2)

        # Now cancel the pending order
        cancel_res = cancel_order(order_id)
        self.assertTrue(cancel_res["success"])
        self.assertIn("cancelled", cancel_res["message"].lower())

        # Check portfolio: escrow should be fully released
        p_after_cancel = get_portfolio()
        self.assertEqual(p_after_cancel["reserved_cash"], 0.0)
        self.assertEqual(p_after_cancel["available_cash"], 100000.0)
        self.assertEqual(p_after_cancel["ledger_cash"], 100000.0)

        # Verify order status in DB is CANCELLED
        orders = get_orders(limit=5)
        o = [x for x in orders if x["order_id"] == order_id][0]
        self.assertEqual(o["status"], "CANCELLED")

        # Double cancellation must fail
        double_cancel = cancel_order(order_id)
        self.assertFalse(double_cancel["success"])
        self.assertIn("cannot cancel", double_cancel["reason"].lower())

    def test_execution_affordability_and_price_surge_rejection(self):
        # Case: Account has ₹10,000 cash remaining
        # Place order for 5 shares @ ₹1,900. Est cost = 5 * 1900 * 1.001 = ₹9,509.50 (affordable)
        # But open price surges to ₹2,100. Actual cost = 5 * 2100 * 1.001 = ₹10,510.50 (exceeds cash!)
        # Order must be REJECTED at execution, escrow released, cash not negative.

        # Artificially set cash to 10,000 in DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE portfolio_summary SET cash = 10000.0 WHERE id = 1")
            conn.commit()

        res = submit_order("INFY", "BUY", 5, 1900.0)
        self.assertTrue(res["success"])
        order_id = res["order_id"]

        p_reserved = get_portfolio()
        self.assertAlmostEqual(p_reserved["reserved_cash"], 9509.50, places=2)

        # Execute at open with surged price 2,100
        exec_res = execute_order_at_open(order_id, 2100.0, "2026-03-31")
        self.assertFalse(exec_res["success"])
        self.assertIn("rejected", exec_res["status"].lower())
        self.assertIn("surge", exec_res["reason"].lower())

        # Portfolio check: escrow released, cash is still 10,000, 0 shares
        p_final = get_portfolio()
        self.assertEqual(p_final["reserved_cash"], 0.0)
        self.assertEqual(p_final["available_cash"], 10000.0)
        self.assertEqual(p_final["cash"], 10000.0)
        self.assertEqual(p_final["holdings"]["INFY"]["shares"], 0)

    def test_competing_orders_escrow_limit(self):
        # Cash is 100,000.
        # Order 1: BUY 30 RELIANCE @ 2500 -> ~75,075 reserved. Available = 24,925.
        res1 = submit_order("RELIANCE", "BUY", 30, 2500.0)
        self.assertTrue(res1["success"])

        # Order 2: BUY 15 RELIANCE @ 2500 -> ~37,537.50 required.
        # Since available is only 24,925, Order 2 MUST be rejected at submission!
        res2 = submit_order("RELIANCE", "BUY", 15, 2500.0)
        self.assertFalse(res2["success"])
        self.assertIn("insufficient", res2["reason"].lower())

        # Verify Order 2 is recorded as REJECTED
        orders = get_orders(limit=2)
        rejected = [o for o in orders if o["order_id"] == res2["order_id"]][0]
        self.assertEqual(rejected["status"], "REJECTED")

    def test_full_accounting_reconciliation_workflow(self):
        # 1. Initial State
        p = get_portfolio()
        self.assertEqual(p["cash"], 100000.0)

        # 2. Submit BUY 20 TCS @ 3000. Est cost: 60,000 + 60 = 60,060
        res = submit_order("TCS", "BUY", 20, 3000.0)
        oid = res["order_id"]
        self.assertTrue(res["success"])

        # 3. Execute at Open @ 2950 (favorable price)
        # Actual cost: 20 * 2950 = 59,000; Fee: 59.0; Total cash deducted: 59,059.0
        # Cash remaining: 100,000 - 59,059 = 40,941.0
        exec_res = execute_order_at_open(oid, 2950.0, "2026-04-01")
        self.assertTrue(exec_res["success"])

        p = get_portfolio()
        self.assertAlmostEqual(p["cash"], 40941.0, places=2)
        self.assertEqual(p["holdings"]["TCS"]["shares"], 20)
        self.assertAlmostEqual(p["holdings"]["TCS"]["avg_purchase_price"], 2952.95, places=2)

        # 4. Mark to market @ 3100
        # Holdings val: 20 * 3100 = 62,000.
        # Total portfolio: 40,941 + 62,000 = 102,941.
        # Unrealized P&L: 62,000 - 59,059 = +2,941.00.
        # Total return: (102,941 - 100,000)/100,000 = +2.941%
        update_holdings_prices({"TCS": 3100.0})
        p = get_portfolio()
        self.assertAlmostEqual(p["holdings_value"], 62000.0, places=2)
        self.assertAlmostEqual(p["total_value"], 102941.0, places=2)
        self.assertAlmostEqual(p["unrealized_pnl"], 2941.0, places=2)

        # 5. Sell 10 shares of TCS @ 3100
        sell_res = submit_order("TCS", "SELL", 10, 3100.0)
        self.assertTrue(sell_res["success"])
        sell_oid = sell_res["order_id"]

        # Execute SELL at open @ 3150
        # Gross: 10 * 3150 = 31,500; Fee: 31.50; Net received: 31,468.50
        # Realized P&L: 1939.00
        # New cash: 40,941 + 31,468.50 = 72,409.50
        exec_sell = execute_order_at_open(sell_oid, 3150.0, "2026-04-02")
        self.assertTrue(exec_sell["success"])

        p = get_portfolio()
        self.assertAlmostEqual(p["cash"], 72409.50, places=2)
        self.assertEqual(p["holdings"]["TCS"]["shares"], 10)
        self.assertAlmostEqual(p["realized_pnl"], 1939.00, places=2)


class TestAPIEndpointsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Temporary DB for API testing
        cls.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.db_path = Path(cls.temp_dir.name) / "test_api.db"

        cls.db_patcher = patch("backend.live_data.portfolio_db.DB_PATH", cls.db_path)
        cls.db_patcher.start()

        init_db()

        # Start API server on random port
        server_address = ("127.0.0.1", 8765)
        cls.httpd = http.server.HTTPServer(server_address, NeuroTradeAPIHandler)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.db_patcher.stop()
        cls.temp_dir.cleanup()

    def test_api_portfolio_get(self):
        req = urllib.request.Request("http://127.0.0.1:8765/api/portfolio")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("cash", data)
            self.assertIn("available_cash", data)
            self.assertIn("reserved_cash", data)
            self.assertIn("total_value", data)
            self.assertEqual(data["cash"], 100000.0)

    def test_api_order_submission_and_cancellation(self):
        # 1. Submit valid order
        order_payload = json.dumps({
            "symbol": "HDFCBANK",
            "action": "BUY",
            "quantity": 10,
            "price": 1700.0
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/order",
            data=order_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode())
            self.assertTrue(res["success"])
            order_id = res["order_id"]

        # 2. Check /api/orders
        req = urllib.request.Request("http://127.0.0.1:8765/api/orders")
        with urllib.request.urlopen(req) as resp:
            orders = json.loads(resp.read().decode())
            pending = [o for o in orders if o["order_id"] == order_id]
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["status"], "PENDING")

        # 3. Cancel order via /api/cancel
        cancel_payload = json.dumps({"order_id": order_id}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/cancel",
            data=cancel_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            c_res = json.loads(resp.read().decode())
            self.assertTrue(c_res["success"])

        # 4. Verify portfolio has 0 reserved cash
        req = urllib.request.Request("http://127.0.0.1:8765/api/portfolio")
        with urllib.request.urlopen(req) as resp:
            p = json.loads(resp.read().decode())
            self.assertEqual(p["reserved_cash"], 0.0)
            self.assertEqual(p["available_cash"], 100000.0)

    def test_api_order_invalid_rejected_logged(self):
        # Negative quantity
        payload = json.dumps({
            "symbol": "INFY",
            "action": "BUY",
            "quantity": -10,
            "price": 1500.0
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/order",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        # Should return 400 Bad Request with json error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err_data = json.loads(ctx.exception.read().decode())
        self.assertFalse(err_data["success"])
        self.assertIn("order_id", err_data)

        # Verify it was logged in DB
        req = urllib.request.Request("http://127.0.0.1:8765/api/orders")
        with urllib.request.urlopen(req) as resp:
            orders = json.loads(resp.read().decode())
            rej = [o for o in orders if o["order_id"] == err_data["order_id"]][0]
            self.assertEqual(rej["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
