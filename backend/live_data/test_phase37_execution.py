"""
Comprehensive Phase 3.7 Execution Lifecycle & Integrity Test Suite for NeuroTrade.

Covers all 17 requirements:
1. Pending BUY executes at the first eligible subsequent session Open.
2. Pending SELL executes at the first eligible subsequent session Open.
3. An order cannot execute using a candle dated before its submission.
4. Stale market data does not cause retrospective execution.
5. Weekend gaps are handled correctly (Friday -> Monday Open).
6. Holiday gaps are handled using actual available trading sessions.
7. Insufficient actual BUY funds cause rejection.
8. Other pending BUY reservations remain protected during execution.
9. Insufficient SELL shares cause rejection.
10. Other pending SELL reservations remain protected during execution.
11. Rejected orders release escrow automatically.
12. Cancelled orders release escrow automatically.
13. Repeated processing does not duplicate transactions.
14. Concurrent processing cannot execute the same order twice (thread-safe).
15. API submission rejects untrusted browser prices.
16. API execution endpoint (/api/process-pending) uses backend-verified market data.
17. Portfolio accounting remains consistent after multiple executions.
"""

import unittest
import tempfile
import sqlite3
import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from backend.live_data.portfolio_db import (
    init_db, get_portfolio, submit_order, execute_order_at_open,
    cancel_order, get_orders, get_transactions, update_holdings_prices,
    process_pending_orders_service, process_pending_orders_against_market,
    TARGET_STOCKS, TRANSACTION_FEE_RATE
)
from backend.live_data.api_server import NeuroTradeAPIHandler
import http.server


class TestPhase37ExecutionLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_p37.db"
        self.db_patcher = patch("backend.live_data.portfolio_db.DB_PATH", self.db_path)
        self.db_patcher.start()
        init_db()

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    # 1. Pending BUY executes at the first eligible subsequent session Open
    def test_01_pending_buy_executes_at_subsequent_open(self):
        # Order submitted on 2026-11-20
        res = submit_order("RELIANCE", "BUY", 10, 1300.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        self.assertTrue(res["success"])
        oid = res["order_id"]

        market_bars = {
            "RELIANCE": [
                {"date": "2026-11-19", "open": 1280.0, "close": 1290.0},
                {"date": "2026-11-20", "open": 1295.0, "close": 1300.0},
                {"date": "2026-11-21", "open": 1310.0, "close": 1320.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 1)
        tx = exec_res["executed_orders"][0]
        self.assertEqual(tx["order_id"], oid)
        self.assertEqual(tx["execution_price"], 1310.0)
        self.assertEqual(tx["execution_date"], "2026-11-21")

        p = get_portfolio()
        self.assertEqual(p["holdings"]["RELIANCE"]["shares"], 10)
        self.assertEqual(p["reserved_cash"], 0.0)

    # 2. Pending SELL executes at the first eligible subsequent session Open
    def test_02_pending_sell_executes_at_subsequent_open(self):
        # Setup: give portfolio 10 shares of TCS
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            UPDATE holdings
            SET shares = 10, avg_purchase_price = 2400.0, total_invested = 24024.0, current_price = 2400.0, current_value = 24000.0
            WHERE symbol = 'TCS';
            """)
            conn.commit()

        res = submit_order("TCS", "SELL", 5, 2400.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        self.assertTrue(res["success"])
        oid = res["order_id"]

        market_bars = {
            "TCS": [
                {"date": "2026-11-20", "open": 2390.0, "close": 2400.0},
                {"date": "2026-11-21", "open": 2420.0, "close": 2430.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 1)
        tx = exec_res["executed_orders"][0]
        self.assertEqual(tx["order_id"], oid)
        self.assertEqual(tx["execution_price"], 2420.0)

        p = get_portfolio()
        self.assertEqual(p["holdings"]["TCS"]["shares"], 5)

    # 3. An order cannot execute using a candle dated before its submission
    def test_03_cannot_execute_using_candle_before_submission(self):
        # Submission date is 2026-11-20
        res = submit_order("INFY", "BUY", 10, 1100.0, decision_date="2026-11-20")
        oid = res["order_id"]

        # Only older bars available
        market_bars = {
            "INFY": [
                {"date": "2026-11-17", "open": 1080.0, "close": 1090.0},
                {"date": "2026-11-18", "open": 1090.0, "close": 1100.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 0)
        self.assertEqual(exec_res["deferred_count"], 1)

        orders = get_orders(limit=5)
        o = [x for x in orders if x["order_id"] == oid][0]
        self.assertEqual(o["status"], "PENDING")

    # 4. Stale market data does not cause retrospective execution
    def test_04_stale_data_does_not_cause_retrospective_execution(self):
        # Order created today (e.g. 2026-11-20), even if decision_date was 2026-11-17 (stale candle)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO orders (symbol, side, quantity, decision_date, decision_time, agent_recommendation,
                                agent_probabilities, user_action, estimated_price, status, created_at)
            VALUES ('INFY', 'BUY', 10, '2026-11-17', '12:00:00', 'HOLD', '{}', 'BUY', 1130.0, 'PENDING', '2026-11-20T12:00:00');
            """)
            conn.commit()

        # Market data has a bar from 2026-11-18 (after 11-17, but BEFORE order submission 11-20)
        market_bars = {
            "INFY": [
                {"date": "2026-11-17", "open": 1120.0, "close": 1130.0},
                {"date": "2026-11-18", "open": 1135.0, "close": 1140.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 0)
        self.assertEqual(exec_res["deferred_count"], 1)

    # 5. Weekend gaps are handled correctly (Friday -> Monday Open)
    def test_05_weekend_gap_handled_correctly(self):
        # Friday 2026-11-20 submission
        res = submit_order("ITC", "BUY", 20, 290.0, decision_date="2026-11-20")
        oid = res["order_id"]

        # Next session is Monday 2026-11-23
        market_bars = {
            "ITC": [
                {"date": "2026-11-20", "open": 288.0, "close": 290.0},
                {"date": "2026-11-23", "open": 292.0, "close": 295.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 1)
        tx = exec_res["executed_orders"][0]
        self.assertEqual(tx["order_id"], oid)
        self.assertEqual(tx["execution_date"], "2026-11-23")
        self.assertEqual(tx["execution_price"], 292.0)

    # 6. Holiday gaps are handled using actual available trading sessions
    def test_06_holiday_gap_handled_correctly(self):
        # Order placed before a mid-week exchange holiday (e.g. holiday on 11-24)
        res = submit_order("HDFCBANK", "BUY", 10, 800.0, decision_date="2026-11-23")
        oid = res["order_id"]

        # Next available market session is 2026-11-25 (skipping holiday 11-24)
        market_bars = {
            "HDFCBANK": [
                {"date": "2026-11-23", "open": 798.0, "close": 800.0},
                {"date": "2026-11-25", "open": 805.0, "close": 810.0},
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["executed_count"], 1)
        self.assertEqual(exec_res["executed_orders"][0]["execution_date"], "2026-11-25")

    # 7. Insufficient actual BUY funds cause rejection
    def test_07_insufficient_actual_buy_funds_causes_rejection(self):
        # Set cash to 10,000
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE portfolio_summary SET cash = 10000.0 WHERE id = 1")
            conn.commit()

        # Buy 5 shares @ estimated 1900 (~9509.50)
        res = submit_order("INFY", "BUY", 5, 1900.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        oid = res["order_id"]

        # Next open surges to 2200 (actual cost: 5 * 2200 * 1.001 = 11,011 > 10,000 cash)
        market_bars = {
            "INFY": [
                {"date": "2026-11-21", "open": 2200.0, "close": 2210.0}
            ]
        }
        exec_res = process_pending_orders_service(market_bars)
        self.assertEqual(exec_res["rejected_count"], 1)
        self.assertEqual(exec_res["executed_count"], 0)

        orders = get_orders(limit=2)
        rej_o = [x for x in orders if x["order_id"] == oid][0]
        self.assertEqual(rej_o["status"], "REJECTED")

    # 8. Other pending BUY reservations remain protected
    def test_08_other_pending_buy_reservations_protected(self):
        # Portfolio has ₹15,000 cash
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE portfolio_summary SET cash = 15000.0 WHERE id = 1")
            conn.commit()

        # Order 1 (TCS): BUY 5 @ 1000 -> reserves 5005.0. Available = 9995.0
        res1 = submit_order("TCS", "BUY", 5, 1000.0, decision_date="2026-11-20")
        # Order 2 (INFY): BUY 8 @ 1000 -> reserves 8008.0. Available = 1987.0
        res2 = submit_order("INFY", "BUY", 8, 1000.0, decision_date="2026-11-20")

        # Now execute Order 1, but its Open surges to ₹1,500!
        # Total cost for Order 1 at 1500 = 5 * 1500 * 1.001 = 7,507.50.
        # But Order 2 has reserved 8,008.00!
        # Cash available without consuming Order 2's reservation is: 15,000 - 8,008 = 6,992.00!
        # 7,507.50 > 6,992.00! Executing Order 1 would encroach on Order 2's reserved cash!
        # Order 1 MUST be rejected, protecting Order 2's reservation!
        res_exec1 = execute_order_at_open(res1["order_id"], 1500.0, "2026-11-21")
        self.assertFalse(res_exec1["success"])
        self.assertIn("affordability check failed", res_exec1["reason"].lower())

        # Verify Order 2 is still PENDING and cash is still 15,000
        p = get_portfolio()
        self.assertEqual(p["cash"], 15000.0)
        self.assertAlmostEqual(p["reserved_cash"], 8008.0, places=1)

    # 9. Insufficient SELL shares cause rejection
    def test_09_insufficient_sell_shares_cause_rejection(self):
        # Directly insert a SELL order for 10 shares when user has 0 shares
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO orders (symbol, side, quantity, decision_date, decision_time, agent_recommendation,
                                agent_probabilities, user_action, estimated_price, status, created_at)
            VALUES ('RELIANCE', 'SELL', 10, '2026-11-20', '12:00:00', 'SELL', '{}', 'SELL', 1300.0, 'PENDING', '2026-11-20T12:00:00');
            """)
            conn.commit()

        exec_res = execute_order_at_open(1, 1310.0, "2026-11-21")
        self.assertFalse(exec_res["success"])
        self.assertIn("Insufficient unreserved shares", exec_res["reason"])

    # 10. Other pending SELL reservations remain protected
    def test_10_other_pending_sell_reservations_protected(self):
        # User owns 10 shares of RELIANCE
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            UPDATE holdings
            SET shares = 10, avg_purchase_price = 1200.0, total_invested = 12012.0, current_price = 1200.0, current_value = 12000.0
            WHERE symbol = 'RELIANCE';
            """)
            conn.commit()

        # Order 1: SELL 6 shares (reserves 6, leaves 4 available)
        res1 = submit_order("RELIANCE", "SELL", 6, 1200.0, decision_date="2026-11-20")
        # Order 2: SELL 4 shares (reserves 4, leaves 0 available)
        res2 = submit_order("RELIANCE", "SELL", 4, 1200.0, decision_date="2026-11-20")

        # Now if someone attempts to execute a rogue order to sell 5 shares,
        # it must be rejected because 6 shares are reserved for Order 1!
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO orders (symbol, side, quantity, decision_date, decision_time, agent_recommendation,
                                agent_probabilities, user_action, estimated_price, status, created_at)
            VALUES ('RELIANCE', 'SELL', 5, '2026-11-20', '12:00:00', 'SELL', '{}', 'SELL', 1200.0, 'PENDING', '2026-11-20T12:00:00');
            """)
            rogue_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()

        rogue_res = execute_order_at_open(rogue_id, 1250.0, "2026-11-21")
        self.assertFalse(rogue_res["success"])
        self.assertIn("Insufficient unreserved shares", rogue_res["reason"])

    # 11. Rejected orders release escrow
    def test_11_rejected_orders_release_escrow(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE portfolio_summary SET cash = 5000.0 WHERE id = 1")
            conn.commit()

        res = submit_order("INFY", "BUY", 3, 1500.0, decision_date="2026-11-20")
        p_res = get_portfolio()
        self.assertGreater(p_res["reserved_cash"], 0.0)

        # Execute at surged open 2500 -> Rejected
        execute_order_at_open(res["order_id"], 2500.0, "2026-11-21")
        p_after = get_portfolio()
        self.assertEqual(p_after["reserved_cash"], 0.0)
        self.assertEqual(p_after["available_cash"], 5000.0)

    # 12. Cancelled orders release escrow
    def test_12_cancelled_orders_release_escrow(self):
        res = submit_order("ITC", "BUY", 10, 300.0, decision_date="2026-11-20")
        self.assertGreater(get_portfolio()["reserved_cash"], 0.0)

        cancel_order(res["order_id"])
        self.assertEqual(get_portfolio()["reserved_cash"], 0.0)
        self.assertEqual(get_portfolio()["available_cash"], 100000.0)

    # 13. Repeated processing does not duplicate transactions
    def test_13_repeated_processing_does_not_duplicate_transactions(self):
        res = submit_order("TCS", "BUY", 5, 2000.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        market_bars = {
            "TCS": [{"date": "2026-11-21", "open": 2010.0, "close": 2020.0}]
        }
        # First execution
        exec1 = process_pending_orders_service(market_bars)
        self.assertEqual(exec1["executed_count"], 1)

        # Second execution with same data
        exec2 = process_pending_orders_service(market_bars)
        self.assertEqual(exec2["executed_count"], 0)
        self.assertEqual(exec2["processed_count"], 0)

        txs = get_transactions()
        self.assertEqual(len(txs), 1)

    # 14. Concurrent processing cannot execute the same order twice
    def test_14_concurrent_processing_cannot_double_execute(self):
        res = submit_order("RELIANCE", "BUY", 5, 1300.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        oid = res["order_id"]

        results = []
        def worker():
            r = execute_order_at_open(oid, 1305.0, "2026-11-21")
            results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r.get("success")]
        self.assertEqual(len(successes), 1)

        txs = get_transactions()
        matching_txs = [t for t in txs if t["order_id"] == oid]
        self.assertEqual(len(matching_txs), 1)

    # 17. Portfolio accounting remains consistent after multiple executions
    def test_17_portfolio_accounting_consistency(self):
        # 1. Buy RELIANCE 10 @ 1000
        res1 = submit_order("RELIANCE", "BUY", 10, 1000.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")
        # 2. Buy TCS 5 @ 2000
        res2 = submit_order("TCS", "BUY", 5, 2000.0, decision_date="2026-11-20", created_at="2026-11-20T10:00:00")

        market_bars = {
            "RELIANCE": [{"date": "2026-11-21", "open": 1000.0, "close": 1050.0}],
            "TCS": [{"date": "2026-11-21", "open": 2000.0, "close": 2050.0}],
        }
        res_all = process_pending_orders_service(market_bars)
        self.assertEqual(res_all["executed_count"], 2)

        p = get_portfolio()
        # Initial: 100,000.
        # Cost 1: 10 * 1000 * 1.001 = 10,010.00
        # Cost 2: 5 * 2000 * 1.001 = 10,010.00
        # Total cost: 20,020.00
        # Cash remaining: 79,980.00
        self.assertAlmostEqual(p["cash"], 79980.0, places=2)
        self.assertAlmostEqual(p["reserved_cash"], 0.0, places=2)
        self.assertAlmostEqual(p["available_cash"], 79980.0, places=2)
        # Holdings val: 10 * 1050 + 5 * 2050 = 10,500 + 10,250 = 20,750.00
        self.assertAlmostEqual(p["holdings_value"], 20750.0, places=2)
        # Total val = 79,980 + 20,750 = 100,730.00
        self.assertAlmostEqual(p["total_value"], 100730.0, places=2)


class TestPhase37APIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.db_path = Path(cls.temp_dir.name) / "test_api_p37.db"
        cls.db_patcher = patch("backend.live_data.portfolio_db.DB_PATH", cls.db_path)
        cls.db_patcher.start()
        init_db()

        server_address = ("127.0.0.1", 8769)
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

    # 15. API submission rejects untrusted browser prices
    def test_15_api_rejects_untrusted_browser_prices(self):
        # Mock backend price returning 0 (unverified)
        with patch("backend.live_data.api_server.get_trusted_market_price_and_rec", return_value=(0.0, "HOLD", {}, "2026-11-20")):
            payload = json.dumps({
                "symbol": "RELIANCE",
                "side": "BUY",
                "quantity": 10,
                "price": 999.0  # Browser claims it's 999
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://127.0.0.1:8769/api/order",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 400)
            err = json.loads(ctx.exception.read().decode())
            self.assertFalse(err["success"])
            self.assertIn("client-supplied prices are not accepted", err["reason"].lower())

    # 16. API execution endpoint uses backend-verified market data
    def test_16_api_process_pending_endpoint(self):
        # Submit valid order
        with patch("backend.live_data.api_server.get_trusted_market_price_and_rec", return_value=(1300.0, "BUY", {}, "2026-11-20")):
            payload = json.dumps({
                "symbol": "RELIANCE",
                "side": "BUY",
                "quantity": 5
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8769/api/order",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)

        # Call POST /api/process-pending
        # Mock load_from_cache to provide a subsequent bar strictly after today
        today_str = datetime.now().strftime("%Y-%m-%d")
        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        fake_cache = {
            "historical": {
                "RELIANCE": [
                    {"date": today_str, "open": 1290.0, "close": 1300.0},
                    {"date": tomorrow_str, "open": 1305.0, "close": 1310.0},
                ]
            }
        }
        with patch("backend.live_data.api_server.load_from_cache", return_value=fake_cache):
            proc_req = urllib.request.Request(
                "http://127.0.0.1:8769/api/process-pending",
                data=b"{}",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(proc_req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode())
                self.assertTrue(data["success"])
                self.assertEqual(data["executed_count"], 1)


if __name__ == "__main__":
    unittest.main()
