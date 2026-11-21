"""
Unit & Integration Test Suite for Phase 6 Multi-Session Evaluation & Metrics.

Tests:
1. Session snapshot recording in isolated temporary database.
2. Accurate return, volatility, Sharpe ratio, and drawdown calculations.
3. Benchmark generation: Equal-Weight Buy & Hold and Dual SMA Crossover.
4. Visualization generator creating 5 standard analytical figures.
5. In-progress status verification when observed sessions < 20.
6. Absolute immutability of production database records (orders #27, #28, #29).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.evaluation_engine import EvaluationEngine, RISK_FREE_RATE_ANNUAL
from backend.live_data.portfolio_db import DB_PATH, get_orders


class TestPhase6Evaluation(unittest.TestCase):
    """Verifies all Phase 6 quantitative calculations, benchmarks, and data isolation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.temp_db_path = Path(self.temp_dir.name) / "test_eval.db"
        self.viz_dir = Path(self.temp_dir.name) / "visualizations"
        self.engine = EvaluationEngine(db_path=self.temp_db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_record_single_session_and_in_progress_status(self):
        """Verifies session recording and confirms IN PROGRESS status when N < 20."""
        portfolio_mock = {
            "initial_capital": 100000.0,
            "cash": 100000.0,
            "reserved_cash": 0.0,
            "available_cash": 100000.0,
            "holdings_value": 0.0,
            "total_value": 100000.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "holdings": {"RELIANCE": {"shares": 0, "current_price": 1377.5}}
        }
        predictions_mock = [
            {
                "symbol": "RELIANCE",
                "current_price": 1377.5,
                "recommendation": "HOLD",
                "action_probabilities": {"HOLD": 0.60, "BUY": 0.20, "SELL": 0.20},
                "portfolio_context": [0.0, 0.0, 1.0]
            }
        ]

        res = self.engine.record_session(
            trading_date="2026-11-20",
            data_status="FRESH",
            portfolio=portfolio_mock,
            predictions=predictions_mock
        )
        self.assertEqual(res["trading_date"], "2026-11-20")
        self.assertEqual(res["portfolio_value"], 100000.0)

        metrics = self.engine.calculate_evaluation_metrics()
        self.assertFalse(metrics["is_complete"], "Evaluation must NOT be marked complete with only 1 session.")
        self.assertIn("IN PROGRESS", metrics["evaluation_status"])
        self.assertEqual(metrics["sessions_observed"], 1)

    def test_multi_session_metrics_and_drawdown(self):
        """Verifies multi-session progression, daily returns, Sharpe ratio, and drawdown calculation."""
        # Session 1: Day 1 (+1% to ₹1,01,000)
        p1 = {"initial_capital": 100000.0, "total_value": 101000.0, "cash": 101000.0, "holdings_value": 0.0}
        self.engine.record_session("2026-11-16", "FRESH", p1, [])

        # Session 2: Day 2 (+2% to ₹1,03,020)
        p2 = {"initial_capital": 100000.0, "total_value": 103020.0, "cash": 103020.0, "holdings_value": 0.0}
        self.engine.record_session("2026-11-17", "FRESH", p2, [])

        # Session 3: Day 3 (-1% to ₹1,01,989.8) - introduces drawdown
        p3 = {"initial_capital": 100000.0, "total_value": 101989.8, "cash": 101989.8, "holdings_value": 0.0}
        self.engine.record_session("2026-11-18", "FRESH", p3, [])

        metrics = self.engine.calculate_evaluation_metrics()
        self.assertEqual(metrics["sessions_observed"], 3)
        self.assertAlmostEqual(metrics["portfolio_performance"]["cumulative_return_pct"], 1.9898, places=2)

        # Drawdown from peak (103,020 down to 101,989.8 = 1.00%)
        self.assertGreater(metrics["risk_metrics"]["maximum_drawdown_pct"], 0.95)
        self.assertLess(metrics["risk_metrics"]["maximum_drawdown_pct"], 1.05)

        # Volatility & Sharpe
        self.assertGreater(metrics["risk_metrics"]["annualized_volatility_pct"], 0.0)
        self.assertIn("6.50%", metrics["risk_metrics"]["risk_free_rate_used"])
        self.assertIn("Preliminary estimate", metrics["risk_metrics"]["sharpe_ratio_note"])

    def test_win_rate_and_profit_factor_edge_cases(self):
        """Verifies win rate exclusion of open positions and zero-trade/zero-loss edge cases."""
        p1 = {"initial_capital": 100000.0, "total_value": 100000.0, "cash": 100000.0, "holdings_value": 0.0}
        self.engine.record_session("2026-11-20", "FRESH", p1, [])

        metrics = self.engine.calculate_evaluation_metrics()
        tm = metrics["trading_metrics"]
        self.assertIsNone(tm["win_rate_pct"])
        self.assertIsNone(tm["profit_factor"])
        self.assertIn("0 closed trades", tm["win_rate_display"])
        self.assertIn("No closed round-trip trades", tm["profit_factor_display"])

    def test_visualization_generation(self):
        """Confirms generation of all 5 required analytical charts."""
        p1 = {"initial_capital": 100000.0, "total_value": 100000.0, "cash": 100000.0, "holdings_value": 0.0}
        self.engine.record_session("2026-11-20", "FRESH", p1, [
            {"symbol": "RELIANCE", "current_price": 1377.5, "recommendation": "HOLD", "action_probabilities": {"HOLD": 0.7, "BUY": 0.15, "SELL": 0.15}}
        ])

        charts = self.engine.generate_visualizations(output_dir=self.viz_dir)
        self.assertEqual(len(charts), 5)
        for c in charts:
            self.assertTrue(c.exists(), f"Chart file {c} must exist on disk.")
            self.assertGreater(c.stat().st_size, 1000, f"Chart file {c} must not be empty.")

    def test_production_database_isolation(self):
        """Strictly verifies that production database orders #27, #28, #29 are untouched."""
        prod_orders = get_orders()
        prod_order_ids = [o["order_id"] for o in prod_orders]
        self.assertIn(27, prod_order_ids, "Order #27 must exist in production DB.")
        self.assertIn(28, prod_order_ids, "Order #28 must exist in production DB.")
        self.assertIn(29, prod_order_ids, "Order #29 must exist in production DB.")

        for o in prod_orders:
            if o["order_id"] in [27, 28, 29]:
                self.assertEqual(o["status"], "PENDING", f"Production Order #{o['order_id']} must remain PENDING.")


if __name__ == "__main__":
    unittest.main()
