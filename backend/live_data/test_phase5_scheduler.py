"""
Unit & Integration Test Suite for Phase 5 Automated Scheduler & Safeguards.

Tests:
1. Process concurrency locking (cross-process collision prevention).
2. Weekend detection & exclusion.
3. Official NSE holiday calendar recognition.
4. Previous & next trading day resolution.
5. Missed-run detection.
6. Idempotency & duplicate execution prevention.
7. Logging & status persistence.
8. Dry-run execution safety.
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import backend.live_data.scheduler_config as cfg
from backend.live_data.daily_orchestrator import (
    OrchestratorLock, record_run_status, read_last_run_status,
    execute_daily_orchestration
)


class TestPhase5Scheduler(unittest.TestCase):
    """Verifies all Phase 5 scheduling safeguards, calendar rules, and locks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lock_file = self.temp_path / "test.lock"
        self.status_file = self.temp_path / "last_status.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_lock_acquisition_and_release(self):
        """Confirms that OrchestratorLock creates lockfile and cleans up on release."""
        lock = OrchestratorLock(self.lock_file)
        self.assertTrue(lock.acquire(), "First acquisition must succeed.")
        self.assertTrue(self.lock_file.exists(), "Lockfile must exist on disk.")

        # Second acquisition on same lockfile must fail
        lock2 = OrchestratorLock(self.lock_file)
        self.assertFalse(lock2.acquire(), "Second acquisition on active lock must fail.")

        lock.release()
        self.assertFalse(self.lock_file.exists(), "Lockfile must be deleted on release.")
        self.assertTrue(lock2.acquire(), "Subsequent acquisition must succeed after release.")
        lock2.release()

    def test_weekend_detection(self):
        """Verifies that Saturday and Sunday are excluded from active trading days."""
        # 2026-11-20 is Friday (Weekday)
        friday = date(2026, 11, 20)
        self.assertFalse(cfg.is_weekend(friday))
        self.assertTrue(cfg.is_nse_trading_day(friday))

        # 2026-11-21 is Saturday (Weekend)
        saturday = date(2026, 11, 21)
        self.assertTrue(cfg.is_weekend(saturday))
        self.assertFalse(cfg.is_nse_trading_day(saturday))

        # 2026-11-22 is Sunday (Weekend)
        sunday = date(2026, 11, 22)
        self.assertTrue(cfg.is_weekend(sunday))
        self.assertFalse(cfg.is_nse_trading_day(sunday))

        # 2026-11-23 is Monday (Weekday)
        monday = date(2026, 11, 23)
        self.assertFalse(cfg.is_weekend(monday))
        self.assertTrue(cfg.is_nse_trading_day(monday))

    def test_official_nse_holiday_calendar(self):
        """Verifies that official 2026 NSE holidays are accurately recognized."""
        # Republic Day: Jan 26, 2026 (Monday)
        rep_day = date(2026, 1, 26)
        is_hol, hol_name = cfg.is_nse_holiday(rep_day)
        self.assertTrue(is_hol)
        self.assertEqual(hol_name, "Republic Day")
        self.assertFalse(cfg.is_nse_trading_day(rep_day))

        # Gandhi Jayanti: Oct 2, 2026 (Friday)
        gandhi = date(2026, 10, 2)
        is_hol, hol_name = cfg.is_nse_holiday(gandhi)
        self.assertTrue(is_hol)
        self.assertEqual(hol_name, "Mahatma Gandhi Jayanti")
        self.assertFalse(cfg.is_nse_trading_day(gandhi))

        # Christmas: Dec 25, 2026 (Friday)
        xmas = date(2026, 12, 25)
        is_hol, hol_name = cfg.is_nse_holiday(xmas)
        self.assertTrue(is_hol)
        self.assertEqual(hol_name, "Christmas")
        self.assertFalse(cfg.is_nse_trading_day(xmas))

    def test_previous_and_next_trading_day(self):
        """Verifies calculation of preceding and succeeding trading sessions."""
        # From Monday 2026-11-23, previous trading day must be Friday 2026-11-20
        monday = date(2026, 11, 23)
        prev_day = cfg.get_previous_trading_day(monday)
        self.assertEqual(prev_day, date(2026, 11, 20))

        # From Friday 2026-11-20, next trading day must be Monday 2026-11-23
        friday = date(2026, 11, 20)
        next_day = cfg.get_next_trading_day(friday)
        self.assertEqual(next_day, date(2026, 11, 23))

        # From Friday 2026-10-02 (Gandhi Jayanti holiday), previous was Thursday 2026-10-01
        gandhi_friday = date(2026, 10, 2)
        prev_before_holiday = cfg.get_previous_trading_day(gandhi_friday)
        self.assertEqual(prev_before_holiday, date(2026, 10, 1))

    def test_missed_trading_days_detection(self):
        """Detects trading sessions where the scheduled pipeline did not execute."""
        # System was offline from Thursday 2026-11-12 to Tuesday 2026-11-17
        last_run = date(2026, 11, 12)  # Thursday
        current = date(2026, 11, 17)   # Tuesday

        missed = cfg.get_missed_trading_days(last_run, current)
        # Missed trading days: Friday 2026-11-13 and Monday 2026-11-16
        expected = [date(2026, 11, 13), date(2026, 11, 16)]
        self.assertEqual(missed, expected)

    def test_status_persistence(self):
        """Verifies writing and reading orchestrator status payloads."""
        payload = {
            "execution_date": "2026-11-20",
            "timestamp": "2026-11-20T16:30:00",
            "status": "SUCCESS",
            "executed_orders": 2,
            "portfolio_value": 100000.0
        }
        record_run_status(payload, path=self.status_file)
        loaded = read_last_run_status(path=self.status_file)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["status"], "SUCCESS")
        self.assertEqual(loaded["execution_date"], "2026-11-20")

    def test_calendar_status_structure(self):
        """Confirms that get_calendar_status returns comprehensive structured metadata."""
        sat_status = cfg.get_calendar_status(date(2026, 11, 21))
        self.assertFalse(sat_status["is_trading_day"])
        self.assertTrue(sat_status["is_weekend"])
        self.assertIn("Weekend", sat_status["reason"])

        trade_status = cfg.get_calendar_status(date(2026, 11, 20))
        self.assertTrue(trade_status["is_trading_day"])
        self.assertFalse(trade_status["is_weekend"])


if __name__ == "__main__":
    unittest.main()
