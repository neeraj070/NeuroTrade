"""
Daily Operational Orchestrator for NeuroTrade Algorithmic Trading System.

Automates the complete end-to-end trading lifecycle:
1. Process concurrency locking (cross-process mutex preventing overlapping runs).
2. NSE market-calendar validation (weekend & official holiday exclusion).
3. Missed-run detection & recovery audit.
4. Idempotency enforcement (prevents duplicate runs for the same trading session).
5. Market data ingestion from NSE MCP with retry backoff & 180s cache awareness.
6. Execution of eligible pending orders at verified next-session Open prices.
7. Mark-to-market portfolio valuation.
8. Deep learning (LSTM 32-D) & RL (PPO 49-D) inference with dynamic portfolio context.
9. Persistent multi-session performance tracking (Phase 6 evaluation integration).
10. Live dashboard updates & structured audit logging.
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.scheduler_config import (
    LOCK_FILE_PATH, STATUS_FILE_PATH, LOGS_DIR, LIVE_DATA_DIR,
    is_nse_trading_day, get_calendar_status, get_missed_trading_days,
    MAX_RETRIES, RETRY_BACKOFF_BASE
)
from backend.live_data.data_fetcher import fetch_all_market_data, TARGET_STOCKS
from backend.live_data.inference_engine import LiveInferenceEngine
from backend.live_data.dashboard import print_console_dashboard, export_html_dashboard
from backend.live_data.portfolio_db import (
    init_db, get_portfolio, update_holdings_prices,
    process_pending_orders_service, submit_order
)
from backend.live_data.paper_trader import get_dynamic_portfolio_context

logger = logging.getLogger("DailyOrchestrator")


def _is_pid_alive(pid: int) -> bool:
    """Checks whether the process with given PID is still active on the OS."""
    if pid <= 0:
        return False
    try:
        import ctypes
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        process = ctypes.windll.kernel32.OpenProcess(0x1000, 0, pid)
        if process:
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        return False
    except Exception:
        # Fall back to checking via os.kill on POSIX or assuming alive
        try:
            os.kill(pid, 0)
            return True
        except (OSError, AttributeError):
            return False


class OrchestratorLock:
    """
    Cross-process file lock ensuring single-instance execution on Windows and Linux.
    Writes PID and timestamp to lockfile. Cleans up stale or dead PID locks.
    """
    def __init__(self, lock_path: Path = LOCK_FILE_PATH):
        self.lock_path = lock_path
        self.acquired = False
        self._file_handle = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Check if existing lockfile is stale (older than 30 minutes or dead PID)
            if self.lock_path.exists():
                try:
                    with open(self.lock_path, "r") as f:
                        lock_data = json.load(f)
                    lock_pid = lock_data.get("pid", 0)
                    lock_time = datetime.fromisoformat(lock_data.get("acquired_at", "1970-01-01"))
                    age_seconds = (datetime.now() - lock_time).total_seconds()

                    if not _is_pid_alive(lock_pid):
                        logger.warning(f"Removing dead process lock from inactive PID {lock_pid} (acquired at {lock_time})")
                        self.lock_path.unlink(missing_ok=True)
                    elif age_seconds > 1800:
                        logger.warning(f"Removing timed-out lock from PID {lock_pid} (age {age_seconds:.0f}s > 1800s)")
                        self.lock_path.unlink(missing_ok=True)
                    else:
                        logger.error(f"Process lock already held by active PID {lock_pid} since {lock_time}")
                        return False
                except Exception:
                    self.lock_path.unlink(missing_ok=True)

            # Atomic creation on filesystem
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            with os.fdopen(fd, "w") as f:
                json.dump({
                    "pid": os.getpid(),
                    "acquired_at": datetime.now().isoformat()
                }, f)
            self.acquired = True
            return True
        except (FileExistsError, OSError) as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release(self):
        if self.acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink(missing_ok=True)
                self.acquired = False
            except Exception as e:
                logger.warning(f"Error removing lockfile {self.lock_path}: {e}")

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire orchestrator process lock. Another instance is active.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def setup_orchestrator_logging(log_date_str: str) -> logging.Logger:
    """Configures structured file and console logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"daily_orchestrator_{log_date_str}.log"

    orchestrator_logger = logging.getLogger("DailyOrchestrator")
    orchestrator_logger.setLevel(logging.INFO)
    orchestrator_logger.handlers = []

    # File handler
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_fmt)
    orchestrator_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("  %(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    orchestrator_logger.addHandler(ch)

    return orchestrator_logger


def record_run_status(status_payload: Dict[str, Any], path: Path = STATUS_FILE_PATH) -> None:
    """Persists the latest run status to JSON for scheduler telemetry and recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)


def read_last_run_status(path: Path = STATUS_FILE_PATH) -> Optional[Dict[str, Any]]:
    """Loads the last recorded run status if available."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


async def execute_daily_orchestration(
    target_date: Optional[date] = None,
    force: bool = False,
    force_refresh: bool = False,
    dry_run: bool = False,
    auto_approve_agent: bool = False
) -> Dict[str, Any]:
    """
    Main daily execution flow executed by Windows Task Scheduler or CLI.
    """
    now = datetime.now()
    cur_date = target_date or now.date()
    cur_date_str = cur_date.strftime("%Y-%m-%d")

    orchestrator_logger = setup_orchestrator_logging(cur_date_str)
    orchestrator_logger.info("=" * 80)
    orchestrator_logger.info(f"STARTING NEUROTRADE DAILY ORCHESTRATOR FOR DATE: {cur_date_str}")
    orchestrator_logger.info(f"Execution Mode: Force={force}, Refresh={force_refresh}, DryRun={dry_run}, AutoApprove={auto_approve_agent}")
    orchestrator_logger.info("=" * 80)

    # Step 1: Calendar validation
    cal_status = get_calendar_status(cur_date)
    orchestrator_logger.info(f"NSE Calendar Status: {cal_status['reason']} (Trading Day: {cal_status['is_trading_day']})")

    if not cal_status["is_trading_day"] and not force:
        status_code = "SKIPPED_WEEKEND" if cal_status["is_weekend"] else "SKIPPED_HOLIDAY"
        orchestrator_logger.info(f"Skipping scheduled run: {cal_status['reason']}. Use --force to override.")
        result = {
            "execution_date": cur_date_str,
            "timestamp": now.isoformat(),
            "status": status_code,
            "message": f"Execution skipped: {cal_status['reason']}",
            "calendar": cal_status,
        }
        record_run_status(result)
        return result

    # Step 2: Idempotency check
    last_status = read_last_run_status()
    if last_status and not force:
        if (
            last_status.get("execution_date") == cur_date_str
            and last_status.get("status") == "SUCCESS"
        ):
            orchestrator_logger.info(f"Trading session {cur_date_str} was already successfully processed at {last_status.get('timestamp')}.")
            orchestrator_logger.info("Skipping duplicate execution. Use --force to re-run.")
            result = {
                "execution_date": cur_date_str,
                "timestamp": now.isoformat(),
                "status": "SKIPPED_ALREADY_PROCESSED",
                "message": f"Trading session {cur_date_str} already completed. Duplicate run prevented.",
                "previous_run": last_status,
            }
            record_run_status(result)
            return result

    # Step 3: Missed-run detection
    if last_status and last_status.get("execution_date"):
        last_date = last_status["execution_date"]
        missed = get_missed_trading_days(last_date, cur_date)
        if missed:
            missed_strs = [d.strftime("%Y-%m-%d") for d in missed]
            orchestrator_logger.warning(f"DETECTED {len(missed)} MISSED TRADING SESSIONS: {', '.join(missed_strs)}")
            orchestrator_logger.warning("Pipeline will process all completed historical bars to restore synchronization.")

    # Initialize DB
    init_db()

    # Step 4: Retrieve Market Data with retry logic
    orchestrator_logger.info("\n[Stage 1/5] Ingesting Market Data from official NSE MCP servers...")
    market_data = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            orchestrator_logger.info(f"Fetching market data (Attempt {attempt}/{MAX_RETRIES})...")
            market_data = await fetch_all_market_data(force_refresh=force_refresh)
            if market_data and market_data.get("data_status") in ["FRESH", "CACHED"]:
                orchestrator_logger.info(f"Market data retrieved successfully. Status: [{market_data['data_status']}]")
                break
        except Exception as e:
            last_error = e
            orchestrator_logger.warning(f"Data retrieval attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                sleep_sec = RETRY_BACKOFF_BASE ** attempt
                orchestrator_logger.info(f"Backing off for {sleep_sec:.1f}s before retry...")
                await asyncio.sleep(sleep_sec)

    if not market_data:
        err_msg = f"Failed to ingest market data after {MAX_RETRIES} attempts. Error: {last_error}"
        orchestrator_logger.error(err_msg)
        result = {
            "execution_date": cur_date_str,
            "timestamp": now.isoformat(),
            "status": "FAILED_DATA_INGESTION",
            "error": str(last_error),
            "message": err_msg
        }
        record_run_status(result)
        return result

    data_status = market_data.get("data_status", "UNKNOWN")
    historical = market_data.get("historical", {})
    quotes = market_data.get("quotes", {})
    orchestrator_logger.info(f"Data Status: [{data_status}] | Cache Age: {market_data.get('cache_age_seconds', 0.0)}s")

    if data_status == "FALLBACK_STALE":
        orchestrator_logger.warning("SAFEGUARD ALERT: Market data is STALE. Next-session order execution and auto-trades require fresh verified bars.")

    # Step 5: Process Pending Orders (Next-Session Settlement)
    orchestrator_logger.info("\n[Stage 2/5] Processing pending paper orders against verified session Open bars...")
    execution_result = {"processed_count": 0, "executed_count": 0, "rejected_count": 0, "deferred_count": 0}
    if not dry_run:
        try:
            execution_result = process_pending_orders_service(historical)
            orchestrator_logger.info(
                f"Settlement Summary: {execution_result['executed_count']} Executed, "
                f"{execution_result['rejected_count']} Rejected, "
                f"{execution_result['deferred_count']} Deferred (awaiting subsequent session Open)."
            )
            for ex in execution_result.get("executed_orders", []):
                orchestrator_logger.info(f"  -> EXECUTED Order #{ex['order_id']}: {ex['side']} {ex['quantity']} {ex['symbol']} @ ₹{ex['execution_price']:,.2f}")
            for rj in execution_result.get("rejected_orders", []):
                orchestrator_logger.warning(f"  -> REJECTED Order #{rj.get('order_id')}: {rj.get('reason')}")
        except Exception as e:
            orchestrator_logger.error(f"Error during order settlement: {e}")
    else:
        orchestrator_logger.info("Dry-run active: Skipped order settlement.")

    # Step 6: Mark Portfolio to Market
    orchestrator_logger.info("\n[Stage 3/5] Marking portfolio holdings to market...")
    current_prices = {}
    for s in TARGET_STOCKS:
        p = 0.0
        if s in quotes and quotes[s].get("lastTradedPrice"):
            p = float(quotes[s]["lastTradedPrice"])
        elif s in historical and historical[s]:
            p = float(historical[s][-1].get("close") or historical[s][-1].get("Close") or 0.0)
        current_prices[s] = p

    if not dry_run:
        update_holdings_prices(current_prices)
    portfolio = get_portfolio()
    orchestrator_logger.info(
        f"Portfolio Valuation: Total ₹{portfolio.get('total_value', 100000.0):,.2f} "
        f"(Cash: ₹{portfolio.get('cash', 100000.0):,.2f}, Reserved: ₹{portfolio.get('reserved_cash', 0.0):,.2f}, "
        f"Holdings: ₹{portfolio.get('holdings_value', 0.0):,.2f}, Return: {portfolio.get('total_return_pct', 0.0):+.2f}%)"
    )

    # Step 7: Deep Learning (LSTM 32-D) & PPO (49-D) Inference
    orchestrator_logger.info("\n[Stage 4/5] Executing LSTM & PPO Model Inference...")
    predictions = []
    try:
        engine = LiveInferenceEngine()
        for symbol in TARGET_STOCKS:
            bars = historical.get(symbol, [])
            quote = quotes.get(symbol, {})
            price = current_prices.get(symbol, 0.0)
            if not bars:
                continue

            dyn_context = get_dynamic_portfolio_context(symbol, price, portfolio)
            pred = engine.predict_stock(symbol, bars, quote, portfolio_context=dyn_context)
            predictions.append(pred)

            rec = pred["recommendation"]
            p = pred["action_probabilities"]
            orchestrator_logger.info(
                f"  -> {symbol:<10} | Price: ₹{price:>9,.2f} | PPO Rec: {rec:<4} "
                f"| Probabilities: HOLD {p['HOLD']*100:>4.1f}%, BUY {p['BUY']*100:>4.1f}%, SELL {p['SELL']*100:>4.1f}%"
            )

            # Auto-approve agent recommendation if flag is set (for automated multi-session evaluation)
            if auto_approve_agent and not dry_run and data_status == "FRESH":
                if rec in ["BUY", "SELL"]:
                    target_qty = 10  # Standard evaluation lot size
                    res = submit_order(
                        symbol=symbol,
                        side=rec,
                        quantity=target_qty,
                        current_price=price,
                        agent_recommendation=rec,
                        agent_probabilities=p,
                        decision_date=cur_date_str
                    )
                    orchestrator_logger.info(f"  -> [Auto-Agent] Queued {rec} {target_qty} {symbol}: {res.get('message', res.get('reason'))}")
    except Exception as e:
        orchestrator_logger.error(f"Inference error: {e}")

    # Step 8: Multi-Session Evaluation Snapshot (Phase 6 Integration)
    orchestrator_logger.info("\n[Stage 5/5] Updating Evaluation Records & Dashboards...")
    try:
        from backend.live_data.evaluation_engine import EvaluationEngine
        eval_engine = EvaluationEngine()
        eval_engine.record_session(
            trading_date=cur_date_str,
            data_status=data_status,
            portfolio=portfolio,
            predictions=predictions,
            execution_summary=execution_result
        )
        orchestrator_logger.info("Evaluation session snapshot successfully recorded in evaluation ledger.")
    except Exception as e:
        orchestrator_logger.warning(f"Could not update evaluation session record: {e}")

    # Step 9: Render & Export Dashboards
    pipeline_output = {
        "stocks": predictions,
        "cache_info": {
            "data_status": data_status,
            "cache_age_seconds": market_data.get("cache_age_seconds", 0.0),
            "retrieval_timestamp": market_data.get("retrieval_timestamp"),
            "market_timestamp": market_data.get("market_timestamp"),
        }
    }
    if not dry_run:
        html_file = export_html_dashboard(pipeline_output)
        orchestrator_logger.info(f"Dashboard updated at: {html_file}")

    run_summary = {
        "execution_date": cur_date_str,
        "timestamp": now.isoformat(),
        "status": "SUCCESS",
        "data_status": data_status,
        "stocks_evaluated": len(predictions),
        "executed_orders": execution_result.get("executed_count", 0),
        "rejected_orders": execution_result.get("rejected_count", 0),
        "deferred_orders": execution_result.get("deferred_count", 0),
        "portfolio_value": portfolio.get("total_value", 100000.0),
        "total_return_pct": portfolio.get("total_return_pct", 0.0),
    }
    record_run_status(run_summary)
    orchestrator_logger.info("=" * 80)
    orchestrator_logger.info(f"DAILY ORCHESTRATION COMPLETED SUCCESSFULLY FOR {cur_date_str}")
    orchestrator_logger.info("=" * 80)
    return run_summary


def main():
    parser = argparse.ArgumentParser(description="NeuroTrade Automated Daily Operational Orchestrator")
    parser.add_argument("--force", action="store_true", help="Bypass weekend/holiday and duplicate-run checks.")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass 180s cache and fetch fresh data from NSE MCP.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without mutating portfolio or orders.")
    parser.add_argument("--auto-approve-agent", action="store_true", help="Automatically submit agent recommendations as pending orders.")
    parser.add_argument("--date", type=str, default=None, help="Override target date (YYYY-MM-DD) for backtesting or catchup.")
    args = parser.parse_args()

    target_dt = None
    if args.date:
        try:
            target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {args.date}. Expected YYYY-MM-DD.")
            sys.exit(1)

    lock = OrchestratorLock()
    if not lock.acquire():
        print("[ERROR] Another NeuroTrade orchestrator process is currently running. Exiting.")
        sys.exit(2)

    try:
        res = asyncio.run(execute_daily_orchestration(
            target_date=target_dt,
            force=args.force,
            force_refresh=args.force_refresh,
            dry_run=args.dry_run,
            auto_approve_agent=args.auto_approve_agent
        ))
        print(f"\n[Orchestrator Result] Status: {res.get('status')} | Message: {res.get('message', 'Completed')}")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
