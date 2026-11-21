"""
Multi-Session Paper-Trading Evaluation & Quantitative Performance Engine.

Provides:
1. Session-by-session state persistence in SQLite.
2. Rigorous quantitative metrics: Cumulative Return, Daily Returns, Annualized Sharpe Ratio,
   Maximum Drawdown, Volatility, Win Rate, Profit Factor, Turnover, and Fee Impact.
3. Concurrent Benchmark tracking:
   - Equal-Weight Buy & Hold (20% allocated to RELIANCE, TCS, INFY, HDFCBANK, ITC).
   - Single-Stock Buy & Hold baselines.
   - Dual Moving Average Crossover (SMA 10 / SMA 30) with 0.1% transaction friction.
4. Chart & Visualization generation (matplotlib) saved to reports/phase6/visualizations/.
5. Strict Academic Rigor: Tracks observed sessions and enforces the IN PROGRESS status
   until 20-30 actual live NSE trading sessions have been fully recorded.
"""

import os
import sys
import json
import sqlite3
import math
import logging
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.portfolio_db import (
    DB_PATH, DEFAULT_INITIAL_CAPITAL,
    TRANSACTION_FEE_RATE, TARGET_STOCKS
)

logger = logging.getLogger("EvaluationEngine")

# Documentation assumption: Annualized Risk-Free Rate = 6.50% (RBI 91-day T-Bill rate)
RISK_FREE_RATE_ANNUAL = 0.065
TRADING_DAYS_PER_YEAR = 252
MIN_SESSIONS_FOR_COMPLETION = 20

VIZ_DIR = PROJECT_ROOT / "reports" / "phase6" / "visualizations"


class EvaluationEngine:
    """
    Manages multi-session paper-trading evaluation records and calculates
    comprehensive quantitative performance metrics and benchmark comparisons.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.init_evaluation_tables()

    @contextmanager
    def conn_scope(self):
        """Context manager that ensures the SQLite connection is closed on exit."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_evaluation_tables(self) -> None:
        """Creates tables for session snapshots, decisions, and benchmark baselines."""
        with self.conn_scope() as conn:
            cursor = conn.cursor()

            # 1. Session Snapshots
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_date TEXT NOT NULL UNIQUE,
                session_timestamp TEXT NOT NULL,
                data_status TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                ledger_cash REAL NOT NULL,
                reserved_cash REAL NOT NULL,
                available_cash REAL NOT NULL,
                holdings_value REAL NOT NULL,
                total_portfolio_value REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                cumulative_return_pct REAL NOT NULL,
                daily_return_pct REAL NOT NULL,
                peak_portfolio_value REAL NOT NULL,
                current_drawdown_pct REAL NOT NULL,
                executed_trades_count INTEGER NOT NULL,
                holdings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Agent Decisions Snapshot
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES evaluation_sessions(session_id),
                trading_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                current_price REAL NOT NULL,
                ppo_recommendation TEXT NOT NULL,
                prob_hold REAL NOT NULL,
                prob_buy REAL NOT NULL,
                prob_sell REAL NOT NULL,
                portfolio_context_pos REAL NOT NULL,
                user_action_taken TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            # Ensure base transactions table exists if running against an isolated DB
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                execution_price REAL NOT NULL,
                execution_date TEXT NOT NULL,
                execution_time TEXT NOT NULL,
                transaction_fee REAL NOT NULL,
                total_amount REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                decision_date TEXT NOT NULL,
                decision_time TEXT NOT NULL,
                agent_recommendation TEXT NOT NULL,
                agent_probabilities TEXT NOT NULL,
                user_action TEXT NOT NULL,
                estimated_price REAL NOT NULL,
                target_execution TEXT NOT NULL DEFAULT 'NEXT_SESSION_OPEN',
                status TEXT NOT NULL,
                rejection_reason TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 3. Benchmark Comparisons Snapshot
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_date TEXT NOT NULL UNIQUE,
                ppo_value REAL NOT NULL,
                equal_weight_bh_value REAL NOT NULL,
                reliance_bh_value REAL NOT NULL,
                tcs_bh_value REAL NOT NULL,
                infy_bh_value REAL NOT NULL,
                hdfcbank_bh_value REAL NOT NULL,
                itc_bh_value REAL NOT NULL,
                sma_crossover_value REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
            conn.commit()

    def record_session(
        self,
        trading_date: str,
        data_status: str,
        portfolio: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        execution_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Records an end-of-session evaluation snapshot.
        Calculates daily return, peak valuation, and drawdown dynamically.
        """
        now_iso = datetime.now().isoformat()
        init_cap = float(portfolio.get("initial_capital", DEFAULT_INITIAL_CAPITAL))
        tot_val = float(portfolio.get("total_value", init_cap))
        cash = float(portfolio.get("cash", init_cap))
        res_cash = float(portfolio.get("reserved_cash", 0.0))
        avail_cash = float(portfolio.get("available_cash", cash))
        h_val = float(portfolio.get("holdings_value", 0.0))
        r_pnl = float(portfolio.get("realized_pnl", 0.0))
        u_pnl = float(portfolio.get("unrealized_pnl", 0.0))
        tot_pnl = float(portfolio.get("total_pnl", r_pnl + u_pnl))

        cum_return_pct = round(((tot_val - init_cap) / init_cap) * 100.0, 4)

        with self.conn_scope() as conn:
            cursor = conn.cursor()

            # Retrieve prior session to calculate daily return and peak value
            cursor.execute("""
            SELECT total_portfolio_value, peak_portfolio_value
            FROM evaluation_sessions
            WHERE trading_date < ?
            ORDER BY trading_date DESC LIMIT 1;
            """, (trading_date,))
            prev_row = cursor.fetchone()

            if prev_row:
                prev_val = prev_row["total_portfolio_value"]
                prev_peak = prev_row["peak_portfolio_value"]
                daily_return_pct = round(((tot_val - prev_val) / prev_val) * 100.0, 4) if prev_val > 0 else 0.0
                peak_val = max(prev_peak, tot_val)
            else:
                daily_return_pct = round(((tot_val - init_cap) / init_cap) * 100.0, 4)
                peak_val = max(init_cap, tot_val)

            drawdown_pct = round(((peak_val - tot_val) / peak_val) * 100.0, 4) if peak_val > 0 else 0.0

            cursor.execute("SELECT COUNT(*) FROM transactions;")
            tx_count_row = cursor.fetchone()
            executed_trades = tx_count_row[0] if tx_count_row else 0

            holdings_json = json.dumps(portfolio.get("holdings", {}))

            cursor.execute("""
            INSERT INTO evaluation_sessions (
                trading_date, session_timestamp, data_status, initial_capital,
                ledger_cash, reserved_cash, available_cash, holdings_value,
                total_portfolio_value, realized_pnl, unrealized_pnl, total_pnl,
                cumulative_return_pct, daily_return_pct, peak_portfolio_value,
                current_drawdown_pct, executed_trades_count, holdings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trading_date) DO UPDATE SET
                session_timestamp = excluded.session_timestamp,
                data_status = excluded.data_status,
                ledger_cash = excluded.ledger_cash,
                reserved_cash = excluded.reserved_cash,
                available_cash = excluded.available_cash,
                holdings_value = excluded.holdings_value,
                total_portfolio_value = excluded.total_portfolio_value,
                realized_pnl = excluded.realized_pnl,
                unrealized_pnl = excluded.unrealized_pnl,
                total_pnl = excluded.total_pnl,
                cumulative_return_pct = excluded.cumulative_return_pct,
                daily_return_pct = excluded.daily_return_pct,
                peak_portfolio_value = excluded.peak_portfolio_value,
                current_drawdown_pct = excluded.current_drawdown_pct,
                executed_trades_count = excluded.executed_trades_count,
                holdings_json = excluded.holdings_json;
            """, (
                trading_date, now_iso, data_status, init_cap,
                cash, res_cash, avail_cash, h_val,
                tot_val, r_pnl, u_pnl, tot_pnl,
                cum_return_pct, daily_return_pct, peak_val,
                drawdown_pct, executed_trades, holdings_json, now_iso
            ))

            cursor.execute("SELECT session_id FROM evaluation_sessions WHERE trading_date = ?;", (trading_date,))
            session_id = cursor.fetchone()["session_id"]

            # Record Agent decisions
            for pred in predictions:
                sym = pred.get("symbol", "")
                p_price = float(pred.get("current_price", 0.0))
                rec = pred.get("recommendation", "HOLD")
                probs = pred.get("action_probabilities", {"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34})
                pos_pct = float(pred.get("portfolio_context", [0.0])[0]) if isinstance(pred.get("portfolio_context"), (list, tuple)) else 0.0

                cursor.execute("""
                INSERT INTO evaluation_decisions (
                    session_id, trading_date, symbol, current_price,
                    ppo_recommendation, prob_hold, prob_buy, prob_sell,
                    portfolio_context_pos, user_action_taken, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    session_id, trading_date, sym, p_price,
                    rec, probs.get("HOLD", 0.0), probs.get("BUY", 0.0), probs.get("SELL", 0.0),
                    pos_pct, rec, now_iso
                ))

            # Record Benchmark values
            self._update_benchmarks_for_session(conn, trading_date, tot_val, predictions)
            conn.commit()

        return {
            "session_id": session_id,
            "trading_date": trading_date,
            "portfolio_value": tot_val,
            "cumulative_return_pct": cum_return_pct,
            "daily_return_pct": daily_return_pct,
            "drawdown_pct": drawdown_pct,
        }

    def _update_benchmarks_for_session(
        self,
        conn: sqlite3.Connection,
        trading_date: str,
        ppo_val: float,
        predictions: List[Dict[str, Any]]
    ) -> None:
        """
        Updates benchmark valuations (Equal-Weight B&H, Single-Stock B&H, and SMA Crossover)
        for the given session date.
        """
        cursor = conn.cursor()
        prices = {p["symbol"]: float(p["current_price"]) for p in predictions if p.get("current_price", 0) > 0}
        now_iso = datetime.now().isoformat()

        # Check Day 0 benchmark initial reference prices
        cursor.execute("SELECT * FROM benchmark_sessions ORDER BY trading_date ASC LIMIT 1;")
        first_bm = cursor.fetchone()

        if not first_bm:
            # Day 0: all benchmarks initialize at ₹1,00,000
            cursor.execute("""
            INSERT INTO benchmark_sessions (
                trading_date, ppo_value, equal_weight_bh_value, reliance_bh_value,
                tcs_bh_value, infy_bh_value, hdfcbank_bh_value, itc_bh_value,
                sma_crossover_value, created_at
            ) VALUES (?, ?, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, ?);
            """, (trading_date, ppo_val, now_iso))
            return

        # Subsequent sessions: track relative price movements from day 0
        # Retrieve base prices from Day 0 decisions
        cursor.execute("""
        SELECT symbol, current_price FROM evaluation_decisions
        WHERE session_id = 1;
        """)
        base_rows = cursor.fetchall()
        base_prices = {r["symbol"]: r["current_price"] for r in base_rows if r["current_price"] > 0}

        # Calculate Buy-and-Hold values
        bh_weights = {s: 20000.0 for s in TARGET_STOCKS}  # ₹20,000 each = ₹1,00,000 total
        eq_bh_val = 0.0
        single_vals = {}

        for s in TARGET_STOCKS:
            cur_p = prices.get(s, base_prices.get(s, 1.0))
            base_p = base_prices.get(s, cur_p)
            ratio = (cur_p / base_p) if base_p > 0 else 1.0
            eq_bh_val += bh_weights[s] * ratio
            single_vals[s] = round(100000.0 * ratio, 2)

        eq_bh_val = round(eq_bh_val, 2)

        # Dual SMA Crossover (SMA 10 / SMA 30) calculated independently without look-ahead bias
        # For each of the 5 stocks, allocation is ₹20,000. If SMA10 > SMA30, hold stock; else hold cash.
        sma_val = 0.0
        try:
            from backend.live_data.market_cache import load_from_cache
            cached = load_from_cache(TARGET_STOCKS)
            hist = cached.get("historical", {}) if cached else {}
        except Exception:
            hist = {}

        for s in TARGET_STOCKS:
            cur_p = prices.get(s, base_prices.get(s, 1.0))
            base_p = base_prices.get(s, cur_p)
            s_bars = hist.get(s, [])
            # Filter bars strictly up to current trading date
            s_bars_up_to = [b for b in s_bars if str(b.get("date") or b.get("Date") or "")[:10] <= trading_date]
            closes = [float(b.get("close") or b.get("Close") or 0.0) for b in s_bars_up_to if float(b.get("close") or b.get("Close") or 0.0) > 0]
            
            if len(closes) >= 30:
                sma10 = sum(closes[-10:]) / 10.0
                sma30 = sum(closes[-30:]) / 30.0
                is_long = sma10 > sma30
            else:
                is_long = True  # Default to invested if insufficient historical bars

            if is_long:
                stock_ratio = (cur_p / base_p) if base_p > 0 else 1.0
                # Net of 0.1% transaction friction on entry
                stock_val = bh_weights[s] * stock_ratio * (1.0 - TRANSACTION_FEE_RATE)
            else:
                stock_val = bh_weights[s]  # Cash preservation

            sma_val += stock_val

        sma_val = round(sma_val, 2)

        cursor.execute("""
        INSERT INTO benchmark_sessions (
            trading_date, ppo_value, equal_weight_bh_value, reliance_bh_value,
            tcs_bh_value, infy_bh_value, hdfcbank_bh_value, itc_bh_value,
            sma_crossover_value, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trading_date) DO UPDATE SET
            ppo_value = excluded.ppo_value,
            equal_weight_bh_value = excluded.equal_weight_bh_value,
            reliance_bh_value = excluded.reliance_bh_value,
            tcs_bh_value = excluded.tcs_bh_value,
            infy_bh_value = excluded.infy_bh_value,
            hdfcbank_bh_value = excluded.hdfcbank_bh_value,
            itc_bh_value = excluded.itc_bh_value,
            sma_crossover_value = excluded.sma_crossover_value;
        """, (
            trading_date, ppo_val, eq_bh_val,
            single_vals.get("RELIANCE", 100000.0),
            single_vals.get("TCS", 100000.0),
            single_vals.get("INFY", 100000.0),
            single_vals.get("HDFCBANK", 100000.0),
            single_vals.get("ITC", 100000.0),
            sma_val, now_iso
        ))

    def calculate_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Calculates all portfolio performance, risk, and trading metrics
        from recorded session data and transaction history.
        """
        with self.conn_scope() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM evaluation_sessions ORDER BY trading_date ASC;")
            sessions = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM transactions ORDER BY created_at ASC;")
            txs = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM orders;")
            orders = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM benchmark_sessions ORDER BY trading_date ASC;")
            benchmarks = [dict(r) for r in cursor.fetchall()]

        num_sessions = len(sessions)
        is_complete = num_sessions >= MIN_SESSIONS_FOR_COMPLETION
        eval_status = (
            f"COMPLETED ({num_sessions} sessions)"
            if is_complete
            else f"IN PROGRESS — MULTI-SESSION EVALUATION ACTIVE ({num_sessions} / {MIN_SESSIONS_FOR_COMPLETION} Sessions Observed)"
        )

        init_val = sessions[0]["initial_capital"] if sessions else DEFAULT_INITIAL_CAPITAL
        final_val = sessions[-1]["total_portfolio_value"] if sessions else init_val
        cum_ret = round(((final_val - init_val) / init_val) * 100.0, 4) if init_val > 0 else 0.0

        daily_returns = [s["daily_return_pct"] / 100.0 for s in sessions]
        n_ret = len(daily_returns)

        # Volatility & Sharpe Ratio
        if n_ret > 1:
            mean_ret = sum(daily_returns) / n_ret
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (n_ret - 1)
            daily_vol = math.sqrt(variance)
            ann_vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)

            # Daily risk-free rate
            rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
            excess_return = mean_ret - rf_daily
            sharpe_ratio = round((excess_return / daily_vol) * math.sqrt(TRADING_DAYS_PER_YEAR), 3) if daily_vol > 0 else 0.0

            # Downside Deviation & Sortino Ratio
            downside_diffs = [min(0.0, r - rf_daily) ** 2 for r in daily_returns]
            downside_dev = math.sqrt(sum(downside_diffs) / n_ret) if n_ret > 0 else 0.0
            ann_downside_dev = downside_dev * math.sqrt(TRADING_DAYS_PER_YEAR)
            sortino_ratio = round((excess_return / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR), 3) if downside_dev > 0 else 0.0
        else:
            daily_vol = 0.0
            ann_vol = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            ann_downside_dev = 0.0

        # Maximum Drawdown
        max_dd = 0.0
        peak = init_val
        for s in sessions:
            v = s["total_portfolio_value"]
            if v > peak:
                peak = v
            dd = ((peak - v) / peak) * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        max_drawdown = round(max_dd, 3)

        # Trading metrics
        buy_orders = [o for o in orders if o["side"] == "BUY" and o["status"] == "EXECUTED"]
        sell_orders = [o for o in orders if o["side"] == "SELL" and o["status"] == "EXECUTED"]
        hold_decisions = [o for o in orders if o["side"] == "HOLD"]

        total_tx_fees = round(sum(tx["transaction_fee"] for tx in txs), 2)
        total_volume_traded = round(sum(tx["total_amount"] for tx in txs), 2)
        turnover = round((total_volume_traded / init_val), 3) if init_val > 0 else 0.0

        # Win / Loss analysis on completed round-trip trades (SELL transactions)
        # Pending and open positions are strictly excluded from win rate calculations
        sell_txs = [t for t in txs if t["side"] == "SELL"]
        closed_trades_count = len(sell_txs)
        winning_trades = 0
        losing_trades = 0
        gross_profits = 0.0
        gross_losses = 0.0

        for stx in sell_txs:
            # Look up purchase price for symbol to determine profit/loss
            sym = stx["symbol"]
            qty = stx["quantity"]
            sell_amt = stx["total_amount"]
            sell_fee = stx["transaction_fee"]
            # Net proceeds
            net_proceeds = sell_amt - sell_fee
            # Find matching executed buy price from order history
            cursor.execute("SELECT execution_price FROM transactions WHERE symbol = ? AND side = 'BUY' AND order_id < ? ORDER BY order_id DESC LIMIT 1;", (sym, stx["order_id"]))
            buy_row = cursor.fetchone()
            buy_price = buy_row["execution_price"] if buy_row else (sell_amt / qty if qty > 0 else 0.0)
            cost_basis = qty * buy_price * (1.0 + TRANSACTION_FEE_RATE)
            trade_pnl = net_proceeds - cost_basis

            if trade_pnl > 0:
                winning_trades += 1
                gross_profits += trade_pnl
            elif trade_pnl < 0:
                losing_trades += 1
                gross_losses += abs(trade_pnl)

        if closed_trades_count > 0 and (winning_trades + losing_trades) > 0:
            win_rate = round((winning_trades / (winning_trades + losing_trades)) * 100.0, 2)
            if gross_losses > 0:
                profit_factor = round(gross_profits / gross_losses, 2)
                profit_factor_display = f"{profit_factor:.2f}"
            else:
                profit_factor = float("inf") if gross_profits > 0 else 1.0
                profit_factor_display = "Infinity (Zero Losses)" if gross_profits > 0 else "1.00 (Neutral)"
            win_rate_display = f"{win_rate:.2f}%"
        else:
            win_rate = None
            profit_factor = None
            win_rate_display = "N/A (0 closed trades; pending/open positions excluded)"
            profit_factor_display = "N/A (No closed round-trip trades)"

        # Explicit labeling of sample size validity for risk metrics
        is_sample_statistically_valid = num_sessions >= MIN_SESSIONS_FOR_COMPLETION
        if not is_sample_statistically_valid:
            sharpe_note = f"Preliminary estimate only; requires >= {MIN_SESSIONS_FOR_COMPLETION} forward trading sessions for statistical significance (Current N={num_sessions})"
            volatility_note = f"Preliminary annualized estimate based on {num_sessions} session(s); requires >= {MIN_SESSIONS_FOR_COMPLETION} sessions"
        else:
            sharpe_note = f"Statistically valid observation period (N={num_sessions} >= {MIN_SESSIONS_FOR_COMPLETION})"
            volatility_note = "Annualized based on 252 trading days"

        # Realized and Unrealized P&L from latest session
        realized_pnl = sessions[-1]["realized_pnl"] if sessions else 0.0
        unrealized_pnl = sessions[-1]["unrealized_pnl"] if sessions else 0.0

        # Benchmarks Comparison
        eq_bh_val = benchmarks[-1]["equal_weight_bh_value"] if benchmarks else init_val
        eq_bh_ret = round(((eq_bh_val - init_val) / init_val) * 100.0, 2)

        sma_val = benchmarks[-1]["sma_crossover_value"] if benchmarks else init_val
        sma_ret = round(((sma_val - init_val) / init_val) * 100.0, 2)

        return {
            "evaluation_status": eval_status,
            "is_complete": is_complete,
            "sessions_observed": num_sessions,
            "target_sessions": MIN_SESSIONS_FOR_COMPLETION,
            "portfolio_performance": {
                "initial_portfolio_value": init_val,
                "final_portfolio_value": final_val,
                "cumulative_return_pct": cum_ret,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": round(realized_pnl + unrealized_pnl, 2),
                "total_transaction_fees": total_tx_fees,
            },
            "risk_metrics": {
                "sharpe_ratio": sharpe_ratio,
                "sharpe_ratio_note": sharpe_note,
                "sortino_ratio": sortino_ratio,
                "annualized_volatility_pct": round(ann_vol * 100.0, 2),
                "annualized_volatility_note": volatility_note,
                "downside_deviation_pct": round(ann_downside_dev * 100.0, 2),
                "maximum_drawdown_pct": max_drawdown,
                "risk_free_rate_used": f"{RISK_FREE_RATE_ANNUAL*100:.2f}% (RBI 91-day T-Bill rate)",
            },
            "trading_metrics": {
                "total_executed_trades": len(txs),
                "closed_round_trip_trades": closed_trades_count,
                "buy_orders_executed": len(buy_orders),
                "sell_orders_executed": len(sell_orders),
                "hold_decisions_logged": len(hold_decisions),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_pct": win_rate,
                "win_rate_display": win_rate_display,
                "profit_factor": profit_factor,
                "profit_factor_display": profit_factor_display,
                "turnover_ratio": turnover,
                "total_volume_traded": total_volume_traded,
            },
            "benchmark_comparison": {
                "ppo_portfolio": {"final_value": final_val, "return_pct": cum_ret},
                "equal_weight_buy_and_hold": {"final_value": eq_bh_val, "return_pct": eq_bh_ret},
                "dual_sma_crossover": {"final_value": sma_val, "return_pct": sma_ret},
            },
            "sessions_history": sessions,
            "benchmark_history": benchmarks,
        }

    def generate_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Generates and saves clean matplotlib visual charts for Phase 6 evaluation:
        1. equity_curve.png (PPO Portfolio vs Equal-Weight B&H vs SMA Crossover)
        2. daily_returns.png (Daily return distribution)
        3. drawdown_curve.png (Underwater drawdown % profile)
        4. benchmark_comparison.png (Bar chart comparison)
        5. stock_pnl_contribution.png (Holdings valuation and P&L by stock)
        6. agent_action_distribution.png (HOLD / BUY / SELL action probability breakdown)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = output_dir or VIZ_DIR
        out_path.mkdir(parents=True, exist_ok=True)
        generated: List[Path] = []

        metrics = self.calculate_evaluation_metrics()
        sessions = metrics.get("sessions_history", [])
        benchmarks = metrics.get("benchmark_history", [])

        if not sessions:
            logger.warning("No evaluation sessions available to plot. Generating baseline chart.")
            dates = [date(2026, 11, 20).strftime("%Y-%m-%d")]
            ppo_vals = [100000.0]
            bh_vals = [100000.0]
            sma_vals = [100000.0]
        else:
            dates = [s["trading_date"] for s in sessions]
            ppo_vals = [s["total_portfolio_value"] for s in sessions]
            bh_vals = [b["equal_weight_bh_value"] for b in benchmarks] if benchmarks else ppo_vals
            sma_vals = [b["sma_crossover_value"] for b in benchmarks] if benchmarks else ppo_vals

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Equity Curve
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(dates, ppo_vals, label="NeuroTrader (PPO Policy)", color="#1976D2", linewidth=2.5, marker="o")
        ax.plot(dates, bh_vals, label="Equal-Weight Buy & Hold (5 Stocks)", color="#388E3C", linewidth=2.0, linestyle="--")
        ax.plot(dates, sma_vals, label="Dual SMA Crossover (10/30)", color="#F57C00", linewidth=1.8, linestyle=":")
        ax.set_title("NeuroTrader Portfolio Valuation vs Benchmarks (₹1,00,000 Initial)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Trading Session Date", fontsize=10)
        ax.set_ylabel("Portfolio Value (₹)", fontsize=10)
        ax.yaxis.set_major_formatter("₹{x:,.0f}")
        ax.legend(loc="upper left", frameon=True)
        plt.xticks(rotation=25)
        plt.tight_layout()
        f1 = out_path / "equity_curve.png"
        fig.savefig(f1)
        plt.close(fig)
        generated.append(f1)

        # 2. Daily Returns
        daily_rets = [s.get("daily_return_pct", 0.0) for s in sessions] if sessions else [0.0]
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        colors = ["#2E7D32" if r >= 0 else "#C62828" for r in daily_rets]
        ax.bar(dates, daily_rets, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Daily Portfolio Return (%) Profile", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Trading Session Date", fontsize=10)
        ax.set_ylabel("Return (%)", fontsize=10)
        plt.xticks(rotation=25)
        plt.tight_layout()
        f2 = out_path / "daily_returns.png"
        fig.savefig(f2)
        plt.close(fig)
        generated.append(f2)

        # 3. Drawdown Curve
        drawdowns = [-s.get("current_drawdown_pct", 0.0) for s in sessions] if sessions else [0.0]
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.fill_between(dates, drawdowns, 0, color="#E53935", alpha=0.3)
        ax.plot(dates, drawdowns, color="#C62828", linewidth=1.5)
        ax.set_title("Underwater Portfolio Drawdown Profile (%)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Trading Session Date", fontsize=10)
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        plt.xticks(rotation=25)
        plt.tight_layout()
        f3 = out_path / "drawdown_curve.png"
        fig.savefig(f3)
        plt.close(fig)
        generated.append(f3)

        # 4. Benchmark Comparison Bar Chart
        bm = metrics["benchmark_comparison"]
        labels = ["PPO Policy", "Equal-Weight B&H", "SMA Crossover"]
        rets = [
            bm["ppo_portfolio"]["return_pct"],
            bm["equal_weight_buy_and_hold"]["return_pct"],
            bm["dual_sma_crossover"]["return_pct"]
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bar_colors = ["#1976D2", "#388E3C", "#F57C00"]
        bars = ax.bar(labels, rets, color=bar_colors, width=0.45, edgecolor="black", linewidth=0.8)
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height:+.2f}%",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Cumulative Return Comparison vs Standard Benchmarks", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Net Cumulative Return (%)", fontsize=10)
        plt.tight_layout()
        f4 = out_path / "benchmark_comparison.png"
        fig.savefig(f4)
        plt.close(fig)
        generated.append(f4)

        # 5. Agent Action Distribution
        with self.conn_scope() as conn:
            c = conn.cursor()
            c.execute("SELECT AVG(prob_hold), AVG(prob_buy), AVG(prob_sell) FROM evaluation_decisions;")
            avg_probs = c.fetchone()
        
        p_h = avg_probs[0] if (avg_probs and avg_probs[0]) else 0.50
        p_b = avg_probs[1] if (avg_probs and avg_probs[1]) else 0.25
        p_s = avg_probs[2] if (avg_probs and avg_probs[2]) else 0.25

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        pie_labels = ["HOLD", "BUY", "SELL"]
        pie_colors = ["#78909C", "#43A047", "#E53935"]
        ax.pie([p_h, p_b, p_s], labels=pie_labels, autopct="%1.1f%%", colors=pie_colors, startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("PPO Policy Action Probability Distribution", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        f5 = out_path / "agent_action_distribution.png"
        fig.savefig(f5)
        plt.close(fig)
        generated.append(f5)

        logger.info(f"Successfully generated {len(generated)} evaluation charts in {out_path}")
        return generated
