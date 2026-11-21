"""
SQLite Persistence Layer for NeuroTrade Paper Trading.

Database location: data/live/paper_trading.db
Keeps paper-trading data strictly isolated from training datasets and checkpoints.

Tables:
- portfolio_summary: initial_capital, cash, total_value, realized_pnl, created_at, updated_at
- holdings: symbol, shares, avg_purchase_price, total_invested, current_price, current_value, unrealized_pnl, updated_at
- orders: order_id, symbol, side (BUY/SELL/HOLD), quantity, decision_date, decision_time, agent_recommendation, agent_probabilities, user_action, estimated_price, status (PENDING/EXECUTED/REJECTED), rejection_reason, created_at
- transactions: tx_id, order_id, symbol, side, quantity, execution_price, execution_date, execution_time, transaction_fee, total_amount, created_at
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "data" / "live"
DB_PATH = DB_DIR / "paper_trading.db"

DEFAULT_INITIAL_CAPITAL = 100000.0  # ₹1,00,000
TRANSACTION_FEE_RATE = 0.001        # 0.1% transaction fee

TARGET_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

# Re-entrant thread lock to guarantee exactly-once execution and race-free concurrent processing
_execution_lock = threading.RLock()


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite paper trading database with WAL mode."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> None:
    """Initializes tables and seeds default portfolio if not already present."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Portfolio summary table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            initial_capital REAL NOT NULL,
            cash REAL NOT NULL,
            total_value REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

        # 2. Holdings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            symbol TEXT PRIMARY KEY,
            shares INTEGER NOT NULL DEFAULT 0,
            avg_purchase_price REAL NOT NULL DEFAULT 0.0,
            total_invested REAL NOT NULL DEFAULT 0.0,
            current_price REAL NOT NULL DEFAULT 0.0,
            current_value REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL
        );
        """)

        # 3. Orders table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL', 'HOLD')),
            quantity INTEGER NOT NULL,
            decision_date TEXT NOT NULL,
            decision_time TEXT NOT NULL,
            agent_recommendation TEXT NOT NULL,
            agent_probabilities TEXT NOT NULL,
            user_action TEXT NOT NULL,
            estimated_price REAL NOT NULL,
            target_execution TEXT NOT NULL DEFAULT 'NEXT_SESSION_OPEN',
            status TEXT NOT NULL CHECK (status IN ('PENDING', 'EXECUTED', 'REJECTED', 'CANCELLED')),
            rejection_reason TEXT,
            created_at TEXT NOT NULL
        );
        """)

        # Migration: Ensure existing orders table supports CANCELLED status
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='orders';")
        orders_sql_row = cursor.fetchone()
        if orders_sql_row and "'CANCELLED'" not in orders_sql_row[0]:
            cursor.execute("PRAGMA foreign_keys = OFF;")
            cursor.execute("""
            CREATE TABLE orders_migration_temp (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL', 'HOLD')),
                quantity INTEGER NOT NULL,
                decision_date TEXT NOT NULL,
                decision_time TEXT NOT NULL,
                agent_recommendation TEXT NOT NULL,
                agent_probabilities TEXT NOT NULL,
                user_action TEXT NOT NULL,
                estimated_price REAL NOT NULL,
                target_execution TEXT NOT NULL DEFAULT 'NEXT_SESSION_OPEN',
                status TEXT NOT NULL CHECK (status IN ('PENDING', 'EXECUTED', 'REJECTED', 'CANCELLED')),
                rejection_reason TEXT,
                created_at TEXT NOT NULL
            );
            """)
            cursor.execute("INSERT INTO orders_migration_temp SELECT * FROM orders;")
            cursor.execute("DROP TABLE orders;")
            cursor.execute("ALTER TABLE orders_migration_temp RENAME TO orders;")
            cursor.execute("PRAGMA foreign_keys = ON;")

        # 4. Transactions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(order_id),
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
            quantity INTEGER NOT NULL,
            execution_price REAL NOT NULL,
            execution_date TEXT NOT NULL,
            execution_time TEXT NOT NULL,
            transaction_fee REAL NOT NULL,
            total_amount REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        # Seed initial portfolio if empty
        cursor.execute("SELECT COUNT(*) FROM portfolio_summary WHERE id = 1;")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.now().isoformat()
            cursor.execute("""
            INSERT INTO portfolio_summary (id, initial_capital, cash, total_value, realized_pnl, created_at, updated_at)
            VALUES (1, ?, ?, ?, 0.0, ?, ?);
            """, (initial_capital, initial_capital, initial_capital, now_iso, now_iso))

            for symbol in TARGET_STOCKS:
                cursor.execute("""
                INSERT OR IGNORE INTO holdings (symbol, shares, avg_purchase_price, total_invested, current_price, current_value, unrealized_pnl, updated_at)
                VALUES (?, 0, 0.0, 0.0, 0.0, 0.0, 0.0, ?);
                """, (symbol, now_iso))

        conn.commit()


def get_portfolio() -> Dict[str, Any]:
    """Retrieves current portfolio summary, holdings, and return statistics."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM portfolio_summary WHERE id = 1;")
        summary_row = cursor.fetchone()
        if not summary_row:
            return {}

        cursor.execute("SELECT * FROM holdings ORDER BY symbol ASC;")
        holdings_rows = cursor.fetchall()

    holdings = {}
    total_holding_value = 0.0
    total_unrealized_pnl = 0.0

    for r in holdings_rows:
        sym = r["symbol"]
        shares = r["shares"]
        avg_price = r["avg_purchase_price"]
        cur_price = r["current_price"]
        cur_val = r["current_value"]
        u_pnl = r["unrealized_pnl"]

        total_holding_value += cur_val
        total_unrealized_pnl += u_pnl

        holdings[sym] = {
            "shares": shares,
            "avg_purchase_price": avg_price,
            "total_invested": r["total_invested"],
            "current_price": cur_price,
            "current_value": cur_val,
            "unrealized_pnl": u_pnl,
            "unrealized_pnl_pct": round((u_pnl / r["total_invested"] * 100), 2) if r["total_invested"] > 0 else 0.0,
            "updated_at": r["updated_at"],
        }

    init_cap = summary_row["initial_capital"]
    cash = summary_row["cash"]
    realized_pnl = summary_row["realized_pnl"]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(quantity * estimated_price * (1.0 + ?)) as res_cash FROM orders WHERE side = 'BUY' AND status = 'PENDING';", (TRANSACTION_FEE_RATE,))
        res_cash_row = cursor.fetchone()
        reserved_cash = round(res_cash_row["res_cash"] or 0.0, 2)

    available_cash = round(max(0.0, cash - reserved_cash), 2)
    total_val = cash + total_holding_value
    total_return_pct = round(((total_val - init_cap) / init_cap) * 100.0, 2)

    return {
        "initial_capital": init_cap,
        "cash": cash,
        "ledger_cash": cash,
        "reserved_cash": reserved_cash,
        "available_cash": available_cash,
        "holdings_value": round(total_holding_value, 2),
        "total_value": round(total_val, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_pnl": round(realized_pnl + total_unrealized_pnl, 2),
        "total_return_pct": total_return_pct,
        "holdings": holdings,
        "updated_at": summary_row["updated_at"],
    }


def update_holdings_prices(current_prices: Dict[str, float]) -> None:
    """Updates current market prices and mark-to-market valuations for holdings."""
    init_db()
    now_iso = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        total_holding_val = 0.0

        for sym, price in current_prices.items():
            if price <= 0:
                continue
            cursor.execute("SELECT shares, avg_purchase_price, total_invested FROM holdings WHERE symbol = ?;", (sym,))
            row = cursor.fetchone()
            if not row:
                continue

            shares = row["shares"]
            cur_val = round(shares * price, 2)
            invested = row["total_invested"]
            u_pnl = round(cur_val - invested, 2) if shares > 0 else 0.0
            total_holding_val += cur_val

            cursor.execute("""
            UPDATE holdings
            SET current_price = ?, current_value = ?, unrealized_pnl = ?, updated_at = ?
            WHERE symbol = ?;
            """, (price, cur_val, u_pnl, now_iso, sym))

        cursor.execute("SELECT cash FROM portfolio_summary WHERE id = 1;")
        cash_row = cursor.fetchone()
        if cash_row:
            total_portfolio_val = round(cash_row["cash"] + total_holding_val, 2)
            cursor.execute("""
            UPDATE portfolio_summary
            SET total_value = ?, updated_at = ?
            WHERE id = 1;
            """, (total_portfolio_val, now_iso))

        conn.commit()


def submit_order(
    symbol: str,
    side: str,
    quantity: int,
    current_price: float,
    agent_recommendation: str = "USER_DIRECT",
    agent_probabilities: Optional[Dict[str, float]] = None,
    decision_date: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submits a user paper trading order.
    
    Validation:
    - Side must be BUY, SELL, or HOLD.
    - Quantity must be positive integer (for BUY/SELL).
    - Available cash must cover quantity * price * (1 + TRANSACTION_FEE_RATE) for BUY.
    - Holdings must have at least quantity shares for SELL.
    - HOLD creates no transaction and requires quantity=0.
    
    Execution Timing:
    - Order is created with status PENDING for execution at next session Open (t+1).
    - If validation fails, order is logged in SQLite with status REJECTED and reason for auditability.
    """
    init_db()
    side_str = str(side).upper() if side else ""
    sym_str = str(symbol).upper() if symbol else ""
    now = datetime.now()
    if created_at is None:
        created_at = now.isoformat()
    now_iso = created_at
    now_time = created_at[11:19] if len(created_at) >= 19 else now.strftime("%H:%M:%S")

    if agent_probabilities is None:
        agent_probabilities = {"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34}
    if decision_date is None:
        decision_date = now.strftime("%Y-%m-%d")

    # 1. Validation checks
    rejection_reason = None
    if sym_str not in TARGET_STOCKS:
        rejection_reason = f"Invalid symbol: {symbol}. Must be one of {TARGET_STOCKS}"
        sym_str = "RELIANCE"  # safe fallback symbol for db column constraint

    if side_str not in ["BUY", "SELL", "HOLD"]:
        rejection_reason = f"Invalid action: {side}. Must be BUY, SELL, or HOLD"
        side_str = "HOLD"

    if rejection_reason is None:
        if side_str == "HOLD":
            if quantity != 0:
                quantity = 0  # HOLD maintains position with 0 quantity
        else:
            if not isinstance(quantity, int) or quantity <= 0:
                rejection_reason = f"Quantity must be a positive integer, got: {quantity}"
                quantity = 0

    # Handle valid HOLD decision
    if rejection_reason is None and side_str == "HOLD":
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO orders (
                symbol, side, quantity, decision_date, decision_time,
                agent_recommendation, agent_probabilities, user_action,
                estimated_price, target_execution, status, rejection_reason, created_at
            ) VALUES (?, 'HOLD', 0, ?, ?, ?, ?, 'HOLD', ?, 'NONE', 'EXECUTED', 'HOLD action maintains position', ?);
            """, (
                sym_str, decision_date, now_time, agent_recommendation,
                json.dumps(agent_probabilities), current_price, now_iso
            ))
            order_id = cursor.lastrowid
            conn.commit()

        return {
            "success": True,
            "order_id": order_id,
            "status": "EXECUTED",
            "side": "HOLD",
            "symbol": sym_str,
            "message": "HOLD recorded. Position maintained with zero transaction fees."
        }

    # For BUY/SELL or rejected orders
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cash FROM portfolio_summary WHERE id = 1;")
        cash_row = cursor.fetchone()
        cash = cash_row["cash"] if cash_row else 0.0

        cursor.execute("SELECT shares FROM holdings WHERE symbol = ?;", (sym_str,))
        hold_row = cursor.fetchone()
        shares_held = hold_row["shares"] if hold_row else 0

        # Check pending BUY orders already reserving cash
        cursor.execute("""
        SELECT SUM(quantity * estimated_price * (1.0 + ?)) as reserved_cash
        FROM orders WHERE side = 'BUY' AND status = 'PENDING';
        """, (TRANSACTION_FEE_RATE,))
        res_cash_row = cursor.fetchone()
        reserved_cash = res_cash_row["reserved_cash"] or 0.0
        available_cash = cash - reserved_cash

        # Check pending SELL orders already reserving shares
        cursor.execute("""
        SELECT SUM(quantity) as reserved_shares
        FROM orders WHERE symbol = ? AND side = 'SELL' AND status = 'PENDING';
        """, (sym_str,))
        res_shares_row = cursor.fetchone()
        reserved_shares = res_shares_row["reserved_shares"] or 0
        available_shares = shares_held - reserved_shares

        est_fee = round(quantity * current_price * TRANSACTION_FEE_RATE, 2)
        est_total = round(quantity * current_price + est_fee, 2)

        # Funds / holdings check if not already rejected
        if rejection_reason is None:
            if side_str == "BUY" and est_total > available_cash:
                rejection_reason = (
                    f"Insufficient available cash. Required: ₹{est_total:,.2f} "
                    f"(including ₹{est_fee:.2f} fee), Available: ₹{available_cash:,.2f} "
                    f"(Total Cash: ₹{cash:,.2f}, Reserved in Pending: ₹{reserved_cash:,.2f})"
                )
            elif side_str == "SELL" and quantity > available_shares:
                rejection_reason = (
                    f"Insufficient shares held. Required: {quantity}, "
                    f"Available: {available_shares} (Owned: {shares_held}, Reserved in Pending: {reserved_shares})"
                )

        status = "REJECTED" if rejection_reason else "PENDING"

        cursor.execute("""
        INSERT INTO orders (
            symbol, side, quantity, decision_date, decision_time,
            agent_recommendation, agent_probabilities, user_action,
            estimated_price, target_execution, status, rejection_reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEXT_SESSION_OPEN', ?, ?, ?);
        """, (
            sym_str, side_str, quantity, decision_date, now_time,
            agent_recommendation, json.dumps(agent_probabilities),
            str(side), current_price, status, rejection_reason, now_iso
        ))
        order_id = cursor.lastrowid
        conn.commit()

    if rejection_reason:
        return {"success": False, "order_id": order_id, "status": "REJECTED", "reason": rejection_reason}

    return {
        "success": True,
        "order_id": order_id,
        "status": "PENDING",
        "symbol": sym_str,
        "side": side_str,
        "quantity": quantity,
        "decision_date": decision_date,
        "target_execution": "NEXT_SESSION_OPEN",
        "estimated_price": current_price,
        "estimated_fee": est_fee,
        "estimated_total": est_total,
        "message": f"Order #{order_id} queued as PENDING. Will execute at next trading session Open."
    }


def execute_order_at_open(order_id: int, open_price: float, open_date: str) -> Dict[str, Any]:
    """
    Executes a single PENDING order at the next session's verified Open price.
    Applies the TRANSACTION_FEE_RATE fee, updates portfolio cash and holdings atomically.
    Protects other pending orders' reservations and prevents race conditions via thread locking.
    """
    if open_price <= 0:
        raise ValueError(f"Invalid open price: {open_price}")

    with _execution_lock:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ? AND status = 'PENDING';", (order_id,))
            order = cursor.fetchone()
            if not order:
                return {"success": False, "reason": f"No pending order found with ID {order_id}"}

            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            gross_amt = round(quantity * open_price, 2)
            fee = round(gross_amt * TRANSACTION_FEE_RATE, 2)
            now_iso = datetime.now().isoformat()
            now_time = datetime.now().strftime("%H:%M:%S")

            cursor.execute("SELECT cash, realized_pnl FROM portfolio_summary WHERE id = 1;")
            summary = cursor.fetchone()
            cash = summary["cash"]
            realized_pnl = summary["realized_pnl"]

            cursor.execute("SELECT shares, avg_purchase_price, total_invested FROM holdings WHERE symbol = ?;", (symbol,))
            hold = cursor.fetchone()
            shares = hold["shares"]
            avg_price = hold["avg_purchase_price"]
            invested = hold["total_invested"]

            if side == "BUY":
                total_cost = round(gross_amt + fee, 2)

                # Task 4: Calculate total reserved amount for all OTHER pending BUY orders
                cursor.execute("""
                SELECT SUM(quantity * estimated_price * (1.0 + ?)) as other_res
                FROM orders
                WHERE side = 'BUY' AND status = 'PENDING' AND order_id != ?;
                """, (TRANSACTION_FEE_RATE, order_id))
                other_res_row = cursor.fetchone()
                other_reserved_cash = round(other_res_row["other_res"] or 0.0, 2)

                # Cash available to this execution without consuming other orders' reservations
                available_for_execution = round(cash - other_reserved_cash, 2)

                if total_cost > available_for_execution:
                    rejection_msg = (
                        f"Execution affordability check failed (price surge or insufficient unreserved funds): "
                        f"Required ₹{total_cost:,.2f} (Gross ₹{gross_amt:,.2f} + Fee ₹{fee:.2f}) exceeds available unreserved cash "
                        f"₹{available_for_execution:,.2f} (Ledger cash: ₹{cash:,.2f}, "
                        f"Reserved for other pending orders: ₹{other_reserved_cash:,.2f})"
                    )
                    cursor.execute("UPDATE orders SET status = 'REJECTED', rejection_reason = ? WHERE order_id = ?;",
                                   (rejection_msg, order_id))
                    conn.commit()
                    return {"success": False, "order_id": order_id, "status": "REJECTED", "reason": rejection_msg}

                new_cash = round(cash - total_cost, 2)
                new_shares = shares + quantity
                new_invested = round(invested + gross_amt + fee, 2)  # cost basis includes transaction fee
                new_avg_price = round(new_invested / new_shares, 2) if new_shares > 0 else 0.0

                cursor.execute("""
                UPDATE holdings
                SET shares = ?, avg_purchase_price = ?, total_invested = ?, current_price = ?,
                    current_value = ?, unrealized_pnl = ?, updated_at = ?
                WHERE symbol = ?;
                """, (new_shares, new_avg_price, new_invested, open_price, round(new_shares * open_price, 2),
                      round((new_shares * open_price) - new_invested, 2), now_iso, symbol))

                cursor.execute("UPDATE portfolio_summary SET cash = ?, updated_at = ? WHERE id = 1;", (new_cash, now_iso))

            elif side == "SELL":
                # Task 5: Calculate shares reserved by all OTHER pending SELL orders for this symbol
                cursor.execute("""
                SELECT SUM(quantity) as other_shares
                FROM orders
                WHERE symbol = ? AND side = 'SELL' AND status = 'PENDING' AND order_id != ?;
                """, (symbol, order_id))
                other_shares_row = cursor.fetchone()
                other_reserved_shares = int(other_shares_row["other_shares"] or 0)

                available_shares_for_exec = shares - other_reserved_shares

                if quantity > available_shares_for_exec:
                    rejection_msg = (
                        f"Insufficient unreserved shares at Open: Needed {quantity}, "
                        f"have {available_shares_for_exec} available (Owned: {shares}, "
                        f"Reserved for other pending SELL orders: {other_reserved_shares})"
                    )
                    cursor.execute("UPDATE orders SET status = 'REJECTED', rejection_reason = ? WHERE order_id = ?;",
                                   (rejection_msg, order_id))
                    conn.commit()
                    return {"success": False, "order_id": order_id, "status": "REJECTED", "reason": rejection_msg}

                net_proceeds = round(gross_amt - fee, 2)
                new_cash = round(cash + net_proceeds, 2)
                new_shares = shares - quantity

                # Proportionate cost of shares sold
                cost_of_sold = round((quantity / shares) * invested, 2) if shares > 0 else 0.0
                gain_on_sale = round(net_proceeds - cost_of_sold, 2)
                new_realized_pnl = round(realized_pnl + gain_on_sale, 2)
                new_invested = round(invested - cost_of_sold, 2) if new_shares > 0 else 0.0
                new_avg_price = avg_price if new_shares > 0 else 0.0

                cursor.execute("""
                UPDATE holdings
                SET shares = ?, avg_purchase_price = ?, total_invested = ?, current_price = ?,
                    current_value = ?, unrealized_pnl = ?, updated_at = ?
                WHERE symbol = ?;
                """, (new_shares, new_avg_price, new_invested, open_price, round(new_shares * open_price, 2),
                      round((new_shares * open_price) - new_invested, 2) if new_shares > 0 else 0.0, now_iso, symbol))

                cursor.execute("""
                UPDATE portfolio_summary
                SET cash = ?, realized_pnl = ?, updated_at = ?
                WHERE id = 1;
                """, (new_cash, new_realized_pnl, now_iso))

            # Record transaction
            cursor.execute("""
            INSERT INTO transactions (
                order_id, symbol, side, quantity, execution_price,
                execution_date, execution_time, transaction_fee, total_amount, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (order_id, symbol, side, quantity, open_price, open_date, now_time, fee, gross_amt, now_iso))

            # Mark order EXECUTED
            cursor.execute("UPDATE orders SET status = 'EXECUTED' WHERE order_id = ?;", (order_id,))
            conn.commit()

        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "execution_price": open_price,
            "execution_date": open_date,
            "fee": fee,
            "gross_amount": gross_amt,
            "status": "EXECUTED"
        }


def cancel_order(order_id: int) -> Dict[str, Any]:
    """
    Cancels a pending order, releasing any escrowed cash or shares atomically.
    Only orders with status PENDING can be cancelled.
    """
    init_db()
    with _execution_lock:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ?;", (order_id,))
            order = cursor.fetchone()
            if not order:
                return {"success": False, "status": "NOT_FOUND", "reason": f"Order #{order_id} not found."}

            current_status = order["status"]
            if current_status != "PENDING":
                return {
                    "success": False,
                    "status": current_status,
                    "reason": f"Cannot cancel order #{order_id} because its status is already '{current_status}'."
                }

            cursor.execute("""
            UPDATE orders
            SET status = 'CANCELLED', rejection_reason = 'Cancelled by user'
            WHERE order_id = ?;
            """, (order_id,))
            conn.commit()

        return {
            "success": True,
            "order_id": order_id,
            "status": "CANCELLED",
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": order["quantity"],
            "message": f"Order #{order_id} ({order['side']} {order['quantity']} {order['symbol']}) cancelled successfully. Escrow released."
        }


def process_pending_orders_service(market_bars_dict: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """
    Single central execution service that processes all eligible PENDING orders.

    Guarantees:
    - Exactly-once execution (thread-safe and atomic).
    - Strict freshness and cutoff date validation: an order NEVER executes against a historical
      candle that predates its actual submission time or decision date.
    - Weekend and holiday handling via verified completed daily session bars.
    - Affordability safeguards protecting escrow of all other pending orders.
    - Rejection of unaffordable orders with immediate escrow release.
    - Returns structured summary of executed, rejected, and deferred orders.
    """
    init_db()
    with _execution_lock:
        if market_bars_dict is None:
            try:
                from backend.live_data.market_cache import load_from_cache
                cached = load_from_cache(TARGET_STOCKS)
                market_bars_dict = cached.get("historical", {}) if cached else {}
            except Exception as e:
                logger.warning(f"Failed to load cache in process_pending_orders_service: {e}")
                market_bars_dict = {}

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE status = 'PENDING' ORDER BY order_id ASC;")
            pending_orders = cursor.fetchall()

        if not pending_orders:
            return {
                "success": True,
                "processed_count": 0,
                "executed_count": 0,
                "rejected_count": 0,
                "deferred_count": 0,
                "executed_orders": [],
                "rejected_orders": [],
                "deferred_orders": [],
                "portfolio": get_portfolio(),
            }

        executed_list = []
        rejected_list = []
        deferred_list = []

        for order in pending_orders:
            order_id = order["order_id"]
            sym = order["symbol"]
            dec_date = str(order["decision_date"])[:10]
            submission_date = str(order["created_at"])[:10]
            # Strict cutoff date: execution bar must be strictly AFTER both submission date and decision date
            cutoff_date = max(dec_date, submission_date)

            bars = market_bars_dict.get(sym, [])
            if not bars:
                deferred_list.append({
                    "order_id": order_id,
                    "symbol": sym,
                    "status": "PENDING",
                    "reason": f"No completed market bars available for {sym}."
                })
                continue

            # Sort completed daily bars chronologically ascending
            sorted_bars = sorted(bars, key=lambda x: str(x.get("date") or x.get("Date") or "")[:10])

            # Find earliest verified daily bar strictly AFTER cutoff_date with open > 0
            next_bar = None
            for b in sorted_bars:
                b_date = str(b.get("date") or b.get("Date") or "")[:10]
                b_open = float(b.get("open") or b.get("Open") or 0.0)
                if b_date and b_date > cutoff_date and b_open > 0:
                    next_bar = b
                    break

            if next_bar:
                open_price = float(next_bar.get("open") or next_bar.get("Open") or 0.0)
                open_date = str(next_bar.get("date") or next_bar.get("Date"))[:10]
                res = execute_order_at_open(order_id, open_price, open_date)
                if res.get("success"):
                    executed_list.append(res)
                else:
                    rejected_list.append(res)
            else:
                latest_bar_date = str(sorted_bars[-1].get("date") or sorted_bars[-1].get("Date") or "UNKNOWN")[:10] if sorted_bars else "N/A"
                deferred_list.append({
                    "order_id": order_id,
                    "symbol": sym,
                    "status": "PENDING",
                    "reason": (
                        f"Awaiting subsequent session Open: Order created on {submission_date} "
                        f"(decision cutoff {cutoff_date}). Latest verified completed bar in dataset is {latest_bar_date}."
                    )
                })

        # Update mark-to-market prices if any orders were executed
        if executed_list:
            current_prices = {}
            for s in TARGET_STOCKS:
                s_bars = market_bars_dict.get(s, [])
                if s_bars:
                    s_sorted = sorted(s_bars, key=lambda x: str(x.get("date") or x.get("Date") or "")[:10])
                    p = float(s_sorted[-1].get("close") or s_sorted[-1].get("Close") or 0.0)
                    if p > 0:
                        current_prices[s] = p
            if current_prices:
                update_holdings_prices(current_prices)

        return {
            "success": True,
            "processed_count": len(executed_list) + len(rejected_list),
            "executed_count": len(executed_list),
            "rejected_count": len(rejected_list),
            "deferred_count": len(deferred_list),
            "executed_orders": executed_list,
            "rejected_orders": rejected_list,
            "deferred_orders": deferred_list,
            "portfolio": get_portfolio(),
        }


def process_pending_orders_against_market(market_bars_dict: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper that delegates to process_pending_orders_service
    and returns a combined list of executed and rejected result dictionaries.
    """
    res = process_pending_orders_service(market_bars_dict)
    return res["executed_orders"] + res["rejected_orders"]


def get_orders(limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves order history."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT * FROM orders WHERE status = ? ORDER BY order_id DESC LIMIT ?;", (status_filter, limit))
        else:
            cursor.execute("SELECT * FROM orders ORDER BY order_id DESC LIMIT ?;", (limit,))
        rows = cursor.fetchall()

    orders = []
    for r in rows:
        probs = {}
        try:
            probs = json.loads(r["agent_probabilities"])
        except Exception:
            pass

        orders.append({
            "order_id": r["order_id"],
            "symbol": r["symbol"],
            "side": r["side"],
            "quantity": r["quantity"],
            "decision_date": r["decision_date"],
            "decision_time": r["decision_time"],
            "agent_recommendation": r["agent_recommendation"],
            "agent_probabilities": probs,
            "user_action": r["user_action"],
            "estimated_price": r["estimated_price"],
            "target_execution": r["target_execution"],
            "status": r["status"],
            "rejection_reason": r["rejection_reason"],
            "created_at": r["created_at"],
        })
    return orders


def get_transactions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves executed trade history."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions ORDER BY tx_id DESC LIMIT ?;", (limit,))
        rows = cursor.fetchall()

    txs = []
    for r in rows:
        txs.append({
            "tx_id": r["tx_id"],
            "order_id": r["order_id"],
            "symbol": r["symbol"],
            "side": r["side"],
            "quantity": r["quantity"],
            "execution_price": r["execution_price"],
            "execution_date": r["execution_date"],
            "execution_time": r["execution_time"],
            "transaction_fee": r["transaction_fee"],
            "total_amount": r["total_amount"],
            "created_at": r["created_at"],
        })
    return txs


def reset_portfolio(confirmed: bool = False, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> bool:
    """
    Resets the paper trading portfolio to initial virtual cash, zero holdings,
    and clears all paper orders and transactions.
    
    Requires confirmed=True to prevent accidental resets.
    NEVER deletes market data, training datasets, or checkpoints.
    """
    if not confirmed:
        raise ValueError("Portfolio reset requires explicit confirmation (confirmed=True).")

    init_db(initial_capital)
    now_iso = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions;")
        cursor.execute("DELETE FROM orders;")
        cursor.execute("""
        UPDATE portfolio_summary
        SET initial_capital = ?, cash = ?, total_value = ?, realized_pnl = 0.0, updated_at = ?
        WHERE id = 1;
        """, (initial_capital, initial_capital, initial_capital, now_iso))

        for symbol in TARGET_STOCKS:
            cursor.execute("""
            UPDATE holdings
            SET shares = 0, avg_purchase_price = 0.0, total_invested = 0.0,
                current_price = 0.0, current_value = 0.0, unrealized_pnl = 0.0, updated_at = ?
            WHERE symbol = ?;
            """, (now_iso, symbol))

        conn.commit()
    logger.info(f"Portfolio successfully reset to ₹{initial_capital:,.2f} initial capital.")
    return True

