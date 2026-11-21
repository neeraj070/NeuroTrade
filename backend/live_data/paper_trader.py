"""
Paper Trading Controller for NeuroTrade.

Coordinates:
- Order validation and submission
- Dynamic PPO portfolio context calculation
- Next-session execution timing (Decision at close t -> Execution at open t+1)
- Portfolio state and performance tracking
- CLI interface for manual paper trading actions
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.portfolio_db import (
    init_db, get_portfolio, update_holdings_prices, submit_order,
    execute_order_at_open, process_pending_orders_against_market,
    get_orders, get_transactions, cancel_order, reset_portfolio,
    TARGET_STOCKS, TRANSACTION_FEE_RATE
)


def get_dynamic_portfolio_context(
    symbol: str,
    current_price: float,
    portfolio: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Computes the 3-D dynamic portfolio context vector for a given stock,
    strictly replicating the mathematical semantics of MarketTradingEnv._get_observation:
    
    1. pos_ratio  = (shares * close_t) / V_t   in [0.0, 1.0]
    2. cash_ratio = cash / V_t                  in [0.0, 1.0]
    3. norm_val   = V_t / initial_capital       in [0.0, inf)
    
    When flat: returns np.array([0.0, 1.0, 1.0], dtype=np.float32).
    """
    if portfolio is None:
        portfolio = get_portfolio()

    curr_val = portfolio.get("total_value", 100000.0)
    init_cap = portfolio.get("initial_capital", 100000.0)
    cash = portfolio.get("cash", 100000.0)
    holdings = portfolio.get("holdings", {})

    shares = holdings.get(symbol, {}).get("shares", 0)
    equity_val = shares * current_price

    if curr_val <= 1e-8:
        pos_ratio = 0.0
        cash_ratio = 0.0
        norm_val = 0.0
    else:
        pos_ratio = float(equity_val / curr_val)
        cash_ratio = float(cash / curr_val)
        norm_val = float(curr_val / init_cap)

    # Clamp ratios to valid bounds
    pos_ratio = max(0.0, min(1.0, pos_ratio))
    cash_ratio = max(0.0, min(1.0, cash_ratio))

    return np.array([pos_ratio, cash_ratio, norm_val], dtype=np.float32)


def print_portfolio_summary():
    """Prints a clean summary of the virtual paper trading portfolio."""
    p = get_portfolio()
    init_cap = p["initial_capital"]
    cash = p["cash"]
    avail_cash = p.get("available_cash", cash)
    res_cash = p.get("reserved_cash", 0.0)
    h_val = p["holdings_value"]
    total = p["total_value"]
    r_pnl = p["realized_pnl"]
    u_pnl = p["unrealized_pnl"]
    ret_pct = p["total_return_pct"]

    print("\n" + "=" * 75)
    print("  VIRTUAL PAPER TRADING PORTFOLIO SUMMARY")
    print("=" * 75)
    print(f"  Initial Capital  : ₹{init_cap:>12,.2f}")
    print(f"  Total Ledger Cash: ₹{cash:>12,.2f}")
    print(f"  Available Cash   : ₹{avail_cash:>12,.2f}  ({avail_cash/total*100:.1f}% of total)")
    print(f"  Reserved (Escrow): ₹{res_cash:>12,.2f}")
    print(f"  Holdings Value   : ₹{h_val:>12,.2f}  ({h_val/total*100:.1f}% of total)")
    print(f"  Total Portfolio  : ₹{total:>12,.2f}")
    print(f"  Realized P&L     : ₹{r_pnl:>+12,.2f}")
    print(f"  Unrealized P&L   : ₹{u_pnl:>+12,.2f}")
    print(f"  Overall Return   : {ret_pct:>+12.2f}%")
    print("-" * 75)

    print(f"  {'Symbol':<10} {'Shares':>7} {'Avg Cost':>10} {'LTP':>10} {'Value':>12} {'P&L (₹)':>10} {'P&L %':>7}")
    print("  " + "-" * 70)

    for sym, h in p["holdings"].items():
        sh = h["shares"]
        avg_c = f"₹{h['avg_purchase_price']:,.2f}" if sh > 0 else "-"
        ltp = f"₹{h['current_price']:,.2f}" if h["current_price"] > 0 else "-"
        val = f"₹{h['current_value']:,.2f}"
        pnl = f"{h['unrealized_pnl']:+,.2f}"
        pnl_pct = f"{h['unrealized_pnl_pct']:+.2f}%"

        print(f"  {sym:<10} {sh:>7} {avg_c:>10} {ltp:>10} {val:>12} {pnl:>10} {pnl_pct:>7}")

    print("=" * 75 + "\n")


def print_order_history(limit: int = 15):
    """Prints recent orders and their execution status."""
    orders = get_orders(limit=limit)
    print("\n" + "=" * 90)
    print(f"  PAPER TRADING ORDERS (Recent {len(orders)})")
    print("=" * 90)
    print(f"  {'ID':<5} {'Symbol':<10} {'Side':<5} {'Qty':>5} {'Est Price':>10} {'Agent Rec':<9} {'User Act':<9} {'Status':<10} {'Date':<10}")
    print("  " + "-" * 85)

    for o in orders:
        oid = o["order_id"]
        sym = o["symbol"]
        side = o["side"]
        qty = o["quantity"]
        p = f"₹{o['estimated_price']:,.2f}"
        rec = o["agent_recommendation"]
        user_a = o["user_action"]
        status = o["status"]
        dt = o["decision_date"]

        print(f"  #{oid:<4} {sym:<10} {side:<5} {qty:>5} {p:>10} {rec:<9} {user_a:<9} {status:<10} {dt:<10}")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NeuroTrade Paper Trading CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: portfolio
    subparsers.add_parser("portfolio", help="View current portfolio and holdings")

    # Command: history
    hist_parser = subparsers.add_parser("history", help="View order and transaction history")
    hist_parser.add_argument("--limit", type=int, default=15, help="Number of records to show")

    # Command: order
    order_parser = subparsers.add_parser("order", help="Submit a paper trading order")
    order_parser.add_argument("--symbol", type=str, required=True, choices=TARGET_STOCKS, help="Stock symbol")
    order_parser.add_argument("--action", type=str, required=True, choices=["BUY", "SELL", "HOLD"], help="Trading action")
    order_parser.add_argument("--quantity", type=int, default=0, help="Number of shares (for BUY/SELL)")
    order_parser.add_argument("--price", type=float, default=0.0, help="Current market price (optional)")

    # Command: cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pending paper trading order")
    cancel_parser.add_argument("--order-id", type=int, required=True, help="Order ID to cancel")

    # Command: reset
    reset_parser = subparsers.add_parser("reset", help="Reset virtual portfolio to initial state")
    reset_parser.add_argument("--confirm", action="store_true", help="Explicit confirmation flag")

    args = parser.parse_args()

    if args.command == "portfolio":
        print_portfolio_summary()

    elif args.command == "history":
        print_order_history(limit=args.limit)

    elif args.command == "cancel":
        res = cancel_order(order_id=args.order_id)
        print("\nOrder Cancellation Result:")
        print(json.dumps(res, indent=2))
        print_portfolio_summary()

    elif args.command == "order":
        from backend.live_data.market_cache import load_from_cache

        price = args.price
        if price <= 0:
            cached = load_from_cache([args.symbol])
            if cached and args.symbol in cached.get("quotes", {}):
                price = float(cached["quotes"][args.symbol].get("lastTradedPrice", 0.0))
            if price <= 0 and cached and args.symbol in cached.get("historical", {}):
                bars = cached["historical"][args.symbol]
                if bars:
                    price = float(bars[-1].get("close", 0.0))

        if price <= 0:
            print(f"Error: Unable to determine price for {args.symbol}. Pass --price explicitly.")
            return

        res = submit_order(
            symbol=args.symbol,
            side=args.action,
            quantity=args.quantity,
            current_price=price,
            agent_recommendation="USER_DIRECT",
            agent_probabilities={"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34},
            decision_date=datetime.now().strftime("%Y-%m-%d"),
        )
        print("\nOrder Submission Result:")
        print(json.dumps(res, indent=2))
        print_portfolio_summary()

    elif args.command == "reset":
        if not args.confirm:
            print("ERROR: Portfolio reset requires explicit confirmation. Pass --confirm.")
            return
        reset_portfolio(confirmed=True)
        print("Portfolio reset complete.")
        print_portfolio_summary()

    else:
        print_portfolio_summary()


if __name__ == "__main__":
    main()
