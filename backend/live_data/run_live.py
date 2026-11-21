"""
Master Live Execution & Paper Trading Pipeline for NeuroTrade.

Workflow:
1. Fetch latest market data from NSE MCP (enforcing 3-minute local cache)
2. Process pending orders if a new market session bar has opened
3. Mark portfolio holdings to market at latest prices
4. Generate dynamic 3-D portfolio context for each stock from actual virtual holdings
5. Run LSTM/PPO inference with dynamic 49-D state vector
6. Display integrated console and HTML dashboard
7. Provide interactive/CLI paper-trading controls
"""

import sys
import os
import argparse
import asyncio
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_data.data_fetcher import fetch_all_market_data, TARGET_STOCKS
from backend.live_data.inference_engine import LiveInferenceEngine
from backend.live_data.dashboard import print_console_dashboard, export_html_dashboard, HTML_OUTPUT_PATH
from backend.live_data.portfolio_db import (
    init_db, get_portfolio, update_holdings_prices,
    process_pending_orders_against_market, submit_order, reset_portfolio
)
from backend.live_data.paper_trader import get_dynamic_portfolio_context, print_portfolio_summary


async def run_pipeline(
    force_refresh: bool = False,
    trade_symbol: str = None,
    trade_action: str = None,
    trade_quantity: int = 0,
    do_reset: bool = False
):
    print("=" * 80)
    print("  NEUROTRADE — LIVE MARKET DATA, PPO INFERENCE & PAPER TRADING")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize Portfolio Database
    init_db()

    # Handle Portfolio Reset if requested
    if do_reset:
        reset_portfolio(confirmed=True)
        print("  -> Portfolio reset to initial ₹1,00,000 cash. All holdings cleared.")

    # Step 1: Fetch or load market data (with 3-minute caching)
    print("\n[Step 1/5] Retrieving market data (checking 3-minute cache)...")
    try:
        market_data = await fetch_all_market_data(force_refresh=force_refresh)
        status = market_data["data_status"]
        age = market_data.get("cache_age_seconds", 0.0)
        print(f"  -> Data Status: [{status}] (Age: {age}s / 180s TTL)")
        print(f"  -> Source Info: {market_data.get('message', '')}")
    except Exception as e:
        print(f"  [ERROR] Data retrieval failed: {e}")
        return

    historical = market_data.get("historical", {})
    quotes = market_data.get("quotes", {})

    # Step 2: Check & process pending orders against new session bars
    print("\n[Step 2/5] Processing pending orders against market data...")
    executed_txs = process_pending_orders_against_market(historical)
    if executed_txs:
        for tx in executed_txs:
            if tx.get("success"):
                print(f"  -> EXECUTED Order #{tx['order_id']}: {tx['side']} {tx['quantity']} {tx['symbol']} @ ₹{tx['execution_price']:,.2f} on {tx['execution_date']} (Fee: ₹{tx['fee']:.2f})")
            else:
                print(f"  -> REJECTED Order #{tx.get('order_id')}: {tx.get('reason')}")
    else:
        print("  -> No pending orders eligible for execution (waiting for next session Open).")

    # Step 3: Mark portfolio holdings to market
    current_prices = {}
    for s in TARGET_STOCKS:
        p = 0.0
        if s in quotes and quotes[s].get("lastTradedPrice"):
            p = float(quotes[s]["lastTradedPrice"])
        elif s in historical and historical[s]:
            p = float(historical[s][-1].get("close", 0.0))
        current_prices[s] = p

    update_holdings_prices(current_prices)
    portfolio = get_portfolio()

    # Step 4: Run inference with DYNAMIC portfolio context
    print("\n[Step 3/5] Running LSTM & PPO inference with dynamic portfolio context...")
    try:
        engine = LiveInferenceEngine()
    except Exception as e:
        print(f"  [ERROR] Failed to load model artifacts: {e}")
        return

    predictions = []
    pred_by_sym = {}

    for symbol in TARGET_STOCKS:
        bars = historical.get(symbol, [])
        quote = quotes.get(symbol, {})
        price = current_prices.get(symbol, 0.0)

        if not bars:
            continue

        # Compute dynamic 3-D portfolio context from actual holdings
        dyn_context = get_dynamic_portfolio_context(symbol, price, portfolio)

        try:
            pred = engine.predict_stock(symbol, bars, quote, portfolio_context=dyn_context)
            predictions.append(pred)
            pred_by_sym[symbol] = pred
            rec = pred["recommendation"]
            p = pred["action_probabilities"]
            pos_pct = dyn_context[0] * 100
            print(f"  -> {symbol:<10} | Price: ₹{pred['current_price']:>9,.2f} | Context Pos: {pos_pct:>5.1f}% | Rec: {rec:<4} | HOLD: {p['HOLD']*100:>5.1f}% | BUY: {p['BUY']*100:>5.1f}% | SELL: {p['SELL']*100:>5.1f}%")
        except Exception as e:
            print(f"  [ERROR] Inference failed for {symbol}: {e}")

    # Handle manual trade submission if requested via CLI
    if trade_symbol and trade_action:
        print(f"\n[Trade Submission] User Action: {trade_action} {trade_quantity} {trade_symbol}")
        current_p = current_prices.get(trade_symbol, 0.0)
        agent_p = pred_by_sym.get(trade_symbol, {})
        agent_rec = agent_p.get("recommendation", "HOLD")
        agent_probs = agent_p.get("action_probabilities", {"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34})
        decision_dt = agent_p.get("candle_date", datetime.now().strftime("%Y-%m-%d"))

        res = submit_order(
            symbol=trade_symbol,
            side=trade_action,
            quantity=trade_quantity,
            current_price=current_p,
            agent_recommendation=agent_rec,
            agent_probabilities=agent_probs,
            decision_date=decision_dt
        )
        if res.get("success"):
            print(f"  -> Order #{res.get('order_id')} Recorded: {res.get('message')}")
        else:
            print(f"  -> Order REJECTED: {res.get('reason')}")

    # Step 5: Render Dashboards
    print("\n[Step 4/5] Rendering Dashboards...")
    pipeline_output = {
        "stocks": predictions,
        "cache_info": {
            "data_status": market_data.get("data_status"),
            "cache_age_seconds": market_data.get("cache_age_seconds", 0.0),
            "retrieval_timestamp": market_data.get("retrieval_timestamp"),
            "market_timestamp": market_data.get("market_timestamp"),
        }
    }

    # Console output
    print_console_dashboard(pipeline_output)

    # HTML output
    html_file = export_html_dashboard(pipeline_output)
    print(f"  -> Standalone HTML Dashboard updated: {html_file}")
    print(f"     Open in browser: file:///{str(html_file).replace(chr(92), '/')}")


def main():
    parser = argparse.ArgumentParser(description="NeuroTrade Live Inference & Paper Trading Pipeline")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the 3-minute local cache and fetch fresh data from NSE MCP."
    )
    parser.add_argument(
        "--buy",
        nargs=2,
        metavar=("SYMBOL", "QTY"),
        help="Submit paper BUY order: --buy RELIANCE 10"
    )
    parser.add_argument(
        "--sell",
        nargs=2,
        metavar=("SYMBOL", "QTY"),
        help="Submit paper SELL order: --sell RELIANCE 10"
    )
    parser.add_argument(
        "--hold",
        metavar="SYMBOL",
        help="Record paper HOLD decision: --hold RELIANCE"
    )
    parser.add_argument(
        "--reset-portfolio",
        action="store_true",
        help="Reset virtual portfolio to initial ₹1,00,000 cash balance."
    )

    args = parser.parse_args()

    trade_symbol = None
    trade_action = None
    trade_qty = 0

    if args.buy:
        trade_symbol = args.buy[0].upper()
        trade_action = "BUY"
        trade_qty = int(args.buy[1])
    elif args.sell:
        trade_symbol = args.sell[0].upper()
        trade_action = "SELL"
        trade_qty = int(args.sell[1])
    elif args.hold:
        trade_symbol = args.hold.upper()
        trade_action = "HOLD"
        trade_qty = 0

    asyncio.run(run_pipeline(
        force_refresh=args.force_refresh,
        trade_symbol=trade_symbol,
        trade_action=trade_action,
        trade_quantity=trade_qty,
        do_reset=args.reset_portfolio
    ))


if __name__ == "__main__":
    main()
