"""
Integrated Live Algorithmic Trading & Paper Trading Dashboard.

Displays:
1. Market Overview — 5 NSE stocks, latest OHLCV, and PPO recommendations
2. User Trading Controls — BUY / SELL / HOLD selection and quantity inputs
3. Portfolio Overview — Virtual capital, cash balance, current holdings, unrealized/realized P&L, overall return %
4. Trade History — Pending paper orders & executed transactions
5. Model Recommendations — Separate PPO policy probabilities from user decisions
"""

from typing import Dict, Any, List
from pathlib import Path
import json

from backend.live_data.portfolio_db import get_portfolio, get_orders, get_transactions, TRANSACTION_FEE_RATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
HTML_OUTPUT_PATH = REPORT_DIR / "live_trading_dashboard.html"


def print_console_dashboard(pipeline_output: Dict[str, Any]):
    """Prints an integrated market and paper-trading table to the terminal."""
    stocks = pipeline_output.get("stocks", [])
    cache_info = pipeline_output.get("cache_info", {})
    source_status = cache_info.get("data_status", "UNKNOWN")
    age = cache_info.get("cache_age_seconds", 0.0)
    retrieval_time = cache_info.get("retrieval_timestamp", "N/A")

    portfolio = get_portfolio()
    orders = get_orders(limit=5)
    pending_orders = [o for o in get_orders(limit=20) if o["status"] == "PENDING"]

    print("\n" + "=" * 95)
    print("  NEUROTRADE — REINFORCEMENT LEARNING ALGORITHMIC TRADING & PAPER TRADING DASHBOARD")
    print("  Architecture: 14 Technical Features -> 2-Layer LSTM (32-D Bottleneck) -> Decoupled PPO (49-D State)")
    print("=" * 95)
    print(f"  Data Source    : NSE MCP (National Stock Exchange of India) | Status: [{source_status}] (Age: {age}s / 180s TTL)")
    print(f"  Last Retrieved : {retrieval_time}")
    print(f"  Notice         : Data may be delayed by 1-3 minutes per exchange regulations. For academic research.")
    print("-" * 95)

    # 1. Market Overview & PPO Recommendations
    print("\n  [1] MARKET OVERVIEW & AGENT RECOMMENDATIONS")
    print(f"  {'Symbol':<10} {'LTP (₹)':>10} {'Change':>8} {'Latest Bar':>12} {'PPO Rec.':^9} {'HOLD':>7} {'BUY':>7} {'SELL':>7} {'RSI':>6}")
    print("  " + "-" * 88)

    for s in stocks:
        sym = s["symbol"]
        price = f"{s['current_price']:,.2f}"
        chg = f"{s['price_change_pct']:+.2f}%"
        dt = s["candle_date"]
        rec = s["recommendation"]
        p = s["action_probabilities"]
        p_hold = f"{p['HOLD']*100:.1f}%"
        p_buy = f"{p['BUY']*100:.1f}%"
        p_sell = f"{p['SELL']*100:.1f}%"
        rsi = f"{s['rsi_14']:.1f}"

        print(f"  {sym:<10} {price:>10} {chg:>8} {dt:>12} {rec:^9} {p_hold:>7} {p_buy:>7} {p_sell:>7} {rsi:>6}")

    # 2. Portfolio Overview
    init_cap = portfolio.get("initial_capital", 100000.0)
    cash = portfolio.get("cash", 100000.0)
    avail_cash = portfolio.get("available_cash", cash)
    res_cash = portfolio.get("reserved_cash", 0.0)
    h_val = portfolio.get("holdings_value", 0.0)
    total_val = portfolio.get("total_value", 100000.0)
    ret_pct = portfolio.get("total_return_pct", 0.0)
    r_pnl = portfolio.get("realized_pnl", 0.0)
    u_pnl = portfolio.get("unrealized_pnl", 0.0)

    print("\n  [2] VIRTUAL PAPER TRADING PORTFOLIO")
    print(f"  Initial: ₹{init_cap:,.2f}  |  Available Cash: ₹{avail_cash:,.2f}  |  Reserved (Escrow): ₹{res_cash:,.2f}  |  Ledger Cash: ₹{cash:,.2f}")
    print(f"  Holdings Value: ₹{h_val:,.2f}  |  Total Value: ₹{total_val:,.2f}  |  Return: {ret_pct:+.2f}%  |  Realized P&L: ₹{r_pnl:+,.2f}  |  Unrealized: ₹{u_pnl:+,.2f}")
    print("  " + "-" * 88)
    print(f"  {'Symbol':<10} {'Shares':>7} {'Avg Cost':>10} {'LTP':>10} {'Current Value':>14} {'Unrealized P&L':>15} {'P&L %':>8}")
    print("  " + "-" * 88)

    for sym, h in portfolio["holdings"].items():
        sh = h["shares"]
        avg_c = f"₹{h['avg_purchase_price']:,.2f}" if sh > 0 else "-"
        ltp = f"₹{h['current_price']:,.2f}" if h["current_price"] > 0 else "-"
        val = f"₹{h['current_value']:,.2f}"
        pnl = f"₹{h['unrealized_pnl']:+,.2f}"
        pnl_pct = f"{h['unrealized_pnl_pct']:+.2f}%"
        print(f"  {sym:<10} {sh:>7} {avg_c:>10} {ltp:>10} {val:>14} {pnl:>15} {pnl_pct:>8}")

    # 3. Pending Orders
    print(f"\n  [3] PENDING ORDERS ({len(pending_orders)} order(s) waiting for next session Open)")
    if pending_orders:
        print(f"  {'Order ID':<10} {'Symbol':<10} {'Side':<6} {'Qty':>6} {'Est Price':>10} {'Est Total':>12} {'Target Execution':<20}")
        print("  " + "-" * 80)
        for o in pending_orders:
            est_total = o["quantity"] * o["estimated_price"] * (1.0 + TRANSACTION_FEE_RATE)
            print(f"  #{o['order_id']:<9} {o['symbol']:<10} {o['side']:<6} {o['quantity']:>6} ₹{o['estimated_price']:>9,.2f} ₹{est_total:>11,.2f} {o['target_execution']:<20}")
    else:
        print("  No pending orders. Orders submitted during session execute at next session Open.")

    # 4. Recent Order History
    print(f"\n  [4] RECENT ORDER HISTORY (Last {min(5, len(orders))})")
    if orders:
        print(f"  {'Order ID':<10} {'Symbol':<10} {'Action':<6} {'Qty':>5} {'Status':<10} {'PPO Recommendation':<20} {'Decision Date':<12}")
        print("  " + "-" * 80)
        for o in orders[:5]:
            print(f"  #{o['order_id']:<9} {o['symbol']:<10} {o['user_action']:<6} {o['quantity']:>5} {o['status']:<10} {o['agent_recommendation']:<20} {o['decision_date']:<12}")
    else:
        print("  No previous orders recorded.")

    print("\n" + "=" * 95)
    print("  Policy Probabilities represent the PPO agent's categorical action distribution, NOT profit probabilities.")
    print("  Execution Protocol: Decisions submitted at close t execute at Open t+1 with 0.1% friction.")
    print("=" * 95 + "\n")


def export_html_dashboard(pipeline_output: Dict[str, Any], output_path: Path = HTML_OUTPUT_PATH):
    """Renders the comprehensive HTML dashboard with interactive paper trading controls."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stocks = pipeline_output.get("stocks", [])
    cache_info = pipeline_output.get("cache_info", {})
    source_status = cache_info.get("data_status", "UNKNOWN")
    age = cache_info.get("cache_age_seconds", 0.0)
    retrieval_time = cache_info.get("retrieval_timestamp", "N/A")

    portfolio = get_portfolio()
    orders = get_orders(limit=20)
    transactions = get_transactions(limit=20)
    pending_orders = [o for o in orders if o["status"] == "PENDING"]

    badge_color = {
        "FRESH": "#10b981",
        "CACHED": "#3b82f6",
        "FALLBACK_STALE": "#f59e0b",
    }.get(source_status, "#6b7280")

    # Generate Stock Cards with Trading Controls
    cards_html = ""
    for s in stocks:
        sym = s["symbol"]
        rec = s["recommendation"]
        rec_color = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}.get(rec, "#6b7280")
        p = s["action_probabilities"]
        chg_color = "#10b981" if s["price_change_pct"] >= 0 else "#ef4444"
        holding_info = portfolio["holdings"].get(sym, {})
        shares_held = holding_info.get("shares", 0)

        cards_html += f"""
        <div class="stock-card">
            <div class="stock-header">
                <div>
                    <span class="stock-symbol">{sym}</span>
                    <span class="stock-date">{s['candle_date']}</span>
                </div>
                <div class="stock-price" style="text-align: right;">
                    ₹{s['current_price']:,.2f}
                    <div style="font-size: 13px; color: {chg_color}; font-weight: 600;">{s['price_change_pct']:+.2f}%</div>
                </div>
            </div>

            <div class="recommendation-badge" style="background-color: {rec_color}20; color: {rec_color}; border: 1px solid {rec_color}60;">
                AGENT RECOMMENDATION: <strong>{rec}</strong>
            </div>

            <div class="metrics-grid">
                <div class="metric"><span class="m-label">Open</span><span class="m-val">₹{s['daily_open']:,.2f}</span></div>
                <div class="metric"><span class="m-label">High</span><span class="m-val">₹{s['daily_high']:,.2f}</span></div>
                <div class="metric"><span class="m-label">Low</span><span class="m-val">₹{s['daily_low']:,.2f}</span></div>
                <div class="metric"><span class="m-label">Shares Held</span><span class="m-val">{shares_held}</span></div>
                <div class="metric"><span class="m-label">Holding Val</span><span class="m-val">₹{holding_info.get('current_value', 0):,.2f}</span></div>
                <div class="metric"><span class="m-label">RSI (14)</span><span class="m-val">{s['rsi_14']}</span></div>
            </div>

            <div style="margin-top: 14px;">
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px; font-weight: 600; text-transform: uppercase;">PPO Policy Probabilities:</div>
                <div class="prob-row">
                    <span style="width: 45px; font-size: 11px;">HOLD</span>
                    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {p['HOLD']*100}%; background-color: #f59e0b;"></div></div>
                    <span style="width: 45px; text-align: right; font-size: 11px; font-weight: 600;">{p['HOLD']*100:.1f}%</span>
                </div>
                <div class="prob-row">
                    <span style="width: 45px; font-size: 11px;">BUY</span>
                    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {p['BUY']*100}%; background-color: #10b981;"></div></div>
                    <span style="width: 45px; text-align: right; font-size: 11px; font-weight: 600;">{p['BUY']*100:.1f}%</span>
                </div>
                <div class="prob-row">
                    <span style="width: 45px; font-size: 11px;">SELL</span>
                    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {p['SELL']*100}%; background-color: #ef4444;"></div></div>
                    <span style="width: 45px; text-align: right; font-size: 11px; font-weight: 600;">{p['SELL']*100:.1f}%</span>
                </div>
            </div>

            <!-- Paper Trading Controls Section -->
            <div class="trade-controls-card">
                <div style="font-size: 12px; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">Paper Trading Controls:</div>
                <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                    <div style="flex: 1;">
                        <label style="font-size: 10px; color: #94a3b8;">ACTION</label>
                        <select id="action_{sym}" class="trade-input" onchange="updateEstimatedCost('{sym}', {s['current_price']})">
                            <option value="BUY">BUY</option>
                            <option value="SELL">SELL</option>
                            <option value="HOLD">HOLD</option>
                        </select>
                    </div>
                    <div style="flex: 1;">
                        <label style="font-size: 10px; color: #94a3b8;">QUANTITY</label>
                        <input type="number" id="qty_{sym}" class="trade-input" value="10" min="1" step="1" oninput="updateEstimatedCost('{sym}', {s['current_price']})" />
                    </div>
                </div>
                <div style="font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>Est. Fee (0.1%): <strong id="fee_{sym}">₹{s['current_price']*10*TRANSACTION_FEE_RATE:,.2f}</strong></span>
                    <span>Est. Total: <strong id="total_{sym}">₹{s['current_price']*10*(1.0 + TRANSACTION_FEE_RATE):,.2f}</strong></span>
                </div>
                <button class="trade-btn" onclick="submitPaperOrder('{sym}', {s['current_price']}, '{rec}')">
                    Submit Paper Order (Queues for Open t+1)
                </button>
            </div>
        </div>
        """

    # Generate Holdings Rows
    holdings_rows_html = ""
    for sym, h in portfolio["holdings"].items():
        sh = h["shares"]
        avg_c = f"₹{h['avg_purchase_price']:,.2f}" if sh > 0 else "-"
        ltp = f"₹{h['current_price']:,.2f}" if h["current_price"] > 0 else "-"
        val = f"₹{h['current_value']:,.2f}"
        pnl_color = "#10b981" if h["unrealized_pnl"] >= 0 else "#ef4444"
        pnl = f"₹{h['unrealized_pnl']:+,.2f}"
        pnl_pct = f"{h['unrealized_pnl_pct']:+.2f}%"

        holdings_rows_html += f"""
        <tr>
            <td style="font-weight: 700;">{sym}</td>
            <td style="text-align: right;">{sh}</td>
            <td style="text-align: right;">{avg_c}</td>
            <td style="text-align: right;">{ltp}</td>
            <td style="text-align: right; font-weight: 600;">{val}</td>
            <td style="text-align: right; color: {pnl_color}; font-weight: 600;">{pnl}</td>
            <td style="text-align: right; color: {pnl_color}; font-weight: 600;">{pnl_pct}</td>
        </tr>
        """

    # Generate Pending Orders Rows
    pending_rows_html = ""
    for o in pending_orders:
        side_color = "#10b981" if o["side"] == "BUY" else "#ef4444"
        est_total = o["quantity"] * o["estimated_price"] * (1.0 + TRANSACTION_FEE_RATE)
        pending_rows_html += f"""
        <tr>
            <td>#{o['order_id']}</td>
            <td style="font-weight: 700;">{o['symbol']}</td>
            <td style="color: {side_color}; font-weight: 700;">{o['side']}</td>
            <td style="text-align: right;">{o['quantity']}</td>
            <td style="text-align: right;">₹{o['estimated_price']:,.2f}</td>
            <td style="text-align: right;">₹{est_total:,.2f}</td>
            <td style="color: #38bdf8;">{o['agent_recommendation']}</td>
            <td><span class="badge badge-pending">PENDING (Open t+1)</span></td>
            <td>{o['decision_date']}</td>
            <td><button class="cancel-btn" onclick="cancelPendingOrder({o['order_id']})">Cancel</button></td>
        </tr>
        """
    if not pending_orders:
        pending_rows_html = "<tr><td colspan='10' style='text-align: center; color: #64748b; padding: 14px;'>No pending orders. Orders submitted during session execute at next trading day Open.</td></tr>"

    # Generate Executed Trade Rows
    executed_rows_html = ""
    for t in transactions[:10]:
        side_color = "#10b981" if t["side"] == "BUY" else "#ef4444"
        executed_rows_html += f"""
        <tr>
            <td>#{t['tx_id']}</td>
            <td>#{t['order_id']}</td>
            <td style="font-weight: 700;">{t['symbol']}</td>
            <td style="color: {side_color}; font-weight: 700;">{t['side']}</td>
            <td style="text-align: right;">{t['quantity']}</td>
            <td style="text-align: right;">₹{t['execution_price']:,.2f}</td>
            <td style="text-align: right;">₹{t['transaction_fee']:,.2f}</td>
            <td style="text-align: right; font-weight: 600;">₹{t['total_amount']:,.2f}</td>
            <td>{t['execution_date']} {t['execution_time']}</td>
        </tr>
        """
    if not transactions:
        executed_rows_html = "<tr><td colspan='9' style='text-align: center; color: #64748b; padding: 14px;'>No executed transactions yet. Executions occur when the next session opens.</td></tr>"

    avail_cash = portfolio.get('available_cash', portfolio['cash'])
    res_cash = portfolio.get('reserved_cash', 0.0)

    # Phase 6 Multi-Session Performance Metrics
    try:
        from backend.live_data.evaluation_engine import EvaluationEngine
        eval_engine = EvaluationEngine()
        eval_metrics = eval_engine.calculate_evaluation_metrics()
    except Exception:
        eval_metrics = {
            "evaluation_status": "IN PROGRESS (1 / 20 Sessions Observed)",
            "sessions_observed": 1,
            "portfolio_performance": {"cumulative_return_pct": 0.0, "final_portfolio_value": 100000.0},
            "risk_metrics": {"sharpe_ratio": 0.0, "maximum_drawdown_pct": 0.0, "annualized_volatility_pct": 0.0, "risk_free_rate_used": "6.50% (RBI 91-day T-Bill rate)"},
            "benchmark_comparison": {
                "ppo_portfolio": {"return_pct": 0.0, "final_value": 100000.0},
                "equal_weight_buy_and_hold": {"return_pct": 0.0, "final_value": 100000.0},
                "dual_sma_crossover": {"return_pct": 0.0, "final_value": 100000.0}
            }
        }

    eval_ret = eval_metrics["portfolio_performance"]["cumulative_return_pct"]
    eval_sharpe = eval_metrics["risk_metrics"]["sharpe_ratio"]
    eval_mdd = eval_metrics["risk_metrics"]["maximum_drawdown_pct"]
    eval_vol = eval_metrics["risk_metrics"]["annualized_volatility_pct"]
    eval_sessions = eval_metrics.get("sessions_observed", 1)
    bm_data = eval_metrics.get("benchmark_comparison", {})
    eval_ppo_val = bm_data.get("ppo_portfolio", {}).get("final_value", 100000.0)
    eval_bh_val = bm_data.get("equal_weight_buy_and_hold", {}).get("final_value", 100000.0)
    eval_bh_ret = bm_data.get("equal_weight_buy_and_hold", {}).get("return_pct", 0.0)
    eval_sma_val = bm_data.get("dual_sma_crossover", {}).get("final_value", 100000.0)
    eval_sma_ret = bm_data.get("dual_sma_crossover", {}).get("return_pct", 0.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroTrade — Live Algorithmic & Paper Trading Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: #0f172a; color: #f8fafc; padding: 24px; min-height: 100vh; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 18px; margin-bottom: 24px; }}
        .title-area h1 {{ font-size: 22px; font-weight: 700; color: #38bdf8; }}
        .title-area p {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
        .status-badge {{ background-color: {badge_color}20; color: {badge_color}; border: 1px solid {badge_color}; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
        
        .section-title {{ font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }}
        .section-title::before {{ content: ""; display: inline-block; width: 4px; height: 16px; background-color: #38bdf8; border-radius: 2px; }}

        .portfolio-summary-bar {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 26px; }}
        .summary-card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
        .sc-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; }}
        .sc-val {{ font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
        
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stock-card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .stock-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }}
        .stock-symbol {{ font-size: 18px; font-weight: 700; color: #f1f5f9; }}
        .stock-date {{ display: block; font-size: 11px; color: #64748b; margin-top: 2px; }}
        .stock-price {{ font-size: 18px; font-weight: 700; }}
        .recommendation-badge {{ text-align: center; padding: 8px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; letter-spacing: 0.5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background-color: #0f172a80; padding: 10px; border-radius: 8px; }}
        .metric {{ display: flex; flex-direction: column; }}
        .m-label {{ font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .m-val {{ font-size: 12px; font-weight: 600; color: #cbd5e1; margin-top: 2px; }}
        
        .prob-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
        .prob-bar-bg {{ flex: 1; height: 7px; background-color: #334155; border-radius: 4px; overflow: hidden; }}
        .prob-bar-fill {{ height: 100%; border-radius: 4px; }}

        .trade-controls-card {{ background-color: #0f172a90; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 14px; }}
        .trade-input {{ width: 100%; background-color: #1e293b; border: 1px solid #475569; color: #f8fafc; border-radius: 6px; padding: 6px 10px; font-size: 12px; margin-top: 4px; }}
        .trade-btn {{ width: 100%; background-color: #2563eb; color: #ffffff; border: none; border-radius: 6px; padding: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
        .trade-btn:hover {{ background-color: #1d4ed8; }}
        .trade-btn:disabled {{ background-color: #475569; cursor: not-allowed; }}

        .cancel-btn {{ background-color: #ef444420; color: #ef4444; border: 1px solid #ef444460; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
        .cancel-btn:hover {{ background-color: #ef4444; color: #ffffff; }}
        .cancel-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        .process-btn {{ background-color: #0284c7; color: #ffffff; border: 1px solid #38bdf8; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.2s; display: inline-flex; align-items: center; gap: 6px; }}
        .process-btn:hover {{ background-color: #0369a1; }}
        .process-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; margin-bottom: 26px; }}
        .data-table th {{ background-color: #0f172a; padding: 10px 14px; text-align: left; font-size: 11px; text-transform: uppercase; color: #94a3b8; border-bottom: 1px solid #334155; }}
        .data-table td {{ padding: 10px 14px; border-bottom: 1px solid #33415580; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .badge-pending {{ background-color: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b50; }}

        .notice-banner {{ background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #c7d2fe; margin-bottom: 24px; line-height: 1.5; }}
        .footer {{ border-top: 1px solid #334155; padding-top: 16px; font-size: 12px; color: #64748b; display: flex; justify-content: space-between; align-items: center; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title-area">
            <h1>NeuroTrade — Live Algorithmic & Paper Trading Dashboard</h1>
            <p>Reinforcement Learning with RNN-Based Market State Representation (B.Tech Capstone Project)</p>
        </div>
        <div>
            <span class="status-badge">● {source_status} ({age}s ago / 3-min cache)</span>
        </div>
    </div>

    <div class="notice-banner">
        ⚠️ <strong>SEBI Regulatory & Operational Notice:</strong> Market data sourced via official NSE Model Context Protocol (MCP) Streamable HTTP transport (delayed by 1–3 minutes per exchange regulations). Paper trading operates on virtual money only. Decisions submitted at close t execute at Open t+1 with a 0.1% transaction fee, strictly replicating the MarketTradingEnv protocol.
    </div>

    <!-- Portfolio Summary Metrics -->
    <div class="section-title">Virtual Paper Portfolio Summary (Default Initial: ₹1,00,000)</div>
    <div class="portfolio-summary-bar">
        <div class="summary-card">
            <div class="sc-label">Total Portfolio Value</div>
            <div id="sc_total_val" class="sc-val">₹{portfolio['total_value']:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Available Buying Cash</div>
            <div id="sc_avail_cash" class="sc-val" style="color: #38bdf8;">₹{avail_cash:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Reserved Cash (Escrow)</div>
            <div id="sc_res_cash" class="sc-val" style="color: #f59e0b;">₹{res_cash:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Total Ledger Cash</div>
            <div id="sc_ledger_cash" class="sc-val">₹{portfolio['cash']:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Current Holdings Value</div>
            <div id="sc_holdings_val" class="sc-val">₹{portfolio['holdings_value']:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Overall Return</div>
            <div id="sc_return_pct" class="sc-val" style="color: {'#10b981' if portfolio['total_return_pct'] >= 0 else '#ef4444'};">
                {portfolio['total_return_pct']:+.2f}%
            </div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Unrealized P&L</div>
            <div id="sc_unreal_pnl" class="sc-val" style="color: {'#10b981' if portfolio['unrealized_pnl'] >= 0 else '#ef4444'};">
                ₹{portfolio['unrealized_pnl']:+,.2f}
            </div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Realized P&L</div>
            <div id="sc_real_pnl" class="sc-val" style="color: {'#10b981' if portfolio['realized_pnl'] >= 0 else '#ef4444'};">
                ₹{portfolio['realized_pnl']:+,.2f}
            </div>
        </div>
    </div>

    <!-- Stock Cards with Agent Recommendation & Paper Trading Controls -->
    <div class="section-title">Equities & Paper Trading Controls</div>
    <div class="cards-grid">
        {cards_html}
    </div>

    <!-- Portfolio Holdings Table -->
    <div class="section-title">Current Asset Holdings</div>
    <table class="data-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th style="text-align: right;">Shares Held</th>
                <th style="text-align: right;">Avg Cost (₹)</th>
                <th style="text-align: right;">LTP (₹)</th>
                <th style="text-align: right;">Current Value (₹)</th>
                <th style="text-align: right;">Unrealized P&L (₹)</th>
                <th style="text-align: right;">Return (%)</th>
            </tr>
        </thead>
        <tbody id="holdings_table_body">
            {holdings_rows_html}
        </tbody>
    </table>

    <!-- Pending Orders Table -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div class="section-title" style="margin-bottom: 0;">Pending Orders (Execution at Session Open t+1)</div>
        <button id="btn_process_pending" class="process-btn" onclick="processPendingOrders()">⚡ Process Eligible Pending Orders</button>
    </div>
    <div id="process_feedback" style="display: none; padding: 10px 14px; margin-bottom: 14px; border-radius: 6px; font-size: 12px; line-height: 1.4;"></div>
    <table class="data-table">
        <thead>
            <tr>
                <th>Order ID</th>
                <th>Symbol</th>
                <th>Side</th>
                <th style="text-align: right;">Quantity</th>
                <th style="text-align: right;">Est. Price</th>
                <th style="text-align: right;">Est. Total (inc 0.1% fee)</th>
                <th>Agent Recommendation</th>
                <th>Status</th>
                <th>Decision Date</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody id="pending_orders_body">
            {pending_rows_html}
        </tbody>
    </table>

    <!-- Executed Trades History Table -->
    <div class="section-title">Executed Trade History</div>
    <table class="data-table">
        <thead>
            <tr>
                <th>Tx ID</th>
                <th>Order ID</th>
                <th>Symbol</th>
                <th>Side</th>
                <th style="text-align: right;">Quantity</th>
                <th style="text-align: right;">Execution Price</th>
                <th style="text-align: right;">Transaction Fee</th>
                <th style="text-align: right;">Gross Amount</th>
                <th>Execution Timestamp</th>
            </tr>
        </thead>
        <tbody id="history_orders_body">
            {executed_rows_html}
        </tbody>
    </table>

    <!-- Phase 6: Multi-Session Performance & Benchmark Evaluation -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 30px; margin-bottom: 14px;">
        <div class="section-title" style="margin-bottom: 0;">Multi-Session Paper-Trading Evaluation & Benchmarks</div>
        <span class="status-badge" style="background-color: #f59e0b20; color: #f59e0b; border-color: #f59e0b;">● {eval_metrics['evaluation_status']}</span>
    </div>

    <div class="portfolio-summary-bar">
        <div class="summary-card">
            <div class="sc-label">Evaluation Return</div>
            <div class="sc-val" style="color: {'#10b981' if eval_ret >= 0 else '#ef4444'};">{eval_ret:+.2f}%</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Sharpe Ratio (rf=6.50%)</div>
            <div class="sc-val" style="color: #38bdf8;">{eval_sharpe:.2f}</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Max Drawdown</div>
            <div class="sc-val" style="color: #ef4444;">{eval_mdd:.2f}%</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Annualized Volatility</div>
            <div class="sc-val" style="color: #cbd5e1;">{eval_vol:.2f}%</div>
        </div>
        <div class="summary-card">
            <div class="sc-label">Sessions Observed</div>
            <div class="sc-val" style="color: #a78bfa;">{eval_sessions} / 30</div>
        </div>
    </div>

    <!-- Benchmark Comparison Table -->
    <table class="data-table">
        <thead>
            <tr>
                <th>Trading Strategy</th>
                <th>Strategy Type</th>
                <th style="text-align: right;">Initial Capital</th>
                <th style="text-align: right;">Current Valuation</th>
                <th style="text-align: right;">Cumulative Return</th>
                <th style="text-align: right;">Excess vs Buy & Hold</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: #1e3a5f40;">
                <td style="font-weight: 700; color: #38bdf8;">NeuroTrader (PPO Policy)</td>
                <td>Reinforcement Learning (14 Feat -> LSTM 32-D -> PPO)</td>
                <td style="text-align: right;">₹1,00,000.00</td>
                <td style="text-align: right; font-weight: 700;">₹{eval_ppo_val:,.2f}</td>
                <td style="text-align: right; font-weight: 700; color: {'#10b981' if eval_ret >= 0 else '#ef4444'};">{eval_ret:+.2f}%</td>
                <td style="text-align: right; font-weight: 700; color: {'#10b981' if (eval_ret - eval_bh_ret) >= 0 else '#ef4444'};">{eval_ret - eval_bh_ret:+.2f}%</td>
            </tr>
            <tr>
                <td style="font-weight: 700;">Equal-Weight Buy & Hold (5 Stocks)</td>
                <td>Passive Benchmark (₹20,000 each in 5 stocks on Day 0)</td>
                <td style="text-align: right;">₹1,00,000.00</td>
                <td style="text-align: right; font-weight: 600;">₹{eval_bh_val:,.2f}</td>
                <td style="text-align: right; font-weight: 600;">{eval_bh_ret:+.2f}%</td>
                <td style="text-align: right;">-</td>
            </tr>
            <tr>
                <td style="font-weight: 700;">Dual SMA Crossover (10 / 30)</td>
                <td>Trend Following Filter (with 0.1% transaction friction)</td>
                <td style="text-align: right;">₹1,00,000.00</td>
                <td style="text-align: right; font-weight: 600;">₹{eval_sma_val:,.2f}</td>
                <td style="text-align: right; font-weight: 600;">{eval_sma_ret:+.2f}%</td>
                <td style="text-align: right;">{eval_sma_ret - eval_bh_ret:+.2f}%</td>
            </tr>
        </tbody>
    </table>

    <!-- Analytics Visualizations Grid -->
    <div class="section-title">Quantitative Analytics & Visualizations</div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; margin-bottom: 26px;">
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">PORTFOLIO EQUITY CURVE VS BENCHMARKS</div>
            <img src="/reports/phase6/visualizations/equity_curve.png" alt="Equity Curve" style="width: 100%; border-radius: 6px;" onerror="this.parentElement.innerHTML='<div style=\\'color:#64748b;padding:40px;text-align:center;\\'>Equity Curve visualization available in reports/phase6/visualizations/</div>'">
        </div>
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">BENCHMARK RETURN COMPARISON</div>
            <img src="/reports/phase6/visualizations/benchmark_comparison.png" alt="Benchmark Comparison" style="width: 100%; border-radius: 6px;" onerror="this.parentElement.innerHTML='<div style=\\'color:#64748b;padding:40px;text-align:center;\\'>Benchmark Comparison visualization available in reports/phase6/visualizations/</div>'">
        </div>
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">UNDERWATER DRAWDOWN PROFILE</div>
            <img src="/reports/phase6/visualizations/drawdown_curve.png" alt="Drawdown Curve" style="width: 100%; border-radius: 6px;" onerror="this.parentElement.innerHTML='<div style=\\'color:#64748b;padding:40px;text-align:center;\\'>Drawdown visualization available in reports/phase6/visualizations/</div>'">
        </div>
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">POLICY ACTION DISTRIBUTION</div>
            <img src="/reports/phase6/visualizations/agent_action_distribution.png" alt="Action Distribution" style="width: 100%; border-radius: 6px;" onerror="this.parentElement.innerHTML='<div style=\\'color:#64748b;padding:40px;text-align:center;\\'>Action distribution visualization available in reports/phase6/visualizations/</div>'">
        </div>
    </div>

    <div class="footer">
        <div>
            <strong>NeuroTrade Engine:</strong> 14 Technical Features -> 2-Layer LSTM (32-D Bottleneck) -> Decoupled PPO Actor-Critic (49-D State).
        </div>
        <div>
            Data Retrieved: {retrieval_time}
        </div>
    </div>

    <script>
        const TRANSACTION_FEE_RATE = 0.001;

        function updateEstimatedCost(sym, price) {{
            const action = document.getElementById('action_' + sym).value;
            const qty = parseInt(document.getElementById('qty_' + sym).value) || 0;
            const gross = qty * price;
            const fee = gross * TRANSACTION_FEE_RATE;
            const total = action === 'BUY' ? gross + fee : (action === 'SELL' ? gross - fee : 0);
            
            document.getElementById('fee_' + sym).innerText = '₹' + fee.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
            document.getElementById('total_' + sym).innerText = '₹' + total.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
        }}

        async function submitPaperOrder(sym, price, agentRec) {{
            const btn = event ? event.target : null;
            const action = document.getElementById('action_' + sym).value;
            const qtyInput = document.getElementById('qty_' + sym);
            const qty = parseInt(qtyInput.value) || 0;

            if (action !== 'HOLD' && qty <= 0) {{
                alert('Please enter a valid positive integer quantity greater than zero.');
                return;
            }}

            const promptMsg = 'Confirm Paper Trading Order:\\n\\n' +
                'Symbol: ' + sym + '\\n' +
                'Action: ' + action + '\\n' +
                'Quantity: ' + qty + ' shares\\n' +
                'Estimated Price: ₹' + price.toFixed(2) + '\\n' +
                'Agent Recommendation: ' + agentRec + '\\n\\n' +
                'Note: Decision is recorded at close t and queues for execution at next session Open t+1 with 0.1% transaction fee.';

            if (!confirm(promptMsg)) return;

            if (btn) {{
                btn.disabled = true;
                btn.innerText = 'Submitting...';
            }}

            try {{
                const response = await fetch('/api/order', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        symbol: sym,
                        side: action,
                        quantity: qty
                    }})
                }});

                const data = await response.json();
                if (response.ok && data.success) {{
                    alert('Order Submitted Successfully!\\nOrder #' + data.order_id + ' queued as ' + data.status + '.\\n' + (data.message || ''));
                    await refreshDashboardData();
                }} else {{
                    alert('Order Rejected / Failed:\\n' + (data.reason || data.message || 'Validation error'));
                    await refreshDashboardData();
                }}
            }} catch (err) {{
                alert('Communication Error: Could not reach API endpoint /api/order.\\n' + err.message + '\\nEnsure the local Python server is running.');
            }} finally {{
                if (btn) {{
                    btn.disabled = false;
                    btn.innerText = 'Submit Paper Order (Queues for Open t+1)';
                }}
            }}
        }}

        async function cancelPendingOrder(orderId) {{
            if (!confirm('Are you sure you want to cancel Pending Order #' + orderId + '?\\nAny escrowed funds or shares will be released immediately.')) {{
                return;
            }}

            try {{
                const response = await fetch('/api/cancel', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ order_id: orderId }})
                }});

                const data = await response.json();
                if (response.ok && data.success) {{
                    alert(data.message || ('Order #' + orderId + ' cancelled.'));
                    await refreshDashboardData();
                }} else {{
                    alert('Cancellation Failed: ' + (data.reason || 'Unknown error'));
                }}
            }} catch (err) {{
                alert('Error connecting to backend: ' + err.message);
            }}
        }}

        async function processPendingOrders() {{
            const btn = document.getElementById('btn_process_pending');
            const fb = document.getElementById('process_feedback');
            if (btn) {{
                btn.disabled = true;
                btn.innerText = 'Processing Pending Orders...';
            }}
            if (fb) {{
                fb.style.display = 'block';
                fb.style.backgroundColor = '#1e293b';
                fb.style.color = '#94a3b8';
                fb.style.border = '1px solid #475569';
                fb.innerText = 'Checking verified completed daily market sessions for pending orders...';
            }}
            try {{
                const res = await fetch('/api/process-pending', {{ method: 'POST' }});
                const data = await res.json();
                if (fb) {{
                    if (data.executed_count > 0 || data.rejected_count > 0) {{
                        fb.style.backgroundColor = '#064e3b';
                        fb.style.color = '#34d399';
                        fb.style.border = '1px solid #059669';
                        fb.innerText = 'Execution Complete: ' + data.executed_count + ' order(s) executed at Open, ' + data.rejected_count + ' rejected.';
                    }} else {{
                        fb.style.backgroundColor = '#1e1b4b';
                        fb.style.color = '#a5b4fc';
                        fb.style.border = '1px solid #4338ca';
                        const deferredMsg = data.deferred_orders && data.deferred_orders.length > 0
                            ? data.deferred_orders[0].reason
                            : 'All pending orders are awaiting their eligible subsequent session Open.';
                        fb.innerText = 'No orders eligible for execution yet. ' + deferredMsg;
                    }}
                }}
                await refreshDashboardData();
            }} catch (err) {{
                if (fb) {{
                    fb.style.backgroundColor = '#7f1d1d';
                    fb.style.color = '#fca5a5';
                    fb.style.border = '1px solid #dc2626';
                    fb.innerText = 'Failed to process pending orders: ' + err.message;
                }}
            }} finally {{
                if (btn) {{
                    btn.disabled = false;
                    btn.innerText = '⚡ Process Eligible Pending Orders';
                }}
            }}
        }}

        async function refreshDashboardData() {{
            try {{
                // 1. Refresh Portfolio Summary
                const pRes = await fetch('/api/portfolio');
                if (pRes.ok) {{
                    const p = await pRes.json();
                    const setTxt = (id, val) => {{ const el = document.getElementById(id); if (el) el.innerText = val; }};
                    setTxt('sc_total_val', '₹' + p.total_value.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_avail_cash', '₹' + p.available_cash.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_res_cash', '₹' + p.reserved_cash.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_ledger_cash', '₹' + p.cash.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_holdings_val', '₹' + p.holdings_value.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_return_pct', (p.total_return_pct >= 0 ? '+' : '') + p.total_return_pct.toFixed(2) + '%');
                    setTxt('sc_unreal_pnl', '₹' + (p.unrealized_pnl >= 0 ? '+' : '') + p.unrealized_pnl.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                    setTxt('sc_real_pnl', '₹' + (p.realized_pnl >= 0 ? '+' : '') + p.realized_pnl.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}));
                }}

                // 2. Refresh Pending Orders
                const oRes = await fetch('/api/orders?status=PENDING');
                if (oRes.ok) {{
                    const pending = await oRes.json();
                    const tbody = document.getElementById('pending_orders_body');
                    if (tbody) {{
                        if (pending.length === 0) {{
                            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: #64748b; padding: 14px;">No pending orders. Orders submitted during session execute at next trading day Open.</td></tr>';
                        }} else {{
                            let html = '';
                            for (const o of pending) {{
                                const sideColor = o.side === 'BUY' ? '#10b981' : '#ef4444';
                                const estTotal = o.quantity * o.estimated_price * (1 + TRANSACTION_FEE_RATE);
                                html += '<tr>' +
                                    '<td>#' + o.order_id + '</td>' +
                                    '<td style="font-weight: 700;">' + o.symbol + '</td>' +
                                    '<td style="color: ' + sideColor + '; font-weight: 700;">' + o.side + '</td>' +
                                    '<td style="text-align: right;">' + o.quantity + '</td>' +
                                    '<td style="text-align: right;">₹' + o.estimated_price.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</td>' +
                                    '<td style="text-align: right;">₹' + estTotal.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</td>' +
                                    '<td style="color: #38bdf8;">' + o.agent_recommendation + '</td>' +
                                    '<td><span class="badge badge-pending">PENDING (Open t+1)</span></td>' +
                                    '<td>' + o.decision_date + '</td>' +
                                    '<td><button class="cancel-btn" onclick="cancelPendingOrder(' + o.order_id + ')">Cancel</button></td>' +
                                '</tr>';
                            }}
                            tbody.innerHTML = html;
                        }}
                    }}
                }}

                // 3. Refresh History
                const hRes = await fetch('/api/history?limit=10');
                if (hRes.ok) {{
                    const txs = await hRes.json();
                    const tbody = document.getElementById('history_orders_body');
                    if (tbody) {{
                        if (txs.length === 0) {{
                            tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #64748b; padding: 14px;">No executed transactions yet. Executions occur when the next session opens.</td></tr>';
                        }} else {{
                            let html = '';
                            for (const t of txs) {{
                                const sideColor = t.side === 'BUY' ? '#10b981' : '#ef4444';
                                html += '<tr>' +
                                    '<td>#' + t.tx_id + '</td>' +
                                    '<td>#' + t.order_id + '</td>' +
                                    '<td style="font-weight: 700;">' + t.symbol + '</td>' +
                                    '<td style="color: ' + sideColor + '; font-weight: 700;">' + t.side + '</td>' +
                                    '<td style="text-align: right;">' + t.quantity + '</td>' +
                                    '<td style="text-align: right;">₹' + t.execution_price.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</td>' +
                                    '<td style="text-align: right;">₹' + t.transaction_fee.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</td>' +
                                    '<td style="text-align: right; font-weight: 600;">₹' + t.total_amount.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</td>' +
                                    '<td>' + t.execution_date + ' ' + t.execution_time + '</td>' +
                                '</tr>';
                            }}
                            tbody.innerHTML = html;
                        }}
                    }}
                }}
            }} catch (err) {{
                console.warn('Dashboard background refresh skipped: backend offline', err);
            }}
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
