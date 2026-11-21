"""
Lightweight Local HTTP API Server for NeuroTrade Paper Trading.

Binds to 127.0.0.1 and provides REST endpoints for:
- POST /api/order      (Submit BUY/SELL/HOLD order with trusted backend price)
- POST /api/cancel     (Cancel PENDING order and release escrowed cash/shares)
- GET  /api/portfolio  (Current portfolio metrics, available cash, reserved cash)
- GET  /api/orders     (Pending, executed, rejected, cancelled orders)
- GET  /api/history    (Executed transactions history)
- GET  /api/health     (Health check status)
- Static file serving  (Serves reports/live_trading_dashboard.html and UI assets)
"""

import os
import sys
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import backend.live_data.portfolio_db as pdb
from backend.live_data.market_cache import load_from_cache

logger = logging.getLogger(__name__)
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


def get_trusted_market_price_and_rec(symbol: str) -> tuple[float, str, Dict[str, float], str]:
    """
    Obtains trusted market data and model prediction from the backend cache
    to avoid trusting client-supplied execution prices.
    
    Returns:
        (price: float, recommendation: str, probabilities: dict, candle_date: str)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    default_rec = "HOLD"
    default_probs = {"HOLD": 0.5, "BUY": 0.25, "SELL": 0.25}

    try:
        cached = load_from_cache([symbol])
        if not cached:
            return 0.0, default_rec, default_probs, today_str

        price = 0.0
        # 1. Try real-time quote
        quotes = cached.get("quotes", {})
        if symbol in quotes and quotes[symbol].get("lastTradedPrice"):
            price = float(quotes[symbol]["lastTradedPrice"])

        # 2. Fall back to latest completed daily candle close
        historical = cached.get("historical", {})
        candle_date = today_str
        if symbol in historical and historical[symbol]:
            last_bar = historical[symbol][-1]
            if price <= 0:
                price = float(last_bar.get("close") or last_bar.get("Close") or 0.0)
            candle_date = str(last_bar.get("date") or last_bar.get("Date") or today_str)[:10]

        return price, default_rec, default_probs, candle_date
    except Exception as e:
        logger.warning(f"Failed to get trusted price for {symbol}: {e}")
        return 0.0, default_rec, default_probs, today_str


class NeuroTradeAPIHandler(SimpleHTTPRequestHandler):
    """HTTP request handler supporting both static UI assets and REST API endpoints."""

    def __init__(self, *args, **kwargs):
        # Serve static files relative to project root
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json_response(self, data: Any, status_code: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handles CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handles GET requests for static files and REST API endpoints."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Redirect root to dashboard
        if path in ["/", "/dashboard"]:
            self.send_response(302)
            self.send_header("Location", "/reports/live_trading_dashboard.html")
            self.end_headers()
            return

        # API: /api/health
        if path == "/api/health":
            self._send_json_response({
                "status": "online",
                "service": "NeuroTrade Paper Trading Local API",
                "timestamp": datetime.now().isoformat(),
                "database": str(pdb.DB_PATH)
            })
            return

        # API: /api/portfolio
        if path == "/api/portfolio":
            try:
                portfolio = pdb.get_portfolio()
                self._send_json_response(portfolio)
            except Exception as e:
                self._send_json_response({"error": str(e)}, status_code=500)
            return

        # API: /api/orders
        if path == "/api/orders":
            try:
                limit = int(query.get("limit", [50])[0])
                status = query.get("status", [None])[0]
                orders = pdb.get_orders(limit=limit, status_filter=status)
                self._send_json_response(orders)
            except Exception as e:
                self._send_json_response({"error": str(e)}, status_code=500)
            return

        # API: /api/history
        if path == "/api/history":
            try:
                limit = int(query.get("limit", [50])[0])
                txs = pdb.get_transactions(limit=limit)
                self._send_json_response(txs)
            except Exception as e:
                self._send_json_response({"error": str(e)}, status_code=500)
            return

        # API: /api/evaluation (Phase 6 Multi-Session Performance & Benchmarks)
        if path == "/api/evaluation":
            try:
                from backend.live_data.evaluation_engine import EvaluationEngine
                eval_engine = EvaluationEngine()
                metrics = eval_engine.calculate_evaluation_metrics()
                self._send_json_response(metrics)
            except Exception as e:
                self._send_json_response({"error": str(e)}, status_code=500)
            return

        # Otherwise serve static files (HTML, JS, CSS)
        super().do_GET()

    def do_POST(self):
        """Handles POST requests for paper trading orders, cancellations, and execution processing."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Read JSON body if present
        content_length = int(self.headers.get("Content-Length", 0))
        data = {}
        if content_length > 0:
            try:
                raw_body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(raw_body)
            except Exception as e:
                self._send_json_response({"success": False, "reason": f"Malformed JSON: {e}"}, status_code=400)
                return

        # POST /api/process-pending
        if path == "/api/process-pending":
            try:
                # Load verified market data from cache (rejecting untrusted browser prices)
                cached_data = load_from_cache(pdb.TARGET_STOCKS)
                historical = cached_data.get("historical", {}) if cached_data else {}
                result = pdb.process_pending_orders_service(historical)
                self._send_json_response(result, status_code=200)
            except Exception as e:
                logger.error(f"Error processing pending orders: {e}")
                self._send_json_response({"success": False, "reason": str(e)}, status_code=500)
            return

        # POST /api/order
        if path == "/api/order":
            if not data:
                self._send_json_response({"success": False, "reason": "Empty request body"}, status_code=400)
                return

            symbol = data.get("symbol")
            side = data.get("side") or data.get("action")
            quantity = data.get("quantity")

            # Validate Symbol
            if not symbol or str(symbol).upper() not in pdb.TARGET_STOCKS:
                res = pdb.submit_order(
                    symbol=str(symbol) if symbol else "",
                    side=str(side) if side else "HOLD",
                    quantity=int(quantity) if isinstance(quantity, int) else 0,
                    current_price=0.0,
                    agent_recommendation="USER_DIRECT",
                    agent_probabilities={},
                    decision_date=datetime.now().strftime("%Y-%m-%d")
                )
                self._send_json_response(res, status_code=400)
                return

            symbol = str(symbol).upper()

            # Validate Side
            if not side or str(side).upper() not in ["BUY", "SELL", "HOLD"]:
                res = pdb.submit_order(
                    symbol=symbol,
                    side=str(side) if side else "",
                    quantity=int(quantity) if isinstance(quantity, int) else 0,
                    current_price=0.0,
                    agent_recommendation="USER_DIRECT",
                    agent_probabilities={},
                    decision_date=datetime.now().strftime("%Y-%m-%d")
                )
                self._send_json_response(res, status_code=400)
                return

            side = str(side).upper()

            # Validate Quantity
            if side != "HOLD":
                if not isinstance(quantity, int) or quantity <= 0:
                    res = pdb.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity if isinstance(quantity, int) else -1,
                        current_price=0.0,
                        agent_recommendation="USER_DIRECT",
                        agent_probabilities={},
                        decision_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    self._send_json_response(res, status_code=400)
                    return
            else:
                quantity = 0

            # Task 7: Obtain trusted market price from backend ONLY (never accept user-supplied price)
            trusted_price, agent_rec, agent_probs, candle_date = get_trusted_market_price_and_rec(symbol)
            if trusted_price <= 0:
                self._send_json_response({
                    "success": False,
                    "status": "REJECTED",
                    "reason": f"Unable to verify trusted market price for {symbol} from backend. Client-supplied prices are not accepted."
                }, status_code=400)
                return

            submission_date = datetime.now().strftime("%Y-%m-%d")

            # Submit order to SQLite database
            try:
                res = pdb.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    current_price=trusted_price,
                    agent_recommendation=agent_rec,
                    agent_probabilities=agent_probs,
                    decision_date=submission_date
                )
                status_code = 200 if res.get("success") else 400
                self._send_json_response(res, status_code=status_code)
            except Exception as e:
                logger.error(f"Error submitting order: {e}")
                self._send_json_response({"success": False, "status": "ERROR", "reason": str(e)}, status_code=500)
            return

        # POST /api/cancel
        if path == "/api/cancel":
            if not data:
                self._send_json_response({"success": False, "reason": "Empty request body"}, status_code=400)
                return

            order_id = data.get("order_id")
            if not isinstance(order_id, int) or order_id <= 0:
                self._send_json_response({
                    "success": False,
                    "reason": f"Valid positive integer order_id required, got: {order_id}"
                }, status_code=400)
                return

            try:
                res = pdb.cancel_order(order_id)
                status_code = 200 if res.get("success") else 400
                self._send_json_response(res, status_code=status_code)
            except Exception as e:
                self._send_json_response({"success": False, "reason": str(e)}, status_code=500)
            return

        # Unknown endpoint
        self._send_json_response({"error": f"Endpoint not found: {path}"}, status_code=404)


def run_server(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST):
    """Runs the HTTP API server."""
    pdb.init_db()
    server_address = (host, port)
    httpd = HTTPServer(server_address, NeuroTradeAPIHandler)
    print("=" * 75)
    print(f"  NeuroTrade Live API & Dashboard Server running at:")
    print(f"  -> http://{host}:{port}/reports/live_trading_dashboard.html")
    print(f"  Database connected: {pdb.DB_PATH}")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroTrade Live API & Dashboard Server")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8000)")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind (default: 127.0.0.1)")
    args, unknown = parser.parse_known_args()
    run_server(port=args.port, host=args.host)
