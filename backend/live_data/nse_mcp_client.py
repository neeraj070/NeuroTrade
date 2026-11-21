"""
NSE MCP Client — Connection test for NeuroTrader project.

Connects to official NSE MCP servers using Streamable HTTP transport:
  - CM Market: https://mcp.nseindia.in/cmmkt/mcp  (near-live quotes)
  - Bhavcopy:  https://mcp.nseindia.in/bhavcopy/cm/mcp  (historical EOD data)

No API key or authentication required.
"""

import asyncio
import json
import sys
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_LIVE_DIR = PROJECT_ROOT / "data" / "live"
DATA_LIVE_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

CM_MARKET_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"

# ---------------------------------------------------------------------------
# Results collector
# ---------------------------------------------------------------------------

results = {
    "timestamp": datetime.now().isoformat(),
    "cm_market": {
        "url": CM_MARKET_URL,
        "connected": False,
        "tools": [],
        "quotes": {},
        "errors": [],
    },
    "bhavcopy": {
        "url": BHAVCOPY_URL,
        "connected": False,
        "tools": [],
        "historical": {},
        "errors": [],
    },
}


# ---------------------------------------------------------------------------
# Helper: safe tool call
# ---------------------------------------------------------------------------

async def safe_call_tool(client, tool_name, arguments, label=""):
    """Call a tool and return (success, result_content_list)."""
    try:
        result = await client.call_tool(tool_name, arguments)
        return True, result.content
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Phase 1: CM Market (live quotes)
# ---------------------------------------------------------------------------

async def test_cm_market():
    """Connect to CM Market endpoint, discover tools, fetch quotes."""
    from mcp import Client

    print("=" * 70)
    print("PHASE 1: CM MARKET — Near-live quotes")
    print(f"  Endpoint: {CM_MARKET_URL}")
    print("=" * 70)

    try:
        async with Client(CM_MARKET_URL) as client:
            results["cm_market"]["connected"] = True
            print("  ✓ Connected successfully\n")

            # --- Tool discovery ---
            tools_resp = await client.list_tools()
            tool_list = []
            for t in tools_resp.tools:
                info = {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                tool_list.append(info)
                print(f"  Tool: {t.name}")
                print(f"        {t.description}")
            results["cm_market"]["tools"] = tool_list
            print(f"\n  Total tools discovered: {len(tool_list)}\n")

            # --- Try fetching individual stock quotes ---
            # Look for a likely quote tool
            quote_tool = None
            for t in tool_list:
                name_lower = t["name"].lower()
                if "quote" in name_lower or "stock" in name_lower or "equity" in name_lower:
                    quote_tool = t["name"]
                    break

            if quote_tool is None:
                # Try the first tool that seems relevant, or just try common names
                candidate_names = ["get_stock_quote", "get_equity_quote", "stock_quote",
                                   "get_quote", "equity_quote", "getStockQuote"]
                for cn in candidate_names:
                    if any(t["name"] == cn for t in tool_list):
                        quote_tool = cn
                        break

            if quote_tool:
                print(f"  Using quote tool: {quote_tool}\n")
                for symbol in STOCKS:
                    print(f"  Fetching quote for {symbol}...")
                    # Try different parameter names
                    for param_set in [
                        {"symbol": symbol},
                        {"stock": symbol},
                        {"ticker": symbol},
                        {"name": symbol},
                    ]:
                        ok, content = await safe_call_tool(
                            client, quote_tool, param_set, f"{symbol} quote"
                        )
                        if ok:
                            # Parse content
                            parsed = []
                            for item in content:
                                if hasattr(item, "text"):
                                    parsed.append(item.text)
                                else:
                                    parsed.append(str(item))
                            results["cm_market"]["quotes"][symbol] = parsed
                            print(f"    ✓ Got response ({len(parsed)} items)")
                            break
                    else:
                        results["cm_market"]["quotes"][symbol] = "No response with any parameter set"
                        print(f"    ✗ No response")
            else:
                # If no obvious quote tool, try calling every tool with a stock symbol
                print("  No obvious quote tool found. Trying all tools with RELIANCE...")
                for t in tool_list[:5]:  # limit to first 5
                    for param_set in [{"symbol": "RELIANCE"}, {"stock": "RELIANCE"}]:
                        ok, content = await safe_call_tool(
                            client, t["name"], param_set, t["name"]
                        )
                        if ok:
                            parsed = []
                            for item in content:
                                if hasattr(item, "text"):
                                    parsed.append(item.text)
                                else:
                                    parsed.append(str(item))
                            results["cm_market"]["quotes"][f"RELIANCE_via_{t['name']}"] = parsed
                            print(f"    ✓ {t['name']} returned data")
                            break

    except Exception as e:
        results["cm_market"]["errors"].append(f"Connection error: {traceback.format_exc()}")
        print(f"  ✗ Connection FAILED: {e}")


# ---------------------------------------------------------------------------
# Phase 2: Bhavcopy (historical EOD data)
# ---------------------------------------------------------------------------

async def test_bhavcopy():
    """Connect to Bhavcopy endpoint, discover tools, fetch historical data."""
    from mcp import Client

    print("\n" + "=" * 70)
    print("PHASE 2: BHAVCOPY — Historical EOD data")
    print(f"  Endpoint: {BHAVCOPY_URL}")
    print("=" * 70)

    try:
        async with Client(BHAVCOPY_URL) as client:
            results["bhavcopy"]["connected"] = True
            print("  ✓ Connected successfully\n")

            # --- Tool discovery ---
            tools_resp = await client.list_tools()
            tool_list = []
            for t in tools_resp.tools:
                info = {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                tool_list.append(info)
                print(f"  Tool: {t.name}")
                print(f"        {t.description}")
            results["bhavcopy"]["tools"] = tool_list
            print(f"\n  Total tools discovered: {len(tool_list)}\n")

            # --- Try fetching historical price data ---
            history_tool = None
            for t in tool_list:
                name_lower = t["name"].lower()
                if "price" in name_lower and "history" in name_lower:
                    history_tool = t["name"]
                    break
                if "history" in name_lower or "historical" in name_lower:
                    history_tool = t["name"]
                    break
                if "bhavcopy" in name_lower:
                    history_tool = t["name"]
                    break

            if history_tool is None:
                candidate_names = ["get_price_history", "price_history",
                                   "get_historical_data", "historical_data",
                                   "get_bhavcopy"]
                for cn in candidate_names:
                    if any(t["name"] == cn for t in tool_list):
                        history_tool = cn
                        break

            # Determine date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            if history_tool:
                print(f"  Using history tool: {history_tool}")
                print(f"  Date range: {start_date} to {end_date}\n")
                for symbol in STOCKS:
                    print(f"  Fetching history for {symbol}...")
                    for param_set in [
                        {"symbol": symbol, "from_date": start_date, "to_date": end_date},
                        {"symbol": symbol, "start_date": start_date, "end_date": end_date},
                        {"symbol": symbol, "from": start_date, "to": end_date},
                        {"symbol": symbol},
                    ]:
                        ok, content = await safe_call_tool(
                            client, history_tool, param_set, f"{symbol} history"
                        )
                        if ok:
                            parsed = []
                            for item in content:
                                if hasattr(item, "text"):
                                    parsed.append(item.text)
                                else:
                                    parsed.append(str(item))
                            results["bhavcopy"]["historical"][symbol] = parsed
                            print(f"    ✓ Got response ({len(parsed)} items)")
                            break
                    else:
                        results["bhavcopy"]["historical"][symbol] = "No response with any parameter set"
                        print(f"    ✗ No response")
            else:
                print("  No obvious history tool found. Trying all tools with RELIANCE...")
                for t in tool_list[:5]:
                    for param_set in [
                        {"symbol": "RELIANCE"},
                        {"symbol": "RELIANCE", "from_date": start_date, "to_date": end_date},
                    ]:
                        ok, content = await safe_call_tool(
                            client, t["name"], param_set, t["name"]
                        )
                        if ok:
                            parsed = []
                            for item in content:
                                if hasattr(item, "text"):
                                    parsed.append(item.text)
                                else:
                                    parsed.append(str(item))
                            results["bhavcopy"]["historical"][f"RELIANCE_via_{t['name']}"] = parsed
                            print(f"    ✓ {t['name']} returned data")
                            break

    except Exception as e:
        results["bhavcopy"]["errors"].append(f"Connection error: {traceback.format_exc()}")
        print(f"  ✗ Connection FAILED: {e}")


# ---------------------------------------------------------------------------
# Phase 3: Save results
# ---------------------------------------------------------------------------

def save_results():
    """Save all results to data/live/."""
    outpath = DATA_LIVE_DIR / "nse_mcp_test_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {outpath}")

    # Save individual stock CSVs if we got parseable historical data
    for symbol, data_list in results["bhavcopy"]["historical"].items():
        if isinstance(data_list, list) and len(data_list) > 0:
            csv_path = DATA_LIVE_DIR / f"{symbol}_live.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                # Write raw text content — we'll parse format in the report
                for item in data_list:
                    f.write(str(item) + "\n")
            print(f"  Saved raw data: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("\n" + "=" * 70)
    print("NeuroTrader — NSE MCP Connection Test")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Stocks: " + ", ".join(STOCKS))
    print("=" * 70 + "\n")

    await test_cm_market()
    await test_bhavcopy()
    save_results()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print(f"  CM Market connected: {results['cm_market']['connected']}")
    print(f"  CM Market tools: {len(results['cm_market']['tools'])}")
    print(f"  CM Market quotes retrieved: {len(results['cm_market']['quotes'])}")
    print(f"  Bhavcopy connected: {results['bhavcopy']['connected']}")
    print(f"  Bhavcopy tools: {len(results['bhavcopy']['tools'])}")
    print(f"  Bhavcopy historical retrieved: {len(results['bhavcopy']['historical'])}")
    if results["cm_market"]["errors"]:
        print(f"  CM Market errors: {len(results['cm_market']['errors'])}")
    if results["bhavcopy"]["errors"]:
        print(f"  Bhavcopy errors: {len(results['bhavcopy']['errors'])}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
