"""
NSE MCP Client — Retry with corrected tool names and parameters.

Findings from run 1:
  - Both servers connected successfully.
  - 15 CM Market tools, 21 Bhavcopy tools discovered.
  - All data calls returned Java NullPointerException (wrong tool/params).

This retry uses:
  - cm_get_stock_quote (symbol=X) for live quotes
  - get_stock_history (symbol=X, months=1) for historical OHLCV
  - get_bulk_quote (symbols=[...]) as fallback
"""

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_LIVE_DIR = PROJECT_ROOT / "data" / "live"

STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

CM_MARKET_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"

results = {
    "timestamp": datetime.now().isoformat(),
    "cm_market": {"connected": False, "quotes": {}, "errors": []},
    "bhavcopy": {"connected": False, "historical": {}, "errors": []},
}

def extract_text(content_list):
    """Extract text from MCP response content items."""
    texts = []
    for item in content_list:
        if hasattr(item, "text"):
            texts.append(item.text)
        else:
            texts.append(str(item))
    return texts


async def test_cm_quotes():
    """Test CM Market with cm_get_stock_quote and fallback to cm_get_equity_stocks."""
    from mcp import Client

    print("=" * 70)
    print("PHASE 1: CM MARKET — Stock Quotes")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Note: Market hours 09:15-15:30 IST. Outside hours = last available data.")
    print("=" * 70)

    try:
        async with Client(CM_MARKET_URL) as client:
            results["cm_market"]["connected"] = True
            print("  ✓ Connected\n")

            # --- Attempt 1: cm_get_stock_quote (symbol=X) ---
            print("  Strategy 1: cm_get_stock_quote")
            for symbol in STOCKS:
                try:
                    resp = await client.call_tool("cm_get_stock_quote", {"symbol": symbol})
                    text = extract_text(resp.content)
                    results["cm_market"]["quotes"][symbol] = {"tool": "cm_get_stock_quote", "data": text}
                    # Check if it's an error
                    joined = " ".join(text)
                    if "Exception" in joined or "Error" in joined or "null" in joined.lower():
                        print(f"    {symbol}: Server error — {joined[:120]}")
                    else:
                        print(f"    {symbol}: ✓ OK ({len(joined)} chars)")
                except Exception as e:
                    results["cm_market"]["quotes"][symbol] = {"tool": "cm_get_stock_quote", "error": str(e)}
                    print(f"    {symbol}: ✗ {e}")

            # --- Attempt 2: get bulk quote if individual failed ---
            first_result = results["cm_market"]["quotes"].get("RELIANCE", {})
            first_data = first_result.get("data", [""])[0] if isinstance(first_result, dict) else ""
            if "Exception" in str(first_data):
                print("\n  Strategy 2: cm_get_equity_stocks (symbolFilter)")
                for symbol in STOCKS:
                    try:
                        resp = await client.call_tool("cm_get_equity_stocks", {"symbolFilter": symbol})
                        text = extract_text(resp.content)
                        joined = " ".join(text)
                        if "Exception" not in joined:
                            results["cm_market"]["quotes"][f"{symbol}_equity"] = {"tool": "cm_get_equity_stocks", "data": text}
                            print(f"    {symbol}: ✓ OK ({len(joined)} chars)")
                        else:
                            print(f"    {symbol}: Server error")
                    except Exception as e:
                        print(f"    {symbol}: ✗ {e}")

            # --- Attempt 3: cm_get_data_status to check if data is available ---
            print("\n  Checking data freshness...")
            try:
                resp = await client.call_tool("cm_get_data_status", {})
                text = extract_text(resp.content)
                results["cm_market"]["data_status"] = text
                for t in text:
                    print(f"    {t[:200]}")
            except Exception as e:
                print(f"    Data status check failed: {e}")

    except Exception as e:
        results["cm_market"]["errors"].append(traceback.format_exc())
        print(f"  ✗ Connection FAILED: {e}")


async def test_bhavcopy_history():
    """Test Bhavcopy with get_stock_history (months param) and fallbacks."""
    from mcp import Client

    print("\n" + "=" * 70)
    print("PHASE 2: BHAVCOPY — Historical OHLCV")
    print("=" * 70)

    try:
        async with Client(BHAVCOPY_URL) as client:
            results["bhavcopy"]["connected"] = True
            print("  ✓ Connected\n")

            # --- Attempt 1: get_stock_history with months=1 ---
            print("  Strategy 1: get_stock_history (symbol, months=1)")
            for symbol in STOCKS:
                try:
                    resp = await client.call_tool("get_stock_history", {"symbol": symbol, "months": 1})
                    text = extract_text(resp.content)
                    joined = " ".join(text)
                    if "Exception" in joined or "Error" in joined:
                        print(f"    {symbol}: Server error — {joined[:120]}")
                        results["bhavcopy"]["historical"][symbol] = {"tool": "get_stock_history_m1", "error": joined[:200]}
                    else:
                        results["bhavcopy"]["historical"][symbol] = {"tool": "get_stock_history", "data": text}
                        print(f"    {symbol}: ✓ OK ({len(joined)} chars)")
                except Exception as e:
                    results["bhavcopy"]["historical"][symbol] = {"tool": "get_stock_history", "error": str(e)}
                    print(f"    {symbol}: ✗ {e}")

            # --- Attempt 2: If months didn't work, try without months ---
            first_result = results["bhavcopy"]["historical"].get("RELIANCE", {})
            if "error" in first_result:
                print("\n  Strategy 2: get_stock_history (symbol only, no months)")
                try:
                    resp = await client.call_tool("get_stock_history", {"symbol": "RELIANCE"})
                    text = extract_text(resp.content)
                    joined = " ".join(text)
                    if "Exception" not in joined:
                        results["bhavcopy"]["historical"]["RELIANCE_nomonths"] = {"tool": "get_stock_history_noargs", "data": text}
                        print(f"    RELIANCE: ✓ OK ({len(joined)} chars)")
                    else:
                        print(f"    RELIANCE: Server error — {joined[:120]}")
                except Exception as e:
                    print(f"    RELIANCE: ✗ {e}")

            # --- Attempt 3: get_bulk_quote ---
            if "error" in first_result:
                print("\n  Strategy 3: get_bulk_quote (symbols list)")
                try:
                    resp = await client.call_tool("get_bulk_quote", {"symbols": STOCKS})
                    text = extract_text(resp.content)
                    joined = " ".join(text)
                    if "Exception" not in joined:
                        results["bhavcopy"]["historical"]["bulk_quote"] = {"tool": "get_bulk_quote", "data": text}
                        print(f"    Bulk: ✓ OK ({len(joined)} chars)")
                    else:
                        print(f"    Bulk: Server error — {joined[:120]}")
                except Exception as e:
                    print(f"    Bulk: ✗ {e}")

            # --- Attempt 4: get_ltp_by_date (single date test) ---
            print("\n  Strategy 4: get_ltp_by_date (RELIANCE, 2026-11-14)")
            try:
                resp = await client.call_tool("get_ltp_by_date", {"symbol": "RELIANCE", "date": "2026-11-14"})
                text = extract_text(resp.content)
                joined = " ".join(text)
                results["bhavcopy"]["historical"]["RELIANCE_ltp_test"] = {"tool": "get_ltp_by_date", "data": text}
                if "Exception" not in joined:
                    print(f"    RELIANCE LTP: ✓ OK — {joined[:200]}")
                else:
                    print(f"    RELIANCE LTP: Server error — {joined[:120]}")
            except Exception as e:
                print(f"    RELIANCE LTP: ✗ {e}")

            # --- Attempt 5: search_symbols ---
            print("\n  Strategy 5: search_symbols (RELIANCE)")
            try:
                resp = await client.call_tool("search_symbols", {"query": "RELIANCE"})
                text = extract_text(resp.content)
                joined = " ".join(text)
                results["bhavcopy"]["historical"]["search_test"] = {"tool": "search_symbols", "data": text}
                if "Exception" not in joined:
                    print(f"    Search: ✓ OK — {joined[:300]}")
                else:
                    print(f"    Search: Server error — {joined[:120]}")
            except Exception as e:
                print(f"    Search: ✗ {e}")

    except Exception as e:
        results["bhavcopy"]["errors"].append(traceback.format_exc())
        print(f"  ✗ Connection FAILED: {e}")


async def main():
    print("\n" + "=" * 70)
    print("NeuroTrader — NSE MCP Connection Test (Retry)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    await test_cm_quotes()
    await test_bhavcopy_history()

    # Save full results
    outpath = DATA_LIVE_DIR / "nse_mcp_retry_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {outpath}")

    print("\n" + "=" * 70)
    print("RETRY SUMMARY")
    print(f"  CM Market connected: {results['cm_market']['connected']}")
    cm_ok = sum(1 for v in results['cm_market']['quotes'].values()
                if isinstance(v, dict) and "data" in v and "Exception" not in str(v.get("data", "")))
    print(f"  CM Market successful quotes: {cm_ok}/{len(results['cm_market']['quotes'])}")
    bh_ok = sum(1 for v in results['bhavcopy']['historical'].values()
                if isinstance(v, dict) and "data" in v and "Exception" not in str(v.get("data", "")))
    print(f"  Bhavcopy successful data: {bh_ok}/{len(results['bhavcopy']['historical'])}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
