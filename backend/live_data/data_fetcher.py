"""
NSE MCP Data Fetcher with 3-Minute Caching.

Fetches:
1. Completed daily OHLCV bars (~120 days) from Bhavcopy endpoint: https://mcp.nseindia.in/bhavcopy/cm/mcp
2. Near-live snapshot quotes from CM Market endpoint: https://mcp.nseindia.in/cmmkt/mcp
3. Integrates with market_cache.py to enforce 3-minute cache validity.

Transport notes:
    NSE MCP servers use Streamable HTTP transport but do NOT support session
    termination (DELETE).  The MCP SDK sends a DELETE request when the Client
    context manager exits with terminate_on_close=True (the default for plain
    URL strings).  NSE returns HTTP 501 → "Session termination failed: 501"
    warning.  We pass terminate_on_close=False to suppress that harmless noise.

    Fetching 5 stocks × 2 chunks (Bhavcopy) + 5 quotes (CM Market) takes
    25-120 seconds depending on NSE server load.  The timeout must be generous
    enough to cover this.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from .market_cache import get_cache_status, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
CM_MARKET_URL = "https://mcp.nseindia.in/cmmkt/mcp"

TARGET_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

# NSE MCP servers respond slowly under load.  Sequential fetch of 5 stocks ×
# 2 history chunks + 5 quotes can take 25-120 s.  180 s is generous without
# hanging indefinitely.
MCP_FETCH_TIMEOUT_SECONDS = 180


def _make_mcp_client(url: str) -> Client:
    """Create an MCP Client with terminate_on_close=False.

    NSE MCP servers return HTTP 501 for the DELETE request the SDK sends on
    context exit.  Setting terminate_on_close=False avoids that harmless but
    noisy warning without affecting data retrieval.
    """
    transport = streamable_http_client(url, terminate_on_close=False)
    return Client(transport)


async def fetch_stock_history(client: Client, symbol: str, retries: int = 2) -> List[Dict[str, Any]]:
    """
    Fetches ~120 completed daily OHLCV bars using two 3-month chained chunks with retries.
    """
    # Chunk 1: latest 3 months
    bars1 = []
    next_end = None
    for attempt in range(retries + 1):
        try:
            res1 = await client.call_tool(
                "get_stock_history",
                {"symbol": symbol, "months": 3, "endDate": "today"}
            )
            data1_raw = [item.text for item in res1.content if hasattr(item, "text")][0]
            data1 = json.loads(data1_raw)
            bars1 = data1.get("data", [])
            next_end = data1.get("next_end_date")
            break
        except Exception as e:
            if attempt == retries:
                logger.warning(f"[{symbol}] Failed to fetch recent chunk after {retries} retries: {e}")
            else:
                await asyncio.sleep(1.0)

    # Chunk 2: prior 3 months if next_end_date is available
    bars2 = []
    if next_end:
        for attempt in range(retries + 1):
            try:
                res2 = await client.call_tool(
                    "get_stock_history",
                    {"symbol": symbol, "months": 3, "endDate": next_end}
                )
                data2_raw = [item.text for item in res2.content if hasattr(item, "text")][0]
                data2 = json.loads(data2_raw)
                bars2 = data2.get("data", [])
                break
            except Exception as e:
                if attempt == retries:
                    logger.warning(f"[{symbol}] Failed to fetch older chunk after {retries} retries: {e}")
                else:
                    await asyncio.sleep(1.0)

    # Combine in chronological order: chunk 2 (older) + chunk 1 (newer)
    combined = bars2 + bars1

    # Deduplicate by date
    seen = set()
    deduped = []
    for bar in combined:
        dt = bar.get("date")
        if dt and dt not in seen:
            seen.add(dt)
            deduped.append(bar)

    deduped.sort(key=lambda x: x.get("date", ""))
    return deduped


async def fetch_stock_quote(client: Client, symbol: str, retries: int = 2) -> Dict[str, Any]:
    """
    Fetches the latest live quote snapshot from CM Market with retries.
    """
    for attempt in range(retries + 1):
        try:
            res = await client.call_tool("cm_get_stock_quote", {"symbol": symbol})
            text = [item.text for item in res.content if hasattr(item, "text")][0]
            data = json.loads(text)
            return data.get("stock", {})
        except Exception as e:
            if attempt == retries:
                logger.warning(f"[{symbol}] Failed to fetch live quote after {retries} retries: {e}")
                return {}
            await asyncio.sleep(1.0)
    return {}


async def fetch_all_market_data(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Main data retrieval function.
    
    1. Checks if a valid (<3 min old) local cache exists.
    2. If valid and not force_refresh, returns cached data immediately.
    3. Otherwise connects to official NSE MCP endpoints to fetch fresh data.
    4. If network fails, falls back to existing cache (if any) with warning.
    
    Returns:
        Dict containing:
            - 'historical': {symbol: list of bars}
            - 'quotes': {symbol: quote dict}
            - 'retrieval_timestamp': ISO timestamp
            - 'market_timestamp': NSE market timestamp
            - 'data_status': 'FRESH' | 'CACHED' | 'FALLBACK_STALE'
            - 'cache_age_seconds': float
    """
    cache_status = get_cache_status()

    # Reuse cache if valid and not forcing refresh
    if cache_status["is_valid"] and not force_refresh:
        cached_data = load_from_cache(TARGET_STOCKS)
        if cached_data:
            return {
                **cached_data,
                "data_status": "CACHED",
                "cache_age_seconds": cache_status["age_seconds"],
                "message": f"Reused local cache ({cache_status['age_seconds']}s old; 3-min TTL)"
            }

    # Fetch fresh from NSE MCP
    historical: Dict[str, List[Dict[str, Any]]] = {}
    quotes: Dict[str, Dict[str, Any]] = {}
    market_timestamp = None

    try:
        async def _do_mcp_fetch():
            nonlocal market_timestamp

            # Use terminate_on_close=False to avoid NSE's HTTP 501 on DELETE
            async with _make_mcp_client(BHAVCOPY_URL) as bhav_client:
                for symbol in TARGET_STOCKS:
                    bars = await fetch_stock_history(bhav_client, symbol)
                    historical[symbol] = bars
                    logger.info(f"[Bhavcopy] {symbol}: {len(bars)} bars fetched")

            async with _make_mcp_client(CM_MARKET_URL) as cm_client:
                for symbol in TARGET_STOCKS:
                    q = await fetch_stock_quote(cm_client, symbol)
                    quotes[symbol] = q
                    if not market_timestamp and q.get("latestTimestamp"):
                        market_timestamp = q.get("latestTimestamp")
                    logger.info(f"[CM Market] {symbol}: LTP ₹{q.get('lastTradedPrice')}")

        await asyncio.wait_for(_do_mcp_fetch(), timeout=MCP_FETCH_TIMEOUT_SECONDS)

        # Package and save to cache
        fresh_data = {
            "historical": historical,
            "quotes": quotes,
            "market_timestamp": market_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_to_cache(fresh_data)

        return {
            **fresh_data,
            "retrieval_timestamp": datetime.now().isoformat(),
            "data_status": "FRESH",
            "cache_age_seconds": 0.0,
            "message": "Fresh data retrieved from official NSE MCP servers."
        }

    except asyncio.TimeoutError:
        err_msg = f"NSE MCP fetch timed out after {MCP_FETCH_TIMEOUT_SECONDS}s"
        logger.error(err_msg)
        cached_data = load_from_cache(TARGET_STOCKS)
        if cached_data:
            return {
                **cached_data,
                "data_status": "FALLBACK_STALE",
                "error": err_msg,
                "message": f"NSE MCP timeout. Falling back to cached data from {cached_data.get('retrieval_timestamp')}."
            }
        raise RuntimeError(f"NSE MCP fetch timed out and no local cache exists.")

    except Exception as e:
        logger.error(f"Error fetching fresh data from NSE MCP: {e}")
        # Fallback to existing cache if available
        cached_data = load_from_cache(TARGET_STOCKS)
        if cached_data:
            return {
                **cached_data,
                "data_status": "FALLBACK_STALE",
                "error": str(e),
                "message": f"NSE connection error. Falling back to cached data from {cached_data.get('retrieval_timestamp')}."
            }
        raise RuntimeError(f"Failed to fetch market data from NSE MCP and no local cache exists: {e}")
