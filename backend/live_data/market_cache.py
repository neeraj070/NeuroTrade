"""
Market Data Cache Manager for NeuroTrade Live Pipeline.

Provides 3-minute caching for NSE MCP market data:
- Stores cache files in data/live/cache/
- Keeps data separate from data/raw/ and data/processed/
- Tracks retrieval timestamp vs NSE market timestamp
- Automatically validates 3-minute TTL (180 seconds)
- Provides graceful fallback to cached data if NSE is temporarily unreachable
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

CACHE_TTL_SECONDS = 180  # 3 minutes

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "live" / "cache"
METADATA_FILE = CACHE_DIR / "cache_metadata.json"


def ensure_cache_dir() -> Path:
    """Ensures the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_status() -> Dict[str, Any]:
    """
    Checks the status of the local market data cache.

    Returns:
        dict: {
            "exists": bool,
            "is_valid": bool,
            "age_seconds": float,
            "retrieval_timestamp": Optional[str],
            "ttl_seconds": int,
            "cached_stocks": list
        }
    """
    ensure_cache_dir()

    if not METADATA_FILE.exists():
        return {
            "exists": False,
            "is_valid": False,
            "age_seconds": None,
            "retrieval_timestamp": None,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "cached_stocks": [],
        }

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        retrieval_iso = metadata.get("retrieval_timestamp")
        if not retrieval_iso:
            return {"exists": True, "is_valid": False, "age_seconds": None, "retrieval_timestamp": None, "ttl_seconds": CACHE_TTL_SECONDS, "cached_stocks": []}

        retrieval_dt = datetime.fromisoformat(retrieval_iso)
        age = (datetime.now() - retrieval_dt).total_seconds()
        is_valid = (age >= 0) and (age < CACHE_TTL_SECONDS)

        return {
            "exists": True,
            "is_valid": is_valid,
            "age_seconds": round(age, 1),
            "retrieval_timestamp": retrieval_iso,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "cached_stocks": metadata.get("cached_stocks", []),
            "source": metadata.get("source", "NSE_MCP"),
        }
    except Exception as e:
        return {
            "exists": True,
            "is_valid": False,
            "age_seconds": None,
            "retrieval_timestamp": None,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "cached_stocks": [],
            "error": str(e),
        }


def save_to_cache(data: Dict[str, Any]) -> None:
    """
    Saves fetched market data and metadata to local cache.

    Args:
        data: Dict containing:
            - 'historical': {symbol: list of OHLCV bars}
            - 'quotes': {symbol: quote dict}
            - 'market_timestamp': Optional str
    """
    ensure_cache_dir()
    now_iso = datetime.now().isoformat()

    # Save per-stock data
    historical = data.get("historical", {})
    quotes = data.get("quotes", {})
    saved_stocks = list(historical.keys())

    for symbol, bars in historical.items():
        stock_file = CACHE_DIR / f"{symbol}_history.json"
        with open(stock_file, "w", encoding="utf-8") as f:
            json.dump(bars, f, indent=2)

    for symbol, quote in quotes.items():
        quote_file = CACHE_DIR / f"{symbol}_quote.json"
        with open(quote_file, "w", encoding="utf-8") as f:
            json.dump(quote, f, indent=2)

    # Save metadata
    meta = {
        "retrieval_timestamp": now_iso,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "source": "NSE_MCP",
        "cached_stocks": saved_stocks,
        "market_timestamp": data.get("market_timestamp"),
        "total_stocks": len(saved_stocks),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_from_cache(symbols: list) -> Optional[Dict[str, Any]]:
    """
    Loads all cached market data for requested symbols.

    Returns:
        Dict with 'historical', 'quotes', 'retrieval_timestamp', 'market_timestamp'
        or None if required files are missing.
    """
    ensure_cache_dir()
    if not METADATA_FILE.exists():
        return None

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        historical = {}
        quotes = {}

        for symbol in symbols:
            h_file = CACHE_DIR / f"{symbol}_history.json"
            q_file = CACHE_DIR / f"{symbol}_quote.json"

            if not h_file.exists():
                return None

            with open(h_file, "r", encoding="utf-8") as f:
                historical[symbol] = json.load(f)

            if q_file.exists():
                with open(q_file, "r", encoding="utf-8") as f:
                    quotes[symbol] = json.load(f)
            else:
                quotes[symbol] = {}

        return {
            "historical": historical,
            "quotes": quotes,
            "retrieval_timestamp": meta.get("retrieval_timestamp"),
            "market_timestamp": meta.get("market_timestamp"),
            "cache_age_seconds": round((datetime.now() - datetime.fromisoformat(meta["retrieval_timestamp"])).total_seconds(), 1),
            "source": "LOCAL_CACHE",
        }
    except Exception:
        return None
