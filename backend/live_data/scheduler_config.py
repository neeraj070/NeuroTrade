"""
Scheduler Configuration & NSE Market Calendar Module for NeuroTrade.

Defines:
1. Official National Stock Exchange of India (NSE) holiday schedule for 2026.
2. Weekend exclusion (Saturday/Sunday) and trading day validation.
3. Missed-execution detection and previous/next trading session calculators.
4. Operational settings: scheduled run times, lockfile locations, log paths, retry parameters.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Operational File Paths
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
LOGS_DIR = PROJECT_ROOT / "reports" / "logs"
LOCK_FILE_PATH = LIVE_DATA_DIR / "orchestrator.lock"
STATUS_FILE_PATH = LOGS_DIR / "last_run_status.json"

# Scheduling Timing (NSE post-market Bhavcopy is typically released 16:00-16:30 IST)
DEFAULT_SCHEDULED_TIME = "16:30:00"
CACHE_TTL_SECONDS = 180
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

# Official NSE Trading Holidays for 2026 (Capital Market Segment)
NSE_HOLIDAYS_2026: Dict[str, str] = {
    "2026-01-26": "Republic Day",
    "2026-03-17": "Mahashivratri",
    "2026-03-25": "Holi",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-04-20": "Id-Ul-Fitr (Ramzan Id)",
    "2026-05-01": "Maharashtra Day",
    "2026-06-26": "Bakri Id (Eid ul-Adha)",
    "2026-07-27": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-08": "Diwali Laxmi Pujan (Special Muhurat Trading Only)",
    "2026-11-09": "Diwali Balipratipada",
    "2026-11-24": "Gurunanak Jayanti",
    "2026-12-25": "Christmas",
}


def _parse_date(dt: Union[date, datetime, str]) -> date:
    """Normalizes input to a standard datetime.date instance."""
    if isinstance(dt, datetime):
        return dt.date()
    elif isinstance(dt, date):
        return dt
    elif isinstance(dt, str):
        return datetime.strptime(str(dt)[:10], "%Y-%m-%d").date()
    else:
        raise ValueError(f"Cannot parse date from type {type(dt)}: {dt}")


def is_weekend(dt: Union[date, datetime, str]) -> bool:
    """Returns True if the date falls on a Saturday (5) or Sunday (6)."""
    d = _parse_date(dt)
    return d.weekday() >= 5


def is_nse_holiday(dt: Union[date, datetime, str]) -> Tuple[bool, Optional[str]]:
    """
    Checks if date is a declared NSE holiday.
    Returns: (is_holiday: bool, holiday_name: Optional[str])
    """
    d = _parse_date(dt)
    d_str = d.strftime("%Y-%m-%d")
    if d_str in NSE_HOLIDAYS_2026:
        return True, NSE_HOLIDAYS_2026[d_str]
    return False, None


def is_nse_trading_day(dt: Union[date, datetime, str]) -> bool:
    """Returns True if the date is an active NSE trading day (weekday and non-holiday)."""
    if is_weekend(dt):
        return False
    is_hol, _ = is_nse_holiday(dt)
    return not is_hol


def get_calendar_status(dt: Union[date, datetime, str]) -> Dict[str, Any]:
    """
    Returns comprehensive calendar information for the given date.
    """
    d = _parse_date(dt)
    d_str = d.strftime("%Y-%m-%d")
    weekday_name = d.strftime("%A")

    if is_weekend(d):
        return {
            "date": d_str,
            "weekday": weekday_name,
            "is_trading_day": False,
            "reason": f"Weekend ({weekday_name})",
            "is_weekend": True,
            "is_holiday": False,
            "holiday_name": None,
        }

    is_hol, hol_name = is_nse_holiday(d)
    if is_hol:
        return {
            "date": d_str,
            "weekday": weekday_name,
            "is_trading_day": False,
            "reason": f"NSE Market Holiday: {hol_name}",
            "is_weekend": False,
            "is_holiday": True,
            "holiday_name": hol_name,
        }

    return {
        "date": d_str,
        "weekday": weekday_name,
        "is_trading_day": True,
        "reason": "Active NSE Trading Day",
        "is_weekend": False,
        "is_holiday": False,
        "holiday_name": None,
    }


def get_previous_trading_day(dt: Union[date, datetime, str]) -> date:
    """Calculates the closest preceding active NSE trading day."""
    curr = _parse_date(dt) - timedelta(days=1)
    while not is_nse_trading_day(curr):
        curr -= timedelta(days=1)
    return curr


def get_next_trading_day(dt: Union[date, datetime, str]) -> date:
    """Calculates the closest subsequent active NSE trading day."""
    curr = _parse_date(dt) + timedelta(days=1)
    while not is_nse_trading_day(curr):
        curr += timedelta(days=1)
    return curr


def get_missed_trading_days(
    last_run_date: Union[date, datetime, str],
    current_date: Union[date, datetime, str]
) -> List[date]:
    """
    Identifies any active NSE trading days between last_run_date and current_date
    that were missed (e.g. system was offline or scheduled run failed).
    """
    start = _parse_date(last_run_date) + timedelta(days=1)
    end = _parse_date(current_date)
    missed: List[date] = []

    curr = start
    while curr < end:
        if is_nse_trading_day(curr):
            missed.append(curr)
        curr += timedelta(days=1)

    return missed
