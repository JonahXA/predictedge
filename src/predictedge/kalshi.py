"""Thin client for Kalshi's public (unauthenticated) trade API.

Only public market-data endpoints are used: market listings and
candlestick history. No account, no keys, no order flow.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .cache import get_json
from .config import KALSHI_BASE


def settled_markets(series_ticker: str) -> list[dict]:
    """All settled markets currently visible for a series. Kalshi's
    public listing retains roughly two months, which is why ingestion
    archives these permanently."""
    out: list[dict] = []
    cursor = None
    while True:
        params = {"limit": 1000, "status": "settled", "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        # Listings change daily as markets settle and age out; always refresh.
        d = get_json(f"{KALSHI_BASE}/markets", params, refresh=True)
        ms = d.get("markets", [])
        out.extend(ms)
        cursor = d.get("cursor")
        if not cursor or not ms:
            return out


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def candlesticks(series_ticker: str, market: dict, period_minutes: int = 60) -> list[dict]:
    """Hourly candles (trade OHLC + yes bid/ask) over a settled market's
    whole life. Settled markets are immutable, so these cache forever."""
    start = _ts(market["open_time"])
    end = _ts(market["close_time"])
    d = get_json(
        f"{KALSHI_BASE}/series/{series_ticker}/markets/{market['ticker']}/candlesticks",
        {"start_ts": start, "end_ts": end, "period_interval": period_minutes},
    )
    return d.get("candlesticks", [])
