"""Market-side parsing: strikes, event structure, snapshot prices.

Each Kalshi daily-high event (e.g. KXHIGHNY-26JUL23) is a mutually
exclusive partition of the temperature line into bins: a bottom tail
("less"), interior "between" bins, and a top tail ("greater"). Exactly
one bin resolves yes. The market's forecast is the de-vigged vector of
bin mid-prices at the snapshot; ours is a predictive distribution
integrated over the same bins.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import SNAPSHOT_UTC_HOUR


def event_date(event_ticker: str) -> date:
    """KXHIGHNY-26JUL23 -> 2026-07-23 (the event's local calendar day)."""
    tail = event_ticker.rsplit("-", 1)[1]
    return datetime.strptime(tail, "%y%b%d").date()


def snapshot_ts(d: date, day_offset: int = 0, hour: int = SNAPSHOT_UTC_HOUR) -> int:
    """Snapshot instant for an event on day d: `hour` UTC on d+day_offset
    (day_offset=-1 → the day before the event)."""
    base = datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc) + timedelta(days=day_offset)
    return int(base.timestamp())


def implied_result(strike_type: str, floor: float, cap: float, value: float) -> str:
    """What the strike rules say the result should be, given the settled
    value — used to validate our parsing against Kalshi's own results."""
    if strike_type == "greater":
        return "yes" if value > floor else "no"
    if strike_type == "less":
        return "yes" if value < cap else "no"
    if strike_type == "between":
        return "yes" if floor <= value <= cap else "no"
    return "unknown"


def validate_strikes(markets: pd.DataFrame) -> pd.DataFrame:
    """Rows where our strike parsing disagrees with Kalshi's settled
    result. Anything here means the bin probabilities would be wrong —
    the backtest refuses to run unless this is empty."""
    m = markets.dropna(subset=["expiration_value"])
    implied = [
        implied_result(r.strike_type, r.floor_strike, r.cap_strike, r.expiration_value)
        for r in m.itertuples()
    ]
    return m[np.array(implied) != m["result"].to_numpy()]


def snapshot_quotes(candles: pd.DataFrame, snap_ts: int) -> pd.DataFrame:
    """Last candle at or before the snapshot for every market: yes bid,
    ask, mid, spread, last trade. One row per ticker."""
    c = candles[candles["end_period_ts"] <= snap_ts]
    c = c.sort_values("end_period_ts").groupby("ticker").tail(1).copy()
    c["mid"] = (c["yes_bid_close"] + c["yes_ask_close"]) / 2
    c["spread"] = c["yes_ask_close"] - c["yes_bid_close"]
    return c.set_index("ticker")[["mid", "spread", "yes_bid_close", "yes_ask_close", "price_close", "end_period_ts"]]


def devig(mids: np.ndarray) -> np.ndarray:
    """Normalize bin mid-prices of a mutually exclusive event to sum
    to 1 — the standard multiplicative de-vig."""
    s = mids.sum()
    return mids / s if s > 0 else mids
