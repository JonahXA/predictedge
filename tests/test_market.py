from datetime import date

import numpy as np
import pandas as pd

from predictedge.market import devig, event_date, implied_result, snapshot_quotes, snapshot_ts


def test_event_date_parses_ticker():
    assert event_date("KXHIGHNY-26JUL23") == date(2026, 7, 23)
    assert event_date("KXHIGHLAX-26MAY05") == date(2026, 5, 5)


def test_snapshot_is_0900_utc():
    assert snapshot_ts(date(2026, 7, 23)) % 86400 == 9 * 3600


def test_implied_result_rules():
    assert implied_result("greater", 86.0, np.nan, 87.0) == "yes"
    assert implied_result("greater", 86.0, np.nan, 86.0) == "no"
    assert implied_result("less", np.nan, 79.0, 78.0) == "yes"
    assert implied_result("less", np.nan, 79.0, 79.0) == "no"
    assert implied_result("between", 80.0, 81.0, 81.0) == "yes"
    assert implied_result("between", 80.0, 81.0, 82.0) == "no"


def test_devig_normalizes():
    p = devig(np.array([0.3, 0.4, 0.5]))
    assert p.sum() == 1.0


def test_snapshot_quotes_takes_last_candle_before_cutoff():
    candles = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "end_period_ts": [100, 200, 300],
        "yes_bid_close": [0.10, 0.20, 0.90],
        "yes_ask_close": [0.20, 0.30, 1.00],
        "price_close": [0.15, 0.25, 0.95],
    })
    q = snapshot_quotes(candles, snap_ts=250)
    assert q.loc["A", "end_period_ts"] == 200
    assert q.loc["A", "mid"] == 0.25


def test_local_snapshot_precedes_the_whole_local_day():
    """Daily lows occur pre-dawn, so the study snapshot must sit at local
    midnight — before any of the local day — in every time zone."""
    import datetime as dt

    for tz, off in [("America/New_York", 4), ("America/Los_Angeles", 7),
                    ("America/Chicago", 5), ("America/Denver", 6)]:
        ts = snapshot_ts(date(2026, 7, 23), 0, 0, tz)
        utc = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
        assert utc.hour == off  # local midnight expressed in UTC
        local = utc.astimezone(dt.timezone(dt.timedelta(hours=-off)))
        assert (local.hour, local.date()) == (0, date(2026, 7, 23))


def test_utc_snapshot_unchanged_without_tz():
    assert snapshot_ts(date(2026, 7, 23)) % 86400 == 9 * 3600
