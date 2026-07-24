"""Issue live, pre-registered forecasts for open weather markets.

This is the forward-looking counterpart of the backtest: for every open
daily-high event, compute the model's bin probabilities and snapshot the
market's current quotes, then APPEND them to forecasts/weather.csv with
an issue timestamp. Rows are never modified or deleted — the git commit
that adds them, made before the event resolves, is the pre-registration.

Causality at issue time: mu uses the *current* Open-Meteo run for the
event's date (legitimately available now), and the bias/sigma state is
built from settled events strictly before today.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import ingest, market, weather
from .cache import get_json
from .config import FORECASTS_DIR, KALSHI_BASE, WEATHER_SERIES
from .models.baseline import ErrorState, bin_probs

LIVE_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _live_highs(lat: float, lon: float, tz: str) -> dict[date, float]:
    d = get_json(LIVE_FORECAST, {
        "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit", "timezone": tz, "forecast_days": 3,
    }, refresh=True)
    daily = d["daily"]
    return {date.fromisoformat(t): v for t, v in zip(daily["time"], daily["temperature_2m_max"])
            if v is not None}


def _error_state(series_ticker: str, today: date) -> ErrorState:
    """Walk-forward bias/sigma from archived settled events before today."""
    m = ingest.load_markets()
    m = m[(m["series_ticker"] == series_ticker) & m["expiration_value"].notna()]
    officials = (
        m.assign(d=m["event_ticker"].map(market.event_date))
        .groupby("d")["expiration_value"].first().sort_index()
    )
    officials = officials[officials.index < today]
    state = ErrorState()
    if len(officials) == 0:
        return state
    cfg = WEATHER_SERIES[series_ticker]
    fc = weather.day_ahead_highs(cfg["lat"], cfg["lon"], cfg["tz"],
                                 officials.index.min(), officials.index.max())
    for d, official in officials.items():
        if d in fc.index:
            state.add(float(official) - float(fc[d]))
    return state


def run() -> pd.DataFrame:
    issue_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    today = datetime.now(timezone.utc).date()
    rows = []
    for st, cfg in WEATHER_SERIES.items():
        d = get_json(f"{KALSHI_BASE}/markets",
                     {"limit": 1000, "status": "open", "series_ticker": st}, refresh=True)
        ms = pd.DataFrame(d.get("markets", []))
        if ms.empty:
            continue
        for col in ("floor_strike", "cap_strike"):
            if col not in ms:
                ms[col] = float("nan")
            ms[col] = pd.to_numeric(ms[col], errors="coerce")
        highs = _live_highs(cfg["lat"], cfg["lon"], cfg["tz"])
        state = _error_state(st, today)
        # Pure day-ahead issuance, matching the backtest design: only
        # events whose local calendar day hasn't started yet. Issuing on
        # a day already in progress would let the forecast (and the
        # market quote next to it) see part of the outcome.
        local_today = datetime.now(ZoneInfo(cfg["tz"])).date()
        for ev, g in ms.groupby("event_ticker"):
            ev_date = market.event_date(ev)
            if ev_date <= local_today or ev_date not in highs:
                continue
            g = g.sort_values(["floor_strike", "cap_strike"], na_position="first")
            mu = highs[ev_date] + state.bias
            bins = list(zip(g["strike_type"], g["floor_strike"], g["cap_strike"]))
            p = bin_probs(mu, state.sigma, bins)
            for i, t in enumerate(g.itertuples()):
                bid = float(t.yes_bid_dollars) if pd.notna(t.yes_bid_dollars) else None
                ask = float(t.yes_ask_dollars) if pd.notna(t.yes_ask_dollars) else None
                rows.append({
                    "issue_ts": issue_ts, "series": st, "event_ticker": ev,
                    "event_date": ev_date, "ticker": t.ticker,
                    "strike_type": t.strike_type, "floor": t.floor_strike, "cap": t.cap_strike,
                    "p_model": round(float(p[i]), 4), "mu": round(mu, 2),
                    "sigma": round(state.sigma, 2), "n_errors": len(state.errors),
                    "yes_bid": bid, "yes_ask": ask,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        print("no open weather events with a forecast available")
        return out
    FORECASTS_DIR.mkdir(exist_ok=True)
    path = FORECASTS_DIR / "weather.csv"
    # Append-only: issued forecasts are never overwritten.
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"issued {len(out)} bin forecasts across "
          f"{out['event_ticker'].nunique()} events at {issue_ts} -> {path}")
    return out
