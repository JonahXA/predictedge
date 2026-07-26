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
from .weather import ENSEMBLE_MODELS, drop_duplicate_members

# The live model mirrors the best backtested variant: multi-model mean
# with spread-conditional sigma. Recorded in every issued row so the
# pre-registered record stays interpretable across model changes.
MODEL_TAG = "ensemble-spread-v2"

LIVE_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _live_members(lat: float, lon: float, tz: str) -> pd.DataFrame:
    """Live multi-model forecasts of the daily high, one column per NWP
    member, indexed by date.

    Unlike the backtest — which is capped at `previous_day1` because that
    is the shortest lead the archive exposes — the live path may use each
    model's *current* run. Nothing has resolved yet, so there is no leak,
    and issued forecasts are therefore built on fresher data than the
    backtest could measure."""
    d = get_json(LIVE_FORECAST, {
        "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit", "timezone": tz, "forecast_days": 3,
        "models": ",".join(ENSEMBLE_MODELS),
    }, refresh=True)
    daily = d["daily"]
    idx = [date.fromisoformat(t) for t in daily["time"]]
    cols = {
        k.replace("temperature_2m_max_", ""): v
        for k, v in daily.items()
        if k.startswith("temperature_2m_max")
    }
    if not cols:  # single-model response shape
        cols = {"default": daily["temperature_2m_max"]}
    return drop_duplicate_members(pd.DataFrame(cols, index=idx).dropna(how="all"))


def _error_state(series_ticker: str, today: date) -> ErrorState:
    """Walk-forward bias/sigma from archived settled events before today,
    measured against the same multi-model mean the live path issues."""
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
    members = weather.ensemble_highs(cfg["lat"], cfg["lon"], cfg["tz"],
                                     officials.index.min(), officials.index.max())
    mean, spread = members.mean(axis=1), members.std(axis=1, ddof=0)
    for d, official in officials.items():
        if d in mean.index and pd.notna(mean[d]):
            state.add(float(official) - float(mean[d]),
                      float(spread[d]) if pd.notna(spread[d]) else None)
    return state


def _append(path, new: pd.DataFrame) -> None:
    """Append issued forecasts, tolerating schema growth.

    Naive CSV append breaks the moment the model gains a column, so when
    the schema changes the file is rewritten over the union of columns —
    older rows keep every value they were issued with and simply carry
    blanks for fields that did not exist yet. No issued value is ever
    modified, and git preserves the original bytes either way."""
    if not path.exists():
        new.to_csv(path, index=False)
        return
    old = pd.read_csv(path)
    if list(old.columns) == list(new.columns):
        new.to_csv(path, mode="a", header=False, index=False)
        return
    cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    pd.concat([old, new], ignore_index=True).reindex(columns=cols).to_csv(path, index=False)


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
        members = _live_members(cfg["lat"], cfg["lon"], cfg["tz"])
        highs = members.mean(axis=1)
        spreads = members.std(axis=1, ddof=0)
        state = _error_state(st, today)
        # Pure day-ahead issuance, matching the backtest design: only
        # events whose local calendar day hasn't started yet. Issuing on
        # a day already in progress would let the forecast (and the
        # market quote next to it) see part of the outcome.
        local_today = datetime.now(ZoneInfo(cfg["tz"])).date()
        for ev, g in ms.groupby("event_ticker"):
            ev_date = market.event_date(ev)
            if ev_date <= local_today or ev_date not in highs.index or pd.isna(highs[ev_date]):
                continue
            g = g.sort_values(["floor_strike", "cap_strike"], na_position="first")
            mu = float(highs[ev_date]) + state.bias
            spread = float(spreads[ev_date]) if pd.notna(spreads[ev_date]) else None
            sigma = state.sigma_for(spread)
            bins = list(zip(g["strike_type"], g["floor_strike"], g["cap_strike"]))
            p = bin_probs(mu, sigma, bins)
            for i, t in enumerate(g.itertuples()):
                bid = float(t.yes_bid_dollars) if pd.notna(t.yes_bid_dollars) else None
                ask = float(t.yes_ask_dollars) if pd.notna(t.yes_ask_dollars) else None
                rows.append({
                    "issue_ts": issue_ts, "series": st, "event_ticker": ev,
                    "event_date": ev_date, "ticker": t.ticker,
                    "strike_type": t.strike_type, "floor": t.floor_strike, "cap": t.cap_strike,
                    "p_model": round(float(p[i]), 4), "mu": round(mu, 2),
                    "sigma": round(sigma, 2), "n_errors": len(state.errors),
                    "yes_bid": bid, "yes_ask": ask,
                    "model": MODEL_TAG,
                    "n_members": int(members.shape[1]),
                    "spread": round(spread, 2) if spread is not None else None,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        print("no open weather events with a forecast available")
        return out
    FORECASTS_DIR.mkdir(exist_ok=True)
    path = FORECASTS_DIR / "weather.csv"
    _append(path, out)
    print(f"issued {len(out)} bin forecasts across "
          f"{out['event_ticker'].nunique()} events at {issue_ts} -> {path}")
    return out
