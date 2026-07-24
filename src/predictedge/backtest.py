"""Walk-forward backtest: baseline model vs the market, event by event.

For every settled daily-high event with usable data:

  1. Take the day-ahead forecast (issued the previous day) for the
     event's station.
  2. Apply the city's walk-forward bias/sigma error state (built only
     from strictly earlier days).
  3. Integrate the predictive Normal over the event's strike bins.
  4. Take the market's de-vigged bin mids at the 09:00 UTC snapshot.
  5. Score both against the settled outcome (Brier + log loss).

Events are processed in date order and the error state is updated only
after an event is scored, so no feature ever sees the future.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import ingest, market, weather
from .config import MAX_BIN_SPREAD, REPORTS_DIR, WEATHER_SERIES
from .models.baseline import ErrorState, bin_probs
from .scoring import brier, logloss


def _weather_events(markets: pd.DataFrame) -> pd.DataFrame:
    m = markets[markets["series_ticker"].isin(WEATHER_SERIES)].copy()
    m = m[m["status"].isin(["finalized", "settled"])]
    m["date"] = m["event_ticker"].map(market.event_date)
    return m


def run() -> pd.DataFrame:
    markets = _weather_events(ingest.load_markets())
    candles = ingest.load_candles()

    bad = market.validate_strikes(markets)
    if len(bad):
        raise RuntimeError(
            f"strike parsing disagrees with Kalshi's settled results on "
            f"{len(bad)} markets (e.g. {bad['ticker'].head().tolist()}) — fix before trusting any probabilities"
        )

    # One day-ahead forecast series per city, fetched once.
    forecasts: dict[str, pd.Series] = {}
    for st, cfg in WEATHER_SERIES.items():
        dates = markets.loc[markets["series_ticker"] == st, "date"]
        if len(dates) == 0:
            continue
        forecasts[st] = weather.day_ahead_highs(
            cfg["lat"], cfg["lon"], cfg["tz"], dates.min(), dates.max()
        )

    states = {st: ErrorState() for st in WEATHER_SERIES}
    rows, bin_rows = [], []
    ordered = markets[["event_ticker", "date", "series_ticker"]].drop_duplicates().sort_values(["date", "event_ticker"])

    for ev in ordered.itertuples():
        g = markets[markets["event_ticker"] == ev.event_ticker]
        st = ev.series_ticker
        skip = None
        outcome = g[g["result"] == "yes"]
        official = g["expiration_value"].dropna()
        fc = forecasts.get(st, pd.Series(dtype=float)).get(ev.date)
        if len(outcome) != 1:
            skip = "no_unique_outcome"
        elif len(official) == 0:
            skip = "no_official_value"
        elif fc is None or pd.isna(fc):
            skip = "no_forecast"

        if skip is None:
            g = g.sort_values(["floor_strike", "cap_strike"], na_position="first")
            snap = market.snapshot_quotes(candles[candles["ticker"].isin(g["ticker"])], market.snapshot_ts(ev.date))
            quotes = snap.reindex(g["ticker"])
            state = states[st]
            mu = fc + state.bias
            sigma = state.sigma
            bins = list(zip(g["strike_type"], g["floor_strike"], g["cap_strike"]))
            p_model = bin_probs(mu, sigma, bins)
            mids = quotes["mid"].to_numpy()
            spreads = quotes["spread"].to_numpy()
            outcome_idx = int((g["result"] == "yes").to_numpy().argmax())
            quoted = pd.notna(mids).all()
            included = bool(quoted and (spreads <= MAX_BIN_SPREAD).all())
            row = {
                "event_ticker": ev.event_ticker, "series": st,
                "city": WEATHER_SERIES[st]["city"], "date": ev.date,
                "n_bins": len(g), "official_high": float(official.iloc[0]),
                "forecast_high": float(fc), "mu": float(mu), "sigma": float(sigma),
                "n_errors": len(state.errors),
                "brier_model": brier(p_model, outcome_idx),
                "logloss_model": logloss(p_model, outcome_idx),
                "included": included, "max_spread": float(pd.Series(spreads).max()),
            }
            if quoted:
                p_mkt = market.devig(mids)
                row |= {
                    "brier_market": brier(p_mkt, outcome_idx),
                    "logloss_market": logloss(p_mkt, outcome_idx),
                }
            rows.append(row)
            for i, t in enumerate(g.itertuples()):
                bin_rows.append({
                    "event_ticker": ev.event_ticker, "ticker": t.ticker, "date": ev.date,
                    "series": st, "strike_type": t.strike_type,
                    "floor": t.floor_strike, "cap": t.cap_strike,
                    "p_model": p_model[i], "mid": mids[i],
                    "spread": spreads[i], "result": t.result,
                })
            # Update the walk-forward state only after scoring.
            states[st].add(float(official.iloc[0]) - float(fc))
        else:
            rows.append({"event_ticker": ev.event_ticker, "series": st, "date": ev.date, "skip": skip})

    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "backtest_events.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(REPORTS_DIR / "backtest_bins.csv", index=False)
    scored = out.dropna(subset=["brier_model"]) if "brier_model" in out else out.iloc[0:0]
    print(f"events scored: {len(scored)} / {len(out)}  "
          f"(included in primary: {int(scored['included'].sum()) if len(scored) else 0})")
    return out


def evaluate() -> pd.DataFrame:
    """Aggregate Brier / log loss, model vs market, primary + all."""
    bt = pd.read_csv(REPORTS_DIR / "backtest_events.csv")
    bt = bt.dropna(subset=["brier_model", "brier_market"])
    rows = []
    for label, g in [("primary (tight quotes)", bt[bt["included"] == True]), ("all quoted events", bt)]:  # noqa: E712
        rows.append({
            "sample": label, "events": len(g),
            "brier_model": g["brier_model"].mean(), "brier_market": g["brier_market"].mean(),
            "logloss_model": g["logloss_model"].mean(), "logloss_market": g["logloss_market"].mean(),
        })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPORTS_DIR / "evaluation.csv", index=False)
    print(out.to_string(index=False))
    return out
