"""Walk-forward backtest: baseline model vs the market, event by event.

For every settled daily-high event with usable data:

  1. Take the NWP forecast for the event's station at the config's lead
     (previous_dayN — issued N days ahead, strictly pre-snapshot).
  2. Apply the city's walk-forward bias/sigma error state, built only
     from days at least `error_lag` days before the event (a day's high
     is only fully known after local midnight, so day-before snapshots
     must not see the immediately preceding day's error).
  3. Integrate the predictive Normal over the event's strike bins.
  4. Take the market's de-vigged bin mids at the snapshot.
  5. Score both against the settled outcome (Brier + log loss).

`run()` executes the primary pre-specified config (09:00 UTC on the
event day, day-ahead forecast). `sweep()` repeats the experiment at
earlier decision times — market open + 2h through the primary — to ask
*when* the market sharpens past the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import ingest, market, weather
from .config import MAX_BIN_SPREAD, REPORTS_DIR, WEATHER_SERIES
from .models.baseline import ErrorState, bin_probs
from .scoring import brier, logloss


@dataclass(frozen=True)
class Snapshot:
    label: str
    day_offset: int  # snapshot day relative to the event day
    hour: int  # UTC hour of the snapshot
    lead_days: int  # forecast lead (previous_dayN), strictly pre-snapshot
    error_lag: int  # min days between an error's date and the event


PRIMARY = Snapshot("D 09:00Z", 0, 9, 1, 1)

# Earlier decision times. Day-before snapshots need the 2-day-lead
# forecast (the 1-day run isn't issued yet) and a 2-day error lag; the
# late-evening-UTC snapshots keep lag 2 because the previous local day
# isn't over on the west coast.
SWEEP = [
    Snapshot("D-1 16:00Z (open+2h)", -1, 16, 2, 2),
    Snapshot("D-1 21:00Z", -1, 21, 2, 2),
    Snapshot("D 01:00Z", 0, 1, 1, 2),
    Snapshot("D 05:00Z", 0, 5, 1, 2),
    PRIMARY,
]


def _weather_events(markets: pd.DataFrame) -> pd.DataFrame:
    m = markets[markets["series_ticker"].isin(WEATHER_SERIES)].copy()
    m = m[m["status"].isin(["finalized", "settled"])]
    m["date"] = m["event_ticker"].map(market.event_date)
    return m


def _forecasts(markets: pd.DataFrame, lead_days: int) -> dict[str, pd.Series]:
    out = {}
    for st, cfg in WEATHER_SERIES.items():
        dates = markets.loc[markets["series_ticker"] == st, "date"]
        if len(dates):
            out[st] = weather.day_ahead_highs(
                cfg["lat"], cfg["lon"], cfg["tz"], dates.min(), dates.max(), lead_days=lead_days
            )
    return out


def _score(markets: pd.DataFrame, candles: pd.DataFrame, snap: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = _forecasts(markets, snap.lead_days)
    history: dict[str, list[tuple]] = {st: [] for st in WEATHER_SERIES}
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
            snap_ts = market.snapshot_ts(ev.date, snap.day_offset, snap.hour)
            quotes = market.snapshot_quotes(
                candles[candles["ticker"].isin(g["ticker"])], snap_ts
            ).reindex(g["ticker"])
            usable = [e for d, e in history[st] if (ev.date - d).days >= snap.error_lag]
            state = ErrorState(usable)
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
                "n_errors": len(usable),
                "brier_model": brier(p_model, outcome_idx),
                "logloss_model": logloss(p_model, outcome_idx),
                "included": included,
                "max_spread": float(pd.Series(spreads).max()) if quoted else None,
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
            # Observed only after the event; gated on read by error_lag.
            history[st].append((ev.date, float(official.iloc[0]) - float(fc)))
        else:
            rows.append({"event_ticker": ev.event_ticker, "series": st, "date": ev.date, "skip": skip})

    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def run() -> pd.DataFrame:
    markets = _weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    bad = market.validate_strikes(markets)
    if len(bad):
        raise RuntimeError(
            f"strike parsing disagrees with Kalshi's settled results on "
            f"{len(bad)} markets (e.g. {bad['ticker'].head().tolist()}) — fix before trusting any probabilities"
        )
    out, bins = _score(markets, candles, PRIMARY)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "backtest_events.csv", index=False)
    bins.to_csv(REPORTS_DIR / "backtest_bins.csv", index=False)
    scored = out.dropna(subset=["brier_model"]) if "brier_model" in out else out.iloc[0:0]
    print(f"events scored: {len(scored)} / {len(out)}  "
          f"(included in primary: {int(scored['included'].sum()) if len(scored) else 0})")
    return out


def sweep() -> pd.DataFrame:
    """Model-vs-market gap at each decision time, with significance."""
    from .significance import cluster_bootstrap, diebold_mariano

    markets = _weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    rows = []
    for snap in SWEEP:
        ev, _ = _score(markets, candles, snap)
        g = ev.dropna(subset=["brier_model", "brier_market"])
        g = g[g["included"] == True]  # noqa: E712
        d = (g["brier_model"] - g["brier_market"]).to_numpy(float)
        lo, hi, p_boot = cluster_bootstrap(d, g["date"].to_numpy())
        per_date = pd.DataFrame({"d": d, "date": g["date"]}).groupby("date")["d"].mean().sort_index()
        dm, p_dm = diebold_mariano(per_date.to_numpy())
        rows.append({
            "snapshot": snap.label, "lead_days": snap.lead_days,
            "events": len(g),
            "brier_model": round(g["brier_model"].mean(), 4),
            "brier_market": round(g["brier_market"].mean(), 4),
            "d_brier": round(float(d.mean()), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_bootstrap": round(p_boot, 4), "dm_stat": round(dm, 2), "p_dm": round(p_dm, 4),
            "logloss_model": round(g["logloss_model"].mean(), 4),
            "logloss_market": round(g["logloss_market"].mean(), 4),
        })
        print(f"{snap.label}: done ({len(g)} events)")
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "snapshot_sweep.csv", index=False)
    print(out.to_string(index=False))
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

    by_city = (
        bt.groupby("city")
        .agg(events=("brier_model", "size"),
             brier_model=("brier_model", "mean"), brier_market=("brier_market", "mean"),
             logloss_model=("logloss_model", "mean"), logloss_market=("logloss_market", "mean"))
        .round(4)
    )
    by_city.to_csv(REPORTS_DIR / "evaluation_by_city.csv")
    print()
    print(by_city.to_string())
    return out
