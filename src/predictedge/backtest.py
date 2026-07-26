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

import numpy as np
import pandas as pd

from . import ingest, market, weather
from .config import MAX_BIN_SPREAD, MIN_N_CALIB, PRIOR_N, REPORTS_DIR, WEATHER_SERIES
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


@dataclass(frozen=True)
class Variant:
    """A model configuration. `ensemble` averages several NWP systems at
    the same lead instead of using Open-Meteo's default (which tracks
    GFS) — identical information timing, better central estimate.
    `spread_sigma` additionally conditions the predictive width on how
    much those members disagree that day."""
    name: str
    ensemble: bool = False
    spread_sigma: bool = False
    weighted: bool = False
    empirical_shape: bool = False
    pooled_weights: bool = False  # learn member skill across all cities, not one
    linear_calib: bool = False  # correct conditional bias with a slope, not just a shift
    # Variant this one differs from by exactly one change, so the
    # variant-vs-variant test attributes the difference to that change.
    parent: str | None = None


BASELINE = Variant("baseline (single NWP)")
ENSEMBLE = Variant("ensemble (multi-model mean)", ensemble=True, parent=BASELINE.name)
ENSEMBLE_SPREAD = Variant("ensemble + spread-conditional sigma", ensemble=True,
                          spread_sigma=True, parent=ENSEMBLE.name)
WEIGHTED = Variant("+ skill-weighted members", ensemble=True, spread_sigma=True,
                   weighted=True, parent=ENSEMBLE_SPREAD.name)
EMPIRICAL = Variant("+ empirical residual shape", ensemble=True, spread_sigma=True,
                    weighted=True, empirical_shape=True, parent=WEIGHTED.name)
POOLED = Variant("+ pooled member skill", ensemble=True, spread_sigma=True,
                 weighted=True, pooled_weights=True, parent=WEIGHTED.name)
CALIB = Variant("+ linear calibration", ensemble=True, spread_sigma=True,
                weighted=True, linear_calib=True, parent=WEIGHTED.name)
VARIANTS = [BASELINE, ENSEMBLE, ENSEMBLE_SPREAD, WEIGHTED, EMPIRICAL, POOLED, CALIB]

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


def _forecasts(markets: pd.DataFrame, lead_days: int,
               variant: Variant = BASELINE) -> dict[str, pd.DataFrame]:
    """Per-series frame indexed by date: one column per ensemble member
    (or a single column for the baseline). The combination into a point
    forecast happens per event, because weighted variants need the
    walk-forward member skill available at that moment."""
    out = {}
    for st, cfg in WEATHER_SERIES.items():
        dates = markets.loc[markets["series_ticker"] == st, "date"]
        if not len(dates):
            continue
        args = (cfg["lat"], cfg["lon"], cfg["tz"], dates.min(), dates.max())
        if variant.ensemble:
            out[st] = weather.ensemble_highs(*args, lead_days=lead_days)
        else:
            out[st] = weather.day_ahead_highs(*args, lead_days=lead_days).to_frame("default")
    return out


def _calibration(fcs: list[float], truths: list[float]) -> tuple[float, float]:
    """Walk-forward linear calibration truth ~ a + b*forecast.

    A pure bias shift can only move the forecast up or down; a slope also
    corrects *conditional* bias, e.g. NWP compressing extremes toward the
    seasonal mean. Both coefficients are shrunk toward the identity
    mapping (a=0, b=1) by PRIOR_N pseudo-observations so early days stay
    close to the raw forecast."""
    n = len(fcs)
    if n < MIN_N_CALIB:
        return 0.0, 1.0
    x = np.asarray(fcs, float)
    y = np.asarray(truths, float)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 0:
        return 0.0, 1.0
    b = float(((x - xm) * (y - ym)).sum() / denom)
    a = float(ym - b * xm)
    # Shrink toward identity; clip the slope so a wild fit can't invert
    # or explode the forecast.
    b = (n * b + PRIOR_N * 1.0) / (n + PRIOR_N)
    a = (n * a + PRIOR_N * 0.0) / (n + PRIOR_N)
    b = min(max(b, 0.5), 1.5)
    return a, b


def _member_weights(history: list[np.ndarray], k: int) -> np.ndarray:
    """Inverse-MSE member weights from past per-member errors, shrunk
    toward equal weighting by PRIOR_N pseudo-observations. Equal weights
    until there is any history."""
    equal = np.full(k, 1.0 / k)
    if not history:
        return equal
    errs = np.vstack(history)
    n = len(errs)
    with np.errstate(invalid="ignore"):
        mse = np.nanmean(errs**2, axis=0)
    if not np.isfinite(mse).all() or (mse <= 0).any():
        return equal
    inv = 1.0 / mse
    fitted = inv / inv.sum()
    w = (n * fitted + PRIOR_N * equal) / (n + PRIOR_N)
    return w / w.sum()


def _score(markets: pd.DataFrame, candles: pd.DataFrame, snap: Snapshot,
           variant: Variant = BASELINE) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = _forecasts(markets, snap.lead_days, variant)
    history: dict[str, list[tuple]] = {st: [] for st in WEATHER_SERIES}
    member_history: dict[str, list[tuple]] = {st: [] for st in WEATHER_SERIES}
    history_full: dict[str, list[tuple]] = {st: [] for st in WEATHER_SERIES}
    rows, bin_rows = [], []
    ordered = markets[["event_ticker", "date", "series_ticker"]].drop_duplicates().sort_values(["date", "event_ticker"])

    for ev in ordered.itertuples():
        g = markets[markets["event_ticker"] == ev.event_ticker]
        st = ev.series_ticker
        skip = None
        outcome = g[g["result"] == "yes"]
        official = g["expiration_value"].dropna()
        fdf = forecasts.get(st)
        row_fc = fdf.loc[ev.date] if fdf is not None and ev.date in fdf.index else None
        if row_fc is None:
            members = None
            fc = spread = None
        else:
            members = row_fc.to_numpy(float)
            if variant.weighted:
                # Pooled draws on every city's history; the lag rule still
                # excludes same-day observations, so no city ever informs
                # another about a day that hasn't finished.
                src = (
                    [x for hs in member_history.values() for x in hs]
                    if variant.pooled_weights else member_history[st]
                )
                usable_m = [e for d, e in src if (ev.date - d).days >= snap.error_lag]
                w = _member_weights(usable_m, len(members))
                ok = np.isfinite(members)
                fc = float((w[ok] * members[ok]).sum() / w[ok].sum()) if ok.any() else None
            else:
                fc = float(np.nanmean(members)) if np.isfinite(members).any() else None
            spread = float(np.nanstd(members)) if len(members) > 1 else float("nan")
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
            usable = [(e, sp) for d, e, sp in history[st] if (ev.date - d).days >= snap.error_lag]
            state = ErrorState([e for e, _ in usable], [sp for _, sp in usable])
            if variant.linear_calib:
                hist = [(f, t_) for d, _, _, f, t_ in history_full[st]
                        if (ev.date - d).days >= snap.error_lag]
                a, b = _calibration([f for f, _ in hist], [t_ for _, t_ in hist])
                mu = a + b * fc
            else:
                mu = fc + state.bias
            sigma = state.sigma_for(spread) if variant.spread_sigma else state.sigma
            bins = list(zip(g["strike_type"], g["floor_strike"], g["cap_strike"]))
            cdf = state.residual_cdf(variant.spread_sigma) if variant.empirical_shape else None
            p_model = bin_probs(mu, sigma, bins, cdf)
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
                "spread": float(spread) if spread is not None and pd.notna(spread) else None,
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
            truth = float(official.iloc[0])
            history[st].append((ev.date, truth - float(fc),
                               float(spread) if pd.notna(spread) else float("nan")))
            history_full[st].append((ev.date, truth - float(fc),
                                     float(spread) if pd.notna(spread) else float("nan"),
                                     float(fc), truth))
            member_history[st].append((ev.date, truth - members))
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


def sweep(variants: list[Variant] | None = None) -> pd.DataFrame:
    """Model-vs-market gap at each decision time, with significance.

    Run across variants, this answers the question the estimator
    improvements raised: if the market's remaining edge is *fresher
    information* rather than skill, the gap should be smallest at the
    earliest snapshot — where the market is also working from day-old
    model runs — and widen as the market gains runs we cannot access.
    """
    from .significance import cluster_bootstrap, diebold_mariano

    markets = _weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    rows = []
    for variant in variants or [BASELINE]:
      for snap in SWEEP:
        ev, _ = _score(markets, candles, snap, variant)
        g = ev.dropna(subset=["brier_model", "brier_market"])
        g = g[g["included"] == True]  # noqa: E712
        d = (g["brier_model"] - g["brier_market"]).to_numpy(float)
        lo, hi, p_boot = cluster_bootstrap(d, g["date"].to_numpy())
        per_date = pd.DataFrame({"d": d, "date": g["date"]}).groupby("date")["d"].mean().sort_index()
        dm, p_dm = diebold_mariano(per_date.to_numpy())
        rows.append({
            "variant": variant.name,
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
        print(f"{variant.name} @ {snap.label}: done ({len(g)} events)")
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "snapshot_sweep.csv", index=False)
    print(out.to_string(index=False))
    return out


def compare() -> pd.DataFrame:
    """Model variants against the market and against each other.

    Both variants are scored on identical events, bins and snapshot, so
    the variant-vs-variant differential isolates the single change (the
    forecast input) with no other moving part.
    """
    from .significance import cluster_bootstrap, diebold_mariano

    markets = _weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    scored: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        ev, _ = _score(markets, candles, PRIMARY, variant)
        g = ev.dropna(subset=["brier_model", "brier_market"])
        scored[variant.name] = g[g["included"] == True].set_index("event_ticker")  # noqa: E712
        print(f"{variant.name}: {len(scored[variant.name])} events scored")

    def _test(d: np.ndarray, dates: np.ndarray, label: str, metric: str, n: int) -> dict:
        lo, hi, p_boot = cluster_bootstrap(d, dates)
        per_date = pd.DataFrame({"d": d, "date": dates}).groupby("date")["d"].mean().sort_index()
        dm, p_dm = diebold_mariano(per_date.to_numpy())
        return {
            "comparison": label, "metric": metric, "events": n,
            "mean_diff": round(float(d.mean()), 5),
            "ci_low": round(lo, 5), "ci_high": round(hi, 5),
            "p_bootstrap": round(p_boot, 4),
            "dm_stat": round(dm, 3), "p_dm": round(p_dm, 4),
        }

    rows = []
    for metric in ("brier", "logloss"):
        for name, g in scored.items():
            d = (g[f"{metric}_model"] - g[f"{metric}_market"]).to_numpy(float)
            rows.append(_test(d, g["date"].to_numpy(), f"{name} vs market", metric, len(d)))
        # Each variant against its declared parent, aligned on shared
        # events, so every row isolates a single change.
        for cur in VARIANTS:
            if cur.parent is None:
                continue
            a, b = scored[cur.name], scored[cur.parent]
            common = a.index.intersection(b.index)
            d = (a.loc[common, f"{metric}_model"] - b.loc[common, f"{metric}_model"]).to_numpy(float)
            rows.append(_test(d, a.loc[common, "date"].to_numpy(),
                              f"{cur.name} vs {cur.parent}", metric, len(common)))

    summary = pd.DataFrame([
        {"variant": name, "events": len(g),
         "brier_model": round(g["brier_model"].mean(), 4),
         "brier_market": round(g["brier_market"].mean(), 4),
         "logloss_model": round(g["logloss_model"].mean(), 4),
         "logloss_market": round(g["logloss_market"].mean(), 4),
         "forecast_mae": round(float((g["official_high"] - g["forecast_high"]).abs().mean()), 3)}
        for name, g in scored.items()
    ])
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORTS_DIR / "variants.csv", index=False)
    out.to_csv(REPORTS_DIR / "variant_significance.csv", index=False)
    print()
    print(summary.to_string(index=False))
    print()
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
