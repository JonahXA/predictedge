"""As-of weather forecasts from Open-Meteo (free, unauthenticated).

The model input for an event on day D is the *day-ahead* forecast — what
the weather model predicted for D as of D-1. Open-Meteo's Previous Runs
API archives exactly this: `temperature_2m_previous_day1` is, for each
hour, the value predicted for that hour by the run one day earlier. The
daily max of those hours is the day-ahead forecast of the day's high.
Using it (rather than the current-run forecast or the observed high)
is what keeps the feature strictly pre-snapshot.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .cache import get_json

PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Independent NWP systems Open-Meteo archives at the same lead. Averaging
# these is a "poor man's ensemble": same information timing as any single
# member, but less run-specific noise. Verified to return genuinely
# different forecasts (spreads up to ~12 deg F on hard days); the API's
# default best_match tracks GFS, so the single-model baseline is GFS.
ENSEMBLE_MODELS = [
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless",
    "jma_seamless",
    "meteofrance_seamless",
    # Added after probing: these return genuinely distinct series for US
    # points. Many other ids are aliases that silently duplicate a member
    # (gfs_hrrr and ncep_hrrr_conus return gfs_seamless; arpege_world
    # returns meteofrance; dmi/metno return knmi), which would double-
    # weight that model — `ensemble_highs` drops duplicates defensively.
    "ukmo_seamless",
    "ecmwf_aifs025_single",
    "cma_grapes_global",
    "knmi_seamless",
]

# NOTE: there is no lead-0 variable. Requesting
# `temperature_2m_previous_day0` silently returns plain `temperature_2m`,
# which is the latest analysis — it would leak the outcome. Never use it.


def _extreme(df: pd.DataFrame, kind: str) -> pd.Series:
    """Daily max or min of the hourly forecast, by local calendar day."""
    g = df.groupby(df["time"].dt.date)["t"]
    return g.max() if kind == "high" else g.min()


def _fetch_highs(lat: float, lon: float, tz: str, start: date, end: date,
                 var: str, model: str | None, kind: str = "high") -> pd.Series:
    params = {
        "latitude": lat, "longitude": lon, "hourly": var,
        "temperature_unit": "fahrenheit", "timezone": tz,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    }
    if model:
        params["models"] = model
    for base in (PREVIOUS_RUNS, HISTORICAL_FORECAST):
        try:
            d = get_json(base, params)
        except Exception:
            continue
        hours = d.get("hourly", {})
        if not hours.get("time") or var not in hours:
            continue
        df = pd.DataFrame({"time": pd.to_datetime(hours["time"]), "t": hours[var]}).dropna()
        if df.empty:
            continue
        return _extreme(df, kind)
    raise RuntimeError(f"no forecast data for ({lat},{lon}) {start}..{end} var={var} model={model}")


def day_ahead_highs(lat: float, lon: float, tz: str, start: date, end: date,
                    lead_days: int = 1, kind: str = "high") -> pd.Series:
    """Day-ahead forecast of the daily extreme (deg F) for each local
    calendar day in [start, end], indexed by date."""
    return _fetch_highs(lat, lon, tz, start, end,
                        f"temperature_2m_previous_day{lead_days}", None, kind)


def ensemble_highs(lat: float, lon: float, tz: str, start: date, end: date,
                   lead_days: int = 1, models: list[str] | None = None,
                   kind: str = "high") -> pd.DataFrame:
    """One column per NWP model, all at the same lead, indexed by date.

    Every member is drawn from the same `previous_dayN` archive, so the
    ensemble carries exactly the same information timing as the
    single-model baseline — only the estimate is better."""
    var = f"temperature_2m_previous_day{lead_days}"
    cols = {}
    for m in models or ENSEMBLE_MODELS:
        try:
            cols[m] = _fetch_highs(lat, lon, tz, start, end, var, m, kind)
        except RuntimeError:
            continue  # a member missing for this window is dropped, not fatal
    if not cols:
        raise RuntimeError(f"no ensemble members available for ({lat},{lon}) {start}..{end}")
    return drop_duplicate_members(pd.DataFrame(cols))


def drop_duplicate_members(df: pd.DataFrame) -> pd.DataFrame:
    """Drop members whose series is identical to an earlier one.

    Several Open-Meteo model ids are aliases of the same underlying run.
    Keeping both would silently give that model double weight in the
    mean and understate the inter-model spread, so the first name wins
    and the rest are dropped."""
    keep: list[str] = []
    for col in df.columns:
        if not any(df[col].equals(df[k]) for k in keep):
            keep.append(col)
    return df[keep]
