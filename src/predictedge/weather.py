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


def day_ahead_highs(lat: float, lon: float, tz: str, start: date, end: date,
                    lead_days: int = 1) -> pd.Series:
    """Day-ahead forecast of the daily high (deg F) for each local
    calendar day in [start, end], indexed by date."""
    var = f"temperature_2m_previous_day{lead_days}"
    frames = []
    for base in (PREVIOUS_RUNS, HISTORICAL_FORECAST):
        try:
            d = get_json(base, {
                "latitude": lat, "longitude": lon, "hourly": var,
                "temperature_unit": "fahrenheit", "timezone": tz,
                "start_date": start.isoformat(), "end_date": end.isoformat(),
            })
        except Exception:
            continue
        hours = d.get("hourly", {})
        if not hours.get("time"):
            continue
        df = pd.DataFrame({"time": pd.to_datetime(hours["time"]), "t": hours[var]})
        frames.append(df)
        break
    if not frames:
        raise RuntimeError(f"no day-ahead forecast data for ({lat},{lon}) {start}..{end}")
    df = frames[0].dropna()
    return df.groupby(df["time"].dt.date)["t"].max()
