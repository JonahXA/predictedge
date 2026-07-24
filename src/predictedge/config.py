"""Project paths, data sources, and the study configuration.

Everything that defines the experiment — which series, which snapshot
time, which forecast lead — lives here so the backtest is reproducible
from a single place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARCHIVE_DIR = DATA_DIR / "archive"
REPORTS_DIR = ROOT / "reports"
FORECASTS_DIR = ROOT / "forecasts"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Daily-high-temperature series and the NWS station each one settles on.
# Coordinates are the station's, so Open-Meteo forecasts target the same
# point the market settles against.
WEATHER_SERIES: dict[str, dict] = {
    "KXHIGHNY": {"city": "New York (Central Park)", "lat": 40.783, "lon": -73.967, "tz": "America/New_York"},
    "KXHIGHCHI": {"city": "Chicago (Midway)", "lat": 41.786, "lon": -87.752, "tz": "America/Chicago"},
    "KXHIGHMIA": {"city": "Miami (Intl Airport)", "lat": 25.788, "lon": -80.317, "tz": "America/New_York"},
    "KXHIGHAUS": {"city": "Austin (Camp Mabry)", "lat": 30.321, "lon": -97.760, "tz": "America/Chicago"},
    "KXHIGHDEN": {"city": "Denver (Intl Airport)", "lat": 39.847, "lon": -104.656, "tz": "America/Denver"},
    "KXHIGHLAX": {"city": "Los Angeles (LAX)", "lat": 33.938, "lon": -118.389, "tz": "America/Los_Angeles"},
}

# Series snapshotted for future studies (not enough settled history yet
# to backtest; Kalshi's ~2-month public retention means we archive now
# and analyze later).
FORWARD_SERIES = ["KXCPIYOY", "KXCPI", "KXPAYROLLS", "KXFED"]

# Decision time: 09:00 UTC on the event's local calendar day (05:00 New
# York, 02:00 Los Angeles) — before the day's high develops anywhere in
# the continental US. Model forecast and market price are both taken
# as of this instant.
SNAPSHOT_UTC_HOUR = 9

# The model's weather input is the day-ahead forecast (issued the
# previous day), which is strictly available before the snapshot.
FORECAST_LEAD_DAYS = 1

# Walk-forward error model prior: pseudo-observations for the
# forecast-error sigma before any city-specific history accumulates.
PRIOR_SIGMA_F = 3.0  # deg F, typical day-ahead high-temp forecast RMSE
PRIOR_N = 10

# Primary analysis includes an event only if every bin has a two-sided
# quote at the snapshot with spread <= this. Reported sensitivity: all.
MAX_BIN_SPREAD = 0.15
