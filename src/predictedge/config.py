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

# Weather stations these markets settle on. Coordinates are the station's,
# so Open-Meteo forecasts target the same point the market settles against.
#
# NOTE: Austin was previously configured as Camp Mabry, but the contract
# rules settle on Austin-Bergstrom. Corrected 2026-07-26; the first
# published six-city results were computed with the wrong Austin station
# and are superseded (git preserves both).
CITIES: dict[str, dict] = {
    "ATL": {"name": "Atlanta", "lat": 33.640, "lon": -84.427, "tz": "America/New_York"},
    "AUS": {"name": "Austin (Bergstrom)", "lat": 30.194, "lon": -97.670, "tz": "America/Chicago"},
    "BOS": {"name": "Boston", "lat": 42.363, "lon": -71.006, "tz": "America/New_York"},
    "CHI": {"name": "Chicago (Midway)", "lat": 41.786, "lon": -87.752, "tz": "America/Chicago"},
    "DAL": {"name": "Dallas (DFW)", "lat": 32.897, "lon": -97.038, "tz": "America/Chicago"},
    "DC": {"name": "Washington DC (DCA)", "lat": 38.848, "lon": -77.034, "tz": "America/New_York"},
    "DEN": {"name": "Denver", "lat": 39.847, "lon": -104.656, "tz": "America/Denver"},
    "HOU": {"name": "Houston (IAH)", "lat": 29.980, "lon": -95.360, "tz": "America/Chicago"},
    "LAX": {"name": "Los Angeles (LAX)", "lat": 33.938, "lon": -118.389, "tz": "America/Los_Angeles"},
    "LV": {"name": "Las Vegas", "lat": 36.080, "lon": -115.152, "tz": "America/Los_Angeles"},
    "MIA": {"name": "Miami", "lat": 25.788, "lon": -80.317, "tz": "America/New_York"},
    "MIN": {"name": "Minneapolis", "lat": 44.883, "lon": -93.229, "tz": "America/Chicago"},
    "NOLA": {"name": "New Orleans", "lat": 29.993, "lon": -90.258, "tz": "America/Chicago"},
    "NY": {"name": "New York (Central Park)", "lat": 40.783, "lon": -73.967, "tz": "America/New_York"},
    "OKC": {"name": "Oklahoma City", "lat": 35.389, "lon": -97.601, "tz": "America/Chicago"},
    "PHIL": {"name": "Philadelphia", "lat": 39.872, "lon": -75.241, "tz": "America/New_York"},
    "PHX": {"name": "Phoenix", "lat": 33.428, "lon": -112.004, "tz": "America/Phoenix"},
    "SATX": {"name": "San Antonio", "lat": 29.544, "lon": -98.484, "tz": "America/Chicago"},
    "SEA": {"name": "Seattle", "lat": 47.444, "lon": -122.314, "tz": "America/Los_Angeles"},
    "SFO": {"name": "San Francisco", "lat": 37.619, "lon": -122.375, "tz": "America/Los_Angeles"},
}

# Kalshi series -> (city key, which daily extreme it settles on). High and
# low markets for the same city share a station and a day but differ ~20x
# in traded volume, which is the natural experiment the thin-market study
# is built on.
_SERIES = {
    "KXHIGHTATL": ("ATL", "high"), "KXLOWTATL": ("ATL", "low"),
    "KXHIGHAUS": ("AUS", "high"), "KXLOWTAUS": ("AUS", "low"),
    "KXHIGHTBOS": ("BOS", "high"), "KXLOWTBOS": ("BOS", "low"),
    "KXHIGHCHI": ("CHI", "high"), "KXLOWTCHI": ("CHI", "low"),
    "KXHIGHTDAL": ("DAL", "high"), "KXLOWTDAL": ("DAL", "low"),
    "KXHIGHTDC": ("DC", "high"), "KXLOWTDC": ("DC", "low"),
    "KXHIGHDEN": ("DEN", "high"), "KXLOWTDEN": ("DEN", "low"),
    "KXHIGHTHOU": ("HOU", "high"), "KXLOWTHOU": ("HOU", "low"),
    "KXHIGHLAX": ("LAX", "high"), "KXLOWTLAX": ("LAX", "low"),
    "KXHIGHTLV": ("LV", "high"), "KXLOWTLV": ("LV", "low"),
    "KXHIGHMIA": ("MIA", "high"), "KXLOWTMIA": ("MIA", "low"),
    "KXHIGHTMIN": ("MIN", "high"), "KXLOWTMIN": ("MIN", "low"),
    "KXHIGHTNOLA": ("NOLA", "high"), "KXLOWTNOLA": ("NOLA", "low"),
    "KXHIGHNY": ("NY", "high"), "KXLOWTNYC": ("NY", "low"),
    "KXHIGHTOKC": ("OKC", "high"), "KXLOWTOKC": ("OKC", "low"),
    "KXHIGHPHIL": ("PHIL", "high"), "KXLOWTPHIL": ("PHIL", "low"),
    "KXHIGHTPHX": ("PHX", "high"), "KXLOWTPHX": ("PHX", "low"),
    "KXHIGHTSATX": ("SATX", "high"), "KXLOWTSATX": ("SATX", "low"),
    "KXHIGHTSEA": ("SEA", "high"), "KXLOWTSEA": ("SEA", "low"),
    "KXHIGHTSFO": ("SFO", "high"), "KXLOWTSFO": ("SFO", "low"),
}

WEATHER_SERIES: dict[str, dict] = {
    ticker: {
        "city": f"{CITIES[c]['name']} {kind}",
        "city_key": c,
        "kind": kind,
        "lat": CITIES[c]["lat"],
        "lon": CITIES[c]["lon"],
        "tz": CITIES[c]["tz"],
    }
    for ticker, (c, kind) in _SERIES.items()
}

# The six series the original pre-registered study used, kept so the
# primary result can still be reproduced in isolation.
PRIMARY_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX"]

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

# Spread-conditional sigma: days where the NWP members disagree deserve
# wider predictive distributions. The variance model var = a + b*spread^2
# is fit walk-forward, so these bounds keep an early or degenerate fit
# from producing absurd distributions.
MIN_N_SPREAD_FIT = 15  # observations before the spread fit is trusted at all
MIN_N_SHAPE_FIT = 25  # observations before the empirical residual shape is trusted
MIN_N_CALIB = 20  # observations before the linear calibration is fit at all
MIN_SIGMA_F = 1.0
MAX_SIGMA_F = 8.0
MAX_SPREAD_SLOPE = 4.0

# Primary analysis includes an event only if every bin has a two-sided
# quote at the snapshot with spread <= this. Reported sensitivity: all.
MAX_BIN_SPREAD = 0.15
