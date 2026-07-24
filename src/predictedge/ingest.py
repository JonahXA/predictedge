"""Archive Kalshi settled markets and their price history.

Kalshi's public API only lists roughly the last two months of settled
markets — anything older is unreachable (verified: direct ticker fetches
404). The archive under data/archive/ is therefore the durable record:
each run merges newly visible settled markets and their candlesticks
into committed parquet files, never dropping rows that have aged out
upstream.
"""

from __future__ import annotations

import pandas as pd

from . import kalshi
from .config import ARCHIVE_DIR, FORWARD_SERIES, WEATHER_SERIES

MARKET_COLS = [
    "series_ticker", "event_ticker", "ticker", "market_type", "title",
    "yes_sub_title", "strike_type", "floor_strike", "cap_strike",
    "open_time", "close_time", "expected_expiration_time", "settlement_ts",
    "status", "result", "expiration_value", "volume_fp", "open_interest_fp",
    "liquidity_dollars", "last_price_dollars",
]

CANDLE_FIELDS = {
    "end_period_ts": ("end_period_ts",),
    "price_close": ("price", "close_dollars"),
    "price_mean": ("price", "mean_dollars"),
    "yes_bid_close": ("yes_bid", "close_dollars"),
    "yes_ask_close": ("yes_ask", "close_dollars"),
    "volume_fp": ("volume_fp",),
    "open_interest_fp": ("open_interest_fp",),
}


def _merge(path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Merge new rows into an archive parquet, new rows winning on key
    collisions, and never deleting old rows."""
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=keys, keep="last").sort_values(keys)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def _flatten_candle(c: dict) -> dict:
    row = {}
    for col, keypath in CANDLE_FIELDS.items():
        v: object = c
        for k in keypath:
            v = v.get(k) if isinstance(v, dict) else None
        row[col] = v
    return row


def _archived_candle_tickers() -> set[str]:
    path = ARCHIVE_DIR / "candles.parquet"
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["ticker"])["ticker"].unique())


def ingest_series(series_ticker: str, have_candles: set[str] | None = None) -> tuple[int, int]:
    """Fetch all visible settled markets for one series plus hourly
    candles, and merge into the archive. Settled markets are immutable,
    so candles are only fetched for tickers not already archived —
    a daily run touches ~one day's worth of new markets.
    Returns (n_markets, n_new_candles)."""
    markets = kalshi.settled_markets(series_ticker)
    have = _archived_candle_tickers() if have_candles is None else have_candles
    rows, candle_rows = [], []
    for m in markets:
        rows.append({c: m.get(c) for c in MARKET_COLS} | {"series_ticker": series_ticker})
        if m["ticker"] in have:
            continue
        for c in kalshi.candlesticks(series_ticker, m):
            candle_rows.append({"ticker": m["ticker"]} | _flatten_candle(c))
    if rows:
        _merge(ARCHIVE_DIR / "markets.parquet", pd.DataFrame(rows), ["ticker"])
    if candle_rows:
        df = pd.DataFrame(candle_rows)
        for col in df.columns:
            if col != "ticker":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _merge(ARCHIVE_DIR / "candles.parquet", df, ["ticker", "end_period_ts"])
    return len(rows), len(candle_rows)


def run(include_forward: bool = True) -> None:
    for st in list(WEATHER_SERIES) + (FORWARD_SERIES if include_forward else []):
        n_m, n_c = ingest_series(st)
        print(f"{st}: {n_m} settled markets, {n_c} candles merged")


def load_markets() -> pd.DataFrame:
    df = pd.read_parquet(ARCHIVE_DIR / "markets.parquet")
    for col in ("floor_strike", "cap_strike", "expiration_value", "volume_fp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_candles() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE_DIR / "candles.parquet")
