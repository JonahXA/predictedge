"""Cached, rate-limited HTTP GET.

Every upstream request goes through here: responses are cached on disk
keyed by URL+params, so a source is never hit twice for the same
resource (settled markets and their candlesticks are immutable).
Requests are throttled and retried with backoff on 429/5xx.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import requests

from .config import CACHE_DIR

_MIN_INTERVAL = 0.25  # seconds between real HTTP hits
_last_hit = 0.0
_session = requests.Session()
_session.headers["User-Agent"] = "predictedge-research/0.1 (public data; cached; contact via github.com/JonahXA)"


def _key(url: str, params: dict | None) -> str:
    blob = url + "?" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def get_json(url: str, params: dict | None = None, *, refresh: bool = False) -> Any:
    """GET a JSON resource with disk caching. `refresh=True` bypasses the
    cache (for listings that change over time, e.g. open-market pages)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(url, params)}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    global _last_hit
    for attempt in range(6):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_hit)
        if wait > 0:
            time.sleep(wait)
        _last_hit = time.monotonic()
        resp = _session.get(url, params=params, timeout=30)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        path.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"gave up after retries: {url}")
