"""Export a single JSON data file for the public dashboard.

Everything the site renders comes from committed artifacts — the reports
written by `backtest`/`evaluate`/`significance`/`sweep`, plus the
append-only pre-registered forecasts. The dashboard is a view of the
research record, never a separate computation.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import ingest
from .config import FORECASTS_DIR, REPORTS_DIR, ROOT, WEATHER_SERIES

DEST = ROOT / "dashboard" / "public" / "data.json"
N_BINS = 10


def _read(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _reliability(bins: pd.DataFrame) -> list[dict]:
    """Pooled reliability curve for both forecasters: every bin's issued
    probability against whether that bin resolved yes."""
    b = bins.dropna(subset=["mid"]).copy()
    # De-vig the market's mids within each event so both vectors are
    # normalized the same way before pooling.
    b["p_market"] = b["mid"] / b.groupby("event_ticker")["mid"].transform("sum")
    hit = (b["result"] == "yes").to_numpy(float)
    out = []
    for col, name in [("p_model", "model"), ("p_market", "market")]:
        probs = b[col].to_numpy(float)
        idx = np.clip((probs * N_BINS).astype(int), 0, N_BINS - 1)
        for k in range(N_BINS):
            mask = idx == k
            if mask.sum() < 10:
                continue
            out.append({
                "series": name,
                "bin_mid": round((k + 0.5) / N_BINS, 2),
                "predicted": round(float(probs[mask].mean()), 4),
                "observed": round(float(hit[mask].mean()), 4),
                "n": int(mask.sum()),
            })
    return out


def _daily(events: pd.DataFrame) -> list[dict]:
    g = events.groupby("date").agg(
        model_brier=("brier_model", "mean"),
        market_brier=("brier_market", "mean"),
        n=("event_ticker", "size"),
    )
    return [
        {"date": d, "model_brier": round(r.model_brier, 4),
         "market_brier": round(r.market_brier, 4), "n": int(r.n)}
        for d, r in g.iterrows()
    ]


def _live() -> dict:
    """Pre-registered live forecasts, split into still-open and resolved.

    Resolved rows are joined to the archive's settled results, so the
    site shows how issued forecasts actually scored — the payoff of the
    pre-registration loop."""
    path = FORECASTS_DIR / "weather.csv"
    if not path.exists():
        return {"open": [], "resolved": [], "summary": None}
    f = pd.read_csv(path)
    # An event may be forecast on several days; keep the latest issue.
    f = f.sort_values("issue_ts").drop_duplicates(subset=["ticker"], keep="last")

    markets = ingest.load_markets()[["ticker", "result", "expiration_value"]]
    m = f.merge(markets, on="ticker", how="left")
    m["city"] = m["series"].map(lambda s: WEATHER_SERIES[s]["city"])
    m["mid"] = (m["yes_bid"] + m["yes_ask"]) / 2

    resolved = m[m["result"].notna()].copy()
    open_ = m[m["result"].isna()].copy()

    summary = None
    if not resolved.empty:
        hit = (resolved["result"] == "yes").to_numpy(float)
        ev = resolved.groupby("event_ticker")
        p_mkt = resolved["mid"] / ev["mid"].transform("sum")
        summary = {
            "events": int(resolved["event_ticker"].nunique()),
            "bins": int(len(resolved)),
            "brier_model": round(float(((resolved["p_model"] - hit) ** 2).sum()
                                       / resolved["event_ticker"].nunique()), 4),
            "brier_market": round(float(((p_mkt - hit) ** 2).sum()
                                        / resolved["event_ticker"].nunique()), 4),
        }

    cols = ["issue_ts", "city", "event_ticker", "event_date", "ticker", "strike_type",
            "floor", "cap", "p_model", "mu", "sigma", "yes_bid", "yes_ask"]
    return {
        "open": json.loads(open_[cols].to_json(orient="records")),
        "resolved": json.loads(resolved[cols + ["result", "expiration_value"]]
                               .tail(240).to_json(orient="records")),
        "summary": summary,
    }


def run() -> None:
    payload: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cities": {st: cfg["city"] for st, cfg in WEATHER_SERIES.items()},
    }

    events = _read("backtest_events.csv")
    if events is not None:
        scored = events.dropna(subset=["brier_model", "brier_market"])
        payload["backtest"] = {
            "events": int(len(scored)),
            "start": str(scored["date"].min()),
            "end": str(scored["date"].max()),
            "daily": _daily(scored),
            "forecast_mae": round(float((scored["official_high"] - scored["forecast_high"]).abs().mean()), 2),
        }

    for key, name in [("evaluation", "evaluation.csv"), ("by_city", "evaluation_by_city.csv"),
                      ("significance", "significance.csv"), ("sweep", "snapshot_sweep.csv"),
                      ("variants", "variants.csv"), ("variant_significance", "variant_significance.csv")]:
        df = _read(name)
        if df is not None:
            payload[key] = json.loads(df.to_json(orient="records"))

    bins = _read("backtest_bins.csv")
    if bins is not None:
        payload["reliability"] = _reliability(bins)
        modal = bins.groupby("event_ticker").apply(
            lambda g: pd.Series({
                "model": g.loc[g["p_model"].idxmax(), "result"] == "yes",
                "market": g.loc[g["mid"].idxmax(), "result"] == "yes" if g["mid"].notna().any() else np.nan,
            }),
            include_groups=False,
        )
        payload["modal_hit"] = {
            "model": round(float(modal["model"].mean()), 4),
            "market": round(float(modal["market"].mean()), 4),
        }

    payload["live"] = _live()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {DEST} ({DEST.stat().st_size / 1024:.0f} KB)")
