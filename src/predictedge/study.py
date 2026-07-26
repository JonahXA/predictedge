"""Thin-market study: is market sharpness bought with attention?

The original result — that Kalshi's daily high-temperature markets beat
a public-data model decisively — was measured on the most heavily traded
weather series on the exchange. That is the least favourable place to
test the project's actual thesis, which was that *thin, recreational*
markets are not yet sharpened.

This module runs the identical model against 40 series spanning a ~60x
range in traded volume. Every series is the same modelling problem
(a daily temperature extreme at one station, binned into mutually
exclusive strikes, resolved by the NWS), so the only thing varying
across them is how much money and attention the market attracts.

Two designs, both reported:

  * **Cross-series regression.** Model-minus-market Brier against
    log10 volume per market. A positive slope means the market's edge
    grows with attention.
  * **Paired high-vs-low.** For each city, the daily high and daily low
    market share a station, a day and a resolution source but differ
    roughly 5-20x in volume. Differencing within a city removes any
    city-level effect, so this is the cleanest contrast available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest, ingest
from .config import REPORTS_DIR, WEATHER_SERIES
from .significance import cluster_bootstrap, diebold_mariano


def _volume_per_market() -> pd.Series:
    m = ingest.load_markets()
    m = m[m["series_ticker"].isin(WEATHER_SERIES)]
    v = m.groupby("series_ticker").agg(vol=("volume_fp", "sum"), n=("ticker", "size"))
    return (v["vol"] / v["n"]).rename("vol_per_market")


def per_series(variant: backtest.Variant | None = None) -> pd.DataFrame:
    """Score every series once, then summarise the model-market gap for
    each. `_score` keeps its walk-forward state per series, so a single
    pass is equivalent to running each series independently."""
    variant = variant or backtest.WEIGHTED
    markets = backtest._weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    ev, _ = backtest._score(markets, candles, backtest.STUDY, variant)
    ev = ev.dropna(subset=["brier_model", "brier_market"])
    ev = ev[ev["included"] == True]  # noqa: E712

    vpm = _volume_per_market()
    rows = []
    for st, g in ev.groupby("series"):
        d = (g["brier_model"] - g["brier_market"]).to_numpy(float)
        if len(d) < 30:
            continue
        lo, hi, p_boot = cluster_bootstrap(d, g["date"].to_numpy())
        cfg = WEATHER_SERIES[st]
        rows.append({
            "series": st, "city": cfg["city_key"], "kind": cfg["kind"],
            "events": len(d),
            "vol_per_market": float(vpm.get(st, np.nan)),
            "brier_model": g["brier_model"].mean(),
            "brier_market": g["brier_market"].mean(),
            "d_brier": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "p_bootstrap": p_boot,
            "forecast_mae": float((g["official_high"] - g["forecast_high"]).abs().mean()),
        })
    out = pd.DataFrame(rows).sort_values("vol_per_market", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.round(5).to_csv(REPORTS_DIR / "thin_market.csv", index=False)
    return out


def _regression(x: np.ndarray, y: np.ndarray) -> dict:
    r = stats.linregress(x, y)
    return {
        "slope": float(r.slope), "intercept": float(r.intercept),
        "r": float(r.rvalue), "p": float(r.pvalue), "stderr": float(r.stderr),
        "n": int(len(x)),
    }


def run() -> pd.DataFrame:
    df = per_series()
    print(df[["series", "kind", "events", "vol_per_market", "brier_model",
              "brier_market", "d_brier", "forecast_mae"]]
          .round(4).to_string(index=False))

    findings = []

    # 1. Does the gap grow with traded volume?
    ok = df["vol_per_market"].notna() & (df["vol_per_market"] > 0)
    x = np.log10(df.loc[ok, "vol_per_market"].to_numpy(float))
    y = df.loc[ok, "d_brier"].to_numpy(float)
    reg = _regression(x, y)
    findings.append({"test": "d_brier ~ log10(volume per market), all series", **reg})

    for kind in ("high", "low"):
        s = df[ok & (df["kind"] == kind)]
        if len(s) >= 5:
            reg_k = _regression(np.log10(s["vol_per_market"].to_numpy(float)),
                                s["d_brier"].to_numpy(float))
            findings.append({"test": f"d_brier ~ log10(volume), {kind} only", **reg_k})

    # 2. Paired within-city: high vs low.
    piv = df.pivot_table(index="city", columns="kind", values="d_brier")
    piv = piv.dropna()
    if len(piv) >= 5:
        diff = (piv["high"] - piv["low"]).to_numpy(float)
        t = stats.ttest_rel(piv["high"], piv["low"])
        findings.append({
            "test": "paired within-city: d_brier(high) - d_brier(low)",
            "n": int(len(diff)), "mean_diff": float(diff.mean()),
            "p": float(t.pvalue), "slope": np.nan, "intercept": np.nan,
            "r": np.nan, "stderr": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        })

    out = pd.DataFrame(findings)
    out.round(5).to_csv(REPORTS_DIR / "thin_market_tests.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))

    means = df.groupby("kind").agg(
        series=("series", "size"), events=("events", "sum"),
        vol_per_market=("vol_per_market", "mean"),
        brier_model=("brier_model", "mean"), brier_market=("brier_market", "mean"),
        d_brier=("d_brier", "mean"), forecast_mae=("forecast_mae", "mean"),
    ).round(4)
    means.to_csv(REPORTS_DIR / "thin_market_by_kind.csv")
    print()
    print(means.to_string())
    return df
