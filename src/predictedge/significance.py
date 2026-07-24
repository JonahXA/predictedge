"""Significance tests on per-event score differentials.

Ported from ClosingLine and adapted for one wrinkle: weather outcomes
are correlated *across cities on the same day* (synoptic systems span
the continent), so treating the ~6 same-day events as independent would
understate the standard error. Both tests therefore cluster by date:

  * Paired bootstrap: resample DATES with replacement, keeping all
    events of a sampled date together; CI and two-sided p on the mean
    per-event differential.
  * Diebold-Mariano on the per-date mean differential series (date
    order), with Newey-West HAC variance (Bartlett kernel, lag n^{1/3})
    for any residual serial correlation across days.

d_i = score_model_i - score_market_i, so positive means the market wins
(lower is better for both metrics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import REPORTS_DIR

RNG = np.random.default_rng(20260724)
N_BOOT = 10000


def diebold_mariano(d: np.ndarray) -> tuple[float, float]:
    """DM statistic and two-sided p for mean(d)=0 with Newey-West HAC
    variance (lag = n^{1/3}, Bartlett kernel)."""
    n = len(d)
    dbar = d.mean()
    demeaned = d - dbar
    lag = max(1, int(round(n ** (1 / 3))))
    var = (demeaned @ demeaned) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (demeaned[k:] @ demeaned[:-k]) / n
    se = np.sqrt(var / n)
    dm = dbar / se
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def cluster_bootstrap(d: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """(ci_low, ci_high, two-sided p) for mean(d), resampling whole
    clusters (dates) with replacement."""
    uniq = np.unique(clusters)
    groups = [d[clusters == c] for c in uniq]
    sums = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    means = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac = (means <= 0).mean() if d.mean() > 0 else (means >= 0).mean()
    return float(lo), float(hi), float(min(1.0, 2 * frac))


def _compare(g: pd.DataFrame, metric: str, sample: str) -> dict:
    d = (g[f"{metric}_model"] - g[f"{metric}_market"]).to_numpy(float)
    dates = g["date"].to_numpy()
    lo, hi, p_boot = cluster_bootstrap(d, dates)
    per_date = pd.DataFrame({"d": d, "date": dates}).groupby("date")["d"].mean().sort_index()
    dm, p_dm = diebold_mariano(per_date.to_numpy())
    return {
        "comparison": "baseline model vs market (de-vigged mid)",
        "sample": sample, "metric": metric, "events": len(d),
        "dates": len(per_date), "mean_diff": round(float(d.mean()), 5),
        "ci_low": round(lo, 5), "ci_high": round(hi, 5),
        "p_bootstrap": round(p_boot, 4), "dm_stat": round(dm, 3), "p_dm": round(p_dm, 4),
    }


def run() -> pd.DataFrame:
    bt = pd.read_csv(REPORTS_DIR / "backtest_events.csv").dropna(subset=["brier_model", "brier_market"])
    rows = []
    for metric in ("brier", "logloss"):
        rows.append(_compare(bt[bt["included"] == True], metric, "primary"))  # noqa: E712
        rows.append(_compare(bt, metric, "all quoted"))
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "significance.csv", index=False)
    print(out.to_string(index=False))
    return out
