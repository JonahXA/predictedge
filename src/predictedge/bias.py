"""Is the market's *pricing* biased, even where its forecast is better?

Every result so far asked whether our forecast beats the market's. It
does not. But a market can be more accurate overall and still misprice
particular regions of the probability scale — and exploiting that needs
no forecasting edge at all, only the distortion.

The classic form is the **favourite-longshot bias**: cheap contracts
trade rich, expensive ones trade cheap. It is one of the most reproduced
findings in betting-market research, so it is worth measuring here
directly rather than assumed absent because the market wins on Brier.

The honest test has two halves, and the second is the one that matters:

  1. Does realized frequency differ from price, bucketed by price?
  2. Does any such gap **survive the bid-ask spread and exchange fees**?

An anomaly smaller than the cost of correcting it is exactly what an
efficient market looks like from the inside. Reporting (1) without (2)
would be the single most misleading thing this project could publish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REPORTS_DIR

# Price buckets, finer at the cheap end where the bias is expected.
BUCKETS = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.01]

# Exchange fees are charged per contract and are largest in the middle of
# the probability range. Kalshi's published schedule is of the form
# fee = rate * price * (1 - price); the rate here is a placeholder used
# for sensitivity only. VERIFY THE CURRENT SCHEDULE before reading any
# net number as real — at these edge sizes the fee term decides the sign.
FEE_RATE = 0.07


def calibration(bins: pd.DataFrame, fee_rate: float = FEE_RATE) -> pd.DataFrame:
    """Realized frequency vs price by bucket, then net of costs."""
    b = bins.dropna(subset=["mid", "spread"]).copy()
    b["p_mkt"] = b["mid"] / b.groupby("event_ticker")["mid"].transform("sum")
    b["hit"] = (b["result"] == "yes").astype(float)
    b["bucket"] = pd.cut(b["p_mkt"], BUCKETS)

    g = b.groupby("bucket", observed=True).agg(
        n=("hit", "size"), price=("p_mkt", "mean"), realized=("hit", "mean"),
        spread=("spread", "mean"),
    )
    g["gross_edge"] = g["price"] - g["realized"]  # >0: contract trades rich
    g["se"] = np.sqrt(g["realized"] * (1 - g["realized"]) / g["n"])
    g["z"] = g["gross_edge"] / g["se"]
    g["half_spread"] = g["spread"] / 2
    g["fee"] = fee_rate * g["price"] * (1 - g["price"])
    # Taking the mispriced side means crossing the spread and paying fees.
    g["net_edge"] = g["gross_edge"].abs() - g["half_spread"] - g["fee"]
    return g


def run() -> pd.DataFrame:
    from . import backtest, ingest

    markets = backtest._weather_events(ingest.load_markets())
    candles = ingest.load_candles()
    _, bins = backtest._score(markets, candles, backtest.STUDY, backtest.WEIGHTED)
    g = calibration(bins)

    cols = ["n", "price", "realized", "gross_edge", "z", "spread", "fee", "net_edge"]
    REPORTS_DIR.mkdir(exist_ok=True)
    g[cols].round(5).to_csv(REPORTS_DIR / "price_bias.csv")
    print(g[cols].round(4).to_string())

    rich = g[g["z"] > 2]
    cheap = g[g["z"] < -2]
    survive = g[g["net_edge"] > 0]
    print(f"\nbuckets priced significantly rich (z>2):  {len(rich)}")
    print(f"buckets priced significantly cheap (z<-2): {len(cheap)}")
    print(f"buckets whose |edge| survives spread + fees: {len(survive)}")
    if len(survive):
        print(survive[["price", "gross_edge", "net_edge"]].round(4).to_string())
    return g
