# PredictEdge

Pre-registered probabilistic forecasting research on **prediction markets** — starting with Kalshi's daily high-temperature contracts. The question: can honest, public-data modeling beat a market that isn't sharp yet?

**No money is ever involved.** Everything here is paper measurement: forecasts are committed to git before events resolve, then scored against the market's own price. The gap to the market — positive or negative — is the research result.

## Thesis, and how this differs from ClosingLine

[ClosingLine](https://github.com/JonahXA/closingline) asked whether public data could beat the sharpest number in sports betting — the closing line — and the honest answer was **no**: the model got within ~2% of the market's Brier score, and the remaining gap was statistically significant. That was a finding about *sharp* markets.

Prediction markets are the same modeling problem against **thinner, more recreational money**, on recurring events with real statistical structure (daily weather, economic data releases). The bet is that the market's forecast is beatable there — or at least that the gap is small enough to matter. Either way, we measure it the same way ClosingLine did:

- **Pre-registration.** Every forecast is committed with a timestamp before the event resolves; git history is the tamper-evident record. Issued forecasts are never overwritten.
- **Proper scoring.** Brier score and log loss, reported together.
- **The market is the benchmark.** De-vigged bin mid-prices at a fixed decision snapshot, scored on identical events and bins.
- **Strict walk-forward.** A forecast may only use information available before the snapshot; a unit test proves that adding future events never changes past features.
- **Significance-test every claim.** Paired bootstrap (clustered by date, since same-day weather outcomes correlate across cities) plus Diebold-Mariano with Newey-West HAC variance. A gap whose CI crosses zero is not a finding.
- **Negatives are reported as carefully as positives.**

## The data (recon findings, July 2026)

All sources are public and unauthenticated:

- **Kalshi trade API** (`api.elections.kalshi.com/trade-api/v2`): settled markets include the outcome (`result`) *and* the officially measured value (`expiration_value`, e.g. the NWS high temperature). Hourly candlesticks (trade OHLC + yes bid/ask) are public for every market.
- **Retention caveat:** the public listing only reaches back **~2 months** — older markets 404 even by direct ticker. `predictedge ingest` therefore merges every visible settled market into a committed parquet archive (`data/archive/`); the archive grows daily and never drops rows. This repo *is* the durable record.
- **Open-Meteo** previous-runs API: archived day-ahead forecasts (`temperature_2m_previous_day1`), i.e. what the weather model said *yesterday* about today — the leak-free model input.
- **Polymarket** (Gamma + CLOB `prices-history`) is also publicly reachable and archived as a future direction; its recurring-event catalog is weaker, so Kalshi goes first.

**Contract type #1 — daily high temperature**, six series with settled history and real statistical structure: NYC (Central Park), Chicago (Midway), Miami, Austin (Camp Mabry), Denver, Los Angeles (LAX). Each day is one mutually exclusive event of ~6 temperature bins; exactly one bin resolves yes. Economic-indicator series (CPI, payrolls, Fed) don't have enough visible settled history yet — they're archived forward for a later study.

## Study design

For each city-day event:

1. **Model**: the day's high is `Normal(mu, sigma)`, where `mu` = Open-Meteo day-ahead forecast plus a walk-forward bias correction (shrunk mean of the city's past forecast-vs-official errors), and `sigma` = walk-forward error spread shrunk toward a 3°F prior. Bin probabilities integrate this density over the strike bins (continuity-corrected — settlement is in whole °F).
2. **Market**: de-vigged (sum-normalized) bin mids from the last hourly candle at or before **09:00 UTC** on the event day (05:00 New York, 02:00 Los Angeles — before the day's high develops anywhere).
3. **Score** both vectors against the settled outcome: multiclass Brier and log loss.
4. The **primary sample** requires every bin two-sidedly quoted with spread ≤ 15¢ at the snapshot; all-quoted-events is reported as a sensitivity.

Causality notes: the walk-forward error state for day D uses only days strictly before D (a day's high is publicly observable from hourly obs by local midnight, hours before the snapshot). The day-ahead forecast for D was issued on D−1. `tests/test_causality.py` proves the walk-forward state is invariant to future data.

## First result (2026-07-24): the market wins, decisively

408 city-day events (six cities × 68 days, 2026-05-17 → 2026-07-23), walk-forward, every event scored for both forecasters on identical bins:

| sample | events | Brier model | Brier market | log loss model | log loss market |
|---|---|---|---|---|---|
| primary (all bins quoted, spread ≤ 15¢) | 407 | 0.765 | **0.581** | 1.583 | **1.066** |
| all quoted events | 408 | 0.765 | **0.581** | 1.583 | **1.065** |

Significance of the gap (positive = market better), clustered by date:

| metric | mean diff | 95% CI (bootstrap) | DM stat | p |
|---|---|---|---|---|
| Brier | +0.184 | [+0.148, +0.221] | 7.94 | < 0.0001 |
| log loss | +0.517 | [+0.417, +0.621] | 8.90 | < 0.0001 |

The market beats the day-ahead-NWP baseline in every one of the six cities, and picks the correct bin as its modal outcome 52% of the time vs the model's 37%. **This gap is real, not noise — and it is the honest starting line.**

Why the baseline loses, and why that's informative: its only weather input is a ~24-hour-old forecast run (`previous_day1`, day-ahead MAE ≈ 2.3°F), while traders at the 09:00 UTC snapshot have overnight model runs (00Z) plus human synthesis. Kalshi's weather markets, at least at this snapshot time, are **not** the soft target the thesis hoped for — they price in fresher information than a naive public baseline does. The measured gap (ΔBrier 0.18) is now the budget any model improvement has to close: fresher NWP inputs at the snapshot, multi-model ensembles, and earlier snapshots (where the market has had less time to sharpen) are the next pre-registered experiments.

Alongside the backtest, `predictedge forecast` (run daily by CI) issues live day-ahead forecasts for open markets into `forecasts/weather.csv`, append-only — the commit timestamp before resolution is the pre-registration.

## Reproduce

```
pip install -e .
predictedge ingest         # archive settled markets + candlesticks (cached, throttled)
predictedge backtest       # walk-forward model vs market
predictedge evaluate       # aggregate Brier / log loss table
predictedge significance   # clustered bootstrap + Diebold-Mariano
```

## Guardrails

No trading, no wagering, no accounts, no API keys. Public unauthenticated endpoints only, cached and rate-limited. This is market-efficiency research, not financial advice.
