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

## Second result: the market is already sharp at the open

Same design swept across earlier decision times (`predictedge sweep`). Day-before snapshots use the 2-day-lead forecast and a 2-day error lag, so every input stays strictly pre-snapshot:

| snapshot | lead | events | Brier model | Brier market | ΔBrier | 95% CI | p |
|---|---|---|---|---|---|---|---|
| D−1 16:00Z (open + 2h) | 2d | 404 | 0.765 | 0.647 | +0.118 | [+0.089, +0.148] | < 0.0001 |
| D−1 21:00Z | 2d | 408 | 0.765 | 0.636 | +0.128 | [+0.097, +0.160] | < 0.0001 |
| D 01:00Z | 1d | 408 | 0.768 | 0.626 | +0.142 | [+0.109, +0.176] | < 0.0001 |
| D 05:00Z | 1d | 408 | 0.768 | 0.602 | +0.166 | [+0.130, +0.203] | < 0.0001 |
| D 09:00Z (primary) | 1d | 407 | 0.765 | 0.581 | +0.184 | [+0.147, +0.220] | < 0.0001 |

Two findings. First, there is **no soft window**: two hours after these markets open, with thin books, they already beat the public-NWP baseline decisively. Second, the market's Brier improves monotonically as the event approaches (0.647 → 0.581) while the baseline's stays flat — the widening gap is a direct measurement of information flowing into the price over the ~17 hours before the event day starts. Whatever traders are using (fresher model runs, human synthesis), they price it in early and keep accruing it.

## Third result: 39% of the "market edge" was our own weak input

The obvious confound in the first two results: was the market genuinely smarter, or was our forecast just under-powered? Open-Meteo's default `best_match` turns out to track **GFS exactly** — the baseline was one model, not a considered choice. Averaging six independent NWP systems (`ecmwf_ifs025`, `gfs`, `icon`, `gem`, `jma`, `meteofrance`), **all drawn from the same `previous_day1` archive**, changes the estimate without changing the information timing by a single minute.

| variant | forecast MAE | Brier | log loss | ΔBrier vs market |
|---|---|---|---|---|
| baseline (single NWP) | 2.33°F | 0.767 | 1.597 | +0.183 |
| ensemble (6-model mean) | **1.91°F** | **0.696** | **1.361** | **+0.112** |

The ensemble improvement is real and significant: ΔBrier **−0.071** (95% CI [−0.099, −0.042], DM −4.66, p < 0.0001), Δlog loss −0.235. It closes **39% of the Brier gap** and 45% of the log-loss gap — with zero new information, purely by not being sloppy about the estimator.

**But the market still wins decisively**: +0.112 Brier (CI [+0.088, +0.136], DM 7.86, p < 0.0001). So the honest conclusion splits in two — a large share of what looked like trader skill was our own measurement weakness, and a large, unambiguous remainder is not.

This is why the confound mattered. Reporting result #1 as "the market is smart" would have been partly wrong.

> Methodology note: the primary specification (single NWP, D 09:00Z) stays pre-registered and unchanged; the ensemble is reported as a disclosed follow-up experiment, not a retroactive redefinition of the baseline. Both are in git.

## The improvement ladder — including what didn't work

Each variant differs from its parent by exactly one change, and is scored on identical events, bins and snapshot (`predictedge compare`):

| variant | MAE | Brier | log loss | vs parent (Brier) | significant? |
|---|---|---|---|---|---|
| baseline (single NWP) | 2.33°F | 0.7674 | 1.5966 | — | — |
| ensemble (6-model mean) | 1.91°F | 0.6964 | 1.3614 | −0.0709, p < 0.0001 | ✅ |
| + spread-conditional σ | 1.91°F | 0.6882 | 1.3377 | −0.0083, p = 0.003 | ✅ |
| + skill-weighted members | **1.72°F** | **0.6787** | **1.3086** | −0.0095, p = 0.031 | ✅ |
| + empirical residual shape | 1.72°F | 0.6782 | 1.3005 | −0.0005, p = 0.78 | ❌ |
| + pooled member skill | 1.76°F | 0.6774 | 1.3091 | −0.0012, p = 0.71 | ❌ |

**What worked.** Averaging six NWP systems; widening σ on days the members disagree (fit walk-forward as var = a + b·spread²); weighting members by inverse walk-forward MSE (skill is very uneven — ICON RMSE 2.52 vs ECMWF 4.32).

**What didn't, and is reported anyway.** Residuals are *significantly* non-normal (skew −0.71, excess kurtosis 1.41, D'Agostino p = 1.8e-10), so we replaced the Normal with a walk-forward empirical residual CDF — and it changed nothing measurable (p = 0.78). Real misspecification, immaterial consequence. Pooling member skill across cities also did nothing (p = 0.71) and slightly hurt MAE. Both stay in the variant table rather than being quietly dropped.

## Sixth result: how much of the gap is skill, and how much is access?

Re-running the decision-time sweep with the improved model separates the two:

| snapshot | model Brier | market Brier | ΔBrier |
|---|---|---|---|
| D−1 16:00Z (open + 2h) | 0.716 | 0.650 | +0.066 |
| D−1 21:00Z | 0.717 | 0.639 | +0.078 |
| **D 01:00Z** | 0.682 | 0.629 | **+0.054** |
| D 05:00Z | 0.682 | 0.604 | +0.078 |
| D 09:00Z (primary) | 0.679 | 0.584 | +0.094 |

The model's Brier is **flat at 0.682** from 01:00Z to 09:00Z — it is frozen at the day-ahead forecast — while the market improves 0.629 → 0.584. The gap is narrowest exactly where the information vintages are most comparable, and widens precisely during the hours when 00Z/06Z runs arrive.

So the +0.094 primary gap decomposes roughly as:

- **~0.054 — genuine remaining edge** at comparable information vintage. Real, and we have not closed it.
- **~0.040 — information access.** Open-Meteo's archive exposes no lead shorter than `previous_day1`; traders at 09:00Z are using runs we cannot retrieve historically. This is a limitation of our data, not evidence about traders.

That distinction matters: roughly 43% of what remains isn't a modeling failure at all, it's an archive we don't have.

## Dashboard

**[jonahxa.github.io/predictedge](https://jonahxa.github.io/predictedge/)**

The public site renders the committed research record — reports and pre-registered forecasts already in git. It never recomputes anything at build time, so what's published is exactly what's in the repo. `predictedge export` writes `dashboard/public/data.json`; `.github/workflows/pages.yml` builds the Next.js static export and deploys it on every push.

## Reproduce

```
pip install -e .
predictedge ingest         # archive settled markets + candlesticks (cached, throttled)
predictedge backtest       # walk-forward model vs market
predictedge evaluate       # aggregate Brier / log loss table
predictedge significance   # clustered bootstrap + Diebold-Mariano
predictedge sweep          # gap across decision-time snapshots
predictedge forecast       # issue pre-registered forecasts for open markets
predictedge export         # write dashboard data.json
```

## Guardrails

No trading, no wagering, no accounts, no API keys. Public unauthenticated endpoints only, cached and rate-limited. This is market-efficiency research, not financial advice.
