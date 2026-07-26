# PredictEdge

Pre-registered probabilistic forecasting research on **prediction markets** — Kalshi's daily temperature contracts. The question: can honest, public-data modeling beat a market that isn't sharp yet?

**The answer, measured four independent ways, is no.** Kalshi's weather complex is efficient — at every decision time we can observe, across a 64× range of traded volume, and even where its own prices are provably biased. The founding bet that thin, recreational money leaves these markets unsharpened is answered, and it is wrong. The four negative results, and the ~57% of the apparent "market edge" that turned out to be *our own* measurement weakness, are the contribution.

**No money is ever involved.** Everything here is paper measurement: forecasts are committed to git before events resolve, then scored against the market's own price. The gap to the market — positive or negative — is the research result.

| finding | result |
|---|---|
| Beat the market's forecast | 13.6% behind on Brier at the decision point, p < 0.0001 |
| Find a soft decision window | market leads at all 5 snapshots, including 2h after open |
| Find soft thin markets | no attention effect over 64× volume, within-city control |
| Exploit a pricing bias | favourite-longshot bias real (z = 5.3) but smaller than costs |

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

## Third result: over half the "market edge" was our own weak input

The obvious confound in the first two results: was the market genuinely smarter, or was our forecast just under-powered? Open-Meteo's default `best_match` turns out to track **GFS exactly** — the baseline was one model, not a considered choice. Averaging six independent NWP systems (`ecmwf_ifs025`, `gfs`, `icon`, `gem`, `jma`, `meteofrance`), **all drawn from the same `previous_day1` archive**, changes the estimate without changing the information timing by a single minute.

| variant | forecast MAE | Brier | log loss | ΔBrier vs market |
|---|---|---|---|---|
| baseline (single NWP) | 2.33°F | 0.767 | 1.597 | +0.183 |
| best model (see ladder) | **1.68°F** | **0.664** | **1.252** | **+0.079** |

The ensemble improvement alone is ΔBrier **−0.082** (95% CI [−0.112, −0.051], DM −5.15, p < 0.0001). With the rest of the ladder the model closes **57% of the Brier gap** — with zero new information, purely by not being sloppy about the estimator.

**But the market still wins decisively**: +0.079 Brier (CI [+0.056, +0.104], DM 4.82, p < 0.0001). So the honest conclusion splits in two — most of what looked like trader skill was our own measurement weakness, and a significant remainder is not.

This is why the confound mattered. Reporting result #1 as "the market is smart" would have been partly wrong.

> Methodology note: the primary specification (single NWP, D 09:00Z) stays pre-registered and unchanged; the ensemble is reported as a disclosed follow-up experiment, not a retroactive redefinition of the baseline. Both are in git.

## The improvement ladder — including what didn't work

Each variant differs from its parent by exactly one change, and is scored on identical events, bins and snapshot (`predictedge compare`):

| variant | MAE | Brier | log loss | vs parent (Brier) | significant? |
|---|---|---|---|---|---|
| baseline (single NWP) | 2.33°F | 0.7674 | 1.5966 | — | — |
| ensemble (10-model mean) | 1.88°F | 0.6858 | 1.3247 | −0.0815, p < 0.0001 | ✅ |
| + spread-conditional σ | 1.88°F | 0.6743 | 1.2852 | −0.0115, p < 0.0001 | ✅ |
| + skill-weighted members | **1.68°F** | **0.6639** | **1.2515** | −0.0104, p = 0.022 | ✅ |
| + empirical residual shape | 1.68°F | 0.6648 | 1.2528 | +0.0009, p = 0.72 | ❌ |
| + pooled member skill | 1.76°F | 0.6679 | 1.2683 | +0.0040, p = 0.19 | ❌ |
| + linear calibration | 1.68°F | 0.6650 | 1.2503 | +0.0011, p = 0.89 | ❌ |
| + time-lagged members | 1.70°F | 0.6733 | 1.2827 | +0.0093, p = 0.011 | ❌ **worse** |

**What worked.** Averaging ten NWP systems (ECMWF-IFS, GFS, ICON, GEM, JMA, Météo-France, UKMO, ECMWF-AIFS, CMA, KNMI); widening σ on days the members disagree (fit walk-forward as var = a + b·spread²); weighting members by inverse walk-forward MSE (skill is very uneven — ICON RMSE 2.52 vs ECMWF 4.32).

A trap worth naming: several Open-Meteo model ids are **aliases that return byte-identical series** (`gfs_hrrr` and `ncep_hrrr_conus` both return `gfs_seamless`; `arpege_world` returns Météo-France; `dmi`/`metno` return KNMI). Adding them naively would silently double-weight a model and shrink the apparent ensemble spread, so members are de-duplicated on ingest and a test pins that behavior.

**What didn't, and is reported anyway.** Four attempts in a row failed:

- **Empirical residual shape.** Residuals are *significantly* non-normal (skew −0.71, excess kurtosis 1.41, D'Agostino p = 1.8e-10), so we replaced the Normal with a walk-forward empirical residual CDF. It changed nothing measurable (p = 0.72). Real misspecification, immaterial consequence.
- **Pooled member skill** across cities: nothing (p = 0.19), slightly worse MAE.
- **Linear calibration** (truth ~ a + b·forecast, to catch NWP compressing extremes): nothing (p = 0.89). The slope has nothing left to correct once the ensemble and bias shift are in.
- **Time-lagged members** (adding the previous run cycle): **significantly worse** (+0.0093 Brier, p = 0.011). Weights shrink toward equal, so consistently staler members can't be discounted enough to pay for themselves.

Four consecutive failures — three null, one harmful — is itself the finding: **estimation is plateaued.** Everything cheap and legitimate has been tried, and the model has stopped moving. What remains is not technique.

## Sixth result: how much of the gap is skill, and how much is access?

Re-running the decision-time sweep with the improved model separates the two:

| snapshot | model Brier | market Brier | ΔBrier |
|---|---|---|---|
| D−1 16:00Z (open + 2h) | 0.714 | 0.650 | +0.064 |
| D−1 21:00Z | 0.714 | 0.639 | +0.075 |
| **D 01:00Z** | 0.667 | 0.629 | **+0.038** |
| D 05:00Z | 0.667 | 0.604 | +0.063 |
| D 09:00Z (primary) | 0.664 | 0.584 | +0.079 |

The model's Brier is **flat at 0.667** from 01:00Z to 05:00Z — it is frozen at the day-ahead forecast — while the market improves 0.629 → 0.604 → 0.584. The gap is narrowest exactly where the information vintages are most comparable, and widens precisely during the hours when 00Z/06Z runs arrive.

So the +0.079 primary gap decomposes roughly as:

- **~0.038 — genuine remaining edge** at comparable information vintage (still significant, p = 0.009). Real, and we have not closed it.
- **~0.041 — information access.** Open-Meteo's archive exposes no lead shorter than `previous_day1`; traders at 09:00Z are using runs we cannot retrieve historically. This is a limitation of our data, not evidence about traders.

That distinction matters: roughly **half of what remains isn't a modeling failure at all**, it's an archive we don't have. Of the original +0.183 baseline gap, 57% was our estimator, ~22% is information access, and ~21% is a genuine edge we have not explained.

## Eighth result: market sharpness is not bought with attention

The first three results were all measured on the most heavily traded weather series on the exchange — the least favourable place to test a thesis about *thin, recreational* markets. So we ran the identical model against **40 series (20 cities × daily high and daily low), 2,725 events, spanning a 64× range in traded volume** (1,349 → 86,013 per market).

The liquidity gradient is unambiguously real:

> thinner markets quote **much** wider — `mean_spread ~ log10(volume)`, r = **−0.81**, p < 0.0001. Low-temperature markets quote 11–15¢ spreads against 3–4¢ for the thick high markets.

It buys no accuracy at all:

| test | effect | 95% CI | p |
|---|---|---|---|
| ΔBrier ~ log10(volume), all 40 series | +0.014 / decade | [−0.012, +0.040] | 0.27 |
| high-temperature series only | −0.002 | [−0.050, +0.046] | 0.93 |
| low-temperature series only | +0.041 | [−0.052, +0.133] | 0.37 |
| **paired within city (high − low)** | **+0.012** | [−0.009, +0.033] | 0.25 |
| sensitivity: tight-spread events only | +0.010 | [−0.018, +0.038] | 0.47 |

The paired design is the strongest piece: a city's high and low markets share a station, a day and an NWS resolution source, and differ ~6× in volume, so geography and forecast difficulty cancel out. The aggregate direction is if anything *opposite* to the thesis — thin low markets show a smaller gap (+0.053) than thick high markets (+0.065), at near-identical forecast difficulty (MAE 1.81 vs 1.79°F).

Slope CIs are reported rather than bare p-values, so this states what it rules out: across the full observed range the effect is at most ~0.07 Brier and may be zero or negative. **A market trading 1,349 per contract prices temperature about as well as one trading 86,013.**

Two design decisions this result depends on, both of which would have silently broken it:

- **Daily lows happen before dawn.** A fixed 09:00 UTC snapshot is already 05:00 in New York, so the low may have *occurred* — the market would partly know the outcome while a day-ahead forecast would not, making thin low markets look artificially sharp. The study snapshots at **local midnight**, putting the whole local day ahead of the decision everywhere.
- **Thin markets quote wider**, so the headline backtest's ≤15¢ filter would preferentially discard the very series under study. The study reports all quoted events and carries the restricted sample as a disclosed sensitivity.

## Ninth result: the price is measurably wrong — by less than the cost of fixing it

Every result above asks whether our *forecast* beats the market's. A different question: is the market's *pricing* biased in a way that needs no forecasting edge to exploit? The classic form is the favourite-longshot bias.

It is present, and it is the strongest statistical signal in the project (16,350 bin-observations):

| market price | realized | z |
|---|---|---|
| 0.95% | **0.49%** | **+4.1** |
| 3.3% | **1.9%** | **+5.3** |
| 7.2% | 5.6% | +2.9 |
| 14.8% | 13.1% | +2.2 |
| 41.5% | **44.2%** | **−2.5** |
| 57.7% | **61.8%** | **−2.3** |

Perfectly monotonic: longshots trade rich, favourites trade cheap.

Then subtract what it costs to act on it — half the bid-ask spread, plus exchange fees:

| price | gross edge | half-spread | fee | **net** |
|---|---|---|---|---|
| 0.95¢ | +0.46¢ | 0.66¢ | 0.07¢ | **−0.26¢** |
| **3.3¢** | +1.38¢ | 1.12¢ | 0.22¢ | **+0.04¢** |
| 7.2¢ | +1.55¢ | 1.52¢ | 0.47¢ | **−0.44¢** |
| 41¢ | −2.77¢ | 2.53¢ | 1.70¢ | **−1.46¢** |

**One bucket of eight survives, by 0.04¢ on a 3.3¢ contract** — and that survivor is one marginal positive out of eight tests, discovered in-sample, resting on an unverified fee placeholder, priced against a mid you cannot actually trade at, and subject to adverse selection in exactly the thin books where it appears. It is the noise floor, not an edge.

This is what efficiency looks like from the inside: **the price is wrong in a highly significant way, by less than the cost of correcting it.** Reporting the gross bias without the net column would have been the most misleading thing this project could publish.

## Where this leaves the thesis

Four independent attacks, all closed:

1. **Beat the forecast** — 13.6% behind on Brier at the decision point (p < 0.0001), after closing 57% of the original gap.
2. **Find a soft decision window** — the market leads at all five snapshots, including two hours after open.
3. **Find soft thin markets** — no attention effect across a 64× volume range, with a within-city control.
4. **Exploit a pricing bias** — real and highly significant, but sub-cost.

These are not four scattered nulls; they are one coherent finding. **Kalshi's weather complex is efficient.** The founding bet — that thinner, more recreational money leaves these markets unsharpened — is answered, and the answer is no.

The economics series remain structurally the most attractive target, because a scheduled BLS/BEA release has no overnight model run to lose a race to. But a survey of every daily and weekly Economics series found the best candidate has **58 resolved events** against the 2,725 that got weather to p < 0.0001. That study is premature, not impossible, so those series are now archived daily and the question is deferred rather than guessed at.

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
