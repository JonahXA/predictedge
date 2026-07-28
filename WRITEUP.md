# I tried four ways to beat prediction markets. Most of the gap was my own sloppiness.

Prediction markets were supposed to be the soft target.

I'd just finished a project called ClosingLine, which asked whether public data could beat the sharpest number in sports betting — the closing line. The answer was no: my model got within ~2% of the market's Brier score, and the remaining gap was statistically significant. Fine. That's a *sharp* market, pounded into shape by enormous informed money.

Prediction markets looked different. Thinner books. More recreational money. Kalshi runs daily contracts on things with real statistical structure — will the high in NYC exceed 86°F today? — where a decent weather model should have something to say. The bet was that nobody had sharpened those yet.

So I built the instrument to find out, pre-registered, before risking anything. Every forecast committed to git before the event resolved. Brier score and log loss against the market's own de-vigged price. Walk-forward, with a unit test proving that adding future events can't change past features. Paired bootstrap plus Diebold-Mariano with Newey-West variance on every claim.

Then I pointed it at 40 daily-temperature series across 20 US cities.

## The market won, decisively

First result, 413 city-day events at a fixed 09:00 UTC decision snapshot:

| | model | market |
|---|---|---|
| Brier | 0.7674 | **0.5845** |
| log loss | 1.5966 | **1.0699** |

ΔBrier **+0.183**, 95% CI [+0.147, +0.220], DM 7.94, p < 0.0001. The market beat me in all six cities I started with. Its modal bin was right 52% of the time against my 37%.

I also swept earlier decision times to find the soft window — surely two hours after these markets open, on thin books, before anyone's paying attention? No. The market led at every snapshot, including that one.

That could have been the whole post: *markets efficient, model loses, film at eleven.* Except the obvious confound was still sitting there.

## Then I checked whether it was actually their skill

My model's only weather input was a ~24-hour-old forecast run. Traders at the snapshot had fresher data. Was the market genuinely smarter, or was my input just weak?

Two things turned up when I looked.

First, Open-Meteo's default `best_match` model — which I'd been using without thinking — tracks GFS exactly. My "baseline NWP" was one model, chosen by default rather than by decision.

Second, and worse: I'd been about to use a variable called `temperature_2m_previous_day0`, assuming it meant a same-day forecast run. It doesn't exist. The API silently returns plain `temperature_2m` instead — the *latest analysis*, which incorporates what actually happened. It would have leaked the outcome straight into my features and produced a beautiful, entirely fake edge.

So I averaged ten independent NWP systems — ECMWF, GFS, ICON, GEM, JMA, Météo-France, UKMO, ECMWF-AIFS, CMA, KNMI — all pulled from the *same* day-ahead archive. Identical information timing. Only a better estimator.

(A trap here too: several Open-Meteo model IDs are aliases returning byte-identical data. `gfs_hrrr` and `ncep_hrrr_conus` both return `gfs_seamless`. Adding them naively would have silently double-weighted GFS and shrunk the apparent ensemble spread. Members get de-duplicated, with a test pinning it.)

Then I weighted members by inverse walk-forward MSE — member skill is wildly uneven, ICON at 2.52°F RMSE against ECMWF's 4.32 — and widened the predictive distribution on days the members disagreed.

| variant | forecast MAE | Brier |
|---|---|---|
| baseline (single NWP) | 2.33°F | 0.7674 |
| **10-model, skill-weighted** | **1.68°F** | **0.6639** |

That closed **57% of the gap** — with zero new information. Purely by not being sloppy.

**The market still won**: +0.079 Brier, CI [+0.056, +0.104], p < 0.0001. But the conclusion had split in two. Most of what looked like trader skill was my own measurement weakness. Reporting the first result as "the market is smart" would have been substantially wrong.

## How much of what's left is skill, and how much is access?

Re-running the decision-time sweep with the better model separated them:

| snapshot | model Brier | market Brier | ΔBrier |
|---|---|---|---|
| D−1 16:00Z (open + 2h) | 0.714 | 0.650 | +0.064 |
| **D 01:00Z** | 0.667 | 0.628 | **+0.038** |
| D 05:00Z | 0.667 | 0.604 | +0.063 |
| D 09:00Z | 0.664 | 0.585 | +0.079 |

My model's Brier is **flat at 0.667** from 01:00Z to 05:00Z — it's frozen at the day-ahead forecast — while the market improves 0.628 → 0.604 → 0.585. The gap is narrowest where the information vintages match, and widens precisely during the hours when 00Z and 06Z model runs land.

So the +0.079 splits roughly: **~0.038 genuine remaining edge** at matched vintage (still p = 0.009), and **~0.041 information I can't retrieve** — Open-Meteo's archive exposes no lead shorter than one day. About half of what remained wasn't a modeling failure at all. It was an archive I don't have.

## The thesis, finally tested properly

Here's the thing I'd been avoiding: all of that was measured on the *most heavily traded* weather series on the exchange. The least favourable place to test a thesis about thin, recreational markets.

So: 40 series, 2,725 events, spanning a **64× range in traded volume** (1,349 to 86,013 per contract). Each city's daily *low* market shares a station, a day, and an NWS resolution source with its *high* market — but trades ~6× thinner. Differencing within a city removes geography and forecast difficulty entirely.

One trap worth naming, because it would have inverted the result: daily lows occur just before dawn. A fixed 09:00 UTC snapshot is already 05:00 in New York — the low may have *already happened*, so the market would partly know the outcome while my day-ahead forecast wouldn't. That would have made thin markets look artificially sharp. Everything moved to a local-midnight snapshot, putting the whole local day ahead of the decision in every time zone.

The liquidity gradient is unambiguously real — thinner markets quote much wider spreads (r = −0.81, p < 0.0001; 11–15¢ against 3–4¢).

It buys no accuracy whatsoever:

| test | effect | 95% CI | p |
|---|---|---|---|
| ΔBrier ~ log₁₀(volume), 40 series | +0.014/decade | [−0.012, +0.040] | 0.27 |
| **paired within-city (high − low)** | **+0.012** | [−0.009, +0.033] | 0.25 |

The aggregate direction is, if anything, *opposite* to the thesis: thin low-markets show a smaller gap (+0.053) than thick high-markets (+0.065), at near-identical forecast difficulty.

**A contract trading 1,349 prices temperature about as well as one trading 86,013.**

## The pricing is wrong. It doesn't matter.

One more idea: forget beating their forecast. Is their *pricing* biased in a way that needs no forecasting edge at all?

Across 16,350 bin-observations, a textbook favourite–longshot bias, perfectly monotonic:

| market price | realized | z |
|---|---|---|
| 0.95% | **0.49%** | **+4.1** |
| 3.3% | **1.9%** | **+5.3** |
| 41.5% | **44.2%** | **−2.5** |
| 57.7% | **61.8%** | **−2.3** |

Longshots trade rich, favourites trade cheap. This is the strongest statistical signal in the entire project.

Then subtract what it costs to act on it — half the bid-ask spread, plus exchange fees:

| price | gross edge | half-spread | fee | **net** |
|---|---|---|---|---|
| 0.95¢ | +0.46¢ | 0.66¢ | 0.07¢ | **−0.26¢** |
| 3.3¢ | +1.38¢ | 1.12¢ | 0.22¢ | **+0.04¢** |
| 41¢ | −2.77¢ | 2.53¢ | 1.70¢ | **−1.46¢** |

**One bucket of eight survives, by 0.04¢ on a 3.3¢ contract.** That's one marginal positive out of eight tests, found in-sample, resting on a fee estimate I couldn't verify, priced against a mid you can't actually trade at, and adversely selected in exactly the thin books where it appears.

It's the noise floor, not an edge.

This is what efficiency looks like from the inside: **the price is measurably, highly significantly wrong — by less than the cost of correcting it.** Publishing the top table without the bottom one would have been the most misleading thing this project could do.

## What I'd take from it

**Four attempts, four closed doors**: beat the forecast, find a soft decision window, find soft thin markets, exploit a pricing bias. Those aren't four scattered nulls; they're one coherent finding.

But the result I actually care about is the second one. **57% of the market's apparent edge was mine, not theirs.** I'd built the whole apparatus — pre-registration, significance tests, causality proofs — to catch the market being wrong. What it caught was me.

The trap-hunting was worth more than the modeling. A silently-nonexistent API variable would have manufactured a fake edge. Aliased model IDs would have quietly double-weighted GFS. A pre-dawn snapshot would have inverted the thin-market result. Sample counts mistaken for census counts made crowded categories look empty. Every one of those produces a confident, wrong answer that looks exactly like a real one.

Four honest negatives cost a day. A model I believed in would have cost considerably more.

---

*Code, data and the full result set: [github.com/JonahXA/predictedge](https://github.com/JonahXA/predictedge) · live dashboard: [jonahxa.github.io/predictedge](https://jonahxa.github.io/predictedge/). Everything is paper measurement — no trading, no positions, ever.*
