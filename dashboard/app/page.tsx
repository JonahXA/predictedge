import raw from "../public/data.json";
import DailyBrier from "../components/DailyBrier";
import Reliability from "../components/Reliability";
import SnapshotSweep from "../components/SnapshotSweep";

type Eval = {
  sample: string;
  events: number;
  brier_model: number;
  brier_market: number;
  logloss_model: number;
  logloss_market: number;
};

type City = Eval & { city: string };

type Sig = {
  comparison: string;
  sample: string;
  metric: string;
  events: number;
  dates: number;
  mean_diff: number;
  ci_low: number;
  ci_high: number;
  p_bootstrap: number;
  dm_stat: number;
  p_dm: number;
};

type Sweep = {
  snapshot: string;
  lead_days: number;
  events: number;
  brier_model: number;
  brier_market: number;
  d_brier: number;
  ci_low: number;
  ci_high: number;
  dm_stat: number;
  p_dm: number;
};

type Forecast = {
  issue_ts: string;
  city: string;
  event_ticker: string;
  event_date: string;
  ticker: string;
  strike_type: string;
  floor: number | null;
  cap: number | null;
  p_model: number;
  mu: number;
  sigma: number;
  yes_bid: number | null;
  yes_ask: number | null;
  result?: string;
  expiration_value?: number;
};

const data = raw as unknown as {
  generated_at: string;
  cities: Record<string, string>;
  backtest?: {
    events: number;
    start: string;
    end: string;
    forecast_mae: number;
    daily: { date: string; model_brier: number; market_brier: number; n: number }[];
  };
  evaluation?: Eval[];
  by_city?: City[];
  significance?: Sig[];
  sweep?: Sweep[];
  variants?: {
    variant: string;
    events: number;
    brier_model: number;
    brier_market: number;
    logloss_model: number;
    logloss_market: number;
    forecast_mae: number;
  }[];
  variant_significance?: {
    comparison: string;
    metric: string;
    events: number;
    mean_diff: number;
    ci_low: number;
    ci_high: number;
    dm_stat: number;
    p_dm: number;
  }[];
  reliability?: { series: string; bin_mid: number; predicted: number; observed: number; n: number }[];
  modal_hit?: { model: number; market: number };
  live: { open: Forecast[]; resolved: Forecast[]; summary: { events: number; brier_model: number; brier_market: number } | null };
};

const pct = (v: number) => `${Math.round(v * 100)}%`;
const p4 = (v: number) => v.toFixed(4);
const pval = (v: number) => (v < 0.0001 ? "< 0.0001" : v.toFixed(4));

/** "81° or below", "82–83°", "above 86°" from a strike row. */
function strikeLabel(f: Forecast) {
  if (f.strike_type === "less") return `${(f.cap ?? 0) - 1}° or below`;
  if (f.strike_type === "greater") return `${(f.floor ?? 0) + 1}° or above`;
  return f.floor === f.cap ? `${f.floor}°` : `${f.floor}–${f.cap}°`;
}

export default function Home() {
  const bt = data.backtest;
  const primary = data.evaluation?.find((e) => e.sample.startsWith("primary"));
  const sigBrier = data.significance?.find((s) => s.metric === "brier" && s.sample === "primary");
  const gapPct = primary ? ((primary.brier_model - primary.brier_market) / primary.brier_market) * 100 : null;
  const earliest = data.sweep?.[0];

  return (
    <main>
      <div className="hero">
        <span className="badge">Paper measurement · no trading</span>
        <h1>PredictEdge</h1>
        <p className="tagline">
          Probabilistic forecasts for Kalshi daily-temperature contracts, committed to git before
          resolution and scored against the market&rsquo;s own price. The question is not whether the
          model is accurate — it is whether the model is <em>more</em> accurate than the price you
          would be trading against.
        </p>
      </div>

      {primary && bt && (
        <div className="stat-row">
          <div className="stat-tile">
            <div className="value">{bt.events}</div>
            <div className="label">
              events scored · {bt.start} → {bt.end}
            </div>
          </div>
          <div className="stat-tile">
            <div className="value">{p4(primary.brier_model)}</div>
            <div className="label">Brier — baseline model</div>
          </div>
          <div className="stat-tile">
            <div className="value" style={{ color: "var(--good)" }}>
              {p4(primary.brier_market)}
            </div>
            <div className="label">Brier — Kalshi market</div>
          </div>
          <div className="stat-tile">
            <div className="value">{gapPct ? `+${gapPct.toFixed(1)}%` : "—"}</div>
            <div className="label">model&rsquo;s deficit to the market</div>
          </div>
        </div>
      )}

      <section className="card">
        <h2>Headline finding: the market wins, decisively</h2>
        <p className="sub">
          Walk-forward, six cities, every event scored for both forecasters on identical strike bins.
        </p>
        <ul className="findings">
          {sigBrier && (
            <li>
              The market beats the day-ahead-NWP baseline by <strong>ΔBrier {sigBrier.mean_diff.toFixed(3)}</strong>{" "}
              (95% CI [{sigBrier.ci_low.toFixed(3)}, {sigBrier.ci_high.toFixed(3)}], Diebold-Mariano{" "}
              {sigBrier.dm_stat.toFixed(1)}, p {pval(sigBrier.p_dm)}). Clustered by date, because same-day
              weather outcomes correlate across cities.
            </li>
          )}
          {data.modal_hit && (
            <li>
              The market&rsquo;s modal bin is correct <strong>{pct(data.modal_hit.market)}</strong> of the
              time; the model&rsquo;s, {pct(data.modal_hit.model)}.
            </li>
          )}
          {earliest && (
            <li>
              There is <strong>no soft window</strong>: even at {earliest.snapshot}, on thin books, the
              market is already ahead (ΔBrier {earliest.d_brier.toFixed(3)}, p {pval(earliest.p_dm)}).
            </li>
          )}
          {bt && (
            <li>
              Why the baseline loses: its only input is a ~24h-old forecast run (day-ahead MAE{" "}
              <strong>{bt.forecast_mae}°F</strong>), while traders price in fresher model runs plus human
              synthesis. Kalshi weather markets are <strong>not</strong> the soft target the thesis hoped
              for.
            </li>
          )}
        </ul>
      </section>

      {data.sweep && data.sweep.length > 0 && (
        <section className="card">
          <h2>When does the market sharpen?</h2>
          <p className="sub">
            The same experiment repeated at earlier decision times. Day-before snapshots use the
            2-day-lead forecast and a 2-day error lag, so every input stays strictly pre-snapshot.
          </p>
          <SnapshotSweep rows={data.sweep} />
          <div className="table-scroll" style={{ marginTop: 18 }}>
            <table>
              <thead>
                <tr>
                  <th>Snapshot</th>
                  <th>Lead</th>
                  <th>Events</th>
                  <th>Model Brier</th>
                  <th>Market Brier</th>
                  <th>ΔBrier</th>
                  <th>95% CI</th>
                  <th>p</th>
                </tr>
              </thead>
              <tbody>
                {data.sweep.map((r) => (
                  <tr key={r.snapshot}>
                    <td>{r.snapshot}</td>
                    <td>{r.lead_days}d</td>
                    <td>{r.events}</td>
                    <td>{p4(r.brier_model)}</td>
                    <td className="win">{p4(r.brier_market)}</td>
                    <td>+{r.d_brier.toFixed(3)}</td>
                    <td>
                      [{r.ci_low.toFixed(3)}, {r.ci_high.toFixed(3)}]
                    </td>
                    <td>{pval(r.p_dm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {data.variants && data.variants.length > 1 && (
        <section className="card">
          <h2>How much of the gap was our own weak input?</h2>
          <p className="sub">
            Averaging six independent NWP systems, all drawn from the same day-ahead archive — a better
            estimate on identical information timing, not fresher data.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>Forecast MAE</th>
                  <th>Brier</th>
                  <th>Log loss</th>
                  <th>ΔBrier vs market</th>
                </tr>
              </thead>
              <tbody>
                {data.variants.map((v) => (
                  <tr key={v.variant}>
                    <td>{v.variant}</td>
                    <td>{v.forecast_mae.toFixed(2)}°F</td>
                    <td>{p4(v.brier_model)}</td>
                    <td>{p4(v.logloss_model)}</td>
                    <td>+{(v.brier_model - v.brier_market).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.variant_significance && (
            <ul className="findings" style={{ marginTop: 18 }}>
              {data.variant_significance
                .filter((s) => s.comparison.includes(" vs ") && s.comparison.includes("baseline") && !s.comparison.includes("market"))
                .map((s) => (
                  <li key={s.metric}>
                    The ensemble improves <strong>{s.metric}</strong> by{" "}
                    <strong>{Math.abs(s.mean_diff).toFixed(3)}</strong> (95% CI [{s.ci_low.toFixed(3)},{" "}
                    {s.ci_high.toFixed(3)}], DM {s.dm_stat.toFixed(2)}, p {pval(s.p_dm)}) — with zero new
                    information.
                  </li>
                ))}
              <li>
                A large share of the apparent &ldquo;market edge&rdquo; was our own measurement weakness.
                The remainder — still significant at p &lt; 0.0001 — is not.
              </li>
            </ul>
          )}
        </section>
      )}

      {bt && bt.daily.length > 0 && (
        <section className="card">
          <h2>Daily Brier, model vs market</h2>
          <p className="sub">Mean across the six cities, one point per day.</p>
          <DailyBrier rows={bt.daily} />
        </section>
      )}

      {data.by_city && (
        <section className="card">
          <h2>By city</h2>
          <p className="sub">The market wins in every market we measured.</p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>City</th>
                  <th>Events</th>
                  <th>Model Brier</th>
                  <th>Market Brier</th>
                  <th>Model log loss</th>
                  <th>Market log loss</th>
                </tr>
              </thead>
              <tbody>
                {data.by_city.map((r) => (
                  <tr key={r.city}>
                    <td>{r.city}</td>
                    <td>{r.events}</td>
                    <td className={r.brier_model <= r.brier_market ? "win" : ""}>{p4(r.brier_model)}</td>
                    <td className={r.brier_market < r.brier_model ? "win" : ""}>{p4(r.brier_market)}</td>
                    <td className={r.logloss_model <= r.logloss_market ? "win" : ""}>{p4(r.logloss_model)}</td>
                    <td className={r.logloss_market < r.logloss_model ? "win" : ""}>{p4(r.logloss_market)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {data.reliability && data.reliability.length > 0 && (
        <section className="card">
          <h2>Calibration</h2>
          <p className="sub">
            Issued probability vs realized frequency, pooled over every strike bin. On the dashed
            diagonal is perfect.
          </p>
          <Reliability rows={data.reliability} />
        </section>
      )}

      {data.significance && (
        <section className="card">
          <h2>Significance</h2>
          <p className="sub">
            Paired differentials (model − market), bootstrap resampling whole dates, plus
            Diebold-Mariano with Newey-West HAC variance. Positive means the market scored better.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Metric</th>
                  <th>Events</th>
                  <th>Dates</th>
                  <th>Mean diff</th>
                  <th>95% CI</th>
                  <th>DM</th>
                  <th>p (DM)</th>
                </tr>
              </thead>
              <tbody>
                {data.significance.map((s) => (
                  <tr key={`${s.sample}-${s.metric}`}>
                    <td>{s.sample}</td>
                    <td>{s.metric}</td>
                    <td>{s.events}</td>
                    <td>{s.dates}</td>
                    <td>+{s.mean_diff.toFixed(4)}</td>
                    <td>
                      [{s.ci_low.toFixed(3)}, {s.ci_high.toFixed(3)}]
                    </td>
                    <td>{s.dm_stat.toFixed(2)}</td>
                    <td>{pval(s.p_dm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Live pre-registered forecasts</h2>
        <p className="sub">
          Issued daily by CI for events that have not started, appended to{" "}
          <code>forecasts/weather.csv</code> and never overwritten. The commit timestamp before
          resolution is the pre-registration.
        </p>
        {data.live.summary && (
          <div className="stat-row" style={{ margin: "0 0 20px" }}>
            <div className="stat-tile">
              <div className="value">{data.live.summary.events}</div>
              <div className="label">live events resolved</div>
            </div>
            <div className="stat-tile">
              <div className="value">{p4(data.live.summary.brier_model)}</div>
              <div className="label">Brier — model (live)</div>
            </div>
            <div className="stat-tile">
              <div className="value">{p4(data.live.summary.brier_market)}</div>
              <div className="label">Brier — market (live)</div>
            </div>
          </div>
        )}
        {data.live.open.length === 0 ? (
          <p className="empty">No open forecasts right now — the next batch is issued after the daily ingest.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Event date</th>
                  <th>City</th>
                  <th>Bin</th>
                  <th>Model</th>
                  <th>Market bid/ask</th>
                  <th>Issued</th>
                </tr>
              </thead>
              <tbody>
                {data.live.open.map((f) => (
                  <tr key={f.ticker}>
                    <td>{f.event_date}</td>
                    <td>{f.city}</td>
                    <td>{strikeLabel(f)}</td>
                    <td>{pct(f.p_model)}</td>
                    <td>
                      {f.yes_bid === null || f.yes_ask === null
                        ? "—"
                        : `${Math.round(f.yes_bid * 100)}¢ / ${Math.round(f.yes_ask * 100)}¢`}
                    </td>
                    <td>{f.issue_ts.replace("T", " ").replace("+00:00", "Z")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Method</h2>
        <ul className="findings">
          <li>
            <strong>Model.</strong> The day&rsquo;s high is Normal(μ, σ): μ is the Open-Meteo day-ahead
            forecast plus a walk-forward bias correction, σ is the walk-forward error spread shrunk
            toward a 3°F prior. Bin probabilities integrate that density over each strike bin, with a
            continuity correction for whole-degree settlement.
          </li>
          <li>
            <strong>Market.</strong> De-vigged (sum-normalized) bin mid-prices from the last hourly
            candle at or before the snapshot.
          </li>
          <li>
            <strong>No leakage.</strong> The walk-forward error state for a day uses only strictly
            earlier days, and a unit test proves that appending future events never changes past
            features.
          </li>
          <li>
            <strong>Data.</strong> Public unauthenticated endpoints only, cached and rate-limited.
            Kalshi&rsquo;s API retains only ~2 months of settled markets, so a daily Action archives
            everything to git — this repo is the durable record.
          </li>
        </ul>
      </section>

      <footer>
        <p>
          Generated {data.generated_at.replace("T", " ").replace("+00:00", "")} UTC ·{" "}
          <a href="https://github.com/JonahXA/predictedge">source</a> ·{" "}
          <a href="https://github.com/JonahXA/closingline">ClosingLine</a>, the sportsbook predecessor.
        </p>
        <p style={{ marginTop: 8 }}>
          Market-efficiency research, not financial advice. No wagering, no accounts, no money — every
          position on this site is hypothetical.
        </p>
      </footer>
    </main>
  );
}
