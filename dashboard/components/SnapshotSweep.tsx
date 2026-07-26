"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { tooltipStyle, usePalette } from "./palette";

type Row = {
  variant?: string;
  snapshot: string;
  events: number;
  brier_model: number;
  brier_market: number;
  d_brier: number;
};

const SERIES = {
  baseline: "Baseline model",
  improved: "Improved model",
  market: "Kalshi market",
} as const;

/** Brier at each decision time. The market's line falls as the event
 *  approaches while both models stay flat — the widening gap is
 *  information entering the price that our archive cannot follow. */
export default function SnapshotSweep({ rows }: { rows: Row[] }) {
  const p = usePalette();
  const variants = [...new Set(rows.map((r) => r.variant).filter(Boolean))] as string[];
  const baselineName = variants.find((v) => v.toLowerCase().includes("baseline"));
  const improvedName = variants.find((v) => v !== baselineName);

  const order = rows
    .filter((r) => !baselineName || r.variant === baselineName)
    .map((r) => r.snapshot);
  const merged = order.map((snapshot) => {
    const b = rows.find((r) => r.snapshot === snapshot && (!baselineName || r.variant === baselineName));
    const i = improvedName ? rows.find((r) => r.snapshot === snapshot && r.variant === improvedName) : undefined;
    return {
      snapshot,
      baseline: b?.brier_model ?? null,
      improved: i?.brier_model ?? null,
      market: b?.brier_market ?? null,
      events: b?.events ?? 0,
    };
  });

  const lines: [keyof typeof SERIES, string, number][] = [
    ["baseline", p.muted, 1.5],
    ["improved", p.model, 2],
    ["market", p.market, 2],
  ];

  return (
    <>
      <div className="legend">
        {improvedName && (
          <span>
            <span className="swatch" style={{ background: "var(--muted)" }} />
            {SERIES.baseline}
          </span>
        )}
        <span>
          <span className="swatch" style={{ background: "var(--series-model)" }} />
          {improvedName ? SERIES.improved : SERIES.baseline}
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--series-market)" }} />
          {SERIES.market} (de-vigged)
        </span>
      </div>
      <div className="chart-scroll">
        <ResponsiveContainer width="100%" height={300} minWidth={560}>
          <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis dataKey="snapshot" tick={{ fill: p.muted, fontSize: 11 }} stroke={p.baseline} />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fill: p.muted, fontSize: 12 }}
              stroke={p.baseline}
              width={52}
              tickFormatter={(v: number) => v.toFixed(2)}
              label={{
                value: "Brier (lower is better)",
                angle: -90,
                position: "insideLeft",
                style: { fill: p.muted, fontSize: 11, textAnchor: "middle" },
              }}
            />
            <Tooltip
              contentStyle={tooltipStyle(p)}
              formatter={(v: number, name: string) => [v.toFixed(4), SERIES[name as keyof typeof SERIES] ?? name]}
              labelFormatter={(l: string, payload) =>
                `${l} · ${payload?.[0]?.payload?.events ?? "?"} events`
              }
            />
            {lines.map(([key, color, width]) =>
              key === "baseline" && !improvedName ? null : (
                <Line
                  key={key}
                  dataKey={key}
                  stroke={color}
                  strokeWidth={width}
                  strokeDasharray={key === "baseline" ? "4 3" : undefined}
                  dot={{ r: 3, fill: color, strokeWidth: 0 }}
                  connectNulls
                  isAnimationActive={false}
                />
              )
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
