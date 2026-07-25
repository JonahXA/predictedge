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
import { Legend, tooltipStyle, usePalette } from "./palette";

type Row = {
  snapshot: string;
  events: number;
  brier_model: number;
  brier_market: number;
  d_brier: number;
};

/** Brier score at each decision time, earliest (market open) to latest.
 *  The market's line falls as the event approaches while the baseline's
 *  stays flat — the widening gap is information entering the price. */
export default function SnapshotSweep({ rows }: { rows: Row[] }) {
  const p = usePalette();

  return (
    <>
      <Legend modelLabel="Baseline model" marketLabel="Kalshi market (de-vigged)" />
      <div className="chart-scroll">
        <ResponsiveContainer width="100%" height={300} minWidth={560}>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis
              dataKey="snapshot"
              tick={{ fill: p.muted, fontSize: 11 }}
              stroke={p.baseline}
            />
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
              formatter={(v: number, name: string) => [
                v.toFixed(4),
                name === "brier_model" ? "Baseline model" : "Market",
              ]}
              labelFormatter={(l: string, payload) =>
                `${l} · ${payload?.[0]?.payload?.events ?? "?"} events`
              }
            />
            <Line
              dataKey="brier_model"
              stroke={p.model}
              strokeWidth={2}
              dot={{ r: 3, fill: p.model, strokeWidth: 0 }}
              isAnimationActive={false}
            />
            <Line
              dataKey="brier_market"
              stroke={p.market}
              strokeWidth={2}
              dot={{ r: 3, fill: p.market, strokeWidth: 0 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
