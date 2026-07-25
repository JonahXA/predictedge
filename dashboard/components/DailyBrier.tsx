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

type Row = { date: string; model_brier: number; market_brier: number; n: number };

/** Per-day mean Brier across the six cities. */
export default function DailyBrier({ rows }: { rows: Row[] }) {
  const p = usePalette();

  return (
    <>
      <Legend modelLabel="Baseline model" marketLabel="Kalshi market (de-vigged)" />
      <div className="chart-scroll">
        <ResponsiveContainer width="100%" height={280} minWidth={560}>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: p.muted, fontSize: 11 }}
              stroke={p.baseline}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              domain={[0, "auto"]}
              tick={{ fill: p.muted, fontSize: 12 }}
              stroke={p.baseline}
              width={48}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={tooltipStyle(p)}
              formatter={(v: number, name: string) => [
                v.toFixed(4),
                name === "model_brier" ? "Baseline model" : "Market",
              ]}
              labelFormatter={(l: string, payload) =>
                `${l} · ${payload?.[0]?.payload?.n ?? "?"} cities`
              }
            />
            <Line
              dataKey="model_brier"
              stroke={p.model}
              strokeWidth={1.6}
              dot={false}
              activeDot={{ r: 4, stroke: p.surface, strokeWidth: 2 }}
              isAnimationActive={false}
            />
            <Line
              dataKey="market_brier"
              stroke={p.market}
              strokeWidth={1.6}
              dot={false}
              activeDot={{ r: 4, stroke: p.surface, strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
