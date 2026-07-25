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

type Row = { series: string; bin_mid: number; predicted: number; observed: number; n: number };

/** Reliability curve: issued probability vs realized frequency, pooled
 *  over every strike bin. The dashed diagonal is perfect calibration. */
export default function Reliability({ rows }: { rows: Row[] }) {
  const p = usePalette();
  const model = rows.filter((r) => r.series === "model");
  const market = rows.filter((r) => r.series === "market");
  const merged = [...new Set(rows.map((r) => r.bin_mid))]
    .sort((a, b) => a - b)
    .map((bin) => ({
      bin_mid: bin,
      model: model.find((r) => r.bin_mid === bin)?.observed ?? null,
      market: market.find((r) => r.bin_mid === bin)?.observed ?? null,
      ideal: bin,
      n_model: model.find((r) => r.bin_mid === bin)?.n ?? 0,
    }));

  return (
    <>
      <Legend modelLabel="Baseline model" marketLabel="Kalshi market (de-vigged)" />
      <div className="chart-scroll">
        <ResponsiveContainer width="100%" height={300} minWidth={520}>
          <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis
              dataKey="bin_mid"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: p.muted, fontSize: 11 }}
              stroke={p.baseline}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              label={{
                value: "Forecast probability",
                position: "insideBottom",
                offset: -2,
                style: { fill: p.muted, fontSize: 11 },
              }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: p.muted, fontSize: 12 }}
              stroke={p.baseline}
              width={48}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle(p)}
              formatter={(v: number, name: string) => [
                `${(v * 100).toFixed(1)}%`,
                name === "model" ? "Model observed" : name === "market" ? "Market observed" : "Perfect",
              ]}
              labelFormatter={(l: number) => `Forecast ${Math.round(l * 100)}%`}
            />
            <Line
              dataKey="ideal"
              stroke={p.baseline}
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="model"
              stroke={p.model}
              strokeWidth={2}
              dot={{ r: 3, fill: p.model, strokeWidth: 0 }}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              dataKey="market"
              stroke={p.market}
              strokeWidth={2}
              dot={{ r: 3, fill: p.market, strokeWidth: 0 }}
              connectNulls
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
