"use client";

import {
  CartesianGrid,
  Legend as RLegend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { tooltipStyle, usePalette } from "./palette";

type Row = {
  series: string;
  city: string;
  kind: string;
  events: number;
  vol_per_market: number;
  d_brier: number;
};

/** Each point is one market series: how much the market beats the model
 *  (y) against how much money trades in it (x, log scale). If sharpness
 *  is bought with attention, the cloud slopes upward. */
export default function ThinMarket({ rows }: { rows: Row[] }) {
  const p = usePalette();
  const pt = (r: Row) => ({ x: r.vol_per_market, y: r.d_brier, city: r.city, series: r.series });
  const highs = rows.filter((r) => r.kind === "high").map(pt);
  const lows = rows.filter((r) => r.kind === "low").map(pt);

  return (
    <div className="chart-scroll">
      <ResponsiveContainer width="100%" height={340} minWidth={560}>
        <ScatterChart margin={{ top: 8, right: 20, bottom: 18, left: 0 }}>
          <CartesianGrid stroke={p.grid} strokeWidth={1} />
          <XAxis
            type="number"
            dataKey="x"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fill: p.muted, fontSize: 11 }}
            stroke={p.baseline}
            tickFormatter={(v: number) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`)}
            label={{
              value: "traded volume per market (log)",
              position: "insideBottom",
              offset: -8,
              style: { fill: p.muted, fontSize: 11 },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            tick={{ fill: p.muted, fontSize: 12 }}
            stroke={p.baseline}
            width={56}
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{
              value: "market's Brier edge",
              angle: -90,
              position: "insideLeft",
              style: { fill: p.muted, fontSize: 11, textAnchor: "middle" },
            }}
          />
          <ZAxis range={[60, 60]} />
          <Tooltip
            contentStyle={tooltipStyle(p)}
            cursor={{ stroke: p.baseline, strokeDasharray: "3 3" }}
            formatter={(v: number, _name, item) =>
              item?.dataKey === "x"
                ? [Math.round(v).toLocaleString(), "volume/market"]
                : [v.toFixed(4), "ΔBrier"]
            }
            labelFormatter={() => ""}
          />
          <RLegend
            verticalAlign="top"
            height={28}
            wrapperStyle={{ fontSize: 12, color: p.ink2 }}
          />
          <Scatter name="daily high" data={highs} fill={p.model} isAnimationActive={false} />
          <Scatter name="daily low (thin)" data={lows} fill={p.market} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
