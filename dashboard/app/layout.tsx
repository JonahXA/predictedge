import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PredictEdge — pre-registered forecasts vs prediction markets",
  description:
    "Probabilistic forecasts for Kalshi daily-temperature contracts, committed to git before resolution and benchmarked against the market's own price. Paper measurement only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
