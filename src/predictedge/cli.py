"""predictedge CLI: ingest / backtest / evaluate / significance."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="predictedge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="archive settled Kalshi markets + candlesticks")
    sub.add_parser("backtest", help="walk-forward backtest vs the market")
    sub.add_parser("evaluate", help="aggregate Brier/log-loss summary")
    sub.add_parser("significance", help="bootstrap + Diebold-Mariano on the gap")
    sub.add_parser("forecast", help="issue pre-registered forecasts for open markets")
    sub.add_parser("sweep", help="model-vs-market gap across decision-time snapshots")
    args = ap.parse_args()

    if args.cmd == "ingest":
        from . import ingest
        ingest.run()
    elif args.cmd == "backtest":
        from . import backtest
        backtest.run()
    elif args.cmd == "evaluate":
        from . import backtest
        backtest.evaluate()
    elif args.cmd == "significance":
        from . import significance
        significance.run()
    elif args.cmd == "forecast":
        from . import forecast
        forecast.run()
    elif args.cmd == "sweep":
        from . import backtest
        backtest.sweep()


if __name__ == "__main__":
    main()
