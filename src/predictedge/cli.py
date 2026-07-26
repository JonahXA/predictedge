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
    sub.add_parser("compare", help="model variants vs market and vs each other")
    sub.add_parser("thin", help="thin-market study: does the gap grow with volume?")
    sub.add_parser("export", help="write dashboard/public/data.json")
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
        backtest.sweep([backtest.BASELINE, backtest.WEIGHTED])
    elif args.cmd == "compare":
        from . import backtest
        backtest.compare()
    elif args.cmd == "thin":
        from . import study
        study.run()
    elif args.cmd == "export":
        from . import export
        export.run()


if __name__ == "__main__":
    main()
