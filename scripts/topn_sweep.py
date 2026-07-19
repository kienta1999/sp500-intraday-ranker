#!/usr/bin/env python3
"""Portfolio-construction sweep on the saved OOS predictions.

Round-4 finding: IC improved at the 10d horizon but the top-10 portfolio got
worse — with ~62 rebalances, 10 names sample the ranking too thinly. This
sweeps basket width (and rebalance offsets) WITHOUT retraining: it reuses
data/processed/oos_predictions.parquet from the last train run.

CLI:
    python scripts/topn_sweep.py --rebalance-days 10
    python scripts/topn_sweep.py --rebalance-days 10 --top-ns 5,10,20,40
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from backtest import (  # noqa: E402
    N_OFFSETS,
    build_price_matrices,
    compute_stats,
    model_picks,
    simulate,
)
from labels import FORWARD_DAYS  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebalance-days", type=int, default=FORWARD_DAYS)
    ap.add_argument("--top-ns", default="5,10,20,40")
    args = ap.parse_args()
    top_ns = [int(x) for x in args.top_ns.split(",")]

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(dates)
    print(f"OOS: {len(oos):,} rows, {oos['ticker'].nunique()} tickers, "
          f"rebalance every {args.rebalance_days} sessions", flush=True)

    prices = build_price_matrices(sorted(oos["ticker"].unique()))

    results: dict[str, dict] = {}
    for n in top_ns:
        navs = []
        for off in range(N_OFFSETS):
            reb = list(dates[off::args.rebalance_days])
            picks = model_picks(oos, reb, top_n=n)
            nav, to = simulate(picks, prices, calendar, fill_mode="next_open")
            navs.append((nav, to))
        # offset 0 is the headline; band across offsets shows schedule luck
        nav0, to0 = navs[0]
        st = compute_stats(nav0)
        totals = sorted(nav.iloc[-1] / nav.iloc[0] - 1 for nav, _ in navs)
        st["avg_turnover"] = round(to0, 3)
        st["offset_total_return_min_max"] = [round(totals[0], 4), round(totals[-1], 4)]
        results[f"top_{n}"] = st
        print(f"  top-{n:<3} CAGR={st['cagr']:+.2%}  Sharpe={st['sharpe']:+.2f}  "
              f"MaxDD={st['max_drawdown']:+.2%}  "
              f"offsetTotRet=[{totals[0]:+.0%} … {totals[-1]:+.0%}]", flush=True)

    out = os.path.join(REPORTS_DIR, f"topn_sweep_h{args.rebalance_days}d.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
