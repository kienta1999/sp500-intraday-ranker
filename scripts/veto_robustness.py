#!/usr/bin/env python3
"""Robustness battery for the momentum-with-veto lead (round 8).

The +31.5%/0.97 cell (k15→n10, veto 30%) was the best of 4 tested configs —
before believing it, map the neighborhood: if the edge is real it should be
smooth across pool size / veto threshold / schedule offsets, and its per-year
attribution should explain WHERE the veto earns (hypothesis: crash regimes,
by dodging momentum's blowups). All backtest-only on the saved OOS.

Outputs reports/veto_robustness.json + printed tables.

CLI:
    HORIZON_DAYS=10 python scripts/veto_robustness.py --rebalance-days 10
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
    simulate,
)
from labels import FORWARD_DAYS  # noqa: E402
from momentum_blend import momentum_series, picks_momentum_veto, picks_pure  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")


def yearly_returns(nav: pd.Series) -> dict:
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    return {str(idx.year): round(float(v), 4) for idx, v in yearly.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebalance-days", type=int, default=FORWARD_DAYS)
    args = ap.parse_args()

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    all_dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(all_dates)
    oos_by_date = dict(tuple(oos.groupby("date")))

    prices = build_price_matrices(sorted(oos["ticker"].unique()))
    mom = momentum_series(prices["close"])

    results: dict = {"stability_surface": {}, "offsets": {}, "yearly": {}}

    # 1. Stability surface around the winning cell (offset 0).
    print("Stability surface (CAGR / Sharpe):", flush=True)
    reb0 = list(all_dates[0 :: args.rebalance_days])
    for k in (12, 15, 20, 25):
        for v in (0.2, 0.3, 0.4):
            picks = picks_momentum_veto(oos_by_date, mom, reb0, k, 10, v)
            nav, _ = simulate(picks, prices, calendar, fill_mode="next_open")
            st = compute_stats(nav)
            results["stability_surface"][f"k{k}_v{int(v*100)}"] = st
            print(f"  k={k:<3} veto={v:.0%}:  CAGR={st['cagr']:+.2%}  Sharpe={st['sharpe']:+.2f}",
                  flush=True)

    # 2. Offset bands for the winner and pure momentum.
    print("\nOffset robustness (5 schedules):", flush=True)
    for name, fn in (
        ("veto_k15_n10_v30", lambda d: picks_momentum_veto(oos_by_date, mom, d, 15, 10, 0.3)),
        ("pure_momentum_n10", lambda d: picks_pure(oos_by_date, mom, d, "momentum", 10)),
    ):
        navs = []
        for off in range(N_OFFSETS):
            reb = list(all_dates[off :: args.rebalance_days])
            nav, _ = simulate(fn(reb), prices, calendar, fill_mode="next_open")
            navs.append(nav)
        cagrs = sorted(compute_stats(n)["cagr"] for n in navs)
        results["offsets"][name] = [round(c, 4) for c in cagrs]
        print(f"  {name:<20} CAGR across offsets: "
              + " ".join(f"{c:+.1%}" for c in cagrs), flush=True)
        if name.startswith("veto"):
            results["yearly"][name] = yearly_returns(navs[0])
        else:
            results["yearly"][name] = yearly_returns(navs[0])

    # 3. Per-year attribution (offset 0).
    print("\nPer-year returns (offset 0):", flush=True)
    years = sorted(results["yearly"]["veto_k15_n10_v30"].keys())
    print("  year   veto      momentum   edge")
    for y in years:
        ve = results["yearly"]["veto_k15_n10_v30"][y]
        mo = results["yearly"]["pure_momentum_n10"][y]
        print(f"  {y}  {ve:+8.1%}  {mo:+8.1%}  {ve - mo:+8.1%}")

    with open(os.path.join(REPORTS_DIR, "veto_robustness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote reports/veto_robustness.json", flush=True)


if __name__ == "__main__":
    main()
