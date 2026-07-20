#!/usr/bin/env python3
"""Full pool/veto/hold grid for the veto strategy, with a rebalance-count
diagnostic.

Two open questions from the round-9 surface:
  1. Does holding FEWER than 10 (n=5) work, or is it too noisy?
  2. Are the small-pool cells (k=12) inflated by SKIPPED rebalances? When
     fewer than n names survive the veto, picks_momentum_veto emits nothing
     for that date, so the book silently rides on — which flatters a config
     for trading less rather than vetoing better.

Reports realized rebalance count alongside performance so cells that skipped
their way to a good number are visible.

CLI:
    HORIZON_DAYS=10 python scripts/veto_grid.py --rebalance-days 10
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

from backtest import build_price_matrices, compute_stats, simulate  # noqa: E402
from labels import FORWARD_DAYS  # noqa: E402
from momentum_blend import momentum_series, picks_momentum_veto, picks_pure  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

POOLS = (8, 10, 12, 15, 20, 25)
VETOS = (0.2, 0.3, 0.4)
HOLDS = (5, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebalance-days", type=int, default=FORWARD_DAYS)
    args = ap.parse_args()

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(dates)
    oos_by_date = dict(tuple(oos.groupby("date")))
    reb = list(dates[0 :: args.rebalance_days])
    n_scheduled = len(reb)

    prices = build_price_matrices(sorted(oos["ticker"].unique()))
    mom = momentum_series(prices["close"])

    results: dict[str, dict] = {}

    # Baselines: pure momentum at each hold width (no model involved).
    for n in HOLDS:
        picks = picks_pure(oos_by_date, mom, reb, "momentum", n)
        nav, to = simulate(picks, prices, calendar, fill_mode="next_open")
        st = compute_stats(nav) | {"rebalances": len(picks), "scheduled": n_scheduled,
                                   "avg_turnover": round(to, 3)}
        results[f"pure_momentum_n{n}"] = st
        print(f"  pure momentum n={n:<3}            CAGR={st['cagr']:+.2%}  "
              f"Sharpe={st['sharpe']:+.2f}  MaxDD={st['max_drawdown']:+.1%}  "
              f"rebal={len(picks)}/{n_scheduled}", flush=True)

    print(flush=True)
    for n in HOLDS:
        for k in POOLS:
            if k < n:
                continue
            for v in VETOS:
                picks = picks_momentum_veto(oos_by_date, mom, reb, k, n, v)
                if not picks:
                    continue
                nav, to = simulate(picks, prices, calendar, fill_mode="next_open")
                st = compute_stats(nav) | {"rebalances": len(picks),
                                           "scheduled": n_scheduled,
                                           "avg_turnover": round(to, 3)}
                results[f"veto_k{k}_n{n}_v{int(v*100)}"] = st
                flag = "  <-- SKIPPED REBALANCES" if len(picks) < 0.9 * n_scheduled else ""
                print(f"  k={k:<3} hold={n:<3} veto={v:.0%}  CAGR={st['cagr']:+.2%}  "
                      f"Sharpe={st['sharpe']:+.2f}  MaxDD={st['max_drawdown']:+.1%}  "
                      f"rebal={len(picks)}/{n_scheduled}{flag}", flush=True)

    with open(os.path.join(REPORTS_DIR, "veto_grid.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote reports/veto_grid.json", flush=True)


if __name__ == "__main__":
    main()
