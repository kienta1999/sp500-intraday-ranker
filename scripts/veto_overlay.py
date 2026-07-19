#!/usr/bin/env python3
"""Deployment realism for the promoted veto strategy: cost drag by account
size + vol-target overlay on the drawdown.

1. Order accounting: exact order counts from the pick sequence → $1/order
   drag at $10k/$50k/$100k (spread drag is proportional, already in the sim).
2. Vol-target overlay (sibling's trick): scale exposure to
   min(1, target / SPY 20d realized vol), applied on the daily return series
   with a 1-day lag — no lookahead.

Writes reports/veto_deploy.json.

CLI:
    HORIZON_DAYS=10 python scripts/veto_overlay.py
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from backtest import build_price_matrices, compute_stats, simulate  # noqa: E402
from data import load_spy_daily  # noqa: E402
from momentum_blend import momentum_series, picks_momentum_veto  # noqa: E402
from strategy import VETO_PCT, VETO_POOL, VETO_REBALANCE_DAYS  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

VOL_TARGET = 0.20
VOL_LOOKBACK = 20


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(dates)
    oos_by_date = dict(tuple(oos.groupby("date")))

    prices = build_price_matrices(sorted(oos["ticker"].unique()))
    mom = momentum_series(prices["close"])
    reb = list(dates[0::VETO_REBALANCE_DAYS])
    picks = picks_momentum_veto(oos_by_date, mom, reb, VETO_POOL, 10, VETO_PCT)

    # 1. Order accounting from the pick sequence.
    baskets = [picks[d] for d in sorted(picks)]
    orders = len(baskets[0]) if baskets else 0
    for prev, cur in zip(baskets, baskets[1:]):
        orders += len(set(prev) ^ set(cur))
    years = len(calendar) / 252
    orders_per_year = orders / years
    drag = {f"${c//1000}k": round(orders_per_year * 1.0 / c, 4)
            for c in (10_000, 50_000, 100_000)}

    # 2. Base sim + vol-target overlay.
    nav, to = simulate(picks, prices, calendar, fill_mode="next_open")
    base = compute_stats(nav)

    spy = load_spy_daily()["Close"].reindex(calendar).ffill()
    spy_vol = spy.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    weight = (VOL_TARGET / spy_vol).clip(upper=1.0).shift(1).fillna(1.0)
    r = nav.pct_change().fillna(0.0)
    overlay_nav = (1 + weight * r).cumprod() * nav.iloc[0]
    overlay = compute_stats(overlay_nav)

    out = {
        "orders_per_year": round(orders_per_year, 1),
        "fixed_cost_drag_per_year": drag,
        "avg_turnover": round(to, 3),
        "base": base,
        "vol_target_overlay": overlay,
        "avg_exposure": round(float(weight.mean()), 3),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(REPORTS_DIR, "veto_deploy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote reports/veto_deploy.json", flush=True)


if __name__ == "__main__":
    main()
