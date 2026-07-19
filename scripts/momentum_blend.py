#!/usr/bin/env python3
"""Momentum × model combination experiments (backtest-only, no retraining).

The disruptive-regime play: 12-1 momentum owns this market (+55% CAGR OOS),
so instead of fighting it head-on, ride it and add the model's edge on top.
Two families, evaluated on the saved OOS predictions:

  two_stage   candidates = top-K by 12-1 momentum that day, then pick the
              top-N by model score WITHIN them ("momentum finds the wave,
              the model picks the surfers")
  blend       composite = alpha * momentum-rank + (1-alpha) * model-rank,
              buy the top-N of the composite

References: pure momentum and pure model at the same widths.

CLI:
    HORIZON_DAYS=10 python scripts/momentum_blend.py --rebalance-days 10
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
    build_price_matrices,
    compute_stats,
    simulate,
)
from labels import FORWARD_DAYS  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

MOM_SKIP, MOM_LOOKBACK = 21, 252


def momentum_series(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(MOM_SKIP) / close.shift(MOM_LOOKBACK) - 1


def picks_two_stage(oos_by_date, mom, dates, k: int, n: int) -> dict:
    out = {}
    for d in dates:
        day = oos_by_date.get(d)
        if day is None or d not in mom.index:
            continue
        m = mom.loc[d, [t for t in day["ticker"] if t in mom.columns]].dropna()
        pool = set(m.nlargest(k).index)
        cand = day[day["ticker"].isin(pool)]
        if len(cand) >= n:
            out[d] = cand.nlargest(n, "y_pred")["ticker"].tolist()
    return out


def picks_blend(oos_by_date, mom, dates, alpha: float, n: int) -> dict:
    out = {}
    for d in dates:
        day = oos_by_date.get(d)
        if day is None or d not in mom.index:
            continue
        df = day.set_index("ticker")
        m = mom.loc[d].reindex(df.index)
        score = alpha * m.rank(pct=True) + (1 - alpha) * df["y_pred"].rank(pct=True)
        score = score.dropna()
        if len(score) >= n:
            out[d] = score.nlargest(n).index.tolist()
    return out


def picks_pure(oos_by_date, mom, dates, source: str, n: int) -> dict:
    out = {}
    for d in dates:
        day = oos_by_date.get(d)
        if day is None:
            continue
        if source == "model":
            if len(day) >= n:
                out[d] = day.nlargest(n, "y_pred")["ticker"].tolist()
        else:
            if d not in mom.index:
                continue
            m = mom.loc[d, [t for t in day["ticker"] if t in mom.columns]].dropna()
            if len(m) >= n:
                out[d] = m.nlargest(n).index.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebalance-days", type=int, default=FORWARD_DAYS)
    args = ap.parse_args()

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    all_dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(all_dates)
    reb = list(all_dates[0 :: args.rebalance_days])
    oos_by_date = dict(tuple(oos.groupby("date")))

    prices = build_price_matrices(sorted(oos["ticker"].unique()))
    mom = momentum_series(prices["close"])

    variants: dict[str, dict] = {}
    for k, n in ((50, 10), (100, 10), (100, 20)):
        variants[f"two_stage_k{k}_n{n}"] = picks_two_stage(oos_by_date, mom, reb, k, n)
    for a, n in ((0.75, 20), (0.5, 20), (0.25, 20)):
        variants[f"blend_a{int(a*100)}_n{n}"] = picks_blend(oos_by_date, mom, reb, a, n)
    for n in (10, 20):
        variants[f"pure_momentum_n{n}"] = picks_pure(oos_by_date, mom, reb, "momentum", n)
    variants["pure_model_n20"] = picks_pure(oos_by_date, mom, reb, "model", 20)

    results: dict[str, dict] = {}
    for name, picks in variants.items():
        nav, to = simulate(picks, prices, calendar, fill_mode="next_open")
        st = compute_stats(nav)
        st["avg_turnover"] = round(to, 3)
        results[name] = st
        print(f"  {name:<22} CAGR={st['cagr']:+.2%}  Sharpe={st['sharpe']:+.2f}  "
              f"MaxDD={st['max_drawdown']:+.2%}  turnover={st['avg_turnover']}",
              flush=True)

    out = os.path.join(REPORTS_DIR, f"momentum_blend_h{args.rebalance_days}d.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
