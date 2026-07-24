#!/usr/bin/env python3
"""Transfer test: does the frozen veto config survive off the cross-section it
was selected on?

WHY THIS EXISTS
---------------
The veto family's parameters (VETO_POOL=15, VETO_PCT=0.30, TOP_N=10,
VETO_REBALANCE_DAYS=10) were chosen by testing ~40 variants against the full
point-in-time S&P 500 over 2019-04..2026-07. That is selection pressure on a
single window, and the README says so. Running *more* variants on that same
window cannot undo it.

What can be tested without new data: freeze every parameter and re-run on
cross-sections the parameters were never selected against. If the veto's
value-add over pure 12-1 momentum only exists on the full universe, it is a
fitting artifact. If it shows up in slices that had no say in choosing the
parameters, the effect is more likely real.

WHAT IS MEASURED
----------------
Not the level (CAGR/Sharpe of the veto) — the level is contaminated and also
varies with slice composition. What is measured is the DELTA: veto minus pure
12-1 momentum *within the same slice*, using the same rebalance calendar. The
claim under test is "the model's veto improves momentum", so the delta is the
claim.

SLICES
------
  full             reference; should reproduce reports/veto_deploy.json
  ex_tech          Information Technology dropped — is this a tech artifact?
  ex_<sector>      leave-one-sector-out, one run per major sector
  random_half_<i>  disjoint random half-universes (breadth: broad edge, or a
                   handful of names carrying it?)
  first/second_half  same config on each half of the OOS window (delta stability)

HONEST LIMITS
-------------
This is a generalization test, not a true out-of-universe test. Every slice is
drawn from the same point-in-time S&P 500 the parameters saw, so the slices are
not independent of the selection process — they are weaker evidence than a
genuinely unseen universe would be.

A real out-of-universe test (S&P 400 midcaps, Nasdaq-100) is NOT done here on
purpose: point-in-time membership history exists only for the S&P 500, so any
other index would have to be backtested on *current* membership. That is
survivorship bias, and it would inflate the result in exactly the way that
invalidated `technical-analysis-stock-scanner`. A biased transfer test is worse
than none, because it produces a number people trust.

The only fully clean test remains forward paper tracking (scripts/scorecard.py).

CLI:
    python scripts/transfer_test.py
    python scripts/transfer_test.py --n-random 8 --seed 7
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
from momentum_blend import momentum_series, picks_momentum_veto, picks_pure  # noqa: E402
from strategy import TOP_N, VETO_PCT, VETO_POOL, VETO_REBALANCE_DAYS  # noqa: E402
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

# Sector map lives in the sibling repo (current GICS membership). Sector labels
# are as-of-today, not point-in-time — sectors are stable enough for slicing,
# but delisted names simply have no label and land in "Unknown".
SECTOR_CSV = os.path.join(
    os.path.dirname(_ROOT), "ranker-21d-sp500", "data", "universe", "sp500_sectors.csv"
)

MIN_NAMES = VETO_POOL + 5   # a slice must comfortably support the momentum pool
MIN_COVERAGE = 0.80         # ...on at least this fraction of rebalance dates


def load_sectors() -> pd.Series:
    if not os.path.exists(SECTOR_CSV):
        print(f"  ! sector map not found at {SECTOR_CSV} — sector slices skipped")
        return pd.Series(dtype=object)
    df = pd.read_csv(SECTOR_CSV)
    return df.set_index("Ticker")["GICS Sector"]


def run_slice(name, tickers, oos, mom, prices, reb, calendar):
    """Frozen config on one sub-universe. Returns veto vs pure-momentum stats
    and the delta, or None if the slice cannot support the strategy."""
    tickers = set(tickers)
    sub = oos[oos["ticker"].isin(tickers)]
    if sub.empty:
        return None

    by_date = dict(tuple(sub.groupby("date")))
    ok_dates = [d for d in reb if len(by_date.get(d, ())) >= MIN_NAMES]
    coverage = len(ok_dates) / len(reb) if reb else 0.0
    if coverage < MIN_COVERAGE:
        print(f"  ~ {name}: only {coverage:.0%} of rebalance dates have "
              f">={MIN_NAMES} names — skipped")
        return None

    cols = [t for t in mom.columns if t in tickers]
    mom_sub = mom[cols]

    p_veto = picks_momentum_veto(by_date, mom_sub, reb, VETO_POOL, TOP_N, VETO_PCT)
    p_mom = picks_pure(by_date, mom_sub, reb, "momentum", TOP_N)
    if not p_veto or not p_mom:
        return None

    nav_v, _ = simulate(p_veto, prices, calendar, fill_mode="next_open")
    nav_m, _ = simulate(p_mom, prices, calendar, fill_mode="next_open")
    sv, sm = compute_stats(nav_v), compute_stats(nav_m)

    return {
        "slice": name,
        "n_tickers": len(tickers),
        "n_rebalances": len(p_veto),
        "coverage": round(coverage, 3),
        "veto": sv,
        "momentum": sm,
        "delta_cagr": round(sv["cagr"] - sm["cagr"], 4),
        "delta_sharpe": round(sv["sharpe"] - sm["sharpe"], 3),
        "delta_maxdd": round(sv["max_drawdown"] - sm["max_drawdown"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-random", type=int, default=6,
                    help="number of random half-universe splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Frozen config under test: "
          f"VETO_POOL={VETO_POOL} VETO_PCT={VETO_PCT} TOP_N={TOP_N} "
          f"REBALANCE={VETO_REBALANCE_DAYS}\n")

    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(dates)
    reb = list(dates[0::VETO_REBALANCE_DAYS])
    universe = sorted(oos["ticker"].unique())

    prices = build_price_matrices(universe)
    mom = momentum_series(prices["close"])

    results = []

    # --- reference ---------------------------------------------------------
    print("\n== full universe (reference) ==")
    ref = run_slice("full", universe, oos, mom, prices, reb, calendar)
    if ref:
        results.append(ref)
        print(f"  veto {ref['veto']['cagr']:+.1%} / Sharpe {ref['veto']['sharpe']:.2f}"
              f"   momentum {ref['momentum']['cagr']:+.1%} / {ref['momentum']['sharpe']:.2f}"
              f"   delta {ref['delta_cagr']:+.1%} CAGR, {ref['delta_sharpe']:+.2f} Sharpe")

    # --- sector slices -----------------------------------------------------
    sectors = load_sectors()
    if not sectors.empty:
        labelled = {t: sectors.get(t, "Unknown") for t in universe}
        counts = pd.Series(labelled).value_counts()
        major = [s for s, c in counts.items() if s != "Unknown" and c >= 25]

        print("\n== leave-one-sector-out ==")
        for sec in major:
            keep = [t for t in universe if labelled[t] != sec]
            tag = "ex_tech" if sec == "Information Technology" else \
                  "ex_" + sec.lower().replace(" ", "_")
            r = run_slice(tag, keep, oos, mom, prices, reb, calendar)
            if r:
                r["dropped_sector"] = sec
                results.append(r)
                print(f"  {tag:<28} delta {r['delta_cagr']:+.1%} CAGR, "
                      f"{r['delta_sharpe']:+.2f} Sharpe   "
                      f"(veto {r['veto']['cagr']:+.1%})")

    # --- random half-universes --------------------------------------------
    print("\n== random half-universes (breadth) ==")
    rng = np.random.default_rng(args.seed)
    for i in range(args.n_random):
        half = rng.permutation(universe)[: len(universe) // 2]
        r = run_slice(f"random_half_{i}", half, oos, mom, prices, reb, calendar)
        if r:
            results.append(r)
            print(f"  random_half_{i:<16} delta {r['delta_cagr']:+.1%} CAGR, "
                  f"{r['delta_sharpe']:+.2f} Sharpe   "
                  f"(veto {r['veto']['cagr']:+.1%})")

    # --- calendar halves ---------------------------------------------------
    # Not a universe slice: the same frozen config on each half of the OOS
    # window. The README reports the edge is positive in all 8 years, but that
    # is the LEVEL. This checks the DELTA over momentum survives in each half —
    # a veto that only adds value in one regime is a regime bet.
    print("\n== calendar halves (delta stability over time) ==")
    mid = dates[len(dates) // 2]
    for tag, lo, hi in (("first_half", dates[0], mid), ("second_half", mid, dates[-1])):
        sub_dates = [d for d in dates if lo <= d <= hi]
        sub_reb = [d for d in reb if lo <= d <= hi]
        if len(sub_reb) < 8:
            continue
        r = run_slice(tag, universe, oos, mom, prices, sub_reb,
                      pd.DatetimeIndex(sub_dates))
        if r:
            r["window"] = [str(pd.Timestamp(lo).date()), str(pd.Timestamp(hi).date())]
            results.append(r)
            print(f"  {tag:<28} delta {r['delta_cagr']:+.1%} CAGR, "
                  f"{r['delta_sharpe']:+.2f} Sharpe   "
                  f"(veto {r['veto']['cagr']:+.1%})")

    # --- verdict -----------------------------------------------------------
    tested = [r for r in results if r["slice"] != "full"]
    pos_cagr = sum(1 for r in tested if r["delta_cagr"] > 0)
    pos_sharpe = sum(1 for r in tested if r["delta_sharpe"] > 0)
    n = len(tested)

    print("\n" + "=" * 70)
    print(f"  Slices where the veto BEAT pure momentum: "
          f"{pos_cagr}/{n} on CAGR, {pos_sharpe}/{n} on Sharpe")
    if n:
        print(f"  Median delta: {np.median([r['delta_cagr'] for r in tested]):+.2%} CAGR, "
              f"{np.median([r['delta_sharpe'] for r in tested]):+.2f} Sharpe")
        worst = min(tested, key=lambda r: r["delta_cagr"])
        print(f"  Worst slice: {worst['slice']} at {worst['delta_cagr']:+.1%} CAGR")
    print("=" * 70)
    print("  Reading it: the veto claim is that the model improves momentum.")
    print("  A broad majority of positive deltas supports the claim; deltas that")
    print("  collapse once one sector is removed mean the edge lives in that")
    print("  sector, not in the model.")
    print("  This is a generalization test, NOT proof — see module docstring.")

    out = {
        "frozen_config": {
            "VETO_POOL": VETO_POOL, "VETO_PCT": VETO_PCT,
            "TOP_N": TOP_N, "VETO_REBALANCE_DAYS": VETO_REBALANCE_DAYS,
        },
        "n_slices_tested": n,
        "slices_veto_beat_momentum_cagr": pos_cagr,
        "slices_veto_beat_momentum_sharpe": pos_sharpe,
        "results": results,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "transfer_test.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.relpath(path, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
