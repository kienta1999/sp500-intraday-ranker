#!/usr/bin/env python3
"""Paper-tracking scorecard: grade every picks_*.csv against what actually
happened afterward.

This is the strategy's only unfakeable test — no backtest search bias can
touch returns that occur after the picks file was written. For each file:
equal-weight basket return from the pick date's close to +1/3/5/10 sessions
(as far as data allows), minus SPY over the same span.

Writes reports/paper_scorecard.csv; prints the running tally.

CLI:
    python scripts/scorecard.py
"""

import argparse
import os
import re
import sys
import warnings
from glob import glob

warnings.filterwarnings("ignore")

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import load_daily, load_spy_daily  # noqa: E402

_ROOT = os.path.dirname(_HERE)
PICKS_DIR = os.path.join(_ROOT, "picks")
REPORTS_DIR = os.path.join(_ROOT, "reports")
HORIZONS = (1, 3, 5, 10)


def fwd_return(close: pd.Series, d0: pd.Timestamp, h: int) -> float | None:
    idx = close.index
    pos = idx.searchsorted(d0)
    if pos >= len(idx) or idx[pos] != d0 or pos + h >= len(idx):
        return None
    return float(close.iloc[pos + h] / close.iloc[pos] - 1)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    spy = load_spy_daily()["Close"]

    rows = []
    for path in sorted(glob(os.path.join(PICKS_DIR, "picks_*.csv"))):
        m = re.search(r"picks_(\d{4}-\d{2}-\d{2})\.csv", path)
        if not m:
            continue
        d0 = pd.Timestamp(m.group(1))
        tickers = pd.read_csv(path)["ticker"].tolist()
        closes = {t: load_daily(t) for t in tickers}
        row: dict = {"date": d0.date(), "n": len(tickers)}
        for h in HORIZONS:
            rets = [
                r for t in tickers
                if closes[t] is not None
                and (r := fwd_return(closes[t]["Close"], d0, h)) is not None
            ]
            spy_r = fwd_return(spy, d0, h)
            if rets and spy_r is not None:
                row[f"excess_{h}d"] = round(sum(rets) / len(rets) - spy_r, 4)
        rows.append(row)

    if not rows:
        print("No picks files yet — run today.py first.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPORTS_DIR, "paper_scorecard.csv"), index=False)
    print("Paper scorecard (equal-weight basket return minus SPY):")
    print(df.to_string(index=False))
    done = [c for c in df.columns if c.startswith("excess_")]
    if done:
        print("\nMean per horizon:", {c: round(df[c].mean(), 4) for c in done})


if __name__ == "__main__":
    main()
