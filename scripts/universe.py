#!/usr/bin/env python3
"""Build the frozen trading universe: current S&P 500 members ranked by
dollar volume.

Bootstraps entirely from the sibling project's caches — zero API calls:
  * membership + GICS sectors: ../ml-stock-forward-return/data/universe/sp500_sectors.csv
  * dollar volume: ../ml-stock-forward-return/data/raw/{TICKER}.parquet
    (daily OHLCV; 63-trading-day median of Close*Volume)

Output: data/universe/universe.csv — columns ticker, dollar_vol_63d, rank,
gics_sector, as_of. The file is COMMITTED to git and never auto-refreshed:
silently churning the universe would orphan cached intraday history and shift
every cross-sectional rank. Re-run manually (~quarterly); backfill additions
with `data.py --backfill --tickers NEW1,NEW2`.

Known limitation (accepted deliberately): fixing present-day membership over
4 years of history carries survivorship bias. The alternative — point-in-time
membership with intraday bars for departed names — isn't worth the complexity
for this project; read the results with that caveat.

CLI:
    python scripts/universe.py                     # build + print top of table
    python scripts/universe.py --sibling-root PATH # override sibling location
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
UNIVERSE_DIR = os.path.join(_ROOT, "data", "universe")
UNIVERSE_CSV = os.path.join(UNIVERSE_DIR, "universe.csv")

SIBLING_ROOT = os.path.join(os.path.dirname(_ROOT), "ml-stock-forward-return")
DOLLAR_VOL_WINDOW = 63  # trading days (~1 quarter) for the median dollar volume


def load_universe() -> list[str]:
    """Tickers in dollar-volume rank order — the single source of truth for
    data.py and features.py. Raises if the universe hasn't been built yet.
    """
    if not os.path.exists(UNIVERSE_CSV):
        raise SystemExit(f"{UNIVERSE_CSV} not found. Run scripts/universe.py first.")
    df = pd.read_csv(UNIVERSE_CSV)
    return df.sort_values("rank")["ticker"].tolist()


def load_sectors() -> dict[str, str]:
    """ticker -> GICS sector from the committed universe file."""
    df = pd.read_csv(UNIVERSE_CSV)
    return dict(zip(df["ticker"], df["gics_sector"]))


def build_universe(sibling_root: str = SIBLING_ROOT) -> pd.DataFrame:
    sectors_csv = os.path.join(sibling_root, "data", "universe", "sp500_sectors.csv")
    raw_dir = os.path.join(sibling_root, "data", "raw")
    if not os.path.exists(sectors_csv):
        raise SystemExit(
            f"{sectors_csv} not found — pass --sibling-root pointing at the "
            "ml-stock-forward-return checkout."
        )

    sectors = pd.read_csv(sectors_csv)
    rows: list[dict] = []
    missing: list[str] = []
    for _, r in sectors.iterrows():
        ticker = str(r["Ticker"]).strip()
        p = os.path.join(raw_dir, f"{ticker}.parquet")
        if not os.path.exists(p):
            missing.append(ticker)
            continue
        px = pd.read_parquet(p, columns=["Close", "Volume"])
        tail = px.dropna().tail(DOLLAR_VOL_WINDOW)
        if len(tail) < DOLLAR_VOL_WINDOW // 2:
            missing.append(ticker)
            continue
        rows.append(
            {
                "ticker": ticker,
                "dollar_vol_63d": float((tail["Close"] * tail["Volume"]).median()),
                "gics_sector": r["GICS Sector"],
            }
        )

    uni = pd.DataFrame(rows).sort_values("dollar_vol_63d", ascending=False)
    uni["rank"] = range(1, len(uni) + 1)
    uni["as_of"] = date.today().isoformat()
    uni = uni[["ticker", "dollar_vol_63d", "rank", "gics_sector", "as_of"]]

    if missing:
        print(
            f"{len(missing)} members skipped (no/short daily cache in sibling): "
            + ", ".join(missing),
            flush=True,
        )
    return uni.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sibling-root", default=SIBLING_ROOT)
    args = ap.parse_args()

    uni = build_universe(args.sibling_root)
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    tmp = UNIVERSE_CSV + ".tmp"
    uni.to_csv(tmp, index=False)
    os.replace(tmp, UNIVERSE_CSV)

    print(f"\nWrote {UNIVERSE_CSV}: {len(uni)} tickers.", flush=True)
    with pd.option_context("display.float_format", "{:,.0f}".format):
        print("\nTop 15 by 63d median dollar volume:")
        print(uni.head(15).to_string(index=False))
        print("\nBottom 5:")
        print(uni.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
