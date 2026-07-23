#!/usr/bin/env python3
"""Point-in-time S&P 500 membership + the frozen backfill universe.

Survivorship-bias control, ported from the sibling project
(ranker-21d-sp500/scripts/universe.py): the model only ever sees a
(ticker, date) row if that stock was ACTUALLY in the index on that date.
Ranking today's hottest names over a past window would otherwise bake
"picked the winners in hindsight" into every backtest number.

Bootstrapped from the sibling's caches — zero HTTP calls here:
  * membership history: sibling data/universe/sp500_history.parquet
    (long (date, ticker) built from github.com/fja05680/sp500 change events
    + synthetic Wikipedia "today" snapshots, refreshed by the sibling's cron)
  * current sectors:    sibling data/universe/sp500_sectors.csv

Local caches (data/universe/):
  sp500_history.parquet   copy of the sibling history (refreshed each run)
  universe.csv            COMMITTED — every member during the backfill window,
                          dollar-volume ranked (delisted names, which have no
                          sibling daily cache, sort last). Drives data.py pulls.

Public API
----------
    load_universe()                 backfill ticker list (rank order)
    members_on(date)                tickers in the index on/just-before date
    all_historical_tickers(since=)  every ticker in the index since `since`
    filter_to_members(panel)        drop rows where ticker wasn't a member on date
    load_sectors()                  ticker -> GICS sector (current members only)

CLI:
    python scripts/universe.py                     # refresh history + rebuild universe.csv
    python scripts/universe.py --sibling-root PATH # override sibling location
"""

import argparse
import os
import shutil
import sys
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
UNIVERSE_DIR = os.path.join(_ROOT, "data", "universe")
UNIVERSE_CSV = os.path.join(UNIVERSE_DIR, "universe.csv")
HISTORY_FILE = os.path.join(UNIVERSE_DIR, "sp500_history.parquet")

SIBLING_ROOT = os.path.join(os.path.dirname(_ROOT), "ranker-21d-sp500")
DOLLAR_VOL_WINDOW = 63   # trading days for the median dollar volume (backfill order)
BACKFILL_YEARS = 10.0    # membership window for universe.csv — keep == data.DEFAULT_YEARS
                         # (10y ≈ Alpaca's full SIP archive, back to 2016)
UNKNOWN_SECTOR = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Membership history (point-in-time)
# ─────────────────────────────────────────────────────────────────────────────


def refresh_history(sibling_root: str = SIBLING_ROOT) -> pd.DataFrame:
    """Copy the sibling's membership history into our cache and return it.

    The sibling's daily cron keeps its parquet current (fja05680 CSV +
    synthetic Wikipedia snapshots); we just mirror it. Falls back to parsing
    the sibling's raw change-event CSV if the parquet is missing.
    """
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    src = os.path.join(sibling_root, "data", "universe", "sp500_history.parquet")
    if os.path.exists(src) and os.path.getsize(src) > 0:
        if (
            not os.path.exists(HISTORY_FILE)
            or os.path.getmtime(HISTORY_FILE) < os.path.getmtime(src)
        ):
            shutil.copy2(src, HISTORY_FILE)
        return pd.read_parquet(HISTORY_FILE)

    raw = os.path.join(sibling_root, "data", "universe", "SP_500_Historical_Component.csv")
    if not os.path.exists(raw):
        raise SystemExit(
            f"Neither {src} nor {raw} found — pass --sibling-root pointing at "
            "the ranker-21d-sp500 checkout."
        )
    df = pd.read_csv(raw)
    df["date"] = pd.to_datetime(df["date"])
    rows = [
        (d, t.strip().replace(".", "-"))
        for d, tickers in zip(df["date"], df["tickers"])
        for t in tickers.split(",")
    ]
    hist = (
        pd.DataFrame(rows, columns=["date", "ticker"])
        .drop_duplicates()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    tmp = HISTORY_FILE + ".tmp"
    hist.to_parquet(tmp)
    os.replace(tmp, HISTORY_FILE)
    return hist


def load_history() -> pd.DataFrame:
    """Local membership history (long (date, ticker)); refresh if absent."""
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        return pd.read_parquet(HISTORY_FILE)
    return refresh_history()


def members_on(when, history: pd.DataFrame | None = None) -> list[str]:
    """Tickers in the S&P 500 on `when` (most-recent snapshot <= when)."""
    if history is None:
        history = load_history()
    snap_dates = np.sort(history["date"].unique())
    pos = snap_dates.searchsorted(np.datetime64(pd.Timestamp(when), "ns"), side="right") - 1
    if pos < 0:
        return []
    return history.loc[history["date"] == snap_dates[pos], "ticker"].tolist()


def all_historical_tickers(
    since=None, history: pd.DataFrame | None = None
) -> list[str]:
    """Every ticker in the index in at least one snapshot on/after `since`."""
    if history is None:
        history = load_history()
    if since is not None:
        # Include the snapshot in effect AT `since`, not just snapshots after it.
        snap_dates = np.sort(history["date"].unique())
        pos = snap_dates.searchsorted(
            np.datetime64(pd.Timestamp(since), "ns"), side="right"
        ) - 1
        cutoff = snap_dates[max(pos, 0)]
        history = history[history["date"] >= cutoff]
    return sorted(history["ticker"].unique())


def filter_to_members(
    panel: pd.DataFrame,
    history: pd.DataFrame | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Keep only rows where `ticker` was an S&P 500 member on `date`.

    Interval-compression port of the sibling's implementation: membership of
    snapshot i applies until snapshot i+1 − 1 day; consecutive present-
    snapshots merge into (start, end) spans, so add → remove → re-add doesn't
    backdate membership during the gap.
    """
    if history is None:
        history = load_history()

    snap_dates = pd.DatetimeIndex(np.sort(history["date"].unique()))
    snap_pos = {d: i for i, d in enumerate(snap_dates)}
    ticker_idx: dict[str, list[int]] = (
        history.assign(_si=history["date"].map(snap_pos))
        .groupby("ticker")["_si"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )
    snap_end_after = list(snap_dates[1:] - pd.Timedelta(days=1)) + [
        pd.Timestamp("2999-12-31")
    ]

    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ticker, idxs in ticker_idx.items():
        spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        a = b = idxs[0]
        for k in idxs[1:]:
            if k == b + 1:
                b = k
            else:
                spans.append((snap_dates[a], snap_end_after[b]))
                a = b = k
        spans.append((snap_dates[a], snap_end_after[b]))
        intervals[ticker] = spans

    keep = np.zeros(len(panel), dtype=bool)
    for ticker, group in panel.groupby(ticker_col, sort=False):
        spans = intervals.get(ticker)
        if not spans:
            continue
        gd = group[date_col].to_numpy()
        m = np.zeros(len(group), dtype=bool)
        for start, end in spans:
            m |= (gd >= start.to_datetime64()) & (gd <= end.to_datetime64())
        keep[group.index.to_numpy()] = m

    dropped = len(panel) - int(keep.sum())
    if dropped:
        print(
            f"filter_to_members: dropped {dropped:,} of {len(panel):,} rows "
            f"(ticker not in index on date).",
            flush=True,
        )
    return panel[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backfill universe (universe.csv)
# ─────────────────────────────────────────────────────────────────────────────


def load_universe() -> list[str]:
    """Tickers to pull, in dollar-volume rank order (delisted names last)."""
    if not os.path.exists(UNIVERSE_CSV):
        raise SystemExit(f"{UNIVERSE_CSV} not found. Run scripts/universe.py first.")
    df = pd.read_csv(UNIVERSE_CSV)
    return df.sort_values("rank")["ticker"].tolist()


def load_sectors() -> dict[str, str]:
    """ticker -> GICS sector from the committed universe file."""
    df = pd.read_csv(UNIVERSE_CSV)
    return dict(zip(df["ticker"], df["gics_sector"]))


def build_universe(sibling_root: str = SIBLING_ROOT) -> pd.DataFrame:
    history = refresh_history(sibling_root)
    start = pd.Timestamp(datetime.now() - timedelta(days=365.25 * BACKFILL_YEARS))
    tickers = all_historical_tickers(since=start, history=history)

    sectors_csv = os.path.join(sibling_root, "data", "universe", "sp500_sectors.csv")
    sectors = (
        pd.read_csv(sectors_csv).set_index("Ticker")["GICS Sector"].to_dict()
        if os.path.exists(sectors_csv) else {}
    )
    raw_dir = os.path.join(sibling_root, "data", "raw")

    rows: list[dict] = []
    for t in tickers:
        dv = np.nan
        p = os.path.join(raw_dir, f"{t}.parquet")
        if os.path.exists(p):
            try:
                px = pd.read_parquet(p, columns=["Close", "Volume"]).dropna()
                tail = px.tail(DOLLAR_VOL_WINDOW)
                if len(tail) >= DOLLAR_VOL_WINDOW // 2:
                    dv = float((tail["Close"] * tail["Volume"]).median())
            except Exception:
                pass
        rows.append(
            {
                "ticker": t,
                "dollar_vol_63d": dv,
                "gics_sector": sectors.get(t, UNKNOWN_SECTOR),
            }
        )

    uni = pd.DataFrame(rows).sort_values(
        "dollar_vol_63d", ascending=False, na_position="last"
    )
    uni["rank"] = range(1, len(uni) + 1)
    uni["as_of"] = date.today().isoformat()
    return uni[["ticker", "dollar_vol_63d", "rank", "gics_sector", "as_of"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sibling-root", default=SIBLING_ROOT)
    args = ap.parse_args()

    uni = build_universe(args.sibling_root)
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    tmp = UNIVERSE_CSV + ".tmp"
    uni.to_csv(tmp, index=False)
    os.replace(tmp, UNIVERSE_CSV)

    history = load_history()
    n_current = len(members_on(pd.Timestamp.today(), history=history))
    n_delisted = int(uni["dollar_vol_63d"].isna().sum())
    print(
        f"\nWrote {UNIVERSE_CSV}: {len(uni)} members during the last "
        f"{BACKFILL_YEARS:.0f}y window ({n_current} current, "
        f"~{n_delisted} departed/no-daily-cache).",
        flush=True,
    )
    print(
        f"History: {history['date'].nunique()} snapshot dates, "
        f"{history['date'].min().date()} → {history['date'].max().date()}."
    )
    with pd.option_context("display.float_format", "{:,.0f}".format):
        print("\nTop 10 by dollar volume:")
        print(uni.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
