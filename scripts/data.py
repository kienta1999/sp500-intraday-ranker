#!/usr/bin/env python3
"""Download and cache 5-minute OHLCV bars for the universe + SPY daily closes.

Per-ticker parquets:  data/raw/{TICKER}.parquet
    tz-aware UTC bar-start index `date`; Open/High/Low/Close/Volume/VWAP;
    regular trading hours only (NY-local bar starts 09:30–15:55).
Market series:        data/market/SPY_daily.parquet

Source is Alpaca Market Data (SIP feed, split-adjusted — see alpaca_client.py
for the conventions and why the daily top-up must stay on the same source as
the backfill: mixing IBKR volume into an Alpaca history would put a seam in
every volume feature right at the live edge).

Behaviour:
  * --backfill: pull DEFAULT_YEARS of history for every universe ticker,
    skipping tickers whose cache already reaches back far enough. ~500
    tickers x 4y ≈ 30-60 min (the SDK paginates 10k bars/request; Alpaca's
    free tier allows ~200 req/min).
  * --topup: incremental tail-append from each ticker's last cached bar to
    now-16min. Whole universe in a couple of minutes; suitable for cron.
  * Writes are atomic (.tmp + os.replace) so an interrupted run never
    corrupts a cache; rerunning resumes ticker-by-ticker.
  * SPY daily closes are refreshed on every invocation (labels + benchmark).

CLI:
    python scripts/data.py --backfill                       # full universe
    python scripts/data.py --backfill --tickers AAPL,MSFT   # subset
    python scripts/data.py --backfill --years 0.1           # short smoke pull
    python scripts/data.py --topup                          # daily increment
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from alpaca_client import (  # noqa: E402
    BAR_1DAY,
    BAR_5MIN,
    fetch_bars,
    get_client,
    latest_queryable_end,
)
from universe import load_universe  # noqa: E402

_ROOT = os.path.dirname(_HERE)
RAW_DIR = os.path.join(_ROOT, "data", "raw")
MARKET_DIR = os.path.join(_ROOT, "data", "market")
SPY_DAILY_PATH = os.path.join(MARKET_DIR, "SPY_daily.parquet")

DEFAULT_YEARS = 5.0  # 252d long-feature warmup eats ~1y; 5y keeps ~2y of OOS
NY_TZ = "America/New_York"
# Regular trading hours, in bar-START times: 09:30 first bar, 15:55 last
# (covers 15:55–16:00). Alpaca returns pre/post-market bars; drop them at
# ingest — overnight bars have garbage liquidity and would poison features.
RTH_START = "09:30"
RTH_LAST_BAR = "15:55"
# Cache counts as "reaching back far enough" if its first bar is within this
# many days of the requested start (new listings/IPOs legitimately start late).
BACKFILL_SLACK_DAYS = 14


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cache_path(ticker: str) -> str:
    return os.path.join(RAW_DIR, f"{ticker}.parquet")


def _load_cached(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_atomic(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars whose NY-local start time is within regular trading hours."""
    if df.empty:
        return df
    local = df.index.tz_convert(NY_TZ)
    times = pd.Series(local.strftime("%H:%M"), index=df.index)
    return df[(times >= RTH_START) & (times <= RTH_LAST_BAR)]


def _merge_save(ticker: str, cached: pd.DataFrame | None, new: pd.DataFrame) -> int:
    merged = new if cached is None or cached.empty else pd.concat([cached, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _save_atomic(merged, _cache_path(ticker))
    return len(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch modes
# ─────────────────────────────────────────────────────────────────────────────


def backfill_ticker(client, ticker: str, start: datetime) -> tuple[str, int, str]:
    """Pull [start → now] 5-min bars, reusing whatever tail is cached.

    Returns (ticker, n_rows, status) with status in
    {"new", "updated", "fresh", "failed", "empty"}.
    """
    cached = _load_cached(ticker)
    if cached is not None and len(cached):
        first, last = cached.index.min(), cached.index.max()
        deep_enough = first <= start + timedelta(days=BACKFILL_SLACK_DAYS)
        if deep_enough:
            # History reaches back; just top up the live edge.
            new = fetch_bars(client, ticker, BAR_5MIN, start=last + timedelta(minutes=5))
            if new is None:
                return ticker, len(cached), "failed"
            if new.empty:
                return ticker, len(cached), "fresh"
            return ticker, _merge_save(ticker, cached, _rth_only(new)), "updated"
        # Cache exists but is shallow (e.g. an earlier short smoke pull):
        # refetch the whole window — one clean request beats hole-stitching.

    new = fetch_bars(client, ticker, BAR_5MIN, start=start)
    if new is None:
        return ticker, 0, "failed"
    new = _rth_only(new)
    if new.empty:
        return ticker, 0, "empty"
    return ticker, _merge_save(ticker, cached, new), "new"


def topup_ticker(client, ticker: str) -> tuple[str, int, str]:
    """Tail-append from the last cached bar. Skips tickers never backfilled."""
    cached = _load_cached(ticker)
    if cached is None or not len(cached):
        return ticker, 0, "missing"
    last = cached.index.max()
    if latest_queryable_end() - last < timedelta(minutes=10):
        return ticker, len(cached), "fresh"
    new = fetch_bars(client, ticker, BAR_5MIN, start=last + timedelta(minutes=5))
    if new is None:
        return ticker, len(cached), "failed"
    new = _rth_only(new)
    if new.empty:
        return ticker, len(cached), "fresh"
    return ticker, _merge_save(ticker, cached, new), "updated"


def fetch_spy_daily(client, years: float = DEFAULT_YEARS) -> pd.DataFrame:
    """Refresh data/market/SPY_daily.parquet (full re-pull — one cheap request).

    Daily bars, same feed/adjustment as the 5-min bars so labels and benchmark
    share the stocks' conventions.
    """
    start = datetime.now(timezone.utc) - timedelta(days=365.25 * years + 30)
    df = fetch_bars(client, "SPY", BAR_1DAY, start=start)
    if df is None or df.empty:
        raise SystemExit("SPY daily download failed — labels need it.")
    # Daily bar timestamps → naive dates (one row per session).
    df.index = pd.to_datetime(df.index.tz_convert(NY_TZ).date)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    _save_atomic(df, SPY_DAILY_PATH)
    print(f"SPY daily: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})", flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders (used by features.py / labels.py / backtest.py)
# ─────────────────────────────────────────────────────────────────────────────


def load_bars(ticker: str) -> pd.DataFrame | None:
    """Cached 5-min RTH bars for one ticker (tz-aware UTC index), or None."""
    return _load_cached(ticker)


def cached_tickers() -> list[str]:
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(
        f.removesuffix(".parquet") for f in os.listdir(RAW_DIR) if f.endswith(".parquet")
    )


def load_daily(ticker: str) -> pd.DataFrame | None:
    """Daily OHLCV derived from the cached 5-min bars (one row per session).

    Open = first RTH bar's Open, Close = last bar's Close (~16:00 print),
    High/Low = session extremes, Volume = session sum. Index: naive dates.
    """
    bars = _load_cached(ticker)
    if bars is None or bars.empty:
        return None
    local = bars.tz_convert(NY_TZ) if bars.index.tz is not None else bars
    session = pd.Series(local.index.tz_convert(NY_TZ).date, index=bars.index)
    g = bars.groupby(session.values)
    daily = pd.DataFrame(
        {
            "Open": g["Open"].first(),
            "High": g["High"].max(),
            "Low": g["Low"].min(),
            "Close": g["Close"].last(),
            "Volume": g["Volume"].sum(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily.sort_index()


def load_spy_daily() -> pd.DataFrame:
    if not os.path.exists(SPY_DAILY_PATH):
        raise SystemExit(f"{SPY_DAILY_PATH} not found. Run scripts/data.py first.")
    return pd.read_parquet(SPY_DAILY_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _summarize(results: list[tuple[str, int, str]]) -> None:
    summary = pd.DataFrame(results, columns=["Ticker", "Rows", "Status"])
    counts = summary["Status"].value_counts().to_dict()
    print(
        "\nDone. "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        flush=True,
    )
    for status in ("failed", "missing", "empty"):
        bad = summary[summary["Status"] == status]
        if len(bad):
            print(f"{status}: " + ", ".join(bad["Ticker"].tolist()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true", help="Pull full history")
    mode.add_argument("--topup", action="store_true", help="Incremental tail-append")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS)
    ap.add_argument("--tickers", help="Comma-separated subset (default: universe)")
    ap.add_argument("--skip-spy", action="store_true")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_universe()

    client = get_client()

    if not args.skip_spy:
        fetch_spy_daily(client, years=args.years)

    results: list[tuple[str, int, str]] = []
    if args.backfill:
        start = datetime.now(timezone.utc) - timedelta(days=365.25 * args.years)
        print(
            f"Backfilling {len(tickers)} tickers from "
            f"{start.date()} (5-min bars, SIP, RTH only)...",
            flush=True,
        )
        for t in tqdm(tickers, desc="Backfill"):
            results.append(backfill_ticker(client, t, start=start))
    else:
        print(f"Topping up {len(tickers)} tickers...", flush=True)
        for t in tqdm(tickers, desc="Topup"):
            results.append(topup_ticker(client, t))

    _summarize(results)


if __name__ == "__main__":
    main()
