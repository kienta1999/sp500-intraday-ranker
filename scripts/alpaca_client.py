#!/usr/bin/env python3
"""Shared Alpaca Market Data plumbing used by data.py.

Owns the API-key check, the historical-data client factory, and a retried
bar-fetch helper that normalizes Alpaca's response into the repo's raw-bar
schema (tz-aware UTC index named `date`; Open/High/Low/Close/Volume/VWAP).

Conventions baked in here (and relied on downstream):
  * feed = SIP — the full consolidated tape, so volume is comparable across
    the whole history and never mixes sources. Alpaca's free tier serves SIP
    *historical* data as long as the query ends >= 15 minutes in the past
    (only live streaming is IEX-limited), so every request here clamps `end`
    to now-16min.
  * adjustment = SPLIT — split-adjusted but NOT dividend-adjusted, matching
    the convention of IBKR TRADES bars (the eventual execution venue) and
    keeping 5-day return math honest (dividend drift is negligible at this
    horizon).
  * Timestamps are bar-START times in UTC. Conversion to America/New_York
    happens only inside features.py.

Keys come from env vars ALPACA_API_KEY / ALPACA_SECRET_KEY (free account at
https://alpaca.markets — no funding needed for market data).
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

RETRIES = 3
RETRY_SLEEP = 2.0
# Free-tier SIP historical queries require end >= 15 min in the past.
SIP_DELAY_MIN = 16

BAR_5MIN = TimeFrame(5, TimeFrameUnit.Minute)
BAR_1DAY = TimeFrame(1, TimeFrameUnit.Day)

_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vwap": "VWAP",
}


def get_client() -> StockHistoricalDataClient:
    """Build the historical-data client, failing fast if keys are missing."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set.\n"
            "Create a free account at https://alpaca.markets, generate an API "
            "key pair (paper is fine), and export both env vars."
        )
    return StockHistoricalDataClient(api_key=key, secret_key=secret)


def latest_queryable_end() -> datetime:
    """Most recent `end` the free tier will serve from the SIP feed."""
    return datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MIN)


def fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime | None = None,
) -> pd.DataFrame | None:
    """Fetch bars for one symbol with retry. Returns None on hard failure.

    The SDK paginates internally (10k bars/page) and retries 429s on its own;
    the loop here covers transient network/5xx failures on top of that.
    Result: DataFrame indexed by tz-aware UTC bar-start `date`, columns
    Open/High/Low/Close/Volume/VWAP, sorted ascending. Empty ranges (e.g. a
    holiday week) return an empty DataFrame, not None.
    """
    end = min(end or latest_queryable_end(), latest_queryable_end())
    if start >= end:
        return pd.DataFrame(columns=list(_COLUMN_MAP.values()))

    # Class shares: the universe file uses yfinance's dash form (BRK-B, BF-B,
    # inherited from the sibling); Alpaca wants dots (BRK.B). Cache filenames
    # keep the dash form — only the API request is translated.
    api_symbol = symbol.replace("-", ".")
    req = StockBarsRequest(
        symbol_or_symbols=[api_symbol],
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.SIP,
    )
    for attempt in range(1, RETRIES + 1):
        try:
            barset = client.get_stock_bars(req)
            df = barset.df
            if df is None or df.empty:
                return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
            # BarSet.df is MultiIndex (symbol, timestamp) — drop the symbol level.
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            df = df.rename(columns=_COLUMN_MAP)
            df = df[[c for c in _COLUMN_MAP.values() if c in df.columns]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "date"
            return df.sort_index()
        except Exception as e:
            msg = str(e)
            if "subscription" in msg.lower():
                raise SystemExit(
                    f"Alpaca rejected the SIP request for {symbol}: {msg}\n"
                    "Free-tier SIP history requires the query end to be >= 15 "
                    "minutes old — this should never trip here, so check the "
                    "account's data subscription status."
                )
            if attempt == RETRIES:
                print(f"  [{symbol}] failed after {RETRIES} retries: {e}", flush=True)
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None
