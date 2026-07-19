#!/usr/bin/env python3
"""Attach the target to the sampled feature rows → data/processed/panel.parquet.

Target: forward 5-trading-day close-to-close return in EXCESS of SPY,
    forward_5d_excess_spy = clip(close[t+5]/close[t] − 1, ±20%) − spy_ret_5d
computed from daily closes (a stock's session close = its last RTH 5-min
bar's close; SPY from data/market/SPY_daily.parquet — same feed/adjustment).

Why SPY-excess rather than the sibling's date-demeaning: subtracting a
per-date constant never changes within-date ordering, so rank IC and top-N
picks are identical under either; SPY-excess matches the written project
plan and the gate-2 benchmark, and the SPY series is needed anyway.

Rows at the live edge (last 5 sessions) keep NaN labels — they're the rows a
future `today.py` would score; dataset.py drops them for training.

CLI:
    python scripts/labels.py
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import load_daily, load_spy_daily  # noqa: E402
from features import SAMPLED_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
PANEL_PATH = os.path.join(_ROOT, "data", "processed", "panel.parquet")

# Label horizon in trading days. Overridable via env for horizon experiments
# (the decay curve keeps voting for 10-21d):  HORIZON_DAYS=21 uv run ...
# dataset.py's purge gap and evaluate.py's Newey-West lag follow it.
FORWARD_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
# Clip scales with sqrt(horizon) so the tail treatment stays comparable
# (±20% at 5d → ±41% at 21d, ≈ sibling's ±50% convention at its 21d horizon).
CLIP_PCT = round(0.20 * (FORWARD_DAYS / 5) ** 0.5, 2)
RAW_LABEL_COL = f"forward_{FORWARD_DAYS}d_return"
LABEL_COL = f"forward_{FORWARD_DAYS}d_excess_spy"


def forward_returns(close: pd.Series, days: int = FORWARD_DAYS) -> pd.Series:
    """close[t+days]/close[t] − 1 on the series' own session index."""
    return close.shift(-days) / close - 1.0


def add_label(sampled: pd.DataFrame) -> pd.DataFrame:
    spy_close = load_spy_daily()["Close"]
    spy_fwd = forward_returns(spy_close)

    parts: list[pd.DataFrame] = []
    for t, grp in tqdm(sampled.groupby("ticker"), desc="Labels"):
        daily = load_daily(t)
        if daily is None:
            continue
        fwd = forward_returns(daily["Close"])
        g = grp.copy()
        g[RAW_LABEL_COL] = fwd.reindex(g["date"]).to_numpy()
        parts.append(g)

    panel = pd.concat(parts, ignore_index=True)
    clipped = panel[RAW_LABEL_COL].clip(-CLIP_PCT, CLIP_PCT)
    panel[LABEL_COL] = clipped - spy_fwd.reindex(panel["date"]).to_numpy()
    return panel


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    if not os.path.exists(SAMPLED_PATH):
        raise SystemExit(f"{SAMPLED_PATH} not found. Run scripts/features.py first.")
    sampled = pd.read_parquet(SAMPLED_PATH)
    panel = add_label(sampled)

    tmp = PANEL_PATH + ".tmp"
    panel.to_parquet(tmp)
    os.replace(tmp, PANEL_PATH)

    labeled = panel[LABEL_COL].notna()
    y = panel.loc[labeled, LABEL_COL]
    print(
        f"\nWrote {PANEL_PATH}: {len(panel):,} rows "
        f"({labeled.sum():,} labeled, {(~labeled).sum():,} live-edge NaN).\n"
        f"{LABEL_COL}: mean={y.mean():+.4f} std={y.std():.4f} "
        f"pos%={100 * (y > 0).mean():.1f} "
        f"clip-hits={100 * (panel.loc[labeled, RAW_LABEL_COL].abs() >= CLIP_PCT).mean():.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
