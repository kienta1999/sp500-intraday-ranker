#!/usr/bin/env python3
"""Panel loading, walk-forward window bookkeeping, and the lookahead guard.

Walk-forward protocol (chronological, never shuffled):
    train  = expanding window, at least MIN_TRAIN_DAYS (504 ≈ 2y) sessions
    [purge PURGE_GAP=5 sessions — the label horizon, so no forward-5d label
     straddles a boundary]
    val    = next VAL_DAYS (63) sessions   (early stopping + grid selection)
    [purge 5]
    test   = next TEST_DAYS (63) sessions  (out-of-sample; pooled across windows)
    roll test start forward by TEST_DAYS and repeat, train end expanding.

Lookahead guard: sample random (ticker, timestamp) panel rows, truncate the
raw 5-min bars to <= timestamp, recompute features, and assert the last row
matches the panel. Cross-sectional ranks are skipped (they need the full
universe cross-section, but they're timestamp-aligned so not a leak vector).

CLI:
    python scripts/dataset.py                  # window summary + lookahead check
    python scripts/dataset.py --quick          # skip the lookahead check
    python scripts/dataset.py --n-samples 200  # more thorough check
"""

import argparse
import os
import sys
import warnings
from typing import NamedTuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import load_bars  # noqa: E402
from features import (  # noqa: E402
    ALL_FEATURES,
    PER_TICKER_FEATURES,
    compute_features,
)
from labels import LABEL_COL, PANEL_PATH, RAW_LABEL_COL  # noqa: E402
from strategy import DEFAULT_SEED  # noqa: E402

MIN_TRAIN_DAYS = 504
VAL_DAYS = 63
TEST_DAYS = 63
PURGE_GAP = 5           # == label horizon
MIN_TEST_DAYS = 21      # accept a shorter final window down to this

FEATURE_COLS: list[str] = ALL_FEATURES
TARGET_COL: str = LABEL_COL


class Window(NamedTuple):
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────


def load_panel(path: str = PANEL_PATH, drop_na: bool = True) -> pd.DataFrame:
    """Read the labeled panel; by default drop rows unusable for training
    (rolling-warmup NaN features or live-edge NaN labels)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run scripts/labels.py first.")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])

    if drop_na:
        before = len(panel)
        panel = panel.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
        print(
            f"Loaded {before:,} rows; dropped {before - len(panel):,} "
            f"(warmup NaN features or live-edge NaN labels) → {len(panel):,} usable.",
            flush=True,
        )
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward windows
# ─────────────────────────────────────────────────────────────────────────────


def walk_forward_windows(
    dates: pd.DatetimeIndex | pd.Series,
    min_train_days: int = MIN_TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    test_days: int = TEST_DAYS,
    purge_gap: int = PURGE_GAP,
    min_test_days: int = MIN_TEST_DAYS,
) -> list[Window]:
    """Build the expanding-train walk-forward schedule over unique sessions."""
    u = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    windows: list[Window] = []
    train_end_idx = min_train_days - 1  # expanding train: [0, train_end_idx]
    wid = 0
    while True:
        val_start_idx = train_end_idx + 1 + purge_gap
        val_end_idx = val_start_idx + val_days - 1
        test_start_idx = val_end_idx + 1 + purge_gap
        test_end_idx = min(test_start_idx + test_days - 1, len(u) - 1)
        if test_end_idx - test_start_idx + 1 < min_test_days:
            break
        windows.append(
            Window(
                wid,
                u[0], u[train_end_idx],
                u[val_start_idx], u[val_end_idx],
                u[test_start_idx], u[test_end_idx],
            )
        )
        wid += 1
        train_end_idx += test_days
    return windows


def window_slices(
    panel: pd.DataFrame, w: Window
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = panel["date"]
    train = panel[(d >= w.train_start) & (d <= w.train_end)]
    val = panel[(d >= w.val_start) & (d <= w.val_end)]
    test = panel[(d >= w.test_start) & (d <= w.test_end)]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead guard
# ─────────────────────────────────────────────────────────────────────────────


def assert_no_lookahead(
    panel: pd.DataFrame,
    n_samples: int = 50,
    seed: int = DEFAULT_SEED,
    tol: float = 1e-6,
) -> None:
    """Recompute sampled rows from truncated raw bars; raise on any mismatch."""
    has_ts = panel["timestamp"].notna()
    sample = panel[has_ts].sample(
        n=min(n_samples, has_ts.sum()), random_state=seed
    ).reset_index(drop=True)

    cache: dict[str, pd.DataFrame] = {}
    mismatches: list[tuple] = []
    n_checked = 0

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Lookahead check"):
        t, ts = row["ticker"], pd.Timestamp(row["timestamp"])
        if t not in cache:
            bars = load_bars(t)
            if bars is None:
                continue
            cache[t] = bars
        trunc = cache[t].loc[:ts]
        if len(trunc) == 0 or trunc.index[-1] != ts:
            continue  # timestamp not in this cache (stale panel) — skip

        recomputed = compute_features(trunc).iloc[-1]
        for col in PER_TICKER_FEATURES:
            actual, new = row[col], recomputed[col]
            if pd.isna(actual) and pd.isna(new):
                continue
            if pd.isna(actual) or pd.isna(new) or not np.isclose(
                actual, new, rtol=tol, atol=tol
            ):
                mismatches.append((t, ts, col, actual, new))
        n_checked += 1

    if mismatches:
        print(f"\n{len(mismatches)} lookahead mismatches (showing up to 10):")
        for m in mismatches[:10]:
            print(f"  {m[0]} {m[1]} {m[2]}: panel={m[3]!r:<22} recomputed={m[4]!r}")
        raise AssertionError("Lookahead leak detected — see mismatches above.")
    print(
        f"\n[OK] No lookahead detected: {n_checked} rows × "
        f"{len(PER_TICKER_FEATURES)} features verified.",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_windows(panel: pd.DataFrame, windows: list[Window]) -> None:
    print(f"\n{len(windows)} walk-forward window(s):")
    for w in windows:
        n_test = ((panel["date"] >= w.test_start) & (panel["date"] <= w.test_end)).sum()
        print(
            f"  #{w.window_id}: train {w.train_start.date()}→{w.train_end.date()}  "
            f"val {w.val_start.date()}→{w.val_end.date()}  "
            f"test {w.test_start.date()}→{w.test_end.date()} ({n_test:,} rows)"
        )
    y = panel[TARGET_COL]
    print(
        f"\n{TARGET_COL}: mean={y.mean():+.4f} std={y.std():.4f} "
        f"pos%={100 * (y > 0).mean():.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--quick", action="store_true", help="Skip the lookahead check")
    args = ap.parse_args()

    panel = load_panel()
    windows = walk_forward_windows(panel["date"])
    if not windows:
        print(
            f"\nWARNING: not enough history for a full walk-forward window "
            f"({panel['date'].nunique()} sessions < "
            f"{MIN_TRAIN_DAYS + 2 * PURGE_GAP + VAL_DAYS + MIN_TEST_DAYS} needed).",
            flush=True,
        )
    else:
        _print_windows(panel, windows)

    if not args.quick:
        print(f"\nRunning lookahead check ({args.n_samples} samples)...")
        assert_no_lookahead(panel, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
