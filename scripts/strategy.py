#!/usr/bin/env python3
"""Shared strategy primitives and canonical constants.

Used by train.py (seed), backtest.py, and evaluate.py. Mirrors the sibling's
pattern: this module owns the canonical values; callers expose CLI overrides.

Imports feature lists directly from features.py (never via dataset.py) to
avoid a circular import — dataset.py imports DEFAULT_SEED from here.
"""

import os
import sys

import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from features import ALL_FEATURES as FEATURE_COLS  # noqa: E402, F401

# Portfolio rules (the evaluation portfolio; gate 2).
TOP_N = 10
REBALANCE_DAYS = 5            # weekly (every 5 trading days)

# The PROMOTED strategy (round 9, 2026-07-19): momentum-with-veto.
# 12-1 momentum picks its top VETO_POOL names; the model excludes any it
# ranks in its bottom VETO_PCT percentile that day; hold the top TOP_N
# survivors by momentum, rebalance every VETO_REBALANCE_DAYS sessions.
# Survived the robustness battery: smooth config neighborhood, wins on all
# 5 offsets and in all 8 OOS years (reports/veto_robustness.json).
VETO_POOL = 15
VETO_PCT = 0.30
VETO_REBALANCE_DAYS = 10
COST_PER_ORDER = 1.00         # dollars per order ($1/order broker commission)
SPREAD_BPS = 3.0              # half-spread cost per side, basis points
DEFAULT_CAPITAL = 100_000.0
# Buffer ranks (hold an incumbent while it stays inside the top BUFFER_RANK):
# a phase-4 turnover-reduction rule — only wired into live execution if the
# gates pass; the evaluation backtest stays plain top-N so gate 2 is honest.
BUFFER_RANK = 15

FILL_MODES = ("next_open", "same_day_1535")

# Default random seed everywhere an RNG appears (XGBoost random_state,
# sampling in assert_no_lookahead, random baseline seeds offset from it).
DEFAULT_SEED = 15


def load_model(path: str) -> xgb.XGBRegressor:
    booster = xgb.XGBRegressor()
    booster.load_model(path)
    return booster


def predict(df: pd.DataFrame, booster: xgb.XGBRegressor) -> pd.DataFrame:
    """Score rows; returns a copy with a `predicted_return` column."""
    out = df.copy()
    out["predicted_return"] = booster.predict(out[FEATURE_COLS])
    return out


def top_picks(day_panel: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Top-N rows by predicted_return on a single date's slice."""
    return day_panel.nlargest(top_n, "predicted_return")


def compute_weights(tickers: list[str]) -> dict[str, float]:
    """Equal weight across the basket (sums to 1.0)."""
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}
