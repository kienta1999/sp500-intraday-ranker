#!/usr/bin/env python3
"""Score the latest session and print today's top-N picks.

Uses the MOST RECENT walk-forward model (models/xgb_wf_<date>.json with the
latest date) — no retraining. Scores the newest panel rows (the live edge,
where the forward label is still NaN but features are complete) and writes
picks/picks_<session>.csv.

NOTE: picks are for paper tracking until the evaluation gates pass (see
README). This script does not place orders.

CLI:
    python scripts/today.py
    python scripts/today.py --top-n 5
"""

import argparse
import os
import sys
import warnings
from glob import glob

warnings.filterwarnings("ignore")

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import load_panel  # noqa: E402
from features import ALL_FEATURES, NULLABLE_FEATURES  # noqa: E402
from strategy import TOP_N, load_model, predict, top_picks  # noqa: E402
from train import MODELS_DIR  # noqa: E402

_ROOT = os.path.dirname(_HERE)
PICKS_DIR = os.path.join(_ROOT, "picks")


def latest_model_path() -> str:
    paths = sorted(glob(os.path.join(MODELS_DIR, "xgb_wf_*.json")))
    if not paths:
        raise SystemExit(
            f"No models in {MODELS_DIR} — run scripts/train.py (or run_all.py --retrain) first."
        )
    return paths[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    panel = load_panel(drop_na=False)
    latest = panel["date"].max()
    day = panel[panel["date"] == latest].copy()
    # Live-edge rows have NaN labels (fine — we're predicting, not training),
    # but features must be valid. Long features are legitimately NaN for
    # young listings (XGBoost handles missing natively).
    required = [c for c in ALL_FEATURES if c not in NULLABLE_FEATURES]
    day = day.dropna(subset=required)
    if day.empty:
        raise SystemExit(f"No scoreable rows on {latest.date()} — run run_all.py first.")

    model_path = latest_model_path()
    scored = predict(day, load_model(model_path))
    picks = top_picks(scored, args.top_n)[
        ["ticker", "predicted_return", "close", "timestamp"]
    ].reset_index(drop=True)
    picks.index += 1

    print(f"\nSession {latest.date()} — decision bar {picks['timestamp'].iloc[0]}")
    print(f"Model: {os.path.basename(model_path)}  |  universe scored: {len(day)} tickers\n")
    from labels import FORWARD_DAYS  # noqa: E402 (local: keeps import graph flat)

    print(
        picks.drop(columns="timestamp")
        .rename(columns={"predicted_return": f"pred_{FORWARD_DAYS}d_excess_spy"})
        .to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 1 else f"{x:,.2f}")
    )
    print("\n(paper tracking only until the README gates pass)")

    os.makedirs(PICKS_DIR, exist_ok=True)
    out = os.path.join(PICKS_DIR, f"picks_{latest.date()}.csv")
    picks.to_csv(out, index_label="rank")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
