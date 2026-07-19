#!/usr/bin/env python3
"""How fast do the walk-forward models rot?

Each saved window model (models/xgb_wf_<test_start>.json) was trained on data
up to its own window. Here every model is also scored on every LATER window's
test slice — all still strictly out-of-sample for that model — giving IC as a
function of model age. Aggregating across models by age offset answers: how
much IC does a model lose per quarter of staleness? That number is the
evidence for (or against) the quarterly retraining cadence.

    age 0 = the model's own test window (the normal walk-forward result)
    age k = scored on the window k quarters after its own

Reads:  models/xgb_wf_*.json, reports/walkforward_params.json, panel.parquet
Writes: reports/model_aging.csv / .png

CLI:
    python scripts/model_aging.py
"""

import argparse
import json
import os
import sys
import warnings
from glob import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import FEATURE_COLS, TARGET_COL, load_panel  # noqa: E402
from strategy import load_model  # noqa: E402
from train import MODELS_DIR, PARAMS_PATH, daily_ic  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    with open(PARAMS_PATH) as f:
        wf = json.load(f)
    windows = [(w["window_id"], w["test_start"], w["test_end"]) for w in wf]
    panel = load_panel()

    # Pre-slice each window's test rows once.
    slices: dict[int, pd.DataFrame] = {}
    for wid, ts, te in windows:
        d = panel["date"]
        slices[wid] = panel[(d >= ts) & (d <= te)]

    model_paths = {os.path.basename(p): p for p in glob(os.path.join(MODELS_DIR, "xgb_wf_*.json"))}
    rows: list[dict] = []
    for wid, ts, te in windows:
        name = f"xgb_wf_{ts}.json"
        if name not in model_paths:
            print(f"  (skipping window {wid}: {name} not found)", flush=True)
            continue
        model = load_model(model_paths[name])
        for wid2, ts2, te2 in windows:
            if wid2 < wid:
                continue  # only same-or-later windows are OOS for this model
            sl = slices[wid2]
            if sl.empty:
                continue
            ic = daily_ic(sl["date"], sl[TARGET_COL], model.predict(sl[FEATURE_COLS]))
            rows.append(
                {"model_window": wid, "scored_window": wid2, "age_quarters": wid2 - wid,
                 "test_start": ts2, "ic": round(ic, 4)}
            )
        print(f"model w{wid} ({ts}): scored {sum(r['model_window'] == wid for r in rows)} windows",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPORTS_DIR, "model_aging.csv"), index=False)

    curve = df.groupby("age_quarters")["ic"].agg(["mean", "std", "count"])
    print("\nIC by model age (quarters since training):")
    print(curve.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o",
                capsize=3, color="tab:blue")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("model age (quarters since end of training)")
    ax.set_ylabel("mean daily IC")
    ax.set_title("Model aging — does a frozen model rot?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "model_aging.png"), dpi=110)
    print(f"\nWrote reports/model_aging.csv and model_aging.png", flush=True)


if __name__ == "__main__":
    main()
