#!/usr/bin/env python3
"""Walk-forward training harness.

Per window (see dataset.walk_forward_windows):
  * fit XGBRegressor (squared error, hist) on the expanding train slice with
    early stopping (RMSE) on the validation tail, n_estimators=1500 cap;
  * grid-search a SMALL deterministic grid (not Optuna — each window's val
    slice is only ~63 daily cross-sections, which 50 TPE trials would
    overfit), selecting by mean daily Spearman IC on val;
  * window 0 runs the full grid; later windows re-run only window 0's top
    TOP_CONFIGS configs (params are stable across adjacent windows; keeps
    cost linear);
  * predict the test slice out-of-sample.

Artifacts:
    models/xgb_wf_{test_start}.json          one model per window
    reports/walkforward_params.json          chosen params + val IC per window
    reports/feature_importance.csv           gain importance, averaged over windows
    data/processed/oos_predictions.parquet   pooled OOS table — the SOLE input
                                             to backtest.py and evaluate.py

CLI:
    python scripts/train.py             # full walk-forward
    python scripts/train.py --smoke     # tiny windows + 2 configs (plumbing test)
"""

import argparse
import itertools
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import (  # noqa: E402
    FEATURE_COLS,
    TARGET_COL,
    Window,
    load_panel,
    walk_forward_windows,
    window_slices,
)
from labels import RAW_LABEL_COL  # noqa: E402
from strategy import DEFAULT_SEED  # noqa: E402

_ROOT = os.path.dirname(_HERE)
MODELS_DIR = os.path.join(_ROOT, "models")
REPORTS_DIR = os.path.join(_ROOT, "reports")
OOS_PATH = os.path.join(_ROOT, "data", "processed", "oos_predictions.parquet")
PARAMS_PATH = os.path.join(REPORTS_DIR, "walkforward_params.json")

N_ESTIMATORS = 1500
EARLY_STOPPING_ROUNDS = 100
TOP_CONFIGS = 6  # configs carried from window 0 into later windows

# --quick: skip the grid entirely, one sane fixed config per window (early
# stopping still tunes n_estimators). For fast iteration rounds.
QUICK_PARAMS = dict(max_depth=4, learning_rate=0.03, min_child_weight=5, subsample=0.9)

GRID: list[dict] = [
    dict(max_depth=d, learning_rate=lr, min_child_weight=mcw, subsample=ss)
    for d, lr, mcw, ss in itertools.product(
        (3, 4, 6), (0.01, 0.03, 0.1), (5, 20), (0.6, 0.9)
    )
]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def daily_ic_series(dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray) -> pd.Series:
    """Per-date Spearman rank correlation between predictions and actuals."""
    df = pd.DataFrame(
        {"date": np.asarray(dates), "y_true": np.asarray(y_true), "y_pred": y_pred}
    )
    return df.groupby("date").apply(
        lambda g: g["y_pred"].corr(g["y_true"], method="spearman")
        if len(g) > 1 else np.nan
    ).dropna()


def daily_ic(dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Mean daily Spearman IC — the metric a cross-sectional ranker cares about."""
    return float(daily_ic_series(dates, y_true, y_pred).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────


def fit_one(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = DEFAULT_SEED,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=seed,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_window(
    panel: pd.DataFrame,
    w: Window,
    configs: list[dict],
    seed: int = DEFAULT_SEED,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> tuple[xgb.XGBRegressor, dict, float, pd.DataFrame]:
    """Grid over `configs`; return (best model, best params, val IC, test preds)."""
    feature_cols = feature_cols or FEATURE_COLS
    train, val, test = window_slices(panel, w)

    best: tuple[float, dict, xgb.XGBRegressor] | None = None
    for params in configs:
        model = fit_one(
            params, train[feature_cols], train[target_col],
            val[feature_cols], val[target_col], seed=seed,
        )
        ic = daily_ic(val["date"], val[target_col], model.predict(val[feature_cols]))
        if best is None or ic > best[0]:
            best = (ic, params, model)

    val_ic, best_params, model = best
    preds = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "ticker": test["ticker"].to_numpy(),
            "y_pred": model.predict(test[feature_cols]),
            "y_true_raw": test[RAW_LABEL_COL].to_numpy(),
            "y_true_excess": test[target_col].to_numpy(),
            "window_id": w.window_id,
        }
    )
    return model, best_params, val_ic, preds


def rank_configs(
    panel: pd.DataFrame,
    w: Window,
    seed: int = DEFAULT_SEED,
    grid: list[dict] | None = None,
) -> list[dict]:
    """Full grid on one window; return configs sorted by val IC (best first)."""
    grid = grid or GRID
    train, val, _ = window_slices(panel, w)
    scored: list[tuple[float, dict]] = []
    for params in grid:
        model = fit_one(
            params, train[FEATURE_COLS], train[TARGET_COL],
            val[FEATURE_COLS], val[TARGET_COL], seed=seed,
        )
        ic = daily_ic(val["date"], val[TARGET_COL], model.predict(val[FEATURE_COLS]))
        scored.append((ic, params))
        print(f"  grid {params}: val IC={ic:+.4f}", flush=True)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def run_walk_forward(
    panel: pd.DataFrame,
    windows: list[Window],
    seed: int = DEFAULT_SEED,
    params_per_window: list[dict] | None = None,
    grid: list[dict] | None = None,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
    save_models: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run every window; return (pooled OOS predictions, per-window info).

    `params_per_window` (from a prior run's walkforward_params.json) skips the
    grid entirely — evaluate.py's permutation/ablation runs use this so they
    re-fit with the SAME hyperparameters rather than re-tuning on shuffled data.
    """
    oos_parts: list[pd.DataFrame] = []
    info: list[dict] = []
    configs: list[dict] | None = None

    for w in windows:
        if params_per_window is not None:
            window_configs = [params_per_window[w.window_id]]
        elif configs is None:
            if verbose:
                print(f"\nWindow #{w.window_id}: full grid ({len(grid or GRID)} configs)...",
                      flush=True)
            configs = rank_configs(panel, w, seed=seed, grid=grid)[:TOP_CONFIGS]
            window_configs = configs
        else:
            window_configs = configs

        model, best_params, val_ic, preds = run_window(
            panel, w, window_configs, seed=seed,
            feature_cols=feature_cols, target_col=target_col,
        )
        oos_parts.append(preds)
        test_ic = daily_ic(preds["date"], preds["y_true_excess"], preds["y_pred"].to_numpy())
        info.append(
            {
                "window_id": w.window_id,
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "params": best_params,
                "best_iteration": int(model.best_iteration),
                "val_ic": round(val_ic, 4),
                "test_ic": round(test_ic, 4),
                "importance": dict(
                    zip(feature_cols or FEATURE_COLS,
                        [float(x) for x in model.feature_importances_])
                ),
            }
        )
        if verbose:
            print(
                f"Window #{w.window_id} [{w.test_start.date()}→{w.test_end.date()}]: "
                f"params={best_params} val_ic={val_ic:+.4f} test_ic={test_ic:+.4f}",
                flush=True,
            )
        if save_models:
            os.makedirs(MODELS_DIR, exist_ok=True)
            model.save_model(
                os.path.join(MODELS_DIR, f"xgb_wf_{w.test_start.date()}.json")
            )

    return pd.concat(oos_parts, ignore_index=True), info


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--smoke", action="store_true",
        help="Tiny windows + 2-config grid — plumbing test, metrics meaningless.",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="Real windows, fixed params, no grid search — fast iteration.",
    )
    args = ap.parse_args()

    panel = load_panel()

    if args.smoke:
        windows = walk_forward_windows(
            panel["date"], min_train_days=40, val_days=15, test_days=15,
            purge_gap=5, min_test_days=5,
        )
        grid = GRID[:2]
    else:
        windows = walk_forward_windows(panel["date"])
        grid = [QUICK_PARAMS] if args.quick else GRID

    if not windows:
        raise SystemExit(
            "Not enough history for a single walk-forward window — "
            "extend the backfill (data.py --backfill --years N) or use --smoke."
        )

    oos, info = run_walk_forward(
        panel, windows, seed=args.seed, grid=grid, save_models=True
    )

    os.makedirs(os.path.dirname(OOS_PATH), exist_ok=True)
    tmp = OOS_PATH + ".tmp"
    oos.to_parquet(tmp)
    os.replace(tmp, OOS_PATH)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(info, f, indent=2)

    imp = pd.DataFrame([i["importance"] for i in info]).mean().sort_values(ascending=False)
    imp.rename("gain").to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"))

    pooled_ic = daily_ic(oos["date"], oos["y_true_excess"], oos["y_pred"].to_numpy())
    print(
        f"\nPooled OOS: {len(oos):,} predictions across {len(windows)} windows "
        f"({oos['date'].min().date()} → {oos['date'].max().date()}), "
        f"pooled IC={pooled_ic:+.4f}",
        flush=True,
    )
    print(f"Wrote {OOS_PATH}, {PARAMS_PATH}, feature_importance.csv", flush=True)
    print("\nTop 10 features by mean gain:")
    print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
