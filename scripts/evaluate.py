#!/usr/bin/env python3
"""Evaluation gates + validation report on the pooled walk-forward OOS.

GATES (committed before any results existed — see README):
  1 ic          mean daily Spearman IC > 0.02 AND Newey-West(lag=5) t > 2
                (overlapping 5-day labels autocorrelate; plain t also reported)
  2 portfolio   model top-10 (next-open fills, net of costs) beats SPY AND the
                12-1 momentum baseline on total OOS return
  3 permutation real pooled IC > 95th percentile of N_PERM within-date
                label-permutation runs (refit with the SAME per-window params —
                no re-tuning on shuffled data)
  4 ablation    (informational) ΔIC from dropping volume / rank feature groups
  5 decay       (informational) IC of frozen predictions vs realized
                1/3/5/10/21-day excess returns

Decision rule: fail gate 1 or 2 → improve features or accept the momentum
baseline; do NOT deploy the ML anyway.

Reads:  oos_predictions.parquet, walkforward_params.json, panel.parquet,
        backtest_stats.json (+ price caches for decay & cost sensitivity)
Writes: reports/validation.html (self-contained), validation_metrics.json,
        individual .png figures

CLI:
    python scripts/evaluate.py
    python scripts/evaluate.py --n-perm 5 --skip-ablation   # faster pass
"""

import argparse
import base64
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtest import build_price_matrices, model_picks, simulate  # noqa: E402
from dataset import load_panel, walk_forward_windows  # noqa: E402
from features import (  # noqa: E402
    ALL_FEATURES,
    INTRADAY_FEATURES,
    LONG_FEATURES,
    RANK_FEATURES,
    VOLUME_FEATURES,
)
from labels import FORWARD_DAYS, LABEL_COL  # noqa: E402
from strategy import DEFAULT_SEED, REBALANCE_DAYS, TOP_N  # noqa: E402
from train import (  # noqa: E402
    OOS_PATH,
    PARAMS_PATH,
    daily_ic_series,
    run_walk_forward,
)

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

GATES = {
    "ic_mean_min": 0.02,
    "ic_tstat_min": 2.0,
    "perm_percentile": 95,
}
N_PERM = 20
DECAY_HORIZONS = (1, 3, 5, 10, 21)
COST_GRID_ORDERS = (0.0, 1.0, 2.0)
COST_GRID_BPS = (0.0, 3.0, 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: IC + Newey-West t-stat
# ─────────────────────────────────────────────────────────────────────────────


def newey_west_tstat(x: pd.Series, lags: int = 5) -> float:
    """t-stat of mean(x) with Newey-West (Bartlett) HAC standard errors."""
    x = x.dropna().to_numpy()
    n = len(x)
    if n < lags + 2:
        return np.nan
    e = x - x.mean()
    lrv = float(e @ e) / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        lrv += 2 * w * float(e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(lrv / n))


def ic_gate(oos: pd.DataFrame) -> tuple[dict, pd.Series]:
    ics = daily_ic_series(oos["date"], oos["y_true_excess"], oos["y_pred"].to_numpy())
    mean_ic = float(ics.mean())
    t_nw = newey_west_tstat(ics, lags=FORWARD_DAYS)
    t_plain = float(mean_ic / (ics.std() / np.sqrt(len(ics))))
    res = {
        "mean_ic": round(mean_ic, 4),
        "tstat_newey_west_5": round(t_nw, 2),
        "tstat_plain": round(t_plain, 2),
        "n_days": int(len(ics)),
        "pass": bool(mean_ic > GATES["ic_mean_min"] and t_nw > GATES["ic_tstat_min"]),
    }
    return res, ics


# ─────────────────────────────────────────────────────────────────────────────
# Gate 3: permutation control
# ─────────────────────────────────────────────────────────────────────────────


def _permute_within_date(panel: pd.DataFrame, col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = panel.copy()
    out[col] = (
        out.groupby("date")[col]
        .transform(lambda s: rng.permutation(s.to_numpy()))
    )
    return out


def permutation_gate(
    panel: pd.DataFrame, windows, params_per_window: list[dict],
    real_ic: float, n_perm: int = N_PERM,
) -> dict:
    perm_ics: list[float] = []
    for i in range(n_perm):
        shuffled = _permute_within_date(panel, LABEL_COL, seed=DEFAULT_SEED + 1000 + i)
        oos, _ = run_walk_forward(
            shuffled, windows, params_per_window=params_per_window, verbose=False,
        )
        ic = float(
            daily_ic_series(oos["date"], oos["y_true_excess"], oos["y_pred"].to_numpy()).mean()
        )
        perm_ics.append(ic)
        print(f"  permutation {i + 1}/{n_perm}: IC={ic:+.4f}", flush=True)
    threshold = float(np.percentile(perm_ics, GATES["perm_percentile"]))
    return {
        "real_ic": round(real_ic, 4),
        "perm_ic_p95": round(threshold, 4),
        "perm_ic_max": round(max(perm_ics), 4),
        "n_perm": n_perm,
        "pass": bool(real_ic > threshold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gate 4: ablations
# ─────────────────────────────────────────────────────────────────────────────


def ablation_study(
    panel: pd.DataFrame, windows, params_per_window: list[dict], real_ic: float,
) -> dict:
    out = {"full_ic": round(real_ic, 4)}

    def _without(group: list[str]) -> list[str]:
        dropped = set(group) | {f"{g}_rank" for g in group}
        return [f for f in ALL_FEATURES if f not in dropped]

    groups = {
        "drop_volume": _without(VOLUME_FEATURES),
        "drop_intraday": _without(INTRADAY_FEATURES),
        "drop_long": _without(LONG_FEATURES),
        "drop_ranks": [f for f in ALL_FEATURES if f not in RANK_FEATURES],
    }
    for name, cols in groups.items():
        oos, _ = run_walk_forward(
            panel, windows, params_per_window=params_per_window,
            feature_cols=cols, verbose=False,
        )
        ic = float(
            daily_ic_series(oos["date"], oos["y_true_excess"], oos["y_pred"].to_numpy()).mean()
        )
        out[name] = {"ic": round(ic, 4), "delta_vs_full": round(ic - real_ic, 4)}
        print(f"  ablation {name}: IC={ic:+.4f} (Δ={ic - real_ic:+.4f})", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gate 5: decay curve
# ─────────────────────────────────────────────────────────────────────────────


def decay_curve(oos: pd.DataFrame, close: pd.DataFrame, spy_close: pd.Series) -> dict:
    """IC of the frozen 5d-trained predictions vs realized h-day excess returns."""
    out = {}
    for h in DECAY_HORIZONS:
        fwd = close.shift(-h) / close - 1
        spy_fwd = (spy_close.shift(-h) / spy_close - 1).reindex(fwd.index)
        excess = fwd.sub(spy_fwd, axis=0)
        stacked = excess.stack().rename("y_h").reset_index()
        stacked.columns = ["date", "ticker", "y_h"]
        merged = oos.merge(stacked, on=["date", "ticker"], how="inner").dropna(subset=["y_h"])
        ic = float(
            daily_ic_series(merged["date"], merged["y_h"], merged["y_pred"].to_numpy()).mean()
        )
        out[f"{h}d"] = round(ic, 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cost sensitivity (model next-open picks only)
# ─────────────────────────────────────────────────────────────────────────────


def cost_sensitivity(oos: pd.DataFrame, prices: dict) -> dict:
    dates = sorted(pd.to_datetime(oos["date"]).unique())
    calendar = pd.DatetimeIndex(dates)
    # Holding period follows the label horizon (matches backtest --rebalance-days).
    picks = model_picks(oos, list(dates[0::FORWARD_DAYS]))
    table = {}
    for cpo in COST_GRID_ORDERS:
        for bps in COST_GRID_BPS:
            nav, _ = simulate(picks, prices, calendar, fill_mode="next_open",
                              cost_per_order=cpo, spread_bps=bps)
            table[f"${cpo:.0f}/order + {bps:.0f}bps"] = round(
                float(nav.iloc[-1] / nav.iloc[0] - 1), 4
            )
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Figures + HTML
# ─────────────────────────────────────────────────────────────────────────────


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_ic_series(ics: pd.Series, path: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ics.index, ics.values, width=1.2, color="tab:blue", alpha=0.35, label="daily IC")
    ax.plot(ics.rolling(21).mean(), color="tab:red", lw=2, label="21d rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(GATES["ic_mean_min"], color="green", ls="--", lw=1, label="gate (0.02)")
    ax.set_title("Daily rank IC — pooled walk-forward OOS")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return _fig_to_b64(fig)


def plot_decay(decay: dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [int(k.rstrip("d")) for k in decay]
    ax.plot(xs, list(decay.values()), marker="o", color="tab:purple")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("horizon (trading days)")
    ax.set_ylabel("mean daily IC")
    ax.set_title("Signal decay — prediction IC vs realized h-day excess return")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return _fig_to_b64(fig)


def plot_equity_png_b64() -> str | None:
    p = os.path.join(REPORTS_DIR, "backtest_equity.png")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _dict_table(d: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items())
    return f"<table><tbody>{rows}</tbody></table>"


def write_html(metrics: dict, imgs: dict[str, str | None], importance: pd.Series | None) -> str:
    def img(key: str, title: str) -> str:
        if not imgs.get(key):
            return ""
        return f"<h2>{title}</h2><img src='data:image/png;base64,{imgs[key]}' style='max-width:100%'>"

    gate_rows = ""
    for name, res in metrics["gates"].items():
        ok = res.get("pass")
        badge = ("<b style='color:green'>PASS</b>" if ok
                 else "<b style='color:red'>FAIL</b>" if ok is False else "info")
        detail = {k: v for k, v in res.items() if k != "pass"}
        gate_rows += f"<tr><td>{name}</td><td>{badge}</td><td><code>{detail}</code></td></tr>"

    imp_html = ""
    if importance is not None:
        imp_html = "<h2>Feature importance (mean gain, top 15)</h2>" + \
            importance.head(15).to_frame("importance").to_html()

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>sp500-intraday-ranker — validation</title>
<style>
 body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }}
 table {{ border-collapse: collapse; margin: 0.5em 0; }}
 td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
 code {{ font-size: 0.85em; }}
</style></head><body>
<h1>Validation report — {metrics['generated']}</h1>
<p>Pooled walk-forward OOS: {metrics['oos_rows']:,} predictions,
{metrics['oos_range']}, {metrics['n_windows']} windows.</p>
<h2>Gates</h2>
<table><thead><tr><th>gate</th><th>result</th><th>detail</th></tr></thead>
<tbody>{gate_rows}</tbody></table>
{img('ic', 'IC time series')}
{img('equity', 'Equity curves (see backtest_stats.json for per-variant stats)')}
{img('decay', 'Signal decay')}
<h2>Cost sensitivity — model next-open total OOS return</h2>
{_dict_table(metrics['cost_sensitivity'])}
{imp_html}
</body></html>"""
    path = os.path.join(REPORTS_DIR, "validation.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Match train.py --smoke window geometry (plumbing test).")
    args = ap.parse_args()

    if not os.path.exists(OOS_PATH):
        raise SystemExit(f"{OOS_PATH} not found. Run scripts/train.py first.")
    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    with open(PARAMS_PATH) as f:
        wf_info = json.load(f)
    params_per_window = [i["params"] for i in wf_info]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics: dict = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "oos_rows": int(len(oos)),
        "oos_range": f"{oos['date'].min().date()} → {oos['date'].max().date()}",
        "n_windows": int(oos["window_id"].nunique()),
        "gates": {},
    }

    # Gate 1 — IC
    print("Gate 1: IC...", flush=True)
    g1, ics = ic_gate(oos)
    metrics["gates"]["1_ic"] = g1

    # Gate 2 — portfolio (from backtest_stats.json; run backtest.py first)
    stats_path = os.path.join(REPORTS_DIR, "backtest_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            bt = json.load(f)
        model_ret = bt["model_next_open"]["total_return"]
        g2 = {
            "model_total_return": model_ret,
            "spy_total_return": bt["spy"]["total_return"],
            "momentum_total_return": bt["momentum_12_1"]["total_return"],
            "model_same_day_total_return": bt["model_same_day_1535"]["total_return"],
            "pass": bool(
                model_ret > bt["spy"]["total_return"]
                and model_ret > bt["momentum_12_1"]["total_return"]
            ),
        }
    else:
        g2 = {"error": "backtest_stats.json missing — run scripts/backtest.py", "pass": False}
    metrics["gates"]["2_portfolio"] = g2

    # Shared refit ingredients for gates 3 & 4
    panel = load_panel()
    if args.smoke:
        windows = walk_forward_windows(
            panel["date"], min_train_days=40, val_days=15, test_days=15,
            purge_gap=5, min_test_days=5,
        )
    else:
        windows = walk_forward_windows(panel["date"])
    windows = windows[: len(params_per_window)]

    if not args.skip_perm:
        print(f"Gate 3: permutation control ({args.n_perm} refit runs)...", flush=True)
        metrics["gates"]["3_permutation"] = permutation_gate(
            panel, windows, params_per_window, g1["mean_ic"], n_perm=args.n_perm
        )
    if not args.skip_ablation:
        print("Gate 4: ablations...", flush=True)
        metrics["gates"]["4_ablation"] = ablation_study(
            panel, windows, params_per_window, g1["mean_ic"]
        )

    # Gate 5 — decay + cost sensitivity (need price matrices)
    print("Gate 5: decay curve + cost sensitivity...", flush=True)
    from data import load_spy_daily  # noqa: E402  (late import keeps top tidy)

    prices = build_price_matrices(sorted(oos["ticker"].unique()))
    decay = decay_curve(oos, prices["close"], load_spy_daily()["Close"])
    metrics["gates"]["5_decay"] = decay
    metrics["cost_sensitivity"] = cost_sensitivity(oos, prices)

    # Figures + report
    imgs = {
        "ic": plot_ic_series(ics, os.path.join(REPORTS_DIR, "ic_series.png")),
        "decay": plot_decay(decay, os.path.join(REPORTS_DIR, "decay_curve.png")),
        "equity": plot_equity_png_b64(),
    }
    imp_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    importance = (
        pd.read_csv(imp_path).set_index("feature")["importance"]
        if os.path.exists(imp_path) else None
    )
    html_path = write_html(metrics, imgs, importance)

    with open(os.path.join(REPORTS_DIR, "validation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {html_path} and validation_metrics.json", flush=True)
    hard_gates = [metrics["gates"].get("1_ic", {}), metrics["gates"].get("2_portfolio", {})]
    if "3_permutation" in metrics["gates"]:
        hard_gates.append(metrics["gates"]["3_permutation"])
    verdict = all(g.get("pass") for g in hard_gates)
    print(f"\nVERDICT: {'ALL HARD GATES PASS' if verdict else 'GATES FAILED'} "
          f"— {'proceed to phase 4 (paper trading)' if verdict else 'improve features or accept the momentum baseline'}",
          flush=True)


if __name__ == "__main__":
    main()
