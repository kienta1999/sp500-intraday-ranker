#!/usr/bin/env python3
"""Portfolio backtest on the pooled walk-forward OOS predictions.

Strategy: every REBALANCE_DAYS (5) sessions, rank that day's 15:25 decision
rows by y_pred and hold the equal-weight top TOP_N (10). Two fill modes,
both reported (the gap between them measures overnight-drift dependence):
    next_open      fill at the NEXT session's open (first RTH bar's Open) —
                   the conservative, binding variant for gate 2
    same_day_1535  fill at the same day's 15:35 print (15:35 bar's Open)

Costs: COST_PER_ORDER dollars per order + SPREAD_BPS basis points of traded
notional per side.

Baselines (same simulator, same costs):
    momentum   12-1 momentum top-10 (close[t-21]/close[t-252] − 1)
    random     10 uniformly-random names, 20 seeds → mean + 10/90 band
    SPY        buy-and-hold

Rebalance-offset robustness: the model strategy is run at 5 schedule offsets
(0–4); stats report the mean and the band.

Reads:  data/processed/oos_predictions.parquet + raw bar caches (prices)
Writes: reports/backtest_equity.csv / .png, reports/backtest_stats.json

CLI:
    python scripts/backtest.py
    python scripts/backtest.py --capital 50000
"""

import argparse
import json
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

from data import NY_TZ, load_bars, load_daily, load_spy_daily  # noqa: E402
from strategy import (  # noqa: E402
    COST_PER_ORDER,
    DEFAULT_CAPITAL,
    DEFAULT_SEED,
    REBALANCE_DAYS,
    SPREAD_BPS,
    TOP_N,
)
from train import OOS_PATH  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")

N_RANDOM_SEEDS = 20
N_OFFSETS = 5
FILL_1535 = "15:35"  # NY-local bar start whose Open is the same-day fill print


# ─────────────────────────────────────────────────────────────────────────────
# Price matrices
# ─────────────────────────────────────────────────────────────────────────────


def build_price_matrices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """date × ticker matrices over each ticker's FULL cached history:
    close (session close), open (session open), p1535 (15:35 print).
    Full history (not just OOS) so the 12-1 momentum baseline has its
    252-session lookback."""
    closes, opens, p1535s = {}, {}, {}
    for t in tqdm(tickers, desc="Prices"):
        daily = load_daily(t)
        if daily is None or daily.empty:
            continue
        closes[t] = daily["Close"]
        opens[t] = daily["Open"]
        bars = load_bars(t)
        local = bars.index.tz_convert(NY_TZ)
        at = bars[local.strftime("%H:%M") == FILL_1535]
        s = pd.Series(at["Open"].to_numpy(),
                      index=pd.to_datetime(at.index.tz_convert(NY_TZ).date))
        p1535s[t] = s[~s.index.duplicated(keep="last")]
    return {
        "close": pd.DataFrame(closes).sort_index(),
        "open": pd.DataFrame(opens).sort_index(),
        "p1535": pd.DataFrame(p1535s).sort_index(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────


def simulate(
    picks_by_date: dict[pd.Timestamp, list[str]],
    prices: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    fill_mode: str = "next_open",
    capital: float = DEFAULT_CAPITAL,
    cost_per_order: float = COST_PER_ORDER,
    spread_bps: float = SPREAD_BPS,
) -> tuple[pd.Series, float]:
    """Daily-NAV simulation. Returns (NAV series on `calendar`, mean turnover
    per rebalance). `picks_by_date` maps DECISION dates → target baskets."""
    close, open_, p1535 = prices["close"], prices["open"], prices["p1535"]
    cal = list(calendar)
    idx_of = {d: i for i, d in enumerate(cal)}

    # Decision date → (fill date, fill-price row getter)
    fills: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
    for d, basket in picks_by_date.items():
        if fill_mode == "next_open":
            i = idx_of.get(d)
            if i is None or i + 1 >= len(cal):
                continue
            fd = cal[i + 1]
            fills[fd] = (d, open_.loc[fd] if fd in open_.index else None)
        else:  # same_day_1535
            if d not in idx_of:
                continue
            row = p1535.loc[d] if d in p1535.index else None
            # Half-days have no 15:35 bar — fall back to the session close.
            if row is None or row.isna().all():
                row = close.loc[d] if d in close.index else None
            fills[d] = (d, row)

    shares: dict[str, float] = {}
    cash = capital
    nav_path: list[float] = []
    turnovers: list[float] = []

    for d in cal:
        px_close = close.loc[d] if d in close.index else None

        if d in fills and fills[d][1] is not None:
            decision_date, fill_px = fills[d]
            basket = [
                t for t in picks_by_date[decision_date]
                if t in fill_px.index and pd.notna(fill_px[t])
            ]
            if basket:
                # Mark existing book at fill prices (last close where missing).
                value = cash
                for t, sh in shares.items():
                    p = fill_px.get(t)
                    if pd.isna(p) or p is None:
                        p = close[t].loc[:d].dropna().iloc[-1] if t in close else 0.0
                    value += sh * float(p)

                target_val = value / len(basket)
                new_shares = {t: target_val / float(fill_px[t]) for t in basket}

                traded = 0.0
                n_orders = 0
                for t in set(shares) | set(new_shares):
                    old = shares.get(t, 0.0)
                    new = new_shares.get(t, 0.0)
                    p = fill_px.get(t)
                    if pd.isna(p) or p is None:
                        p = close[t].loc[:d].dropna().iloc[-1] if t in close else 0.0
                    delta_notional = abs(new - old) * float(p)
                    if delta_notional > 1e-9:
                        traded += delta_notional
                        n_orders += 1
                costs = n_orders * cost_per_order + traded * spread_bps / 1e4
                turnovers.append(traded / value if value > 0 else 0.0)

                spent = sum(new_shares[t] * float(fill_px[t]) for t in basket)
                cash = value - spent - costs
                shares = new_shares

        nav = cash
        for t, sh in shares.items():
            if px_close is not None and t in px_close.index and pd.notna(px_close[t]):
                p = float(px_close[t])
            else:
                hist = close[t].loc[:d].dropna() if t in close else pd.Series(dtype=float)
                p = float(hist.iloc[-1]) if len(hist) else 0.0
            nav += sh * p
        nav_path.append(nav)

    return pd.Series(nav_path, index=calendar, name="nav"), float(np.mean(turnovers or [0.0]))


def compute_stats(nav: pd.Series) -> dict:
    ret = nav.pct_change().dropna()
    n_years = len(nav) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return {
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy + baseline pick builders
# ─────────────────────────────────────────────────────────────────────────────


def model_picks(oos: pd.DataFrame, rebalance_dates: list[pd.Timestamp],
                top_n: int = TOP_N) -> dict:
    out = {}
    by_date = dict(tuple(oos.groupby("date")))
    for d in rebalance_dates:
        day = by_date.get(d)
        if day is not None and len(day) >= top_n:
            out[d] = day.nlargest(top_n, "y_pred")["ticker"].tolist()
    return out


def momentum_picks(close: pd.DataFrame, rebalance_dates: list[pd.Timestamp],
                   universe_by_date: dict, top_n: int = TOP_N) -> dict:
    """12-1 momentum: close[t-21]/close[t-252] − 1, ranked among that day's
    predictable universe."""
    mom = close.shift(21) / close.shift(252) - 1
    out = {}
    for d in rebalance_dates:
        if d not in mom.index:
            continue
        candidates = [t for t in universe_by_date.get(d, []) if t in mom.columns]
        row = mom.loc[d, candidates].dropna()
        if len(row) >= top_n:
            out[d] = row.nlargest(top_n).index.tolist()
    return out


def random_picks(rebalance_dates: list[pd.Timestamp], universe_by_date: dict,
                 seed: int, top_n: int = TOP_N) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for d in rebalance_dates:
        pool = universe_by_date.get(d, [])
        if len(pool) >= top_n:
            out[d] = list(rng.choice(pool, size=top_n, replace=False))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def run_backtest(
    oos: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    capital: float = DEFAULT_CAPITAL,
    cost_per_order: float = COST_PER_ORDER,
    spread_bps: float = SPREAD_BPS,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run all variants. Returns (equity paths DataFrame, stats dict).
    Exposed for evaluate.py's cost-sensitivity table."""
    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    dates = sorted(oos["date"].unique())
    calendar = pd.DatetimeIndex(dates)
    universe_by_date = oos.groupby("date")["ticker"].agg(list).to_dict()

    kw = dict(capital=capital, cost_per_order=cost_per_order, spread_bps=spread_bps)
    equity: dict[str, pd.Series] = {}
    stats: dict[str, dict] = {}

    # Model, both fill modes, offset 0 (headline) + offset band on next_open.
    offset_navs = []
    for off in range(N_OFFSETS):
        reb = list(dates[off::REBALANCE_DAYS])
        picks = model_picks(oos, reb)
        nav, to = simulate(picks, prices, calendar, fill_mode="next_open", **kw)
        offset_navs.append(nav)
        if off == 0:
            equity["model_next_open"] = nav
            stats["model_next_open"] = compute_stats(nav) | {"avg_turnover": round(to, 3)}
            nav_sd, to_sd = simulate(picks, prices, calendar,
                                     fill_mode="same_day_1535", **kw)
            equity["model_same_day_1535"] = nav_sd
            stats["model_same_day_1535"] = compute_stats(nav_sd) | {"avg_turnover": round(to_sd, 3)}

    offs = pd.concat(offset_navs, axis=1)
    stats["model_next_open"]["offset_band_total_return"] = [
        round(float(x), 4) for x in
        (offs.iloc[-1] / offs.iloc[0] - 1).quantile([0.1, 0.5, 0.9]).tolist()
    ]

    reb0 = list(dates[0::REBALANCE_DAYS])

    picks = momentum_picks(prices["close"], reb0, universe_by_date)
    nav, to = simulate(picks, prices, calendar, fill_mode="next_open", **kw)
    equity["momentum_12_1"] = nav
    stats["momentum_12_1"] = compute_stats(nav) | {"avg_turnover": round(to, 3)}

    rand_navs = []
    for s in range(N_RANDOM_SEEDS):
        picks = random_picks(reb0, universe_by_date, seed=DEFAULT_SEED + s)
        nav, _ = simulate(picks, prices, calendar, fill_mode="next_open", **kw)
        rand_navs.append(nav)
    rand = pd.concat(rand_navs, axis=1)
    equity["random_10_mean"] = rand.mean(axis=1)
    stats["random_10"] = compute_stats(equity["random_10_mean"]) | {
        "total_return_band_10_90": [
            round(float(x), 4) for x in
            (rand.iloc[-1] / rand.iloc[0] - 1).quantile([0.1, 0.9]).tolist()
        ]
    }

    spy = load_spy_daily()["Close"].reindex(calendar).ffill()
    equity["spy"] = spy / spy.iloc[0] * capital
    stats["spy"] = compute_stats(equity["spy"])

    eq = pd.DataFrame(equity)
    if not quiet:
        for name, st in stats.items():
            print(f"  {name:<22} {st}", flush=True)
    return eq, stats


def plot_equity(eq: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    styles = {
        "model_next_open": dict(color="tab:blue", lw=2),
        "model_same_day_1535": dict(color="tab:cyan", lw=1.5, ls="--"),
        "momentum_12_1": dict(color="tab:orange", lw=1.5),
        "random_10_mean": dict(color="tab:gray", lw=1),
        "spy": dict(color="black", lw=1.5),
    }
    for col in eq.columns:
        ax.plot(eq.index, eq[col], label=col, **styles.get(col, {}))
    ax.set_yscale("log")
    ax.set_ylabel("NAV ($, log)")
    ax.set_title("Walk-forward OOS equity — top-10 weekly rebalance, net of costs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = ap.parse_args()

    if not os.path.exists(OOS_PATH):
        raise SystemExit(f"{OOS_PATH} not found. Run scripts/train.py first.")
    oos = pd.read_parquet(OOS_PATH)
    tickers = sorted(oos["ticker"].unique())
    print(f"OOS table: {len(oos):,} rows, {len(tickers)} tickers, "
          f"{oos['date'].min()} → {oos['date'].max()}", flush=True)

    prices = build_price_matrices(tickers)
    print("\nRunning variants...", flush=True)
    eq, stats = run_backtest(oos, prices, capital=args.capital)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    eq.to_csv(os.path.join(REPORTS_DIR, "backtest_equity.csv"))
    with open(os.path.join(REPORTS_DIR, "backtest_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    plot_equity(eq, os.path.join(REPORTS_DIR, "backtest_equity.png"))
    print(f"\nWrote reports/backtest_equity.csv/.png and backtest_stats.json", flush=True)


if __name__ == "__main__":
    main()
