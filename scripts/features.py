#!/usr/bin/env python3
"""Compute intraday features from cached 5-min bars.

Two outputs:
  * data/processed/features/{TICKER}.parquet — full-resolution per-ticker
    feature frames (every RTH bar), used by diagnostics and by the lookahead
    test in dataset.py.
  * data/processed/sampled_features.parquet — ONE decision row per
    (ticker, session): the bar starting SAMPLE_TIME (15:25 ET, data through
    15:30 — late enough to act the same day, early enough to get filled),
    falling back to the session's last bar on half-days. Cross-sectional
    percentile ranks are added on this slice (they need the full universe at
    a shared timestamp). labels.py attaches the target to this file.

Every feature uses only bars <= its own timestamp — daily aggregates
(ret_5d, dist_mean_20d, volume baselines) are built from *prior completed
sessions* via shift(1)-style indexing. dataset.py's assert_no_lookahead
recomputes sampled rows from truncated raw bars to enforce this.

minutes_to_close uses the scheduled 16:00 close, not the session's actual
last bar — knowing the actual close time would leak on half-days, and the
scheduled value is what's known at decision time.

Feature buckets are exported as module constants; train/dataset/strategy
import them from here (never via each other — avoids circular imports).

CLI:
    python scripts/features.py                    # all cached universe tickers
    python scripts/features.py --tickers AAPL,MSFT
    python scripts/features.py --ticker AAPL      # smoke: print last rows, no write
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

from data import NY_TZ, cached_tickers, load_bars  # noqa: E402
from universe import load_universe  # noqa: E402

_ROOT = os.path.dirname(_HERE)
FEATURES_DIR = os.path.join(_ROOT, "data", "processed", "features")
SAMPLED_PATH = os.path.join(_ROOT, "data", "processed", "sampled_features.parquet")

SAMPLE_TIME = "15:25"  # NY-local bar start; the 15:25 bar carries data through 15:30
BARS_PER_DAY = 78
ANNUALIZER = np.sqrt(BARS_PER_DAY * 252)
MIN_SESSIONS = 60  # skip tickers with less history than the deepest daily window

# ─────────────────────────────────────────────────────────────────────────────
# Feature buckets
# ─────────────────────────────────────────────────────────────────────────────

# Buckets are organized by TIMESCALE. Coverage is deliberately densest in the
# 1–50 day band (0.5×–10× the 5-day label horizon — the scale the label lives
# on), with intraday features for event/abnormal-day detection and a light
# long-scale group for trend/regime context. evaluate.py ablates the intraday
# and long groups so the first real run quantifies each group's contribution.

INTRADAY_FEATURES = [
    # minutes → 1 day, computed on 5-min bars
    "ret_12b", "ret_26b", "ret_78b",   # 1h / ~2h / 1-day trailing returns
    "rsi_14b",                         # Wilder RSI over 14 bars (~70 min)
    "rvol_26b", "rvol_78b",            # std of 1-bar returns, annualized
    "atr14b_pct",                      # ATR(14 bars, Wilder) as % of price
    "range_expansion",                 # bar range / 20-bar mean range
    "range_pos_78b",                   # position in trailing-78-bar high/low range
    "dist_vwap",                       # price / cumulative session VWAP − 1
    "rel_vol_tod",                     # bar volume / same time-of-day mean, prior 20 sessions
    "cum_vol_vs_20d",                  # session cum volume vs same point-in-day, prior 20 sessions
    "volume_price_corr_26b",           # corr(volume, close) over trailing 26 bars
]
HORIZON_FEATURES = [
    # 1–50 days — the core band for a 5d label. Daily aggregates use prior
    # COMPLETED sessions (shift(1)); "px" terms use the live bar price.
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_21d",  # vs close k sessions ago
    "gap_1d",                          # today's session open / prior close − 1
    "up_streak_d",                     # signed count of consecutive up/down days
    "dist_mean_20d", "dist_mean_50d",  # price / mean of prior 20/50 daily closes − 1
    "range_pos_20d",                   # position in prior-20-session high/low range
    "rvol_5d", "rvol_10d",             # std of daily returns, annualized
    "vol_5d_vs_20d",                   # mean daily volume 5d vs 20d
    "rsi_14d",                         # Wilder RSI, 14 completed daily closes
    "macd_hist",                       # (MACD 12/26 − signal 9) / price, daily
    "zscore_20d",                      # (price − mean20d) / std20d of prior closes
    "bb_width_20d",                    # Bollinger band width (squeeze detector)
    "cmf_20d",                         # Chaikin Money Flow, 20 sessions
    "obv_zscore_20d",                  # On-Balance Volume z-score vs its 20d history
    "mfi_14d",                         # Money Flow Index, 14 sessions
    "cci_20d",                         # Commodity Channel Index, 20 sessions
    "aroon_25d",                       # Aroon oscillator, 25 sessions
]
LONG_FEATURES = [
    # 63–252 days — light regime/trend context ("see after training").
    "ret_63d", "ret_126d",             # quarterly / half-year momentum
    "mom_12_1",                        # close[t-21]/close[t-252] − 1 (classic 12-1)
    "dist_52w_high",                   # price / prior-252-session high − 1
    "dist_mean_200d",                  # price / mean of prior 200 daily closes − 1
]
TIME_FEATURES = [
    "bar_of_day", "day_of_week", "minutes_since_open", "minutes_to_close",
]

# Ablation grouping (overlaps the timescale buckets): everything volume-driven.
VOLUME_FEATURES = [
    "rel_vol_tod", "cum_vol_vs_20d", "volume_price_corr_26b",
    "vol_5d_vs_20d", "cmf_20d", "obv_zscore_20d", "mfi_14d", "dollar_vol_rank",
]

# Cross-sectional percentile ranks (added on the sampled slice, where the full
# universe shares a timestamp). dollar_vol_rank is already a rank; time
# features are calendar facts — neither gets re-ranked.
RANKABLE = INTRADAY_FEATURES + HORIZON_FEATURES + LONG_FEATURES
RANK_FEATURES = [f"{c}_rank" for c in RANKABLE]

ALL_FEATURES = (
    INTRADAY_FEATURES + HORIZON_FEATURES + LONG_FEATURES
    + ["dollar_vol_rank"] + TIME_FEATURES + RANK_FEATURES
)

# Long features need 252 completed sessions of warmup — a recent listing would
# lose its entire history to a NaN-dropna. XGBoost handles missing natively,
# so these (and their ranks) are exempt from dataset.py's dropna gate.
NULLABLE_FEATURES = LONG_FEATURES + [f"{c}_rank" for c in LONG_FEATURES]

# Features recomputable from a single ticker's truncated raw bars — what the
# lookahead test verifies. Excludes cross-sectional columns (need the full
# universe, but are timestamp-aligned so not a leak vector).
PER_TICKER_FEATURES = (
    INTRADAY_FEATURES + HORIZON_FEATURES + LONG_FEATURES + TIME_FEATURES
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All per-ticker features for one ticker's full 5-min RTH bar history.

    `bars`: tz-aware UTC index, Open/High/Low/Close/Volume (+ optional VWAP).
    Returns a frame on the same index with feature columns plus auxiliary
    columns: session (naive session date), close, dollar_vol_21d.
    """
    local = bars.index.tz_convert(NY_TZ)
    session = pd.Series(pd.to_datetime(local.date), index=bars.index)
    tod = pd.Series(local.strftime("%H:%M"), index=bars.index)
    px = bars["Close"]
    vol = bars["Volume"].astype(float)

    out = pd.DataFrame(index=bars.index)
    out["session"] = session
    out["close"] = px

    # ── Trailing bar returns ──
    for k in (12, 26, 78):
        out[f"ret_{k}b"] = px.pct_change(k)

    # ── Daily aggregates from prior COMPLETED sessions ──
    # daily_*[i] = session i's aggregate; a bar in session i may only reference
    # daily aggregates up to session i-1 (shift >= 1 below). The exception is
    # gap_1d's session OPEN — the first bar of the bar's own session is past
    # data by construction.
    grouped = bars.groupby(session.values)
    daily_close = grouped["Close"].last()
    daily_close.index = pd.DatetimeIndex(daily_close.index)
    daily_idx = daily_close.index
    daily_high = grouped["High"].max().set_axis(daily_idx)
    daily_low = grouped["Low"].min().set_axis(daily_idx)
    daily_open = grouped["Open"].first().set_axis(daily_idx)
    daily_vol = grouped["Volume"].sum().astype(float).set_axis(daily_idx)
    daily_ret = daily_close.pct_change()
    sess_of_bar = session.values  # aligns bars → their session date

    def _map_daily(s: pd.Series) -> np.ndarray:
        return s.reindex(sess_of_bar).to_numpy()

    for k in (1, 3, 5, 10, 21, 63, 126):
        out[f"ret_{k}d"] = px.to_numpy() / _map_daily(daily_close.shift(k)) - 1.0
    out["mom_12_1"] = _map_daily(daily_close.shift(21) / daily_close.shift(252) - 1.0)
    for k in (20, 50, 200):
        mean_k = daily_close.shift(1).rolling(k).mean()
        out[f"dist_mean_{k}d"] = px.to_numpy() / _map_daily(mean_k) - 1.0
    out["dist_52w_high"] = (
        px.to_numpy() / _map_daily(daily_high.shift(1).rolling(252).max()) - 1.0
    )

    # Horizon-band price/vol dynamics
    out["gap_1d"] = _map_daily(daily_open) / _map_daily(daily_close.shift(1)) - 1.0
    sgn = np.sign(daily_ret)
    streak_grp = (sgn != sgn.shift()).cumsum()
    streak = (sgn.groupby(streak_grp).cumcount() + 1) * sgn
    out["up_streak_d"] = _map_daily(streak.shift(1))
    hi20 = daily_high.shift(1).rolling(20).max()
    lo20 = daily_low.shift(1).rolling(20).min()
    out["range_pos_20d"] = (px.to_numpy() - _map_daily(lo20)) / _map_daily(
        (hi20 - lo20).replace(0.0, np.nan)
    )
    for k in (5, 10):
        out[f"rvol_{k}d"] = _map_daily(
            daily_ret.rolling(k).std().shift(1)
        ) * np.sqrt(252)
    out["vol_5d_vs_20d"] = _map_daily(
        (daily_vol.rolling(5).mean() / daily_vol.rolling(20).mean()).shift(1)
    )

    # ── Intraday VWAP distance ──
    # Cumulative from the session open. Alpaca supplies a per-bar VWAP; fall
    # back to typical price if a cache predates that column.
    bar_vwap = bars["VWAP"] if "VWAP" in bars.columns else (
        (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    )
    pv = (bar_vwap * vol).groupby(session.values).cumsum()
    cv = vol.groupby(session.values).cumsum()
    sess_vwap = pv / cv.replace(0.0, np.nan)
    out["dist_vwap"] = px / sess_vwap - 1.0

    # ── Range position ──
    hi78 = bars["High"].rolling(78).max()
    lo78 = bars["Low"].rolling(78).min()
    rng = (hi78 - lo78).replace(0.0, np.nan)
    out["range_pos_78b"] = (px - lo78) / rng

    # ── Volatility ──
    r1 = px.pct_change()
    out["rvol_26b"] = r1.rolling(26).std() * ANNUALIZER
    out["rvol_78b"] = r1.rolling(78).std() * ANNUALIZER
    prev_close = px.shift(1)
    tr = pd.concat(
        [
            bars["High"] - bars["Low"],
            (bars["High"] - prev_close).abs(),
            (bars["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14b_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / px
    bar_range = bars["High"] - bars["Low"]
    out["range_expansion"] = bar_range / bar_range.rolling(20).mean().replace(0.0, np.nan)

    # ── Volume (time-of-day normalized — intraday volume is U-shaped, so raw
    # ratios are meaningless without matching the same bar across sessions) ──
    out["rel_vol_tod"] = vol / vol.groupby(tod.values).transform(
        lambda s: s.shift(1).rolling(20).mean()
    ).replace(0.0, np.nan)
    cumvol = vol.groupby(session.values).cumsum()
    out["cum_vol_vs_20d"] = cumvol / cumvol.groupby(tod.values).transform(
        lambda s: s.shift(1).rolling(20).mean()
    ).replace(0.0, np.nan)
    out["volume_price_corr_26b"] = vol.rolling(26).corr(px)

    # ── Oscillators ──
    def _wilder_rsi(s: pd.Series, period: int) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        return 100 - 100 / (1 + up / dn.replace(0.0, np.nan))

    out["rsi_14b"] = _wilder_rsi(px, 14)
    out["rsi_14d"] = _map_daily(_wilder_rsi(daily_close, 14).shift(1))
    ema12 = daily_close.ewm(span=12, adjust=False).mean()
    ema26 = daily_close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = _map_daily(((macd - signal) / daily_close).shift(1))
    mean20 = daily_close.shift(1).rolling(20).mean()
    std20 = daily_close.shift(1).rolling(20).std()
    out["zscore_20d"] = (px.to_numpy() - _map_daily(mean20)) / _map_daily(
        std20.replace(0.0, np.nan)
    )
    out["bb_width_20d"] = _map_daily(4.0 * std20 / mean20)

    # Daily volume-flow oscillators (all shift(1) → completed sessions only)
    rng_d = (daily_high - daily_low).replace(0.0, np.nan)
    mfm = ((daily_close - daily_low) - (daily_high - daily_close)) / rng_d
    out["cmf_20d"] = _map_daily(
        ((mfm * daily_vol).rolling(20).sum() / daily_vol.rolling(20).sum()).shift(1)
    )
    obv = (sgn.fillna(0.0) * daily_vol).cumsum()
    out["obv_zscore_20d"] = _map_daily(
        ((obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0.0, np.nan)).shift(1)
    )
    tp = (daily_high + daily_low + daily_close) / 3.0
    mf = tp * daily_vol
    d_tp = tp.diff()
    pos_mf = mf.where(d_tp > 0, 0.0).rolling(14).sum()
    neg_mf = mf.where(d_tp < 0, 0.0).rolling(14).sum().replace(0.0, np.nan)
    out["mfi_14d"] = _map_daily((100 - 100 / (1 + pos_mf / neg_mf)).shift(1))
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["cci_20d"] = _map_daily(
        ((tp - sma_tp) / (0.015 * mad_tp.replace(0.0, np.nan))).shift(1)
    )
    days_since_high = daily_high.rolling(25).apply(
        lambda x: len(x) - 1 - np.argmax(x), raw=True
    )
    days_since_low = daily_low.rolling(25).apply(
        lambda x: len(x) - 1 - np.argmin(x), raw=True
    )
    out["aroon_25d"] = _map_daily(
        (((25 - days_since_high) - (25 - days_since_low)) / 25 * 100).shift(1)
    )

    # Auxiliary: 21d mean daily dollar volume through session t-1 (ranked
    # cross-sectionally later as dollar_vol_rank).
    daily_dollar = (px * vol).groupby(session.values).sum()
    daily_dollar.index = daily_close.index
    out["dollar_vol_21d"] = _map_daily(daily_dollar.shift(1).rolling(21).mean())

    # ── Time ──
    out["bar_of_day"] = out.groupby(session.values).cumcount()
    out["day_of_week"] = local.dayofweek
    minutes = local.hour * 60 + local.minute
    out["minutes_since_open"] = minutes - (9 * 60 + 30)
    out["minutes_to_close"] = (16 * 60) - minutes  # scheduled close (see docstring)

    return out


def sample_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """One decision row per session: the SAMPLE_TIME bar, else the last bar."""
    local = feats.index.tz_convert(NY_TZ)
    tod = pd.Series(local.strftime("%H:%M"), index=feats.index)
    at_sample = feats[tod == SAMPLE_TIME]
    # Half-days (or data gaps) have no 15:25 bar — take the session's last bar.
    missing = feats[~feats["session"].isin(at_sample["session"])]
    fallback = missing.groupby(missing["session"].values).tail(1)
    out = pd.concat([at_sample, fallback]).sort_index()
    out = out.reset_index().rename(columns={"date": "timestamp", "session": "date"})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional ranks + build
# ─────────────────────────────────────────────────────────────────────────────


def add_cross_sectional_ranks(sampled: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank features across the universe on each session date.

    The sampled slice is date-aligned (everyone's row is the same decision
    bar), so ranking by `date` == ranking at a shared timestamp.
    """
    g = sampled.groupby("date")
    for col in RANKABLE:
        sampled[f"{col}_rank"] = g[col].rank(pct=True)
    sampled["dollar_vol_rank"] = g["dollar_vol_21d"].rank(pct=True)
    return sampled


def build(tickers: list[str]) -> pd.DataFrame:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    sampled_parts: list[pd.DataFrame] = []
    skipped: list[str] = []

    for t in tqdm(tickers, desc="Features"):
        bars = load_bars(t)
        if bars is None or bars.empty:
            skipped.append(t)
            continue
        feats = compute_features(bars)
        if feats["session"].nunique() < MIN_SESSIONS:
            skipped.append(t)
            continue
        tmp = os.path.join(FEATURES_DIR, f"{t}.parquet.tmp")
        feats.to_parquet(tmp)
        os.replace(tmp, os.path.join(FEATURES_DIR, f"{t}.parquet"))
        part = sample_rows(feats)
        part.insert(0, "ticker", t)
        sampled_parts.append(part)

    if skipped:
        print(f"Skipped {len(skipped)} tickers (no/short bar cache): "
              + ", ".join(skipped), flush=True)
    if not sampled_parts:
        raise SystemExit("No tickers produced features — run data.py first.")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = add_cross_sectional_ranks(sampled)
    sampled = sampled.sort_values(["date", "ticker"]).reset_index(drop=True)

    tmp = SAMPLED_PATH + ".tmp"
    sampled.to_parquet(tmp)
    os.replace(tmp, SAMPLED_PATH)
    print(
        f"\nWrote {SAMPLED_PATH}: {len(sampled):,} rows, "
        f"{sampled['ticker'].nunique()} tickers, "
        f"{sampled['date'].min().date()} → {sampled['date'].max().date()}",
        flush=True,
    )
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers", help="Comma-separated subset (default: cached universe)")
    ap.add_argument("--ticker", help="Smoke mode: compute one ticker, print, no write")
    args = ap.parse_args()

    if args.ticker:
        t = args.ticker.strip().upper()
        bars = load_bars(t)
        if bars is None or bars.empty:
            raise SystemExit(f"No cached bars for {t} — run data.py first.")
        feats = compute_features(bars)
        cols = ["session", "close", "dist_vwap", "ret_26b", "rel_vol_tod",
                "bar_of_day", "minutes_to_close"]
        print(feats[cols].tail(12).to_string())
        print(f"\n{t}: {len(feats):,} bars, {feats['session'].nunique()} sessions.")
        sampled = sample_rows(feats)
        print(f"Sampled rows: {len(sampled)} (last: {sampled['timestamp'].iloc[-1]})")
        return

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        universe = set(load_universe())
        tickers = [t for t in cached_tickers() if t in universe]
    build(tickers)


if __name__ == "__main__":
    main()
