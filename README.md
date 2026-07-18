# sp500-intraday-ranker

Short-term cross-sectional momentum ranker on the S&P 500: predict each stock's
**forward 5-trading-day return in excess of SPY** from intraday (5-minute bar)
features, rank the universe daily at the 15:25 ET bar, hold the top 10 with a
weekly rebalance.

Sibling project: [`ml-stock-forward-return`](../ml-stock-forward-return) — the
21-day / daily-bar version whose architecture this repo mirrors. What changes
here: data is 5-minute bars from **Alpaca Market Data** (SIP consolidated tape),
features are intraday (VWAP distance, time-of-day-normalized volume, bar-level
momentum/volatility), validation is **walk-forward**, and the label horizon is
5 days.

**IBKR is used for trade execution only** (phase 4, gated — see below). All
data, both the one-time backfill and the nightly top-up, comes from Alpaca so
volume features never straddle two sources (IBKR historical volume excludes
off-exchange prints; SIP is the full tape).

## Evaluation gates (written before any results existed)

The model ships to paper trading **only if** it passes these, evaluated on
pooled walk-forward out-of-sample predictions:

| # | Gate | Pass criterion |
|---|------|----------------|
| 1 | Rank IC | mean daily Spearman IC > 0.02 AND Newey-West(5) t-stat > 2 |
| 2 | Portfolio | top-10 weekly-rebalance net return (next-open fills, $1/order + 3 bps/side) beats SPY buy-and-hold AND the 12-1 momentum top-10 baseline |
| 3 | Permutation | real pooled IC > 95th percentile of 20 within-date label-permuted runs |
| 4 | Ablations (informational) | ΔIC from dropping volume features / cross-sectional ranks |
| 5 | Decay (informational) | IC of frozen predictions vs realized 1/3/5/10/21-day excess returns |

**Decision rule:** if the model can't beat the 12-1 momentum baseline after
costs, improve features or accept the baseline — do **not** deploy the ML anyway.

## Setup

```bash
uv sync
```

Data comes from Alpaca's free tier (full-market SIP historical bars; no funding
needed). Create a free account at https://alpaca.markets, generate an API key
pair (paper is fine), and put it in a `.env` at the repo root (gitignored;
picked up automatically) — or export the same names in your shell:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

## Pipeline

```bash
uv run python scripts/universe.py          # build data/universe/universe.csv (from sibling's daily cache)
uv run python scripts/data.py --backfill   # one-time: 4y of 5-min bars, ~30-60 min for ~500 tickers
uv run python scripts/run_all.py           # daily: topup → features → labels → lookahead check
uv run python scripts/run_all.py --retrain # + walk-forward train → backtest → evaluate
```

Results land in `reports/validation.html` (gate table, IC time series, decay
curve, equity curves for both fill modes, turnover, cost sensitivity).

## Design notes

- **Label**: `forward_5d_excess_spy` from daily closes (session close = last
  RTH 5-min bar), clipped ±20%. SPY-excess rather than date-demeaning: identical
  within-date ordering, and SPY is needed for the benchmark anyway.
- **Decision row**: one per (ticker, session) — the bar starting 15:25 ET
  (data through 15:30), so signals allow same-day execution near the close.
  The backtest reports both next-open and same-day-15:35 fills; the gap
  measures how much of the edge is overnight drift.
- **Walk-forward**: expanding train window (min 2 years) → 63-trading-day
  validation tail (early stopping + small grid) → 63-day out-of-sample test →
  roll by 63 days, with a 5-trading-day purge gap at every boundary (the label
  horizon), so no label straddles a split.
- **No-lookahead guard**: `dataset.py` recomputes sampled feature rows from
  raw bars truncated to the decision timestamp and asserts equality — run it
  after any feature change.
- **Universe**: all current S&P 500 members, dollar-volume ranked, frozen in
  `data/universe/universe.csv` (committed). Fixed present-day membership over
  4 years of history carries survivorship bias — accepted deliberately to
  avoid point-in-time intraday complexity; documented here so results are read
  with that caveat.
- **Bars**: SIP feed, split-adjusted (not dividend-adjusted), regular trading
  hours only (bar starts 09:30–15:55 ET), stored tz-aware UTC per ticker under
  `data/raw/`.
