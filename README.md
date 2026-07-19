# sp500-intraday-ranker

Short-term cross-sectional momentum ranker on the S&P 500: predict each stock's
**forward 5-trading-day return in excess of SPY** from intraday (5-minute bar)
features, rank the universe daily at the 15:25 ET bar, hold the top 10 with a
weekly rebalance.

> **STATUS (2026-07-18): gates FAILED at 5d/full-universe** — model ≈ SPY
> after costs (+19.9% CAGR vs SPY +20.6%), far behind the 12-1 momentum
> baseline (+63.9%). Signal is real (permutation-clean IC 0.014) but weak
> and slow (IC rises toward 21d horizons). Improvement experiments in
> progress: longer label horizons (10d/21d), simpler models, liquid subset.
> See the Results log below. Not deployable in current form.

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

| #   | Gate                      | Pass criterion                                                                                                                           |
| --- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Rank IC                   | mean daily Spearman IC > 0.02 AND Newey-West(5) t-stat > 2                                                                               |
| 2   | Portfolio                 | top-10 weekly-rebalance net return (next-open fills, $1/order + 3 bps/side) beats SPY buy-and-hold AND the 12-1 momentum top-10 baseline |
| 3   | Permutation               | real pooled IC > 95th percentile of 20 within-date label-permuted runs                                                                   |
| 4   | Ablations (informational) | ΔIC from dropping volume features / cross-sectional ranks                                                                                |
| 5   | Decay (informational)     | IC of frozen predictions vs realized 1/3/5/10/21-day excess returns                                                                      |

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

## Running things

### ⭐ Coming back after a while? Run this:

```bash
uv run python scripts/run_all.py --retrain
```

That one command does everything: pulls the missing prices since last run
(top-up), rebuilds features + labels, runs the fatal lookahead check, retrains
the walk-forward models, backtests, and regenerates `reports/validation.html`.
Add `--dry-run` first if you want to see the plan without executing.

### Everyday commands

```bash
uv run python scripts/run_all.py            # 👈 DAILY go-to: topup → features → labels → check → today's top-10 picks (no retrain)
uv run python scripts/run_all.py --retrain  # ⭐ the above + train → backtest → evaluate (run weekly-ish / after code changes)
uv run python scripts/run_all.py --dry-run  # print what would run, execute nothing
uv run python scripts/today.py              # just re-print today's picks (uses the latest trained model)
```

The plain (no-flag) run scores today's stocks with the **most recent trained
model** — fast, no retraining. Picks land in `picks/picks_<date>.csv` and are
paper-tracking only until the gates above pass.

### One-time setup (already done, kept for reference)

```bash
uv sync                                     # install dependencies
uv run python scripts/universe.py           # rebuild data/universe/universe.csv (from sibling's daily cache)
uv run python scripts/data.py --backfill    # 5y of 5-min bars for ~500 tickers, ~3h; resumable — rerun if interrupted
```

### Individual steps (debugging / iterating on one stage)

```bash
uv run python scripts/data.py --topup                     # incremental price pull only (~2 min)
uv run python scripts/data.py --backfill --tickers AAPL   # (re)pull specific tickers
uv run python scripts/features.py                         # rebuild features for all cached tickers
uv run python scripts/features.py --ticker AAPL           # smoke: print one ticker's features, no write
uv run python scripts/labels.py                           # attach forward 5d SPY-excess labels
uv run python scripts/dataset.py                          # window summary + lookahead check
uv run python scripts/dataset.py --n-samples 200          # more thorough lookahead check (after feature changes)
uv run python scripts/train.py                            # full walk-forward train (grid search, slow)
uv run python scripts/train.py --quick                    # walk-forward with fixed params, no grid (fast)
uv run python scripts/backtest.py                         # portfolio sim + baselines from the OOS predictions
uv run python scripts/evaluate.py                         # gates + validation.html (permutation test is slow)
uv run python scripts/evaluate.py --n-perm 5              # faster evaluation pass
uv run python scripts/model_aging.py                      # IC vs model staleness (after a retrain)
uv run python scripts/today.py                            # today's top-10 from the newest model
```

Results land in `reports/validation.html` (gate table, IC time series, decay
curve, equity curves for both fill modes, turnover, cost sensitivity) —
machine-readable copies in `reports/validation_metrics.json` and
`reports/backtest_stats.json`.

## How it works — the 5-minute tour

**One training example** = one stock on one day, photographed at the 15:25 ET
bar. Inputs: 85 features — the stock's returns over the last hour/day/week/
month/quarter, VWAP and moving-average distances, oscillators (RSI/MACD/
Bollinger/…), time-of-day-normalized volume abnormality, and its percentile
rank vs the rest of the universe on each measure at that same moment. Target:
its next-5-trading-day return minus SPY's, clipped ±20%. The model learns
"stocks that look like *this* at 15:25 tend to beat/lag SPY by *this much*
over the next week"; at prediction time we only use the *ranking* of its
scores. Pool: 611 point-in-time tickers × ~1,250 sessions ≈ 630k examples.

**Walk-forward windows.** The 5 years are sliced into ~10 overlapping
experiments ("windows"), each one quarter later than the last:

| Window | Trains on | Validates on | Tests on (its OOS quarter) |
|---|---|---|---|
| 0 | 2021-07 → 2023-09 (~2.2y) | next 63 days | 2024-01 → 2024-04 |
| 1 | 2021-07 → 2023-12 | next 63 days | 2024-04 → 2024-07 |
| ⋮ | (train end slides forward a quarter each window) | ⋮ | ⋮ |
| 9 | 2021-07 → 2026-01 (~4.5y) | next 63 days | 2026-04 → 2026-07 |

5-trading-day purge gaps sit at every boundary (the label horizon) so no
label straddles a split. The test quarters tile 2024-01 → 2026-07 with no
overlap; pooled, they are THE out-of-sample record every reported number
comes from. Each window's test data becomes the next window's training data
— used for testing exactly once, then absorbed, like live operation.

**Why so many windows/models?** One window would rest the entire verdict on
~13 weekly bets in one quarter — luck, not evidence. Ten windows give a
2.5-year OOS record (623 days), cover several regimes, and cost nothing but
compute. Many windows is how a backtest stops being an anecdote and becomes
a sample. One window = one model file (`models/xgb_wf_<test_start>.json`,
named by its test quarter's first day); the backtest never loads these — it
consumes only the pooled `oos_predictions.parquet`. The newest model file is
what `today.py` uses for live picks.

**Retraining cadence.** Weekly rebalances just *score* with the frozen
newest model (seconds). Retraining is quarterly-ish (`--retrain`) — the
walk-forward is a rehearsal of exactly that ritual. Whether quarterly is
actually necessary is an empirical question: `scripts/model_aging.py` scores
every window model on every later quarter (still OOS for it) and plots IC vs
model age (`reports/model_aging.png`) — a flat curve means retraining is
overkill, a steep one means the cadence earns its cost.

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
- **Universe — point-in-time, survivorship-bias controlled**: membership
  history (fja05680/sp500 change events + Wikipedia snapshots, mirrored from
  the sibling project) determines which stocks existed in the index on each
  date. `universe.csv` (committed) lists every member during the backfill
  window — including departed names (ATVI, TWTR, SIVB, PXD, ...), whose bars
  Alpaca serves up to their exact delisting dates — and
  `features.py` applies `filter_to_members` so no (ticker, date) row exists
  unless the stock was in the index that day. Residual caveat: rows in the
  final 5 sessions before a delisting lose their forward label (no future
  close), so the very last days of collapses are under-represented.
- **Bars**: SIP feed, split-adjusted (not dividend-adjusted), regular trading
  hours only (bar starts 09:30–15:55 ET), stored tz-aware UTC per ticker under
  `data/raw/`.

## Results log

**Rounds 5-6 — 2026-07-19** · Portfolio construction + momentum blends on the
saved 10d predictions (backtest-only; reports/topn_sweep_h10d.json,
momentum_blend_h10d.json):

- **Top-N sweep**: conversion improves monotonically with basket width —
  top-5 +1.1% / top-10 +4.0% / top-20 +9.7% / **top-40 +11.8% CAGR**
  (Sharpe 0.61). Confirms the thin-basket diagnosis, but even top-40 < SPY.
- **Momentum×model blends**: every combination LOSES to pure momentum at the
  same width (two-stage k50 +43.0% vs pure mom-10 +54.9%; best rank-blend
  α=.75 +37.6% vs pure mom-20 +42.7%) and triples momentum's turnover. In
  this regime the model's within-momentum selection subtracts value.
- Next: momentum-with-veto (model only excludes its bottom-ranked names from
  momentum's pool) + prediction-decile diagnostic to test whether the
  model's skill is loser-flagging rather than winner-picking.

**Round 4 — 2026-07-18** · Horizon experiments (full universe, quick params,
holding period = horizon; archives in reports/h{5,10,21}_*.json):

| Horizon | IC | NW-t | Model CAGR (net) | Momentum CAGR | Gate 1 |
|---|---|---|---|---|---|
| 5d | 0.0136 | 1.34 | +19.9% | +63.9% | fail |
| **10d** | **0.0259** | **1.95** | +4.0% | +54.9% | fail (barely) |
| 21d | 0.0183 | 1.00 | +9.1% | +55.1% | fail |

Split decision: the decay curve was right — IC nearly doubles at 10d and the
t-stat almost clears the gate — but the portfolio result COLLAPSED at longer
horizons. Diagnosis: fewer rebalances (~62 at 10d, ~26 at 21d) × a top-10
basket = too thin a sample of the ranking to convert IC into return (rank
quality ≠ top-of-book quality). Next: portfolio-construction sweep (top-N ∈
{5,10,20,40}) on the saved 10d predictions — backtest-only, no retraining.

**Round 3b — 2026-07-18** · FULL universe (611 point-in-time tickers, 313k OOS
predictions), full grid + MIN_TREES guard, 20 permutations:

| Gate | Result |
|---|---|
| 1 IC | **FAIL** — mean 0.0136, NW t 1.34 (plain t 2.5) |
| 2 Portfolio | **FAIL** — model +19.9% CAGR ≈ SPY (+20.6%); momentum +63.9%; random +11.1% |
| 3 Permutation | PASS — real 0.0136 vs shuffled p95 0.0031 (signal real but weak) |
| 4 Ablations | ranks −0.0043 and long −0.0035 help at full breadth; volume −0.0012; intraday ≈ 0 |
| 5 Decay | still rises with horizon: 0.008 (1d) → 0.014 (5d) → 0.023 (21d) |

Model-aging study (reports/model_aging.png): **flat** — frozen models show no
measurable IC decay over 2.5y (quarterly retraining not yet justified by data).

Diagnostics worth keeping: 63-day val slices are near-uninformative for config
selection (val IC anti-correlates with test IC across windows); early stopping
on RMSE is fragile (4/10 windows had every config stop under 50 trees — the
MIN_TREES fallback caught them); paradoxically those 1-tree fallback models
scored the best test ICs, i.e. one coarse split on `dist_mean_200d` beat fully
trained models — the extra 500 trees learn mostly noise at this breadth.
Verdict per decision rule: not deployable at 5d/full-universe as configured.
Next candidates: IC-based early stopping (align stop/select/eval metrics),
liquid-subset evaluation, and the 10–21d horizon the decay curve keeps voting
for.

**Round 3 — 2026-07-18** · VOID — grid selection bug: configs early-stopping
at 0 trees could win the val-IC contest (3/10 windows picked 1-tree models);
portfolio landed on the random baseline. Fixed by MIN_TREES disqualification.

**Round 2 — 2026-07-18** · 155 tickers (backfill in progress), point-in-time
membership active, `--quick` train (no grid), OOS 2024-01 → 2026-07 (10 windows):

| Gate | Result |
|---|---|
| 1 IC | **PASS** — mean 0.0269, NW t-stat 2.14 |
| 2 Portfolio | **FAIL vs momentum** — model +42.5% CAGR net (Sharpe 1.23, MaxDD −33%) vs SPY +20.6% / 12-1 momentum +64.2% / random-10 +24.5% |
| 3 Permutation | **PASS** — real IC 0.0269 vs shuffled p95 0.0037 |
| 4 Ablations (ΔIC when dropped) | volume **−0.0145**, intraday **−0.0088**, ranks −0.0027, long **+0.0047** (long features hurt) |
| 5 Decay | IC rises with horizon: 0.014 (1d) → 0.027 (5d) → 0.058 (21d) — signal is slow |

Notable: random-10 collapsed from +62.6% CAGR (round 1, survivorship-biased
53-ticker snapshot) to +24.5% ≈ SPY-ish — the bias controls are working.
Volume and intraday features are now the top contributing groups; long-scale
features slightly hurt at the 5d horizon despite dominating gain importance
(the model over-trusts long momentum — matches the decay curve).

**Round 1 — 2026-07-18** · 53 most-liquid tickers, no membership filter —
numbers inflated by construction (random-10 "made" +62% CAGR); kept only as
the plumbing-validation round.

**Next**: round 3 on the full 611-name universe with the full grid, once the
backfill completes. Candidate experiment: retrain without LONG_FEATURES
(ablation says +0.005 IC) and consider the 10–21d horizon the decay curve
keeps pointing at.

## Notes

- IBKR account `U27177562` (pending open) — the phase-4 execution account.
- Built in Claude Code session `155a9292-8ff6-4874-a88d-5f19d9308fa1`.
- claude --resume 155a9292-8ff6-4874-a88d-5f19d9308fa1 --dangerously-skip-permissions
