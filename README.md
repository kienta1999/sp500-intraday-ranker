# sp500-intraday-ranker

Short-term cross-sectional ranker on the S&P 500, built from intraday (5-minute
bar) features and evaluated walk-forward. The research question was whether
intraday features predict short-horizon (5-10 day) excess returns well enough
to pick stocks; the answer was no — but the model earned a job as a **veto** on
top of a 12-1 momentum strategy, which is what this repo now runs.

> **STATUS (2026-07-19, after rounds 1-11 — research phase complete).**
> The founding hypothesis **failed**: as a standalone stock picker the ML
> ranker is dead (decade IC 0.0045, t=0.5; portfolio ≈ random at every basket
> width; intraday features NEGATIVE over 10y — a daily-bar model would have
> scored slightly better). What survived is a different use of the same model:
> it can't pick winners, but it flags which momentum names are about to break.
> **That veto strategy is the deliverable** (spec below) and is now in paper
> tracking. Nothing trades real money.

## The strategy (what this repo actually produces)

**Momentum picks, the model vetoes, volatility sizes it:**

1. Rank the point-in-time S&P 500 by **12-1 momentum**; take the top **15**.
2. Score those with the walk-forward XGBoost model; **drop any in its bottom
   30%** that day (the veto — this is the model's whole job).
3. Hold the **top 10 survivors** by momentum, equal weight.
4. **Rebalance every 10 sessions**; size total exposure at
   `min(1, 0.20 / SPY 20d realized vol)`.

Out-of-sample 2019-04 → 2026-07 (29 walk-forward quarters, net of $1/order +
3 bps), from `reports/veto_deploy.json` and `veto_robustness.json`:

| Strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **Veto + vol overlay** | **+30.8%** | **1.03** | **−30.4%** |
| Veto, no overlay | +31.5% | 0.97 | −39.6% |
| Pure 12-1 momentum top-10 | +21.3% | 0.71 | −42.5% |
| SPY buy-and-hold | +13.9% | 0.77 | −34.2% |

Robustness: smooth config neighborhood (12 cells, k∈{12..25} × veto∈{20..40}%
all +17-33% CAGR); beats pure momentum on **all 5** rebalance offsets; edge
positive in **all 8** years, largest in 2020-2022 when momentum bled;
replicates independently at a 21-day horizon; 116 orders/yr (0.12% cost drag
at $100k). `today.py` emits these picks daily.

**The honest caveat**: ~40 variants were tested against this same 7-year
out-of-sample window, so some selection pressure is baked in — the veto family
is only semi-pre-registered (it *failed* on 2024-26 data before being confirmed
on the decade). The one uncontaminated test is forward paper tracking:
`scripts/scorecard.py` grades each `picks_*.csv` against what happens after it
was written. Live execution stays gated on that record.

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

### ⭐ The four commands that matter

```bash
uv run python scripts/run_all.py          # 1. DAILY — writes today's picks      (~2 min)
uv run python scripts/veto_overlay.py     # 2. the deployed strategy's numbers   (~15 min)
uv run python scripts/scorecard.py        # 3. how the real picks actually did    (seconds)
uv run python scripts/run_all.py --retrain  # 4. rebuild models — QUARTERLY at most (~4 h)
```

**1 — `run_all.py` is the one you run every day.** Pulls prices since the last
run, rebuilds features + labels, runs the fatal lookahead check, and writes
`picks/picks_<date>.csv` using the pinned model. Run it any time between the
close (~16:15 ET / 13:15 PT — Alpaca's free SIP feed won't serve the 15:25 bar
until it is 15 min old) and the next open; the validated spec fills at the
**next session's open**, so evening or pre-market both reproduce the backtest.
Note the spec rebalances every 10 sessions — running daily keeps data fresh and
feeds the scorecard, but you only place orders on rebalance days.

**2 — `veto_overlay.py` prints the deployed strategy's performance.** ⚠️ Do
NOT use `backtest.py` for this: that script evaluates the *research* strategy
(rank everything by raw model score, buy the top N), which failed its gates.
It says so on startup now.

**3 — `scorecard.py` grades every `picks_*.csv`** against what actually
happened afterward. This is the only evidence no amount of backtest searching
can contaminate, and the gate for ever going live.

**4 — `--retrain` is quarterly at most, not daily.** The model-aging study
(`reports/model_aging.png`) found a **flat** IC-vs-age curve — a model frozen
for two years ranks about as well as a fresh one. After a retrain, **repin the
deployed model**:

```bash
cp models/xgb_wf_<newest-date>.json models/deployed.json
```

`models/` accumulates a file per walk-forward window per experiment round (139
and counting, across three horizons), so `today.py` uses the explicit
`deployed.json` pin rather than guessing. Repinning is deliberate by design.

### One-time setup (already done, kept for reference)

```bash
uv sync                                     # install dependencies
uv run python scripts/universe.py           # rebuild data/universe/universe.csv (point-in-time membership)
uv run python scripts/data.py --backfill    # 10y of 5-min bars for ~720 tickers, ~6h; resumable — rerun if interrupted
```

### Individual steps (debugging / iterating on one stage)

```bash
uv run python scripts/run_all.py --dry-run                # print the plan, execute nothing
uv run python scripts/today.py                            # re-print today's picks (veto strategy)
uv run python scripts/today.py --mode model               # legacy pure-model ranking (research only)
uv run python scripts/data.py --topup                     # incremental price pull only (~2 min)
uv run python scripts/data.py --backfill --tickers AAPL   # (re)pull specific tickers
uv run python scripts/features.py                         # rebuild features for all cached tickers
uv run python scripts/features.py --ticker AAPL           # smoke: print one ticker's features, no write
uv run python scripts/labels.py                           # attach forward-return labels (HORIZON_DAYS, default 10)
uv run python scripts/dataset.py                          # window summary + lookahead check
uv run python scripts/dataset.py --n-samples 200          # more thorough lookahead check (after feature changes)
uv run python scripts/train.py                            # full walk-forward train (grid search, ~2h on GPU)
uv run python scripts/train.py --quick                    # fixed params, no grid (fast)
uv run python scripts/backtest.py                         # RESEARCH strategy sim + baselines (not the deployed spec)
uv run python scripts/evaluate.py                         # gates + validation.html (permutation test is slow)
uv run python scripts/model_aging.py                      # IC vs model staleness
uv run python scripts/momentum_blend.py --rebalance-days 10   # veto/blend family comparison
uv run python scripts/veto_robustness.py --rebalance-days 10  # config surface, offsets, per-year attribution
uv run python scripts/topn_sweep.py --rebalance-days 10       # basket-width sweep
```

The horizon is set by the `HORIZON_DAYS` env var (default **10**, the deployed
spec). Experiments at other horizons write models and predictions that will
overwrite the current ones — archive first, and repin afterwards.

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

## Lessons from building this

Written 2026-07-19 after 11 experiment rounds. These are the things that cost
real time to learn — read this before starting the next strategy.

### On being fooled

**1. The dartboard test catches survivorship bias instantly.** Always backtest
a *random* portfolio alongside the real one. Round 1 showed the model earning
+126% CAGR — and randomly chosen stocks earning **+62%**. When darts triple
the market, the universe is rigged, not the strategy good. The cause: ranking
by *today's* dollar volume and backtesting the past, i.e. picking from a basket
someone already filled with winners. Fix: point-in-time membership (a stock
appears on a date only if it was in the index that day, delisted names
included). After the fix, random-10 fell to ≈ SPY and the model's "edge"
mostly evaporated with it.

**2. A dumb baseline is the only honest yardstick.** 12-1 momentum — one line
of pandas, published 1993 — beat this entire 85-feature pipeline for most of
the project. Any ML result that isn't compared against the obvious rule is
decoration. Corollary: pick the baseline *before* seeing your own results.

**3. Regime monoculture flatters everything.** On 2024-26 data, momentum
returned +64%/yr and every strategy looked brilliant. Over the full decade
(incl. COVID and 2022) the same baseline returned +21% with a Sharpe *below*
SPY's. Two and a half years of backtest is an anecdote; a decade spanning
drawdowns is evidence.

**4. Count your selection pressure.** ~40 variants were tested against the same
out-of-sample window here. Every extra variant makes the surviving winner less
trustworthy — the best cell of any table is partly luck. Mitigations used:
never re-pick the best neighbor after seeing a surface, keep failed rounds in
the log, and treat forward paper tracking as the only clean test.

### On the machinery

**5. Align the metrics: stop, select, and evaluate on the SAME thing.** Early
stopping watched RMSE while model selection watched rank IC. On a noisy target
these disagree violently — whole grids stopped at ~0 trees, and near-constant
models won the selection contest by fluke. Fixed by early-stopping on daily
rank IC directly (`_make_daily_ic_metric` in train.py).

**6. Guard against under-trained models winning.** A 1-tree model can post a
great validation score on a short window. `MIN_TREES = 50` disqualifies
configs that never really trained. Round 3 was voided by exactly this bug.

**7. Short validation windows lie.** With 63 trading days, validation IC
*anti-correlated* with test IC — worse than useless for selection. Doubling to
126 days fixed it. If your val slice can't rank configs, no amount of grid
search helps.

**8. Match feature timescales to the label horizon.** A 50-minute RSI cannot
predict a 5-day return; the rule of thumb is lookbacks within ~0.5×–10× of the
horizon. Features were reorganized into intraday / horizon-band (1-50d) /
long (63-252d) buckets, with the densest coverage around the horizon, and
each bucket ablated separately to measure its real contribution.

**9. Measure the retraining cadence instead of assuming it.** `model_aging.py`
scores every walk-forward model on every *later* quarter. The curve came out
flat — a two-year-old model ranked as well as a fresh one. Quarterly retraining
is a ritual here, not a necessity.

### On the result

**10. A negative result can be repurposed.** As a stock picker the model is
dead (decade IC 0.0045, returns ≈ random, intraday features *negative*). But
its prediction deciles were monotone — it separated good from bad, just too
weakly to pick winners. Flipping its role from **picker to veto** (remove
momentum's worst names) turned a failed model into +9 CAGR points and −12
points of drawdown. Ask what a weak signal *can* do before discarding it.

**11. Portfolio construction is a first-class variable.** The same predictions
returned +1.1% (top-5) to +11.8% (top-40) depending only on basket width.
Ranking skill and top-of-book skill are different things; a good IC converts to
nothing if the basket is too thin to sample it.

**12. Verify a winner three ways before believing it.** (a) *Neighborhood*: do
nearby configs also win? Real edges are smooth hills, mirages are lone spikes.
(b) *Schedule*: does it survive all rebalance offsets? (c) *Attribution*: which
years produced the edge, and is the mechanism coherent? The veto passed all
three — and its edge being largest in 2020-2022 matched the story that it
avoids blowups.

### On operations

**13. Artifacts silently rot.** `models/` accumulated 139 files across three
horizons; `today.py` picked "the newest filename" and was correct only by luck.
`oos_predictions.parquet` got overwritten by a side experiment, so the
backtest was silently scoring the wrong run. Fixes: an explicit
`models/deployed.json` pin, archived per-round artifacts (`reports/r8_*.json`),
and a startup banner on `backtest.py` saying which strategy it evaluates.

**14. Detach long jobs from the assistant.** Multi-hour pipelines ran as
`setsid nohup` scripts that survive session limits and teardown; the assistant
only polls. One bug worth remembering: a wait-loop that counted *waiting*
iterations against its retry budget gave up on a healthy 6-hour backfill after
36 minutes. Budget relaunches, not patience.

## Results log

**Round 11 — 2026-07-19** · 21d-horizon veto variant (10y retrain, monthly-ish
rebalance; reports/momentum_blend_h21d_10y.json): the veto effect REPLICATES
at 21d — all configs ≥ pure momentum, best +27.2%/Sharpe 0.94 vs +18.3%/0.71
— further evidence the mechanism is real, not a horizon artifact. But the
10d spec remains superior (+30.8%/1.03 with overlay). Hunt phase concludes;
loop shifts toward paper-tracking cadence (scripts/scorecard.py accrues one
unfakeable graded row per trading day).

**Round 10 — 2026-07-19** · Deployment realism (reports/veto_deploy.json):

- **Cost drag is a non-issue**: 116 orders/yr measured → $1/order costs
  1.16%/yr at $10k, 0.23% at $50k, 0.12% at $100k (0% on IBKR Lite).
- **Vol-target overlay (0.20 target, sibling's recipe) earns its place**:
  MaxDD −39.6% → **−30.4%** and Sharpe 0.97 → **1.03** for only −0.7pt CAGR
  (+31.5% → +30.8%); average exposure 0.945.

**THE DEPLOYABLE SPEC (paper-trade candidate)**: 12-1 momentum top-15 →
model vetoes its bottom-30% → hold top-10 by momentum → rebalance every 10
sessions → vol-target 0.20 exposure overlay. 7y OOS, net of costs:
**+30.8% CAGR, Sharpe 1.03, MaxDD −30.4%** (vs SPY +13.9%/0.77/−34.2%,
pure momentum +21.3%/0.71/−42.5%). `today.py` emits these picks daily.
Note: config stays k15/v30 as originally tested — neighbors like k12/v30
score even higher (+33.4%/0.99) but re-picking the best neighbor ex-post
would be the +126% mistake again.

**Round 9 — 2026-07-19** · Veto robustness battery
(reports/veto_robustness.json) — **the lead survived all three tests**:

1. *Stability surface (12 cells, k∈{12,15,20,25} × veto∈{20,30,40}%)*: a
   smooth hill, not a spike — the whole k12-15 region sits at +28-33% CAGR /
   Sharpe 0.88-0.99; wider pools degrade gracefully (+17-27%). The winning
   cell is typical of its neighborhood.
2. *Offset schedules*: veto ≥ pure momentum on ALL 5 rebalance offsets
   (veto +20.1%→+31.5% vs momentum +19.6%→+23.4%); worst veto ≈ best momentum.
3. *Per-year attribution*: edge positive in ALL 8 years — strongest where
   momentum bleeds (2020: +12.6, 2021: +12.7, 2022: +5.9) AND in melt-ups
   (2024: +16.6, 2026: +15.3). Not one lucky year.

Caveats that remain: the veto family was conceived after round 7 (though it
FAILED on 2024-26 data and was confirmed only on the 7y OOS — closer to a
pre-registered hypothesis than a fitted one); project-wide we've tested ~40
variants on this OOS, so some selection pressure exists. Costs ($1 + 3bps)
included; turnover 0.87/rebalance.

**Strategy promoted**: momentum top-15 → model vetoes bottom-30% → hold 10,
10-session rebalance. Next: wire into today.py picks, cost accounting at
account sizes, vol-target overlay for the −39% MaxDD.

**Round 8 — 2026-07-19 (overnight)** · THE DECADE RUN: 10y bars, 721
point-in-time members, 29 OOS quarters (2019-04 → 2026-07, incl. COVID +
2022 bear), 10d horizon, IC-early-stopping trainer, 126d val, full grid.
Archives: reports/r8_10y_*.json, oos_r8_10y.parquet.

*The standalone model is dead:*
- IC 0.0045 (NW-t 0.5) over 1,804 days — permutation-clean but economically nil.
- Portfolio ≈ random at every width (top-5/10/20/40 all +7-10% CAGR, Sharpe
  ~0.4, −61% MaxDD at top-10) vs SPY +13.9%/0.77.
- **Ablation referendum: intraday features are NEGATIVE over the decade**
  (drop → IC +0.0031). Volume also negative. Only cross-sectional ranks help.
- Context: momentum's own edge deflates to +7.5%/yr over SPY (Sharpe 0.71 <
  SPY's 0.77) across the decade — 2024-26 flattered everyone.

*The surviving lead — momentum-with-veto (model as loser-flagger only):*
| Variant | CAGR | Sharpe | MaxDD | vs pure momentum same width |
|---|---|---|---|---|
| veto k15→n10 v30 | **+31.5%** | **0.97** | −39.6% | +21.3% / 0.71 |
| veto k30→n20 v30 | +19.6% | 0.76 | −39.3% | +18.4% / 0.70 |
| veto k30→n20 v50 | +20.9% | 0.79 | −34.3% | " |
| veto k40→n20 v30 | +17.9% | 0.71 | −39.3% | " |

All four ≥ pure momentum; the concentrated one is large. CAUTION: the best
cell was selected ex-post from 4 candidates (mild snooping) and reverses the
2024-26 result (veto hurt there) — consistent with the model's value being
crash-regime loser-avoidance. Robustness battery before believing it:
neighbor-config stability surface, offset bands, per-year attribution,
cost grid.

**Round 7 — 2026-07-19** · Momentum-veto + decile diagnostic
(reports/momentum_blend_h10d.json): veto variants also lose to pure momentum
(best +38.3% vs +42.7% at n=20). Decile table shows the model DOES rank
(bottom decile 0.33%/10d → top 0.78%, mostly monotone) but the edge is too
small to improve momentum in this regime by blending, filtering, or vetoing.
**Momentum-combination avenue exhausted on the 2024-26 window.**
Decision: extend history to 10y (Alpaca's full archive, 2016→) — doubles
training data, triples OOS to ~22 quarters, and adds the 2018/2020/2022
momentum-hostile regimes where the model's defensive tilt can actually be
tested against the baseline. Universe rebuilt point-in-time for the 10y
window: 721 members (105 delisted without sibling daily caches).

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
