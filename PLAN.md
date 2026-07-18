# sp500-intraday-ranker — Implementation Plan

## Context

Mirror `/home/talekien1710/personal_project/ml-stock-forward-return` (XGBoost S&P 500 cross-sectional ranker, 21-day horizon, daily yfinance bars) into a **short-term momentum** version in the new, empty repo `/home/talekien1710/personal_project/sp500-intraday-ranker` (git remote already wired to github.com/kienta1999/sp500-intraday-ranker). The label becomes **forward 5-trading-day return in excess of SPY**; features are intraday 5-min-bar based (VWAP distance, time-of-day-normalized volume, bar-level momentum/vol); validation is **walk-forward** with hard pass/fail gates written before any results exist.

User decisions (confirmed in conversation):
- **Data source: Alpaca Market Data (free tier) for ALL data** — one-time backfill and nightly top-up. Free-tier historical queries serve full-market SIP bars (query end must be ≥15 min old — always true for us), ~200 req/min, 10k bars/request. Full 500-ticker backfill ≈ minutes-to-an-hour, not the ~19 nights IBKR pacing would need. **IBKR is used only for trade execution** (paper/live, phase 4, only if gates pass) — never for data, so volume features have no source seam (IBKR volume excludes off-exchange prints; SIP is the full tape).
- **Universe: all ~500 current S&P members**, dollar-volume ranked.
- **History: 4 years** (superseding the earlier "2 years" answer, which was chosen purely to limit IBKR pull time — with Alpaca the cost difference is minutes). 4y supports the originally-specified walk-forward: ~2y train → ~2y of rolling OOS quarters. Alpaca history goes back to 2016 if more is ever wanted.
- **Backtest fills: report both** next-open and same-day-15:35 fills; the gap measures overnight-drift dependence.

Decision gate (from the user's plan, restated in README from day 1): if the model can't beat the 12-1 momentum baseline after costs, improve features or accept the baseline — do not deploy.

**Prerequisite from the user**: a free Alpaca account and its API key pair, supplied as env vars `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (scripts fail fast with a clear message if unset). No funding or brokerage approval needed for market data.

## Conventions mirrored from the sibling

Flat `scripts/` dir (no package), `sys.path.insert(0, _HERE)` cross-imports, `_HERE`/`_ROOT` path idiom, heavy module docstrings with CLI examples, `print(..., flush=True)`, modern type hints, UPPERCASE module constants + argparse overrides, no config files (env vars for secrets), atomic parquet writes (`.tmp` + `os.replace`, sibling `scripts/universe.py:179-181`), incremental tail-append caches with `duplicated(keep="last")` dedup (sibling `scripts/data.py:146-148`), retry-with-backoff on HTTP (sibling `data.py` RETRIES=3 idiom), `run_all.py` subprocess orchestrator, uv-managed pyproject (Python >=3.11).

## Repo skeleton

```
sp500-intraday-ranker/
├── pyproject.toml        # pandas>=2.2 numpy>=1.26 pyarrow>=15 xgboost>=2.0 scikit-learn>=1.4
│                         # scipy>=1.12 matplotlib>=3.8 tqdm>=4.66 alpaca-py>=0.21
│                         # (ib-async added only in phase 4 if gates pass; no yfinance, no optuna)
├── .python-version       # 3.11
├── .gitignore            # .venv/ __pycache__/ data/raw/ data/market/ data/processed/ cron.log
│                         # (data/universe/universe.csv IS committed — frozen universe contract)
├── README.md             # overview, build order, GATES table written BEFORE results, Alpaca key setup
├── scripts/
│   ├── alpaca_client.py    # thin shared layer: env-key check, StockHistoricalDataClient factory,
│   │                       #   retry/backoff on 429/5xx, batched multi-symbol bar fetch helper
│   ├── universe.py         # all current S&P members ranked by 63d median dollar volume → universe.csv
│   ├── data.py             # puller CLI (--backfill / --topup) + load_bars/load_daily/load_spy_daily
│   ├── features.py         # bucket constants + compute_features(bars) per ticker + cross-sectional ranks
│   ├── labels.py           # forward 5d close-to-close return minus SPY → data/processed/panel/
│   ├── dataset.py          # 15:25-bar sampling, walk-forward Window bookkeeping (purge gap), assert_no_lookahead
│   ├── train.py            # walk-forward harness: per-window grid + early stopping → per-window models + pooled OOS
│   ├── strategy.py         # shared constants (TOP_N=10, REBALANCE_DAYS=5, COST_PER_ORDER=$1, SPREAD_BPS=3,
│   │                       #   BUFFER_RANK=15, DEFAULT_SEED=15) + load_model/predict/top_picks/compute_weights
│   ├── backtest.py         # top-10 weekly-rebalance sim, BOTH fill modes, baselines, offset band
│   ├── evaluate.py         # gates, permutation control, ablations, decay curve → reports/validation.html
│   └── run_all.py          # orchestrator: topup → features → labels → lookahead check → [--retrain: train→backtest→evaluate]
├── data/                   # gitignored except universe/universe.csv
│   ├── universe/universe.csv
│   ├── raw/{TICKER}.parquet           # 5-min bars, tz-aware UTC index, OHLCV columns, RTH only
│   ├── market/SPY_daily.parquet
│   └── processed/{features/,panel/}month=YYYY-MM/  + oos_predictions.parquet
├── models/                 # xgb_wf_{test_start}.json per window
└── reports/                # validation.html, validation_metrics.json, pngs, walkforward_params.json
```

Phase-4 only (built after gates pass, not now): `check_ibkr_conn.py` (near-verbatim sibling port — WSL→Windows host discovery, 4002 paper/4001 live) and an `execute_picks.py` analog; `ib-async` dep added then.

## 1. Data layer (`scripts/alpaca_client.py` + `scripts/data.py`)

**Client (`alpaca_client.py`).** Reads `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` from env (fail fast with setup instructions). `StockBarsRequest` via `alpaca-py`'s `StockHistoricalDataClient`: `timeframe=TimeFrame(5, Minute)`, `feed=DataFeed.SIP`, **`adjustment=Adjustment.SPLIT`** (split- but not dividend-adjusted — same convention as IBKR TRADES bars and the sibling's return math; documented in docstring). Batched multi-symbol requests (~100 symbols/call), SDK handles 10k-bar pagination. Retry 3× with exponential backoff on 429/5xx; a modest `time.sleep(0.3)` between calls keeps us far under the 200 req/min cap — no token-bucket machinery needed.

**Cache model.** One parquet per ticker (`data/raw/{TICKER}.parquet`), tz-aware UTC DatetimeIndex `date` (Alpaca bar timestamps are bar-start UTC), columns Open/High/Low/Close/Volume (sibling schema). **RTH filter applied at ingest**: keep bars whose NY-local start ∈ [09:30, 15:55] — Alpaca returns pre/post-market bars, and the plan requires regular hours only. Atomic write per ticker. Incremental: tail-append from `max(index)` forward + boundary dedup — the sibling `data.py` pattern ported nearly verbatim. No state sidecars / resume cursors / chunk shrinking — a full re-pull costs minutes, so the elaborate IBKR-survival machinery is unnecessary.

**CLI.**
- `data.py --backfill [--years 4] [--tickers ...]`: pulls the full window for every universe ticker (skipping already-complete ones). ~500 tickers × 4y ≈ 39M bars ≈ a few thousand paginated requests ≈ **30–60 minutes, one sitting**. Per-ticker status summary at the end ({new, updated, fresh, failed}) like the sibling.
- `data.py --topup`: per run, one batched request from each ticker's last cached bar to now-15min. Whole universe in ~1–2 minutes — fits the nightly cron trivially.
- **SPY daily**: every run refreshes `data/market/SPY_daily.parquet` (TimeFrame.Day, same feed/adjustment — labels and benchmark use the same source as the features, no seam).
- `load_bars(ticker)`, `load_daily(ticker)` (session-close series derived from last RTH bar), `load_spy_daily()` — the public loaders everything downstream uses.

## 2. Universe (`scripts/universe.py`)

Bootstrap from the sibling's cache — zero API calls: current members from sibling `data/universe/sp500_sectors.csv`; for each with a raw daily parquet in sibling `data/raw/` (730 cached, fresh through 2026-07), compute 63-day median dollar volume; rank descending; write ALL ~500 members to `data/universe/universe.csv` (`ticker, dollar_vol_63d, rank, gics_sector, as_of`). `SIBLING_ROOT` constant with `--sibling-root` override. **Frozen file, committed to git**; manual quarterly refresh; additions backfilled via `--tickers`, removals dropped from csv but kept on disk. Documented limitation: fixed present-day membership over 4y history = survivorship bias, accepted deliberately (README + docstring). `load_universe() -> list[str]` (rank order) is the single source of truth for data.py/features.py.

## 3. Features (`scripts/features.py`)

Bucket constants at top (imported by dataset/train/strategy — import from features.py directly to avoid circular imports):

- `MOMENTUM_FEATURES`: `ret_{1,3,12,26,78}b`, `ret_5d`, `ret_21d` (prior daily closes), `dist_vwap` (px/cumulative-session-VWAP −1), `dist_mean_20d`, `dist_mean_50d`, `range_pos_78b`.
- `VOLATILITY_FEATURES`: `rvol_26b`, `rvol_78b` (1-bar-return std × √(78·252)), `atr14b_pct`, `range_expansion` (bar range / 20-bar mean range).
- `VOLUME_FEATURES`: `rel_vol_tod` (bar volume / mean same-HH:MM bar over prior 20 sessions — U-shape normalization), `cum_vol_vs_20d` (cumulative day volume vs 20-session mean at same bar), `dollar_vol_rank` (21d rolling dollar-vol percentile vs universe), `volume_price_corr_26b`.
- `TIME_FEATURES`: `bar_of_day`, `day_of_week`, `minutes_since_open`, `minutes_to_close`.
- `RANK_FEATURES`: per-timestamp `groupby("date")[col].rank(pct=True)` over the universe for all rankable features.

Mechanics: convert UTC→America/New_York once; session grouping by local date; VWAP = per-session `cumsum(typ_px·vol)/cumsum(vol)`; daily aggregates built from **prior completed sessions only** (session t sees closes through t−1) — the lookahead-critical part; time-of-day volume via (session × HH:MM) pivot with 20-session rolling mean shifted 1 session; half-days align naturally on HH:MM keys. Full per-(timestamp,ticker) table (~39M rows × ~40 cols at 500 tickers × 4y — **process month-by-month partitions**, don't hold the full table in memory), written `to_parquet(partition_cols=["month"])`; daily topup rewrites only the current month. `--ticker AAPL` smoke mode prints last rows.

## 4. Labels (`scripts/labels.py`)

`LABEL_COL = "forward_5d_excess_spy"` = `close[t+5]/close[t] − 1 − spy_ret_5d`, from daily closes (session close = last 5-min bar). SPY-excess chosen over sibling's date-demeaning: identical within-date ordering (per-date constant shift), matches the user's written plan, SPY series needed anyway for the benchmark. Clip raw label ±20%. Labels attach only to **sampled decision rows** (one per ticker-session); full-resolution rows keep NaN labels (used by the decay study). Output: `data/processed/panel/` monthly-partitioned, features + `close` + raw + excess label.

## 5. Dataset + walk-forward (`scripts/dataset.py`)

- `load_sampled_panel()`: one row per (ticker, session) — bar starting `SAMPLE_TIME = "15:25"` ET (data through 15:30); fall back to session's last bar on half-days. ~500 × 1008 ≈ 500k modeling rows at full universe.
- `walk_forward_windows(dates) -> list[Window]` (NamedTuple): **train = expanding, starting at `MIN_TRAIN_DAYS = 504` (2y, per the original spec)**; val = last 63 trading days before test; test = next 63; roll by 63. **`PURGE_GAP = 5` trading days** at train→val and val→test boundaries (label horizon). 4y of data → ~6–7 OOS quarters.
- `assert_no_lookahead(n_samples)`: sample random (ticker, timestamp) rows, truncate raw 5-min bars to ≤ timestamp (SPY daily to < session), recompute features, compare last row `np.isclose(1e-6)` on all non-rank features (ranks skipped — need full cross-section but are timestamp-aligned; same rationale sibling documents). CLI-runnable; the build-order step 2 acceptance test; also runs inside run_all.py (fatal).

## 6. Training (`scripts/train.py`)

XGBRegressor, `reg:squarederror`, `tree_method="hist"`. Per window: **small grid, not Optuna** (63-day val slices would overfit 50 TPE trials; deterministic; cost linear in windows): `max_depth {3,4,6} × learning_rate {0.01,0.03,0.1} × min_child_weight {5,20} × subsample {0.6,0.9}` (24 configs), `n_estimators=1500`, `early_stopping_rounds=100` on the val tail. Selection metric: **mean daily Spearman IC** on val (port sibling `daily_ic`). Full grid on window 1; later windows re-run only window-1's top-6 configs. Artifacts: `models/xgb_wf_{test_start}.json`, `reports/walkforward_params.json`, and the **pooled OOS table** `data/processed/oos_predictions.parquet` (`date, ticker, y_pred, y_true_raw, y_true_excess, window_id`) — sole input to backtest/evaluate. `--smoke`: 1 window, 2 configs, 2-ticker panel.

## 7. Backtest + evaluation (`scripts/backtest.py`, `scripts/evaluate.py`)

**Backtest**: every 5 trading days rank by `y_pred` at the 15:25 sample; equal-weight top-10; **both fill modes reported**: (a) next session's open (first 5-min bar's Open), (b) same-day 15:35 bar; costs $1/order + 3bps half-spread per side of turnover. Baselines: 12-1 momentum top-10 (`ret_252d − ret_21d` from our own daily aggregation — 4y of history spares the lookback year), random-10 (20 seeds, mean + 10/90 band), SPY buy-and-hold. 5 rebalance-offset variants (0–4) with band. Outputs `reports/backtest_equity.{csv,png}`, `backtest_stats.json`.

**Evaluate — GATES dict at top of file and in README before any results:**
1. IC gate: pooled OOS per-date Spearman; pass iff mean > 0.02 AND Newey-West(lag=5) t-stat > 2 (overlapping 5d labels autocorrelate; plain t also reported).
2. Portfolio gate: model top-10 net return > SPY AND > 12-1 momentum baseline over pooled OOS (next-open fills are the binding variant).
3. Permutation control: 20 within-date label permutations, refit with chosen params, pass iff real IC > 95th percentile.
4. Ablations (informational): drop VOLUME_FEATURES, drop RANK_FEATURES → ΔIC; plus gain importances.
5. Decay curve: frozen predictions vs realized 1/3/5/10/21-day excess returns.

Output: **`reports/validation.html`** — self-contained, matplotlib figs base64-embedded, gate pass/fail table, IC time series, decay curve, equity curves (both fill modes), turnover, cost-sensitivity table ($0/1/2 × 0/3/10bps). Also pngs + `validation_metrics.json` (sibling's machine-readable convention).

## 8. Orchestrator (`scripts/run_all.py`)

Default: universe-exists check → `data.py --topup` → `features.py` → `labels.py` → `dataset.py --n-samples 25` (fatal lookahead check) → [`--retrain`: `train.py` → `backtest.py` → `evaluate.py` (non-fatal last step)]. Cron-friendly (resolved-uv path, stop-on-first-failure, exit codes) — sibling idiom.

## 9. Build order

1. **Day 1 — data**: skeleton (pyproject, .gitignore, README w/ GATES + Alpaca key setup) + `alpaca_client.py` + `universe.py` + `data.py`. Smoke-pull 2 tickers, then run the **full 500-ticker × 4y backfill the same day (~30–60 min)**.
2. **Day 2 — features**: `features.py`, `labels.py`, `dataset.py`; lookahead test must pass on real data.
3. **Days 3–4 — model + verdict**: `strategy.py`, `train.py`, `backtest.py`, `evaluate.py`; `run_all.py --retrain`; read `validation.html`; apply the decision gate.
4. **Phase 4 (only if gates 1–2 pass)**: portfolio rules (buffer ranks, earnings exclusion), `check_ibkr_conn.py` port, paper trading via ib-async. If the model can't beat 12-1 momentum after costs: iterate on features or accept the baseline — do not deploy.

## 10. Verification

1. `universe.py` → eyeball printed table (NVDA/AAPL/MSFT/TSLA-class names at top, ~500 rows).
2. **2-ticker smoke pull**: `data.py --backfill --tickers AAPL,MSFT --years 0.1` (~seconds). Verify: 78 bars on full days, tz-aware UTC index, no pre/post-market bars, volumes match consolidated-tape figures (spot-check vs a quote site).
3. Re-run same command → all "fresh", nothing re-downloaded; next day `--topup` → tail-append + boundary dedup.
4. `features.py --ticker AAPL` → hand-check `dist_vwap`, `bar_of_day` against raw bars for one session.
5. `features.py` + `labels.py` + `dataset.py --n-samples 50` on smoke tickers → **lookahead test must pass before any training**.
6. `train.py --smoke` → `backtest.py` → `evaluate.py` → `validation.html` renders end-to-end (metrics meaningless, plumbing proven).
7. Full backfill, then rerun 4–6 on the full universe; `run_all.py --retrain` for the real result.

## Critical reference files (sibling)

- `ml-stock-forward-return/scripts/data.py` — incremental cache/loader + retry pattern
- `ml-stock-forward-return/scripts/dataset.py` — assert_no_lookahead + split bookkeeping to adapt
- `ml-stock-forward-return/scripts/strategy.py` — shared-constants + predict/top_picks pattern
- `ml-stock-forward-return/scripts/run_all.py` — subprocess orchestrator to mirror
- `ml-stock-forward-return/scripts/universe.py:179-181` — atomic-write idiom
- `ml-stock-forward-return/scripts/check_ibkr_conn.py` — phase-4 only (execution plumbing)
