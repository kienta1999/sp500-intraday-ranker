#!/usr/bin/env python3
"""Orchestrate the pipeline (cron-friendly; stops on first failure).

Modes:
    (default)   topup → features → labels → lookahead check
    --retrain   (default) + train → backtest → evaluate
    --dry-run   print the plan, don't execute

The lookahead check is FATAL by design — a leak invalidates everything
downstream. evaluate.py is the non-fatal last step (a plotting hiccup must
not make the cron exit code look like a data problem).

The one-time backfill is NOT part of this script — run it directly:
    uv run python scripts/data.py --backfill
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Resolve `uv` to an absolute path — cron shells may not have ~/.local/bin
# on PATH yet.
_UV = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")


def _run(label: str, cmd: list[str], dry_run: bool, allow_fail: bool = False) -> int:
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print("═" * 70, flush=True)
    if dry_run:
        print("  (dry-run: not executing)")
        return 0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=_ROOT)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  → {status} in {time.time() - t0:.1f}s", flush=True)
    if allow_fail and result.returncode != 0:
        print("  (non-fatal step — continuing despite failure)")
        return 0
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--retrain", action="store_true",
                    help="Also run train + backtest + evaluate.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    py = [_UV, "run", "python"]
    steps: list[tuple[str, list[str]]] = [
        ("Refresh point-in-time universe", py + ["scripts/universe.py"]),
        ("Incremental data top-up (Alpaca)", py + ["scripts/data.py", "--topup"]),
        ("Rebuild features", py + ["scripts/features.py"]),
        ("Rebuild labeled panel", py + ["scripts/labels.py"]),
        ("Lookahead check (fatal)", py + ["scripts/dataset.py", "--n-samples", "25"]),
    ]
    if args.retrain:
        steps += [
            ("Walk-forward train", py + ["scripts/train.py"]),
            ("Backtest", py + ["scripts/backtest.py"]),
            ("Evaluate gates + validation.html", py + ["scripts/evaluate.py"]),
        ]
    # Picks run last and are non-fatal: before the first training there's no
    # model yet, and a scoring hiccup must not make the data refresh look broken.
    steps.append(("Today's picks", py + ["scripts/today.py"]))

    non_fatal = {"Evaluate gates + validation.html", "Today's picks"}

    print(f"\nrun_all.py — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{len(steps)} step(s) planned:")
    for label, _ in steps:
        print(f"  - {label}")

    t0 = time.time()
    for label, cmd in steps:
        code = _run(label, cmd, args.dry_run, allow_fail=label in non_fatal)
        if code != 0:
            print(f"\n!!! Pipeline failed at: {label}")
            return code

    print(f"\n{'═' * 70}")
    print(f"  Pipeline completed in {time.time() - t0:.1f}s ({len(steps)} steps)")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
