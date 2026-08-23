"""Per-variant attribution: exit path, MFE capture, hold time, entry location.

Answers "where is the expectancy going" for one variant without rebuilding the
full round diagnostics bundle.
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.stock.auto.iaric.run_baseline import (
    DATA_DIR,
    END_DATE,
    INITIAL_EQUITY,
    START_DATE,
    run_variant,
    variant_definitions,
)


def _f(trade, *names, default=0.0):
    for name in names:
        value = getattr(trade, name, None)
        if value is not None:
            return float(value)
    return default


def report(name: str, result) -> None:
    trades = result.trades
    if not trades:
        print(f"  {name}: no trades")
        return
    rs = np.array([float(t.r_multiple) for t in trades])
    rps = np.array([max(_f(t, "risk_per_share"), 1e-9) for t in trades])
    mfes = np.array([_f(t, "max_favorable") for t in trades]) / rps
    maes = np.abs(np.array([_f(t, "max_adverse") for t in trades])) / rps

    print(f"\n{'=' * 78}\n  {name}   n={len(trades)}  totR={rs.sum():+.2f}  avgR={rs.mean():+.3f}\n{'=' * 78}")

    total_mfe = mfes.sum()
    if total_mfe > 0:
        print(f"  MFE total={total_mfe:+.2f}R  realized={rs.sum():+.2f}R  "
              f"capture={rs.sum() / total_mfe * 100:.1f}%  giveback={total_mfe - rs.sum():+.2f}R")
    print(f"  mean MFE={mfes.mean():+.3f}  median MFE={np.median(mfes):+.3f}  "
          f"mean MAE={maes.mean():.3f}")

    by_exit: dict[str, list[float]] = defaultdict(list)
    by_exit_mfe: dict[str, list[float]] = defaultdict(list)
    for t, r, m in zip(trades, rs, mfes):
        reason = str(getattr(t, "exit_reason", "") or "UNKNOWN")
        by_exit[reason].append(float(r))
        by_exit_mfe[reason].append(float(m))
    print(f"\n  {'Exit':<18}{'n':>5}{'WR':>8}{'avgR':>9}{'totR':>9}{'meanMFE':>10}{'capture':>9}")
    for reason in sorted(by_exit, key=lambda k: -sum(by_exit[k])):
        vals = np.array(by_exit[reason])
        mv = np.array(by_exit_mfe[reason])
        cap = (vals.sum() / mv.sum() * 100.0) if mv.sum() > 0 else float("nan")
        print(f"  {reason:<18}{len(vals):>5}{(vals > 0).mean() * 100:>7.1f}%"
              f"{vals.mean():>+9.3f}{vals.sum():>+9.2f}{mv.mean():>+10.3f}{cap:>8.1f}%")

    # How much of the book ever had a real profit that was then surrendered?
    losers = rs <= 0
    if losers.any():
        lm = mfes[losers]
        print(f"\n  Losers n={losers.sum()}  meanMFE={lm.mean():+.3f}  "
              f">0.25R={(lm >= 0.25).mean() * 100:.1f}%  >0.50R={(lm >= 0.50).mean() * 100:.1f}%")

    # MFE cohorts: is there path opportunity at all?
    print(f"\n  {'MFE >=':<10}{'n':>5}{'totR':>9}{'nonpos':>9}")
    for thr in (0.25, 0.50, 0.75, 1.00):
        mask = mfes >= thr
        if mask.any():
            print(f"  {thr:<10.2f}{mask.sum():>5}{rs[mask].sum():>+9.2f}"
                  f"{(rs[mask] <= 0).mean() * 100:>8.1f}%")

    holds = np.array([_f(t, "hold_bars", default=np.nan) for t in trades])
    if np.isfinite(holds).any():
        print(f"\n  {'Hold bars':<12}{'n':>5}{'avgR':>9}")
        for lo, hi, label in ((0, 2, "0-2"), (3, 6, "3-6"), (7, 12, "7-12"),
                              (13, 24, "13-24"), (25, 10**6, ">24")):
            mask = (holds >= lo) & (holds <= hi)
            if mask.any():
                print(f"  {label:<12}{mask.sum():>5}{rs[mask].mean():>+9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()

    from backtests.stock.data.replay_cache import load_research_replay_bundle

    defs = variant_definitions()
    if args.variant not in defs:
        print(f"unknown variant; available: {list(defs)}")
        return 2
    replay = load_research_replay_bundle(DATA_DIR, require_bundle=False).data
    out = run_variant(replay, defs[args.variant], START_DATE, END_DATE, INITIAL_EQUITY)
    report(args.variant, out["_result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
