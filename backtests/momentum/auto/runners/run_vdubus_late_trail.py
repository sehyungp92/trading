"""VdubusNQ Late Trail optimization -- single-phase greedy test.

Tests the v4.4 late trail system (independent of plus_1r_partial) across
24 experiment variants covering activation, BE, trail width, and lookback.

Starts from R1 cumulative mutations: plus_1r_partial=False, CHOP_THRESHOLD=40,
MAX_POSITION_BARS_15M=64, MOM_N=65, EARLY_KILL_R=-0.4.

Usage:
    cd trading
    python -u backtests/momentum/auto/runners/run_vdubus_late_trail.py [--max-workers 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_root))

from backtests.momentum._aliases import install

install()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vdubus_late_trail")

# Suppress noisy loggers
logging.getLogger("strategies.momentum.vdub").setLevel(logging.WARNING)
logging.getLogger("backtests.momentum.engine.vdubus_engine").setLevel(logging.WARNING)

from backtests.shared.auto.greedy_optimizer import run_greedy, save_greedy_result
from backtests.shared.auto.types import Experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EQUITY = 10_000.0
DATA_DIR = _root / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = _root / "backtests" / "momentum" / "auto" / "vdubus" / "output" / "late_trail"

# R1 cumulative mutations (baseline for late trail experiments)
BASE_MUTATIONS: dict[str, object] = {
    "flags.plus_1r_partial": False,
    "param_overrides.CHOP_THRESHOLD": 40,
    "param_overrides.MAX_POSITION_BARS_15M": 64,
    "param_overrides.MOM_N": 65,
    "param_overrides.EARLY_KILL_R": -0.4,
}

# Reuse immutable VdubusPlugin scoring (phase 1 weights + rejects)
LATE_TRAIL_PHASE = 1
LATE_TRAIL_HARD_REJECTS: dict[str, float] = {
    "max_dd_pct": 0.22,
    "min_trades": 80,
    "min_pf": 1.5,
}


# ---------------------------------------------------------------------------
# Build 24 experiments
# ---------------------------------------------------------------------------

def _build_candidates() -> list[Experiment]:
    seen: set[str] = set()
    candidates: list[Experiment] = []

    def add(name: str, muts: dict) -> None:
        if name in seen:
            return
        seen.add(name)
        # All experiments enable the late_trail flag
        full_muts = {"flags.late_trail": True}
        full_muts.update(muts)
        candidates.append(Experiment(name=name, mutations=full_muts))

    # --- Group A: Activation Threshold (4) ---
    add("lt_activate_10", {
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 1.0,
        "param_overrides.LATE_TRAIL_BE_R": 0.75,
    })
    add("lt_activate_15", {})  # defaults: ACTIVATE=1.5, BE=1.0
    add("lt_activate_20", {
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 2.0,
    })
    add("lt_activate_25", {
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 2.5,
        "param_overrides.LATE_TRAIL_BE_R": 2.0,
    })

    # --- Group B: Breakeven Threshold (4) ---
    add("lt_be_060", {"param_overrides.LATE_TRAIL_BE_R": 0.60})
    add("lt_be_075", {"param_overrides.LATE_TRAIL_BE_R": 0.75})
    add("lt_be_15", {"param_overrides.LATE_TRAIL_BE_R": 1.5})
    add("lt_no_be", {"param_overrides.LATE_TRAIL_BE_R": 0})

    # --- Group C: Trail Width (4) ---
    add("lt_trail_30", {
        "param_overrides.LATE_TRAIL_MULT": 3.0,
        "param_overrides.LATE_TRAIL_MULT_MIN": 1.5,
    })
    add("lt_trail_40", {})  # defaults: MULT=4.0, MIN=2.0
    add("lt_trail_50", {
        "param_overrides.LATE_TRAIL_MULT": 5.0,
        "param_overrides.LATE_TRAIL_MULT_MIN": 2.5,
    })
    add("lt_trail_60", {
        "param_overrides.LATE_TRAIL_MULT": 6.0,
        "param_overrides.LATE_TRAIL_MULT_MIN": 3.0,
    })

    # --- Group D: Lookback & Tightening (4) ---
    add("lt_lb_12", {"param_overrides.LATE_TRAIL_LOOKBACK": 12})
    add("lt_lb_30", {"param_overrides.LATE_TRAIL_LOOKBACK": 30})
    add("lt_no_tighten", {"param_overrides.LATE_TRAIL_TIGHTEN_R": 999})
    add("lt_fast_tighten", {
        "param_overrides.LATE_TRAIL_TIGHTEN_R": 1.5,
        "param_overrides.LATE_TRAIL_TIGHTEN_DIVISOR": 4.0,
    })

    # --- Group E: Interactions (4) ---
    add("lt_plus_mfe_exempt", {"flags.stale_mfe_exempt": True})
    add("lt_plus_ratchet", {"flags.mfe_ratchet": True})
    add("lt_plus_mfe_exempt_ratchet", {
        "flags.stale_mfe_exempt": True,
        "flags.mfe_ratchet": True,
    })
    add("lt_core_tighten_080", {
        "param_overrides.LATE_TRAIL_WINDOW_MULT": {"OPEN": 1.0, "CORE": 0.80, "CLOSE": 1.0, "EVENING": 1.0},
    })

    # --- Group F: Profiles (4) ---
    add("lt_stale_saver", {
        "param_overrides.LATE_TRAIL_BE_R": 0.60,
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 1.0,
        "param_overrides.LATE_TRAIL_MULT": 3.5,
        "param_overrides.LATE_TRAIL_LOOKBACK": 16,
    })
    add("lt_runner_protector", {
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 2.0,
        "param_overrides.LATE_TRAIL_BE_R": 1.5,
        "param_overrides.LATE_TRAIL_MULT": 5.0,
        "param_overrides.LATE_TRAIL_LOOKBACK": 30,
    })
    add("lt_balanced", {})  # same as lt_activate_15 but named for clarity
    add("lt_be_only", {
        "param_overrides.LATE_TRAIL_BE_R": 0.75,
        "param_overrides.LATE_TRAIL_ACTIVATE_R": 999,
    })

    # Deduplicate (lt_balanced == lt_activate_15 by mutation, keep both names)
    log.info("Built %d unique late-trail candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class _PoolEvaluator:
    def __init__(self, pool: mp.Pool):
        self._pool = pool

    def __call__(self, candidates: list[Experiment], current_mutations: dict):
        from backtests.momentum.auto.vdubus.worker import score_candidate
        args = [
            (c.name, c.mutations, current_mutations,
             LATE_TRAIL_PHASE, None, LATE_TRAIL_HARD_REJECTS)
            for c in candidates
        ]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        pass


class _SequentialEvaluator:
    def __init__(self):
        self._inited = False

    def _ensure_init(self) -> None:
        if self._inited:
            return
        from backtests.momentum.auto.vdubus.worker import init_worker
        init_worker(str(DATA_DIR), EQUITY)
        self._inited = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict):
        self._ensure_init()
        from backtests.momentum.auto.vdubus.worker import score_candidate
        return [
            score_candidate((c.name, c.mutations, current_mutations,
                             LATE_TRAIL_PHASE, None, LATE_TRAIL_HARD_REJECTS))
            for c in candidates
        ]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_workers: int = 2) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== VdubusNQ Late Trail Optimization ===")
    log.info("Base mutations: %s", json.dumps(BASE_MUTATIONS, indent=2, default=str))
    log.info("Hard rejects: %s", LATE_TRAIL_HARD_REJECTS)

    candidates = _build_candidates()

    pool = None
    try:
        from backtests.momentum.auto.vdubus.worker import init_worker
        pool = mp.Pool(processes=max_workers, initializer=init_worker,
                       initargs=(str(DATA_DIR), EQUITY))
        evaluator = _PoolEvaluator(pool)
        log.info("Using pool evaluator (%d workers)", max_workers)
    except Exception as exc:
        log.warning("Pool creation failed (%s), falling back to sequential", exc)
        evaluator = _SequentialEvaluator()

    t0 = time.time()
    try:
        result = run_greedy(
            candidates=candidates,
            base_mutations=dict(BASE_MUTATIONS),
            evaluate_batch=evaluator,
            max_rounds=30,
            min_delta=0.005,
            prune_threshold=0.05,
            checkpoint_path=OUTPUT_DIR / "late_trail_checkpoint.json",
            logger=log,
        )
    finally:
        evaluator.close()
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    log.info("Greedy completed in %.0fs", elapsed)

    # Save results
    save_greedy_result(result, OUTPUT_DIR / "late_trail_result.json")
    log.info("Saved result to %s", OUTPUT_DIR / "late_trail_result.json")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("VDUBUS LATE TRAIL OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBase score:  {result.base_score:.4f}")
    print(f"Final score: {result.final_score:.4f}")
    if result.base_score > 0:
        print(f"Delta:       {(result.final_score - result.base_score) / result.base_score * 100:+.2f}%")
    print(f"Candidates:  {result.total_candidates}")
    print(f"Accepted:    {result.accepted_count}")
    print(f"Elapsed:     {elapsed:.0f}s")

    if result.kept_features:
        print("\nAccepted mutations (in order):")
        for feat in result.kept_features:
            print(f"  + {feat}")

    print("\nFinal mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")

    # --- Metrics comparison ---
    print("\n" + "-" * 70)
    print("METRICS COMPARISON: R1 baseline vs late-trail-optimized")
    print("-" * 70)

    try:
        from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
        from backtest.engine.vdubus_engine import VdubusEngine
        from backtests.momentum.auto.config_mutator import mutate_vdubus_config
        from backtests.momentum.auto.vdubus.scoring import composite_score, extract_vdubus_metrics
        from backtests.momentum.auto.vdubus.worker import load_worker_data

        data = load_worker_data("NQ", DATA_DIR)

        configs = [
            ("R1 baseline", dict(BASE_MUTATIONS)),
            ("Late-trail-opt", dict(result.final_mutations)),
        ]

        all_metrics = []
        for label, muts in configs:
            cfg = mutate_vdubus_config(
                VdubusBacktestConfig(
                    initial_equity=EQUITY,
                    data_dir=DATA_DIR,
                    fixed_qty=10,
                    flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
                ),
                muts,
            )
            engine = VdubusEngine("NQ", cfg)
            r = engine.run(**data)
            m = extract_vdubus_metrics(
                r.trades, list(r.equity_curve), list(r.time_series), EQUITY,
            )
            sc = composite_score(m)
            all_metrics.append((label, m, sc))

        header = f"{'Metric':<22}"
        for label, _, _ in all_metrics:
            header += f" {label:>22}"
        print(header)
        print("-" * (22 + 23 * len(all_metrics)))

        rows = [
            ("Total trades", lambda m: f"{m.total_trades}"),
            ("Win rate", lambda m: f"{m.win_rate:.1%}"),
            ("Profit factor", lambda m: f"{m.profit_factor:.2f}"),
            ("Net return %", lambda m: f"{m.net_return_pct:.1f}%"),
            ("Max DD %", lambda m: f"{m.max_dd_pct:.2%}"),
            ("Calmar", lambda m: f"{m.calmar:.1f}"),
            ("Sharpe", lambda m: f"{m.sharpe:.2f}"),
            ("Sortino", lambda m: f"{m.sortino:.2f}"),
            ("Avg R", lambda m: f"{m.avg_r:.3f}"),
            ("Capture ratio", lambda m: f"{m.capture_ratio:.2f}"),
            ("Stale exit %", lambda m: f"{m.stale_exit_pct:.1%}"),
            ("Avg hold hours", lambda m: f"{m.avg_hold_hours:.1f}"),
        ]

        for label, fmt in rows:
            line = f"{label:<22}"
            for _, m, _ in all_metrics:
                line += f" {fmt(m):>22}"
            print(line)

        # Composite scores
        print()
        print(f"{'Composite score':<22}", end="")
        for _, _, sc in all_metrics:
            print(f" {sc.total:>22.4f}", end="")
        print()

        # Final equity
        print()
        for label, m, _ in all_metrics:
            eq = EQUITY * (1 + m.net_return_pct / 100)
            print(f"{label}: ${eq:,.0f}")

    except Exception as exc:
        log.error("Comparison failed: %s", exc, exc_info=True)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VdubusNQ Late Trail Optimization")
    parser.add_argument("--max-workers", type=int, default=2, help="Max parallel workers")
    args = parser.parse_args()
    main(max_workers=args.max_workers)
