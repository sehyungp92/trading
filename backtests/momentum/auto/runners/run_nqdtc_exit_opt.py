"""NQDTC exit alpha optimization -- adaptive chandelier, grace period, decay, ratchet.

Starts from ablation-optimized config (10 accepted mutations) and tests 48
exit-specific experiments targeting:
  1. Post-TP1 BE zone leak (~75R lost to +0.222R exits)
  2. 1-2 bar immediate stops (-$6,439)
  3. Flat CHANDELIER_ATR_MULT=1.5 replacing all tier defaults

Uses sequential evaluator (Windows OOM safety).

Usage:
    cd trading
    python -u -m backtests.momentum.auto.runners.run_nqdtc_exit_opt 2>&1 | tee backtests/momentum/auto/nqdtc/output/exit_opt.log
"""
from __future__ import annotations

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
log = logging.getLogger("nqdtc_exit_opt")

# Suppress noisy loggers
logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

from backtests.shared.auto.greedy_optimizer import run_greedy, save_greedy_result
from backtests.shared.auto.types import Experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EQUITY = 10_000.0
DATA_DIR = _root / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = _root / "backtests" / "momentum" / "auto" / "nqdtc" / "output"
MAX_WORKERS = 2

# Ablation-optimized baseline (10 mutations)
BASE_MUTATIONS: dict[str, object] = {
    "flags.max_loss_cap": True,
    "flags.max_stop_width": True,
    "param_overrides.TP1_R": 1.056,
    "param_overrides.TP1_PARTIAL_PCT": 0.25,
    "param_overrides.TP2_R": 5.0,
    "param_overrides.TP2_PARTIAL_PCT": 0.25,
    "param_overrides.SCORE_NORMAL": 0.5,
    "flags.block_eth_shorts": True,
    "param_overrides.MIN_BOX_WIDTH": 180,
    "param_overrides.CHANDELIER_ATR_MULT": 1.5,
}

# Exit-focused scoring weights (capture-heavy)
EXIT_OPT_WEIGHTS: dict[str, float] = {
    "net_profit": 0.15,
    "pf": 0.10,
    "calmar": 0.15,
    "inv_dd": 0.10,
    "frequency": 0.05,
    "capture": 0.25,
    "sortino": 0.10,
    "entry_quality": 0.10,
}

# Tight no-regression floor
EXIT_OPT_HARD_REJECTS: dict[str, float] = {
    "max_dd_pct": 0.20,
    "min_trades": 150,
    "min_pf": 2.0,
    "min_sortino": 5.0,
}

EXIT_OPT_PHASE = 6


# ---------------------------------------------------------------------------
# Build 48 experiments
# ---------------------------------------------------------------------------

def _build_candidates() -> list[Experiment]:
    seen: set[str] = set()
    candidates: list[Experiment] = []

    def add(name: str, muts: dict) -> None:
        if name in seen:
            return
        seen.add(name)
        candidates.append(Experiment(name=name, mutations=muts))

    # --- A. Per-Tier Chandelier Mults (15) ---
    add("tier0_mult_2.0", {"param_overrides.CHANDELIER_TIER0_MULT": 2.0})
    add("tier0_mult_2.5", {"param_overrides.CHANDELIER_TIER0_MULT": 2.5})
    add("tier0_mult_3.0", {"param_overrides.CHANDELIER_TIER0_MULT": 3.0})
    add("tier1_mult_1.8", {"param_overrides.CHANDELIER_TIER1_MULT": 1.8})
    add("tier1_mult_2.0", {"param_overrides.CHANDELIER_TIER1_MULT": 2.0})
    add("tier1_mult_2.3", {"param_overrides.CHANDELIER_TIER1_MULT": 2.3})
    add("tier2_mult_1.3", {"param_overrides.CHANDELIER_TIER2_MULT": 1.3})
    add("tier2_mult_1.6", {"param_overrides.CHANDELIER_TIER2_MULT": 1.6})
    add("tier2_mult_1.8", {"param_overrides.CHANDELIER_TIER2_MULT": 1.8})
    add("tier3_mult_1.2", {"param_overrides.CHANDELIER_TIER3_MULT": 1.2})
    add("tier3_mult_1.5", {"param_overrides.CHANDELIER_TIER3_MULT": 1.5})
    add("tier3_mult_1.8", {"param_overrides.CHANDELIER_TIER3_MULT": 1.8})
    add("tier4_mult_0.8", {"param_overrides.CHANDELIER_TIER4_MULT": 0.8})
    add("tier4_mult_1.0", {"param_overrides.CHANDELIER_TIER4_MULT": 1.0})
    add("tier4_mult_1.2", {"param_overrides.CHANDELIER_TIER4_MULT": 1.2})

    # --- B. Grace Period (5) ---
    add("grace_2bars", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 2})
    add("grace_3bars", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 3})
    add("grace_4bars", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 4})
    add("grace_6bars", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 6})
    add("grace_8bars", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 8})

    # --- C. Post-TP1 Mult Decay (6) ---
    add("decay_0.02_floor_1.0", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.02,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })
    add("decay_0.02_floor_1.2", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.02,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.2,
    })
    add("decay_0.03_floor_1.0", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.03,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })
    add("decay_0.05_floor_0.8", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.05,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 0.8,
    })
    add("decay_0.05_floor_1.2", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.05,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.2,
    })
    add("decay_0.10_floor_1.0", {
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.10,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })

    # --- D. Ratchet Parameters (8) ---
    add("ratchet_lock_0.15", {"param_overrides.RATCHET_LOCK_PCT": 0.15})
    add("ratchet_lock_0.20", {"param_overrides.RATCHET_LOCK_PCT": 0.20})
    add("ratchet_lock_0.35", {"param_overrides.RATCHET_LOCK_PCT": 0.35})
    add("ratchet_lock_0.50", {"param_overrides.RATCHET_LOCK_PCT": 0.50})
    add("ratchet_thresh_1.0", {"param_overrides.RATCHET_THRESHOLD_R": 1.0})
    add("ratchet_thresh_1.2", {"param_overrides.RATCHET_THRESHOLD_R": 1.2})
    add("ratchet_thresh_2.0", {"param_overrides.RATCHET_THRESHOLD_R": 2.0})
    add("ratchet_thresh_2.5", {"param_overrides.RATCHET_THRESHOLD_R": 2.5})

    # --- E. Combined Strategies (10) ---
    add("protect_early_tighten_late", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.0,
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.03,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })
    add("aggr_ratchet_grace", {
        "param_overrides.RATCHET_LOCK_PCT": 0.40,
        "param_overrides.RATCHET_THRESHOLD_R": 1.2,
        "param_overrides.CHANDELIER_GRACE_BARS_30M": 3,
    })
    add("orig_tiers_decay", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
        "param_overrides.CHANDELIER_TIER1_MULT": 2.0,
        "param_overrides.CHANDELIER_TIER2_MULT": 1.8,
        "param_overrides.CHANDELIER_TIER3_MULT": 1.8,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.2,
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.03,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })
    add("graduated_tiers", {
        "param_overrides.CHANDELIER_TIER0_MULT": 3.0,
        "param_overrides.CHANDELIER_TIER1_MULT": 2.2,
        "param_overrides.CHANDELIER_TIER2_MULT": 1.6,
        "param_overrides.CHANDELIER_TIER3_MULT": 1.3,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.0,
    })
    add("grace_plus_decay", {
        "param_overrides.CHANDELIER_GRACE_BARS_30M": 4,
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.05,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
    })
    add("full_adaptive", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.0,
        "param_overrides.CHANDELIER_GRACE_BARS_30M": 3,
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.03,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.0,
        "param_overrides.RATCHET_LOCK_PCT": 0.35,
    })
    add("conservative_adaptive", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.0,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.2,
        "param_overrides.CHANDELIER_GRACE_BARS_30M": 2,
        "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.02,
        "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.2,
    })
    add("ratchet_first", {
        "param_overrides.RATCHET_LOCK_PCT": 0.40,
        "param_overrides.RATCHET_THRESHOLD_R": 1.0,
    })
    add("tier0_fix_ratchet", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.0,
        "param_overrides.RATCHET_LOCK_PCT": 0.35,
        "param_overrides.RATCHET_THRESHOLD_R": 1.2,
    })
    add("remove_flat_override", {
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
        "param_overrides.CHANDELIER_TIER1_MULT": 2.0,
        "param_overrides.CHANDELIER_TIER2_MULT": 1.8,
        "param_overrides.CHANDELIER_TIER3_MULT": 1.8,
        "param_overrides.CHANDELIER_TIER4_MULT": 1.2,
    })

    # --- F. Lookback Finetune (4) ---
    add("lookback_6", {"param_overrides.CHANDELIER_LOOKBACK": 6})
    add("lookback_12", {"param_overrides.CHANDELIER_LOOKBACK": 12})
    add("lb6_tier0_fix", {
        "param_overrides.CHANDELIER_LOOKBACK": 6,
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
    })
    add("lb12_tier0_fix", {
        "param_overrides.CHANDELIER_LOOKBACK": 12,
        "param_overrides.CHANDELIER_TIER0_MULT": 2.5,
    })

    log.info("Built %d unique exit-opt candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class _PoolEvaluator:
    def __init__(self, pool: mp.Pool):
        self._pool = pool

    def __call__(self, candidates: list[Experiment], current_mutations: dict):
        from backtests.momentum.auto.nqdtc.worker import score_candidate
        args = [
            (c.name, c.mutations, current_mutations,
             EXIT_OPT_PHASE, EXIT_OPT_WEIGHTS, EXIT_OPT_HARD_REJECTS)
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
        from backtests.momentum.auto.nqdtc.worker import init_worker
        init_worker(str(DATA_DIR), EQUITY)
        self._inited = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict):
        self._ensure_init()
        from backtests.momentum.auto.nqdtc.worker import score_candidate
        return [
            score_candidate((c.name, c.mutations, current_mutations,
                             EXIT_OPT_PHASE, EXIT_OPT_WEIGHTS, EXIT_OPT_HARD_REJECTS))
            for c in candidates
        ]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== NQDTC Exit Alpha Optimization ===")
    log.info("Base mutations: %s", json.dumps(BASE_MUTATIONS, indent=2, default=str))
    log.info("Scoring weights: %s", EXIT_OPT_WEIGHTS)
    log.info("Hard rejects: %s", EXIT_OPT_HARD_REJECTS)

    candidates = _build_candidates()

    pool = None
    try:
        from backtests.momentum.auto.nqdtc.worker import init_worker
        pool = mp.Pool(processes=MAX_WORKERS, initializer=init_worker,
                       initargs=(str(DATA_DIR), EQUITY))
        evaluator = _PoolEvaluator(pool)
        log.info("Using pool evaluator (%d workers)", MAX_WORKERS)
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
            min_delta=0.003,
            prune_threshold=0.05,
            checkpoint_path=OUTPUT_DIR / "exit_opt_checkpoint.json",
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
    save_greedy_result(result, OUTPUT_DIR / "exit_opt_result.json")
    log.info("Saved result to %s", OUTPUT_DIR / "exit_opt_result.json")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("NQDTC EXIT ALPHA OPTIMIZATION RESULTS")
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
    print("METRICS COMPARISON: ablation baseline vs exit-optimized")
    print("-" * 70)

    try:
        from backtest.config_nqdtc import NQDTCBacktestConfig
        from backtest.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.scoring import composite_score, extract_nqdtc_metrics
        from backtests.momentum.auto.nqdtc.worker import load_worker_data

        data = load_worker_data("NQ", DATA_DIR)

        configs = [
            ("Ablation baseline", dict(BASE_MUTATIONS)),
            ("Exit-optimized", dict(result.final_mutations)),
        ]

        all_metrics = []
        for label, muts in configs:
            cfg = mutate_nqdtc_config(
                NQDTCBacktestConfig(initial_equity=EQUITY, data_dir=DATA_DIR, fixed_qty=10),
                muts,
            )
            engine = NQDTCEngine("MNQ", cfg)
            r = engine.run(**data)
            m = extract_nqdtc_metrics(r.trades, list(r.equity_curve), list(r.timestamps), EQUITY)
            sc = composite_score(m, EXIT_OPT_WEIGHTS)
            all_metrics.append((label, m, sc))

        header = f"{'Metric':<22}"
        for label, _, _ in all_metrics:
            header += f" {label:>22}"
        print(header)
        print("-" * (22 + 23 * len(all_metrics)))

        rows = [
            ("Total trades", lambda m: f"{m.total_trades}", lambda m: m.total_trades),
            ("Win rate", lambda m: f"{m.win_rate:.1%}", lambda m: m.win_rate),
            ("Profit factor", lambda m: f"{m.profit_factor:.2f}", lambda m: m.profit_factor),
            ("Net return %", lambda m: f"{m.net_return_pct:.1f}%", lambda m: m.net_return_pct),
            ("Max DD %", lambda m: f"{m.max_dd_pct:.2%}", lambda m: m.max_dd_pct),
            ("Calmar", lambda m: f"{m.calmar:.1f}", lambda m: m.calmar),
            ("Sharpe", lambda m: f"{m.sharpe:.2f}", lambda m: m.sharpe),
            ("Sortino", lambda m: f"{m.sortino:.2f}", lambda m: m.sortino),
            ("Avg R", lambda m: f"{m.avg_r:.3f}", lambda m: m.avg_r),
            ("Capture ratio", lambda m: f"{m.capture_ratio:.3f}", lambda m: m.capture_ratio),
            ("TP1 hit rate", lambda m: f"{m.tp1_hit_rate:.1%}", lambda m: m.tp1_hit_rate),
            ("TP2 hit rate", lambda m: f"{m.tp2_hit_rate:.1%}", lambda m: m.tp2_hit_rate),
            ("Avg hold hours", lambda m: f"{m.avg_hold_hours:.1f}", lambda m: m.avg_hold_hours),
        ]

        for label, fmt, _ in rows:
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

        # Component breakdown
        print(f"\n{'Component':<22}", end="")
        for lbl, _, _ in all_metrics:
            print(f" {lbl:>22}", end="")
        print()
        for comp in ["net_profit", "pf", "calmar", "inv_dd", "frequency", "capture", "sortino", "entry_quality"]:
            print(f"  {comp:<20}", end="")
            for _, _, sc in all_metrics:
                print(f" {getattr(sc, comp):>22.3f}", end="")
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
    main()
