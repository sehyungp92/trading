"""NQDTC ablation greedy optimization -- flat sweep of ALL experiments.

Starts from the optimized config (8 accepted mutations from phased auto)
and tests ALL candidates from phases 1-3 plus finetune +-10/20% variants
and explicit removal candidates. Uses corrected scoring with Sortino
component and recalibrated Calmar scale.

Usage:
    cd trading
    nohup python -u -m backtests.momentum.auto.runners.run_nqdtc_ablation_greedy 2>&1 | tee backtests/momentum/auto/nqdtc/output/ablation_greedy.log &
"""
from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
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
log = logging.getLogger("nqdtc_ablation")

# Suppress noisy loggers
logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

from backtests.momentum.auto.nqdtc.phase_candidates import (
    _phase_1_exit,
    _phase_2_signal,
    _phase_3_timing,
)
from backtests.momentum.auto.nqdtc.scoring import BASE_WEIGHTS
from backtests.shared.auto.greedy_optimizer import run_greedy, save_greedy_result
from backtests.shared.auto.types import Experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EQUITY = 10_000.0
DATA_DIR = _root / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = _root / "backtests" / "momentum" / "auto" / "nqdtc" / "output"
MAX_WORKERS = 2

# Previous optimization result -- starting point for ablation
OPTIMIZED_MUTATIONS: dict[str, object] = {
    "flags.max_loss_cap": True,
    "flags.max_stop_width": True,
    "param_overrides.TP1_R": 0.88,
    "param_overrides.TP1_PARTIAL_PCT": 0.75,
    "param_overrides.TP2_R": 3.5,
    "param_overrides.TP2_PARTIAL_PCT": 0.25,
    "param_overrides.SCORE_NORMAL": 0.5,
    "flags.block_eth_shorts": True,
}

# Balanced weights for ablation (close to BASE_WEIGHTS)
ABLATION_WEIGHTS: dict[str, float] = {
    "net_profit": 0.20,
    "pf": 0.15,
    "calmar": 0.15,
    "inv_dd": 0.10,
    "frequency": 0.10,
    "capture": 0.10,
    "sortino": 0.10,
    "entry_quality": 0.10,
}

# Strict hard rejects to prevent degradation
ABLATION_HARD_REJECTS: dict[str, float] = {
    "max_dd_pct": 0.25,
    "min_trades": 20,
    "min_pf": 1.0,
    "min_sortino": 1.0,
}

# Ablation phase number (use 5 to distinguish from phased auto phases 1-4)
ABLATION_PHASE = 5


# ---------------------------------------------------------------------------
# Build candidate pool
# ---------------------------------------------------------------------------

def _build_candidates(base_mutations: dict) -> list[Experiment]:
    """Pool ALL experiments from phases 1-3 + finetune + explicit removals."""
    seen: set[str] = set()
    candidates: list[Experiment] = []

    def add(name: str, muts: dict) -> None:
        if name in seen:
            return
        seen.add(name)
        candidates.append(Experiment(name=name, mutations=muts))

    # --- Phases 1-3: all static experiments ---
    for name, muts in _phase_1_exit(base_mutations):
        add(name, muts)
    for name, muts in _phase_2_signal(base_mutations):
        add(name, muts)
    for name, muts in _phase_3_timing(base_mutations):
        add(name, muts)

    # --- Explicit removal candidates (ablation of each accepted mutation) ---
    # Test reverting each accepted numeric param to engine defaults
    add("remove_tp1_r", {"param_overrides.TP1_R": 1.5})  # default EXIT_TIERS R
    add("remove_tp1_partial", {"param_overrides.TP1_PARTIAL_PCT": 0.25})  # default
    add("remove_tp2", {"param_overrides.TP2_R": 999.0})  # effectively disable TP2
    add("remove_score_normal", {"param_overrides.SCORE_NORMAL": 1.5})  # default
    add("remove_block_eth_shorts", {"flags.block_eth_shorts": False})

    # --- Finetune candidates: +-10/20% of each accepted numeric mutation ---
    for key, value in base_mutations.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base_name = key.replace("param_overrides.", "").replace("flags.", "")
        for factor, label in [(0.80, "m20"), (0.90, "m10"), (1.10, "p10"), (1.20, "p20")]:
            adjusted = value * factor
            if isinstance(value, int):
                adjusted = int(round(adjusted))
            add(f"finetune_{base_name}_{label}", {key: adjusted})

    # --- Additional experiments targeting known gaps ---
    # Wider TP sweeps around accepted TP1_R=0.88
    add("tp1_r_0.6", {"param_overrides.TP1_R": 0.6})
    add("tp1_r_0.7", {"param_overrides.TP1_R": 0.7})
    add("tp1_r_0.9", {"param_overrides.TP1_R": 0.9})
    add("tp1_r_1.2", {"param_overrides.TP1_R": 1.2})

    # TP2 variants around accepted 3.5R
    add("tp2_at_2.0R", {"param_overrides.TP2_R": 2.0, "param_overrides.TP2_PARTIAL_PCT": 0.25})
    add("tp2_at_3.0R", {"param_overrides.TP2_R": 3.0, "param_overrides.TP2_PARTIAL_PCT": 0.25})
    add("tp2_at_4.0R", {"param_overrides.TP2_R": 4.0, "param_overrides.TP2_PARTIAL_PCT": 0.25})

    # TP1 partial variants around accepted 0.75
    add("tp1_partial_50", {"param_overrides.TP1_R": 0.88, "param_overrides.TP1_PARTIAL_PCT": 0.50})
    add("tp1_partial_65", {"param_overrides.TP1_R": 0.88, "param_overrides.TP1_PARTIAL_PCT": 0.65})
    add("tp1_partial_85", {"param_overrides.TP1_R": 0.88, "param_overrides.TP1_PARTIAL_PCT": 0.85})

    # Combined exit variants
    add("tp1_low_tp2_wide", {
        "param_overrides.TP1_R": 0.7, "param_overrides.TP1_PARTIAL_PCT": 0.60,
        "param_overrides.TP2_R": 5.0, "param_overrides.TP2_PARTIAL_PCT": 0.25,
    })

    log.info("Built %d unique ablation candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Evaluate batch (multiprocessing)
# ---------------------------------------------------------------------------

class _AblationBatchEvaluator:
    """Evaluator that uses multiprocessing pool with balanced scoring."""

    def __init__(self, pool: mp.Pool):
        self._pool = pool

    def __call__(self, candidates: list[Experiment], current_mutations: dict):
        from backtests.momentum.auto.nqdtc.worker import score_candidate

        args = [
            (c.name, c.mutations, current_mutations,
             ABLATION_PHASE, ABLATION_WEIGHTS, ABLATION_HARD_REJECTS)
            for c in candidates
        ]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        pass


class _SequentialEvaluator:
    """Fallback sequential evaluator."""

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
                             ABLATION_PHASE, ABLATION_WEIGHTS, ABLATION_HARD_REJECTS))
            for c in candidates
        ]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== NQDTC Ablation Greedy Optimization ===")
    log.info("Base mutations: %s", json.dumps(OPTIMIZED_MUTATIONS, indent=2, default=str))
    log.info("Scoring weights: %s", ABLATION_WEIGHTS)
    log.info("Hard rejects: %s", ABLATION_HARD_REJECTS)

    candidates = _build_candidates(OPTIMIZED_MUTATIONS)

    # Use sequential evaluator (multiprocessing causes OOM on Windows)
    pool = None
    evaluator = _SequentialEvaluator()
    log.info("Using sequential evaluator (single process)")

    t0 = time.time()
    try:
        result = run_greedy(
            candidates=candidates,
            base_mutations=dict(OPTIMIZED_MUTATIONS),
            evaluate_batch=evaluator,
            max_rounds=30,
            min_delta=0.003,
            prune_threshold=0.05,
            checkpoint_path=OUTPUT_DIR / "ablation_greedy_checkpoint.json",
            logger=log,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    log.info("Greedy completed in %.0fs", elapsed)

    # Save results
    save_greedy_result(result, OUTPUT_DIR / "ablation_greedy_result.json")
    log.info("Saved result to %s", OUTPUT_DIR / "ablation_greedy_result.json")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("NQDTC ABLATION GREEDY OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBase score:  {result.base_score:.4f}")
    print(f"Final score: {result.final_score:.4f}")
    print(f"Delta:       {(result.final_score - result.base_score) / result.base_score * 100:+.2f}%" if result.base_score > 0 else "")
    print(f"Candidates:  {result.total_candidates}")
    print(f"Accepted:    {result.accepted_count}")
    print(f"Elapsed:     {elapsed:.0f}s")

    if result.kept_features:
        print(f"\nAccepted mutations (in order):")
        for feat in result.kept_features:
            print(f"  + {feat}")

    print(f"\nFinal mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")

    # --- Run comparison vs baseline ---
    print("\n" + "-" * 70)
    print("METRICS COMPARISON: initial baseline vs ablation-optimized")
    print("-" * 70)

    try:
        from backtest.config_nqdtc import NQDTCBacktestConfig
        from backtest.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.scoring import (
            NQDTCCompositeScore,
            composite_score,
            extract_nqdtc_metrics,
        )
        from backtests.momentum.auto.nqdtc.worker import load_worker_data

        data = load_worker_data("NQ", DATA_DIR)
        initial_muts = {"flags.max_loss_cap": True, "flags.max_stop_width": True}

        configs = [
            ("Initial baseline", initial_muts),
            ("Pre-ablation optimized", dict(OPTIMIZED_MUTATIONS)),
            ("Ablation-optimized", dict(result.final_mutations)),
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
            sc = composite_score(m, ABLATION_WEIGHTS)
            all_metrics.append((label, m, sc))

        def pct_chg(old, new):
            if old == 0:
                return "N/A"
            return f"{(new - old) / abs(old) * 100:+.1f}%"

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
            ("ETH short trades", lambda m: f"{m.eth_short_trades}", lambda m: m.eth_short_trades),
            ("Burst trade %", lambda m: f"{m.burst_trade_pct:.1%}", lambda m: m.burst_trade_pct),
            ("Avg hold hours", lambda m: f"{m.avg_hold_hours:.1f}", lambda m: m.avg_hold_hours),
        ]

        for label, fmt, extract in rows:
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
