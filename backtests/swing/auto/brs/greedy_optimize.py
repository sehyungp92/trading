"""LEGACY: This module is only used by old runner scripts (run_r*_phased.py).
New optimization rounds should use shared/auto/greedy_optimizer.py via PhaseRunner.

BRS parallel forward selection (greedy optimizer).

Follows swing auto/greedy_optimize.py worker pattern:
  _init_worker() with module-level state, _worker_score(), GreedyRound/GreedyResult
"""
from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per worker)
# ---------------------------------------------------------------------------

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_phase: int = 0
_worker_scoring_weights: dict | None = None


def _init_worker(
    data_dir_str: str,
    equity: float,
    phase: int = 0,
    scoring_weights: dict | None = None,
) -> None:
    """Initialize worker process: load data once."""
    global _worker_data, _worker_config, _worker_equity, _worker_phase, _worker_scoring_weights

    # Force UTF-8 stdout in worker processes (Windows cp949 fix)
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing._aliases import install
    install()

    from backtest.config_brs import BRSConfig
    from backtest.engine.brs_portfolio_engine import load_brs_data

    _worker_equity = equity
    _worker_phase = phase
    _worker_scoring_weights = scoring_weights
    _worker_config = BRSConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
    )
    _worker_data = load_brs_data(_worker_config)


def _worker_score(args: tuple) -> tuple[str, float, bool, str]:
    """Score a single candidate mutation set.

    Args:
        args: (name, candidate_mutations, base_mutations)

    Returns:
        (name, score, rejected, reason)
    """
    name, candidate_muts, base_muts = args

    try:
        from dataclasses import replace
        from backtest.config_brs import BRSConfig
        from backtest.engine.brs_portfolio_engine import run_brs_synchronized
        from backtests.swing.auto.brs.config_mutator import mutate_brs_config
        from backtests.swing.auto.brs.scoring import extract_brs_metrics
        from backtests.swing.auto.brs.phase_scoring import score_phase

        # Merge base + candidate mutations
        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_brs_config(_worker_config, all_muts)
        result = run_brs_synchronized(_worker_data, config)
        metrics = extract_brs_metrics(result, _worker_equity)

        if _worker_phase > 0:
            score = score_phase(_worker_phase, metrics, _worker_scoring_weights)
        else:
            from backtests.swing.auto.brs.scoring import composite_score
            score = composite_score(metrics)

        if score.rejected:
            return (name, 0.0, True, score.reject_reason)
        return (name, score.total, False, "")

    except Exception:
        return (name, 0.0, True, traceback.format_exc())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GreedyRound:
    """Result of a single greedy selection round."""
    round_num: int
    candidates_tested: int
    best_name: str
    best_score: float
    best_delta_pct: float
    kept: bool
    all_scores: list[tuple[str, float, float]]  # (name, score, delta_pct)


@dataclass
class GreedyResult:
    """Full result of greedy forward selection."""
    base_score: float
    final_mutations: dict
    final_score: float
    kept_features: list[str]
    rounds: list[GreedyRound]
    final_trades: int = 0
    final_pf: float = 0.0
    final_dd_pct: float = 0.0
    final_return_pct: float = 0.0
    final_sharpe: float = 0.0
    n_rounds: int = 0
    max_rounds: int = 0
    elapsed_seconds: float = 0.0
    final_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main greedy loop
# ---------------------------------------------------------------------------

def run_greedy(
    data_dir: Path,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 100_000.0,
    base_mutations: dict | None = None,
    max_rounds: int = 0,
    max_workers: int | None = None,
    min_delta: float = 0.001,
    phase: int = 0,
    verbose: bool = True,
    scoring_weight_overrides: dict | None = None,
    checkpoint_dir: Path | None = None,
) -> GreedyResult:
    """Run greedy forward selection for BRS optimization.

    Args:
        data_dir: Path to bar data
        candidates: List of (name, mutations_dict) to evaluate
        initial_equity: Starting equity
        base_mutations: Starting mutations (from prior phases)
        max_rounds: Max selection rounds (0 = unlimited = len(candidates))
        max_workers: Parallel workers (default: cpu_count - 1)
        min_delta: Minimum score improvement to accept candidate
        phase: Optimization phase (for phase-aware scoring)
        verbose: Print progress
        scoring_weight_overrides: Optional per-component scoring weight adjustments
        checkpoint_dir: Optional directory for mid-greedy checkpoint save/resume

    Returns:
        GreedyResult with optimal mutations
    """
    if base_mutations is None:
        base_mutations = {}
    if max_rounds <= 0:
        max_rounds = len(candidates)
    if max_workers is None:
        max_workers = min(len(candidates), max(1, (os.cpu_count() or 4) - 1))

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"BRS Greedy Optimization - Phase {phase}")
        print(f"  Candidates: {len(candidates)}")
        print(f"  Workers: {max_workers}")
        print(f"  Max rounds: {max_rounds}")
        print(f"{'='*60}\n")

    # Score baseline
    if verbose:
        print("Scoring baseline...")

    from backtests.swing._aliases import install
    install()

    from backtest.config_brs import BRSConfig
    from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_synchronized
    from backtests.swing.auto.brs.config_mutator import mutate_brs_config
    from backtests.swing.auto.brs.scoring import extract_brs_metrics
    from backtests.swing.auto.brs.phase_scoring import score_phase
    from backtests.swing.auto.brs.scoring import composite_score as base_composite

    base_config = BRSConfig(initial_equity=initial_equity, data_dir=data_dir)
    base_config = mutate_brs_config(base_config, base_mutations)
    data = load_brs_data(base_config)
    base_result = run_brs_synchronized(data, base_config)
    base_metrics = extract_brs_metrics(base_result, initial_equity)

    if phase > 0:
        base_score_obj = score_phase(phase, base_metrics, scoring_weight_overrides)
    else:
        base_score_obj = base_composite(base_metrics)

    baseline_score = base_score_obj.total if not base_score_obj.rejected else 0.0

    if verbose:
        print(f"Baseline score: {baseline_score:.4f}")

    accepted_mutations = dict(base_mutations)
    current_score = baseline_score
    kept_features: list[str] = []
    rounds: list[GreedyRound] = []
    remaining = list(candidates)

    # Checkpoint resume
    checkpoint_path = checkpoint_dir / "greedy_checkpoint.json" if checkpoint_dir else None
    if checkpoint_path and checkpoint_path.exists():
        resumed = _load_checkpoint(checkpoint_path, base_mutations)
        if resumed:
            accepted_mutations, kept_features, current_score, remaining_names, saved_rounds = resumed
            remaining = [(n, m) for n, m in remaining if n in remaining_names]
            rounds = saved_rounds
            if verbose:
                print(f"  Resumed from checkpoint: {len(kept_features)} kept, "
                      f"score {current_score:.4f}, {len(remaining)} remaining")

    # Greedy rounds
    with mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), initial_equity, phase, scoring_weight_overrides),
    ) as pool:
        for round_num in range(1, max_rounds + 1):
            if not remaining:
                break

            if verbose:
                print(f"\nRound {round_num}/{max_rounds}: testing {len(remaining)} candidates...")

            # Score all remaining candidates
            args_list = [
                (name, muts, accepted_mutations) for name, muts in remaining
            ]
            results = pool.map(_worker_score, args_list)

            # Find best
            scored = []
            for (name, score, rejected, reason) in results:
                if not rejected and score > 0:
                    delta_pct = (score - current_score) / current_score * 100 if current_score > 0 else 0
                    scored.append((name, score, delta_pct))

            scored.sort(key=lambda x: x[1], reverse=True)

            if not scored:
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(remaining),
                    best_name="",
                    best_score=0.0,
                    best_delta_pct=0.0,
                    kept=False,
                    all_scores=[],
                ))
                if verbose:
                    print(f"  No valid candidates - stopping")
                break

            best_name, best_score, best_delta = scored[0]
            improved = best_score > current_score + min_delta

            rounds.append(GreedyRound(
                round_num=round_num,
                candidates_tested=len(remaining),
                best_name=best_name,
                best_score=best_score,
                best_delta_pct=best_delta,
                kept=improved,
                all_scores=scored[:10],  # top 10
            ))

            if improved:
                # Accept this candidate
                cand_muts = dict(next(m for n, m in remaining if n == best_name))
                accepted_mutations.update(cand_muts)
                current_score = best_score
                kept_features.append(best_name)
                remaining = [(n, m) for n, m in remaining if n != best_name]

                # Save checkpoint after each acceptance
                if checkpoint_path:
                    _save_checkpoint(
                        checkpoint_path, accepted_mutations, kept_features,
                        current_score, [n for n, _ in remaining], rounds, base_mutations,
                    )

                if verbose:
                    print(f"  KEPT: {best_name} (score {best_score:.4f}, +{best_delta:.1f}%)")
            else:
                if verbose:
                    print(f"  No improvement > {min_delta:.4f} - stopping")
                break

    elapsed = time.time() - start_time

    # Clean up checkpoint on successful completion
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    # Final metrics
    final_config = mutate_brs_config(
        BRSConfig(initial_equity=initial_equity, data_dir=data_dir),
        accepted_mutations,
    )
    final_result = run_brs_synchronized(data, final_config)
    final_metrics = extract_brs_metrics(final_result, initial_equity)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Greedy complete in {elapsed:.0f}s")
        print(f"  Kept: {len(kept_features)} features")
        print(f"  Score: {baseline_score:.4f} -> {current_score:.4f}")
        print(f"  Trades: {final_metrics.total_trades}")
        print(f"  PF: {final_metrics.profit_factor:.2f}")
        print(f"  DD: {final_metrics.max_dd_pct:.1%}")
        print(f"{'='*60}\n")

    return GreedyResult(
        base_score=baseline_score,
        final_mutations=accepted_mutations,
        final_score=current_score,
        kept_features=kept_features,
        rounds=rounds,
        final_trades=final_metrics.total_trades,
        final_pf=final_metrics.profit_factor,
        final_dd_pct=final_metrics.max_dd_pct,
        final_return_pct=final_metrics.net_return_pct,
        final_sharpe=final_metrics.sharpe,
        n_rounds=len(rounds),
        max_rounds=max_rounds,
        elapsed_seconds=elapsed,
        final_metrics=asdict(final_metrics) if hasattr(final_metrics, '__dataclass_fields__') else {},
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_hash(base_mutations: dict) -> str:
    """Stable hash of base mutations for identity check."""
    import hashlib
    return hashlib.md5(json.dumps(base_mutations, sort_keys=True).encode()).hexdigest()


def _save_checkpoint(
    path: Path,
    accepted_mutations: dict,
    kept_features: list[str],
    current_score: float,
    remaining_names: list[str],
    rounds: list[GreedyRound],
    base_mutations: dict,
) -> None:
    """Save mid-greedy checkpoint (atomic)."""
    data = {
        "accepted_mutations": accepted_mutations,
        "kept_features": kept_features,
        "current_score": current_score,
        "remaining_names": remaining_names,
        "rounds": [asdict(r) for r in rounds],
        "base_mutations_hash": _checkpoint_hash(base_mutations),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpySafeEncoder)
    os.replace(str(tmp), str(path))


def _load_checkpoint(
    path: Path,
    base_mutations: dict,
) -> tuple[dict, list[str], float, set[str], list[GreedyRound]] | None:
    """Load checkpoint if it matches current base mutations.

    Returns (accepted_mutations, kept_features, score, remaining_names, rounds) or None.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("base_mutations_hash") != _checkpoint_hash(base_mutations):
            logger.info("Checkpoint hash mismatch — starting fresh")
            return None
        rounds = [GreedyRound(**r) for r in data.get("rounds", [])]
        return (
            data["accepted_mutations"],
            data["kept_features"],
            data["current_score"],
            set(data["remaining_names"]),
            rounds,
        )
    except Exception:
        logger.warning("Failed to load checkpoint — starting fresh")
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_greedy_result(result: GreedyResult, path: Path) -> None:
    """Save greedy result to JSON (atomic write)."""
    from backtests.swing.auto.brs.phase_state import _atomic_write_json
    _atomic_write_json(asdict(result), path)


def load_greedy_result(path: Path) -> GreedyResult:
    """Load greedy result from JSON."""
    with open(path) as f:
        data = json.load(f)
    rounds = [GreedyRound(**r) for r in data.pop("rounds", [])]
    return GreedyResult(**data, rounds=rounds)
