"""Greedy forward selection for optimal regime MetaConfig.

Algorithm:
  1. Load cached data (one-time)
  2. Score baseline MetaConfig() → base_score
  3. For each round: try each remaining candidate merged with accepted mutations
  4. Accept best if delta >= min_delta
  5. Stop when no improvement

Uses multiprocessing with module-level worker state (same pattern as momentum).
Tiered evaluation: non-HMM candidates reuse cached HMM models for ~2.3× speedup.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level worker state (initialized once per process)
_worker_macro_df = None
_worker_market_df = None
_worker_strat_ret_df = None
_worker_sim_cfg = None
_worker_growth_feature = None
_worker_inflation_feature = None
_worker_phase = None  # Phase number for phase-aware scoring (None = legacy)

# HMM cache state (per-worker, built lazily)
_worker_hmm_cache = None
_worker_cache_key = None


def _init_worker(
    data_dir_str: str,
    initial_equity: float,
    rebalance_cost_bps: float,
    growth_feature: str,
    inflation_feature: str,
    phase: int = 0,
) -> None:
    """Initialize worker process with shared data."""
    global _worker_macro_df, _worker_market_df, _worker_strat_ret_df
    global _worker_sim_cfg, _worker_growth_feature, _worker_inflation_feature
    global _worker_phase

    from research.backtests.regime._aliases import install
    install()

    from research.backtests.regime.config import RegimeBacktestConfig
    from research.backtests.regime.data.downloader import load_cached_data

    data_dir = Path(data_dir_str)
    _worker_macro_df, _worker_market_df, _worker_strat_ret_df = load_cached_data(data_dir)
    _worker_sim_cfg = RegimeBacktestConfig(
        initial_equity=initial_equity,
        rebalance_cost_bps=rebalance_cost_bps,
        data_dir=data_dir,
    )
    _worker_growth_feature = growth_feature
    _worker_inflation_feature = inflation_feature
    _worker_phase = phase if phase > 0 else None


def _worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a MetaConfig via full engine run (HMM fitting included).

    Returns:
        (score, rejected, reject_reason)
    """
    try:
        from regime.config import MetaConfig
        from regime.engine import run_signal_engine

        from research.backtests.regime.auto.config_mutator import mutate_meta_config
        from research.backtests.regime.engine.portfolio_sim import simulate_portfolio

        cfg = mutate_meta_config(MetaConfig(), mutations)

        signals = run_signal_engine(
            macro_df=_worker_macro_df,
            strat_ret_df=_worker_strat_ret_df,
            market_df=_worker_market_df,
            growth_feature=_worker_growth_feature,
            inflation_feature=_worker_inflation_feature,
            cfg=cfg,
        )

        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)

        if _worker_phase is not None:
            from research.backtests.regime.auto.phase_scoring import (
                compute_regime_stats,
                get_phase_scorer,
            )
            scorer = get_phase_scorer(_worker_phase)
            regime_stats = compute_regime_stats(signals)
            score = scorer(result.metrics, regime_stats)
        else:
            from research.backtests.regime.auto.scoring import composite_score
            score = composite_score(result.metrics)

        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Worker failed: %s", exc)
        return 0.0, True, str(exc)


def _worker_score_cached(args: tuple) -> tuple[float, bool, str]:
    """Score a MetaConfig using cached HMM models (skips HMM fitting).

    Args:
        args: (mutations_dict, base_mutations_for_cache)

    Returns:
        (score, rejected, reject_reason)
    """
    mutations, base_muts = args
    global _worker_hmm_cache, _worker_cache_key

    try:
        from regime.config import MetaConfig

        from research.backtests.regime.auto.config_mutator import mutate_meta_config
        from research.backtests.regime.engine.cached_engine import (
            build_hmm_cache,
            hmm_cache_key,
            run_from_cache,
        )
        from research.backtests.regime.engine.portfolio_sim import simulate_portfolio

        # Build/update cache lazily (once per worker per round)
        needed_key = hmm_cache_key(base_muts)
        if _worker_hmm_cache is None or _worker_cache_key != needed_key:
            base_cfg = mutate_meta_config(MetaConfig(), base_muts)
            _worker_hmm_cache = build_hmm_cache(
                _worker_macro_df, _worker_strat_ret_df, _worker_market_df,
                _worker_growth_feature, _worker_inflation_feature, base_cfg,
            )
            _worker_cache_key = needed_key

        cfg = mutate_meta_config(MetaConfig(), mutations)
        signals = run_from_cache(
            _worker_hmm_cache, _worker_strat_ret_df, _worker_market_df, cfg,
        )

        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)

        if _worker_phase is not None:
            from research.backtests.regime.auto.phase_scoring import (
                compute_regime_stats,
                get_phase_scorer,
            )
            scorer = get_phase_scorer(_worker_phase)
            regime_stats = compute_regime_stats(signals)
            score = scorer(result.metrics, regime_stats)
        else:
            from research.backtests.regime.auto.scoring import composite_score
            score = composite_score(result.metrics)

        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Cached worker failed: %s", exc)
        return 0.0, True, str(exc)


def _worker_final_metrics(mutations: dict) -> dict:
    """Run final evaluation in worker and return metrics dict."""
    try:
        from dataclasses import asdict

        from regime.config import MetaConfig
        from regime.engine import run_signal_engine

        from research.backtests.regime.auto.config_mutator import mutate_meta_config
        from research.backtests.regime.engine.portfolio_sim import simulate_portfolio

        cfg = mutate_meta_config(MetaConfig(), mutations)
        signals = run_signal_engine(
            macro_df=_worker_macro_df,
            strat_ret_df=_worker_strat_ret_df,
            market_df=_worker_market_df,
            growth_feature=_worker_growth_feature,
            inflation_feature=_worker_inflation_feature,
            cfg=cfg,
        )
        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)
        return asdict(result.metrics)
    except Exception as exc:
        logger.warning("Failed to get final metrics: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Greedy result types
# ---------------------------------------------------------------------------

@dataclass
class GreedyRound:
    """One round of greedy selection."""
    round_num: int
    candidate_id: str
    candidate_mutations: dict
    score_before: float
    score_after: float
    delta: float
    accepted: bool


@dataclass
class GreedyResult:
    """Final result of greedy optimization."""
    baseline_score: float
    final_score: float
    accepted_mutations: dict = field(default_factory=dict)
    rounds: list[GreedyRound] = field(default_factory=list)
    total_candidates_tested: int = 0
    elapsed_seconds: float = 0.0
    final_metrics: dict = field(default_factory=dict)
    max_rounds: int = 50


def _serialize_value(v: Any) -> Any:
    """Make values JSON-serializable."""
    if isinstance(v, (tuple, list)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _default_workers() -> int:
    """Conservative default worker count (Windows multiprocessing is fragile)."""
    n_cpu = mp.cpu_count()
    if platform.system() == "Windows":
        return min(max(1, n_cpu // 3), 4)
    return max(1, n_cpu - 1)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def run_greedy(
    candidates: list[tuple[str, dict]],
    data_dir: Path | None = None,
    initial_equity: float = 100_000.0,
    rebalance_cost_bps: float = 5.0,
    growth_feature: str = "GROWTH",
    inflation_feature: str = "INFLATION",
    max_workers: int | None = None,
    max_rounds: int = 50,
    min_delta: float = 0.001,
    prune_threshold: float = -0.10,
    base_mutations: dict | None = None,
    verbose: bool = True,
    phase: int = 0,
) -> GreedyResult:
    """Run greedy forward selection on MetaConfig parameters.

    Uses tiered evaluation: candidates that don't affect HMM fitting reuse
    cached HMM models, providing ~2-3× speedup per round.

    Args:
        candidates: List of (name, mutations_dict) to try.
        data_dir: Path to cached parquet data.
        initial_equity: Starting equity.
        rebalance_cost_bps: Per-unit turnover cost.
        growth_feature: Column name for growth in macro_df.
        inflation_feature: Column name for inflation in macro_df.
        max_workers: Pool size (None = auto, conservative on Windows).
        max_rounds: Stop after this many rounds.
        min_delta: Minimum score improvement to accept.
        prune_threshold: Remove candidates with delta below this (default -0.10).
            Also prunes worker failures (rejected/exceptions). Set to None to disable.
        base_mutations: Pre-applied mutations forming the baseline.
        verbose: Print progress.

    Returns:
        GreedyResult with optimal mutations and round-by-round history.
    """
    from research.backtests.regime.engine.cached_engine import mutations_affect_hmm

    if data_dir is None:
        data_dir = Path("research/backtests/regime/data/raw")

    t0 = time.time()
    n_workers = max_workers or _default_workers()
    base_muts = dict(base_mutations) if base_mutations else {}

    if verbose:
        print(f"Starting greedy optimization", flush=True)
        print(f"  Candidates: {len(candidates)}", flush=True)
        print(f"  Workers: {n_workers}", flush=True)
        print(f"  Max rounds: {max_rounds}", flush=True)
        print(f"  Min delta: {min_delta}", flush=True)

    # Create pool once, reuse for baseline + all rounds + final metrics
    initargs = (str(data_dir), initial_equity, rebalance_cost_bps,
                growth_feature, inflation_feature, phase)

    accepted_muts = dict(base_muts)
    current_score = 0.0
    remaining = list(candidates)
    rounds = []
    total_tested = 0

    with mp.Pool(n_workers, initializer=_init_worker, initargs=initargs) as pool:
        # Score baseline
        baseline_result = pool.apply(_worker_score, (base_muts,))
        baseline_score = baseline_result[0]

        if verbose:
            rejected = baseline_result[1]
            reason = baseline_result[2]
            status = f"REJECTED ({reason})" if rejected else "OK"
            print(f"  Baseline score: {baseline_score:.4f} [{status}]", flush=True)

        current_score = baseline_score

        for round_num in range(1, max_rounds + 1):
            if not remaining:
                if verbose:
                    print(f"\nNo candidates remaining. Stopping.", flush=True)
                break

            # Split candidates into HMM-affecting (full run) and cacheable (fast)
            hmm_tasks = []
            fast_tasks = []
            for name, muts in remaining:
                merged = {**accepted_muts, **muts}
                if mutations_affect_hmm(muts):
                    hmm_tasks.append((name, muts, merged))
                else:
                    fast_tasks.append((name, muts, merged))

            n_hmm = len(hmm_tasks)
            n_fast = len(fast_tasks)
            if verbose:
                print(f"\n--- Round {round_num} ({n_hmm + n_fast} candidates: "
                      f"{n_hmm} full + {n_fast} cached) ---", flush=True)

            # Evaluate HMM-affecting candidates (full engine)
            hmm_results = []
            if hmm_tasks:
                t_hmm = time.time()
                hmm_results = pool.map(_worker_score, [t[2] for t in hmm_tasks])
                if verbose:
                    print(f"  HMM evals done: {len(hmm_tasks)} in "
                          f"{time.time() - t_hmm:.0f}s", flush=True)

            # Evaluate non-HMM candidates (cached engine — ~6-10× faster each)
            fast_results = []
            if fast_tasks:
                t_fast = time.time()
                fast_args = [(t[2], accepted_muts) for t in fast_tasks]
                fast_results = pool.map(_worker_score_cached, fast_args)
                if verbose:
                    print(f"  Cached evals done: {len(fast_tasks)} in "
                          f"{time.time() - t_fast:.0f}s", flush=True)

            # Combine results
            all_tasks = hmm_tasks + fast_tasks
            all_results = list(hmm_results) + list(fast_results)
            total_tested += len(all_tasks)

            # Find best
            best_idx = -1
            best_score = current_score
            best_delta = 0.0

            for i, (score, rejected, reason) in enumerate(all_results):
                if rejected:
                    continue
                delta = score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_score = score
                    best_idx = i

            # Prune candidates that failed or scored very negatively
            if prune_threshold is not None:
                prune_names = set()
                for i, (score, rejected, reason) in enumerate(all_results):
                    name_i = all_tasks[i][0]
                    if rejected:
                        prune_names.add(name_i)
                    else:
                        delta_i = score - current_score
                        if delta_i < prune_threshold:
                            prune_names.add(name_i)

                if prune_names and verbose:
                    print(f"  PRUNED {len(prune_names)} candidates "
                          f"(failed or delta < {prune_threshold})", flush=True)
                remaining = [(n, m) for n, m in remaining
                             if n not in prune_names]

            if best_idx >= 0 and best_delta >= min_delta:
                name, muts, _ = all_tasks[best_idx]
                accepted_muts.update(muts)

                round_rec = GreedyRound(
                    round_num=round_num,
                    candidate_id=name,
                    candidate_mutations={k: _serialize_value(v) for k, v in muts.items()},
                    score_before=current_score,
                    score_after=best_score,
                    delta=best_delta,
                    accepted=True,
                )
                rounds.append(round_rec)

                if verbose:
                    print(f"  ACCEPTED: {name}  "
                          f"score={best_score:.4f}  delta=+{best_delta:.4f}", flush=True)

                current_score = best_score
                remaining = [(n, m) for n, m in remaining if n != name]
            else:
                if verbose:
                    print(f"  No improvement >= {min_delta}. Stopping.", flush=True)
                break

        # Get final metrics via a worker (avoids importing regime in main process)
        final_result = pool.apply(_worker_final_metrics, (accepted_muts,))

    elapsed = time.time() - t0
    final_metrics = final_result if isinstance(final_result, dict) else {}

    greedy_result = GreedyResult(
        baseline_score=baseline_score,
        final_score=current_score,
        accepted_mutations={k: _serialize_value(v) for k, v in accepted_muts.items()},
        rounds=rounds,
        total_candidates_tested=total_tested,
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
        max_rounds=max_rounds,
    )

    if verbose:
        print(f"\n=== Greedy Optimization Complete ===", flush=True)
        print(f"  Baseline: {baseline_score:.4f}", flush=True)
        print(f"  Final:    {current_score:.4f}  (+{current_score - baseline_score:.4f})", flush=True)
        print(f"  Rounds:   {len(rounds)}", flush=True)
        print(f"  Tested:   {total_tested}", flush=True)
        print(f"  Elapsed:  {elapsed:.0f}s", flush=True)
        print(f"  Accepted: {list(accepted_muts.keys())}", flush=True)

    return greedy_result


def save_greedy_result(result: GreedyResult, path: Path) -> None:
    """Save greedy result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "baseline_score": result.baseline_score,
        "final_score": result.final_score,
        "accepted_mutations": result.accepted_mutations,
        "total_candidates_tested": result.total_candidates_tested,
        "elapsed_seconds": result.elapsed_seconds,
        "final_metrics": result.final_metrics,
        "rounds": [
            {
                "round_num": r.round_num,
                "candidate_id": r.candidate_id,
                "candidate_mutations": r.candidate_mutations,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "delta": r.delta,
                "accepted": r.accepted,
            }
            for r in result.rounds
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved greedy result to {path}", flush=True)
