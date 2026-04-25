"""Greedy forward selection for optimal swing portfolio config.

Algorithm:
  1. Start with live_parity base config (default UnifiedBacktestConfig)
  2. Score the base → baseline_score
  3. For each round, test every remaining candidate IN PARALLEL
  4. Keep the best candidate if it improves score; stop when none do
  5. Output the final optimal mutations and comparison table

Uses multiprocessing to evaluate candidates concurrently within each round.
Each worker loads data independently (6s one-time cost), then reuses it.
With 12 cores and 10 candidates, round 1 runs in ~350s instead of ~3500s.

Usage:
    from backtests.swing.auto.greedy_optimize import run_greedy
    result = run_greedy(data_dir, candidates, initial_equity=10_000)
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per worker via _init_worker)
# ---------------------------------------------------------------------------
_worker_data = None
_worker_equity: float = 0.0


def _init_worker(data_dir_str: str, equity: float) -> None:
    """Initialize a worker process: install aliases, load data."""
    global _worker_data, _worker_equity

    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import load_unified_data

    _worker_equity = equity
    config = UnifiedBacktestConfig(initial_equity=equity, data_dir=Path(data_dir_str))
    _worker_data = load_unified_data(config)


def _worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a config in a worker process. Returns (score, rejected, reject_reason)."""
    global _worker_data, _worker_equity

    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.auto.scoring import composite_score, extract_metrics
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import run_unified

    try:
        config = UnifiedBacktestConfig(initial_equity=_worker_equity)
        if mutations:
            config = mutate_unified_config(config, mutations)
        result = run_unified(_worker_data, config)
        trades = _collect_trades(result)
        metrics = extract_metrics(
            trades, result.combined_equity,
            result.combined_timestamps, _worker_equity,
        )
        score = composite_score(metrics, _worker_equity, equity_curve=result.combined_equity)
        return score.total if not score.rejected else 0.0, score.rejected, score.reject_reason
    except Exception:
        return 0.0, True, traceback.format_exc()


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


# ---------------------------------------------------------------------------
# Trade collection helper
# ---------------------------------------------------------------------------

def _collect_trades(result) -> list:
    """Collect trades from a UnifiedPortfolioResult."""
    all_trades = []
    for attr in ('atrss_trades', 'helix_trades', 'breakout_trades'):
        trades = getattr(result, attr, [])
        if isinstance(trades, list):
            all_trades.extend(trades)
    strategy_results = getattr(result, 'strategy_results', {})
    if isinstance(strategy_results, dict) and not all_trades:
        for sr in strategy_results.values():
            all_trades.extend(getattr(sr, 'trades', []))
    return all_trades


# ---------------------------------------------------------------------------
# Single-process scoring (for baseline and final run)
# ---------------------------------------------------------------------------

def _score_config(
    data,
    mutations: dict,
    initial_equity: float,
):
    """Build config, run unified engine, return (score, metrics)."""
    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.auto.scoring import composite_score, extract_metrics
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import run_unified

    config = UnifiedBacktestConfig(initial_equity=initial_equity)
    if mutations:
        config = mutate_unified_config(config, mutations)
    result = run_unified(data, config)
    trades = _collect_trades(result)
    metrics = extract_metrics(
        trades, result.combined_equity,
        result.combined_timestamps, initial_equity,
    )
    score = composite_score(metrics, initial_equity, equity_curve=result.combined_equity)
    return score, metrics


# ---------------------------------------------------------------------------
# Main greedy loop
# ---------------------------------------------------------------------------

def run_greedy(
    data,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    base_mutations: dict | None = None,
    data_dir: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = True,
) -> GreedyResult:
    """Run greedy forward selection to find optimal portfolio config.

    Args:
        data: UnifiedPortfolioData (pre-loaded, used for baseline/final only)
        candidates: List of (name, mutations_dict) to test
        initial_equity: Starting equity
        base_mutations: Optional starting mutations (default: live_parity base)
        data_dir: Path to bar data (required for parallel workers to load data)
        max_workers: Number of parallel workers (default: cpu_count - 1)
        verbose: Print progress

    Returns:
        GreedyResult with optimal mutations and round-by-round history
    """
    if base_mutations is None:
        base_mutations = {}
    if max_workers is None:
        max_workers = min(len(candidates), max(1, (os.cpu_count() or 4) - 1))

    if verbose:
        print(f"\n{'='*60}")
        print("GREEDY FORWARD SELECTION: SWING PORTFOLIO")
        print(f"{'='*60}")
        print(f"Base: live_parity + {len(base_mutations)} mutations")
        print(f"Candidate pool: {len(candidates)} features")
        print(f"Workers: {max_workers} parallel processes")
        print(f"{'='*60}\n")

    # Compute baseline in main process (data already loaded)
    base_score_obj, base_metrics = _score_config(data, base_mutations, initial_equity)

    if base_score_obj.rejected:
        print(f"WARNING: Base config rejected ({base_score_obj.reject_reason})")
        baseline_score = 0.0
    else:
        baseline_score = base_score_obj.total

    if verbose:
        print(f"Baseline: score={baseline_score:.4f}")
        if base_metrics:
            print(f"  trades={base_metrics.total_trades}, "
                  f"PF={base_metrics.profit_factor:.2f}, "
                  f"DD={base_metrics.max_drawdown_pct:.1%}, "
                  f"return={base_metrics.net_profit / initial_equity:.1%}, "
                  f"Sharpe={base_metrics.sharpe:.2f}")
        print()

    # Initialize worker pool (workers load data once, reuse for all rounds)
    if data_dir is None:
        data_dir = Path("backtests/swing/data/raw")

    print(f"Spawning {max_workers} worker processes (each loads data ~6s)...")
    t_pool = time.time()
    pool = mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), initial_equity),
    )
    print(f"Pool ready in {time.time() - t_pool:.1f}s\n")

    # Greedy loop
    current_mutations = dict(base_mutations)
    current_score = baseline_score
    remaining = list(candidates)
    kept_features: list[str] = []
    rounds: list[GreedyRound] = []

    try:
        round_num = 0
        while remaining:
            round_num += 1
            if verbose:
                print(f"Round {round_num}: Testing {len(remaining)} candidates "
                      f"(parallel across {min(max_workers, len(remaining))} workers)...")

            t0 = time.time()

            # Build merged mutation dicts for all candidates
            tasks = []
            for name, cand_mutations in remaining:
                merged = {**current_mutations, **cand_mutations}
                tasks.append(merged)

            # Evaluate all candidates in parallel
            results = pool.map(_worker_score, tasks)

            # Collect scores
            round_scores: list[tuple[str, float, float]] = []
            for (name, _), (score_val, rejected, reason) in zip(remaining, results):
                if rejected and score_val == 0.0 and reason:
                    logger.warning("Candidate %s: %s", name, reason[:200])
                delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                round_scores.append((name, score_val, delta))

            elapsed = time.time() - t0

            # Sort by score descending
            round_scores.sort(key=lambda x: x[1], reverse=True)

            if verbose:
                for name, score_val, delta in round_scores:
                    marker = " ***" if score_val == round_scores[0][1] else ""
                    print(f"  {name:35s} score={score_val:.4f}  delta={delta:+.2%}{marker}")
                print(f"  ({elapsed:.1f}s)")

            best_name, best_score, best_delta = round_scores[0]

            if best_score > current_score:
                prev_score = current_score
                best_cand_mutations = next(m for n, m in remaining if n == best_name)
                current_mutations = {**current_mutations, **best_cand_mutations}
                current_score = best_score
                kept_features.append(best_name)
                remaining = [(n, m) for n, m in remaining if n != best_name]

                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(round_scores),
                    best_name=best_name,
                    best_score=best_score,
                    best_delta_pct=best_delta,
                    kept=True,
                    all_scores=round_scores,
                ))

                if verbose:
                    print(f"  -> KEEP {best_name} (score={best_score:.4f}, "
                          f"{best_delta:+.2%} vs {prev_score:.4f})\n")
            else:
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(round_scores),
                    best_name=best_name,
                    best_score=best_score,
                    best_delta_pct=best_delta,
                    kept=False,
                    all_scores=round_scores,
                ))
                if verbose:
                    print("  -> No candidate improves score. Stopping.\n")
                break
    finally:
        pool.close()
        pool.join()

    # Final scoring in main process for detailed metrics
    _, final_metrics = _score_config(data, current_mutations, initial_equity)

    result = GreedyResult(
        base_score=baseline_score,
        final_mutations=current_mutations,
        final_score=current_score,
        kept_features=kept_features,
        rounds=rounds,
    )

    if final_metrics:
        result.final_trades = final_metrics.total_trades
        result.final_pf = final_metrics.profit_factor
        result.final_dd_pct = final_metrics.max_drawdown_pct
        result.final_return_pct = final_metrics.net_profit / initial_equity
        result.final_sharpe = final_metrics.sharpe

    if verbose:
        _print_summary(result, baseline_score)

    return result


def _print_summary(result: GreedyResult, baseline_score: float) -> None:
    """Print final summary."""
    print(f"{'='*60}")
    print("OPTIMAL PORTFOLIO CONFIG")
    print(f"{'='*60}")
    print(f"Added: {', '.join(result.kept_features) if result.kept_features else '(none)'}")
    print(f"Rounds: {len(result.rounds)} ({sum(1 for r in result.rounds if r.kept)} kept)")
    print()
    print(f"Final score:    {result.final_score:.4f}")
    print(f"Baseline score: {baseline_score:.4f}")
    if baseline_score > 0:
        print(f"Improvement:    {(result.final_score - baseline_score) / baseline_score:+.2%}")
    print()
    print(f"Trades:  {result.final_trades}")
    print(f"PF:      {result.final_pf:.2f}")
    print(f"DD:      {result.final_dd_pct:.1%}")
    print(f"Return:  {result.final_return_pct:.1%}")
    print(f"Sharpe:  {result.final_sharpe:.2f}")
    print()
    print("Final mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}")


def save_result(result: GreedyResult, output_path: Path) -> None:
    """Save greedy result to JSON."""
    data = {
        "base_score": result.base_score,
        "final_score": result.final_score,
        "improvement_pct": (
            (result.final_score - result.base_score) / result.base_score
            if result.base_score > 0 else 0.0
        ),
        "kept_features": result.kept_features,
        "final_mutations": result.final_mutations,
        "final_trades": result.final_trades,
        "final_pf": result.final_pf,
        "final_dd_pct": result.final_dd_pct,
        "final_return_pct": result.final_return_pct,
        "final_sharpe": result.final_sharpe,
        "rounds": [
            {
                "round": r.round_num,
                "candidates_tested": r.candidates_tested,
                "best_name": r.best_name,
                "best_score": r.best_score,
                "best_delta_pct": r.best_delta_pct,
                "kept": r.kept,
            }
            for r in result.rounds
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResult saved to {output_path}")


# ---------------------------------------------------------------------------
# Predefined candidate pool from auto experiment results
# ---------------------------------------------------------------------------

# Candidates ordered by individual delta (highest first).
# Strategy-specific params are mapped to unified config routing keys.
PORTFOLIO_CANDIDATES: list[tuple[str, dict]] = [
    # Breakout score threshold removal
    ("brk_no_score_threshold", {"breakout_flags.disable_score_threshold": True}),
    # Helix stale 4H detection
    ("helix_stale_4h_4", {"helix_param.STALE_4H_BARS": 4}),
    # Portfolio-level risk adjustments
    ("helix_risk_1.2pct", {"helix.unit_risk_pct": 0.012}),
    ("helix_risk_1.0pct", {"helix.unit_risk_pct": 0.010}),
    ("overlay_max_70pct", {"overlay_max_pct": 0.70}),
]
