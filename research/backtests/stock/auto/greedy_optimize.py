"""Greedy forward selection for optimal config discovery.

Algorithm:
  1. Start with a base config (e.g. full_recalibration mutations)
  2. Score the base → baseline_score
  3. For each round, test every remaining candidate merged with base
  4. Keep the best candidate if it improves score; stop when none do
  5. Output the final optimal mutations and comparison table

Usage:
    from research.backtests.stock.auto.greedy_optimize import run_greedy
    result = run_greedy(replay, "iaric", 1, base_mutations, candidates)
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.backtests.stock.auto.config_mutator import (
    mutate_alcb_config,
    mutate_iaric_config,
)
from research.backtests.stock.auto.scoring import (
    CompositeScore, IARIC_NORM, composite_score, compute_r_multiples,
    extract_metrics,
)
from research.backtests.stock.models import TradeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallel evaluation support (multiprocessing)
# ---------------------------------------------------------------------------
_mp_replay = None  # Worker-local replay engine
_mp_cfg_kwargs = None  # Worker-local config kwargs


def _init_eval_worker(data_dir_str: str, cfg_kwargs: dict) -> None:
    """Initialize a multiprocessing worker: load data once per process."""
    global _mp_replay, _mp_cfg_kwargs
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine
    _mp_replay = ResearchReplayEngine(data_dir=Path(data_dir_str))
    _mp_replay.load_all_data()
    cfg_kwargs["data_dir"] = Path(cfg_kwargs["data_dir"])
    _mp_cfg_kwargs = cfg_kwargs


def _eval_candidate_mp(args: tuple) -> tuple[str, float, str | None]:
    """Evaluate one candidate in a worker process. Returns (name, score, error)."""
    name, merged_mutations, initial_equity = args
    try:
        config = _make_config(merged_mutations, _mp_cfg_kwargs)
        trades, eq, ts = _run_engine(
            _mp_replay, _mp_cfg_kwargs["strategy"],
            _mp_cfg_kwargs["tier"], config,
        )
        metrics = extract_metrics(trades, eq, ts, initial_equity)
        r_mult = compute_r_multiples(trades)
        norm = IARIC_NORM if _mp_cfg_kwargs["strategy"] == "iaric" else None
        score = composite_score(metrics, initial_equity, r_multiples=r_mult, norm=norm)
        return (name, score.total if not score.rejected else 0.0, None)
    except Exception as exc:
        return (name, 0.0, f"{exc}")


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
    strategy: str
    tier: int
    base_mutations: dict
    base_score: float
    final_mutations: dict
    final_score: float
    kept_features: list[str]
    rounds: list[GreedyRound]
    final_trades: int = 0
    final_pf: float = 0.0
    final_dd_pct: float = 0.0
    final_return_pct: float = 0.0


def _save_checkpoint(
    checkpoint_path: Path,
    strategy: str,
    tier: int,
    base_mutations: dict,
    baseline_score: float,
    current_mutations: dict,
    current_score: float,
    kept_features: list[str],
    rounds: list[GreedyRound],
    remaining: list[tuple[str, dict]],
) -> None:
    """Save greedy loop state to a checkpoint file for resume."""
    data = {
        "strategy": strategy,
        "tier": tier,
        "base_mutations": {k: _serialize_val(v) for k, v in base_mutations.items()},
        "baseline_score": baseline_score,
        "current_mutations": {k: _serialize_val(v) for k, v in current_mutations.items()},
        "current_score": current_score,
        "kept_features": kept_features,
        "rounds": [
            {
                "round_num": r.round_num,
                "candidates_tested": r.candidates_tested,
                "best_name": r.best_name,
                "best_score": r.best_score,
                "best_delta_pct": r.best_delta_pct,
                "kept": r.kept,
                "all_scores": r.all_scores,
            }
            for r in rounds
        ],
        "remaining": [
            (name, {k: _serialize_val(v) for k, v in muts.items()})
            for name, muts in remaining
        ],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        tmp.replace(checkpoint_path)
    except PermissionError:
        # Windows: target may be locked by antivirus/indexer; fall back to copy
        import shutil
        shutil.copy2(tmp, checkpoint_path)
        tmp.unlink(missing_ok=True)


def _serialize_val(v):
    """Convert tuples to lists for JSON serialization."""
    if isinstance(v, tuple):
        return [_serialize_val(x) for x in v]
    if isinstance(v, list):
        return [_serialize_val(x) for x in v]
    return v


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint if it exists, return parsed data or None."""
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        logger.warning("Failed to load checkpoint %s: %s", checkpoint_path, exc)
        return None


def run_greedy(
    replay,
    strategy: str,
    tier: int,
    base_mutations: dict,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    data_dir: str = "research/backtests/stock/data/raw",
    verbose: bool = True,
    max_workers: int = 1,
    checkpoint_path: Path | None = None,
    prune_delta: float = -0.10,
    prune_no_effect: bool = True,
    min_delta: float = 0.001,
) -> GreedyResult:
    """Run greedy forward selection to find optimal config.

    Args:
        replay: ResearchReplayEngine with data already loaded
        strategy: "iaric" or "alcb"
        tier: 1 or 2
        base_mutations: Starting mutations (e.g. full_recalibration)
        candidates: List of (name, mutations_dict) to test
        initial_equity: Starting equity
        start_date: Backtest start date
        end_date: Backtest end date
        data_dir: Path to bar data
        verbose: Print progress
        checkpoint_path: If provided, save/resume checkpoints at this path
        prune_delta: After round 1, drop candidates with delta worse than this
            threshold (default -0.10 = -10%). Set to None to disable pruning.
        prune_no_effect: After round 1, drop candidates with exactly 0% delta
            (no-ops that set parameters to their current values).
        min_delta: Minimum relative improvement to keep a candidate (default
            0.001 = 0.1%). Stops chasing noise in late rounds.

    Returns:
        GreedyResult with optimal mutations and round-by-round history
    """
    cfg_kwargs = dict(
        strategy=strategy, tier=tier, initial_equity=initial_equity,
        start_date=start_date, end_date=end_date, data_dir=Path(data_dir),
    )

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"GREEDY FORWARD SELECTION: {strategy.upper()} T{tier}")
        print(f"{'='*60}")
        print(f"Base mutations: {len(base_mutations)} params")
        print(f"Candidate pool: {len(candidates)} features")
        if max_workers > 1:
            print(f"Parallel workers: {max_workers}")
        print(f"{'='*60}\n", flush=True)

    base_score_obj, base_metrics = _score_config(
        replay, base_mutations, initial_equity, cfg_kwargs,
    )

    if base_score_obj.rejected:
        print(f"WARNING: Base config rejected ({base_score_obj.reject_reason})")
        print("Proceeding anyway — greedy may overcome base limitations.\n")
        baseline_score = 0.0
    else:
        baseline_score = base_score_obj.total

    if verbose:
        print(f"Baseline: score={baseline_score:.4f}", flush=True)
        if base_metrics:
            print(f"  trades={base_metrics.total_trades}, "
                  f"PF={base_metrics.profit_factor:.2f}, "
                  f"DD={base_metrics.max_drawdown_pct:.1%}, "
                  f"return={base_metrics.net_profit / initial_equity:.1%}")
        print(flush=True)

    # --- Resume from checkpoint if available ---
    resumed = False
    if checkpoint_path is not None:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt is not None:
            # Build a lookup of candidate names still remaining in checkpoint
            ckpt_remaining_names = {name for name, _ in ckpt["remaining"]}
            # Rebuild remaining from original candidates to preserve mutation dicts
            remaining_lookup = {name: muts for name, muts in candidates}
            _resumed_remaining = [
                (name, remaining_lookup[name])
                for name in ckpt_remaining_names
                if name in remaining_lookup
            ]
            if _resumed_remaining:
                current_mutations = ckpt["current_mutations"]
                current_score = ckpt["current_score"]
                kept_features = ckpt["kept_features"]
                rounds = [
                    GreedyRound(
                        round_num=r["round_num"],
                        candidates_tested=r["candidates_tested"],
                        best_name=r["best_name"],
                        best_score=r["best_score"],
                        best_delta_pct=r["best_delta_pct"],
                        kept=r["kept"],
                        all_scores=[tuple(s) for s in r["all_scores"]],
                    )
                    for r in ckpt["rounds"]
                ]
                remaining = _resumed_remaining
                baseline_score = ckpt["baseline_score"]
                resumed = True
                if verbose:
                    print(f"*** RESUMED from checkpoint: "
                          f"{len(rounds)} rounds done, "
                          f"{len(kept_features)} features kept, "
                          f"{len(remaining)} candidates remaining, "
                          f"score={current_score:.4f} ***\n", flush=True)

    # Create process pool for parallel evaluation
    pool = None
    if max_workers > 1:
        import multiprocessing as mp
        _cfg_ser = {**cfg_kwargs, "data_dir": str(cfg_kwargs["data_dir"])}
        if verbose:
            print(f"Initializing {max_workers} worker processes...", flush=True)
        pool = mp.Pool(
            processes=max_workers,
            initializer=_init_eval_worker,
            initargs=(str(cfg_kwargs["data_dir"]), _cfg_ser),
        )
        if verbose:
            print(f"  Workers ready.\n", flush=True)

    # Greedy loop — initialize fresh or keep resumed state
    if not resumed:
        current_mutations = dict(base_mutations)
        current_score = baseline_score
        remaining = list(candidates)
        kept_features: list[str] = []
        rounds: list[GreedyRound] = []

    round_num = rounds[-1].round_num if rounds else 0
    try:
        while remaining:
            round_num += 1
            if verbose:
                print(f"Round {round_num}: Testing {len(remaining)} candidates...", flush=True)

            round_scores: list[tuple[str, float, float]] = []
            t0 = time.time()

            if pool is not None:
                # Parallel evaluation
                tasks = [
                    (name, {**current_mutations, **cm}, initial_equity)
                    for name, cm in remaining
                ]
                results = pool.map(_eval_candidate_mp, tasks)
                for name, score_val, err in results:
                    if err:
                        logger.warning("Candidate %s crashed: %s", name, err)
                    delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                    round_scores.append((name, score_val, delta))
            else:
                for ci, (name, cand_mutations) in enumerate(remaining, 1):
                    merged = {**current_mutations, **cand_mutations}
                    t1 = time.time()
                    try:
                        score_obj, _ = _score_config(
                            replay, merged, initial_equity, cfg_kwargs,
                        )
                        score_val = score_obj.total if not score_obj.rejected else 0.0
                    except Exception:
                        logger.warning("Candidate %s crashed:\n%s", name, traceback.format_exc())
                        score_val = 0.0
                    delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                    round_scores.append((name, score_val, delta))
                    if verbose:
                        print(f"  [{ci}/{len(remaining)}] {name:30s} score={score_val:.4f} "
                              f"delta={delta:+.2%} ({time.time()-t1:.0f}s)", flush=True)

            elapsed = time.time() - t0

            # Sort by score descending
            round_scores.sort(key=lambda x: x[1], reverse=True)

            if verbose:
                for name, score_val, delta in round_scores:
                    marker = " ***" if score_val == round_scores[0][1] else ""
                    print(f"  {name:30s} score={score_val:.4f}  delta={delta:+.2%}{marker}")
                print(f"  ({elapsed:.1f}s)", flush=True)

            best_name, best_score, best_delta = round_scores[0]

            if best_score > current_score and best_delta >= min_delta:
                prev_score = current_score
                # Merge best candidate into current
                best_cand_mutations = next(m for n, m in remaining if n == best_name)
                current_mutations = {**current_mutations, **best_cand_mutations}
                current_score = best_score
                kept_features.append(best_name)
                remaining = [(n, m) for n, m in remaining if n != best_name]

                # Prune after round 1
                if round_num >= 1:
                    scores_by_name = {n: d for n, _s, d in round_scores}
                    before_prune = len(remaining)

                    def _keep(n: str) -> bool:
                        d = scores_by_name.get(n, 0.0)
                        if prune_delta is not None and d < prune_delta:
                            return False
                        if prune_no_effect and abs(d) < 1e-6:
                            return False
                        return True

                    remaining = [(n, m) for n, m in remaining if _keep(n)]
                    pruned = before_prune - len(remaining)
                    if pruned > 0 and verbose:
                        print(f"  [pruned {pruned} candidates (delta < "
                              f"{prune_delta:+.0%} or no effect), "
                              f"{len(remaining)} remain]", flush=True)

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
                    print(f"  → KEEP {best_name} (score={best_score:.4f}, "
                          f"{best_delta:+.2%} vs {prev_score:.4f})\n", flush=True)

                # Checkpoint after each kept round
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path, strategy, tier, base_mutations,
                        baseline_score, current_mutations, current_score,
                        kept_features, rounds, remaining,
                    )
                    if verbose:
                        print(f"  [checkpoint saved: round {round_num}]\n", flush=True)
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
                    if best_score > current_score:
                        print(f"  → Best candidate {best_name} ({best_delta:+.2%}) "
                              f"below min_delta ({min_delta:.1%}). Stopping.\n", flush=True)
                    else:
                        print(f"  → No candidate improves score. Stopping.\n", flush=True)
                # Final checkpoint (marks convergence)
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path, strategy, tier, base_mutations,
                        baseline_score, current_mutations, current_score,
                        kept_features, rounds, [],
                    )
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Final scoring for summary metrics
    _, final_metrics = _score_config(
        replay, current_mutations, initial_equity, cfg_kwargs,
    )

    result = GreedyResult(
        strategy=strategy,
        tier=tier,
        base_mutations=base_mutations,
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

    if verbose:
        _print_summary(result, baseline_score)

    return result


def _score_config(
    replay,
    mutations: dict,
    initial_equity: float,
    cfg_kwargs: dict,
) -> tuple[CompositeScore, object | None]:
    """Build config, run engine, return (score, metrics)."""
    config = _make_config(mutations, cfg_kwargs)
    trades, eq, ts = _run_engine(replay, cfg_kwargs["strategy"], cfg_kwargs["tier"], config)
    metrics = extract_metrics(trades, eq, ts, initial_equity)
    r_mult = compute_r_multiples(trades)
    norm = IARIC_NORM if cfg_kwargs["strategy"] == "iaric" else None
    score = composite_score(metrics, initial_equity, r_multiples=r_mult, norm=norm)
    return score, metrics


def _make_config(mutations: dict, cfg_kwargs: dict):
    """Build a config with mutations applied."""
    strategy = cfg_kwargs["strategy"]
    if strategy == "iaric":
        from research.backtests.stock.config_iaric import IARICBacktestConfig
        base = IARICBacktestConfig(
            start_date=cfg_kwargs["start_date"],
            end_date=cfg_kwargs["end_date"],
            initial_equity=cfg_kwargs["initial_equity"],
            tier=cfg_kwargs["tier"],
            data_dir=cfg_kwargs["data_dir"],
        )
        return mutate_iaric_config(base, mutations)
    elif strategy == "alcb":
        from research.backtests.stock.config_alcb import ALCBBacktestConfig
        base = ALCBBacktestConfig(
            start_date=cfg_kwargs["start_date"],
            end_date=cfg_kwargs["end_date"],
            initial_equity=cfg_kwargs["initial_equity"],
            tier=cfg_kwargs["tier"],
            data_dir=cfg_kwargs["data_dir"],
        )
        return mutate_alcb_config(base, mutations)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _run_engine(
    replay,
    strategy: str,
    tier: int,
    config,
) -> tuple[list[TradeRecord], np.ndarray, np.ndarray]:
    """Run the correct engine and return (trades, equity_curve, timestamps)."""
    if strategy == "iaric":
        if tier == 1:
            from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine
            engine = IARICDailyEngine(config, replay)
        else:
            from research.backtests.stock.engine.iaric_intraday_engine_v2 import IARICIntradayEngineV2
            engine = IARICIntradayEngineV2(config, replay)
    elif strategy == "alcb":
        if tier == 1:
            from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine
            engine = ALCBDailyEngine(config, replay)
        else:
            from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
            engine = ALCBIntradayEngine(config, replay)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = engine.run()
    return result.trades, result.equity_curve, result.timestamps


def _print_summary(result: GreedyResult, baseline_score: float) -> None:
    """Print final summary."""
    print(f"{'='*60}")
    print(f"OPTIMAL CONFIG")
    print(f"{'='*60}")
    print(f"Strategy: {result.strategy.upper()} T{result.tier}")
    print(f"Base: full_recalibration")
    print(f"Added: {', '.join(result.kept_features) if result.kept_features else '(none)'}")
    print(f"Rounds: {len(result.rounds)} ({sum(1 for r in result.rounds if r.kept)} kept)")
    print()
    print(f"Final score:    {result.final_score:.4f}")
    print(f"Baseline score: {baseline_score:.4f}")
    if baseline_score > 0:
        print(f"Improvement:    {(result.final_score - baseline_score) / baseline_score:+.2%}")
    print()
    print(f"Trades: {result.final_trades}")
    print(f"PF:     {result.final_pf:.2f}")
    print(f"DD:     {result.final_dd_pct:.1%}")
    print(f"Return: {result.final_return_pct:.1%}")
    print()
    print("Final mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}")


def save_result(result: GreedyResult, output_path: Path) -> None:
    """Save greedy result to JSON."""
    data = {
        "strategy": result.strategy,
        "tier": result.tier,
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
    output_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResult saved to {output_path}")


# ---------------------------------------------------------------------------
# Predefined candidate pools
# ---------------------------------------------------------------------------

IARIC_T1_BASE_MUTATIONS = {
    "param_overrides.carry_only_entry": True,
    "param_overrides.use_close_stop": True,
    "param_overrides.base_risk_fraction": 0.005,
    "param_overrides.max_carry_days": 5,
    "param_overrides.min_carry_r": 0.25,
    "param_overrides.min_conviction_multiplier": 0.25,
    "max_per_sector": 2,
}

IARIC_T2_BASE_MUTATIONS = {
    # Start from T1 optimal params (carry-focused, selective)
    # Note: use_close_stop omitted — T2v2 uses adaptive trailing stops instead
    "param_overrides.carry_only_entry": True,
    "param_overrides.base_risk_fraction": 0.005,
    "param_overrides.min_carry_r": 0.50,
    "param_overrides.min_conviction_multiplier": 0.25,
    "param_overrides.flow_reversal_lookback": 1,
    "param_overrides.regime_b_carry_mult": 0.6,
    "max_per_sector": 2,
}

IARIC_T2_CANDIDATES: list[tuple[str, dict]] = [
    # Adaptive stop tuning
    ("initial_atr_1.0", {"param_overrides.t2_initial_atr_mult": 1.0}),
    ("initial_atr_2.0", {"param_overrides.t2_initial_atr_mult": 2.0}),
    ("breakeven_r_0.3", {"param_overrides.t2_breakeven_r": 0.3}),
    ("breakeven_r_0.75", {"param_overrides.t2_breakeven_r": 0.75}),
    ("profit_trail_0.75", {"param_overrides.t2_profit_trail_atr": 0.75}),
    ("profit_trail_1.5", {"param_overrides.t2_profit_trail_atr": 1.5}),
    # Entry timing
    ("fallback_bar_4", {"param_overrides.t2_fallback_entry_bar": 4}),   # 9:50
    ("fallback_bar_12", {"param_overrides.t2_fallback_entry_bar": 12}),  # 10:30
    ("no_pm_reentry", {"param_overrides.t2_pm_reentry": False}),
    # Carry scoring
    ("carry_thresh_55", {"param_overrides.t2_carry_threshold": 55.0}),
    ("carry_thresh_75", {"param_overrides.t2_carry_threshold": 75.0}),
    # Day-of-week
    ("tue_skip", {"param_overrides.t2_tuesday_mult": 0.0}),
    ("fri_1.5", {"param_overrides.t2_friday_mult": 1.5}),
    # Risk & sizing
    ("risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
    ("risk_010", {"param_overrides.base_risk_fraction": 0.01}),
    # Staleness (defaults: 5h, min_r=0.1 — calibrated for daily-ATR stops)
    ("stale_3h", {"param_overrides.t2_staleness_hours": 3.0}),
    ("stale_4h", {"param_overrides.t2_staleness_hours": 4.0}),
    ("no_stale", {"param_overrides.t2_staleness_hours": 99.0}),
    ("stale_minr_0", {"param_overrides.t2_staleness_min_r": 0.0}),
    ("stale_minr_03", {"param_overrides.t2_staleness_min_r": 0.3}),
]

# ---------------------------------------------------------------------------
# ALCB T2 — Momentum Continuation
# ---------------------------------------------------------------------------

ALCB_T2_BASE_MUTATIONS: dict = {}  # Use StrategySettings defaults

ALCB_T2_CANDIDATES: list[tuple[str, dict]] = [
    # --- Flow reversal (P1: 68% of trades at 14.9% WR, destroying alpha) ---
    ("no_flow_reversal", {"ablation.use_flow_reversal_exit": False}),
    ("fr_grace_12", {"param_overrides.flow_reversal_min_hold_bars": 12}),
    ("fr_grace_24", {"param_overrides.flow_reversal_min_hold_bars": 24}),
    ("fr_below_entry", {"param_overrides.flow_reversal_require_below_entry": True}),
    ("fr_grace12_below", {
        "param_overrides.flow_reversal_min_hold_bars": 12,
        "param_overrides.flow_reversal_require_below_entry": True,
    }),
    # --- Portfolio (P2: shadow shows 59% WR, +0.16R avg blocked by max_pos) ---
    ("max_pos_8", {"param_overrides.max_positions": 8}),
    ("max_pos_10", {"param_overrides.max_positions": 10}),
    ("bigger_portfolio", {
        "param_overrides.max_positions": 8,
        "param_overrides.heat_cap_r": 10.0,
        "param_overrides.max_positions_per_sector": 3,
    }),
    # --- Ablation (P3: test borderline gates) ---
    ("no_avwap_filter", {"ablation.use_avwap_filter": False}),
    ("no_partials", {"ablation.use_partial_takes": False}),
    ("no_carry", {"ablation.use_carry_logic": False}),
    # --- Risk (P4) ---
    ("risk_010", {"param_overrides.base_risk_fraction": 0.010}),
    ("risk_020", {"param_overrides.base_risk_fraction": 0.020}),
    # --- Stop (P5) ---
    ("stop_0.75atr", {"param_overrides.stop_atr_multiple": 0.75}),
    ("stop_1.5atr", {"param_overrides.stop_atr_multiple": 1.5}),
    # --- Entry (P6) ---
    ("score_min_2", {"param_overrides.momentum_score_min": 2}),
    ("cpr_0.5", {"param_overrides.cpr_threshold": 0.5}),
    # --- Carry (P7) ---
    ("carry_days_5", {"param_overrides.max_carry_days": 5}),
    ("carry_min_r_025", {"param_overrides.carry_min_r": 0.25}),
    # --- Opening range (P8) ---
    ("or_3bars", {"param_overrides.opening_range_bars": 3}),
    ("or_12bars", {"param_overrides.opening_range_bars": 12}),
    # --- Regime (P9: Tier B losing money) ---
    ("regime_b_05", {"param_overrides.regime_mult_b": 0.5}),
]

# ---------------------------------------------------------------------------
# ALCB T2 Phase 6 — Diagnostic-Driven Optimization
# ---------------------------------------------------------------------------

ALCB_T2_P6_BASE_MUTATIONS: dict = {}  # Start from defaults

ALCB_T2_P6_CANDIDATES: list[tuple[str, dict]] = [
    # === Inherit all 23 existing Phase 5 candidates ===
    *ALCB_T2_CANDIDATES,

    # === A. RVOL Max Cap (RVOL 3.0+ = -65R, PF=0.78) ===
    ("rvol_max_3", {"param_overrides.rvol_max": 3.0}),
    ("rvol_max_4", {"param_overrides.rvol_max": 4.0}),
    ("rvol_max_5", {"param_overrides.rvol_max": 5.0}),

    # === B. FR MFE Grace (934 trades had MFE>0.3R before FR killed them) ===
    ("fr_mfe_03", {"param_overrides.fr_mfe_grace_r": 0.3}),
    ("fr_mfe_05", {"param_overrides.fr_mfe_grace_r": 0.5}),
    ("fr_mfe_10", {"param_overrides.fr_mfe_grace_r": 1.0}),

    # === C. Thursday Sizing (Thursday = -85R, PF=0.63) ===
    ("thursday_half", {"param_overrides.thursday_sizing_mult": 0.5}),
    ("thursday_skip", {"param_overrides.thursday_sizing_mult": 0.0}),

    # === D. FR Trailing Stop (convert destructive FR into profit-protector) ===
    ("fr_trail_05", {"param_overrides.fr_trailing_activate_r": 0.5, "param_overrides.fr_trailing_distance_r": 0.5}),
    ("fr_trail_075", {"param_overrides.fr_trailing_activate_r": 0.75, "param_overrides.fr_trailing_distance_r": 0.5}),
    ("fr_trail_10_tight", {"param_overrides.fr_trailing_activate_r": 1.0, "param_overrides.fr_trailing_distance_r": 0.3}),

    # === E. Breakeven Stop After MFE (CLOSE_STOP = -365R, protect profits) ===
    ("be_after_03r", {"param_overrides.close_stop_be_after_r": 0.3}),
    ("be_after_05r", {"param_overrides.close_stop_be_after_r": 0.5}),

    # === F. Extended FR Grace Periods (early FR <30min = -378R) ===
    ("fr_grace_36", {"param_overrides.flow_reversal_min_hold_bars": 36}),
    ("fr_grace_48", {"param_overrides.flow_reversal_min_hold_bars": 48}),

    # === G. Momentum Score Raise (Score 4 = -61R, 27% WR) ===
    ("score_min_4", {"param_overrides.momentum_score_min": 4}),
    ("score_min_5", {"param_overrides.momentum_score_min": 5}),

    # === H. Earlier Partial Takes (lock profit sooner, amplify hold-to-EOD) ===
    ("partial_075r", {"param_overrides.partial_r_trigger": 0.75}),
    ("partial_100r", {"param_overrides.partial_r_trigger": 1.0}),

    # === I. Carry Amplification (EOD flatten = +726R alpha source) ===
    ("carry_r_0", {"param_overrides.carry_min_r": 0.0}),
    ("carry_cpr_04", {"param_overrides.carry_min_cpr": 0.4}),

    # === J. Regime B Elimination (Tier B = -74R, PF=0.69) ===
    ("regime_b_0", {"param_overrides.regime_mult_b": 0.0}),

    # === K. Wider Stop (CLOSE_STOP avg hold 0.7h suggests too tight) ===
    ("stop_2atr", {"param_overrides.stop_atr_multiple": 2.0}),

    # === L. RVOL Min Raise (sweet spot is 2.0-3.0) ===
    ("rvol_min_175", {"param_overrides.rvol_threshold": 1.75}),
    ("rvol_min_200", {"param_overrides.rvol_threshold": 2.0}),

    # === M. Compound: FR grace + below_entry + MFE grace ===
    ("fr_compound", {
        "param_overrides.flow_reversal_min_hold_bars": 24,
        "param_overrides.flow_reversal_require_below_entry": True,
        "param_overrides.fr_mfe_grace_r": 0.5,
    }),
]

# ---------------------------------------------------------------------------
# ALCB T2 Phase 7 — Quality & Robustness
# ---------------------------------------------------------------------------

from datetime import time as dt_time

# Base = P6 optimal config (from greedy_optimal_alcb_t2.json)
ALCB_T2_P7_BASE_MUTATIONS: dict = {
    "param_overrides.rvol_max": 5.0,
    "param_overrides.fr_trailing_activate_r": 0.5,
    "param_overrides.fr_trailing_distance_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 24,
    "param_overrides.regime_mult_b": 0.0,
    "param_overrides.base_risk_fraction": 0.02,
    "ablation.use_partial_takes": False,
    "ablation.use_avwap_filter": False,
    "param_overrides.opening_range_bars": 3,
    "param_overrides.max_positions": 8,
    "param_overrides.heat_cap_r": 10.0,
    "param_overrides.max_positions_per_sector": 3,
    "param_overrides.momentum_score_min": 2,
    "param_overrides.stop_atr_multiple": 2.0,
    "param_overrides.carry_min_r": 0.0,
    "param_overrides.carry_min_cpr": 0.4,
}

ALCB_T2_P7_CANDIDATES: list[tuple[str, dict]] = [
    # === A. Daily ATR Dollar Filter (filter weak/low-vol symbols) ===
    ("atr_usd_3", {"param_overrides.min_daily_atr_usd": 3.0}),
    ("atr_usd_5", {"param_overrides.min_daily_atr_usd": 5.0}),
    ("atr_usd_8", {"param_overrides.min_daily_atr_usd": 8.0}),

    # === B. Selection Score Gate (prefer high-quality daily setups) ===
    ("sel_score_6", {"param_overrides.min_selection_score": 6}),
    ("sel_score_8", {"param_overrides.min_selection_score": 8}),

    # === C. Relative Strength Filter (prefer RS leaders) ===
    ("rs_pct_50", {"param_overrides.min_rs_percentile": 0.50}),
    ("rs_pct_70", {"param_overrides.min_rs_percentile": 0.70}),

    # === D. Entry Window Narrowing (10:30+ entries weaker) ===
    ("entry_end_1100", {"param_overrides.entry_window_end": dt_time(11, 0)}),
    ("entry_end_1130", {"param_overrides.entry_window_end": dt_time(11, 30)}),

    # === E. Late Entry Score Escalation (require stronger signal after cutoff) ===
    ("late_score_4", {"param_overrides.late_entry_score_min": 4}),
    ("late_score_5", {"param_overrides.late_entry_score_min": 5}),
    ("late_1030_score4", {
        "param_overrides.late_entry_cutoff": dt_time(10, 30),
        "param_overrides.late_entry_score_min": 4,
    }),

    # === F. FR Refinements (address 2-3h dead zone) ===
    ("fr_grace_36", {"param_overrides.flow_reversal_min_hold_bars": 36}),
    ("fr_below_entry", {"param_overrides.flow_reversal_require_below_entry": True}),
    ("fr_grace36_below", {
        "param_overrides.flow_reversal_min_hold_bars": 36,
        "param_overrides.flow_reversal_require_below_entry": True,
    }),
    ("fr_mfe_grace_03", {"param_overrides.fr_mfe_grace_r": 0.3}),

    # === G. Risk Reduction (for live robustness) ===
    ("risk_015", {"param_overrides.base_risk_fraction": 0.015}),
    ("risk_010", {"param_overrides.base_risk_fraction": 0.010}),

    # === H. Remove dead carry mutations (verify no impact) ===
    ("no_carry_tweaks", {
        "param_overrides.carry_min_r": 0.5,
        "param_overrides.carry_min_cpr": 0.6,
    }),
]

# ---------------------------------------------------------------------------
# ALCB T2 Phase 8 — Exit & Signal Diagnostics
# ---------------------------------------------------------------------------

# Base = P7 optimal config (from optimal_config_alcb_p7.json)
ALCB_T2_P8_BASE_MUTATIONS: dict = {
    # P6 foundation
    "param_overrides.rvol_max": 5.0,
    "param_overrides.fr_trailing_activate_r": 0.5,
    "param_overrides.fr_trailing_distance_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 24,
    "param_overrides.regime_mult_b": 0.0,
    "param_overrides.base_risk_fraction": 0.02,
    "ablation.use_partial_takes": False,
    "ablation.use_avwap_filter": False,
    "param_overrides.opening_range_bars": 3,
    "param_overrides.max_positions": 8,
    "param_overrides.heat_cap_r": 10.0,
    "param_overrides.max_positions_per_sector": 3,
    "param_overrides.momentum_score_min": 2,
    "param_overrides.stop_atr_multiple": 2.0,
    "param_overrides.carry_min_r": 0.0,
    "param_overrides.carry_min_cpr": 0.4,
    # P7 additions (kept by greedy)
    "param_overrides.entry_window_end": dt_time(11, 0),
    "param_overrides.late_entry_score_min": 5,
}

ALCB_T2_P8_CANDIDATES: list[tuple[str, dict]] = [
    # =======================================================================
    # A. EXIT MECHANISM — Flow Reversal (highest priority, ~78.52R drag)
    # =======================================================================

    # A1. FR Ablation: disable FR entirely, rely on CLOSE_STOP + trailing
    ("no_flow_reversal", {"ablation.use_flow_reversal_exit": False}),

    # A2. FR Conditional: only trigger FR when bar CPR is weak
    ("fr_cpr_030", {"param_overrides.fr_cpr_threshold": 0.30}),
    ("fr_cpr_040", {"param_overrides.fr_cpr_threshold": 0.40}),

    # A3. FR Max Hold: disable FR after 48 bars (let trailing stop take over)
    ("fr_max_hold_48", {"param_overrides.fr_max_hold_bars": 48}),
    ("fr_max_hold_72", {"param_overrides.fr_max_hold_bars": 72}),

    # A4. FR MFE Grace: skip FR if position ever hit 0.3R+ (protect profits)
    ("fr_mfe_grace_03", {"param_overrides.fr_mfe_grace_r": 0.3}),

    # A5. Compound FR gating: CPR + MFE grace together
    ("fr_compound_cpr_mfe", {
        "param_overrides.fr_cpr_threshold": 0.30,
        "param_overrides.fr_mfe_grace_r": 0.3,
    }),

    # =======================================================================
    # B. EXIT MECHANISM — Partial Takes & MFE Capture
    # =======================================================================

    # B1. Re-enable partials at various R triggers (36.6% MFE capture → improve)
    ("partial_050r_025f", {
        "ablation.use_partial_takes": True,
        "param_overrides.partial_r_trigger": 0.5,
        "param_overrides.partial_fraction": 0.25,
    }),
    ("partial_075r_033f", {
        "ablation.use_partial_takes": True,
        "param_overrides.partial_r_trigger": 0.75,
        "param_overrides.partial_fraction": 0.33,
    }),
    ("partial_100r_050f", {
        "ablation.use_partial_takes": True,
        "param_overrides.partial_r_trigger": 1.0,
        "param_overrides.partial_fraction": 0.50,
    }),
    ("partial_125r_033f", {
        "ablation.use_partial_takes": True,
        "param_overrides.partial_r_trigger": 1.25,
        "param_overrides.partial_fraction": 0.33,
    }),

    # B2. Breakeven stop acceleration (protect capital earlier)
    ("be_stop_050r", {"param_overrides.close_stop_be_after_r": 0.5}),
    ("be_stop_075r", {"param_overrides.close_stop_be_after_r": 0.75}),
    ("be_stop_100r", {"param_overrides.close_stop_be_after_r": 1.0}),

    # B3. Runner stop tightness (reduce 0.602R MFE giveback)
    ("runner_atr_075", {"param_overrides.runner_structure_atr_base": 0.75}),
    ("runner_ratchet_15r", {"param_overrides.runner_profit_ratchet_r_base": 1.5}),

    # =======================================================================
    # C. EXIT MECHANISM — Time-Based Quick Exit (cut -31R from <6 bar holds)
    # =======================================================================

    ("quick_exit_6bar_02r", {
        "ablation.use_time_based_quick_exit": True,
        "param_overrides.quick_exit_max_bars": 6,
        "param_overrides.quick_exit_min_r": 0.2,
    }),
    ("quick_exit_12bar_02r", {
        "ablation.use_time_based_quick_exit": True,
        "param_overrides.quick_exit_max_bars": 12,
        "param_overrides.quick_exit_min_r": 0.2,
    }),
    ("quick_exit_6bar_00r", {
        "ablation.use_time_based_quick_exit": True,
        "param_overrides.quick_exit_max_bars": 6,
        "param_overrides.quick_exit_min_r": 0.0,
    }),

    # =======================================================================
    # D. SIGNAL — COMBINED_BREAKOUT Quality Gate (10% WR gap vs OR_BREAKOUT)
    # =======================================================================

    ("combined_score_5", {
        "ablation.use_combined_quality_gate": True,
        "param_overrides.combined_breakout_score_min": 5,
    }),
    ("combined_score_6", {
        "ablation.use_combined_quality_gate": True,
        "param_overrides.combined_breakout_score_min": 6,
    }),
    ("combined_rvol_25", {
        "ablation.use_combined_quality_gate": True,
        "param_overrides.combined_breakout_min_rvol": 2.5,
    }),
    ("combined_score5_rvol20", {
        "ablation.use_combined_quality_gate": True,
        "param_overrides.combined_breakout_score_min": 5,
        "param_overrides.combined_breakout_min_rvol": 2.0,
    }),

    # =======================================================================
    # E. SIGNAL — Entry Filters (AVWAP, OR width, breakout distance)
    # =======================================================================

    # E1. AVWAP distance cap (>1% above AVWAP = only +2.57R from 83 trades)
    ("avwap_cap_10pct", {
        "ablation.use_avwap_distance_cap": True,
        "param_overrides.avwap_distance_cap_pct": 0.010,
    }),
    ("avwap_cap_15pct", {
        "ablation.use_avwap_distance_cap": True,
        "param_overrides.avwap_distance_cap_pct": 0.015,
    }),

    # E2. OR width minimum (tight OR <0.1% underperforms)
    ("or_width_min_010", {
        "ablation.use_or_width_min": True,
        "param_overrides.or_width_min_pct": 0.001,
    }),

    # E3. Breakout distance cap (far entries weaker)
    ("breakout_dist_cap_15r", {
        "ablation.use_breakout_distance_cap": True,
        "param_overrides.breakout_distance_cap_r": 1.5,
    }),
    ("breakout_dist_cap_10r", {
        "ablation.use_breakout_distance_cap": True,
        "param_overrides.breakout_distance_cap_r": 1.0,
    }),

    # =======================================================================
    # F. ENTRY — PDH Elimination & RVOL Tiering
    # =======================================================================

    # F1. PDH ablation (only +7.46R total, median R negative)
    ("no_pdh_breakout", {"ablation.use_prior_day_high_breakout": False}),

    # F2. RVOL minimum raise (concentrate on better setups)
    ("rvol_min_175", {"param_overrides.rvol_threshold": 1.75}),
    ("rvol_min_200", {"param_overrides.rvol_threshold": 2.0}),

    # =======================================================================
    # G. PORTFOLIO — Sector-Weighted Sizing
    # =======================================================================

    # Sector-weighted sizing (size up in strong sectors, down in weak)
    ("sector_sz_cons_disc_11", {"param_overrides.sector_mult_consumer_disc": 1.1}),
    ("sector_sz_fin_08", {"param_overrides.sector_mult_financials": 0.8}),
    ("sector_sz_comm_08", {"param_overrides.sector_mult_communication": 0.8}),
    ("sector_sz_ind_05", {"param_overrides.sector_mult_industrials": 0.5}),
    ("sector_sz_weighted", {
        "param_overrides.sector_mult_consumer_disc": 1.1,
        "param_overrides.sector_mult_financials": 0.8,
        "param_overrides.sector_mult_communication": 0.8,
        "param_overrides.sector_mult_industrials": 0.5,
    }),

    # =======================================================================
    # H. SCORING — Momentum Score Threshold Experiments
    # =======================================================================

    # Score 5 is best bucket (67% WR, +0.291R), higher scores worse
    ("score_min_4", {"param_overrides.momentum_score_min": 4}),
    ("score_min_5", {"param_overrides.momentum_score_min": 5}),

    # Tighter trailing stop
    ("fr_trail_tighter", {
        "param_overrides.fr_trailing_activate_r": 0.3,
        "param_overrides.fr_trailing_distance_r": 0.3,
    }),
]


# ---------------------------------------------------------------------------
# ALCB T2 Phase 9 — Quick Exit, MFE Trailing, OR Quality Gate
# ---------------------------------------------------------------------------

# Base = P8 optimal minus sector_mult_consumer_disc (overfitted)
ALCB_T2_P9_BASE_MUTATIONS: dict = {
    # P6 foundation
    "param_overrides.rvol_max": 5.0,
    "param_overrides.fr_trailing_activate_r": 0.5,
    "param_overrides.fr_trailing_distance_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 24,
    "param_overrides.regime_mult_b": 0.0,
    "param_overrides.base_risk_fraction": 0.02,
    "ablation.use_partial_takes": False,
    "ablation.use_avwap_filter": False,
    "param_overrides.opening_range_bars": 3,
    "param_overrides.max_positions": 8,
    "param_overrides.heat_cap_r": 10.0,
    "param_overrides.max_positions_per_sector": 3,
    "param_overrides.momentum_score_min": 2,
    "param_overrides.stop_atr_multiple": 2.0,
    "param_overrides.carry_min_r": 0.0,
    "param_overrides.carry_min_cpr": 0.4,
    # P7 additions
    "param_overrides.entry_window_end": dt_time(11, 0),
    "param_overrides.late_entry_score_min": 5,
    # P8 additions (greedy-kept, minus sector_mult_consumer_disc)
    "ablation.use_time_based_quick_exit": True,
    "param_overrides.quick_exit_max_bars": 6,
    "param_overrides.quick_exit_min_r": 0.2,
    "ablation.use_avwap_distance_cap": True,
    "param_overrides.avwap_distance_cap_pct": 0.01,
}

ALCB_T2_P9_CANDIDATES: list[tuple[str, dict]] = [
    # =======================================================================
    # A. TWO-STAGE QUICK EXIT (8 candidates)
    #    P8 QUICK_EXIT destroys -113.65R (626 trades, 35.8% WR, PF 0.10)
    # =======================================================================

    # A1. Disable quick exit entirely (baseline comparison)
    ("qe_disabled", {"ablation.use_time_based_quick_exit": False}),

    # A1b. Keep quick exit but block slot recycling (loss-limiter only)
    ("qe_no_recycle", {"ablation.use_qe_no_recycle": True}),

    # A2. Lengthen stage 2 window
    ("qe_8bar_02r", {"param_overrides.quick_exit_max_bars": 8}),
    ("qe_10bar_02r", {"param_overrides.quick_exit_max_bars": 10}),

    # A3. Adjust stage 2 threshold
    ("qe_6bar_00r", {"param_overrides.quick_exit_min_r": 0.0}),
    ("qe_6bar_neg02r", {"param_overrides.quick_exit_min_r": -0.2}),

    # A4. Two-stage: aggressive early + lenient late
    ("qe_2stg_3n05_9p01", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 3,
        "param_overrides.qe_stage1_min_r": -0.5,
        "param_overrides.quick_exit_max_bars": 9,
        "param_overrides.quick_exit_min_r": 0.1,
    }),
    ("qe_2stg_3n03_9p02", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 3,
        "param_overrides.qe_stage1_min_r": -0.3,
        "param_overrides.quick_exit_max_bars": 9,
        "param_overrides.quick_exit_min_r": 0.2,
    }),
    ("qe_2stg_4n03_10p01", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 4,
        "param_overrides.qe_stage1_min_r": -0.3,
        "param_overrides.quick_exit_max_bars": 10,
        "param_overrides.quick_exit_min_r": 0.1,
    }),

    # =======================================================================
    # B. MFE-TRIGGERED TRAILING STOP (7 candidates)
    #    Winners give back 0.531R; MFE capture only 31.5%
    #    Current 0.5/0.5 has boundary bug: trail_r=0 at activation
    # =======================================================================

    # B1. Fix boundary bug: tighter trail distance
    ("mfe_trail_05_03", {
        "param_overrides.fr_trailing_activate_r": 0.5,
        "param_overrides.fr_trailing_distance_r": 0.3,
    }),
    ("mfe_trail_03_03", {
        "param_overrides.fr_trailing_activate_r": 0.3,
        "param_overrides.fr_trailing_distance_r": 0.3,
    }),
    ("mfe_trail_075_03", {
        "param_overrides.fr_trailing_activate_r": 0.75,
        "param_overrides.fr_trailing_distance_r": 0.3,
    }),
    ("mfe_trail_05_04", {
        "param_overrides.fr_trailing_activate_r": 0.5,
        "param_overrides.fr_trailing_distance_r": 0.4,
    }),

    # B2. Breakeven stop
    ("be_stop_03r", {"param_overrides.close_stop_be_after_r": 0.3}),
    ("be_stop_05r", {"param_overrides.close_stop_be_after_r": 0.5}),

    # B3. Combined protection
    ("mfe_trail_03_03_be_03", {
        "param_overrides.fr_trailing_activate_r": 0.3,
        "param_overrides.fr_trailing_distance_r": 0.3,
        "param_overrides.close_stop_be_after_r": 0.3,
    }),

    # =======================================================================
    # C. OR_BREAKOUT QUALITY GATE (4 candidates)
    #    ~219 low-score OR trades drag the 747-trade OR pool down
    # =======================================================================

    ("or_score_min_4", {
        "ablation.use_or_quality_gate": True,
        "param_overrides.or_breakout_score_min": 4,
    }),
    ("or_score_min_5", {
        "ablation.use_or_quality_gate": True,
        "param_overrides.or_breakout_score_min": 5,
    }),
    ("or_score_min_6", {
        "ablation.use_or_quality_gate": True,
        "param_overrides.or_breakout_score_min": 6,
    }),
    ("or_score_min_3_rvol2", {
        "ablation.use_or_quality_gate": True,
        "param_overrides.or_breakout_score_min": 3,
        "param_overrides.or_breakout_min_rvol": 2.0,
    }),
]


# ---------------------------------------------------------------------------
# ALCB T2 Phase 10 — Full Re-optimization (new scoring + 2:1 leverage)
# ---------------------------------------------------------------------------

# Empty base: start from defaults. 2:1 leverage is baked into StrategySettings.
# Greedy will rediscover the optimal combination from scratch under new scoring.
ALCB_T2_P10_BASE_MUTATIONS: dict = {}

# Union of ALL candidates from P1-P9 + A/B test winners, deduplicated.
# Later phase versions override earlier for duplicate names.
_p10_all_candidates = [
    *ALCB_T2_P6_CANDIDATES,    # P1-P5 inherited + P6 new
    *ALCB_T2_P7_CANDIDATES,    # P7: quality & robustness
    *ALCB_T2_P8_CANDIDATES,    # P8: exit & signal diagnostics
    *ALCB_T2_P9_CANDIDATES,    # P9: quick exit / MFE trailing / OR gate
    # A/B test winner: ultra-early stage1 QE (bar 2, R < -0.5)
    ("early_qe_2bar_n05", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 2,
        "param_overrides.qe_stage1_min_r": -0.5,
    }),
]
_p10_dedup: dict[str, dict] = {}
for _name, _muts in _p10_all_candidates:
    _p10_dedup[_name] = _muts
ALCB_T2_P10_CANDIDATES: list[tuple[str, dict]] = list(_p10_dedup.items())


IARIC_T2_P5_BASE_MUTATIONS = {
    # T2 greedy optimal (carry-focused, selective)
    **IARIC_T2_BASE_MUTATIONS,
}

IARIC_T2_P5_CANDIDATES: list[tuple[str, dict]] = [
    # --- Shift 1: Capture the open-to-close return ---
    ("open_entry", {"param_overrides.t2_open_entry": True}),
    ("open_entry_half", {"param_overrides.t2_open_entry": True,
                         "param_overrides.t2_open_entry_size_mult": 0.5}),
    ("open_entry_wide_stop", {"param_overrides.t2_open_entry": True,
                              "param_overrides.t2_open_entry_stop_atr": 2.0}),
    # --- Shift 2: Carry-first architecture ---
    ("default_carry", {"param_overrides.t2_default_carry_profitable": True}),
    ("default_carry_r01", {"param_overrides.t2_default_carry_profitable": True,
                           "param_overrides.t2_default_carry_min_r": 0.1}),
    ("default_carry_tight", {"param_overrides.t2_default_carry_profitable": True,
                             "param_overrides.t2_default_carry_stop_atr": 0.5}),
    ("default_carry_wide", {"param_overrides.t2_default_carry_profitable": True,
                            "param_overrides.t2_default_carry_stop_atr": 1.5}),
    # --- Shift 3: Entry strength scoring ---
    ("entry_strength_30", {"param_overrides.t2_entry_strength_gate": 30.0}),
    ("entry_strength_40", {"param_overrides.t2_entry_strength_gate": 40.0}),
    ("entry_strength_50", {"param_overrides.t2_entry_strength_gate": 50.0}),
    ("entry_strength_sizing", {"param_overrides.t2_entry_strength_sizing": True}),
    # --- Extended ORB window ---
    ("orb_window_3", {"param_overrides.t2_orb_window_bars": 3}),
    ("orb_window_5", {"param_overrides.t2_orb_window_bars": 5}),
    ("orb_no_bullish", {"param_overrides.t2_orb_require_bullish": False}),
    ("orb_window_3_any", {"param_overrides.t2_orb_window_bars": 3,
                          "param_overrides.t2_orb_require_bullish": False}),
    # --- Multi-bar VWAP pullback ---
    ("vwap_multibar", {"param_overrides.t2_vwap_pullback_multibar": True}),
    ("vwap_multibar_loose", {"param_overrides.t2_vwap_pullback_multibar": True,
                             "param_overrides.t2_vwap_reclaim_vol_pct": 0.60}),
    # --- Graduated staleness ---
    ("stale_tighten", {"param_overrides.t2_staleness_action": "TIGHTEN"}),
    ("stale_skip", {"param_overrides.t2_staleness_action": "SKIP"}),
    # --- AVWAP breakdown tighten ---
    ("avwap_tighten", {"param_overrides.t2_avwap_breakdown_action": "TIGHTEN"}),
    ("avwap_disable", {"ablation.use_avwap_breakdown_exit": False}),
    # --- Regime B sizing ---
    ("regime_b_half", {"param_overrides.t2_regime_b_sizing_mult": 0.5}),
    ("regime_b_quarter", {"param_overrides.t2_regime_b_sizing_mult": 0.25}),
    # --- Breakeven on closes ---
    ("be_closes", {"param_overrides.t2_breakeven_use_closes": True}),
    # --- Fallback quality gate ---
    ("fallback_above_vwap", {"param_overrides.t2_fallback_require_above_vwap": True}),
    ("fallback_mom2", {"param_overrides.t2_fallback_momentum_bars": 2}),
]

IARIC_T1_CANDIDATES: list[tuple[str, dict]] = [
    # Param upgrades (override existing base values)
    ("risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
    ("risk_010", {"param_overrides.base_risk_fraction": 0.01}),
    ("risk_015", {"param_overrides.base_risk_fraction": 0.015}),
    ("carry_days_7", {"param_overrides.max_carry_days": 7}),
    ("carry_days_10", {"param_overrides.max_carry_days": 10}),
    ("carry_r_050", {"param_overrides.min_carry_r": 0.50}),
    ("carry_r_075", {"param_overrides.min_carry_r": 0.75}),
    ("carry_r_100", {"param_overrides.min_carry_r": 1.00}),
    # New features (add to base)
    ("flow_lb_1", {"param_overrides.flow_reversal_lookback": 1}),
    ("flow_lb_3", {"param_overrides.flow_reversal_lookback": 3}),
    ("stop_cap_010", {"param_overrides.stop_risk_cap_pct": 0.01}),
    ("stop_cap_015", {"param_overrides.stop_risk_cap_pct": 0.015}),
    ("conviction_order", {"param_overrides.entry_order_by_conviction": True}),
    ("intraday_flow", {"param_overrides.intraday_flow_check": True}),
    ("top_quartile", {"param_overrides.carry_top_quartile": True}),
    ("strong_only", {"param_overrides.strong_only_entry": True}),
    ("regime_b_carry_06", {"param_overrides.regime_b_carry_mult": 0.6}),
    ("no_conviction_gate", {"param_overrides.min_conviction_multiplier": 0.0}),

    # ── Tier 1: Structurally defensible ──

    # Entry flow gate: require positive flow proxy at entry
    ("entry_flow_pos_1", {"param_overrides.t1_entry_flow_gate": True, "param_overrides.t1_entry_flow_lookback": 1}),
    ("entry_flow_pos_2", {"param_overrides.t1_entry_flow_gate": True, "param_overrides.t1_entry_flow_lookback": 2}),

    # Intraday partial takes: lock in profit at R-trigger
    ("partial_05r_025f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 0.5, "param_overrides.t1_partial_fraction": 0.25}),
    ("partial_075r_033f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 0.75, "param_overrides.t1_partial_fraction": 0.33}),
    ("partial_10r_050f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 1.0, "param_overrides.t1_partial_fraction": 0.50}),

    # Gap-down entry filter: skip entries where open gaps below prev close
    ("gap_skip_005", {"param_overrides.t1_gap_down_skip_pct": 0.005}),
    ("gap_skip_010", {"param_overrides.t1_gap_down_skip_pct": 0.01}),
    ("gap_skip_015", {"param_overrides.t1_gap_down_skip_pct": 0.015}),

    # Carry close-pct: parameterize hardcoded 0.75 threshold
    ("carry_close_050", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.50}),
    ("carry_close_060", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.60}),
    ("carry_close_065", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.65}),

    # ── Tier 2: Defensible with caveats ──

    # Carry trailing stop: protect accumulated carry profit
    ("carry_trail_d1_15atr", {"param_overrides.carry_trail_activate_days": 1, "param_overrides.carry_trail_atr_mult": 1.5}),
    ("carry_trail_d2_10atr", {"param_overrides.carry_trail_activate_days": 2, "param_overrides.carry_trail_atr_mult": 1.0}),
    ("carry_breakeven_d1", {"param_overrides.carry_trail_activate_days": 1, "param_overrides.carry_trail_atr_mult": 0.0}),

    # Conviction minimum tightening
    ("conviction_035", {"param_overrides.min_conviction_multiplier": 0.35}),
    ("conviction_050", {"param_overrides.min_conviction_multiplier": 0.50}),

    # ── Tier 3: Speculative (flag for out-of-sample validation) ──

    # Day-of-week sizing (NOT statistically significant — include for optimizer test only)
    ("tue_half", {"param_overrides.dow_tuesday_mult": 0.5}),
    ("fri_125", {"param_overrides.dow_friday_mult": 1.25}),
]
