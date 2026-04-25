"""ATRSS worker -- per-process init and candidate scoring.

Workers load data ONCE at pool creation (init_worker) and reuse across all phases.
Phase/weights/rejects are passed per-call via score_candidate args.
"""
from __future__ import annotations

import io
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0


def init_worker(data_dir_str: str, equity: float) -> None:
    """Initialize worker process: install aliases, load data, create base config.

    Called once per worker at pool creation. Data is loaded here and reused
    for all subsequent score_candidate calls across all phases.
    """
    global _worker_data, _worker_config, _worker_equity

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
    from backtests.swing.engine.portfolio_engine import PortfolioData

    # Suppress engine logger noise
    logging.getLogger("backtest.engine.backtest_engine").setLevel(logging.WARNING)

    _worker_equity = equity
    data_dir = Path(data_dir_str)

    _worker_config = BacktestConfig(
        symbols=["QQQ", "GLD"],
        initial_equity=equity,
        fixed_qty=10,
        data_dir=data_dir,
        slippage=SlippageConfig(commission_per_contract=1.00),
        flags=AblationFlags(stall_exit=False),
    )

    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.preprocessing import (
        align_daily_to_hourly,
        build_numpy_arrays,
        filter_rth,
        normalize_timezone,
    )

    _worker_data = PortfolioData()
    for sym in ("QQQ", "GLD"):
        h_df = normalize_timezone(load_bars(data_dir / f"{sym}_1h.parquet"))
        h_df = filter_rth(h_df)
        d_df = normalize_timezone(load_bars(data_dir / f"{sym}_1d.parquet"))
        _worker_data.hourly[sym] = build_numpy_arrays(h_df)
        _worker_data.daily[sym] = build_numpy_arrays(d_df)
        _worker_data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)


def score_candidate(args: tuple) -> ScoredCandidate:
    """Score a single candidate mutation set.

    Args:
        args: (name, candidate_mutations, base_mutations, phase, scoring_weights, hard_rejects)

    Returns:
        ScoredCandidate with score, rejection status, and metrics.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from backtests.swing.engine.portfolio_engine import run_independent
        from backtests.swing.auto.config_mutator import mutate_atrss_config

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_atrss_config(_worker_config, all_muts)
        result = run_independent(_worker_data, config)

        from backtests.swing.auto.atrss.scoring import extract_atrss_metrics
        metrics = extract_atrss_metrics(result, _worker_equity)

        if phase > 0:
            from backtests.swing.auto.atrss.phase_scoring import score_phase_metrics
            score = score_phase_metrics(
                phase, metrics,
                weight_overrides=scoring_weights,
                hard_rejects=hard_rejects,
            )
        else:
            from .scoring import composite_score
            score = composite_score(metrics, hard_rejects=hard_rejects)

        return ScoredCandidate(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics=asdict(metrics),
        )

    except Exception:
        return ScoredCandidate(
            name=name,
            score=0.0,
            rejected=True,
            reject_reason=f"Error: {traceback.format_exc()[-200:]}",
            metrics={},
        )
