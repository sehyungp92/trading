"""NQDTC worker -- multiprocessing-safe candidate evaluation."""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None


def init_worker(data_dir_str: str, equity: float) -> None:
    """Initialize worker: load data once, reuse across all phases/tasks.

    Phase, scoring weights, and hard rejects are passed per-task in
    score_candidate() so the pool can be reused across phases without
    re-initialization (avoids expensive data re-loading).
    """
    global _worker_data, _worker_config

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Suppress verbose box/engine logging (Box ACTIVE/DIRTY) in worker processes
    logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig

    _worker_config = NQDTCBacktestConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
        fixed_qty=10,
    )
    if _worker_data is None:
        _worker_data = load_worker_data("NQ", Path(data_dir_str))


def load_worker_data(symbol: str, data_dir: Path) -> dict:
    """Load NQDTC bar data (same as cli._load_nqdtc_data)."""
    from backtests.momentum.data.replay_cache import load_replay_bundle

    bundle = load_replay_bundle(
        symbol,
        data_dir,
        include_fifteen_min=False,
        include_thirty_min=True,
        include_hourly=True,
        include_four_hour=True,
        include_daily=True,
        include_daily_es=True,
    )
    return {
        "five_min_bars": bundle["five_min"],
        "thirty_min": bundle["thirty_min"],
        "hourly": bundle["hourly"],
        "four_hour": bundle["four_hour"],
        "daily": bundle["daily"],
        "thirty_min_idx_map": bundle["thirty_min_idx_map"],
        "hourly_idx_map": bundle["hourly_idx_map"],
        "four_hour_idx_map": bundle["four_hour_idx_map"],
        "daily_idx_map": bundle["daily_idx_map"],
        "daily_es": bundle.get("daily_es"),
        "daily_es_idx_map": bundle.get("daily_es_idx_map"),
        "cache_key": bundle.get("cache_key"),
        "cache_source_fingerprint": bundle.get("cache_source_fingerprint"),
    }


def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a single candidate.

    Phase/weights/rejects are passed per-task so the worker pool can be
    reused across phases without re-initialization.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from dataclasses import asdict

        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.plugin import score_phase_metrics
        from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_nqdtc_config(_worker_config, all_muts)
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(**replay_engine_kwargs(_worker_data))

        metrics = extract_nqdtc_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_config.initial_equity,
        )
        score = score_phase_metrics(
            phase,
            metrics,
            weight_overrides=scoring_weights,
            hard_rejects=hard_rejects,
        )

        metrics_dict = asdict(metrics)
        if score.rejected:
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=score.reject_reason, metrics=metrics_dict)
        return ScoredCandidate(name=name, score=score.total, metrics=metrics_dict)

    except Exception as exc:
        logger.error("Worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")
