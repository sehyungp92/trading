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

    from backtests.momentum._aliases import install

    install()

    from backtest.config_nqdtc import NQDTCBacktestConfig

    _worker_config = NQDTCBacktestConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
        fixed_qty=10,
    )
    if _worker_data is None:
        _worker_data = load_worker_data("NQ", Path(data_dir_str))


def load_worker_data(symbol: str, data_dir: Path) -> dict:
    """Load NQDTC bar data (same as cli._load_nqdtc_data)."""
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import (
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    five_min_path = data_dir / f"{symbol}_5m.parquet"
    daily_path = data_dir / f"{symbol}_1d.parquet"

    m_df = normalize_timezone(load_bars(five_min_path))
    m_df = filter_eth(m_df)

    m30_df = resample_5m_to_30m(m_df)
    h_df = resample_5m_to_1h(m_df)
    fh_df = resample_5m_to_4h(m_df)

    if daily_path.exists():
        d_df = normalize_timezone(load_bars(daily_path))
    else:
        d_df = resample_5m_to_daily(m_df)

    data = {
        "five_min_bars": build_numpy_arrays(m_df),
        "thirty_min": build_numpy_arrays(m30_df),
        "hourly": build_numpy_arrays(h_df),
        "four_hour": build_numpy_arrays(fh_df),
        "daily": build_numpy_arrays(d_df),
        "thirty_min_idx_map": align_higher_tf_to_5m(m_df, m30_df),
        "hourly_idx_map": align_higher_tf_to_5m(m_df, h_df),
        "four_hour_idx_map": align_higher_tf_to_5m(m_df, fh_df),
        "daily_idx_map": align_higher_tf_to_5m(m_df, d_df),
    }

    es_path = data_dir / "ES_1d.parquet"
    if es_path.exists():
        es_df = normalize_timezone(load_bars(es_path))
        data["daily_es"] = build_numpy_arrays(es_df)
        data["daily_es_idx_map"] = align_higher_tf_to_5m(m_df, es_df)

    return data


def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a single candidate.

    Phase/weights/rejects are passed per-task so the worker pool can be
    reused across phases without re-initialization.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from dataclasses import asdict

        from backtest.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.plugin import score_phase_metrics
        from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_nqdtc_config(_worker_config, all_muts)
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(**_worker_data)

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
