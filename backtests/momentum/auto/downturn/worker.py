from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_phase: int = 1
_worker_weights: dict | None = None
_worker_hard_rejects: dict | None = None


def init_worker(
    data_dir_str: str,
    equity: float,
    phase: int = 1,
    scoring_weights: dict | None = None,
    hard_rejects: dict | None = None,
) -> None:
    global _worker_data, _worker_config, _worker_phase, _worker_weights, _worker_hard_rejects

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.momentum._aliases import install

    install()

    from backtest.config_downturn import DownturnBacktestConfig

    _worker_phase = phase
    _worker_weights = scoring_weights
    _worker_hard_rejects = hard_rejects
    _worker_config = DownturnBacktestConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
    )
    _worker_data = load_worker_data("NQ", Path(data_dir_str))


def load_worker_data(symbol: str, data_dir: Path) -> dict:
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import (
        align_daily_to_5m,
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_15m,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    five_min_path = data_dir / f"{symbol}_5m.parquet"
    daily_path = data_dir / f"{symbol}_1d.parquet"

    m_df = normalize_timezone(load_bars(five_min_path))
    m_df = filter_eth(m_df)

    m15_df = resample_5m_to_15m(m_df)
    m30_df = resample_5m_to_30m(m_df)
    h_df = resample_5m_to_1h(m_df)
    fh_df = resample_5m_to_4h(m_df)

    if daily_path.exists():
        d_df = normalize_timezone(load_bars(daily_path))
    else:
        d_df = resample_5m_to_daily(m_df)

    data = {
        "five_min": build_numpy_arrays(m_df),
        "fifteen_min": build_numpy_arrays(m15_df),
        "thirty_min": build_numpy_arrays(m30_df),
        "hourly": build_numpy_arrays(h_df),
        "four_hour": build_numpy_arrays(fh_df),
        "daily": build_numpy_arrays(d_df),
        "fifteen_min_idx_map": align_higher_tf_to_5m(m_df, m15_df),
        "thirty_min_idx_map": align_higher_tf_to_5m(m_df, m30_df),
        "hourly_idx_map": align_higher_tf_to_5m(m_df, h_df),
        "four_hour_idx_map": align_higher_tf_to_5m(m_df, fh_df),
        "daily_idx_map": align_daily_to_5m(m_df, d_df),
    }

    es_path = data_dir / "ES_1d.parquet"
    if es_path.exists():
        es_df = normalize_timezone(load_bars(es_path))
        data["daily_es"] = build_numpy_arrays(es_df)
        data["daily_es_idx_map"] = align_daily_to_5m(m_df, es_df)
    else:
        data["daily_es"] = None
        data["daily_es_idx_map"] = None

    return data


def score_candidate(args: tuple[str, dict, dict]) -> ScoredCandidate:
    name, candidate_muts, base_muts = args

    try:
        from backtest.engine.downturn_engine import DownturnEngine
        from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
        from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
        from backtests.momentum.auto.downturn.plugin import score_phase_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_downturn_config(_worker_config, all_muts)
        engine = DownturnEngine("NQ", config)
        result = engine.run(**_worker_data)
        metrics = compute_downturn_metrics(result, _worker_data["daily"])
        score = score_phase_metrics(
            _worker_phase,
            metrics,
            weight_overrides=_worker_weights,
            hard_rejects=_worker_hard_rejects,
        )

        if score.rejected:
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=score.reject_reason)
        return ScoredCandidate(name=name, score=score.total)

    except Exception as exc:
        logger.error("Worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")
