from __future__ import annotations

import io
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_data_dir_key: str | None = None


def init_worker(data_dir_str: str, equity: float) -> None:
    global _worker_data, _worker_config, _worker_equity, _worker_data_dir_key

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.momentum.cli import _load_helix_data_cached
    from backtests.momentum.config_helix import Helix4BacktestConfig

    data_dir = Path(data_dir_str)
    data_dir_key = str(data_dir.resolve())
    if _worker_data is None or _worker_data_dir_key != data_dir_key:
        _worker_data = _load_helix_data_cached("NQ", data_dir)
        _worker_data_dir_key = data_dir_key
    _worker_equity = equity
    _worker_config = Helix4BacktestConfig(
        initial_equity=equity,
        fixed_qty=10,
        point_value=2.0,
        data_dir=data_dir,
        track_signals=False,
        track_shadows=False,
    )


def score_candidate(args: tuple) -> ScoredCandidate:
    name = args[0]
    candidate_muts = args[1]
    base_muts = args[2]
    phase = args[3]
    scoring_weights = args[4]
    hard_rejects = args[5]
    target_overrides = args[6] if len(args) > 6 else None

    try:
        from backtests.momentum.auto.config_mutator import mutate_helix_config
        from backtests.momentum.auto.scoring import extract_metrics
        from backtests.momentum.engine.helix_engine import Helix4Engine

        from .scoring import score_phase_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_helix_config(_worker_config, all_muts)
        result = Helix4Engine(symbol="NQ", bt_config=config).run(
            _worker_data["minute_bars"],
            _worker_data["hourly"],
            _worker_data["four_hour"],
            _worker_data["daily"],
            _worker_data["hourly_idx_map"],
            _worker_data["four_hour_idx_map"],
            _worker_data["daily_idx_map"],
        )
        metrics = extract_metrics(
            result.trades,
            result.equity_curve,
            _timestamps_to_numeric(result.timestamps),
            _worker_equity,
        )
        score = score_phase_metrics(
            phase,
            metrics,
            _worker_equity,
            equity_curve=result.equity_curve,
            weight_overrides=scoring_weights,
            hard_rejects=hard_rejects,
            target_overrides=target_overrides,
        )
        metrics_dict = asdict(metrics)
        if score.rejected:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=score.reject_reason,
                metrics=metrics_dict,
            )
        return ScoredCandidate(name=name, score=score.total, metrics=metrics_dict)
    except Exception as exc:
        logger.error("AKC Helix worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")


def _timestamps_to_numeric(timestamps) -> np.ndarray:
    if timestamps is None or len(timestamps) == 0:
        return np.array([])
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        return np.array([dt.timestamp() for dt in timestamps], dtype=float)
    return np.asarray(timestamps)
