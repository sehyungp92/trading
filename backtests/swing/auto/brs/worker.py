from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_phase: int = 0
_worker_scoring_weights: dict | None = None
_worker_hard_rejects: dict | None = None


def init_worker(
    data_dir_str: str,
    equity: float,
    phase: int = 0,
    scoring_weights: dict | None = None,
    hard_rejects: dict | None = None,
) -> None:
    global _worker_data, _worker_config, _worker_equity, _worker_phase, _worker_scoring_weights, _worker_hard_rejects

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing._aliases import install

    install()

    from backtest.config_brs import BRSConfig
    from backtest.engine.brs_portfolio_engine import load_brs_data

    _worker_equity = equity
    _worker_phase = phase
    _worker_scoring_weights = scoring_weights
    _worker_hard_rejects = hard_rejects
    _worker_config = BRSConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
    )
    _worker_data = load_brs_data(_worker_config)


def score_candidate(args: tuple[str, dict, dict]) -> ScoredCandidate:
    name, candidate_muts, base_muts = args

    try:
        from backtest.engine.brs_portfolio_engine import run_brs_independent
        from backtests.swing.auto.brs.config_mutator import mutate_brs_config
        from backtests.swing.auto.brs.plugin import score_phase_metrics
        from backtests.swing.auto.brs.scoring import composite_score, extract_brs_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_brs_config(_worker_config, all_muts)
        result = run_brs_independent(_worker_data, config)
        metrics = extract_brs_metrics(result, _worker_equity)

        if _worker_phase > 0:
            score = score_phase_metrics(
                _worker_phase,
                metrics,
                weight_overrides=_worker_scoring_weights,
                hard_rejects=_worker_hard_rejects,
            )
        else:
            score = composite_score(metrics)

        if score.rejected:
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=score.reject_reason)
        return ScoredCandidate(name=name, score=score.total)

    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())
