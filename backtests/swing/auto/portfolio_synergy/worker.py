from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from backtests.shared.auto.types import ScoredCandidate

from .evaluator import evaluate_portfolio, load_evaluation_data
from .scoring import score_portfolio_metrics

_DATA_DIR: Path | None = None
_INITIAL_EQUITY: float = 0.0
_UNIFIED_DATA: Any = None


def init_worker(data_dir: str, initial_equity: float) -> None:
    global _DATA_DIR, _INITIAL_EQUITY, _UNIFIED_DATA
    _DATA_DIR = Path(data_dir)
    _INITIAL_EQUITY = float(initial_equity)
    _UNIFIED_DATA, _ = load_evaluation_data(_DATA_DIR, _INITIAL_EQUITY)


def score_candidate(args) -> ScoredCandidate:
    name, candidate_mutations, current_mutations, scoring_weights, hard_rejects = args
    try:
        if _DATA_DIR is None:
            raise RuntimeError("portfolio synergy worker was not initialized")
        merged = dict(current_mutations or {})
        merged.update(candidate_mutations or {})
        metrics = evaluate_portfolio(
            merged,
            data_dir=_DATA_DIR,
            initial_equity=_INITIAL_EQUITY,
            unified_data=_UNIFIED_DATA,
        )
        score = score_portfolio_metrics(
            metrics,
            scoring_weights=scoring_weights,
            hard_rejects=hard_rejects,
        )
        return ScoredCandidate(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics={
                **metrics,
                **{f"score_{key}": value for key, value in score.components.items()},
                "score_total": score.total,
            },
        )
    except Exception:
        return ScoredCandidate(
            name=name,
            score=0.0,
            rejected=True,
            reject_reason=traceback.format_exc(),
            metrics={},
        )
