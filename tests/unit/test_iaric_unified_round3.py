from __future__ import annotations

from backtests.stock.auto.runners.run_iaric_unified_round3 import (
    ROUTE_READINESS,
    UNIFIED_SCORE_SPEC,
    _score,
)
from strategies.stock.iaric.core.opportunity import REVERSION_FAMILIES


def _candidate(drawdown: float) -> dict:
    return {
        "metrics": {
            "expected_total_r": 30.0,
            "total_trades": 102.0,
            "max_drawdown_pct": drawdown,
            "avg_r": 0.12,
            "profit_factor": 1.45,
            "entry_realized_discrimination_lift_r": 0.0,
        },
        "validation": {
            "folds": [
                {"avg_r": 0.10},
                {"avg_r": 0.05},
                {"avg_r": 0.08},
            ],
        },
    }


def test_unified_score_is_fixed_at_seven_components_and_rewards_lower_drawdown() -> None:
    low_score, low_components, _ = _score(_candidate(0.03))
    high_score, high_components, _ = _score(_candidate(0.07))

    assert len(UNIFIED_SCORE_SPEC) == 7
    assert sum(item["weight"] for item in UNIFIED_SCORE_SPEC.values()) == 1.0
    assert set(low_components) == set(UNIFIED_SCORE_SPEC)
    assert low_score > high_score
    assert low_components["inverse_mtm_drawdown"] > high_components["inverse_mtm_drawdown"]


def test_every_atlas_reversion_family_has_an_explicit_non_approximate_readiness_gate() -> None:
    assert set(ROUTE_READINESS) == set(REVERSION_FAMILIES)
    assert all(row["detector"] for row in ROUTE_READINESS.values())
    assert all(row["missing"] for row in ROUTE_READINESS.values())
