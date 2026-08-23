from dataclasses import fields
from pathlib import Path

import pytest

from backtests.stock.auto.iaric.phase_candidates import (
    V6R1_BASE_MUTATIONS,
    V6R1_PHASE_CANDIDATES,
)
from backtests.stock.auto.iaric.phase_scoring import (
    V6R1_PHASE_HARD_REJECTS,
    V6R1_PHASE_SCORING_WEIGHTS,
    score_v6r1_pullback_phase,
)
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin
from strategies.stock.iaric.config import StrategySettings


def _baseline_metrics() -> dict[str, float]:
    return {
        "total_trades": 89.0,
        "net_profit": 1_307.46,
        "avg_r": 0.236499957219503,
        "profit_factor": 1.6472990836043948,
        "max_drawdown_pct": 0.0632752553726446,
        "entry_realized_discrimination_lift_r": 0.2604893198240668,
        "robust_avg_r": 0.10,
    }


def test_v6r1_score_is_immutable_and_has_exactly_seven_components() -> None:
    expected = V6R1_PHASE_SCORING_WEIGHTS[1]
    assert len(expected) == 7
    assert sum(expected.values()) == pytest.approx(1.0)
    assert all(weights == expected for weights in V6R1_PHASE_SCORING_WEIGHTS.values())
    assert score_v6r1_pullback_phase(1, _baseline_metrics()) == pytest.approx(0.5)


def test_v6r1_score_rewards_broad_economic_improvement() -> None:
    baseline = _baseline_metrics()
    improved = dict(baseline)
    improved.update(
        total_trades=125.0,
        net_profit=2_200.0,
        avg_r=0.25,
        profit_factor=1.85,
        max_drawdown_pct=0.05,
        entry_realized_discrimination_lift_r=0.30,
    )
    assert score_v6r1_pullback_phase(1, improved) > score_v6r1_pullback_phase(1, baseline)


def test_v6r1_candidates_are_preregistered_and_reachable() -> None:
    setting_names = {field.name for field in fields(StrategySettings)}
    assert set(V6R1_PHASE_CANDIDATES) == {1, 2, 3, 4, 5}
    assert all(1 <= len(candidates) <= 7 for candidates in V6R1_PHASE_CANDIDATES.values())
    for candidates in V6R1_PHASE_CANDIDATES.values():
        for _, mutations in candidates:
            for key in mutations:
                assert key.startswith("param_overrides.")
                assert key.split(".", 1)[1] in setting_names


def test_v6r1_baseline_and_gates_match_repaired_round() -> None:
    assert V6R1_BASE_MUTATIONS["param_overrides.pb_carry_enabled"] is True
    assert V6R1_BASE_MUTATIONS["param_overrides.pb_open_scored_fill_timing"] == "next_5m_open"
    assert V6R1_BASE_MUTATIONS["param_overrides.pb_v2_open_scored_confirmation_policy"] == "band_reclaim"
    baseline = _baseline_metrics()
    for phase, rejects in V6R1_PHASE_HARD_REJECTS.items():
        assert IARICPullbackPlugin._phase_reject_reason(phase, baseline, rejects) == ""


def test_v6r1_refuses_to_open_the_sealed_holdout() -> None:
    with pytest.raises(ValueError, match="holdout is sealed"):
        IARICPullbackPlugin(
            data_dir=Path("."),
            end_date="2026-03-02",
            round_name="v6r1",
        )
