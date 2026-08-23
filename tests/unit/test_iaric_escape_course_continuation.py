from __future__ import annotations

from backtests.stock.auto.runners.run_iaric_escape_course_continuation import (
    _broad_validation_shortlist,
    _diverse_management_parents,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import _signature


def _row(
    name: str,
    family: str,
    score: float,
    total_r: float,
    trades: int,
    pf: float,
    dd: float,
) -> dict[str, object]:
    return {
        "id": name,
        "families": [family],
        "mutations": {
            "param_overrides.pb_aperture_enabled": True,
            "param_overrides.pb_aperture_families": family,
            "variant": name,
        },
        "escape_score": score,
        "metrics": {
            "expected_total_r": total_r,
            "total_trades": trades,
            "avg_r": 0.25,
            "profit_factor": pf,
            "max_drawdown_pct": dd,
        },
        "aperture": {"trades": max(trades - 80, 10), "total_r": 5.0},
    }


def _control() -> dict[str, object]:
    return {
        "id": "incumbent_control",
        "mutations": {},
        "metrics": {
            "expected_total_r": 20.0,
            "total_trades": 80,
            "avg_r": 0.25,
            "profit_factor": 1.6,
            "max_drawdown_pct": 0.04,
        },
        "aperture": {"trades": 0, "total_r": 0.0},
    }


def test_management_parent_beam_preserves_distinct_structures() -> None:
    control = _control()
    phase2 = [
        _row("a_best", "A", 0.90, 40.0, 120, 1.8, 0.03),
        _row("a_second", "A", 0.89, 39.0, 119, 1.9, 0.03),
        _row("b_best", "B", 0.88, 38.0, 130, 1.6, 0.035),
        _row("c_best", "C", 0.87, 35.0, 150, 1.5, 0.04),
    ]

    selected = _diverse_management_parents(phase2, control, 3)

    assert [row["id"] for row in selected] == ["a_best", "b_best", "c_best"]


def test_validation_beam_keeps_primary_and_frequency_diversity() -> None:
    control = _control()
    primary = _row("primary", "A", 0.92, 43.0, 130, 1.8, 0.03)
    high_frequency = _row("frequency", "B", 0.86, 35.0, 170, 1.5, 0.04)
    low_drawdown = _row("low_dd", "C", 0.84, 34.0, 125, 1.7, 0.02)

    selected, reasons = _broad_validation_shortlist(
        [primary, high_frequency, low_drawdown],
        control,
        mandatory_signatures=[_signature(primary["mutations"])],
        limit=8,
    )

    assert {row["id"] for row in selected} == {"primary", "frequency", "low_dd"}
    assert "primary_validated_finalist" in reasons[_signature(primary["mutations"])]
    assert "top_frequency" in reasons[_signature(high_frequency["mutations"])]
    assert "lowest_drawdown" in reasons[_signature(low_drawdown["mutations"])]
