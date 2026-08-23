from __future__ import annotations

from backtests.stock.auto.runners.analyze_stock_opportunity_atlas import (
    _audit_family,
)


def _record(*, fold: str, score: float, value: float, day: int) -> dict:
    return {
        "date": f"2025-01-{day:02d}",
        "fold": fold,
        "score": score,
        "score_components": {
            "dislocation": 0.7,
            "reclaim": 0.7,
            "close_quality": 0.7,
            "relative_volume": 0.5,
            "residual_dislocation": 0.5,
            "prior_down_sequence": 0.5,
            "reversion_room": 0.5,
        },
        "horizon_r": {
            "bar_3": value,
            "bar_6": value,
            "bar_12": value,
            "bar_24": value,
            "bar_48": value,
            "eod": value,
        },
    }


def test_aperture_selection_is_unchanged_by_later_fold_outcomes() -> None:
    accepted_early = [
        _record(fold="early", score=55.0, value=0.20, day=(index % 28) + 1)
        for index in range(35)
    ]
    rejected_early = [
        _record(fold="early", score=30.0, value=-0.20, day=(index % 28) + 1)
        for index in range(35)
    ]
    accepted_later = [
        _record(fold=fold, score=55.0, value=0.10, day=(index % 28) + 1)
        for fold in ("middle", "latest")
        for index in range(30)
    ]
    rejected_later = [
        _record(fold=fold, score=30.0, value=-0.20, day=(index % 28) + 1)
        for fold in ("middle", "latest")
        for index in range(30)
    ]
    negative_accepted_later = [
        _record(fold=record["fold"], score=55.0, value=-0.50, day=(index % 28) + 1)
        for index, record in enumerate(accepted_later)
    ]

    early = accepted_early + rejected_early
    positive = _audit_family(early + accepted_later + rejected_later, simulations=50)
    negative = _audit_family(early + negative_accepted_later + rejected_later, simulations=50)

    assert positive["selected_aperture"] == negative["selected_aperture"]
    assert positive["selected_entry_variant"] == negative["selected_entry_variant"]
    assert positive["selected_horizon"] == negative["selected_horizon"]
    assert positive["route_ready_for_portfolio_replay"] is True
    assert negative["route_ready_for_portfolio_replay"] is False


def test_no_route_is_selected_without_minimum_early_breadth() -> None:
    records = [
        _record(fold="early", score=80.0, value=1.0, day=index + 1)
        for index in range(10)
    ]

    result = _audit_family(records, simulations=20)

    assert result["selected_aperture"] is None
    assert result["route_ready_for_portfolio_replay"] is False
