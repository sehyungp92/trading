from __future__ import annotations

import pytest

from strategies.stock.iaric.core.mechanisms import (
    InformationState,
    SLEEVE_SPECS,
    information_state_veto,
    score_mechanism_components,
    validate_sleeve_specs,
)


def test_mechanism_scores_are_sleeve_specific_and_never_exceed_seven_components() -> None:
    result = validate_sleeve_specs()
    assert result["passed"] is True
    assert result["max_score_components"] <= 7
    reversion_sets = {
        spec.score_components
        for spec in SLEEVE_SPECS.values()
        if spec.role == "reversion"
    }
    assert len(reversion_sets) == 3
    assert len(SLEEVE_SPECS["daily_residual_reversion"].score_components) == 7


def test_information_state_never_vetoes_price_volume_sleeves() -> None:
    assert information_state_veto(
        "intraday_residual_failed_continuation", InformationState.UNKNOWN
    ) is None
    assert information_state_veto(
        "intraday_residual_failed_continuation", InformationState.EARNINGS
    ) is None
    assert information_state_veto(
        "daily_residual_reversion", InformationState.VERIFIED_NO_EVENT
    ) is None


def test_one_sleeve_cannot_borrow_another_sleeves_score() -> None:
    daily = SLEEVE_SPECS["daily_residual_reversion"]
    components = {name: 0.5 for name in daily.score_components}
    assert score_mechanism_components(daily.name, components) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        score_mechanism_components("intraday_residual_failed_continuation", components)


def test_reversion_sleeves_have_no_information_or_quote_vetoes() -> None:
    forbidden = ("news", "information", "quote", "imbalance", "earnings")
    for spec in SLEEVE_SPECS.values():
        if spec.role != "reversion":
            continue
        joined = " ".join(spec.hard_vetoes).lower()
        assert not any(name in joined for name in forbidden)
