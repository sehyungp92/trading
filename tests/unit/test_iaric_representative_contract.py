from __future__ import annotations

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CONTRACT_VERSION,
    CURRENT_INPUT_AUTHORITY,
    DOWNSTREAM_EXECUTION_CONTRACT,
    EXPERIMENT_REGISTRY,
    PHASE_ORDER,
    ANCHOR_REVERSION_SLEEVE,
    SLEEVE_REQUIREMENTS,
    assess_atlas_for_optimization,
    assess_input_authority,
)
from backtests.stock.auto.runners.run_iaric_representative_preflight import (
    build_preflight_payload,
)


def _complete_authority() -> dict[str, bool]:
    return {
        key: True
        for sleeve in SLEEVE_REQUIREMENTS
        for key in sleeve.required_inputs
    }


def test_current_workspace_authority_fails_closed_by_sleeve() -> None:
    result = assess_input_authority(CURRENT_INPUT_AUTHORITY)
    assert result["representative_reversion_baseline_eligible"] is False
    assert all(not row["ready"] for row in result["sleeve_readiness"].values())
    assert result["blockers"]


def test_missing_five_minute_data_blocks_only_secondary_sleeves() -> None:
    authority = _complete_authority()
    authority["five_minute_ohlcv"] = False
    result = assess_input_authority(authority)
    assert result["representative_reversion_baseline_eligible"] is True
    assert result["sleeve_readiness"]["intraday_residual_failed_continuation"]["ready"] is False
    assert result["sleeve_readiness"]["gap_residual_failed_continuation"]["ready"] is False
    assert result["sleeve_readiness"][ANCHOR_REVERSION_SLEEVE]["ready"] is True


def test_complete_authority_still_requires_mechanism_pipeline_and_parity() -> None:
    atlas = {
        "representative_contract_version": CONTRACT_VERSION,
        "input_authority": _complete_authority(),
        "window": {"start": "2024-03-25", "end": CALIBRATION_END},
        "holdout_accessed": False,
        "mechanism_atlas_complete": True,
        "mechanism_candidate_registry_complete": False,
        "qualified_sleeves": [
            "daily_residual_reversion",
        ],
        "economic_input_parity": {
            "passed": True,
            "passed_sleeves": [
                "daily_residual_reversion",
            ],
        },
        "phase_order": list(PHASE_ORDER),
        "downstream_execution_contract": DOWNSTREAM_EXECUTION_CONTRACT,
    }
    result = assess_atlas_for_optimization(atlas)
    assert result["passed"] is False
    assert result["checks"]["mechanism_candidate_registry_complete"] is False


def test_fully_certified_atlas_passes_without_locked_or_holdout_access() -> None:
    atlas = {
        "representative_contract_version": CONTRACT_VERSION,
        "input_authority": _complete_authority(),
        "window": {"start": "2024-03-25", "end": CALIBRATION_END},
        "holdout_accessed": False,
        "mechanism_atlas_complete": True,
        "mechanism_candidate_registry_complete": True,
        "qualified_sleeves": [
            "daily_residual_reversion",
        ],
        "economic_input_parity": {
            "passed": True,
            "passed_sleeves": [
                "daily_residual_reversion",
            ],
        },
        "phase_order": list(PHASE_ORDER),
        "downstream_execution_contract": DOWNSTREAM_EXECUTION_CONTRACT,
    }
    assert assess_atlas_for_optimization(atlas)["passed"] is True


def test_authority_preflight_is_zero_replay_and_explicitly_diagnostic() -> None:
    payload = build_preflight_payload("2024-03-25", CALIBRATION_END)
    assert payload["status"] == "complete_price_data_preflight"
    assert payload["strategy_input_scope"] == "price_volume_only"
    assert payload["news_or_earnings_required"] is False
    assert payload["mechanism_atlas_complete"] is False
    assert payload["economic_input_parity"]["passed"] is False
    assert payload["downstream_execution_contract"] == "authority_only_no_execution"
    assert payload["input_authority_attestation"]["manifest_found"] is False


def test_preflight_rejects_any_unregistered_or_holdout_window() -> None:
    payload = build_preflight_payload("2024-03-25", "2026-03-02")
    assert payload["selection_window_valid"] is False
    assert payload["representative_reversion_baseline_eligible"] is False
    assert payload["holdout_accessed"] is False  # no replay is loaded at preflight
    assert "must exactly match" in " ".join(payload["programme_blockers"])


def test_phase_contract_contains_no_legacy_frozen_phase_continuation() -> None:
    assert len(PHASE_ORDER) == 20
    assert not any("frozen" in phase or "second_dislocation" in phase for phase in PHASE_ORDER)
    assert PHASE_ORDER[-1].endswith("locked_chronological_validation")
    assert not any("shadow" in phase or "capital_pilot" in phase for phase in PHASE_ORDER)
    assert tuple(EXPERIMENT_REGISTRY) == PHASE_ORDER
    assert all(row["experiments"] and row["gate"] for row in EXPERIMENT_REGISTRY.values())


def test_price_volume_contract_has_no_news_quote_or_order_imbalance_dependency() -> None:
    forbidden = ("news", "earnings", "quote", "imbalance", "one_minute")
    authority_names = " ".join(CURRENT_INPUT_AUTHORITY).lower()
    requirement_names = " ".join(
        input_name
        for sleeve in SLEEVE_REQUIREMENTS
        for input_name in sleeve.required_inputs
    ).lower()
    assert not any(name in authority_names for name in forbidden)
    assert not any(name in requirement_names for name in forbidden)
