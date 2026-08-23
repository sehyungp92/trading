from __future__ import annotations

import json

from backtests.stock.auto.runners.run_iaric_portable_alpha_escape import (
    DEFAULT_BASELINE,
    DEFAULT_PHASE8,
    RESIDUAL,
    SCORE_SPEC,
    UPTREND,
    _load_capped_baseline,
    _parse_mapping,
    _phase1_candidates,
    _phase2_candidates,
)


def test_portable_score_is_fixed_to_exactly_seven_components() -> None:
    assert len(SCORE_SPEC) == 7
    assert sum(item["weight"] for item in SCORE_SPEC.values()) == 1.0
    assert set(SCORE_SPEC) == {
        "incremental_total_r",
        "incremental_trades",
        "issuer_neutral_delta_r",
        "issuer_hhi_improvement",
        "worst_segment_delta_r",
        "discrimination_delta_r",
        "drawdown_improvement",
    }


def test_phase1_is_targeted_and_uses_fixed_portable_atlas_policies() -> None:
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    candidates = _phase1_candidates(baseline, smoke=False)
    assert len(candidates) == 6
    assert len({tuple(row["mutations"].items()) for row in candidates}) == 6
    uptrend = next(row for row in candidates if row["id"] == "portable_uptrend")["mutations"]
    residual = next(row for row in candidates if row["id"] == "portable_residual")["mutations"]
    assert _parse_mapping(uptrend["param_overrides.pb_aperture_family_filters"])[UPTREND] == "quiet_deep_room"
    assert _parse_mapping(uptrend["param_overrides.pb_aperture_family_score_floors"])[UPTREND] == "40"
    assert _parse_mapping(residual["param_overrides.pb_aperture_family_filters"])[RESIDUAL] == "relative_exhaustion"
    assert _parse_mapping(residual["param_overrides.pb_aperture_family_transitions"])[RESIDUAL] == "confirm"


def test_phase2_preserves_measured_issuer_caps_and_only_composes_targeted_routes() -> None:
    capped = _load_capped_baseline(DEFAULT_PHASE8)
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    phase1 = _phase1_candidates(baseline, smoke=False)
    for index, row in enumerate(phase1):
        row["portable_score"] = float(index)
        row["metrics"] = {"expected_total_r": float(index)}
    candidates = _phase2_candidates(capped, phase1, smoke=False)
    assert len(candidates) == 4
    for row in candidates:
        assert row["mutations"]["param_overrides.pb_issuer_position_cap"] == 1
        assert row["mutations"]["param_overrides.pb_issuer_daily_entry_cap"] == 1
    composed = next(row for row in candidates if row["id"] == "capped_portable_composition")
    families = set(composed["mutations"]["param_overrides.pb_aperture_families"].split(","))
    assert {UPTREND, RESIDUAL} <= families
