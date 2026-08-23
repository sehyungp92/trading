from __future__ import annotations

from backtests.stock.auto.iaric.round4_scoring import SCORE_SPEC
from backtests.stock.auto.runners.run_iaric_round4_continuation import (
    PHASE_ORDER,
    _behavior_delta,
    _families,
    _merge_rows,
    _parse_mapping,
    _phase5_candidates,
    _phase6_candidates,
    _phase7_candidates,
    _phase9_ablation_candidates,
    _positive_structural,
    _scope,
)
from strategies.stock.iaric.core.lanes import SCORE_COMPONENTS


def _baseline() -> dict[str, object]:
    return {
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_families": (
            "FAILED_BREAKDOWN_RECLAIM,MULTIDAY_HIGHER_LOW_RECLAIM,"
            "PRIOR_DAY_LOW_RECLAIM,UPTREND_PULLBACK_RECLAIM"
        ),
        "param_overrides.pb_aperture_family_score_floors": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:65,PRIOR_DAY_LOW_RECLAIM:70"
        ),
        "param_overrides.pb_aperture_family_transitions": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:next_bar,PRIOR_DAY_LOW_RECLAIM:next_bar"
        ),
        "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
    }


def _parent(candidate_id: str = "phase4") -> dict[str, object]:
    baseline = _baseline()
    return {
        "id": candidate_id,
        "mutations": baseline,
        "focus_scope": _scope(families=_families(baseline)),
    }


def test_round4_uses_representative_order_with_management_before_composition() -> None:
    assert PHASE_ORDER[0] == "phase_0_price_data_integrity_and_parity"
    assert PHASE_ORDER[1] == "phase_1_residual_model_and_horizon_atlas"
    assert PHASE_ORDER[2] == "phase_2_feature_qualification_and_discrimination"
    assert PHASE_ORDER[3] == "phase_3_selection_contract_robustness"
    assert PHASE_ORDER.index("phase_5_residual_anchor_and_half_life_management") < PHASE_ORDER.index(
        "phase_7_protected_integration_and_literal_ablation"
    )
    assert PHASE_ORDER[-1] == "phase_16_locked_chronological_validation"
    assert len(SCORE_SPEC) == 7
    assert len(SCORE_COMPONENTS) == 7


def test_phase5_preserves_control_and_defers_non_equivalent_score_profiles() -> None:
    candidates = _phase5_candidates(_baseline())
    assert [candidate["id"] for candidate in candidates] == ["incumbent_control"]
    assert "param_overrides.pb_aperture_family_score_profiles" not in candidates[0]["mutations"]


def test_phase6_new_lanes_are_causal_separate_and_explicitly_capped() -> None:
    candidates = _phase6_candidates([_parent()])
    assert len(candidates) == 8  # one control and seven activation-capable causal lanes
    for candidate in candidates:
        mutations = candidate["mutations"]
        assert mutations["param_overrides.pb_open_scored_fill_timing"] == "next_5m_open"
        if candidate.get("focus_key"):
            family = candidate["focus_key"]
            family_caps = _parse_mapping(
                mutations["param_overrides.pb_aperture_family_daily_caps"]
            )
            assert family_caps[family] == "1"
            assert family in _families(mutations)


def test_phase7_rearm_is_one_predeclared_second_episode_not_a_cooldown_grid() -> None:
    candidates = _phase7_candidates([_parent()])
    assert len(candidates) == 4
    for candidate in candidates[1:]:
        mutations = candidate["mutations"]
        assert mutations["param_overrides.pb_aperture_rearm_cooldown_bars"] == 12
        events = _parse_mapping(
            mutations["param_overrides.pb_aperture_family_max_events"]
        )
        daily = _parse_mapping(
            mutations["param_overrides.pb_aperture_family_daily_caps"]
        )
        assert set(events.values()) == {"2"}
        assert all(daily[family] == "2" for family in events)


def test_composition_merge_preserves_independent_family_policies_and_caps() -> None:
    base = _baseline()
    phase6 = _phase6_candidates([_parent()])
    gap = next(row for row in phase6 if row.get("focus_key") == "GAP_EXHAUSTION_RECLAIM")
    vwap = next(row for row in phase6 if row.get("focus_key") == "VWAP_DEVIATION_RECLAIM")
    merged = _merge_rows(base, (gap, vwap))
    assert {"GAP_EXHAUSTION_RECLAIM", "VWAP_DEVIATION_RECLAIM"} <= _families(merged)
    caps = _parse_mapping(merged["param_overrides.pb_aperture_family_daily_caps"])
    assert caps["GAP_EXHAUSTION_RECLAIM"] == "1"
    assert caps["VWAP_DEVIATION_RECLAIM"] == "1"
    assert "param_overrides.pb_aperture_family_score_profiles" not in merged


def test_inert_or_cannibalizing_lane_cannot_be_called_positive_structural() -> None:
    trade = {
        "symbol": "AAA",
        "entry_time": "2025-01-02T10:00:00+00:00",
        "route": "APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY",
        "lane": "APERTURE_LEVEL_RECLAIM",
        "r": 1.0,
    }
    control = {
        "id": "control",
        "metrics": {
            "expected_total_r": 10.0,
            "avg_r": 0.3,
            "profit_factor": 1.8,
            "max_drawdown_pct": 0.03,
        },
        "trade_attribution": [trade],
    }
    inert = {
        "id": "inert",
        "metrics": dict(control["metrics"]),
        "trade_attribution": [trade],
        "focus": {"trades": 1, "total_r": 1.0, "profit_factor": 99.0},
    }
    inert["incremental_attribution"] = _behavior_delta(inert, control)
    assert inert["incremental_attribution"]["materially_active"] is False
    assert _positive_structural(inert, control) is False


def test_composition_ablation_manifest_removes_one_literal_source_at_a_time() -> None:
    base = _baseline()
    lanes = _phase6_candidates([_parent()])
    gap = next(row for row in lanes if row.get("focus_key") == "GAP_EXHAUSTION_RECLAIM")
    vwap = next(row for row in lanes if row.get("focus_key") == "VWAP_DEVIATION_RECLAIM")
    composition = {
        "id": "composition",
        "mutations": _merge_rows(base, (gap, vwap)),
        "source_ids": [gap["id"], vwap["id"]],
        "focus_scope": _scope(
            families=("GAP_EXHAUSTION_RECLAIM", "VWAP_DEVIATION_RECLAIM")
        ),
        "round4_score": 0.6,
        "incremental_attribution": {"materially_active": True},
    }
    candidates = _phase9_ablation_candidates(base, [composition], lanes)
    ablations = [candidate for candidate in candidates if candidate.get("removed_source_id")]
    assert len(ablations) == 2
    assert {candidate["removed_source_id"] for candidate in ablations} == {
        gap["id"], vwap["id"]
    }
