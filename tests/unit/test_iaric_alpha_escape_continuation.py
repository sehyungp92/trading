from __future__ import annotations

import json

import pytest

from backtests.stock.auto.runners.run_iaric_alpha_escape_continuation import (
    SCORE_SPEC,
    _apply_profiles,
    _evidence_synthesis_candidates,
    _gates,
    _hybrid_activation_report,
    _hybrid_entry_candidates,
    _prepare_compatible_cache,
    _promotion_rank_key,
    _score,
    _structural_parent_beam,
    _validation_shortlist,
)
from backtests.stock.auto.runners.run_iaric_escape_round3 import _signature


def _row(name: str, trades: int, total_r: float, *, mutations: dict | None = None) -> dict:
    return {
        "id": name,
        "mutations": mutations or {"variant": name},
        "metrics": {
            "total_trades": trades,
            "expected_total_r": total_r,
            "avg_r": total_r / max(trades, 1),
            "profit_factor": 1.7,
            "max_drawdown_pct": 0.035,
            "robust_avg_r": 0.10,
        },
        "aperture": {
            "routes": {
                "APERTURE_ONE_ENTRY": {"trades": 5, "total_r": 1.0},
                "APERTURE_TWO_ENTRY": {"trades": 5, "total_r": 1.0},
                "APERTURE_THREE_ENTRY": {"trades": 5, "total_r": 1.0},
            }
        },
    }


def test_alpha_escape_score_is_exactly_seven_and_frequency_return_led() -> None:
    assert len(SCORE_SPEC) == 7
    assert sum(spec["weight"] for spec in SCORE_SPEC.values()) == pytest.approx(1.0)
    assert SCORE_SPEC["absolute_total_r"]["weight"] + SCORE_SPEC["absolute_trades"]["weight"] == pytest.approx(0.56)
    broad_score = _score(_row("broad", 190, 50.0))[0]
    narrow_score = _score(_row("narrow", 149, 47.1))[0]
    assert broad_score > narrow_score


def test_positive_route_profiles_compose_without_erasing_existing_policies() -> None:
    base = {
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_families": "FAILED_BREAKDOWN_RECLAIM,UPTREND_PULLBACK_RECLAIM",
        "param_overrides.pb_aperture_family_filters": "FAILED_BREAKDOWN_RECLAIM:geometry",
    }
    result = _apply_profiles(base, ("mhl", "pdl", "vwap_next"), candidate_size=0.55)
    families = set(result["param_overrides.pb_aperture_families"].split(","))
    assert {
        "FAILED_BREAKDOWN_RECLAIM",
        "UPTREND_PULLBACK_RECLAIM",
        "MULTIDAY_HIGHER_LOW_RECLAIM",
        "PRIOR_DAY_LOW_RECLAIM",
        "VWAP_DEVIATION_RECLAIM",
    } <= families
    assert "FAILED_BREAKDOWN_RECLAIM:geometry" in result["param_overrides.pb_aperture_family_filters"]
    assert "MULTIDAY_HIGHER_LOW_RECLAIM:65" in result["param_overrides.pb_aperture_family_score_floors"]
    assert "VWAP_DEVIATION_RECLAIM:75" in result["param_overrides.pb_aperture_family_score_floors"]


def test_cache_reuse_requires_absence_of_quality_hybrid_policies(tmp_path) -> None:
    research = tmp_path / "research"
    output = tmp_path / "output"
    research.mkdir()
    output.mkdir()
    old_code = "old-code"
    key = f"source|{old_code}|2024-03-25|2026-03-01|candidate"
    payload = {
        "source_fingerprint": "source",
        "code_fingerprint": old_code,
        "evaluations": {
            key: {
                "mutations": {
                    "param_overrides.pb_aperture_family_filters": "PRIOR_DAY_LOW_RECLAIM:geometry"
                }
            }
        },
    }
    (research / "evaluation_cache.json").write_text(json.dumps(payload), encoding="utf-8")
    _prepare_compatible_cache(
        research,
        output,
        source_fingerprint="source",
        code_fingerprint="new-code",
    )
    migrated = json.loads((output / "evaluation_cache.json").read_text(encoding="utf-8"))
    assert list(migrated["evaluations"]) == [
        "source|new-code|2024-03-25|2026-03-01|candidate"
    ]

    (output / "evaluation_cache.json").unlink()
    payload["evaluations"][key]["mutations"].update({
        "param_overrides.pb_aperture_family_transitions": (
            "VWAP_DEVIATION_RECLAIM:quality_hybrid"
        ),
        "param_overrides.pb_aperture_family_hybrid_next_policies": (
            "VWAP_DEVIATION_RECLAIM:deep_reclaim"
        ),
    })
    (research / "evaluation_cache.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="quality-hybrid"):
        _prepare_compatible_cache(
            research,
            output,
            source_fingerprint="source",
            code_fingerprint="third-code",
        )


def test_hybrid_catalog_bridges_both_vwap_parents_without_cartesian_sweep() -> None:
    base = {
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_families": (
            "FAILED_BREAKDOWN_RECLAIM,GAP_FILL_RECLAIM,MULTIDAY_HIGHER_LOW_RECLAIM,"
            "PRIOR_DAY_LOW_RECLAIM,UPTREND_PULLBACK_RECLAIM,VWAP_DEVIATION_RECLAIM"
        ),
        "param_overrides.pb_aperture_family_transitions": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:next_bar,"
            "PRIOR_DAY_LOW_RECLAIM:next_bar,"
            "VWAP_DEVIATION_RECLAIM:confirm"
        ),
    }
    confirm = _row(
        "improved133__union__mhl_pdl_vwap_confirm_gap",
        159,
        50.2,
        mutations=base,
    )
    next_bar = _row(
        "improved133__union__mhl_pdl_vwap_next_gap",
        174,
        49.3,
        mutations={
            **base,
            "param_overrides.pb_aperture_family_transitions": (
                "MULTIDAY_HIGHER_LOW_RECLAIM:next_bar,"
                "PRIOR_DAY_LOW_RECLAIM:next_bar,"
                "VWAP_DEVIATION_RECLAIM:next_bar"
            ),
        },
    )
    candidates = _hybrid_entry_candidates([confirm, next_bar])
    assert len(candidates) == 3
    policies = {
        row["mutations"]["param_overrides.pb_aperture_family_hybrid_next_policies"]
        for row in candidates
    }
    assert policies == {
        "VWAP_DEVIATION_RECLAIM:deep_reclaim",
        "VWAP_DEVIATION_RECLAIM:residual_reclaim",
        "VWAP_DEVIATION_RECLAIM:room_reclaim",
    }
    assert all(
        "VWAP_DEVIATION_RECLAIM:quality_hybrid"
        in row["mutations"]["param_overrides.pb_aperture_family_transitions"]
        for row in candidates
    )


def test_validation_shortlist_excludes_duplicate_fold_control() -> None:
    improved = _row("improved", 133, 43.4, mutations={"variant": "improved"})
    current = _row("current", 149, 47.0, mutations={"variant": "current"})
    broad = _row("broad", 185, 48.0, mutations={"variant": "broad"})
    for row, score in ((improved, 0.99), (current, 0.80), (broad, 0.90)):
        row["alpha_escape_score"] = score
    shortlist = _validation_shortlist([improved, current, broad], current, improved)
    signatures = {_signature(row["mutations"]) for row in shortlist}
    assert _signature(improved["mutations"]) not in signatures
    assert _signature(current["mutations"]) in signatures
    assert _signature(broad["mutations"]) in signatures


def test_promotion_ranking_prefers_created_value_over_unproductive_breadth() -> None:
    value = _row("value159", 159, 50.206)
    value["metrics"].update({"profit_factor": 1.810, "max_drawdown_pct": 0.02987})
    breadth = _row("breadth174", 174, 49.275)
    breadth["metrics"].update({"profit_factor": 1.767, "max_drawdown_pct": 0.03043})
    for row, score in ((value, 0.6187), (breadth, 0.6365)):
        row["alpha_escape_score"] = score
        row["all_gates_pass"] = True

    assert _promotion_rank_key(value) > _promotion_rank_key(breadth)


def test_user_requested_fold_skip_removes_fold_claims_but_preserves_holdout_gate() -> None:
    improved = _row("improved", 133, 43.4)
    current = _row("current", 149, 47.0)
    candidate = _row("candidate", 185, 50.0)
    candidate["validation_contract"] = {
        "passed": None,
        "fold_validation_performed": False,
        "fold_validation_status": "skipped_by_user_request",
        "holdout_accessed": False,
    }

    gates = _gates(
        candidate,
        current,
        improved,
        fold_validation_enabled=False,
    )

    assert gates["sealed_holdout_excluded"] is True
    assert gates["fold_validation_skipped_by_user_request"] is True
    assert "fold_integrity" not in gates
    assert "chronological_consistency" not in gates
    assert gates["material_frequency_escape"] is True
    assert "frequency_escape_180" not in gates


def test_hybrid_activation_screen_rejects_endpoint_only_policy_masks() -> None:
    parent = _row("parent", 3, 0.3)
    parent["trade_attribution"] = [
        {
            "symbol": f"S{index}",
            "entry_time": f"2025-01-0{index + 1}T10:00:00",
            "route": "APERTURE_VWAP_DEVIATION_RECLAIM_ENTRY",
            "r": 0.1,
            "score_components": {
                "dislocation": 1.0,
                "reclaim": 1.0,
                "close_quality": 0.9,
                "relative_volume": 0.5,
                "residual_dislocation": 0.0,
                "prior_down_sequence": 0.5,
                "reversion_room": 1.0,
            },
        }
        for index in range(3)
    ]

    report = _hybrid_activation_report(parent, "VWAP_DEVIATION_RECLAIM")

    assert report["policies"]["deep_reclaim"]["passed"] == 3
    assert report["policies"]["room_reclaim"]["passed"] == 3
    assert report["policies"]["residual_reclaim"]["passed"] == 0
    assert report["policies"]["deep_reclaim"]["non_degenerate"] is False
    assert report["residual_component_available"] is False
    assert ["deep_reclaim", "room_reclaim"] in report["duplicate_policy_groups"]


def test_evidence_synthesis_uses_positive_gap_sleeve_and_quality_parent() -> None:
    core_ids = (
        "improved133__union__mhl_pdl_vwap_next_gap",
        "improved133__union__mhl_pdl_vwap_confirm_gap",
        "improved133__union__mhl_pdl_gap",
        "course164__union__mhl_pdl_vwap_next",
    )
    core = [
        _row(
            candidate_id,
            trades,
            total_r,
            mutations={
                "variant": candidate_id,
                "param_overrides.pb_aperture_enabled": True,
                "param_overrides.pb_aperture_families": (
                    "FAILED_BREAKDOWN_RECLAIM,UPTREND_PULLBACK_RECLAIM"
                ),
            },
        )
        for candidate_id, trades, total_r in zip(
            core_ids, (174, 159, 160, 196), (49.3, 50.2, 49.8, 42.5), strict=True
        )
    ]
    repair = _row(
        "course164__repair__gap_exhaustion_reclaim__floor_75",
        139,
        47.0,
        mutations={
            "param_overrides.pb_aperture_enabled": True,
            "param_overrides.pb_aperture_families": (
                "FAILED_BREAKDOWN_RECLAIM,GAP_EXHAUSTION_RECLAIM,UPTREND_PULLBACK_RECLAIM"
            ),
        },
    )
    repair["trade_attribution"] = [
        {
            "route": "APERTURE_GAP_EXHAUSTION_RECLAIM_ENTRY",
            "r": value,
        }
        for value in (0.4, 0.3, 0.2, 0.2, 0.1, 0.1)
    ]

    candidates = _evidence_synthesis_candidates(core, [repair])

    assert len(candidates) == 6
    assert candidates[0]["parent_id"] == core_ids[0]
    assert "GAP_EXHAUSTION_RECLAIM:75" in candidates[0]["mutations"][
        "param_overrides.pb_aperture_family_score_floors"
    ]
    assert sum(row["parent_id"] == repair["id"] for row in candidates) == 2


def test_structural_parent_beam_preserves_high_value_sub_150_trade_sleeve() -> None:
    quality = _row("quality139", 139, 52.0)
    breadth = _row("breadth174", 174, 49.0)
    return_leader = _row("return159", 159, 53.0)
    for row, score in ((quality, 0.70), (breadth, 0.72), (return_leader, 0.71)):
        row["alpha_escape_score"] = score

    beam = _structural_parent_beam([quality, breadth, return_leader], 3)

    assert quality in beam
    assert breadth in beam
    assert return_leader in beam
