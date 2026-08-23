from __future__ import annotations

import json

import pytest

from backtests.stock.auto.runners.prepare_iaric_structural_challenger import (
    FAMILIES,
    _catalog,
)
from backtests.stock.auto.runners.run_iaric_structural_challenger import (
    _activation_rescue_candidates,
    _activation_rescue_followup_candidates,
    _conditional_interaction_candidates,
    _entry_candidates,
    _entry_interaction_candidates,
    _execution_screen_score_metrics,
    _lean_management_candidates,
    _management_parent_beam,
    _prepare_cache,
    _primary_interaction_ids,
    _root_candidates,
    _seed_screen_cache_from_full,
    _structural_beam,
)


def _isolation() -> dict[str, dict[str, float | int]]:
    result = {}
    for family in FAMILIES:
        result[family] = {
            "escape_score": 0.4,
            "route_ready": 30,
            "route_trades": 30,
            "route_total_r": -2.0,
            "route_profit_factor": 0.8,
        }
    for index, family in enumerate(
        ("UPTREND_PULLBACK_RECLAIM", "GAP_EXHAUSTION_RECLAIM", "FAILED_BREAKDOWN_RECLAIM")
    ):
        result[family].update(
            escape_score=0.9 - index * 0.05,
            route_trades=20,
            route_total_r=5.0,
            route_profit_factor=1.5,
        )
    for family in ("GAP_FILL_RECLAIM", "PRIOR_DAY_LOW_RECLAIM", "MULTIDAY_HIGHER_LOW_RECLAIM"):
        result[family].update(route_ready=2, route_trades=2)
    return result


def _overlap() -> dict[str, object]:
    matrix = {
        left: {right: (1.0 if left == right else 0.01) for right in FAMILIES}
        for left in FAMILIES
    }
    return {"symbol_day_jaccard": matrix}


def _row(name: str, families: list[str], score: float, trades: int = 130) -> dict[str, object]:
    return {
        "id": name,
        "families": families,
        "focus_families": families,
        "mutations": {"variant": name},
        "escape_score": score,
        "metrics": {
            "avg_r": 0.25,
            "profit_factor": 1.6,
            "max_drawdown_pct": 0.03,
            "expected_total_r": 35.0,
            "total_trades": trades,
        },
        "aperture": {"trades": 40, "total_r": 10.0},
    }


def test_catalog_retains_every_family_and_balances_weak_pair_roles() -> None:
    isolation = _isolation()
    strong = ["UPTREND_PULLBACK_RECLAIM", "GAP_EXHAUSTION_RECLAIM", "FAILED_BREAKDOWN_RECLAIM"]

    catalog = _catalog(isolation, strong, _overlap())

    singles = {
        tuple(row["families"])[0]
        for row in catalog["root_candidates"]
        if len(row["families"]) == 1
    }
    assert singles == set(FAMILIES)
    assert len(catalog["weak_orthogonal_pairs"]) == 8
    assert catalog["generation_policy"]["weak_interaction_quotas"] == {
        "dormant_dormant": 2,
        "dormant_high_supply": 3,
        "high_supply_high_supply": 3,
    }
    dormant = {"GAP_FILL_RECLAIM", "PRIOR_DAY_LOW_RECLAIM", "MULTIDAY_HIGHER_LOW_RECLAIM"}
    bucket_counts = {"dormant_dormant": 0, "dormant_high_supply": 0, "high_supply_high_supply": 0}
    for left, right in catalog["weak_orthogonal_pairs"]:
        count = int(left in dormant) + int(right in dormant)
        bucket_counts[
            "dormant_dormant" if count == 2
            else "dormant_high_supply" if count == 1
            else "high_supply_high_supply"
        ] += 1
    assert bucket_counts == {
        "dormant_dormant": 2,
        "dormant_high_supply": 3,
        "high_supply_high_supply": 3,
    }


def test_structural_beam_preserves_weak_family_before_optional_axes() -> None:
    control = _row("control", [], 0.5, 89)
    control["metrics"]["max_drawdown_pct"] = 0.06
    course = _row("course", ["UPTREND_PULLBACK_RECLAIM"], 0.95)
    course["mandatory_course_control"] = True
    weak = ["GAP_FILL_RECLAIM", "VOLUME_CLIMAX_RECLAIM"]
    rows = [
        course,
        _row("gap", ["UPTREND_PULLBACK_RECLAIM", weak[0]], 0.80),
        _row("volume", ["UPTREND_PULLBACK_RECLAIM", weak[1]], 0.79, 170),
    ]

    selected, reasons = _structural_beam(rows, control, weak, soft_limit=3)

    assert {row["id"] for row in selected} == {"course", "gap", "volume"}
    assert any("best_weak_family:GAP_FILL_RECLAIM" in values for values in reasons.values())
    assert any("best_weak_family:VOLUME_CLIMAX_RECLAIM" in values for values in reasons.values())


def test_positive_focus_beam_rejects_losing_family_hidden_by_good_composition() -> None:
    control = _row("control", [], 0.5, 133)
    control["metrics"]["max_drawdown_pct"] = 0.06
    start = _row("start", ["FAILED_BREAKDOWN_RECLAIM"], 0.9, 133)
    start["mandatory_improved_start"] = True
    course = _row("course", ["UPTREND_PULLBACK_RECLAIM"], 0.8, 120)
    course["mandatory_course_control"] = True
    positive = _row("positive", ["GAP_FILL_RECLAIM"], 0.7, 150)
    positive["trade_attribution"] = [
        {"route": "APERTURE_GAP_FILL_RECLAIM_ENTRY", "r": 0.2}
        for _ in range(3)
    ]
    negative = _row("negative", ["VOLUME_CLIMAX_RECLAIM"], 0.99, 200)
    negative["trade_attribution"] = [
        {"route": "APERTURE_VOLUME_CLIMAX_RECLAIM_ENTRY", "r": -0.2}
        for _ in range(3)
    ]

    selected, _ = _structural_beam(
        [start, course, positive, negative],
        control,
        ["GAP_FILL_RECLAIM", "VOLUME_CLIMAX_RECLAIM"],
        soft_limit=6,
        require_focus_positive=True,
        family_limit=4,
    )

    assert {row["id"] for row in selected} == {"start", "course", "positive"}


def test_management_parent_beam_is_diverse_and_capped_at_four() -> None:
    rows = [_row(f"parent_{index}", ["GAP_FILL_RECLAIM"], 0.9 - index * 0.01, 140 + index)
            for index in range(7)]
    selected, reasons = _management_parent_beam(rows, limit=4)

    assert len(selected) == 4
    assert reasons


def test_improved_133_trade_composition_is_mandatory_start_not_just_generator() -> None:
    catalog = _catalog(
        _isolation(),
        ["UPTREND_PULLBACK_RECLAIM", "GAP_EXHAUSTION_RECLAIM", "FAILED_BREAKDOWN_RECLAIM"],
        _overlap(),
    )
    roots = _root_candidates(
        catalog,
        {"baseline": "incumbent"},
        {"mutations": {"course": "reference"}, "families": ["COURSE"]},
    )
    start = next(row for row in roots if row["id"] == "improved_start_control")

    assert start["mandatory_improved_start"] is True
    assert start["families"] == catalog["current_leader_pair"]
    assert start["mutations"]["param_overrides.pb_aperture_families"] == ",".join(
        sorted(catalog["current_leader_pair"])
    )

    start.update(_row("improved", start["families"], 0.85))
    start["id"] = "improved_start_control"
    start["mandatory_improved_start"] = True
    entry = _entry_candidates([start])
    assert len(entry) == 1


def test_entry_and_management_followups_are_focus_scoped_and_lean() -> None:
    parent = _row(
        "parent",
        ["UPTREND_PULLBACK_RECLAIM", "VOLUME_CLIMAX_RECLAIM"],
        0.8,
    )
    parent["focus_families"] = ["VOLUME_CLIMAX_RECLAIM"]
    parent["mutations"] = {
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_families": "UPTREND_PULLBACK_RECLAIM,VOLUME_CLIMAX_RECLAIM",
    }

    entry = _entry_candidates([parent])
    management = _lean_management_candidates([parent])

    assert len(entry) == 3  # control, one role-aware floor, one causal transition
    assert all(
        "uptrend_pullback_reclaim_confirm" not in row["id"]
        for row in entry
    )
    assert all(
        "param_overrides.pb_aperture_event_score_min" not in row["mutations"]
        for row in entry
    )
    assert any(
        row["mutations"].get("param_overrides.pb_aperture_family_score_floors")
        == "VOLUME_CLIMAX_RECLAIM:65"
        for row in entry
    )
    assert len(management) == 3
    assert {row["id"].rsplit("__", 1)[-1] for row in management} == {
        "management_control",
        "size70",
        "stale4",
    }


def test_entry_interactions_require_measured_floor_activation() -> None:
    family = "VOLUME_CLIMAX_RECLAIM"
    parent = _row("parent", ["UPTREND_PULLBACK_RECLAIM", family], 0.8)
    parent["focus_families"] = [family]
    parent["root_id"] = "parent"
    parent["mutations"] = {
        "param_overrides.pb_aperture_families": f"UPTREND_PULLBACK_RECLAIM,{family}"
    }
    activation = {family: {"role": "high_supply_negative_standalone", "atlas": {}}}
    atomic = _entry_candidates([parent], activation)
    control = next(row for row in atomic if not row.get("policy_atom"))
    floor75 = next(row for row in atomic if row.get("policy_atom") == "family_floor:75")
    transition = next(
        row for row in atomic
        if str(row.get("policy_atom", "")).startswith("transition:")
    )
    control["trade_attribution"] = [
        {"route": f"APERTURE_{family}_ENTRY", "r": 0.2}
        for _ in range(3)
    ]
    floor75["trade_attribution"] = list(control["trade_attribution"])
    transition["trade_attribution"] = list(control["trade_attribution"])

    assert _entry_interaction_candidates([parent], activation, atomic) == []

    floor75["trade_attribution"].append(
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.1}
    )
    transition["trade_attribution"].append(
        {"route": f"APERTURE_{family}_ENTRY", "r": 0.1}
    )
    interactions = _entry_interaction_candidates([parent], activation, atomic)

    assert len(interactions) == 1
    assert interactions[0]["activation_evidence"]["floor75_focus_trades"] == 4


def test_secondary_context_uses_only_policy_needed_for_positive_activation() -> None:
    family = "GAP_FILL_RECLAIM"
    root = _row("secondary", ["GAP_EXHAUSTION_RECLAIM", family], 0.7)
    root["focus_families"] = [family]
    root["mutations"] = {"composition": "secondary"}
    evidence = _row("rescued", ["FAILED_BREAKDOWN_RECLAIM", family], 0.8)
    evidence["focus_families"] = [family]
    evidence["mutations"] = {
        "composition": "primary",
        "param_overrides.pb_aperture_family_score_floors": f"{family}:65",
    }
    evidence["trade_attribution"] = [
        {"route": f"APERTURE_{family}_ENTRY", "r": 0.2}
        for _ in range(3)
    ]

    candidates = _conditional_interaction_candidates(
        [root],
        set(),
        [evidence],
        [family],
    )

    assert len(candidates) == 1
    assert candidates[0]["mutations"] == {
        "composition": "secondary",
        "param_overrides.pb_aperture_family_score_floors": f"{family}:65",
    }


def test_primary_interactions_cover_every_weak_family_and_weak_pair_class() -> None:
    catalog = _catalog(
        _isolation(),
        ["UPTREND_PULLBACK_RECLAIM", "GAP_EXHAUSTION_RECLAIM", "FAILED_BREAKDOWN_RECLAIM"],
        _overlap(),
    )
    selected = _primary_interaction_ids(catalog, _overlap())
    selected_rows = [row for row in catalog["root_candidates"] if row["id"] in selected]

    for family in catalog["weak_or_dormant_families"]:
        assert any(family in row.get("focus_families", []) for row in selected_rows)
        assert any(
            family in row.get("focus_families", [])
            and "leader_pair_plus_orthogonal_family" in row.get("sources", [])
            for row in selected_rows
        )
    assert sum(
        "weak_alone_orthogonal_pair" in row.get("sources", [])
        for row in selected_rows
    ) == len(catalog["weak_orthogonal_pairs"])


def test_negative_high_supply_family_gets_fixed_rescue_filters_and_caps() -> None:
    family = "VOLUME_CLIMAX_RECLAIM"
    parent = _row(
        "leader_plus_volume",
        ["UPTREND_PULLBACK_RECLAIM", family],
        0.7,
        160,
    )
    parent["focus_families"] = [family]
    parent["mutations"] = {
        "param_overrides.pb_aperture_families": f"UPTREND_PULLBACK_RECLAIM,{family}"
    }
    parent["trade_attribution"] = [
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.5},
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.4},
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.3},
    ]
    activation = {
        family: {
            "role": "high_supply_negative_standalone",
            "atlas": {"transitions": {}},
        }
    }

    candidates = _activation_rescue_candidates([parent], activation, [family])

    assert len(candidates) == 4
    mutations = [row["mutations"] for row in candidates]
    assert any(
        row.get("param_overrides.pb_aperture_family_score_floors")
        == f"{family}:75"
        for row in mutations
    )
    assert {
        row.get("param_overrides.pb_aperture_family_filters")
        for row in mutations
        if "param_overrides.pb_aperture_family_filters" in row
    } == {f"{family}:geometry", f"{family}:participation"}
    assert {
        row.get("param_overrides.pb_aperture_family_daily_caps")
        for row in mutations
        if "param_overrides.pb_aperture_family_daily_caps" in row
    } == {f"{family}:1"}


def test_rescue_followup_requires_atom_improvement_and_opens_only_best_joint() -> None:
    family = "VOLUME_CLIMAX_RECLAIM"
    parent = _row("leader_plus_volume", ["UPTREND_PULLBACK_RECLAIM", family], 0.7)
    parent["focus_families"] = [family]
    parent["mutations"] = {"composition": "primary"}
    parent["trade_attribution"] = [
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.3}
        for _ in range(3)
    ]
    activation = {family: {"role": "high_supply_negative_standalone", "atlas": {}}}
    atoms = _activation_rescue_candidates([parent], activation, [family])
    for row in atoms:
        row["trade_attribution"] = list(parent["trade_attribution"])
    best_filter = next(row for row in atoms if row.get("rescue_atom") == "filter:geometry")
    best_filter["trade_attribution"] = [
        {"route": f"APERTURE_{family}_ENTRY", "r": -0.1}
        for _ in range(3)
    ]

    followups = _activation_rescue_followup_candidates(
        [parent], atoms, activation, [family]
    )

    assert len(followups) == 1
    assert followups[0]["rescue_atom"] == "floor75+best_filter"


def test_cache_migration_rekeys_only_policy_absent_legacy_evaluations(tmp_path) -> None:
    output = tmp_path / "structural"
    output.mkdir()
    old_code = "old-code"
    key = f"source|{old_code}|2024-03-25|2026-03-01|candidate"
    payload = {
        "source_fingerprint": "source",
        "code_fingerprint": old_code,
        "evaluations": {
            key: {"mutations": {"param_overrides.pb_aperture_enabled": True}}
        },
    }
    (output / "evaluation_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    _prepare_cache(
        tmp_path,
        output,
        source_fingerprint="source",
        code_fingerprint="new-code",
    )

    migrated = json.loads((output / "evaluation_cache.json").read_text(encoding="utf-8"))
    assert migrated["code_fingerprint"] == "new-code"
    assert list(migrated["evaluations"]) == [
        "source|new-code|2024-03-25|2026-03-01|candidate"
    ]
    assert (output / "evaluation_cache.pre_family_policy_extension.json").exists()

    policy_key = "param_overrides.pb_aperture_family_filters"
    migrated["code_fingerprint"] = "new-code"
    migrated["evaluations"][
        "source|new-code|2024-03-25|2026-03-01|candidate"
    ]["mutations"][policy_key] = "VOLUME_CLIMAX_RECLAIM:geometry"
    (output / "evaluation_cache.json").write_text(json.dumps(migrated), encoding="utf-8")
    with pytest.raises(ValueError, match="family-policy evaluations"):
        _prepare_cache(
            tmp_path,
            output,
            source_fingerprint="source",
            code_fingerprint="third-code",
        )


def test_full_diagnostics_cache_is_reused_only_after_execution_parity(tmp_path) -> None:
    def record(signature: str, trades: int) -> dict[str, object]:
        return {
            "signature": signature,
            "metrics": {
                "total_trades": float(trades),
                "expected_total_r": 4.0,
                "profit_factor": 1.5,
                "max_drawdown_pct": 0.03,
                "avg_r": 0.2,
            },
            "trade_attribution": [
                {"entry_time": "2025-01-02T15:00:00+00:00", "r": 0.2}
                for _ in range(trades)
            ],
            "error": "",
        }

    output = tmp_path
    full = {
        "source_fingerprint": "source",
        "code_fingerprint": "code",
        "evaluations": {"shared": record("shared", 2), "reusable": record("reusable", 3)},
    }
    screen = {
        "source_fingerprint": "source",
        "code_fingerprint": "code",
        "evaluations": {"shared": record("shared", 2)},
    }
    (output / "evaluation_cache.json").write_text(json.dumps(full), encoding="utf-8")
    (output / "structural_screen_cache.json").write_text(json.dumps(screen), encoding="utf-8")

    report = _seed_screen_cache_from_full(output)
    reused = json.loads((output / "structural_screen_cache.json").read_text(encoding="utf-8"))

    assert report["verified_overlap"] == 1
    assert report["reused"] == 1
    assert reused["evaluations"]["reusable"]["screen_cache_provenance"] == (
        "full_diagnostics_execution_superset"
    )
    normalized = _execution_screen_score_metrics(reused["evaluations"]["reusable"])
    assert normalized["entry_realized_discrimination_lift_r"] == 0.2
    assert normalized["robust_avg_r"] == pytest.approx(0.2)

    screen["evaluations"]["shared"]["metrics"]["total_trades"] = 99.0
    (output / "structural_screen_cache.json").write_text(json.dumps(screen), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Diagnostics changed execution"):
        _seed_screen_cache_from_full(output)
