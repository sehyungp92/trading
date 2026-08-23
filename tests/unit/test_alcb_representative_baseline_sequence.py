from __future__ import annotations

from backtests.stock.auto.alcb.run_baseline_recovery import _signature
from backtests.stock.auto.alcb.run_representative_baseline_sequence import (
    ROUND2_CONFIG,
    ROUND3_CONFIG,
    _catalog,
    _challenger_gate,
    _is_ablation_evidence,
    _normalized_config,
)
from backtests.stock.auto.alcb.run_trail_combination_validation import (
    _catalog as _trail_catalog,
    _interaction_delta,
)
from backtests.stock.auto.alcb.run_round3_lineage_rebuild import (
    ACTIVE_ROUND3_PATCH,
    ARCHIVED_ROUND3_PATCH,
    _catalog as _round3_rebuild_catalog,
    _enrich as _round3_rebuild_enrich,
    _full_incremental_gate,
    _metadata_by_signature as _round3_rebuild_metadata,
    _gate_snapshot,
)
from backtests.scripts.alcb_establish_rebuilt_round3 import (
    _live_settings_mismatches,
    _normalize_materialization_metrics,
)


def _configs():
    return _normalized_config(ROUND2_CONFIG), _normalized_config(ROUND3_CONFIG)


def test_catalog_preserves_round2_and_marks_round3_contaminated() -> None:
    round2, round3 = _configs()
    catalog = _catalog(round2, round3)
    by_id = {row["id"]: row for row in catalog}

    assert by_id["control__round2_exact"]["selection_eligible"] is True
    assert by_id["control__round2_exact"]["changed_keys"] == []
    diagnostic = by_id["diagnostic__current_round3"]
    expected_changes = sorted(
        key
        for key in set(round2) | set(round3)
        if round2.get(key) != round3.get(key)
    )
    assert diagnostic["selection_eligible"] is False
    assert diagnostic["changed_keys"] == expected_changes
    assert "param_overrides.rvol_threshold" in expected_changes


def test_catalog_is_signature_unique_and_bounded() -> None:
    round2, round3 = _configs()
    catalog = _catalog(round2, round3)

    assert 20 <= len(catalog) <= 40
    assert len({_signature(row["mutations"]) for row in catalog}) == len(catalog)


def test_ablation_import_contains_only_is_fields() -> None:
    evidence = _is_ablation_evidence()

    assert evidence["literal_mutation_count"] == 50
    assert evidence["is_ablation_result_count"] >= 50
    assert all(
        "oos" not in str(key).lower()
        for row in evidence["results"]
        for key in row
    )


def test_contaminated_candidate_cannot_clear_challenger_gate() -> None:
    metrics = {
        "expected_total_r": 100.0,
        "net_profit": 10_000.0,
        "avg_r": 0.2,
        "profit_factor": 2.0,
        "trades_per_month": 50.0,
        "max_drawdown_pct": 0.02,
    }
    costs = {
        "7.5": {"expected_total_r": 90.0, "profit_factor": 1.8},
        "10.0": {"expected_total_r": 80.0, "profit_factor": 1.6},
        "seven_five_gate": True,
        "ten_gate": True,
    }
    validation = {"robust_eligible": True}
    control = {"metrics": metrics, "costs": costs}
    candidate = {
        "selection_eligible": False,
        "metrics": metrics,
        "costs": costs,
        "validation": validation,
    }

    eligible, reasons = _challenger_gate(candidate, control)

    assert eligible is False
    assert "candidate is diagnostic-only or OOS-contaminated" in reasons


def test_trail_combination_changes_only_requested_settings() -> None:
    catalog = {row["id"]: row for row in _trail_catalog()}
    control = catalog["control__round2_exact"]["mutations"]
    combined = catalog["combined__activation_0p18_timing_30"]["mutations"]
    changed = {key for key in set(control) | set(combined) if control.get(key) != combined.get(key)}

    assert changed == {
        "param_overrides.adaptive_trail_late_activate_r",
        "param_overrides.adaptive_trail_start_bars",
        "param_overrides.adaptive_trail_tighten_bars",
    }
    assert combined["param_overrides.adaptive_trail_late_activate_r"] == 0.18
    assert combined["param_overrides.adaptive_trail_start_bars"] == 30
    assert combined["param_overrides.adaptive_trail_tighten_bars"] == 30


def test_interaction_delta_distinguishes_additive_and_overlapping_effects() -> None:
    control = {"expected_total_r": 100.0}
    activation = {"expected_total_r": 110.0}
    timing = {"expected_total_r": 120.0}
    additive = {"expected_total_r": 130.0}
    overlapping = {"expected_total_r": 122.0}

    assert _interaction_delta(additive, activation, timing, control)["expected_total_r"] == 0.0
    assert _interaction_delta(overlapping, activation, timing, control)["expected_total_r"] == -8.0


def test_round3_rebuild_catalog_uses_combined_trail_as_immutable_control() -> None:
    rows = _round3_rebuild_catalog()
    by_id = {row["id"]: row for row in rows}
    baseline = by_id["baseline__combined_trail"]["mutations"]

    assert by_id["baseline__combined_trail"]["patch"] == {}
    assert by_id["atomic__rvol_1p1"]["patch"] == ACTIVE_ROUND3_PATCH
    assert by_id["bundle__full_archived_lineage"]["patch"] == ARCHIVED_ROUND3_PATCH
    assert by_id["surface__rvol_1p2"]["selection_eligible"] is False
    assert by_id["surface__rvol_1p3"]["selection_eligible"] is False
    assert len({_signature(row["mutations"]) for row in rows}) == len(rows)
    assert all(
        all(
            row["mutations"].get(key) == value
            for key, value in baseline.items()
            if key not in row["patch"]
        )
        for row in rows
    )


def test_round3_full_gate_rejects_return_bought_with_quality_or_drawdown() -> None:
    baseline = {
        "metrics": {
            "expected_total_r": 200.0,
            "avg_r": 0.14,
            "profit_factor": 1.80,
            "max_drawdown_pct": 0.04,
        }
    }
    healthy = {
        "metrics": {
            "expected_total_r": 215.0,
            "avg_r": 0.135,
            "profit_factor": 1.76,
            "max_drawdown_pct": 0.043,
        }
    }
    inflated = {
        "metrics": {
            "expected_total_r": 250.0,
            "avg_r": 0.12,
            "profit_factor": 1.65,
            "max_drawdown_pct": 0.055,
        }
    }

    assert _full_incremental_gate(healthy, baseline) == (True, [])
    passes, reasons = _full_incremental_gate(inflated, baseline)
    assert passes is False
    assert "retained less than 95% of baseline AvgR" in reasons
    assert "retained less than 97% of baseline PF" in reasons
    assert "drawdown exceeded the fixed relative cap" in reasons


def test_round3_rebuild_restores_selection_metadata_after_cached_evaluation() -> None:
    catalog = _round3_rebuild_catalog()
    target = next(row for row in catalog if row["id"] == "atomic__rvol_1p1")
    cached_row = {
        "id": target["id"],
        "family": target["family"],
        "era": target["era"],
        "sources": target["sources"],
        "signature": _signature(target["mutations"]),
        "mutations": target["mutations"],
        "metrics": {},
    }

    enriched = _round3_rebuild_enrich(
        [cached_row], _round3_rebuild_metadata(catalog)
    )[0]

    assert enriched["patch"] == ACTIVE_ROUND3_PATCH
    assert enriched["selection_eligible"] is True
    assert enriched["changed_keys"] == ["param_overrides.rvol_threshold"]


def test_round3_gate_snapshot_compares_incrementally_to_supplied_control() -> None:
    control = {
        "id": "control",
        "metrics": {
            "expected_total_r": 200.0,
            "net_profit": 20_000.0,
            "avg_r": 0.14,
            "profit_factor": 1.80,
            "win_rate": 0.57,
            "trades_per_month": 60.0,
            "max_drawdown_pct": 0.04,
        },
        "validation": {
            "robust_eligible": True,
            "folds": [
                {"fold": f"fold_{index}", "expected_total_r": 40.0}
                for index in range(1, 5)
            ],
        },
        "costs": {
            "7.5": {"expected_total_r": 130.0},
            "10.0": {"expected_total_r": 60.0},
            "seven_five_gate": True,
            "ten_gate": True,
        },
    }
    candidate = {
        "id": "candidate",
        "metrics": {
            **control["metrics"],
            "expected_total_r": 220.0,
            "net_profit": 22_000.0,
        },
        "validation": {
            "robust_eligible": True,
            "folds": [
                {"fold": f"fold_{index}", "expected_total_r": 45.0}
                for index in range(1, 5)
            ],
        },
        "costs": {
            "7.5": {"expected_total_r": 140.0},
            "10.0": {"expected_total_r": 70.0},
            "seven_five_gate": True,
            "ten_gate": True,
        },
    }

    assessment = _gate_snapshot(candidate, control)

    assert assessment["passes"] is True
    assert assessment["control"] == "control"
    assert assessment["fold_wins_vs_control"] == 4
    assert assessment["delta_vs_control"]["expected_total_r"] == 20.0


def test_round3_materialization_aliases_expectancy_as_avg_r() -> None:
    original = {"expectancy": 0.125, "expected_total_r": 25.0}

    normalized = _normalize_materialization_metrics(original)

    assert normalized["avg_r"] == 0.125
    assert original == {"expectancy": 0.125, "expected_total_r": 25.0}


def test_rebuilt_round3_declares_live_setting_mismatches() -> None:
    _, round3 = _configs()

    mismatches = _live_settings_mismatches(round3)

    assert set(mismatches) == {
        "adaptive_trail_late_activate_r",
        "adaptive_trail_start_bars",
        "adaptive_trail_tighten_bars",
        "entry_window_end",
        "late_entry_cutoff",
        "late_entry_score_min",
        "rvol_threshold",
    }
