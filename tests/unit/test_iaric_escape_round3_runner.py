from __future__ import annotations

from types import SimpleNamespace

import pytest

from backtests.stock.auto.runners.run_iaric_escape_round3 import (
    SCORE_SPEC,
    _aperture_expansion_candidates,
    _composition_center_candidates,
    _fold_validate,
    _quality_entry_candidates,
)
from strategies.stock.iaric.core.logic import (
    aperture_event_admitted,
    aperture_hybrid_uses_next_bar,
    aperture_family_daily_cap,
    aperture_family_filter,
    aperture_family_from_route,
    aperture_family_max_bar,
    aperture_family_score_floor,
    aperture_family_transition,
)
from backtests.stock.auto.runners import run_iaric_escape_round3 as escape_runner


def _isolation(family: str) -> dict[str, object]:
    return {"families": [family]}


def test_escape_score_has_exactly_seven_immutable_components() -> None:
    assert len(SCORE_SPEC) == 7
    assert sum(float(spec["weight"]) for spec in SCORE_SPEC.values()) == pytest.approx(1.0)


def test_composition_search_is_center_first_then_selective_expansion() -> None:
    isolations = [
        _isolation("UPTREND_PULLBACK_RECLAIM"),
        _isolation("GAP_EXHAUSTION_RECLAIM"),
        _isolation("FAILED_BREAKDOWN_RECLAIM"),
    ]
    centered = _composition_center_candidates({}, isolations)
    assert len(centered) == 7  # 3 singles + 3 pairs + 1 triple
    assert {
        row["mutations"]["param_overrides.pb_aperture_max_symbols"]
        for row in centered
    } == {120}

    expanded = _aperture_expansion_candidates(centered[:3])
    assert len(expanded) == 9
    assert {
        row["mutations"]["param_overrides.pb_aperture_max_symbols"]
        for row in expanded
    } == {60, 120, 180}


def test_entry_search_only_generates_mechanisms_for_present_families() -> None:
    family = "UPTREND_PULLBACK_RECLAIM"
    parent = {
        "id": "uptrend_parent",
        "families": [family],
        "mutations": {
            "param_overrides.pb_aperture_enabled": True,
            "param_overrides.pb_aperture_families": family,
        },
    }
    candidates = _quality_entry_candidates([parent])
    assert len(candidates) == 5  # control, two floors, confirm, retrace
    assert not any("prior_bar" in row["id"] or "multi_bar" in row["id"] for row in candidates)
    transition_values = {
        row["mutations"].get("param_overrides.pb_aperture_family_transitions", "")
        for row in candidates
    }
    assert f"{family}:confirm" in transition_values
    assert f"{family}:retrace" in transition_values


def test_family_transition_defaults_preserve_existing_live_replay_contract() -> None:
    defaults = SimpleNamespace(
        pb_aperture_family_transitions="",
        pb_aperture_prior_low_transition="retrace",
        pb_aperture_multiday_transition="confirm",
    )
    assert aperture_family_transition(defaults, "UPTREND_PULLBACK_RECLAIM") == "next_bar"
    assert aperture_family_transition(defaults, "PRIOR_DAY_LOW_RECLAIM") == "retrace"
    assert aperture_family_transition(defaults, "MULTIDAY_HIGHER_LOW_RECLAIM") == "confirm"

    mapped = SimpleNamespace(
        pb_aperture_family_transitions=(
            "UPTREND_PULLBACK_RECLAIM:confirm,"
            "FAILED_BREAKDOWN_RECLAIM=retrace"
        )
    )
    assert aperture_family_transition(mapped, "UPTREND_PULLBACK_RECLAIM") == "confirm"
    assert aperture_family_transition(mapped, "FAILED_BREAKDOWN_RECLAIM") == "retrace"


def test_family_transition_rejects_noncausal_or_unknown_mechanisms() -> None:
    settings = SimpleNamespace(
        pb_aperture_family_transitions="UPTREND_PULLBACK_RECLAIM:same_bar"
    )
    with pytest.raises(ValueError, match="next_bar.*confirm.*retrace"):
        aperture_family_transition(settings, "UPTREND_PULLBACK_RECLAIM")


def test_quality_hybrid_routes_causally_from_immutable_signal_components() -> None:
    settings = SimpleNamespace(
        pb_aperture_family_transitions="VWAP_DEVIATION_RECLAIM:quality_hybrid",
        pb_aperture_family_hybrid_next_policies=(
            "VWAP_DEVIATION_RECLAIM:deep_reclaim"
        ),
    )
    assert (
        aperture_family_transition(settings, "VWAP_DEVIATION_RECLAIM")
        == "quality_hybrid"
    )
    event = SimpleNamespace(
        family="VWAP_DEVIATION_RECLAIM",
        score=78.0,
        score_components={
            "dislocation": 0.55,
            "reclaim": 0.35,
            "close_quality": 0.60,
            "relative_volume": 0.20,
            "residual_dislocation": 0.10,
            "prior_down_sequence": 0.25,
            "reversion_room": 0.40,
        },
    )
    assert aperture_hybrid_uses_next_bar(settings, event) is True
    event.score_components["reclaim"] = 0.34
    assert aperture_hybrid_uses_next_bar(settings, event) is False

    settings.pb_aperture_family_hybrid_next_policies = ""
    with pytest.raises(ValueError, match="quality_hybrid requires"):
        aperture_hybrid_uses_next_bar(settings, event)


def test_family_admission_policies_preserve_global_default_and_filter_locally() -> None:
    settings = SimpleNamespace(
        pb_aperture_event_score_min=55.0,
        pb_aperture_family_score_floors="",
        pb_aperture_family_filters="",
        pb_aperture_family_daily_caps="",
    )
    assert aperture_family_score_floor(settings, "OPENING_FLUSH_RECLAIM") == 55.0

    settings.pb_aperture_family_score_floors = "OPENING_FLUSH_RECLAIM:65"
    settings.pb_aperture_family_filters = "OPENING_FLUSH_RECLAIM:geometry"
    event = SimpleNamespace(
        family="OPENING_FLUSH_RECLAIM",
        score=67.0,
        score_components={"reclaim": 0.39, "close_quality": 0.80},
    )
    assert aperture_event_admitted(settings, event) is False
    event.score_components["reclaim"] = 0.40
    assert aperture_event_admitted(settings, event) is True

    settings.pb_aperture_family_daily_caps = "OPENING_FLUSH_RECLAIM:2"
    route = "APERTURE_OPENING_FLUSH_RECLAIM_ENTRY"
    assert aperture_family_from_route(route) == "OPENING_FLUSH_RECLAIM"
    assert aperture_family_daily_cap(settings, route) == 2


def test_breadth_repair_filters_reuse_registered_score_components() -> None:
    settings = SimpleNamespace(
        pb_aperture_event_score_min=65.0,
        pb_aperture_family_score_floors="",
        pb_aperture_family_filters="PRIOR_DAY_LOW_RECLAIM:deep_reclaim",
    )
    event = SimpleNamespace(
        family="PRIOR_DAY_LOW_RECLAIM",
        score=68.0,
        score_components={
            "dislocation": 0.55,
            "reclaim": 0.34,
            "close_quality": 0.70,
            "relative_volume": 0.20,
            "residual_dislocation": 0.45,
            "prior_down_sequence": 0.50,
            "reversion_room": 0.40,
        },
    )
    assert aperture_event_admitted(settings, event) is False
    event.score_components["reclaim"] = 0.35
    assert aperture_event_admitted(settings, event) is True

    settings.pb_aperture_family_filters = "PRIOR_DAY_LOW_RECLAIM:residual_reclaim"
    event.score_components["residual_dislocation"] = 0.34
    assert aperture_event_admitted(settings, event) is False
    event.score_components["residual_dislocation"] = 0.35
    assert aperture_event_admitted(settings, event) is True

    settings.pb_aperture_family_filters = "PRIOR_DAY_LOW_RECLAIM:room_reclaim"
    event.score_components["reversion_room"] = 0.29
    assert aperture_event_admitted(settings, event) is False
    event.score_components["reversion_room"] = 0.30
    assert aperture_event_admitted(settings, event) is True


def test_portable_alpha_filters_are_structural_and_use_the_same_seven_inputs() -> None:
    settings = SimpleNamespace(
        pb_aperture_event_score_min=70.0,
        pb_aperture_family_score_floors="UPTREND_PULLBACK_RECLAIM:40",
        pb_aperture_family_filters="UPTREND_PULLBACK_RECLAIM:quiet_deep_room",
    )
    event = SimpleNamespace(
        family="UPTREND_PULLBACK_RECLAIM",
        score=55.0,
        score_components={
            "dislocation": 0.70,
            "reclaim": 0.60,
            "close_quality": 0.70,
            "relative_volume": 0.50,
            "residual_dislocation": 0.20,
            "prior_down_sequence": 0.25,
            "reversion_room": 0.50,
        },
    )
    assert aperture_family_score_floor(settings, event.family) == 40.0
    assert aperture_event_admitted(settings, event) is True
    event.score_components["relative_volume"] = 0.51
    assert aperture_event_admitted(settings, event) is False

    settings.pb_aperture_family_score_floors = (
        "MARKET_SECTOR_RESIDUAL_RECLAIM:40"
    )
    settings.pb_aperture_family_filters = (
        "MARKET_SECTOR_RESIDUAL_RECLAIM:relative_exhaustion"
    )
    event.family = "MARKET_SECTOR_RESIDUAL_RECLAIM"
    event.score_components["relative_volume"] = 0.80
    event.score_components["residual_dislocation"] = 0.75
    event.score_components["reversion_room"] = 0.25
    assert aperture_event_admitted(settings, event) is True
    event.score_components["reversion_room"] = 0.26
    assert aperture_event_admitted(settings, event) is False


def test_family_max_bar_is_opt_in_and_rejects_unregistered_cutoffs() -> None:
    settings = SimpleNamespace(
        pb_aperture_family_max_bars="",
        pb_aperture_prior_low_max_bar=48,
        pb_aperture_multiday_max_bar=6,
        pb_aperture_default_max_bar=48,
    )
    assert aperture_family_max_bar(settings, "PRIOR_DAY_LOW_RECLAIM") == 48
    assert aperture_family_max_bar(settings, "MULTIDAY_HIGHER_LOW_RECLAIM") == 6
    assert aperture_family_max_bar(settings, "VWAP_DEVIATION_RECLAIM") == 48

    settings.pb_aperture_family_max_bars = "PRIOR_DAY_LOW_RECLAIM:12"
    assert aperture_family_max_bar(settings, "PRIOR_DAY_LOW_RECLAIM") == 12
    settings.pb_aperture_family_max_bars = "PRIOR_DAY_LOW_RECLAIM:18"
    with pytest.raises(ValueError, match="6, 12, 24, or 48"):
        aperture_family_max_bar(settings, "PRIOR_DAY_LOW_RECLAIM")


def test_family_policy_parser_rejects_unregistered_values() -> None:
    settings = SimpleNamespace(
        pb_aperture_event_score_min=70.0,
        pb_aperture_family_score_floors="UNKNOWN_ROUTE:65",
    )
    with pytest.raises(ValueError, match="unknown reversion family"):
        aperture_family_score_floor(settings, "UPTREND_PULLBACK_RECLAIM")

    settings.pb_aperture_family_score_floors = "UPTREND_PULLBACK_RECLAIM:55"
    assert aperture_family_score_floor(settings, "UPTREND_PULLBACK_RECLAIM") == 55.0
    settings.pb_aperture_family_score_floors = "UPTREND_PULLBACK_RECLAIM:101"
    with pytest.raises(ValueError, match="between 0 and 100"):
        aperture_family_score_floor(settings, "UPTREND_PULLBACK_RECLAIM")

    settings.pb_aperture_family_score_floors = ""
    settings.pb_aperture_family_filters = "UPTREND_PULLBACK_RECLAIM:future_leak"
    with pytest.raises(ValueError, match="geometry.*participation"):
        aperture_family_filter(settings, "UPTREND_PULLBACK_RECLAIM")

    settings.pb_aperture_family_filters = ""
    settings.pb_aperture_family_daily_caps = "UPTREND_PULLBACK_RECLAIM:3"
    with pytest.raises(ValueError, match="1 or 2"):
        aperture_family_daily_cap(settings, "UPTREND_PULLBACK_RECLAIM")


def test_fold_validation_keeps_period_results_authoritative(monkeypatch, tmp_path) -> None:
    control = {
        "id": "incumbent_control",
        "mutations": {"variant": "control"},
        "metrics": {"expected_total_r": 10.0, "total_trades": 80.0},
    }
    finalist = {
        "id": "full_period_finalist",
        "mutations": {"variant": "candidate"},
        "stage": "management",
        "families": ["UPTREND_PULLBACK_RECLAIM"],
        "start_date": "2024-03-25",
        "end_date": "2026-03-01",
        "metrics": {"expected_total_r": 999.0, "total_trades": 999.0},
        "trade_attribution": [{"r": 999.0}],
    }

    def fake_evaluate(stage, candidates, *, args, **kwargs):
        rows = []
        for candidate in candidates:
            is_control = candidate["id"] == "incumbent_control"
            rows.append({
                "id": candidate["id"],
                "mutations": candidate["mutations"],
                "start_date": args.start_date,
                "end_date": args.end_date,
                "metrics": {
                    "expected_total_r": 2.0 if is_control else 5.0,
                    "total_trades": 10.0 if is_control else 15.0,
                },
                "aperture": {"trades": 0 if is_control else 5, "total_r": 3.0},
            })
        return rows

    monkeypatch.setattr(escape_runner, "_evaluate", fake_evaluate)
    args = SimpleNamespace(start_date="ignored", end_date="ignored", max_workers=2)
    _fold_validate(
        [finalist],
        control,
        args=args,
        output=tmp_path,
        source_fingerprint="source",
        code_fingerprint="code",
    )

    assert finalist["validation_contract"]["passed"] is True
    assert [fold["metrics"]["total_trades"] for fold in finalist["folds"]] == [15.0] * 3
    assert all(fold["metrics"]["total_trades"] != 999.0 for fold in finalist["folds"])
