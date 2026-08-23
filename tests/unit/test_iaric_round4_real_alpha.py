from __future__ import annotations

from types import SimpleNamespace

import pytest

from backtests.stock.auto.iaric.round4_scoring import (
    SCORE_SPEC,
    fixed_atlas_recall,
    issuer_diagnostics,
    issuer_key,
    score_candidate,
    sector_diagnostics,
)
from backtests.stock.auto.iaric.worker import _compact_trade_attribution
from backtests.stock.auto.runners.run_iaric_round4_real_alpha import (
    _architectural_baseline,
    _baseline_incidence_contract,
    _parity_contract,
    _supply_candidates,
)


def _metrics(**overrides: float) -> dict[str, float]:
    values = {
        "expected_total_r": 47.0,
        "total_trades": 149.0,
        "entry_opportunity_recall": 0.21,
        "entry_realized_discrimination_lift_r": 0.34,
        "max_drawdown_pct": 0.03,
    }
    values.update(overrides)
    return values


def _trade(symbol: str, date: str, value: float) -> dict[str, object]:
    return {"symbol": symbol, "entry_time": f"{date}T10:00:00+00:00", "r": value}


def test_round4_score_is_exactly_seven_fixed_components_and_baseline_is_half() -> None:
    attribution = [
        _trade("AAA", "2024-05-01", 1.0),
        _trade("BBB", "2025-03-01", 1.0),
        _trade("CCC", "2025-10-01", 1.0),
    ]
    score, components, raw, _ = score_candidate(
        _metrics(), attribution, _metrics(), attribution
    )
    assert len(SCORE_SPEC) == 7
    assert len(components) == 7
    assert sum(spec["weight"] for spec in SCORE_SPEC.values()) == pytest.approx(1.0)
    assert score == pytest.approx(0.5)
    assert set(raw) == set(SCORE_SPEC)


def test_score_rewards_broad_alpha_frequency_recall_and_discrimination() -> None:
    control = [
        _trade("GOOG", "2024-05-01", 3.0),
        _trade("GOOGL", "2025-03-01", 3.0),
        _trade("AAA", "2025-10-01", 1.0),
    ]
    candidate = control + [
        _trade("BBB", "2024-06-01", 3.0),
        _trade("CCC", "2025-04-01", 3.0),
    ]
    score, _, raw, _ = score_candidate(
        _metrics(
            expected_total_r=56.0,
            total_trades=179.0,
            entry_opportunity_recall=0.29,
            entry_realized_discrimination_lift_r=0.46,
            max_drawdown_pct=0.025,
        ),
        candidate,
        _metrics(),
        control,
    )
    assert score > 0.5
    assert raw["worst_fold_r_per_month"] > 0.0
    assert raw["issuer_sector_concentration"] > 0.0
    assert raw["net_expected_r_per_month"] > 0.0


def test_locked_internal_validation_attribution_cannot_change_ranking_score() -> None:
    control = [_trade("AAA", "2024-05-01", 1.0)]
    locked_winner = control + [_trade("ZZZ", "2025-10-01", 100.0)]
    score_a, _, raw_a, _ = score_candidate(_metrics(), control, _metrics(), control)
    score_b, _, raw_b, _ = score_candidate(
        _metrics(), locked_winner, _metrics(), control
    )
    assert score_b == pytest.approx(score_a)
    assert raw_b == pytest.approx(raw_a)


def test_run_level_metrics_cannot_leak_locked_outcomes_into_score() -> None:
    attribution = [
        {**_trade("AAA", "2024-05-01", 0.5), "cannibalized_r": 0.1},
        _trade("BBB", "2025-03-01", 0.5),
    ]
    score_a, _, raw_a, _ = score_candidate(
        _metrics(expected_total_r=-1_000.0, total_trades=1.0, cannibalized_r=500.0),
        attribution,
        _metrics(expected_total_r=1_000.0, total_trades=99_999.0),
        attribution,
    )
    score_b, _, raw_b, _ = score_candidate(
        _metrics(expected_total_r=1_000_000.0, total_trades=1_000_000.0),
        attribution,
        _metrics(expected_total_r=-1_000_000.0, total_trades=0.0, cannibalized_r=0.0),
        attribution,
    )
    assert score_b == pytest.approx(score_a)
    assert raw_b == pytest.approx(raw_a)


def test_recall_uses_one_frozen_control_atlas_denominator() -> None:
    control = _metrics(
        entry_potential_total_r=20.0,
        entry_oracle_potential_r=100.0,
        entry_opportunity_recall=0.20,
    )
    wider_candidate = _metrics(
        entry_potential_total_r=25.0,
        entry_oracle_potential_r=250.0,
        entry_opportunity_recall=0.10,
    )
    assert fixed_atlas_recall(control, control) == pytest.approx(0.20)
    assert fixed_atlas_recall(wider_candidate, control) == pytest.approx(0.25)


def test_issuer_concentration_combines_listed_share_classes() -> None:
    assert issuer_key("GOOG") == issuer_key("GOOGL") == "ALPHABET"
    diagnostics = issuer_diagnostics(
        [_trade("GOOG", "2025-01-01", 2.0), _trade("GOOGL", "2025-01-02", 3.0)]
    )
    assert diagnostics["top_positive_issuer"] == "ALPHABET"
    assert diagnostics["top_positive_issuer_r"] == pytest.approx(5.0)
    assert diagnostics["top_positive_issuer_share"] == pytest.approx(1.0)


def test_sector_concentration_is_measured_separately_from_issuer() -> None:
    trades = [
        {**_trade("AAA", "2025-01-01", 2.0), "sector": "Technology"},
        {**_trade("BBB", "2025-01-02", 1.0), "sector": "Technology"},
        {**_trade("CCC", "2025-01-03", 1.0), "sector": "Health Care"},
    ]
    diagnostics = sector_diagnostics(trades)
    assert diagnostics["top_positive_sector"] == "TECHNOLOGY"
    assert diagnostics["top_positive_sector_share"] == pytest.approx(0.75)
    assert diagnostics["effective_positive_sectors"] > 1.0


def test_structural_search_is_bounded_causal_and_uses_typed_shared_settings() -> None:
    baseline = {
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_families": (
            "FAILED_BREAKDOWN_RECLAIM,MULTIDAY_HIGHER_LOW_RECLAIM,"
            "UPTREND_PULLBACK_RECLAIM"
        ),
        "param_overrides.pb_aperture_family_score_floors": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:65"
        ),
        "param_overrides.pb_aperture_family_transitions": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:next_bar"
        ),
        "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
    }
    candidates = _supply_candidates(baseline)
    assert len(candidates) == 8  # control plus seven mechanism-level hypotheses
    assert _parity_contract(candidates)["passed"] is True
    assert all(
        candidate["mutations"]["param_overrides.pb_open_scored_fill_timing"]
        == "next_5m_open"
        for candidate in candidates
    )
    assert not any(
        "same_bar" in str(value)
        for candidate in candidates
        for value in candidate["mutations"].values()
    )


def test_architectural_migration_preserves_incumbent_score_and_tail_contracts() -> None:
    latest = {
        "param_overrides.pb_aperture_event_score_min": 70.0,
        "param_overrides.pb_aperture_family_score_floors": (
            "MULTIDAY_HIGHER_LOW_RECLAIM:65,PRIOR_DAY_LOW_RECLAIM:70"
        ),
    }
    migrated = _architectural_baseline(latest)
    assert migrated["param_overrides.pb_aperture_family_score_floors"] == latest[
        "param_overrides.pb_aperture_family_score_floors"
    ]
    assert migrated["param_overrides.pb_aperture_event_score_min"] == 70.0
    assert migrated["param_overrides.pb_aperture_anchor_exit_enabled"] is False
    assert migrated["param_overrides.pb_issuer_position_cap"] == 1


def test_baseline_incidence_contract_rejects_silent_aperture_expansion() -> None:
    reference = {
        "metrics": {"total_trades": 158, "open_scored_trades": 89},
        "aperture": {
            "trades": 69,
            "routes": {"APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY": {"trades": 10}},
        },
        "funnel_counters": {
            "aperture_ready": 69,
            "lane__aperture_level_reclaim__event_detected": 4605,
            "lane__aperture_trend_pullback__event_detected": 2061,
        },
    }
    observed = {
        "metrics": {"total_trades": 1767, "open_scored_trades": 81},
        "aperture": {
            "trades": 1686,
            "routes": {"APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY": {"trades": 999}},
        },
        "funnel_counters": {
            "aperture_ready": 1840,
            "lane__aperture_level_reclaim__event_detected": 4285,
            "lane__aperture_level_reclaim__score_rejected": 41,
            "lane__aperture_trend_pullback__event_detected": 1856,
            "lane__aperture_trend_pullback__score_rejected": 4,
        },
    }
    contract = _baseline_incidence_contract(observed, reference)
    assert contract["passed"] is False
    assert contract["checks"]["admission_remains_discriminatory"] is False


def test_compact_attribution_retains_structural_score_components() -> None:
    trade = SimpleNamespace(
        metadata={
            "entry_route_family": "APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY",
            "entry_bar_index": 5,
            "signal_bar_index": 4,
            "entry_score_component_residual_dislocation": 0.45,
            "entry_score_component_reversion_room": 0.30,
        },
        fill_bar_index=5,
        signal_bar_index=4,
        entry_type="unused",
        symbol="AAA",
        entry_time=SimpleNamespace(isoformat=lambda: "2025-01-01T10:00:00+00:00"),
        exit_time=SimpleNamespace(isoformat=lambda: "2025-01-01T11:00:00+00:00"),
        exit_reason="STOP_HIT",
        r_multiple=-0.2,
        pnl_net=-10.0,
        entry_price=100.0,
        risk_per_share=2.0,
    )
    compact = _compact_trade_attribution([trade])[0]
    assert compact["lane"] == "APERTURE_LEVEL_RECLAIM"
    assert compact["score_components"]["residual_dislocation"] == pytest.approx(0.45)
    assert compact["score_components"]["reversion_room"] == pytest.approx(0.30)
    assert "daily_signal" in compact["score_components"]
