from __future__ import annotations

import pandas as pd
import pytest

from backtests.regime.crisis_validation import build_event_channel_chronology
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.crisis import config as C
from regime.crisis.actions import resolve_crisis_action
from regime.crisis.context import CrisisContext, _dd_mult_for_level, _risk_mult_for_level
from regime.crisis.detector import (
    compute_advisory_level,
    compute_alert_level,
    is_hard_credit_impulse_warning_candidate,
)
from regime.crisis.hysteresis import HysteresisTracker
from regime.crisis.indicators import ChannelReading, CrisisIndicators
from regime.crisis.integration import apply_crisis_overlay


def _indicators(*, vix: int = 0, spread: int = 0, slope: int = 0,
                corr: int = 0, spy_dd: int = 0) -> CrisisIndicators:
    return CrisisIndicators(
        vix=ChannelReading("VIX", 30.0, vix),
        credit_spread=ChannelReading("CREDIT_SPREAD", 260.0, spread),
        yield_curve=ChannelReading("YIELD_CURVE", -0.6, slope),
        spy_tlt_corr=ChannelReading("SPY_TLT_CORR", 0.45, corr),
        spy_drawdown=ChannelReading("SPY_DRAWDOWN", -0.05, spy_dd),
    )


def _crisis_ctx(level: int) -> CrisisContext:
    return CrisisContext(
        alert_level=C.ALERT_LEVELS[level],
        alert_level_int=level,
        portfolio_action_level=C.ALERT_LEVELS[level] if level >= 2 else C.ALERT_NORMAL,
        portfolio_action_level_int=level if level >= 2 else 0,
        risk_multiplier=_risk_mult_for_level(level),
        dd_tier_multiplier=_dd_mult_for_level(level),
    )


def test_internal_watch_does_not_become_external_advisory() -> None:
    indicators = _indicators(vix=1)

    alert_level, alert_level_int = compute_alert_level(indicators)
    advisory_level, advisory_level_int, reason = compute_advisory_level(
        indicators,
        alert_level_int,
    )

    assert (alert_level, alert_level_int) == (C.ALERT_WATCH, 1)
    assert (advisory_level, advisory_level_int) == (C.ALERT_NORMAL, 0)
    assert reason == "internal watch only"


def test_external_watch_requires_broader_or_deeper_confirmation() -> None:
    broad_watch = _indicators(vix=1, spread=1, slope=1, corr=1)
    one_crisis = _indicators(vix=3)

    assert compute_advisory_level(broad_watch, 1)[:2] == (C.ALERT_WATCH, 1)
    assert compute_advisory_level(one_crisis, 1)[:2] == (C.ALERT_WATCH, 1)


def test_external_advisory_mirrors_actionable_warning() -> None:
    indicators = _indicators(vix=2, spread=2)
    alert_level, alert_level_int = compute_alert_level(indicators)

    assert (alert_level, alert_level_int) == (C.ALERT_WARNING, 2)
    assert compute_advisory_level(indicators, alert_level_int)[:2] == (
        C.ALERT_WARNING,
        2,
    )


def test_stock_crisis_policy_blocks_new_stock_entries_and_tightens_caps() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        directional_cap_long_R=8.0,
        directional_cap_short_R=6.0,
        priority_headroom_R=2.0,
        regime_unit_risk_mult=0.8,
        disabled_strategies=frozenset({"AlreadyDisabled"}),
    )

    result = apply_crisis_overlay(base, _crisis_ctx(3), family_id="stock")

    assert result.regime_unit_risk_mult == pytest.approx(0.24)
    assert result.directional_cap_R == pytest.approx(6.0)
    assert result.directional_cap_long_R == pytest.approx(3.2)
    assert result.directional_cap_short_R == pytest.approx(3.6)
    assert result.priority_headroom_R == pytest.approx(1.0)
    assert result.disabled_strategies == frozenset({
        "ALCB_v1",
        "AlreadyDisabled",
        "IARIC_v1",
    })


def test_momentum_crisis_policy_preserves_short_capacity_and_cuts_contracts() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=4.0,
        directional_cap_long_R=10.0,
        directional_cap_short_R=8.0,
        max_family_contracts_mnq_eq=40,
        regime_unit_risk_mult=1.0,
    )

    result = apply_crisis_overlay(base, _crisis_ctx(3), family_id="momentum")

    assert result.directional_cap_R == pytest.approx(2.2)
    assert result.directional_cap_long_R == pytest.approx(3.5)
    assert result.directional_cap_short_R == pytest.approx(8.0)
    assert result.max_family_contracts_mnq_eq == 20
    assert result.regime_unit_risk_mult == pytest.approx(1.0)
    assert result.regime_unit_risk_long_mult == pytest.approx(0.3)
    assert result.regime_unit_risk_short_mult == pytest.approx(1.0)


def test_shock_formation_watch_can_lightly_tighten_without_warning() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        regime_unit_risk_mult=1.0,
    )
    ctx = CrisisContext(
        alert_level=C.ALERT_WATCH,
        alert_level_int=1,
        advisory_level=C.ALERT_WATCH,
        advisory_level_int=1,
        portfolio_action_level=C.ALERT_WATCH,
        portfolio_action_level_int=1,
        risk_multiplier=C.STRESS_FORMATION_RISK_MULT_SHOCK,
        stress_formation_score=C.STRESS_FORMATION_MIN_SCORE,
        stress_formation_mode="shock",
    )

    result = apply_crisis_overlay(base, ctx, family_id="stock")

    assert result.regime_unit_risk_mult == pytest.approx(0.75)
    assert result.directional_cap_R == pytest.approx(10.0)


def test_credit_impulse_bridge_warning_keeps_early_label_but_uses_lighter_action() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        directional_cap_long_R=8.0,
        priority_headroom_R=2.0,
        regime_unit_risk_mult=1.0,
    )
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        stress_formation_score=C.STRESS_FORMATION_MIN_SCORE,
        stress_formation_mode="credit_impulse",
        primary_warning_count=1,
        primary_crisis_count=0,
    )

    action = resolve_crisis_action(ctx, family_id="stock", regime="G")
    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="G")

    assert action.alert_level == C.ALERT_WARNING
    assert action.action_provenance == "credit_impulse_bridge"
    assert result.regime_unit_risk_mult == pytest.approx(0.75)
    assert result.directional_cap_R == pytest.approx(9.0)
    assert result.directional_cap_long_R == pytest.approx(6.4)
    assert result.priority_headroom_R == pytest.approx(1.8)


def test_confirmed_warning_keeps_full_growth_warning_cut() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        regime_unit_risk_mult=1.0,
    )
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        primary_warning_count=2,
    )

    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="G")

    assert result.regime_unit_risk_mult == pytest.approx(0.65)
    assert result.directional_cap_R == pytest.approx(8.5)


def test_warning_in_stress_regime_adds_smaller_incremental_cut() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        regime_unit_risk_mult=0.75,
    )
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        primary_warning_count=2,
    )

    action = resolve_crisis_action(ctx, family_id="stock", regime="S")
    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="S")

    assert action.action_provenance == "stress_regime"
    assert result.regime_unit_risk_mult == pytest.approx(0.60)
    assert result.directional_cap_R == pytest.approx(9.0)


def test_credit_impulse_hard_bridge_requires_persistence(monkeypatch) -> None:
    monkeypatch.setattr(C, "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS", 3)
    monkeypatch.setattr(C, "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY", 1)

    indicators = _indicators(spread=3)
    indicators.stress_formation_score = C.STRESS_FORMATION_MIN_SCORE
    indicators.stress_formation_mode = "credit_impulse"
    tracker = HysteresisTracker()

    raw_level_ints = []
    final_level_ints = []
    for _ in range(3):
        _, raw_level_int = compute_alert_level(indicators)
        raw_level_int = tracker.apply_hard_credit_impulse_bridge(
            raw_level_int,
            is_hard_credit_impulse_warning_candidate(indicators),
        )
        raw_level_ints.append(raw_level_int)
        final_level_ints.append(tracker.update(raw_level_int))

    assert raw_level_ints == [1, 1, 2]
    assert final_level_ints == [1, 1, 2]


def test_event_channel_chronology_identifies_confirmation_bottleneck() -> None:
    dates = pd.date_range("2020-02-15", periods=5, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 1, 2, 2, 3],
            "raw_level_int": [0, 1, 2, 2, 3],
            "advisory_level_int": [0, 1, 2, 2, 3],
            "vix_level_int": [0, 2, 2, 2, 3],
            "credit_spread_level_int": [0, 0, 2, 2, 2],
            "yield_curve_level_int": [0, 0, 0, 1, 1],
            "spy_tlt_corr_level_int": [0, 0, 0, 0, 0],
            "spy_drawdown_level_int": [0, 1, 1, 1, 3],
        },
        index=dates,
    )

    chronology = build_event_channel_chronology(
        alerts,
        {"Demo": ("2020-02-15", "2020-02-19", "D")},
    )

    demo = chronology["Demo"]
    assert demo["detected_at"] == "2020-02-17"
    assert demo["latency_days"] == 2
    assert demo["bottleneck_channel"] == "CREDIT_SPREAD"
