from __future__ import annotations

import pandas as pd
import pytest

from backtests.regime.auto.crisis.economic import (
    CrisisEconomicPolicy,
    CrisisSleeveEconomicPolicy,
    build_exposure_series,
    build_sleeve_exposure_map,
    evaluate_policy,
)
from backtests.regime.auto.crisis.scoring import extract_crisis_metrics
from regime.crisis import config as C
from regime.crisis.detector import compute_advisory_level
from regime.crisis.indicators import compute_indicators


def test_stress_formation_can_surface_external_advisory_without_action() -> None:
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    market = pd.DataFrame(
        {
            "VIX": [20.0] * 21 + [21.0, 24.0, 30.0, 31.0],
            "SPREAD": [2.0] * 25,
            "SLOPE_10Y2Y": [1.0] * 25,
        },
        index=dates,
    )
    ret = pd.DataFrame(
        {
            "SPY": [0.0] * 20 + [-0.01, -0.02, -0.02, -0.015, 0.0],
            "TLT": [0.0] * 25,
        },
        index=dates,
    )

    indicators = compute_indicators(market, ret, date=dates[-1])
    advisory_level, advisory_level_int, reason = compute_advisory_level(
        indicators,
        action_level_int=0,
    )

    assert indicators.stress_formation_score >= C.STRESS_FORMATION_MIN_SCORE
    assert indicators.stress_formation_mode == "shock"
    assert (advisory_level, advisory_level_int) == (C.ALERT_WATCH, 1)
    assert "stress formation shock" in reason


def test_build_exposure_series_applies_pre_action_then_action_overrides() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 1, 1, 2, 3],
            "portfolio_action_level_int": [0, 0, 0, 2, 3],
            "advisory_level_int": [0, 1, 0, 2, 3],
            "stress_formation_mode": ["", "", "shock", "", ""],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        advisory_mult=0.95,
        shock_mult=0.90,
        warning_mult=0.70,
        crisis_mult=0.40,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [1.0, 0.95, 0.90, 0.70, 0.40]


def test_credit_impulse_pre_action_has_separate_multiplier() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 2],
            "portfolio_action_level_int": [1, 1, 2],
            "advisory_level_int": [1, 1, 2],
            "stress_formation_mode": ["credit_impulse", "shock+credit_impulse", ""],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        advisory_mult=1.0,
        shock_mult=0.85,
        credit_impulse_mult=0.75,
        warning_mult=0.65,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.75, 0.75, 0.65]


def test_credit_bridge_warning_can_be_scored_separately_from_confirmed_warning() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [2, 2, 3],
            "portfolio_action_level_int": [2, 2, 3],
            "advisory_level_int": [2, 2, 3],
            "stress_formation_mode": ["credit_impulse", "", ""],
            "primary_warning_count": [1, 2, 2],
            "primary_crisis_count": [0, 0, 2],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        warning_mult=0.65,
        crisis_mult=0.30,
        credit_bridge_warning_mult=0.75,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.75, 0.65, 0.30]


def test_regime_conditioned_warning_policy_reduces_overcut_in_stress_states() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [2, 2, 2, 3],
            "portfolio_action_level_int": [2, 2, 2, 3],
            "advisory_level_int": [2, 2, 2, 3],
            "hmm_regime": ["G", "S", "D", "D"],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        warning_mult=0.65,
        crisis_mult=0.30,
        stress_regime_warning_mult=0.80,
        defensive_regime_warning_mult=0.85,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.65, 0.80, 0.85, 0.30]


def test_sleeve_policy_can_preserve_short_exposure_separately() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 2, 3, 0],
            "portfolio_action_level_int": [0, 2, 3, 0],
            "advisory_level_int": [0, 2, 3, 0],
            "stress_formation_mode": [""] * 4,
        },
        index=dates,
    )
    policy = CrisisSleeveEconomicPolicy(
        equity_warning_mult=0.65,
        equity_crisis_mult=0.30,
        gld_warning_mult=0.80,
        gld_crisis_mult=0.60,
        short_warning_mult=1.00,
        short_crisis_mult=1.00,
    )

    exposures = build_sleeve_exposure_map(alerts, policy)

    assert exposures["equity_beta"].tolist() == [1.0, 0.65, 0.30, 1.0]
    assert exposures["gld"].tolist() == [1.0, 0.80, 0.60, 1.0]
    assert exposures["short_spy"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_credit_impulse_can_surface_targeted_pre_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "CREDIT_IMPULSE_SPREAD_BPS", 336.0)
    monkeypatch.setattr(C, "CREDIT_IMPULSE_SPY_3D_RETURN", -0.015)
    monkeypatch.setattr(C, "CREDIT_IMPULSE_MIN_VIX", 18.0)

    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    market = pd.DataFrame(
        {
            "VIX": [16.0] * 22 + [18.5, 19.0, 18.2],
            "SPREAD": [3.6] * 25,
            "SLOPE_10Y2Y": [1.0] * 25,
        },
        index=dates,
    )
    ret = pd.DataFrame(
        {
            "SPY": [0.0] * 22 + [-0.006, -0.006, -0.006],
            "TLT": [0.0] * 25,
        },
        index=dates,
    )

    indicators = compute_indicators(market, ret, date=dates[-1])

    assert indicators.stress_formation_score >= C.STRESS_FORMATION_MIN_SCORE
    assert indicators.stress_formation_mode == "credit_impulse"


def test_economic_policy_can_improve_drawdown_on_synthetic_crash() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 2, 2, 0, 0],
            "portfolio_action_level_int": [0, 0, 2, 2, 0, 0],
            "advisory_level_int": [0, 0, 2, 2, 0, 0],
            "stress_formation_mode": [""] * 6,
        },
        index=dates,
    )
    base = pd.Series([0.01, 0.01, -0.10, -0.08, 0.03, 0.03], index=dates)
    cash = pd.Series(0.0, index=dates)

    result = evaluate_policy(
        alerts_df=alerts,
        base_portfolios={"regime_proxy": base},
        cash_returns=cash,
        policy=CrisisEconomicPolicy(warning_mult=0.50, crisis_mult=0.50),
        portfolio_score_weights={"regime_proxy": 1.0},
    )

    deltas = result["portfolio_results"]["regime_proxy"]["deltas"]
    assert deltas["max_drawdown_pct"] < 0
    assert result["score"] > 0


def test_crisis_metrics_track_advisory_and_action_latency_separately() -> None:
    dates = pd.date_range("2020-02-15", periods=6, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 0, 2, 2, 3],
            "advisory_level_int": [1, 1, 1, 2, 2, 3],
            "portfolio_action_level_int": [0, 1, 1, 2, 2, 3],
        },
        index=dates,
    )

    metrics = extract_crisis_metrics(alerts)

    assert metrics.avg_latency == 3.0
    assert metrics.avg_advisory_latency == 0.0
    assert metrics.avg_action_latency == 1.0
    assert metrics.avg_action_latency < metrics.avg_latency
