from __future__ import annotations

from backtests.stock.auto.runners.run_iaric_residual_baseline_diagnostics import (
    BASELINE_CONTRACT_ID,
    baseline_settings,
)


def test_presearch_baseline_contract_is_frozen_and_bounded() -> None:
    settings = baseline_settings()
    assert BASELINE_CONTRACT_ID == "iaric_residual_presearch_exact98_v1"
    assert settings.strategy_mode == "daily_residual_reversion"
    assert settings.daily_residual_factor_model == "market_sector_peer"
    assert settings.daily_residual_formation_sessions == 3
    assert settings.daily_residual_minimum_z == 1.0
    assert settings.daily_residual_score_components == ("residual_extremeness",)
    assert settings.daily_residual_max_positions == 10
    assert settings.daily_residual_max_positions_per_sector == 2
    assert settings.daily_residual_maximum_holding_sessions == 7
    assert len(settings.daily_residual_score_components) <= 5
