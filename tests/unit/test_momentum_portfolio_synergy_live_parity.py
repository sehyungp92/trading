from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _optimized_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "backtests"
        / "output"
        / "momentum"
        / "portfolio_synergy"
        / "round_3"
        / "optimized_config.json"
    )
    return json.loads(path.read_text())


def test_momentum_live_configs_match_round3_portfolio_synergy() -> None:
    from strategies.momentum import coordinator
    from strategies.momentum.downturn import config as downturn_config
    from strategies.momentum.nq_regime import config as nq_regime_config
    from strategies.momentum.nqdtc import config as nqdtc_config
    from strategies.momentum.vdub import config as vdub_config

    optimized = _optimized_config()
    optimized_allocs = {
        row["strategy_id"]: row for row in optimized["strategy_allocations"]
    }

    assert coordinator._PORTFOLIO_HEAT_CAP_R == optimized["heat_cap_R"]
    assert coordinator._PORTFOLIO_DAILY_STOP_R == optimized["portfolio_daily_stop_R"]
    assert coordinator._PORTFOLIO_WEEKLY_STOP_R == optimized["portfolio_weekly_stop_R"]
    assert coordinator._OPTIMIZED_INITIAL_EQUITY == optimized["initial_equity"]
    assert (
        coordinator._OPTIMIZED_REFERENCE_UNIT_RISK_DOLLARS
        == optimized["reference_unit_risk_dollars"]
    )
    assert coordinator._REFERENCE_UNIT_RISK_PCT == pytest.approx(
        optimized["reference_unit_risk_dollars"] / optimized["initial_equity"]
    )
    assert coordinator._MAX_TOTAL_POSITIONS == optimized["max_total_positions"]
    assert (
        coordinator._MAX_FAMILY_CONTRACTS_MNQ_EQ
        == optimized["rules"]["max_family_contracts_mnq_eq"]
    )
    assert coordinator._DD_TIERS == tuple(
        tuple(row) for row in optimized["rules"]["dd_tiers"]
    )
    assert coordinator._STRATEGY_PRIORITIES == tuple(
        tuple(row) for row in optimized["rules"]["strategy_priorities"]
    )
    assert coordinator._STRATEGY_SIZE_MULTIPLIERS == tuple(
        tuple(row) for row in optimized["dynamic_risk"]["strategy_multipliers"]
    )

    live_base_risk = {
        nq_regime_config.STRATEGY_ID: nq_regime_config.BASE_RISK_PCT,
        vdub_config.STRATEGY_ID: vdub_config.BASE_RISK_PCT,
        nqdtc_config.STRATEGY_ID: nqdtc_config.BASE_RISK_PCT,
        downturn_config.STRATEGY_ID: downturn_config.BASE_RISK_PCT,
    }
    live_daily_stops = dict(coordinator._STRATEGY_DAILY_STOPS_R)
    live_max_concurrent = dict(coordinator._MAX_STRATEGY_ACTIVE_POSITIONS)
    live_priorities = dict(coordinator._STRATEGY_PRIORITIES)

    assert set(live_base_risk) == set(optimized_allocs)
    for strategy_id, expected in optimized_allocs.items():
        assert live_base_risk[strategy_id] == expected["base_risk_pct"], strategy_id
        assert live_daily_stops[strategy_id] == expected["daily_stop_R"], strategy_id
        assert live_max_concurrent[strategy_id] == expected["max_concurrent"], strategy_id
        assert live_priorities[strategy_id] == expected["priority"], strategy_id

    assert (
        abs(nqdtc_config.DAILY_STOP_R)
        == optimized_allocs["NQDTC_v2.1"]["daily_stop_R"]
    )
