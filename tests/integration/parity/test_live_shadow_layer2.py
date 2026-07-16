from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.integration.parity.fixtures import load_parity_fixture
from tests.integration.parity.harness import (
    run_layer2_contract,
    run_momentum_r1b_nqdtc_contract,
    run_nq_regime_r1a_contract,
)
from tests.integration.parity.live_shadow_contract import assert_shadow_contract
from tests.integration.parity.runtime_source import runtime_source_fingerprint

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "layer2"


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "fixture_name"),
    [
        ("IARIC", "iaric_entry_fill.json"),
        ("NQ_REGIME", "nq_regime_entry_fill.json"),
        ("TPC", "tpc_entry_fill.json"),
    ],
    ids=["iaric", "nq_regime", "tpc"],
)
async def test_live_shadow_layer2_matches_offline_oms_replay_contract(
    surface: str,
    fixture_name: str,
) -> None:
    assert_shadow_contract(await run_layer2_contract(surface, FIXTURE_ROOT / fixture_name))


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True], ids=["approved_full_fill", "portfolio_denial"])
async def test_nq_regime_r1a_one_child_family_causal_trace(disabled: bool) -> None:
    fixture = _nq_regime_r1a_fixture(disabled=disabled)

    contract = await run_nq_regime_r1a_contract(fixture)

    assert_shadow_contract(contract)
    assert contract.live.source_fingerprint == runtime_source_fingerprint(fixture)
    state = contract.live.state_snapshot or {}
    strategy_state = (state.get("strategy_state", {}) or {}).get("NQ_REGIME", {})
    if disabled:
        assert contract.live.order_intents == []
        assert contract.live.trade_ledger == []
        assert strategy_state.get("position_side") == "flat"
        assert strategy_state.get("last_decision_code") == "ENTRY_DENIED"
    else:
        assert [row["order_role"] for row in contract.live.order_intents].count("ENTRY") == 1
        assert strategy_state.get("position_side") == "long"
        assert strategy_state.get("qty_open", 0) > 0


def test_nq_regime_r1a_fixture_identity_includes_materialized_portfolio_rules() -> None:
    approved = _nq_regime_r1a_fixture(disabled=False)
    denied = _nq_regime_r1a_fixture(disabled=True)

    assert runtime_source_fingerprint(approved) != runtime_source_fingerprint(denied)


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["approved", "denied", "contention"],
)
async def test_momentum_r1b_nqdtc_shared_raw_timeline(scenario: str) -> None:
    fixture = _momentum_r1b_nqdtc_fixture(scenario=scenario)

    contract = await run_momentum_r1b_nqdtc_contract(fixture)

    assert_shadow_contract(contract)
    assert contract.live.source_fingerprint == runtime_source_fingerprint(fixture)
    state = contract.live.state_snapshot or {}
    strategy_state = state.get("strategy_state", {}) or {}
    nqdtc = strategy_state.get("NQDTC_v2.1", {}) or {}
    nq_regime = strategy_state.get("NQ_REGIME", {}) or {}
    blocked_reasons = state.get("blocked_reasons", {}) or {}
    submitted_strategies = [row["strategy_id"] for row in contract.live.order_intents]

    if scenario == "denied":
        assert "NQDTC_v2.1" not in submitted_strategies
        assert nqdtc.get("last_decision_code") == "ENTRY_DENIED"
        assert nqdtc.get("position_open") is False
        assert nq_regime.get("position_side") == "long"
        assert "portfolio_rule:regime_disabled" in blocked_reasons.get(
            "NQDTC_v2.1", []
        )
    elif scenario == "contention":
        assert "NQDTC_v2.1" in submitted_strategies
        assert "NQ_REGIME" not in submitted_strategies
        assert nqdtc.get("position_open") is True
        assert nq_regime.get("last_decision_code") == "ENTRY_DENIED"
        assert nq_regime.get("position_side") == "flat"
        assert "portfolio_rule:directional_cap" in blocked_reasons.get(
            "NQ_REGIME", []
        )
    else:
        assert {"NQDTC_v2.1", "NQ_REGIME"}.issubset(submitted_strategies)
        assert nqdtc.get("position_open") is True
        assert nq_regime.get("position_side") == "long"
        assert {row["strategy_id"] for row in contract.live.terminal_events} == {
            "NQDTC_v2.1",
            "NQ_REGIME",
        }


def _nq_regime_r1a_fixture(*, disabled: bool) -> dict[str, Any]:
    fixture = load_parity_fixture(FIXTURE_ROOT / "nq_regime_entry_fill.json")
    fixture["family_config"] = {
        "family": "momentum",
        "heat_cap_R": 20.0,
        "portfolio_daily_stop_R": 20.0,
        "portfolio_weekly_stop_R": 50.0,
        "strategies": [
            {
                "id": "NQ_REGIME",
                "family": "momentum",
                "unit_risk_dollars": 600.0,
                "daily_stop_R": 20.0,
                "priority": 0,
                "max_heat_R": 20.0,
                "max_working_orders": 4,
            }
        ],
        "portfolio_rules": {
            "family_strategy_ids": ["NQ_REGIME"],
            "directional_cap_R": 0.0,
            "directional_cap_long_R": 0.0,
            "directional_cap_short_R": 0.0,
            "reference_unit_risk_dollars": 500.0,
            "portfolio_heat_cap_R": 0.0,
            "max_trade_risk_R": 0.0,
            "max_total_active_positions": 0,
            "max_strategy_active_positions": [],
            "max_family_contracts_mnq_eq": 0,
            "strategy_priorities": [],
            "priority_headroom_R": 0.0,
            "strategy_size_multipliers": [["NQ_REGIME", 1.0]],
            "existing_position_mult": 1.0,
            "heat_pressure_mult": 1.0,
            "same_direction_pressure_mult": 1.0,
            "fit_to_remaining_heat": False,
            "fit_to_remaining_directional_cap": False,
            "fit_to_remaining_family_cap": False,
            "nqdtc_direction_filter_enabled": False,
            "symbol_collision_action": "none",
            "dd_tiers": [[1.0, 1.0]],
            "disabled_strategies": ["NQ_REGIME"] if disabled else [],
        },
    }
    if disabled:
        fixture["broker_event_script"] = []
    return fixture


def _momentum_r1b_nqdtc_fixture(*, scenario: str) -> dict[str, Any]:
    fixture = load_parity_fixture(FIXTURE_ROOT / "nq_regime_entry_fill.json")
    fixture["initial_strategy_state"]["NQDTC_v2.1"] = {}
    fixture.setdefault("artifacts", {})["nqdtc"] = {
        "r1b_market_input": {
            "symbol": "NQ",
            "timestamp": "2026-05-20T14:15:00+00:00",
            "bars": [
                {
                    "symbol": "NQ",
                    "timeframe": "5m",
                    "timestamp": "2026-05-20T14:05:00+00:00",
                    "open": 20012.0,
                    "high": 20024.0,
                    "low": 20008.0,
                    "close": 20018.0,
                    "volume": 1000.0,
                },
                {
                    "symbol": "NQ",
                    "timeframe": "5m",
                    "timestamp": "2026-05-20T14:10:00+00:00",
                    "open": 20018.0,
                    "high": 20032.0,
                    "low": 20014.0,
                    "close": 20026.0,
                    "volume": 1100.0,
                },
                {
                    "symbol": "NQ",
                    "timeframe": "5m",
                    "timestamp": "2026-05-20T14:15:00+00:00",
                    "open": 20026.0,
                    "high": 20048.0,
                    "low": 20020.0,
                    "close": 20042.0,
                    "volume": 1400.0,
                },
            ],
            "decision_state": {
                "session": "RTH",
                "direction": "LONG",
                "box_high": 20040.0,
                "box_low": 19960.0,
                "box_mid": 20000.0,
                "breakout_bar_high": 20048.0,
                "breakout_bar_low": 20020.0,
                "atr14_30m": 20.0,
                "score": 3.0,
                "disp_metric": 2.0,
                "disp_threshold": 1.0,
                "disp_history": [1.0] * 12,
                "vwap": 20020.0,
                "regime_4h": "TRANSITIONAL",
                "trend_dir_4h": "FLAT",
            },
        }
    }
    directional_cap = 1.2 if scenario == "contention" else 20.0
    fixture["family_config"] = {
        "family": "momentum",
        "heat_cap_R": 20.0,
        "portfolio_daily_stop_R": 20.0,
        "portfolio_weekly_stop_R": 50.0,
        "strategies": [
            {
                "id": "NQDTC_v2.1",
                "family": "momentum",
                "unit_risk_dollars": 450.0,
                "daily_stop_R": 20.0,
                "priority": 0,
                "max_heat_R": 20.0,
                "max_working_orders": 4,
            },
            {
                "id": "NQ_REGIME",
                "family": "momentum",
                "unit_risk_dollars": 600.0,
                "daily_stop_R": 20.0,
                "priority": 1,
                "max_heat_R": 20.0,
                "max_working_orders": 4,
            },
        ],
        "portfolio_rules": {
            "family_strategy_ids": ["NQDTC_v2.1", "NQ_REGIME"],
            "directional_cap_R": directional_cap,
            "directional_cap_long_R": directional_cap,
            "directional_cap_short_R": directional_cap,
            "reference_unit_risk_dollars": 500.0,
            "portfolio_heat_cap_R": 0.0,
            "max_trade_risk_R": 0.0,
            "max_total_active_positions": 0,
            "max_strategy_active_positions": [],
            "max_family_contracts_mnq_eq": 0,
            "strategy_priorities": [["NQDTC_v2.1", 0], ["NQ_REGIME", 1]],
            "priority_headroom_R": 0.0,
            "strategy_size_multipliers": [
                ["NQDTC_v2.1", 1.0],
                ["NQ_REGIME", 1.0],
            ],
            "existing_position_mult": 1.0,
            "heat_pressure_mult": 1.0,
            "same_direction_pressure_mult": 1.0,
            "fit_to_remaining_heat": False,
            "fit_to_remaining_directional_cap": False,
            "fit_to_remaining_family_cap": False,
            "nqdtc_direction_filter_enabled": False,
            "symbol_collision_action": "none",
            "dd_tiers": [[1.0, 1.0]],
            "disabled_strategies": (
                ["NQDTC_v2.1"] if scenario == "denied" else []
            ),
        },
    }
    broker_script = []
    if scenario != "denied":
        broker_script.append(
            {
                "order_match": {
                    "strategy_id": "NQDTC_v2.1",
                    "symbol": "NQ",
                    "role": "ENTRY",
                    "side": "BUY",
                    "sequence": 1,
                },
                "event": "fill",
                "exec_id": f"NQDTC-R1B-{scenario}",
                "price": 20020.25,
                "commission": 0.0,
                "timestamp": "2026-05-20T14:16:00+00:00",
            }
        )
    if scenario != "contention":
        broker_script.append(
            {
                "order_match": {
                    "strategy_id": "NQ_REGIME",
                    "symbol": "MNQ",
                    "role": "ENTRY",
                    "side": "BUY",
                    "sequence": 1,
                },
                "event": "fill",
                "exec_id": f"NQREG-R1B-{scenario}",
                "price": 20036.0,
                "commission": 0.0,
                "timestamp": "2026-05-20T14:17:00+00:00",
            }
        )
    fixture["broker_event_script"] = broker_script
    return fixture
