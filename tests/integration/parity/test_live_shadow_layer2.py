from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.integration.parity.fixtures import load_parity_fixture
from tests.integration.parity.harness import (
    run_layer2_contract,
    run_momentum_r1b_nqdtc_contract,
    run_momentum_r1b_vdub_contract,
    run_momentum_r1b_downturn_contract,
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


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["approved", "denied", "contention"])
async def test_momentum_r1b_vdub_shared_raw_timeline(scenario: str) -> None:
    fixture = _momentum_r1b_vdub_fixture(scenario=scenario)

    contract = await run_momentum_r1b_vdub_contract(fixture)

    assert_shadow_contract(contract)
    assert contract.live.source_fingerprint == runtime_source_fingerprint(fixture)
    state = contract.live.state_snapshot or {}
    strategy_state = state.get("strategy_state", {}) or {}
    vdub = strategy_state.get("VdubusNQ_v4", {}) or {}
    nq_regime = strategy_state.get("NQ_REGIME", {}) or {}
    blocked_reasons = state.get("blocked_reasons", {}) or {}
    submitted = [row["strategy_id"] for row in contract.live.order_intents]

    if scenario == "denied":
        assert "VdubusNQ_v4" not in submitted
        assert vdub.get("last_decision_code") == "ENTRY_DENIED"
        assert vdub.get("position_count") == 0
        assert nq_regime.get("position_side") == "long"
        assert "portfolio_rule:regime_disabled" in blocked_reasons.get(
            "VdubusNQ_v4",
            [],
        )
    elif scenario == "contention":
        assert "VdubusNQ_v4" in submitted
        assert vdub.get("position_count") == 1
        assert "NQ_REGIME" not in submitted
        assert nq_regime.get("last_decision_code") == "ENTRY_DENIED"
        assert "portfolio_rule:directional_cap" in blocked_reasons.get(
            "NQ_REGIME",
            [],
        )
    else:
        assert {"NQDTC_v2.1", "VdubusNQ_v4", "NQ_REGIME"}.issubset(submitted)
        assert vdub.get("position_count") == 1
        assert nq_regime.get("position_side") == "long"


def test_momentum_r1b_vdub_fixture_identities_are_scenario_specific() -> None:
    fingerprints = {
        runtime_source_fingerprint(_momentum_r1b_vdub_fixture(scenario=scenario))
        for scenario in ("approved", "denied", "contention")
    }

    assert len(fingerprints) == 3


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["approved", "denied", "contention"])
async def test_momentum_r1b_downturn_shared_raw_timeline(scenario: str) -> None:
    fixture = _momentum_r1b_downturn_fixture(scenario=scenario)

    contract = await run_momentum_r1b_downturn_contract(fixture)

    assert_shadow_contract(contract)
    assert contract.live.source_fingerprint == runtime_source_fingerprint(fixture)
    state = contract.live.state_snapshot or {}
    strategy_state = state.get("strategy_state", {}) or {}
    downturn = strategy_state.get("DownturnDominator_v1", {}) or {}
    nq_regime = strategy_state.get("NQ_REGIME", {}) or {}
    blocked_reasons = state.get("blocked_reasons", {}) or {}
    submitted = [row["strategy_id"] for row in contract.live.order_intents]

    if scenario == "denied":
        assert "DownturnDominator_v1" not in submitted
        assert downturn.get("last_decision_code") == "ENTRY_DENIED"
        assert downturn.get("position_open") is False
        assert "portfolio_rule:regime_disabled" in blocked_reasons.get(
            "DownturnDominator_v1", []
        )
    elif scenario == "contention":
        assert "DownturnDominator_v1" in submitted
        assert downturn.get("working_entry_count") == 1
        assert "NQ_REGIME" not in submitted
        assert nq_regime.get("last_decision_code") == "ENTRY_DENIED"
        assert "portfolio_rule:directional_cap" in blocked_reasons.get(
            "NQ_REGIME", []
        )
    else:
        assert {
            "NQDTC_v2.1",
            "VdubusNQ_v4",
            "DownturnDominator_v1",
            "NQ_REGIME",
        }.issubset(submitted)
        assert downturn.get("working_entry_count") == 1


def test_momentum_r1b_downturn_fixture_identities_are_scenario_specific() -> None:
    fingerprints = {
        runtime_source_fingerprint(_momentum_r1b_downturn_fixture(scenario=scenario))
        for scenario in ("approved", "denied", "contention")
    }

    assert len(fingerprints) == 3


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "terminal_status"),
    [("release_cancel", "CANCELLED"), ("release_reject", "REJECTED")],
)
async def test_momentum_r1b_releases_working_risk_before_later_child(
    scenario: str,
    terminal_status: str,
) -> None:
    fixture = _momentum_r1b_downturn_fixture(scenario=scenario)

    contract = await run_momentum_r1b_downturn_contract(fixture)

    assert_shadow_contract(contract)
    state = contract.live.state_snapshot or {}
    orders = state.get("orders", []) or []
    downturn_entries = [
        row
        for row in orders
        if row.get("strategy_id") == "DownturnDominator_v1"
        and row.get("role") == "ENTRY"
    ]
    submitted = [row["strategy_id"] for row in contract.live.order_intents]
    downturn = (state.get("strategy_state", {}) or {}).get(
        "DownturnDominator_v1", {}
    )

    assert downturn_entries[0]["status"] == terminal_status
    assert downturn.get("working_entry_count") == 0
    assert "NQ_REGIME" in submitted
    assert "portfolio_rule:directional_cap" not in (
        (state.get("blocked_reasons", {}) or {}).get("NQ_REGIME", [])
    )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_momentum_r1b_nq_partial_exit_updates_family_exposure() -> None:
    fixture = _momentum_r1b_downturn_fixture(scenario="nq_partial_exit")

    contract = await run_momentum_r1b_downturn_contract(fixture)

    assert_shadow_contract(contract)
    state = contract.live.state_snapshot or {}
    nq_state = (state.get("strategy_state", {}) or {}).get("NQ_REGIME", {})
    nq_position = next(
        row
        for row in state.get("positions", []) or []
        if row.get("strategy_id") == "NQ_REGIME"
    )
    portfolio = (state.get("portfolio_risk", []) or [])[0]

    assert nq_state.get("qty_open") == 3
    assert nq_position["net_qty"] == 3
    assert nq_position["open_risk_dollars"] == pytest.approx(168.0)
    assert nq_position["realized_pnl"] == pytest.approx(56.0)
    assert portfolio["open_risk_dollars"] == pytest.approx(876.0)


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_momentum_r1b_realized_loss_resizes_later_downturn_quantity() -> None:
    fixture = _momentum_r1b_downturn_fixture(scenario="realized_loss_resize")

    contract = await run_momentum_r1b_downturn_contract(fixture)

    assert_shadow_contract(contract)
    downturn_entries = [
        row
        for row in contract.live.order_intents
        if row.get("strategy_id") == "DownturnDominator_v1"
        and row.get("order_role") == "ENTRY"
    ]
    state = contract.live.state_snapshot or {}
    portfolio = (state.get("portfolio_risk", []) or [])[0]

    assert [row["qty"] for row in downturn_entries] == [42, 21]
    assert portfolio["daily_realized_pnl"] == pytest.approx(-6000.0)
    assert any(
        row.get("strategy_id") == "NQ_REGIME"
        and row.get("realized_pnl") == pytest.approx(-6000.0)
        for row in state.get("positions", [])
    )
    assert any(
        row.get("strategy_id") == "NQ_REGIME"
        and row.get("entry_price") == pytest.approx(19436.0)
        and row.get("qty") == 5
        for row in contract.live.trade_ledger
    )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_entries", "expected_cooldown", "expected_decision"),
    [
        ("downturn_cooldown_blocked", 1, 1, "EXIT_FILLED"),
        ("downturn_cooldown_reentry", 2, 24, "ENTRY_SUBMITTED"),
    ],
)
async def test_momentum_r1b_downturn_fill_drives_cooldown_and_reentry(
    scenario: str,
    expected_entries: int,
    expected_cooldown: int,
    expected_decision: str,
) -> None:
    fixture = _momentum_r1b_downturn_fixture(scenario=scenario)

    contract = await run_momentum_r1b_downturn_contract(fixture)

    assert_shadow_contract(contract)
    entries = [
        row
        for row in contract.live.order_intents
        if row.get("strategy_id") == "DownturnDominator_v1"
        and row.get("order_role") == "ENTRY"
    ]
    downturn = (
        (contract.live.state_snapshot or {}).get("strategy_state", {}) or {}
    ).get("DownturnDominator_v1", {})

    assert len(entries) == expected_entries
    assert downturn.get("bars_since_last_entry") == expected_cooldown
    assert downturn.get("last_decision_code") == expected_decision
    assert downturn.get("position_open") is False


def test_momentum_r1b_feedback_fixture_identities_are_scenario_specific() -> None:
    scenarios = (
        "release_cancel",
        "release_reject",
        "nq_partial_exit",
        "realized_loss_resize",
        "downturn_cooldown_blocked",
        "downturn_cooldown_reentry",
    )
    fingerprints = {
        runtime_source_fingerprint(_momentum_r1b_downturn_fixture(scenario=scenario))
        for scenario in scenarios
    }

    assert len(fingerprints) == len(scenarios)


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


def _momentum_r1b_vdub_fixture(*, scenario: str) -> dict[str, Any]:
    fixture = _momentum_r1b_nqdtc_fixture(scenario="approved")
    fixture["initial_strategy_state"]["VdubusNQ_v4"] = {}
    fixture.setdefault("artifacts", {})["vdub"] = {
        "r1b_market_input": {
            "symbol": "NQ",
            "timestamp": "2026-05-20T14:15:00+00:00",
            "bars_15m": [
                {
                    "symbol": "NQ",
                    "timeframe": "15m",
                    "timestamp": f"2026-05-20T{hour:02d}:{minute:02d}:00+00:00",
                    "open": 19999.0 + index,
                    "high": 20005.0 + index * 3.0,
                    "low": 19998.0,
                    "close": 20001.0 + index * 3.0,
                    "volume": 1000.0 + index * 100.0,
                }
                for index, (hour, minute) in enumerate(
                    [(13, 30), (13, 45), (14, 0), (14, 15)]
                )
            ],
            "bars_1h": [
                {
                    "symbol": "NQ",
                    "timeframe": "1h",
                    "timestamp": "2026-05-20T13:00:00+00:00",
                    "open": 19990.0,
                    "high": 20008.0,
                    "low": 19980.0,
                    "close": 20002.0,
                    "volume": 4000.0,
                },
                {
                    "symbol": "NQ",
                    "timeframe": "1h",
                    "timestamp": "2026-05-20T14:00:00+00:00",
                    "open": 20002.0,
                    "high": 20020.0,
                    "low": 19995.0,
                    "close": 20010.0,
                    "volume": 4500.0,
                },
            ],
            "decision_state": {
                "direction": "LONG",
                "session": "RTH",
                "sub_window": "OPEN",
                "daily_trend": 1,
                "trend_1h": 1,
                "choppiness": 10.0,
                "vol_state": "Normal",
                "class_mult": 0.7,
                "point_value": 2.0,
                "momentum": [float(value) for value in range(60)],
                "atr15": [20.0, 20.0, 20.0, 20.0],
                "atr1h": [30.0, 30.0],
                "svwap": [20000.0, 20000.0, 20000.0, 20000.0],
                "vwap_a": [20000.0, 20000.0, 20000.0, 20000.0],
            },
        }
    }
    strategies = fixture["family_config"]["strategies"]
    strategies.insert(
        1,
        {
            "id": "VdubusNQ_v4",
            "family": "momentum",
            "unit_risk_dollars": 650.0,
            "daily_stop_R": 20.0,
            "priority": 1,
            "max_heat_R": 20.0,
            "max_working_orders": 4,
        },
    )
    strategies[2]["priority"] = 2
    rules = fixture["family_config"]["portfolio_rules"]
    rules["family_strategy_ids"] = [
        "NQDTC_v2.1",
        "VdubusNQ_v4",
        "NQ_REGIME",
    ]
    rules["strategy_priorities"] = [
        ["NQDTC_v2.1", 0],
        ["VdubusNQ_v4", 1],
        ["NQ_REGIME", 2],
    ]
    rules["strategy_size_multipliers"].insert(1, ["VdubusNQ_v4", 1.0])
    rules["disabled_strategies"] = (
        ["VdubusNQ_v4"] if scenario == "denied" else []
    )
    directional_cap = 2.0 if scenario == "contention" else 20.0
    rules["directional_cap_R"] = directional_cap
    rules["directional_cap_long_R"] = directional_cap
    rules["directional_cap_short_R"] = directional_cap

    broker_script = list(fixture["broker_event_script"])
    if scenario == "contention":
        broker_script = [
            event
            for event in broker_script
            if event["order_match"]["strategy_id"] != "NQ_REGIME"
        ]
    if scenario != "denied":
        broker_script.append(
            {
                "order_match": {
                    "strategy_id": "VdubusNQ_v4",
                    "symbol": "NQ",
                    "role": "ENTRY",
                    "side": "BUY",
                    "sequence": 1,
                },
                "event": "fill",
                "exec_id": f"VDUB-R1B-{scenario}",
                "price": 20015.25,
                "commission": 0.0,
                "timestamp": "2026-05-20T14:16:30+00:00",
            }
        )
    fixture["broker_event_script"] = broker_script
    return fixture


def _momentum_r1b_downturn_fixture(*, scenario: str) -> dict[str, Any]:
    fixture = _momentum_r1b_vdub_fixture(scenario="approved")
    fixture["initial_strategy_state"]["DownturnDominator_v1"] = {}
    fixture.setdefault("artifacts", {})["downturn"] = {
        "r1b_market_input": {
            "timeline_id": "downturn_1",
            "symbol": "MNQ",
            "timestamp": "2026-05-20T14:15:00+00:00",
            "bars_5m": [
                {
                    "symbol": "MNQ",
                    "timeframe": "5m",
                    "timestamp": "2026-05-20T14:15:00+00:00",
                    "open": 20001.0,
                    "high": 20002.0,
                    "low": 19996.0,
                    "close": 19998.0,
                    "volume": 1400.0,
                }
            ],
            "bars_15m": [
                {
                    "symbol": "MNQ",
                    "timeframe": "15m",
                    "timestamp": f"2026-05-20T{hour:02d}:{minute:02d}:00+00:00",
                    "open": 20002.0,
                    "high": 20003.0,
                    "low": 19996.0,
                    "close": 19998.0,
                    "volume": 1200.0,
                }
                for hour, minute in [(13, 0), (13, 15), (13, 30), (13, 45), (14, 0), (14, 15)]
            ],
            "decision_state": {
                "vwap": 20000.0,
                "atr_15m": 20.0,
                "atr_1h": 20.0,
                "atr_30m": 16.0,
                "mom_slope_ok": True,
                "session_type": "core",
                "composite_regime": "emerging_bear",
                "vol_state": "normal",
                "vol_factor": 1.0,
                "strong_bear": False,
                "extension_short": False,
                "in_correction": False,
                "bars_since_last_entry": 999,
            },
        }
    }
    strategies = fixture["family_config"]["strategies"]
    strategies.insert(
        2,
        {
            "id": "DownturnDominator_v1",
            "family": "momentum",
            "unit_risk_dollars": 400.0,
            "daily_stop_R": 20.0,
            "priority": 2,
            "max_heat_R": 20.0,
            "max_working_orders": 4,
        },
    )
    strategies[3]["priority"] = 3
    rules = fixture["family_config"]["portfolio_rules"]
    rules["family_strategy_ids"] = [
        "NQDTC_v2.1",
        "VdubusNQ_v4",
        "DownturnDominator_v1",
        "NQ_REGIME",
    ]
    rules["strategy_priorities"] = [
        ["NQDTC_v2.1", 0],
        ["VdubusNQ_v4", 1],
        ["DownturnDominator_v1", 2],
        ["NQ_REGIME", 3],
    ]
    rules["strategy_size_multipliers"].insert(
        2, ["DownturnDominator_v1", 1.0]
    )
    rules["disabled_strategies"] = (
        ["DownturnDominator_v1"] if scenario == "denied" else []
    )
    if scenario in {"contention", "release_cancel", "release_reject"}:
        rules["directional_cap_R"] = 20.0
        rules["directional_cap_long_R"] = 20.0
        rules["directional_cap_short_R"] = 2.5
        _set_nq_regime_short_breakout(fixture)
    else:
        rules["directional_cap_R"] = 20.0
        rules["directional_cap_long_R"] = 20.0
        rules["directional_cap_short_R"] = 20.0

    broker_script = [
        event
        for event in fixture["broker_event_script"]
        if event["order_match"]["strategy_id"] != "NQ_REGIME"
        or scenario != "contention"
    ]
    if scenario in {"release_cancel", "release_reject"}:
        broker_script = [
            event
            for event in broker_script
            if event["order_match"]["strategy_id"] != "NQ_REGIME"
        ]
        event_type = "status" if scenario == "release_cancel" else "reject"
        release_event = {
            "order_match": {
                "strategy_id": "DownturnDominator_v1",
                "symbol": "MNQ",
                "role": "ENTRY",
                "side": "SELL",
                "sequence": 1,
            },
            "event": event_type,
            "timestamp": "2026-05-20T14:15:30+00:00",
            "apply_after": "downturn_1",
        }
        if event_type == "status":
            release_event.update({"status": "Cancelled", "remaining": 42.0})
        else:
            release_event.update(
                {"reason": "fixture_reject", "error_code": 201, "retryable": False}
            )
        broker_script.append(release_event)
    elif scenario == "nq_partial_exit":
        _phase_nq_entry_fill(broker_script)
        broker_script.append(
            {
                "order_match": {
                    "strategy_id": "NQ_REGIME",
                    "symbol": "MNQ",
                    "role": "TP",
                    "side": "SELL",
                    "sequence": 1,
                },
                "event": "fill",
                "exec_id": "NQREG-R1B-PARTIAL",
                "price": 20050.0,
                "qty": 2,
                "commission": 0.0,
                "timestamp": "2026-05-20T14:17:30+00:00",
                "apply_after": "nq_regime_1",
            }
        )
    elif scenario == "realized_loss_resize":
        rules["dd_tiers"] = [[0.05, 1.0], [0.10, 0.50], [1.0, 0.25]]
        broker_script.append(_downturn_cancel_event())
        _phase_nq_entry_fill(broker_script)
        broker_script.append(
            {
                "order_match": {
                    "strategy_id": "NQ_REGIME",
                    "symbol": "MNQ",
                    "role": "STOP",
                    "side": "SELL",
                    "sequence": 1,
                },
                "event": "fill",
                "exec_id": "NQREG-R1B-LOSS",
                "price": 19436.0,
                "qty": 5,
                "commission": 0.0,
                "timestamp": "2026-05-20T14:18:00+00:00",
                "apply_after": "nq_regime_1",
            }
        )
        _add_downturn_followup(fixture, cooldown_bars=None, allow_no_signal=False)
    elif scenario in {"downturn_cooldown_blocked", "downturn_cooldown_reentry"}:
        broker_script.extend(_downturn_round_trip_events())
        _add_downturn_followup(
            fixture,
            cooldown_bars=(
                1 if scenario == "downturn_cooldown_blocked" else 24
            ),
            allow_no_signal=scenario == "downturn_cooldown_blocked",
        )
    fixture["broker_event_script"] = broker_script
    return fixture


def _set_nq_regime_short_breakout(fixture: dict[str, Any]) -> None:
    fixture["bars"] = [
        {
            "symbol": "NQ",
            "timeframe": "5m",
            "timestamp": "2026-05-20T14:15:00+00:00",
            "open": 19894.0,
            "high": 19898.0,
            "low": 19880.0,
            "close": 19884.0,
            "volume": 3000.0,
        }
    ]
    fixture["initial_strategy_state"]["NQ_REGIME"]["bars_5m"] = [
        {
            "ts": "2026-05-20T10:05:00-04:00",
            "open": 19898.0,
            "high": 19899.0,
            "low": 19892.0,
            "close": 19895.0,
            "volume": 1000.0,
            "vwap": None,
        },
        {
            "ts": "2026-05-20T10:10:00-04:00",
            "open": 19895.0,
            "high": 19897.0,
            "low": 19890.0,
            "close": 19893.0,
            "volume": 1000.0,
            "vwap": None,
        },
    ]
    fixture["artifacts"]["nq_regime"]["daily_context"].update(
        {"pdl": 19600.0, "pdm": 19800.0, "weekly_low": 19000.0}
    )


def _phase_nq_entry_fill(broker_script: list[dict[str, Any]]) -> None:
    for event in broker_script:
        match = event.get("order_match", {}) or {}
        if match.get("strategy_id") == "NQ_REGIME" and match.get("role") == "ENTRY":
            event["apply_after"] = "nq_regime_1"


def _downturn_cancel_event() -> dict[str, Any]:
    return {
        "order_match": {
            "strategy_id": "DownturnDominator_v1",
            "symbol": "MNQ",
            "role": "ENTRY",
            "side": "SELL",
            "sequence": 1,
        },
        "event": "status",
        "status": "Cancelled",
        "remaining": 42.0,
        "timestamp": "2026-05-20T14:15:30+00:00",
        "apply_after": "downturn_1",
    }


def _downturn_round_trip_events() -> list[dict[str, Any]]:
    return [
        {
            "order_match": {
                "strategy_id": "DownturnDominator_v1",
                "symbol": "MNQ",
                "role": "ENTRY",
                "side": "SELL",
                "sequence": 1,
            },
            "event": "fill",
            "exec_id": "DOWNTURN-R1B-ENTRY",
            "price": 19996.5,
            "qty": 42,
            "commission": 0.0,
            "timestamp": "2026-05-20T14:15:30+00:00",
            "apply_after": "downturn_1",
        },
        {
            "order_match": {
                "strategy_id": "DownturnDominator_v1",
                "symbol": "MNQ",
                "role": "STOP",
                "side": "BUY",
                "sequence": 1,
            },
            "event": "fill",
            "exec_id": "DOWNTURN-R1B-STOP",
            "price": 19996.5,
            "qty": 42,
            "commission": 0.0,
            "timestamp": "2026-05-20T14:16:00+00:00",
            "apply_after": "downturn_1",
        },
    ]


def _add_downturn_followup(
    fixture: dict[str, Any],
    *,
    cooldown_bars: int | None,
    allow_no_signal: bool,
) -> None:
    primary = fixture["artifacts"]["downturn"]["r1b_market_input"]
    followup = deepcopy(primary)
    followup["timeline_id"] = "downturn_2"
    followup["timestamp"] = "2026-05-20T14:20:00+00:00"
    if cooldown_bars is None:
        followup["bars_5m"] = [
            {
                **dict(primary["bars_5m"][-1]),
                "timestamp": "2026-05-20T14:20:00+00:00",
            }
        ]
        followup["bars_15m"] = [
            {**row, "timestamp": "2026-05-20T14:20:00+00:00"}
            if index == len(primary["bars_15m"]) - 1
            else dict(row)
            for index, row in enumerate(primary["bars_15m"])
        ]
        followup["decision_state"].pop("bars_since_last_entry", None)
    else:
        followup["bars_5m"] = [
            {
                "symbol": "MNQ",
                "timeframe": "5m",
                "timestamp": "2026-05-20T14:20:00+00:00",
                "open": 19820.0,
                "high": 19824.0,
                "low": 19796.0,
                "close": 19800.0,
                "volume": 1600.0,
            }
        ]
        closes = [20020.0, 20010.0, 20000.0, 19980.0, 19960.0, 19800.0]
        followup["bars_15m"] = [
            {
                "symbol": "MNQ",
                "timeframe": "15m",
                "timestamp": f"2026-05-20T{hour:02d}:{minute:02d}:00+00:00",
                "open": close + 4.0,
                "high": close + 8.0,
                "low": close - 8.0,
                "close": close,
                "volume": 1300.0,
            }
            for close, (hour, minute) in zip(
                closes,
                [(13, 5), (13, 20), (13, 35), (13, 50), (14, 5), (14, 20)],
                strict=True,
            )
        ]
        followup["decision_state"].update(
            {
                "vwap": 19000.0,
                "ema_fast_15m": 20050.0,
                "elapsed_5m_bars": cooldown_bars,
                "allow_no_signal": allow_no_signal,
            }
        )
        followup["decision_state"].pop("bars_since_last_entry", None)
    fixture["artifacts"]["downturn"]["r1b_market_inputs"] = [
        primary,
        followup,
    ]
