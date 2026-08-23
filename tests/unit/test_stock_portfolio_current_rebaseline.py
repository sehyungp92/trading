from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtests.stock.auto.portfolio_synergy.core.logic import run_portfolio_replay
from backtests.stock.auto.portfolio_synergy.core.market import CausalPriceBook
from backtests.stock.auto.portfolio_synergy.evaluator import load_trade_records
from backtests.stock.auto.portfolio_synergy.run_current_rebaseline import neutral_config
from backtests.stock.auto.portfolio_synergy.run_corrected_phased_auto import (
    IS_END,
    _expanding_predictions,
    _filter,
)
from backtests.stock.models import Direction, TradeRecord


def _trade(
    symbol: str,
    entry: datetime,
    exit_: datetime,
    r_multiple: float,
) -> TradeRecord:
    return TradeRecord(
        strategy="test",
        symbol=symbol,
        direction=Direction.LONG,
        entry_time=entry,
        exit_time=exit_,
        entry_price=100.0,
        exit_price=100.0 + r_multiple,
        quantity=10,
        pnl=r_multiple * 100.0,
        r_multiple=r_multiple,
        risk_per_share=10.0,
        commission=0.0,
        slippage=0.0,
        entry_type="DAILY_RESIDUAL_REVERSION",
        sector="Technology",
        metadata={"residual_score": 50.0},
    )


def test_residual_artifact_loader_maps_native_risk_and_quality(tmp_path) -> None:
    path = tmp_path / "trades.json"
    path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "sector": "Technology",
                    "entry_time": "2026-04-01 13:30:00+00:00",
                    "exit_time": "2026-04-02 13:30:00+00:00",
                    "entry_price": 100.0,
                    "exit_price": 102.0,
                    "qty_entry": 10,
                    "initial_risk_dollars": 200.0,
                    "gross_pnl": 20.0,
                    "net_pnl": 19.0,
                    "commission": 1.0,
                    "r_multiple": 0.095,
                    "score": 55.0,
                    "failed_continuation_r": 0.4,
                    "sector_return_5d": -0.01,
                    "factor_model": "market_sector_peer",
                    "formation_sessions": 1,
                    "residual_lane_id": "test",
                    "exit_reason": "time_stop",
                    "held_sessions": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    trade = load_trade_records(path)[0]

    assert trade.entry_type == "DAILY_RESIDUAL_REVERSION"
    assert trade.risk_per_share == 20.0
    assert trade.pnl_net == 19.0
    assert trade.metadata["residual_score"] == 55.0


def test_archived_trade_loader_accepts_named_direction(tmp_path) -> None:
    path = tmp_path / "trades.json"
    path.write_text(
        json.dumps(
            [
                {
                    "strategy": "IARIC_PB",
                    "symbol": "AAPL",
                    "direction": "SHORT",
                    "entry_time": "2026-04-01T13:30:00+00:00",
                    "exit_time": "2026-04-01T14:30:00+00:00",
                    "entry_price": 100.0,
                    "exit_price": 99.0,
                    "quantity": 10,
                    "pnl": 10.0,
                    "r_multiple": 0.1,
                    "risk_per_share": 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_trade_records(path)[0].direction == Direction.SHORT


def test_intraday_slot_reserve_prevents_residual_sleeve_from_consuming_capacity() -> None:
    config = neutral_config()
    config["portfolio_rules"]["max_total_active_positions"] = 3
    config["cross_strategy_rules"]["intraday_reserved_slots"] = 2
    entry = datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc)
    exit_ = datetime(2026, 1, 7, 13, 30, tzinfo=timezone.utc)

    result = run_portfolio_replay([], [_trade("AAPL", entry, exit_, 1.0), _trade("MSFT", entry, exit_, 1.0)], config)

    assert result.metrics["entries_accepted_by_portfolio"] == 1
    assert result.metrics["blocked_reason_intraday_reserved_slots"] == 1


def test_realized_daily_loss_stop_is_enforced_before_next_entry() -> None:
    config = neutral_config()
    config["portfolio_rules"]["portfolio_daily_stop_R"] = 2.0
    config["strategy_allocations"]["ALCB_R3"]["daily_stop_R"] = 0.0
    first_entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    first_exit = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    second_entry = datetime(2026, 1, 5, 15, 5, tzinfo=timezone.utc)
    second_exit = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)

    result = run_portfolio_replay(
        [_trade("AAPL", first_entry, first_exit, -3.0), _trade("MSFT", second_entry, second_exit, 1.0)],
        [],
        config,
    )

    assert result.metrics["entries_accepted_by_portfolio"] == 1
    assert result.metrics["blocked_reason_portfolio_daily_stop"] == 1


def test_corrected_overlay_does_not_reapply_native_strategy_daily_stop() -> None:
    config = neutral_config()
    config["cross_strategy_rules"]["apply_duplicate_native_limits"] = False
    config["strategy_allocations"]["ALCB_R3"]["daily_stop_R"] = 1.0
    first_entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    second_entry = datetime(2026, 1, 5, 15, 5, tzinfo=timezone.utc)
    trades = [
        _trade("AAPL", first_entry, first_entry.replace(hour=15), -2.0),
        _trade("MSFT", second_entry, second_entry.replace(hour=16), 1.0),
    ]

    result = run_portfolio_replay(trades, [], config)

    assert result.metrics["entries_accepted_by_portfolio"] == 2
    assert "blocked_reason_strategy_daily_stop" not in result.metrics


def test_alpha_floor_uses_calibrated_expected_r_not_hindsight_outcome() -> None:
    config = neutral_config()
    config["cross_strategy_rules"].update(
        {
            "alpha_admission_enabled": True,
            "minimum_expected_r": 0.05,
        }
    )
    entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    trade = _trade("AAPL", entry, entry.replace(hour=15), 2.0)
    trade.metadata["portfolio_expected_r"] = -0.10

    result = run_portfolio_replay([trade], [], config)

    assert result.metrics["entries_accepted_by_portfolio"] == 0
    assert result.metrics["blocked_reason_alpha_floor"] == 1


def test_drawdown_halt_preserves_signal_and_block_counts() -> None:
    config = neutral_config()
    config["initial_equity"] = 1_000.0
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.10
    config["portfolio_rules"]["drawdown_tiers"] = ((0.10, 0.0),)
    first_entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    second_entry = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
    trades = [
        _trade("AAPL", first_entry, first_entry.replace(hour=15), -2.0),
        _trade("MSFT", second_entry, second_entry.replace(hour=15), 1.0),
    ]

    result = run_portfolio_replay(trades, [], config)

    assert result.metrics["entry_signals_fired"] == 2
    assert result.metrics["blocked_reason_drawdown_halt"] == 1


def test_corrected_split_purges_trade_whose_outcome_crosses_boundary() -> None:
    entry = datetime(2026, 2, 27, 14, 0, tzinfo=timezone.utc)
    crossing = _trade(
        "AAPL",
        entry,
        datetime(2026, 3, 3, 14, 0, tzinfo=timezone.utc),
        1.0,
    )

    assert _filter([crossing], entry.date(), IS_END) == ()


def test_oos_alpha_prediction_does_not_use_oos_outcome() -> None:
    training = []
    for index in range(50):
        entry = datetime(2025, 12, 1 + index % 20, 14, 0, tzinfo=timezone.utc)
        training.append(
            _trade(
                f"S{index}",
                entry,
                entry.replace(hour=15),
                0.1 + 0.01 * (index % 3),
            )
        )
    oos_entry = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc)
    oos = _trade("OOS", oos_entry, oos_entry.replace(hour=15), 5.0)
    trades = tuple([*training, oos])

    first = _expanding_predictions(trades, ridge=10.0, train_end=IS_END)[-1]
    mutated = tuple([*training, replace(oos, r_multiple=-5.0)])
    second = _expanding_predictions(mutated, ridge=10.0, train_end=IS_END)[-1]

    assert first == second


def test_capacity_resize_reduces_risk_instead_of_blocking_positive_signal() -> None:
    config = neutral_config()
    config["initial_equity"] = 1_000.0
    config["portfolio_rules"].update(
        {
            "reference_risk_pct": 0.01,
            "heat_cap_R": 1.0,
            "max_long_heat_R": 1.0,
            "max_symbol_heat_R": 99.0,
            "max_total_active_positions": 99,
        }
    )
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.0075
    config["cross_strategy_rules"].update(
        {
            "capacity_action": "resize",
            "minimum_capacity_size_mult": 0.30,
            "same_sector_heat_cap_R": 99.0,
        }
    )
    entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    first = _trade("AAPL", entry, entry.replace(hour=16), 1.0)
    second = _trade("MSFT", entry.replace(minute=5), entry.replace(hour=16), 1.0)
    first.quantity = second.quantity = 1_000
    first.risk_per_share = second.risk_per_share = 0.1

    result = run_portfolio_replay([first, second], [], config)

    assert result.metrics["entries_accepted_by_portfolio"] == 2
    assert result.state.accepted_positions[1].risk_dollars < result.state.accepted_positions[0].risk_dollars


def test_integer_share_floor_blocks_unaffordable_risk_unit() -> None:
    config = neutral_config()
    config["initial_equity"] = 100.0
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.001
    entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)

    result = run_portfolio_replay(
        [_trade("AAPL", entry, entry + timedelta(hours=1), 1.0)],
        [],
        config,
    )

    assert result.metrics["entries_accepted_by_portfolio"] == 0
    assert result.metrics["blocked_reason_quantity_below_one"] == 1


def test_shared_account_resizes_to_notional_capacity() -> None:
    config = neutral_config()
    config["initial_equity"] = 1_000.0
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.01
    config["account_rules"] = {
        "enforce_shared_buying_power": True,
        "oversize_action": "resize",
        "max_gross_notional_pct": 0.50,
        "max_net_notional_pct": 0.50,
        "initial_margin_long_pct": 0.50,
    }
    entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    trade = _trade("AAPL", entry, entry + timedelta(hours=1), 1.0)
    trade.risk_per_share = 1.0

    result = run_portfolio_replay([trade], [], config)

    assert result.metrics["entries_accepted_by_portfolio"] == 1
    assert result.state.accepted_positions[0].quantity == 5
    assert result.metrics["gross_leverage_peak"] <= 0.50
    assert result.metrics["margin_breach_count"] == 0


def test_causal_unrealized_loss_reduces_next_trade_size() -> None:
    config = neutral_config()
    config["initial_equity"] = 1_000.0
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.10
    first_entry = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    second_entry = first_entry + timedelta(hours=1)
    exit_ = first_entry + timedelta(hours=3)
    first = _trade("AAPL", first_entry, exit_, 0.0)
    second = _trade("MSFT", second_entry, exit_, 0.0)

    def marks(symbol: str, at: datetime) -> float:
        if symbol == "AAPL" and at >= second_entry:
            return 50.0
        return 100.0

    result = run_portfolio_replay(
        [first, second],
        [],
        config,
        mark_price_provider=marks,
    )

    assert result.state.accepted_positions[0].quantity == 10
    assert result.state.accepted_positions[1].quantity == 5
    assert result.metrics["mark_coverage_ratio"] == 1.0


def test_margin_debit_financing_is_charged_to_equity() -> None:
    config = neutral_config()
    config["initial_equity"] = 1_000.0
    config["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] = 0.02
    config["account_rules"] = {"annual_margin_interest_rate": 0.10}
    entry = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
    trade = _trade("AAPL", entry, entry + timedelta(days=365), 0.0)
    trade.risk_per_share = 1.0

    result = run_portfolio_replay([trade], [], config)

    assert 99.0 < result.metrics["financing_cost"] < 101.0
    assert 899.0 < result.state.equity < 901.0


def test_causal_price_book_uses_only_completed_prior_bars() -> None:
    first = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    second = first + timedelta(minutes=5)
    bars = pd.DataFrame(
        {"close": [100.0, 90.0]},
        index=pd.DatetimeIndex([first, second]),
    )
    daily = pd.DataFrame(
        {"MSFT": [200.0, 210.0]},
        index=pd.DatetimeIndex(
            [
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                datetime(2026, 1, 5, tzinfo=timezone.utc),
            ]
        ),
    )
    book = CausalPriceBook({"AAPL": bars}, daily_close=daily)

    assert book("AAPL", second) == 100.0
    assert book("AAPL", second + timedelta(seconds=1)) == 90.0
    assert book("MSFT", datetime(2026, 1, 5, 15, tzinfo=timezone.utc)) == 200.0
