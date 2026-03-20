import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.oms.models.risk_state import StrategyRiskState
from shared.oms.persistence.schema import RiskDailyPortfolioRow, RiskDailyStrategyRow
from shared.oms.services.factory import (
    _aggregate_strategy_daily_rows,
    _build_portfolio_daily_row,
    _build_strategy_daily_row,
)


AS_OF = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
TRADE_DAY = date(2026, 3, 10)


def test_aggregate_strategy_daily_rows_sums_r_usd_and_strategy_breakdown():
    rows = [
        RiskDailyStrategyRow(
            trade_date=TRADE_DAY,
            strategy_id="Helix",
            daily_realized_r=Decimal("1.25"),
            daily_realized_usd=Decimal("500.0"),
        ),
        RiskDailyStrategyRow(
            trade_date=TRADE_DAY,
            strategy_id="NQDTC",
            daily_realized_r=Decimal("-0.50"),
            daily_realized_usd=Decimal("-140.0"),
        ),
    ]

    daily_r, daily_usd, strategy_daily_pnl = _aggregate_strategy_daily_rows(rows)

    assert daily_r == 0.75
    assert daily_usd == 360.0
    assert strategy_daily_pnl == {"Helix": 500.0, "NQDTC": -140.0}


def test_build_strategy_daily_row_preserves_existing_halt_and_entry_count():
    state = StrategyRiskState(
        strategy_id="Vdubus",
        trade_date=TRADE_DAY,
        daily_realized_pnl=-225.0,
        daily_realized_R=-0.9,
        open_risk_R=0.6,
    )
    existing = RiskDailyStrategyRow(
        trade_date=TRADE_DAY,
        strategy_id="Vdubus",
        filled_entries=3,
        halted=True,
        halt_reason="manual halt",
    )

    row = _build_strategy_daily_row(state, existing, AS_OF)

    assert row.daily_realized_r == Decimal("-0.9")
    assert row.daily_realized_usd == Decimal("-225.0")
    assert row.open_risk_r == Decimal("0.6")
    assert row.filled_entries == 3
    assert row.halted is True
    assert row.halt_reason == "manual halt"
    assert row.last_update_at == AS_OF


def test_build_portfolio_daily_row_aggregates_rows_and_preserves_manual_halt():
    daily_rows = [
        RiskDailyStrategyRow(
            trade_date=TRADE_DAY,
            strategy_id="Helix",
            daily_realized_r=Decimal("1.25"),
            daily_realized_usd=Decimal("500.0"),
        ),
        RiskDailyStrategyRow(
            trade_date=TRADE_DAY,
            strategy_id="NQDTC",
            daily_realized_r=Decimal("-0.50"),
            daily_realized_usd=Decimal("-140.0"),
        ),
    ]
    existing = RiskDailyPortfolioRow(
        trade_date=TRADE_DAY,
        halted=True,
        halt_reason="operator halt",
    )

    row = _build_portfolio_daily_row(
        trade_date=TRADE_DAY,
        daily_rows=daily_rows,
        open_risk_r=1.8,
        existing_row=existing,
        as_of=AS_OF,
    )

    assert row.daily_realized_r == Decimal("0.75")
    assert row.daily_realized_usd == Decimal("360.0")
    assert row.portfolio_open_risk_r == Decimal("1.8")
    assert row.halted is True
    assert row.halt_reason == "operator halt"
    assert row.last_update_at == AS_OF


def test_build_portfolio_daily_row_falls_back_to_existing_snapshot_when_rows_missing():
    existing = RiskDailyPortfolioRow(
        trade_date=TRADE_DAY,
        daily_realized_r=Decimal("-1.5"),
        daily_realized_usd=Decimal("-420.0"),
        halted=False,
    )

    row = _build_portfolio_daily_row(
        trade_date=TRADE_DAY,
        daily_rows=[],
        open_risk_r=0.0,
        existing_row=existing,
        as_of=AS_OF,
    )

    assert row.daily_realized_r == Decimal("-1.5")
    assert row.daily_realized_usd == Decimal("-420.0")
    assert row.portfolio_open_risk_r == Decimal("0.0")
