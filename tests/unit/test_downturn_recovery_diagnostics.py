from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

from backtests.momentum.auto.downturn.phase_diagnostics import (
    _write_d9_dynamic_risk_sizing,
)
from backtests.momentum.auto.downturn.round5_requalify import (
    INDIVIDUAL_STRATEGY_EQUITY,
    INITIAL_EQUITY,
    IS_START,
    OOS_CUTOFF,
    OUTPUT_DIR,
    STUDY_END,
)
from strategies.momentum.downturn.bt_models import DownturnTradeRecord


def test_recovery_runner_uses_canonical_individual_strategy_equity() -> None:
    assert INDIVIDUAL_STRATEGY_EQUITY == 10_000.0
    assert INITIAL_EQUITY == INDIVIDUAL_STRATEGY_EQUITY
    assert "50k" not in OUTPUT_DIR.name.lower()
    assert IS_START.isoformat() == "2024-01-01T00:00:00+00:00"
    assert OOS_CUTOFF.isoformat() == "2026-03-21T00:00:00+00:00"
    assert STUDY_END.isoformat() == "2026-05-02T00:00:00+00:00"


def test_dynamic_sizing_diagnostic_identifies_one_contract_floor() -> None:
    trades = [
        DownturnTradeRecord(
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            entry_price=20_000.0,
            stop0=20_100.0,
            qty=1,
            pnl=100.0,
        ),
        DownturnTradeRecord(
            entry_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
            entry_price=20_000.0,
            stop0=20_150.0,
            qty=1,
            pnl=-50.0,
        ),
    ]
    output = StringIO()

    _write_d9_dynamic_risk_sizing(
        output,
        trades,
        initial_equity=10_000.0,
        point_value=2.0,
        base_risk_pct=0.0064,
    )

    report = output.getvalue()
    assert "Filled quantity distribution: 1x=2" in report
    assert "One-contract floor overrides: 2/2 (100.0%)" in report
    assert "Sizing verdict: INEFFECTIVE" in report
