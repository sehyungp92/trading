from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtests.momentum.analysis.family_portfolio_diagnostics import (
    _distribution,
    _outcome_diagnostics,
    _synergy_assessment,
)
from backtests.momentum.engine.family_portfolio_engine import FamilyPortfolioTrade


def _trade(r_multiple: float, raw_pnl: float) -> FamilyPortfolioTrade:
    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    return FamilyPortfolioTrade(
        strategy_id="NQDTC_v2.1",
        direction=1,
        entry_time=timestamp,
        exit_time=timestamp,
        entry_price=20_000.0,
        exit_price=20_001.0,
        initial_stop=19_999.0,
        raw_pnl_dollars=raw_pnl,
        raw_qty=1,
        r_multiple=r_multiple,
    )


def test_distribution_reports_positive_and_nonpositive_r() -> None:
    result = _distribution([2.0, 1.0, -1.5, -0.5])

    assert result["count"] == 4
    assert result["positive_count"] == 2
    assert result["nonpositive_count"] == 2
    assert result["win_rate"] == 0.5
    assert result["positive_total"] == 3.0
    assert result["nonpositive_total"] == -2.0
    assert result["total"] == 1.0
    assert result["median"] == 0.25


def test_outcome_diagnostics_reconcile_and_price_forgone_alpha() -> None:
    accepted = [_trade(1.0, 100.0), _trade(-0.25, -25.0)]
    blocked = [_trade(2.0, 200.0), _trade(-1.0, -100.0)]

    result = _outcome_diagnostics([*accepted, *blocked], accepted, blocked)

    assert result["reconciliation"] == {
        "fired": 4,
        "accepted": 2,
        "blocked": 2,
        "reconciles": True,
    }
    assert result["avoided_loss_r"] == 1.0
    assert result["forgone_gain_r"] == 2.0
    assert result["net_block_value_r"] == -1.0
    assert result["net_block_value_raw_dollars"] == -100.0
    assert result["blocker_precision_nonpositive"] == 0.5


def test_synergy_verdict_requires_blocker_and_risk_return_quality() -> None:
    optimized_metrics = {
        "net_profit": 90.0,
        "total_trades": 95.0,
        "max_drawdown_pct": 0.08,
        "calmar": 3.0,
        "profit_factor": 2.0,
        "block_rate": 0.05,
        "trades_per_month": 45.0,
    }
    relaxed_metrics = {
        "net_profit": 100.0,
        "total_trades": 100.0,
        "max_drawdown_pct": 0.10,
        "calmar": 2.5,
        "profit_factor": 1.8,
    }
    assessment = _synergy_assessment(
        headline_metrics={"active_strategies": 4.0},
        outcome_diagnostics={
            "net_block_value_r": 2.0,
            "realized_r_discrimination": 0.2,
            "nonpositive_trade_block_rate": 0.20,
            "positive_trade_block_rate": 0.05,
        },
        scenario_comparison={
            "optimized_live_rules": {"metrics": optimized_metrics},
            "same_allocations_relaxed_shared_caps": {"metrics": relaxed_metrics},
        },
        score_diagnostics={
            "optimized": {"score": 0.90},
            "relaxed_shared_caps": {"score": 0.80},
        },
    )

    assert assessment["maximized"] is True
    assert assessment["verdict"] == "maximized_among_tested_evidence_not_global"
    assert assessment["trade_capture_vs_relaxed"] == pytest.approx(0.95)
    assert assessment["blocker_quality_passes"] is True
    assert assessment["risk_return_passes"] is True
