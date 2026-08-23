from __future__ import annotations

from backtests.stock.auto.runners.finalize_iaric_unified_round3 import _metric_agreement


def test_round3_finalization_requires_exact_executable_metric_reconciliation() -> None:
    metrics = {
        "total_trades": 110,
        "expected_total_r": 32.5,
        "avg_r": 0.2954545454545,
        "profit_factor": 1.8,
        "max_drawdown_pct": 0.04,
        "net_profit": 3100.0,
    }

    assert _metric_agreement(metrics, dict(metrics))["passed"] is True

    changed = dict(metrics)
    changed["total_trades"] = 109
    result = _metric_agreement(metrics, changed)
    assert result["passed"] is False
    assert result["checks"]["total_trades"]["passed"] is False
