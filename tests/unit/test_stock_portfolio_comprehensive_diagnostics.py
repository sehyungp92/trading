from __future__ import annotations

import pytest

from backtests.stock.auto.portfolio_synergy.comprehensive_diagnostics import (
    _window_synergy,
)
from backtests.stock.auto.portfolio_synergy.report_matched_performance import (
    _distribution,
)


def _metrics(**overrides: float) -> dict[str, float]:
    baseline = {
        "net_return_pct": 0.10,
        "total_r": 10.0,
        "profit_factor": 1.5,
        "win_rate": 0.55,
        "max_drawdown_pct_mtm_daily": 0.08,
        "certainty_equivalent_growth": 0.20,
    }
    baseline.update(overrides)
    return baseline


def test_distribution_reconciles_block_value_inputs() -> None:
    distribution = _distribution([2.0, 1.0, -1.5, -0.5])

    assert distribution["count"] == 4
    assert distribution["positive_count"] == 2
    assert distribution["nonpositive_count"] == 2
    assert distribution["win_rate"] == 0.5
    assert distribution["positive_total"] == 3.0
    assert distribution["nonpositive_total"] == -2.0
    assert distribution["total"] == 1.0
    assert distribution["average"] == 0.25
    assert distribution["median"] == 0.25


def test_window_synergy_separates_overlay_and_risk_baseline_effects() -> None:
    window = {
        "post_optimization_portfolio": _metrics(
            net_return_pct=0.24,
            total_r=21.0,
            max_drawdown_pct_mtm_daily=0.09,
            certainty_equivalent_growth=0.30,
        ),
        "post_optimization_no_overlay": _metrics(
            net_return_pct=0.20,
            total_r=20.0,
            max_drawdown_pct_mtm_daily=0.11,
            certainty_equivalent_growth=0.27,
        ),
        "pre_optimization_portfolio": _metrics(
            net_return_pct=0.22,
            total_r=19.0,
            max_drawdown_pct_mtm_daily=0.07,
            certainty_equivalent_growth=0.31,
        ),
        "alcb_round3_standalone_native_risk": _metrics(
            net_return_pct=0.15,
            total_r=15.0,
            max_drawdown_pct_mtm_daily=0.04,
        ),
        "iaric_round3_standalone_native_risk": _metrics(
            net_return_pct=0.05,
            total_r=5.0,
            max_drawdown_pct_mtm_daily=0.08,
        ),
        "alcb_standalone_post_risk": _metrics(
            net_return_pct=0.16,
            total_r=16.0,
        ),
        "iaric_standalone_post_risk": _metrics(
            net_return_pct=0.05,
            total_r=5.0,
        ),
    }

    diagnostics = _window_synergy(window)

    assert diagnostics["overlay_return_delta"] == pytest.approx(0.04)
    assert diagnostics["overlay_total_r_delta"] == pytest.approx(1.0)
    assert diagnostics["overlay_drawdown_delta"] == pytest.approx(-0.02)
    assert diagnostics["overlay_ce_delta"] == pytest.approx(0.03)
    assert diagnostics["post_minus_pre_drawdown"] == pytest.approx(0.02)
    assert diagnostics["portfolio_r_capture_vs_native_standalones"] == pytest.approx(
        1.05
    )
    assert diagnostics[
        "portfolio_r_capture_vs_post_risk_standalones"
    ] == pytest.approx(1.0)
    assert diagnostics["post_drawdown_minus_worst_native_standalone"] == pytest.approx(
        0.01
    )
