"""Walk-forward optimization for stock backtests.

Splits the backtest period into rolling windows:
- In-sample (IS): optimize parameters
- Out-of-sample (OOS): validate with best IS parameters

This guards against overfitting by ensuring parameters are always
tested on unseen data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from research.backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import TradeRecord
from research.backtests.stock.optimization.param_space import ParamRange
from research.backtests.stock.optimization.runner import optimize_alcb, optimize_iaric

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single IS/OOS window."""

    window_id: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    best_params: dict[str, float]
    is_metrics: PerformanceMetrics | None = None
    oos_metrics: PerformanceMetrics | None = None
    oos_trades: list[TradeRecord] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result."""

    windows: list[WalkForwardWindow]
    combined_oos_trades: list[TradeRecord]
    combined_metrics: PerformanceMetrics | None = None

    @property
    def efficiency(self) -> float:
        """OOS/IS Sharpe ratio — values > 0.5 suggest robust parameters."""
        if not self.windows:
            return 0.0
        is_sharpes = [w.is_metrics.sharpe for w in self.windows if w.is_metrics]
        oos_sharpes = [w.oos_metrics.sharpe for w in self.windows if w.oos_metrics]
        if not is_sharpes or not oos_sharpes:
            return 0.0
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        return float(avg_oos / avg_is) if avg_is > 0 else 0.0


def _date_add_months(d: date, months: int) -> date:
    """Add months to a date (approximate)."""
    year = d.year + (d.month + months - 1) // 12
    month = (d.month + months - 1) % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)


def walk_forward_alcb(
    replay: ResearchReplayEngine,
    param_space: list[ParamRange],
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    is_months: int = 6,
    oos_months: int = 3,
    step_months: int = 3,
    initial_equity: float = 10_000.0,
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Run walk-forward optimization for ALCB."""
    from dataclasses import replace
    from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    windows: list[WalkForwardWindow] = []
    window_id = 0
    current = start

    while True:
        is_start = current
        is_end = _date_add_months(current, is_months)
        oos_start = is_end
        oos_end = _date_add_months(is_end, oos_months)

        if oos_end > end:
            oos_end = end
        if oos_start >= end:
            break

        logger.info(
            "Window %d: IS [%s, %s] → OOS [%s, %s]",
            window_id, is_start, is_end, oos_start, oos_end,
        )

        # In-sample optimization
        is_config = ALCBBacktestConfig(
            start_date=is_start.isoformat(),
            end_date=is_end.isoformat(),
            initial_equity=initial_equity,
        )
        is_summary = optimize_alcb(replay, is_config, param_space, objective)

        if is_summary.best is None:
            logger.warning("Window %d: no valid IS results, skipping", window_id)
            current = _date_add_months(current, step_months)
            window_id += 1
            continue

        best_params = is_summary.best.params
        is_metrics = is_summary.best.metrics

        # Out-of-sample validation with best IS params
        oos_config = ALCBBacktestConfig(
            start_date=oos_start.isoformat(),
            end_date=oos_end.isoformat(),
            initial_equity=initial_equity,
            param_overrides=best_params,
        )
        oos_engine = ALCBDailyEngine(oos_config, replay)
        oos_result = oos_engine.run()

        oos_metrics = None
        if oos_result.trades:
            pnls = np.array([t.pnl_net for t in oos_result.trades])
            risks = np.array([t.risk_per_share * t.quantity for t in oos_result.trades])
            hold_hours = np.array([t.hold_hours for t in oos_result.trades])
            commissions = np.array([t.commission for t in oos_result.trades])
            oos_metrics = compute_metrics(
                pnls, risks, hold_hours, commissions,
                oos_result.equity_curve, oos_result.timestamps, initial_equity,
            )

        window = WalkForwardWindow(
            window_id=window_id,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            best_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            oos_trades=oos_result.trades,
        )
        windows.append(window)

        logger.info(
            "  IS Sharpe=%.2f → OOS Sharpe=%.2f, params=%s",
            is_metrics.sharpe if is_metrics else 0,
            oos_metrics.sharpe if oos_metrics else 0,
            best_params,
        )

        current = _date_add_months(current, step_months)
        window_id += 1

    # Combine OOS trades
    combined = []
    for w in windows:
        combined.extend(w.oos_trades)
    combined.sort(key=lambda t: t.entry_time)

    result = WalkForwardResult(
        windows=windows,
        combined_oos_trades=combined,
    )

    if combined:
        pnls = np.array([t.pnl_net for t in combined])
        risks = np.array([t.risk_per_share * t.quantity for t in combined])
        hold_hours = np.array([t.hold_hours for t in combined])
        commissions = np.array([t.commission for t in combined])
        # Build approximate equity curve
        eq = [initial_equity]
        for pnl in pnls:
            eq.append(eq[-1] + pnl)
        ts = np.array([np.datetime64(t.entry_time.replace(tzinfo=None)) for t in combined])
        result.combined_metrics = compute_metrics(
            pnls, risks, hold_hours, commissions,
            np.array(eq), ts, initial_equity,
        )
        logger.info(
            "Walk-forward complete: %d windows, %d OOS trades, "
            "combined Sharpe=%.2f, efficiency=%.2f",
            len(windows), len(combined),
            result.combined_metrics.sharpe, result.efficiency,
        )

    return result


def walk_forward_iaric(
    replay: ResearchReplayEngine,
    param_space: list[ParamRange],
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    is_months: int = 6,
    oos_months: int = 3,
    step_months: int = 3,
    initial_equity: float = 10_000.0,
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Run walk-forward optimization for IARIC."""
    from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    windows: list[WalkForwardWindow] = []
    window_id = 0
    current = start

    while True:
        is_start = current
        is_end = _date_add_months(current, is_months)
        oos_start = is_end
        oos_end = _date_add_months(is_end, oos_months)

        if oos_end > end:
            oos_end = end
        if oos_start >= end:
            break

        logger.info(
            "Window %d: IS [%s, %s] → OOS [%s, %s]",
            window_id, is_start, is_end, oos_start, oos_end,
        )

        # In-sample optimization
        is_config = IARICBacktestConfig(
            start_date=is_start.isoformat(),
            end_date=is_end.isoformat(),
            initial_equity=initial_equity,
        )
        is_summary = optimize_iaric(replay, is_config, param_space, objective)

        if is_summary.best is None:
            logger.warning("Window %d: no valid IS results, skipping", window_id)
            current = _date_add_months(current, step_months)
            window_id += 1
            continue

        best_params = is_summary.best.params
        is_metrics = is_summary.best.metrics

        # Out-of-sample validation with best IS params
        oos_config = IARICBacktestConfig(
            start_date=oos_start.isoformat(),
            end_date=oos_end.isoformat(),
            initial_equity=initial_equity,
            param_overrides=best_params,
        )
        oos_engine = IARICDailyEngine(oos_config, replay)
        oos_result = oos_engine.run()

        oos_metrics = None
        if oos_result.trades:
            pnls = np.array([t.pnl_net for t in oos_result.trades])
            risks = np.array([t.risk_per_share * t.quantity for t in oos_result.trades])
            hold_hours = np.array([t.hold_hours for t in oos_result.trades])
            commissions = np.array([t.commission for t in oos_result.trades])
            oos_metrics = compute_metrics(
                pnls, risks, hold_hours, commissions,
                oos_result.equity_curve, oos_result.timestamps, initial_equity,
            )

        window = WalkForwardWindow(
            window_id=window_id,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            best_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            oos_trades=oos_result.trades,
        )
        windows.append(window)

        logger.info(
            "  IS Sharpe=%.2f → OOS Sharpe=%.2f, params=%s",
            is_metrics.sharpe if is_metrics else 0,
            oos_metrics.sharpe if oos_metrics else 0,
            best_params,
        )

        current = _date_add_months(current, step_months)
        window_id += 1

    # Combine OOS trades
    combined = []
    for w in windows:
        combined.extend(w.oos_trades)
    combined.sort(key=lambda t: t.entry_time)

    result = WalkForwardResult(
        windows=windows,
        combined_oos_trades=combined,
    )

    if combined:
        pnls = np.array([t.pnl_net for t in combined])
        risks = np.array([t.risk_per_share * t.quantity for t in combined])
        hold_hours = np.array([t.hold_hours for t in combined])
        commissions = np.array([t.commission for t in combined])
        eq = [initial_equity]
        for pnl in pnls:
            eq.append(eq[-1] + pnl)
        ts = np.array([np.datetime64(t.entry_time.replace(tzinfo=None)) for t in combined])
        result.combined_metrics = compute_metrics(
            pnls, risks, hold_hours, commissions,
            np.array(eq), ts, initial_equity,
        )
        logger.info(
            "Walk-forward complete: %d windows, %d OOS trades, "
            "combined Sharpe=%.2f, efficiency=%.2f",
            len(windows), len(combined),
            result.combined_metrics.sharpe, result.efficiency,
        )

    return result
