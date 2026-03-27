"""Grid search optimization runner for stock backtests."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from research.backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine
from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import TradeRecord
from research.backtests.stock.optimization.param_space import ParamRange, build_grid, grid_size

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""

    params: dict[str, float]
    metrics: PerformanceMetrics
    trades: list[TradeRecord]
    equity_curve: np.ndarray


@dataclass
class OptimizationSummary:
    """Summary of all optimization runs."""

    results: list[OptimizationResult]
    best: OptimizationResult | None = None
    objective: str = "sharpe"

    @property
    def best_params(self) -> dict[str, float]:
        return self.best.params if self.best else {}

    def top_n(self, n: int = 10) -> list[OptimizationResult]:
        """Return top N results by objective."""
        key = self._objective_key()
        sorted_results = sorted(self.results, key=key, reverse=True)
        return sorted_results[:n]

    def _objective_key(self):
        obj = self.objective
        if obj == "sharpe":
            return lambda r: r.metrics.sharpe
        elif obj == "calmar":
            return lambda r: r.metrics.calmar
        elif obj == "profit_factor":
            return lambda r: r.metrics.profit_factor
        elif obj == "expectancy":
            return lambda r: r.metrics.expectancy
        elif obj == "cagr":
            return lambda r: r.metrics.cagr
        elif obj == "net_profit":
            return lambda r: r.metrics.net_profit
        return lambda r: r.metrics.sharpe


def optimize_alcb(
    replay: ResearchReplayEngine,
    base_config: ALCBBacktestConfig,
    param_space: list[ParamRange],
    objective: str = "sharpe",
) -> OptimizationSummary:
    """Run grid search optimization for ALCB."""
    grid = build_grid(param_space)
    total = len(grid)
    logger.info("ALCB optimization: %d parameter combinations", total)

    results: list[OptimizationResult] = []

    for idx, params in enumerate(grid):
        config = replace(base_config, param_overrides=params, verbose=False)
        engine = ALCBDailyEngine(config, replay)
        result = engine.run()

        if not result.trades:
            continue

        pnls = np.array([t.pnl_net for t in result.trades])
        risks = np.array([t.risk_per_share * t.quantity for t in result.trades])
        hold_hours = np.array([t.hold_hours for t in result.trades])
        commissions = np.array([t.commission for t in result.trades])

        metrics = compute_metrics(
            trade_pnls=pnls,
            trade_risks=risks,
            trade_hold_hours=hold_hours,
            trade_commissions=commissions,
            equity_curve=result.equity_curve,
            timestamps=result.timestamps,
            initial_equity=base_config.initial_equity,
        )

        opt_result = OptimizationResult(
            params=params,
            metrics=metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )
        results.append(opt_result)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            logger.info(
                "  [%d/%d] params=%s → Sharpe=%.2f, CAGR=%.2f%%, PF=%.2f",
                idx + 1, total, params, metrics.sharpe, metrics.cagr * 100, metrics.profit_factor,
            )

    summary = OptimizationSummary(results=results, objective=objective)
    if results:
        key = summary._objective_key()
        summary.best = max(results, key=key)
        logger.info(
            "Best: %s → Sharpe=%.2f, CAGR=%.2f%%, PF=%.2f",
            summary.best.params, summary.best.metrics.sharpe,
            summary.best.metrics.cagr * 100, summary.best.metrics.profit_factor,
        )

    return summary


def optimize_iaric(
    replay: ResearchReplayEngine,
    base_config: IARICBacktestConfig,
    param_space: list[ParamRange],
    objective: str = "sharpe",
) -> OptimizationSummary:
    """Run grid search optimization for IARIC."""
    grid = build_grid(param_space)
    total = len(grid)
    logger.info("IARIC optimization: %d parameter combinations", total)

    results: list[OptimizationResult] = []

    for idx, params in enumerate(grid):
        config = replace(base_config, param_overrides=params, verbose=False)
        engine = IARICDailyEngine(config, replay)
        result = engine.run()

        if not result.trades:
            continue

        pnls = np.array([t.pnl_net for t in result.trades])
        risks = np.array([t.risk_per_share * t.quantity for t in result.trades])
        hold_hours = np.array([t.hold_hours for t in result.trades])
        commissions = np.array([t.commission for t in result.trades])

        metrics = compute_metrics(
            trade_pnls=pnls,
            trade_risks=risks,
            trade_hold_hours=hold_hours,
            trade_commissions=commissions,
            equity_curve=result.equity_curve,
            timestamps=result.timestamps,
            initial_equity=base_config.initial_equity,
        )

        opt_result = OptimizationResult(
            params=params,
            metrics=metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )
        results.append(opt_result)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            logger.info(
                "  [%d/%d] params=%s → Sharpe=%.2f, CAGR=%.2f%%, PF=%.2f",
                idx + 1, total, params, metrics.sharpe, metrics.cagr * 100, metrics.profit_factor,
            )

    summary = OptimizationSummary(results=results, objective=objective)
    if results:
        key = summary._objective_key()
        summary.best = max(results, key=key)
        logger.info(
            "Best: %s → Sharpe=%.2f, CAGR=%.2f%%, PF=%.2f",
            summary.best.params, summary.best.metrics.sharpe,
            summary.best.metrics.cagr * 100, summary.best.metrics.profit_factor,
        )

    return summary
