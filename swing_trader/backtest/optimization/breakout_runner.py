"""Breakout optimization runner: LHS coarse search + Optuna TPE refinement.

Mirrors HelixOptimizationRunner but uses Breakout types and parameter space.
Reuses TrialResult, OptimizationResult, composite_objective.
Includes trial persistence (CSV) and permutation importance analysis.
"""
from __future__ import annotations

import csv
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from backtest.analysis.metrics import compute_metrics
from backtest.config_breakout import BreakoutBacktestConfig
from backtest.engine.breakout_portfolio_engine import (
    BreakoutPortfolioData,
    run_breakout_independent,
)
from backtest.optimization.objective import composite_objective
from backtest.optimization.param_space import latin_hypercube_sample
from backtest.optimization.breakout_param_space import (
    BREAKOUT_PARAM_SPACE,
    breakout_params_to_overrides,
)
from backtest.optimization.runner import OptimizationResult, TrialResult

logger = logging.getLogger(__name__)


def _evaluate_single(
    params: dict[str, float],
    data: BreakoutPortfolioData,
    base_config: BreakoutBacktestConfig,
) -> TrialResult:
    """Evaluate one Breakout parameter set."""
    overrides = breakout_params_to_overrides(params)
    config = BreakoutBacktestConfig(
        symbols=base_config.symbols,
        start_date=base_config.start_date,
        end_date=base_config.end_date,
        initial_equity=base_config.initial_equity,
        slippage=base_config.slippage,
        flags=base_config.flags,
        param_overrides=overrides,
        data_dir=base_config.data_dir,
        track_shadows=False,
        track_signals=False,
        warmup_daily=base_config.warmup_daily,
        warmup_hourly=base_config.warmup_hourly,
        warmup_4h=base_config.warmup_4h,
    )

    result = run_breakout_independent(data, config)

    all_pnls = []
    all_risks = []
    all_holds = []
    all_comms = []
    all_syms = []
    for sr in result.symbol_results.values():
        for t in sr.trades:
            all_pnls.append(t.pnl_dollars)
            all_risks.append(abs(t.entry_price - t.stop_price) * t.qty)
            all_holds.append(t.bars_held)
            all_comms.append(t.commission)
            all_syms.append(t.symbol)

    if not all_pnls:
        return TrialResult(params=params, score=-1.0)

    metrics = compute_metrics(
        trade_pnls=np.array(all_pnls),
        trade_risks=np.array(all_risks),
        trade_hold_hours=np.array(all_holds),
        trade_commissions=np.array(all_comms),
        equity_curve=result.combined_equity,
        timestamps=result.combined_timestamps,
        initial_equity=config.initial_equity,
        trade_symbols=all_syms,
    )

    score = composite_objective(metrics)

    return TrialResult(
        params=params,
        score=score,
        total_trades=metrics.total_trades,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe,
        max_dd=metrics.max_drawdown_pct,
        profit_factor=metrics.profit_factor,
        trades_per_month=metrics.trades_per_month,
    )


class BreakoutOptimizationRunner:
    """Two-stage optimizer for Breakout v3.3-ETF strategy."""

    def __init__(
        self,
        base_config: BreakoutBacktestConfig,
        data: BreakoutPortfolioData,
        n_coarse: int = 1000,
        n_refine: int = 300,
        n_jobs: int = -1,
        seed: int = 42,
    ):
        self.base_config = base_config
        self.data = data
        self.n_coarse = n_coarse
        self.n_refine = n_refine
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed

    def run(self) -> OptimizationResult:
        """Execute the full two-stage optimization."""
        logger.info(
            "Starting Breakout optimization: %d coarse + %d refine, %d workers",
            self.n_coarse, self.n_refine, self.n_jobs,
        )

        coarse_results = self._stage_coarse()
        logger.info(
            "Coarse search complete: %d/%d valid",
            sum(1 for r in coarse_results if r.score > 0), len(coarse_results),
        )

        coarse_sorted = sorted(coarse_results, key=lambda r: r.score, reverse=True)
        top_50 = [r for r in coarse_sorted[:50] if r.score > 0]

        if not top_50:
            logger.warning("No valid coarse results, skipping refinement")
            return OptimizationResult(coarse_results=coarse_results)

        refine_results = self._stage_refine(top_50)
        logger.info("Refinement complete: %d trials", len(refine_results))

        all_results = coarse_results + refine_results
        all_sorted = sorted(all_results, key=lambda r: r.score, reverse=True)
        best = all_sorted[0] if all_sorted else TrialResult(params={}, score=-1.0)

        return OptimizationResult(
            best_params=best.params,
            best_score=best.score,
            coarse_results=coarse_results,
            refine_results=refine_results,
            all_sorted=all_sorted[:100],
        )

    def _stage_coarse(self) -> list[TrialResult]:
        """Stage A: Latin Hypercube Sampling coarse search."""
        samples = latin_hypercube_sample(BREAKOUT_PARAM_SPACE, self.n_coarse, self.seed)

        if self.n_jobs <= 1:
            results = []
            for i, sample in enumerate(samples):
                if (i + 1) % 100 == 0:
                    logger.info("Coarse trial %d/%d", i + 1, self.n_coarse)
                try:
                    result = _evaluate_single(sample, self.data, self.base_config)
                    results.append(result)
                except Exception:
                    logger.exception("Error in coarse trial %d", i)
                    results.append(TrialResult(params=sample, score=-1.0))
            return results

        results: list[TrialResult] = [TrialResult(params={}, score=-1.0)] * len(samples)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_idx = {
                executor.submit(_evaluate_single, sample, self.data, self.base_config): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                if completed % 100 == 0:
                    logger.info("Coarse trial %d/%d completed", completed, self.n_coarse)
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Error in coarse trial %d", idx)
                    results[idx] = TrialResult(params=samples[idx], score=-1.0)

        return results

    def _stage_refine(self, top_results: list[TrialResult]) -> list[TrialResult]:
        """Stage B: Optuna TPE refinement around top coarse results."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, skipping refinement stage")
            return []

        study = optuna.create_study(direction="maximize")

        for tr in top_results:
            try:
                study.enqueue_trial(tr.params)
            except Exception:
                pass

        trial_details: dict[int, TrialResult] = {}

        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for p in BREAKOUT_PARAM_SPACE:
                if p.is_int:
                    params[p.name] = trial.suggest_int(
                        p.name, int(p.low), int(p.high),
                        step=int(p.step) if p.step else 1,
                    )
                elif p.step > 0:
                    params[p.name] = trial.suggest_float(
                        p.name, p.low, p.high, step=p.step,
                    )
                else:
                    params[p.name] = trial.suggest_float(p.name, p.low, p.high)

            try:
                result = _evaluate_single(params, self.data, self.base_config)
                trial_details[trial.number] = result
                return result.score
            except Exception:
                return -1.0

        study.optimize(optuna_objective, n_trials=self.n_refine)

        refine_results = []
        for trial in study.trials:
            if trial.value is not None:
                detail = trial_details.get(trial.number)
                if detail is not None:
                    refine_results.append(detail)
                else:
                    refine_results.append(TrialResult(
                        params=trial.params,
                        score=trial.value,
                    ))

        return refine_results


# ---------------------------------------------------------------------------
# Trial persistence
# ---------------------------------------------------------------------------

def save_trials_csv(result: OptimizationResult, path: str | Path) -> Path:
    """Save all optimization trials to CSV for post-hoc analysis.

    Columns: score, total_trades, cagr, sharpe, max_dd, profit_factor,
    trades_per_month, plus one column per parameter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_trials = result.all_sorted
    if not all_trials:
        all_trials = result.coarse_results + result.refine_results

    if not all_trials:
        logger.warning("No trials to save")
        return path

    # Collect all param keys
    param_keys = sorted({k for t in all_trials for k in t.params})
    metric_cols = ["score", "total_trades", "cagr", "sharpe", "max_dd",
                   "profit_factor", "trades_per_month"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metric_cols + param_keys)
        for t in all_trials:
            row = [
                t.score, t.total_trades, t.cagr, t.sharpe,
                t.max_dd, t.profit_factor, t.trades_per_month,
            ]
            row.extend(t.params.get(k, "") for k in param_keys)
            writer.writerow(row)

    logger.info("Saved %d trials to %s", len(all_trials), path)
    return path


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance(
    best_params: dict[str, float],
    data: BreakoutPortfolioData,
    base_config: BreakoutBacktestConfig,
    n_shuffles: int = 5,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Compute permutation importance for each parameter.

    For each parameter, replace its value with random samples from the
    parameter space while holding all others at best_params.  Measures
    the average score drop.

    Returns list of (param_name, importance) sorted by importance desc.
    """
    rng = np.random.default_rng(seed)
    baseline = _evaluate_single(best_params, data, base_config)
    baseline_score = baseline.score

    param_ranges = {p.name: p for p in BREAKOUT_PARAM_SPACE}
    importance: dict[str, float] = {}

    for name in best_params:
        if name not in param_ranges:
            continue
        p = param_ranges[name]
        scores = []
        for _ in range(n_shuffles):
            shuffled = dict(best_params)
            raw = rng.uniform(p.low, p.high)
            shuffled[name] = p.snap(raw)
            try:
                trial = _evaluate_single(shuffled, data, base_config)
                scores.append(trial.score)
            except Exception:
                scores.append(-1.0)
        avg_shuffled = float(np.mean(scores))
        importance[name] = baseline_score - avg_shuffled

    return sorted(importance.items(), key=lambda kv: -kv[1])
