from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time as _time
from pathlib import Path
from typing import Any, Callable

from research.backtests.shared.auto.phase_state import PhaseState
from research.backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from research.backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    greedy_result_from_state,
    mutation_signature,
    resolve_worker_processes,
)
from research.backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import merge_alcb_metrics, score_alcb_phase
from .time_utils import hydrate_time_mutations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-phase static hard rejects (absolute floors)
# P14 uses escalating floors across 3 phases targeting structural alpha leaks.
# ---------------------------------------------------------------------------
PHASE_STATIC_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_expected_total_r": 32.0,
        "min_trades_per_month": 12.0,
        "min_pf": 1.20,
        "max_dd_pct": 0.07,
    },
    2: {
        "min_expected_total_r": 35.0,
        "min_trades_per_month": 12.0,
        "min_pf": 1.25,
        "max_dd_pct": 0.065,
    },
    3: {
        "min_expected_total_r": 38.0,
        "min_trades_per_month": 10.0,
        "min_pf": 1.30,
        "max_dd_pct": 0.06,
    },
}

PHASE_MAX_ROUNDS = {1: 12, 2: 12, 3: 10}
PHASE_PRUNE_THRESHOLDS = {1: 0.05, 2: 0.04, 3: 0.04}
PHASE_MIN_EFFECTIVE_DELTA = {1: 0.003, 2: 0.003, 3: 0.002}

ULTIMATE_TARGETS = {
    "expectancy_dollar": 10.00,
    "expected_total_r": 65.0,
    "trades_per_month": 24.0,
    "profit_factor": 1.80,
    "entry_quality": 0.70,
    "high_rvol_edge": 0.70,
    "short_hold_drag_inverse": 0.70,
    "flow_reversal_short_inverse": 0.50,
    "early_1000_drag_inverse": 0.55,
    "profit_protection": 0.60,
    "long_hold_capture": 0.80,
    "carry_capture": 0.20,
    "inv_dd": 0.75,
    "mfe_capture_efficiency": 0.55,
}


class _PoolBatchEvaluator:
    """Process-pool evaluator that passes phase config per-call."""

    def __init__(
        self,
        pool: mp.Pool,
        phase: int,
        hard_rejects: dict[str, float] | None,
        scoring_weights: dict[str, float] | None,
        on_pool_dead: Callable | None = None,
        num_workers: int = 2,
    ):
        self._pool = pool
        self._phase = phase
        self._hard_rejects = hard_rejects or {}
        self._scoring_weights = scoring_weights or {}
        self._on_pool_dead = on_pool_dead
        self._num_workers = num_workers

        from .worker import set_phase_config

        list(pool.map(
            set_phase_config,
            [(phase, self._hard_rejects, self._scoring_weights)] * (num_workers * 2),
        ))

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate_indexed

        args = [
            (candidate.name, candidate.mutations, current_mutations)
            for candidate in candidates
        ]
        total = len(args)
        if total == 0:
            return []

        start = _time.time()
        logger.info("Dispatching %d candidates to pool...", total)

        indexed_args = [(i, arg) for i, arg in enumerate(args)]
        results: list[Any] = [None] * total
        completed = 0
        progress_step = max(1, total // 10)
        chunksize = max(1, total // (self._num_workers * 4))
        for idx, result in self._pool.imap_unordered(score_candidate_indexed, indexed_args, chunksize=chunksize):
            results[idx] = result
            completed += 1
            if completed % progress_step == 0 or completed == total:
                elapsed = _time.time() - start
                logger.info(
                    "  %d/%d evaluated (%.0fs, ~%.1fs/ea)",
                    completed, total, elapsed, elapsed / completed,
                )
        return results

    def close(self) -> None:
        pass

    def terminate(self) -> None:
        if self._on_pool_dead:
            self._on_pool_dead()
        try:
            self._pool.terminate()
            self._pool.join()
        except Exception:
            pass


class _LocalBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        initial_equity: float,
        phase: int,
        hard_rejects: dict[str, float] | None,
        scoring_weights: dict[str, float] | None,
    ):
        from .worker import init_worker

        init_worker(
            str(data_dir),
            start_date,
            end_date,
            initial_equity,
            phase,
            hard_rejects,
            scoring_weights,
        )
        self._phase = phase
        self._hard_rejects = hard_rejects or {}
        self._scoring_weights = scoring_weights or {}

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        return [
            score_candidate((
                candidate.name, candidate.mutations, current_mutations,
                self._phase, self._hard_rejects, self._scoring_weights,
            ))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class ALCBP14Plugin:
    name = "alcb_p14_phase"
    num_phases = len(PHASE_FOCUS)
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        *,
        start_date: str = "2024-01-01",
        end_date: str = "2026-03-01",
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        experiment_names: set[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.experiment_names = set(experiment_names or [])
        self._cached_replay = None
        self._config_cache: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._last_context: dict[str, Any] = {}
        self._phase_runtime_context: dict[int, dict[str, Any]] = {}
        self._shared_pool: mp.Pool | None = None
        self._resolved_workers: int = 0

    def _get_or_create_pool(self) -> mp.Pool:
        """Return the shared process pool, creating it on first use."""
        if self._shared_pool is None:
            from .worker import init_worker_data_only

            self._resolved_workers = resolve_worker_processes(self.max_workers)
            self._shared_pool = mp.Pool(
                processes=self._resolved_workers,
                initializer=init_worker_data_only,
                initargs=(
                    str(self.data_dir),
                    self.start_date,
                    self.end_date,
                    self.initial_equity,
                ),
            )
        return self._shared_pool

    def close_pool(self) -> None:
        """Shut down the shared process pool if it exists."""
        if self._shared_pool is not None:
            try:
                self._shared_pool.close()
                self._shared_pool.join()
            except Exception:
                try:
                    self._shared_pool.terminate()
                    self._shared_pool.join()
                except Exception:
                    pass
            self._shared_pool = None

    def _invalidate_pool(self) -> None:
        """Mark the shared pool as dead so the next phase recreates it."""
        self._shared_pool = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        focus, focus_metrics = PHASE_FOCUS[phase]
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                experiment_filter=self.experiment_names or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, current_phase=phase: self._gate_criteria(current_phase, metrics),
            scoring_weights=None,
            hard_rejects=PHASE_STATIC_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=PHASE_MIN_EFFECTIVE_DELTA.get(phase, 0.002),
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=PHASE_MAX_ROUNDS.get(phase, len(candidates)),
            prune_threshold=PHASE_PRUNE_THRESHOLDS.get(phase, 0.05),
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        t0 = _time.time()
        base_metrics = self._run_config(cumulative_mutations, store_context=False)["metrics"]
        logger.info("Parent baseline evaluated in %.1fs", _time.time() - t0)
        resolved_hard_rejects = self._resolve_phase_hard_rejects(phase, base_metrics, hard_rejects or {})
        baseline_result = self._seed_result_for_metrics(
            "__baseline__",
            phase,
            base_metrics,
            resolved_hard_rejects,
            scoring_weights,
        )
        baseline_key = mutation_signature(cumulative_mutations)
        self._metrics_cache[baseline_key] = dict(base_metrics)
        self._phase_runtime_context[phase] = {
            "base_metrics": base_metrics,
            "hard_rejects": resolved_hard_rejects,
        }
        evaluation_key = mutation_signature(
            {
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": resolved_hard_rejects,
            }
        )

        def local_factory():
            return _LocalBatchEvaluator(
                self.data_dir,
                self.start_date,
                self.end_date,
                self.initial_equity,
                phase,
                resolved_hard_rejects,
                scoring_weights,
            )

        if self.max_workers == 1 or not _supports_spawn():
            evaluator = local_factory()
        else:
            def pool_factory():
                pool = self._get_or_create_pool()
                return _PoolBatchEvaluator(
                    pool,
                    phase,
                    resolved_hard_rejects,
                    scoring_weights,
                    on_pool_dead=self._invalidate_pool,
                    num_workers=self._resolved_workers,
                )

            evaluator = ResilientBatchEvaluator(
                preferred_factory=pool_factory,
                fallback_factory=local_factory,
                description=f"{self.name} phase {phase} evaluator",
                logger=logger,
            )
        return CachedBatchEvaluator(
            evaluator,
            cache=self._evaluation_cache,
            seed_results={baseline_key: baseline_result},
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        cached = self._metrics_cache.get(mutation_signature(mutations))
        if cached:
            return dict(cached)
        return self._run_config(mutations, store_context=True)["metrics"]

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        runtime = self._phase_runtime_context.get(phase, {})
        return _build_phase_snapshot(
            phase,
            PHASE_FOCUS[phase][0],
            metrics,
            greedy_result,
            base_metrics=runtime.get("base_metrics", {}),
            hard_rejects=runtime.get("hard_rejects", {}),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from research.backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
        from research.backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis

        snapshot = self.run_phase_diagnostics(phase, state, metrics, greedy_result)
        expected_signature = mutation_signature(greedy_result.final_mutations)
        if self._last_context.get("mutation_signature") != expected_signature:
            self._run_config(greedy_result.final_mutations, store_context=True, collect_diagnostics=True)
        trades = self._last_context.get("trades")
        config = self._last_context.get("config")
        if not trades or config is None:
            return snapshot

        max_positions = int(config.param_overrides.get("max_positions", 10))
        return "\n\n".join([
            snapshot,
            alcb_full_diagnostic(
                trades,
                shadow_tracker=self._last_context.get("shadow_tracker"),
                daily_selections=self._last_context.get("daily_selections"),
            ),
            qe_replacement_analysis(trades, max_positions=max_positions),
        ])

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        from research.backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis

        final_phase = max(state.completed_phases) if state.completed_phases else self.num_phases
        final_ctx = self._run_config(state.cumulative_mutations, store_context=True, collect_diagnostics=True)
        base_ctx = self._run_config(BASE_MUTATIONS, store_context=False, collect_diagnostics=True)
        final_metrics = final_ctx["metrics"]
        base_metrics = base_ctx["metrics"]
        final_trades = final_ctx["trades"]
        final_greedy = greedy_result_from_state(state, phase=final_phase, final_metrics=final_metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(final_phase, state, final_metrics, final_greedy)

        dimension_reports = {
            "expected_return": "\n".join([
                f"Net profit {base_metrics['net_profit']:+.2f} -> {final_metrics['net_profit']:+.2f}.",
                f"Expectancy $ {base_metrics['expectancy_dollar']:+.2f} -> {final_metrics['expectancy_dollar']:+.2f}.",
                f"Expected total R {base_metrics['expected_total_r']:+.2f} -> {final_metrics['expected_total_r']:+.2f}.",
                f"Profit factor {base_metrics['profit_factor']:.2f} -> {final_metrics['profit_factor']:.2f}.",
            ]),
            "frequency": "\n".join([
                f"Total trades {int(base_metrics['total_trades'])} -> {int(final_metrics['total_trades'])}.",
                f"Trades/month {base_metrics['trades_per_month']:.2f} -> {final_metrics['trades_per_month']:.2f}.",
                f"Avg hold hours {base_metrics['avg_hold_hours']:.2f} -> {final_metrics['avg_hold_hours']:.2f}.",
            ]),
            "signal_quality": "\n".join([
                f"Entry quality {base_metrics['entry_quality']:.3f} -> {final_metrics['entry_quality']:.3f}.",
                f"High RVOL edge {base_metrics['high_rvol_edge']:.3f} -> {final_metrics['high_rvol_edge']:.3f}.",
                f"OR avg R {base_metrics['or_avg_r']:+.3f} -> {final_metrics['or_avg_r']:+.3f}.",
                f"COMBINED avg R {base_metrics['combined_avg_r']:+.3f} -> {final_metrics['combined_avg_r']:+.3f}.",
                f"PDH avg R {base_metrics['pdh_avg_r']:+.3f} -> {final_metrics['pdh_avg_r']:+.3f}.",
                f"MFE capture eff {base_metrics.get('mfe_capture_efficiency', 0.0):.3f} -> {final_metrics.get('mfe_capture_efficiency', 0.0):.3f}.",
            ]),
            "loss_control": "\n".join([
                f"Short-hold total R {base_metrics['short_hold_total_r']:+.2f} -> {final_metrics['short_hold_total_r']:+.2f}.",
                f"FLOW_REV short R {base_metrics['flow_reversal_short_total_r']:+.2f} -> {final_metrics['flow_reversal_short_total_r']:+.2f}.",
                f"10:00 short R {base_metrics['early_1000_short_total_r']:+.2f} -> {final_metrics['early_1000_short_total_r']:+.2f}.",
                f"Profit protection {base_metrics['profit_protection']:.3f} -> {final_metrics['profit_protection']:.3f}.",
            ]),
            "exit_capture": "\n".join([
                f"Long-hold total R {base_metrics['long_hold_total_r']:+.2f} -> {final_metrics['long_hold_total_r']:+.2f}.",
                f"Long-hold capture {base_metrics['long_hold_capture']:.3f} -> {final_metrics['long_hold_capture']:.3f}.",
                f"Carry capture {base_metrics['carry_capture']:.3f} -> {final_metrics['carry_capture']:.3f}.",
            ]),
            "risk": "\n".join([
                f"Max DD {base_metrics['max_drawdown_pct']:.1%} -> {final_metrics['max_drawdown_pct']:.1%}.",
                f"Sharpe {base_metrics['sharpe']:.2f} -> {final_metrics['sharpe']:.2f}.",
                f"Inv DD {base_metrics['inv_dd']:.3f} -> {final_metrics['inv_dd']:.3f}.",
            ]),
        }
        extra_sections = {
            "qe_replacement": qe_replacement_analysis(
                final_trades,
                max_positions=int(final_ctx["config"].param_overrides.get("max_positions", 10)),
            ),
        }
        overall_verdict = (
            f"Final {self.num_phases}-phase bundle finished at "
            f"expectancy ${final_metrics['expectancy_dollar']:+.2f}, expected_total_r {final_metrics['expected_total_r']:+.2f}, "
            f"{final_metrics['trades_per_month']:.2f} trades/month, PF {final_metrics['profit_factor']:.2f}, "
            f"MFE capture {final_metrics.get('mfe_capture_efficiency', 0.0):.1%}, "
            f"and max DD {final_metrics['max_drawdown_pct']:.1%}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
            extra_sections=extra_sections,
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        gaps: list[str] = []
        if phase == 1:
            if metrics["expected_total_r"] < max(35.0, base_metrics.get("expected_total_r", 0.0) * 0.90):
                gaps.append("MFE conviction is cutting too much alpha -- total R declining.")
            if metrics["trades_per_month"] < max(12.0, base_metrics.get("trades_per_month", 0.0) * 0.60):
                gaps.append("Conviction filter is cutting too much throughput.")
            if metrics["profit_factor"] < max(1.25, base_metrics.get("profit_factor", 0.0) * 1.0):
                gaps.append("Purgatory zone fix has not improved profit factor.")
        elif phase == 2:
            if metrics.get("mfe_capture_efficiency", 0.0) < max(0.42, base_metrics.get("mfe_capture_efficiency", 0.0) * 1.0):
                gaps.append("Adaptive trail has not improved MFE capture efficiency.")
            if metrics["long_hold_capture"] < max(0.45, base_metrics.get("long_hold_capture", 0.0) * 0.95):
                gaps.append("Trail is cutting long-hold alpha zone -- too tight.")
            if metrics["expected_total_r"] < max(35.0, base_metrics.get("expected_total_r", 0.0) * 0.92):
                gaps.append("Trailing stop is eroding total R.")
        elif phase == 3:
            if metrics["entry_quality"] < max(0.55, base_metrics.get("entry_quality", 0.0) * 1.0):
                gaps.append("COMBINED entry filters have not improved entry quality.")
            if metrics["expectancy_dollar"] < base_metrics.get("expectancy_dollar", 0.0) * 0.98:
                gaps.append("Entry refinement is eroding per-trade expectancy.")
            if metrics["trades_per_month"] < max(10.0, base_metrics.get("trades_per_month", 0.0) * 0.80):
                gaps.append("Entry filtering is cutting too much throughput.")
        return gaps

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        del state, greedy_result
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        focus_metrics = PHASE_FOCUS[phase][1]
        deltas = {
            metric_name: {
                "base": float(base_metrics.get(metric_name, 0.0)),
                "final": float(metrics.get(metric_name, 0.0)),
                "delta": float(metrics.get(metric_name, 0.0)) - float(base_metrics.get(metric_name, 0.0)),
            }
            for metric_name in focus_metrics
        }
        return {
            "focus_deltas": deltas,
            "core": {
                "net_profit": metrics["net_profit"],
                "expectancy_dollar": metrics["expectancy_dollar"],
                "expected_total_r": metrics["expected_total_r"],
                "trades_per_month": metrics["trades_per_month"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "mfe_capture_efficiency": metrics.get("mfe_capture_efficiency", 0.0),
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        focus_deltas = extra.get("focus_deltas", {})
        delta_line = ", ".join(
            f"{metric}={values['base']:.3f}->{values['final']:.3f} ({values['delta']:+.3f})"
            for metric, values in focus_deltas.items()
        )
        core = extra.get("core", {})
        core_line = ", ".join(
            [
                f"net_profit={core.get('net_profit', 0.0):+.2f}",
                f"expectancy_dollar={core.get('expectancy_dollar', 0.0):+.2f}",
                f"expected_total_r={core.get('expected_total_r', 0.0):+.2f}",
                f"trades_per_month={core.get('trades_per_month', 0.0):.2f}",
                f"profit_factor={core.get('profit_factor', 0.0):.2f}",
                f"max_drawdown_pct={core.get('max_drawdown_pct', 0.0):.2%}",
                f"mfe_capture_eff={core.get('mfe_capture_efficiency', 0.0):.3f}",
            ]
        )
        return [f"Focus deltas: {delta_line}", f"Core: {core_line}"]

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        dd_budget = self._drawdown_budget(phase, base_metrics)
        if phase == 1:
            return [
                self._min_gate("expected_total_r", metrics["expected_total_r"], max(32.0, base_metrics.get("expected_total_r", 0.0) * 0.85)),
                self._min_gate("trades_per_month", metrics["trades_per_month"], max(12.0, base_metrics.get("trades_per_month", 0.0) * 0.55)),
                self._min_gate("profit_factor", metrics["profit_factor"], max(1.20, base_metrics.get("profit_factor", 0.0) * 1.0)),
                self._min_gate("expectancy_dollar", metrics["expectancy_dollar"], max(3.00, base_metrics.get("expectancy_dollar", 0.0) * 0.95)),
                self._max_gate("max_drawdown_pct", metrics["max_drawdown_pct"], dd_budget),
            ]
        if phase == 2:
            return [
                self._min_gate("mfe_capture_efficiency", metrics.get("mfe_capture_efficiency", 0.0), max(0.40, base_metrics.get("mfe_capture_efficiency", 0.0) * 0.98)),
                self._min_gate("long_hold_capture", metrics["long_hold_capture"], max(0.40, base_metrics.get("long_hold_capture", 0.0) * 0.95)),
                self._min_gate("expected_total_r", metrics["expected_total_r"], max(35.0, base_metrics.get("expected_total_r", 0.0) * 0.90)),
                self._min_gate("expectancy_dollar", metrics["expectancy_dollar"], max(3.00, base_metrics.get("expectancy_dollar", 0.0) * 0.95)),
                self._min_gate("profit_factor", metrics["profit_factor"], max(1.25, base_metrics.get("profit_factor", 0.0) * 0.98)),
                self._max_gate("max_drawdown_pct", metrics["max_drawdown_pct"], dd_budget),
            ]
        # phase 3: entry refinement
        return [
            self._min_gate("entry_quality", metrics["entry_quality"], max(0.50, base_metrics.get("entry_quality", 0.0) * 1.0)),
            self._min_gate("expectancy_dollar", metrics["expectancy_dollar"], max(3.00, base_metrics.get("expectancy_dollar", 0.0) * 0.98)),
            self._min_gate("expected_total_r", metrics["expected_total_r"], max(38.0, base_metrics.get("expected_total_r", 0.0) * 0.95)),
            self._min_gate("profit_factor", metrics["profit_factor"], max(1.30, base_metrics.get("profit_factor", 0.0) * 0.98)),
            self._min_gate("trades_per_month", metrics["trades_per_month"], max(10.0, base_metrics.get("trades_per_month", 0.0) * 0.80)),
            self._max_gate("max_drawdown_pct", metrics["max_drawdown_pct"], dd_budget),
        ]

    def _resolve_phase_hard_rejects(
        self,
        phase: int,
        base_metrics: dict[str, float],
        hard_rejects: dict[str, float],
    ) -> dict[str, float]:
        resolved = dict(PHASE_STATIC_HARD_REJECTS.get(phase, {}))
        resolved.update(hard_rejects or {})

        if phase == 1:
            self._raise_floor(resolved, "min_expected_total_r", base_metrics.get("expected_total_r", 0.0), 0.60, minimum=32.0)
            self._raise_floor(resolved, "min_trades_per_month", base_metrics.get("trades_per_month", 0.0), 0.50, minimum=12.0)
            self._raise_floor(resolved, "min_pf", base_metrics.get("profit_factor", 0.0), 0.85, minimum=1.20)
        elif phase == 2:
            self._raise_floor(resolved, "min_expected_total_r", base_metrics.get("expected_total_r", 0.0), 0.65, minimum=35.0)
            self._raise_floor(resolved, "min_trades_per_month", base_metrics.get("trades_per_month", 0.0), 0.55, minimum=12.0)
            self._raise_floor(resolved, "min_pf", base_metrics.get("profit_factor", 0.0), 0.88, minimum=1.25)
        else:  # phase 3
            self._raise_floor(resolved, "min_expected_total_r", base_metrics.get("expected_total_r", 0.0), 0.72, minimum=38.0)
            self._raise_floor(resolved, "min_trades_per_month", base_metrics.get("trades_per_month", 0.0), 0.45, minimum=10.0)
            self._raise_floor(resolved, "min_pf", base_metrics.get("profit_factor", 0.0), 0.90, minimum=1.30)

        resolved["max_dd_pct"] = min(
            float(resolved.get("max_dd_pct", 0.07)),
            self._drawdown_budget(phase, base_metrics),
        )
        return resolved

    def _ensure_replay(self):
        if self._cached_replay is None:
            from research.backtests.stock.engine.research_replay import ResearchReplayEngine

            replay = ResearchReplayEngine(data_dir=self.data_dir)
            replay.load_all_data()
            self._cached_replay = replay
        return self._cached_replay

    def _run_config(
        self,
        mutations: dict[str, Any],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        store_context: bool = False,
        collect_diagnostics: bool = False,
    ) -> dict[str, Any]:
        from research.backtests.stock._aliases import install
        from research.backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
        from research.backtests.stock.auto.config_mutator import mutate_alcb_config
        from research.backtests.stock.auto.scoring import extract_metrics
        from research.backtests.stock.config_alcb import ALCBBacktestConfig
        from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine

        install()
        mutations = hydrate_time_mutations(mutations)
        effective_start = start_date or self.start_date
        effective_end = end_date or self.end_date
        diagnostics_enabled = bool(collect_diagnostics or store_context)
        signature = mutation_signature(mutations)
        cache_key = (signature, effective_start, effective_end, diagnostics_enabled)
        cached = self._config_cache.get(cache_key)
        if cached is None and not diagnostics_enabled:
            cached = self._config_cache.get((signature, effective_start, effective_end, True))
        if cached is not None:
            if store_context:
                self._last_context = cached
            return cached

        replay = self._ensure_replay()
        config = mutate_alcb_config(
            ALCBBacktestConfig(
                start_date=effective_start,
                end_date=effective_end,
                initial_equity=self.initial_equity,
                tier=2,
                data_dir=self.data_dir,
            ),
            mutations,
        )
        engine = ALCBIntradayEngine(config, replay)
        shadow_tracker = ALCBShadowTracker() if (collect_diagnostics or store_context) else None
        if shadow_tracker is not None:
            engine.shadow_tracker = shadow_tracker
        result = engine.run()
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, self.initial_equity)
        metrics = merge_alcb_metrics(perf, result.trades)
        context = {
            "mutation_signature": signature,
            "metrics": metrics,
            "trades": result.trades,
            "replay": replay,
            "daily_selections": result.daily_selections,
            "shadow_tracker": shadow_tracker,
            "config": config,
        }
        self._config_cache[cache_key] = context
        if diagnostics_enabled:
            self._config_cache[(signature, effective_start, effective_end, False)] = context
        if store_context:
            self._last_context = context
        return context

    def _seed_result_for_metrics(
        self,
        name: str,
        phase: int,
        metrics: dict[str, float],
        hard_rejects: dict[str, float],
        scoring_weights: dict[str, float] | None,
    ) -> ScoredCandidate:
        from .worker import phase_reject_reason

        reject_reason = phase_reject_reason(metrics, hard_rejects, phase=phase)
        if reject_reason:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=reject_reason,
                metrics=dict(metrics),
            )
        return ScoredCandidate(
            name=name,
            score=score_alcb_phase(phase, metrics, scoring_weights),
            metrics=dict(metrics),
        )

    @staticmethod
    def _min_gate(name: str, actual: float, target: float, *, strict: bool = False) -> GateCriterion:
        passed = float(actual) > float(target) if strict else float(actual) >= float(target)
        return GateCriterion(name, float(target), float(actual), passed)

    @staticmethod
    def _max_gate(name: str, actual: float, target: float) -> GateCriterion:
        return GateCriterion(name, float(target), float(actual), float(actual) <= float(target))

    @staticmethod
    def _raise_floor(
        resolved: dict[str, float],
        key: str,
        base_value: float,
        ratio: float,
        *,
        minimum: float = 0.0,
    ) -> None:
        floor = max(float(minimum), float(base_value) * float(ratio)) if float(base_value) > 0.0 else float(minimum)
        resolved[key] = max(float(resolved.get(key, floor)), floor)

    @staticmethod
    def _drawdown_budget(phase: int, base_metrics: dict[str, float]) -> float:
        base_dd = float(base_metrics.get("max_drawdown_pct", 0.0))
        multiplier = {1: 1.20, 2: 1.15, 3: 1.10}.get(phase, 1.15)
        static_cap = {1: 0.07, 2: 0.065, 3: 0.06}.get(phase, 0.07)
        floor = {1: 0.06, 2: 0.055, 3: 0.05}.get(phase, 0.06)
        dynamic = max(base_dd * multiplier, base_dd + 0.015, floor)
        return min(static_cap, dynamic)


def _build_phase_snapshot(
    phase: int,
    focus: str,
    metrics: dict[str, float],
    greedy_result,
    *,
    base_metrics: dict[str, float],
    hard_rejects: dict[str, float],
) -> str:
    floors = ", ".join(
        f"{key}={value:.4f}" if "dd" not in key else f"{key}={value:.2%}"
        for key, value in sorted(hard_rejects.items())
    )
    focus_deltas = ", ".join(
        f"{metric}={base_metrics.get(metric, 0.0):+.3f}->{metrics.get(metric, 0.0):+.3f}"
        for metric in PHASE_FOCUS[phase][1]
    )
    return "\n".join([
        "=" * 70,
        f"ALCB P14 SHARED PHASE {phase} SNAPSHOT",
        "=" * 70,
        f"Focus: {focus}",
        f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f} with {greedy_result.accepted_count} accepted mutations.",
        "",
        (
            f"Core: trades={int(metrics['total_trades'])}, net_profit={metrics['net_profit']:+.2f}, "
            f"expectancy_dollar={metrics['expectancy_dollar']:+.2f}, expected_total_r={metrics['expected_total_r']:+.2f}, "
            f"pf={metrics['profit_factor']:.2f}, dd={metrics['max_drawdown_pct']:.1%}, trades/month={metrics['trades_per_month']:.2f}"
        ),
        (
            f"Quality: entry={metrics['entry_quality']:.3f}, high_rvol={metrics['high_rvol_edge']:.3f}, "
            f"profit_protection={metrics['profit_protection']:.3f}, long_hold_capture={metrics['long_hold_capture']:.3f}, "
            f"carry_capture={metrics['carry_capture']:.3f}, mfe_capture_eff={metrics.get('mfe_capture_efficiency', 0.0):.3f}"
        ),
        (
            f"Pressure points: short_hold_r={metrics['short_hold_total_r']:+.2f}, "
            f"flow_rev_short_r={metrics['flow_reversal_short_total_r']:+.2f}, "
            f"10:00_short_r={metrics['early_1000_short_total_r']:+.2f}"
        ),
        f"Focus metric deltas: {focus_deltas}",
        f"Hard floors: {floors}",
    ])


def _supports_spawn() -> bool:
    if sys.platform != "win32":
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", "")
    return bool(main_path) and not str(main_path).startswith("<")
