"""ATRSS phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
4 phases: exit cleanup, signal filtering, entry optimization, sizing/fine-tune.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    deserialize_experiments,
    greedy_result_from_state,
    greedy_result_to_dict,
    mutation_signature,
    resolve_worker_processes,
)
from backtests.shared.auto.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
)

from .phase_analyzer import get_diagnostic_gaps, suggest_experiments
from .phase_candidates import get_phase_candidates
from .phase_gates import gate_criteria_for_phase
from .phase_scoring import (
    PHASE_FOCUS,
    PHASE_HARD_REJECTS,
    PHASE_WEIGHTS,
    ULTIMATE_TARGETS,
    score_phase_metrics,
)
from .scoring import ATRSSMetrics, composite_score, extract_atrss_metrics

logger = logging.getLogger(__name__)


def _metrics_from_dict(d: dict[str, float]) -> ATRSSMetrics:
    """Reconstruct ATRSSMetrics from a flat dict (safe for missing keys)."""
    return ATRSSMetrics(**{k: v for k, v in d.items() if k in ATRSSMetrics.__dataclass_fields__})


class _SharedPoolBatchEvaluator:
    """Wraps an existing mp.Pool -- does NOT own it (no close/terminate on the pool itself)."""

    def __init__(
        self,
        pool: mp.Pool,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        on_terminate: Any = None,
    ):
        self._pool = pool
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._on_terminate = on_terminate

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [
            (c.name, c.mutations, current_mutations,
             self._phase, self._scoring_weights, self._hard_rejects)
            for c in candidates
        ]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        pass  # Pool is owned by the plugin, not by this evaluator

    def terminate(self) -> None:
        if callable(self._on_terminate):
            self._on_terminate()


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker
        init_worker(str(self._data_dir), self._initial_equity)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate
        return [
            score_candidate((c.name, c.mutations, current_mutations,
                             self._phase, self._scoring_weights, self._hard_rejects))
            for c in candidates
        ]

    def close(self) -> None:
        pass


class ATRSSPlugin:
    """ATRSS phased auto-optimization plugin (duck-typing StrategyPlugin)."""

    name = "atrss"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = None,
    ):
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        # Persistent pool -- created lazily, reused across phases
        self._pool: mp.Pool | None = None
        # Data cache for compute_final_metrics (avoids reloading parquets)
        self._cached_data: Any | None = None
        # Cross-phase caches: evaluation cache is namespaced by signature_prefix
        # so stale scores from a prior phase (different weights/rejects) are never hit.
        # Metrics cache stores raw metrics (settings-independent) for compute_final_metrics reuse.
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}

    # -- PhaseSpec -------------------------------------------------------------

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=n, mutations=m)
            for n, m in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.005,
                diagnostic_gap_fn=get_diagnostic_gaps,
                suggest_experiments_fn=suggest_experiments,
            ),
            max_rounds=None,
        )

    def _gate_criteria(
        self, phase: int, metrics: dict[str, float], state: PhaseState,
    ) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior_phase_metrics = None
        if phase == 4:
            prior_phase_metrics = state.phase_results.get(3, {}).get("final_metrics")
        return gate_criteria_for_phase(phase, m, prior_phase_metrics)

    # -- Evaluator factory -----------------------------------------------------

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        # Settings-aware cache key: same mutations scored with different
        # weights/rejects (different phase or scoring retry) get distinct entries.
        evaluation_key = mutation_signature(
            {"phase": phase, "scoring_weights": scoring_weights or {}, "hard_rejects": hard_rejects or {}},
        )

        def make_parallel():
            self._ensure_pool()
            return _SharedPoolBatchEvaluator(
                self._pool, phase, scoring_weights, hard_rejects,
                self._destroy_pool,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"ATRSS phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    # -- Pool lifecycle --------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None:
            return
        from .worker import init_worker

        processes = resolve_worker_processes(self.max_workers)
        self._pool = mp.Pool(
            processes=processes,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity),
        )
        logger.info("Created worker pool with %d processes", processes)

    def _destroy_pool(self) -> None:
        """Force-kill the pool (called on worker errors via terminate())."""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def close_pool(self) -> None:
        """Gracefully shut down the persistent worker pool.

        Called by PhaseRunner.run_all_phases() at the end of all phases.
        """
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    # -- Metrics ---------------------------------------------------------------

    def _ensure_data(self):
        """Load and cache PortfolioData for compute_final_metrics."""
        if self._cached_data is not None:
            return self._cached_data

        from backtests.swing.data.cache import load_bars
        from backtests.swing.data.preprocessing import (
            align_daily_to_hourly,
            build_numpy_arrays,
            filter_rth,
            normalize_timezone,
        )
        from backtests.swing.engine.portfolio_engine import PortfolioData

        data = PortfolioData()
        for sym in ("QQQ", "GLD"):
            h_df = normalize_timezone(load_bars(self.data_dir / f"{sym}_1h.parquet"))
            h_df = filter_rth(h_df)
            d_df = normalize_timezone(load_bars(self.data_dir / f"{sym}_1d.parquet"))
            data.hourly[sym] = build_numpy_arrays(h_df)
            data.daily[sym] = build_numpy_arrays(d_df)
            data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)
        self._cached_data = data
        return data

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)

        # Check cross-phase metrics cache (raw metrics are settings-independent)
        cached = self._metrics_cache.get(sig)
        if cached:
            return dict(cached)

        from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
        from backtests.swing.engine.portfolio_engine import run_independent
        from backtests.swing.auto.config_mutator import mutate_atrss_config

        base_config = BacktestConfig(
            symbols=["QQQ", "GLD"],
            initial_equity=self.initial_equity,
            fixed_qty=10,
            data_dir=self.data_dir,
            slippage=SlippageConfig(commission_per_contract=1.00),
            flags=AblationFlags(stall_exit=False),
        )
        config = mutate_atrss_config(base_config, mutations)
        data = self._ensure_data()
        result = run_independent(data, config)
        metrics = extract_atrss_metrics(result, self.initial_equity)
        result_dict = asdict(metrics)
        self._metrics_cache[sig] = result_dict
        return result_dict

    # -- Diagnostics -----------------------------------------------------------

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase=phase,
            metrics=_metrics_from_dict(metrics),
            greedy_result=greedy_result_to_dict(greedy_result),
        )

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase=phase,
            metrics=_metrics_from_dict(metrics),
            greedy_result=greedy_result_to_dict(greedy_result),
            force_all_phases=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(
            self.num_phases, state, metrics, final_greedy,
        )

        dimension_reports = {
            "signal_extraction": (
                f"PF={m.profit_factor:.2f}, WR={m.win_rate:.1%}, "
                f"{m.total_trades} trades ({m.trades_per_month:.1f}/month)."
            ),
            "exit_management": (
                f"MFE capture={m.mfe_capture:.3f}, total R={m.total_r:+.1f}, "
                f"avg R={m.avg_r:+.3f}."
            ),
            "risk_management": (
                f"Max DD={m.max_dd_pct:.2%}, Calmar R={m.calmar_r:.1f}, "
                f"Sharpe={m.sharpe:.2f}."
            ),
        }
        overall_verdict = (
            f"ATRSS R1: {m.total_trades} trades, PF={m.profit_factor:.2f}, "
            f"+{m.total_r:.1f}R, DD={m.max_dd_pct:.2%}, "
            f"Calmar R={m.calmar_r:.1f}, MFE capture={m.mfe_capture:.3f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
        )
