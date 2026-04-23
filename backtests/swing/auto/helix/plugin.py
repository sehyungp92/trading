"""Helix phased auto-optimization plugin implementing StrategyPlugin protocol."""
from __future__ import annotations

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
    seen_experiment_names,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

from .phase_candidates import get_phase_candidates
from .scoring import HelixCompositeScore, HelixMetrics, composite_score

# ---------------------------------------------------------------------------
# Phase-specific scoring weights
# ---------------------------------------------------------------------------

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,  # use defaults
    2: {
        "exit_efficiency": 0.22,
        "waste_ratio": 0.18,
        "net_profit": 0.18,
        "pf": 0.12,
        "tail_preservation": 0.10,
        "inv_dd": 0.08,
        "frequency": 0.12,
    },
    3: {
        "net_profit": 0.25,
        "inv_dd": 0.12,
        "pf": 0.18,
        "exit_efficiency": 0.15,
        "waste_ratio": 0.10,
        "tail_preservation": 0.08,
        "frequency": 0.12,
    },
    4: {
        "net_profit": 0.14,
        "pf": 0.14,
        "exit_efficiency": 0.15,
        "waste_ratio": 0.14,
        "tail_preservation": 0.14,
        "inv_dd": 0.14,
        "frequency": 0.15,
    },
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 200, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80},
    2: {"min_trades": 200, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80},
    3: {"min_trades": 200, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80},
    4: {"min_trades": 200, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80},
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("SIGNAL_PRUNING + ENTRY_GATES", ["profit_factor", "total_trades", "waste_ratio"]),
    2: ("EXIT_MANAGEMENT", ["exit_efficiency", "waste_ratio", "net_return_pct"]),
    3: ("VOLATILITY + ADDON", ["net_return_pct", "profit_factor", "max_r_dd"]),
    4: ("SIZING + FINETUNE", ["calmar_r", "net_return_pct", "exit_efficiency"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 100.0,
    "profit_factor": 2.5,
    "max_r_dd": 8.0,
    "exit_efficiency": 0.35,
    "total_trades": 400.0,
}

_GATE_TO_SCORING = {
    "net_return_pct": "net_profit",
    "profit_factor": "pf",
    "total_trades": "frequency",
    "calmar_r": "net_profit",
    "max_r_dd": "inv_dd",
    "exit_efficiency": "exit_efficiency",
    "waste_ratio": "waste_ratio",
    "tail_pct": "tail_preservation",
}


def score_phase_metrics(
    phase: int,
    metrics: HelixMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> HelixCompositeScore:
    """Score metrics with phase-specific weights and hard rejects."""
    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {key: value / total for key, value in base.items()} if total > 0 else base
    return composite_score(metrics, weights, hard_rejects=hard_rejects)


# ---------------------------------------------------------------------------
# Pool-based and sequential batch evaluators
# ---------------------------------------------------------------------------

class _SharedPoolBatchEvaluator:
    """Pool evaluator that keeps the pool alive across close() calls.

    The pool is managed by the plugin -- close() is a no-op so the pool
    survives between greedy runs / phases.  terminate() signals the plugin
    to destroy and recreate the pool on next use.
    """

    def __init__(
        self,
        pool: mp.Pool,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        on_terminate,
    ):
        self._pool = pool
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._on_terminate = on_terminate

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [
            (c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects)
            for c in candidates
        ]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        pass  # Pool stays alive for reuse across phases

    def terminate(self) -> None:
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
            score_candidate((c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            for c in candidates
        ]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HelixPlugin
# ---------------------------------------------------------------------------

class HelixPlugin:
    name = "helix"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 4,
    ):
        if not 1 <= num_phases <= 4:
            raise ValueError(f"HelixPlugin supports 1-4 phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._last_context: dict[str, Any] = {}
        # Persistent pool -- created lazily, reused across phases
        self._pool: mp.Pool | None = None
        self._pool_dirty = False
        # Data cache for compute_final_metrics (avoids reloading)
        self._cached_data: dict | None = None
        # Metrics memoization
        self._last_metrics_sig: str | None = None
        self._last_metrics_result: dict[str, float] | None = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]

        # Phase 4 gate needs P3 metrics for no-regression
        p3_metrics = state.get_phase_metrics(3) if phase == 4 else None

        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, _p=phase, _p3=p3_metrics: self._gate_criteria(_p, metrics, _p3),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.01,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=20,
            prune_threshold=0.05,
        )

    # ------------------------------------------------------------------
    # Persistent pool management
    # ------------------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None and not self._pool_dirty:
            return
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
        from .worker import init_worker

        processes = resolve_worker_processes(self.max_workers)
        self._pool = mp.Pool(
            processes=processes,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity),
        )
        self._pool_dirty = False

    def _on_pool_terminate(self) -> None:
        """Called by _SharedPoolBatchEvaluator.terminate() on error."""
        self._pool_dirty = True

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        def make_parallel():
            self._ensure_pool()
            return _SharedPoolBatchEvaluator(
                self._pool, phase, scoring_weights, hard_rejects,
                self._on_pool_terminate,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"Helix phase {phase}")
        return CachedBatchEvaluator(raw)

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        if sig == self._last_metrics_sig and self._last_metrics_result is not None:
            return self._last_metrics_result

        from backtests.swing._aliases import install
        install()

        from backtest.config_helix import HelixBacktestConfig
        from backtest.engine.helix_portfolio_engine import run_helix_independent
        from .config_mutator import mutate_helix_config
        from .scoring import extract_helix_metrics
        from .worker import load_helix_worker_data

        base_config = HelixBacktestConfig(initial_equity=self.initial_equity, data_dir=self.data_dir)
        config = mutate_helix_config(base_config, mutations)
        if self._cached_data is None:
            # Always load all default symbols so cache stays valid when
            # symbol-pruning mutations run first.
            self._cached_data = load_helix_worker_data(base_config.symbols, base_config.data_dir)

        result = run_helix_independent(self._cached_data, config)
        metrics = extract_helix_metrics(result, self.initial_equity)

        all_trades = []
        for sr in result.symbol_results.values():
            all_trades.extend(sr.trades)

        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "all_trades": all_trades,
        }
        metrics_dict = asdict(metrics)
        self._last_metrics_sig = sig
        self._last_metrics_result = metrics_dict
        return metrics_dict

    def run_phase_diagnostics(self, phase: int, state: PhaseState,
                               metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState,
                                  metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        metrics_obj = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        dimension_reports = {
            "signal_quality": (
                f"PF={metrics_obj.profit_factor:.2f}, Bull PF={metrics_obj.bull_pf:.2f}, "
                f"Bear PF={metrics_obj.bear_pf:.2f}. "
                f"Total trades: {metrics_obj.total_trades}."
            ),
            "exit_management": (
                f"Exit efficiency={metrics_obj.exit_efficiency:.3f}, "
                f"waste ratio={metrics_obj.waste_ratio:.3f}. "
                f"Stale R={metrics_obj.stale_r:.1f}, short-hold R={metrics_obj.short_hold_r:.1f}."
            ),
            "tail_preservation": (
                f"Big winners (>=3R) = {metrics_obj.tail_pct:.1%} of gross win R. "
                f"Big winner R={metrics_obj.big_winner_r:.1f}."
            ),
            "risk_management": (
                f"Max R DD={metrics_obj.max_r_dd:.2f}, Sharpe={metrics_obj.sharpe:.2f}, "
                f"Calmar(R)={metrics_obj.calmar_r:.2f}."
            ),
        }

        overall_verdict = (
            f"Exit efficiency {'meets' if metrics_obj.exit_efficiency >= 0.35 else 'below'} target "
            f"({metrics_obj.exit_efficiency:.3f} vs 0.35 target). "
            f"Waste ratio {'good' if metrics_obj.waste_ratio >= 0.60 else 'needs work'} "
            f"({metrics_obj.waste_ratio:.3f}). "
            f"Net return {metrics_obj.net_return_pct:.1f}% with PF {metrics_obj.profit_factor:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        from .phase_diagnostics import get_diagnostic_gaps
        return get_diagnostic_gaps(phase, _metrics_from_dict(metrics))

    def suggest_experiments(
        self,
        phase: int,
        metrics: dict[str, float],
        weaknesses: list[str],
        state: PhaseState,
    ) -> list[Experiment]:
        m = _metrics_from_dict(metrics)
        tested = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name not in tested:
                tested.add(name)
                suggestions.append(Experiment(name=name, mutations=mutations))

        if m.exit_efficiency < 0.25 and phase <= 2:
            add("sug_trail_base_30", {"param_overrides.TRAIL_BASE": 3.0})
            add("sug_fade_floor_10", {"param_overrides.TRAIL_FADE_FLOOR": 1.0})
            add("sug_stall_onset_5", {"param_overrides.TRAIL_STALL_ONSET": 5})

        if m.waste_ratio < 0.50:
            if phase <= 2:
                add("sug_stale_1h_20", {"param_overrides.STALE_1H_BARS": 20})
                add("sug_bail_d_6", {"param_overrides.CLASS_D_BAIL_BARS": 6})
                add("sug_early_stale_15", {"param_overrides.EARLY_STALE_BARS": 15})

        if m.min_regime_pf < 1.2:
            add("sug_prune_class_a", {"flags.disable_class_a": True})

        if m.tail_pct < 0.40 and phase >= 2:
            add("sug_partial_later", {"param_overrides.R_PARTIAL_2P5": 3.0})
            add("sug_no_partial_2p5", {"flags.disable_partial_2p5r": True})

        if m.profit_factor < 1.7 and phase <= 2:
            add("sug_adx_cap_40", {"param_overrides.ADX_UPPER_GATE": 40})
            add("sug_div_mag_floor_08", {"param_overrides.DIV_MAG_FLOOR": 0.08})

        if m.max_r_dd > 15 and phase >= 3:
            add("sug_emergency_neg15", {"param_overrides.EMERGENCY_STOP_R": -1.5})
            add("sug_circuit_daily_neg20", {"param_overrides.DAILY_STOP_R": -2.0})

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        base_weights = dict(current_weights or PHASE_WEIGHTS.get(phase) or {})
        if not base_weights:
            return None

        boosted = False
        for criterion in gate_result.criteria:
            if criterion.passed:
                continue
            scoring_key = _GATE_TO_SCORING.get(criterion.name.removeprefix("hard_"))
            if scoring_key and scoring_key in base_weights:
                base_weights[scoring_key] *= 1.5
                boosted = True

        for metric_name, progress in analysis.goal_progress.items():
            if progress.get("pct_of_target", 0) < 40:
                scoring_key = _GATE_TO_SCORING.get(metric_name)
                if scoring_key and scoring_key in base_weights:
                    base_weights[scoring_key] *= 1.3
                    boosted = True

        if not boosted:
            return None

        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()} if total > 0 else base_weights

    def build_analysis_extra(self, phase: int, metrics: dict[str, float],
                              state: PhaseState, greedy_result) -> dict[str, Any]:
        """Per-class R attribution for targeted insight."""
        all_trades = self._last_context.get("all_trades", [])
        class_r: dict[str, float] = {}
        for t in all_trades:
            cls = getattr(t, "setup_class", "?")
            class_r[cls] = class_r.get(cls, 0.0) + t.r_multiple
        return {"class_attribution": class_r}

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines = []
        class_r = extra.get("class_attribution", {})
        if class_r:
            parts = [f"{cls}={r:+.1f}R" for cls, r in sorted(class_r.items())]
            lines.append(f"Class R attribution: {', '.join(parts)}")
        return lines

    def close_pool(self) -> None:
        """Called in finally block by PhaseRunner.run_all_phases()."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    # ------------------------------------------------------------------
    # Gate criteria
    # ------------------------------------------------------------------

    def _gate_criteria(self, phase: int, metrics: dict[str, float],
                        p3_metrics: dict[str, float] | None = None) -> list[GateCriterion]:
        from . import phase_gates
        if phase == 1:
            return phase_gates.gate_criteria_phase_1(metrics)
        elif phase == 2:
            return phase_gates.gate_criteria_phase_2(metrics)
        elif phase == 3:
            return phase_gates.gate_criteria_phase_3(metrics)
        elif phase == 4:
            return phase_gates.gate_criteria_phase_4(metrics, p3_metrics)
        return []


def _metrics_from_dict(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)
