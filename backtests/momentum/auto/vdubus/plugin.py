"""VdubusNQ R1 phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
4 phases: trail rehabilitation, entry precision, signal discrimination, fine-tuning.

Key design: immutable scoring weights (no phase-variable scoring).
Phase-specific control is via progressive hard rejects only.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scoring import VdubusCompositeScore, VdubusMetrics

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
from .phase_gates import gate_criteria_for_phase

# ---------------------------------------------------------------------------
# Immutable scoring weights -- same across all phases
# Phase-specific control is via PHASE_HARD_REJECTS only
# ---------------------------------------------------------------------------

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,  # Use BASE_WEIGHTS from scoring.py
    2: None,
    3: None,
    4: None,
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"max_dd_pct": 0.22, "min_trades": 80, "min_pf": 1.5},
    2: {"max_dd_pct": 0.20, "min_trades": 90, "min_pf": 1.8},
    3: {"max_dd_pct": 0.20, "min_trades": 80, "min_pf": 2.0},
    4: {"max_dd_pct": 0.20, "min_trades": 80, "min_pf": 2.0},
}

PHASE_FOCUS = {
    1: ("Trail Rehabilitation & Exit Management", ["capture_ratio", "stale_exit_pct", "profit_factor", "sharpe"]),
    2: ("Entry Precision & Filter Tuning", ["total_trades", "win_rate", "fast_death_pct", "sharpe"]),
    3: ("Signal Discrimination & Regime", ["calmar", "max_dd_pct", "evening_trade_pct", "net_return_pct"]),
    4: ("Fine-tune & Interaction", ["calmar", "net_return_pct", "capture_ratio", "sharpe"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 2000.0,
    "profit_factor": 3.5,
    "max_dd_pct": 0.12,
    "calmar": 20.0,
    "total_trades": 180.0,
    "capture_ratio": 0.55,
    "sharpe": 2.0,
}


def score_phase_metrics(
    phase: int,
    metrics: VdubusMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> VdubusCompositeScore:
    from .scoring import VdubusCompositeScore, composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    if metrics.total_trades < rejects.get("min_trades", 40):
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades})")
    if metrics.max_dd_pct > rejects.get("max_dd_pct", 0.30):
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_dd_pct:.2%})")
    if "min_pf" in rejects and metrics.profit_factor < rejects["min_pf"]:
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f})")

    # Immutable weights -- no phase-specific overrides
    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {k: v / total for k, v in base.items()} if total > 0 else base
    return composite_score(metrics, weights)


# ---------------------------------------------------------------------------
# Batch evaluators
# ---------------------------------------------------------------------------

class _SharedPoolBatchEvaluator:
    """Pool evaluator that keeps the pool alive across close() calls."""

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
# Plugin
# ---------------------------------------------------------------------------

_INITIAL_MUTATIONS = {"flags.plus_1r_partial": False, "param_overrides.CHOP_THRESHOLD": 40}


class VdubusPlugin:
    name = "vdubus"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = _INITIAL_MUTATIONS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 4,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(f"VdubusPlugin supports 1-{max(PHASE_FOCUS)} phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._cached_data: dict[str, Any] | None = None
        self._last_context: dict[str, Any] = {}
        self._pool: mp.Pool | None = None
        self._last_metrics_sig: str = ""
        self._last_metrics_result: dict[str, float] | None = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                state.cumulative_mutations,
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
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
                decide_action_fn=self.decide_phase_action,
            ),
            max_rounds=50,
            prune_threshold=0.05,
        )

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
                self._pool, phase, scoring_weights, hard_rejects, self._destroy_pool,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"vdubus phase {phase}")
        return CachedBatchEvaluator(raw)

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
        """Gracefully shut down the persistent worker pool."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    # -- Metrics ---------------------------------------------------------------

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        if sig == self._last_metrics_sig and self._last_metrics_result is not None:
            return self._last_metrics_result

        from backtests.momentum._aliases import install

        install()

        from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
        from backtest.engine.vdubus_engine import VdubusEngine
        from backtests.momentum.auto.config_mutator import mutate_vdubus_config
        from backtests.momentum.auto.vdubus.scoring import extract_vdubus_metrics
        from backtests.momentum.auto.vdubus.worker import load_worker_data

        if self._cached_data is None:
            self._cached_data = load_worker_data("NQ", self.data_dir)

        config = mutate_vdubus_config(
            VdubusBacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                fixed_qty=10,
                flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
            ),
            mutations,
        )
        engine = VdubusEngine("NQ", config)
        result = engine.run(**self._cached_data)
        metrics = extract_vdubus_metrics(
            result.trades,
            list(result.equity_curve),
            list(result.time_series),
            self.initial_equity,
        )
        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "trades": result.trades,
        }
        metrics_dict = asdict(metrics)
        self._last_metrics_sig = sig
        self._last_metrics_result = metrics_dict
        return metrics_dict

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            None,
            self._last_context.get("trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            None,
            self._last_context.get("trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        extraction = (
            f"Net return {m.net_return_pct:.1f}% with {m.total_trades} trades, "
            f"PF={m.profit_factor:.2f}, DD={m.max_dd_pct:.1%}."
        )
        discrimination = (
            f"Win rate {m.win_rate:.1%}, avg R={m.avg_r:.3f}, "
            f"capture ratio {m.capture_ratio:.2f}."
        )
        management = (
            f"Calmar={m.calmar:.2f}, Sharpe={m.sharpe:.2f}, Sortino={m.sortino:.2f}."
        )
        exits = (
            f"Stale exit pct {m.stale_exit_pct:.1%}, multi-session {m.multi_session_pct:.1%}, "
            f"avg hold {m.avg_hold_hours:.1f}h."
        )
        timing = (
            f"Fast deaths {m.fast_death_pct:.1%}, "
            f"evening trades {m.evening_trade_pct:.1%} (avgR={m.evening_avg_r:+.3f}), "
            f"trades/month {m.trades_per_month:.1f}."
        )
        overall = (
            f"VdubusNQ optimized to {m.net_return_pct:.1f}% return, PF={m.profit_factor:.2f}, "
            f"DD={m.max_dd_pct:.1%}, {m.total_trades} trades. "
            f"Capture ratio {'improved to' if m.capture_ratio > 0.47 else 'at'} {m.capture_ratio:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports={
                "performance": extraction,
                "signal_quality": discrimination,
                "risk_management": management,
                "exit_mechanism": exits,
                "timing": timing,
            },
            overall_verdict=overall,
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
        seen = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name in seen:
                return
            seen.add(name)
            suggestions.append(Experiment(name=name, mutations=mutations))

        weakness_text = " ".join(weaknesses).lower()

        # Capture ratio low -- trail issues
        if m.capture_ratio < 0.45:
            add("suggest_be_wide_45", {"flags.plus_1r_partial": True, "param_overrides.TRAIL_MULT_BASE": 4.5})
            add("suggest_mfe_exempt_040", {"flags.stale_mfe_exempt": True, "param_overrides.STALE_MFE_EXEMPT_R": 0.40})

        # High stale rate
        if m.stale_exit_pct > 0.40 or "stale" in weakness_text:
            add("suggest_stale_bars_10", {"param_overrides.STALE_BARS_15M": 10})
            add("suggest_mfe_exempt_default", {"flags.stale_mfe_exempt": True})

        # Fast deaths
        if m.fast_death_pct > 0.20 or "fast death" in weakness_text:
            add("suggest_disable_early_kill", {"flags.early_kill": False})
            add("suggest_vwap_cap_065", {"param_overrides.VWAP_CAP_CORE": 0.65})

        # High drawdown
        if m.max_dd_pct > 0.15 or "drawdown" in weakness_text:
            add("suggest_lower_risk_008", {"param_overrides.BASE_RISK_PCT": 0.008})

        # Low trade count
        if m.total_trades < 120 or "trade count" in weakness_text:
            add("suggest_relax_floor_015", {"param_overrides.FLOOR_PCT": 0.15})

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        # Immutable weights -- return None to keep defaults
        return None

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        m = _metrics_from_dict(metrics)
        return {
            "exit_profile": {
                "stale_exit_pct": m.stale_exit_pct,
                "capture_ratio": m.capture_ratio,
                "multi_session_pct": m.multi_session_pct,
            },
            "entry_quality": {
                "fast_death_pct": m.fast_death_pct,
                "evening_trade_pct": m.evening_trade_pct,
                "evening_avg_r": m.evening_avg_r,
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        ep = extra.get("exit_profile", {})
        if ep:
            lines.append(
                f"Exit profile: stale={ep.get('stale_exit_pct', 0):.1%}, "
                f"capture={ep.get('capture_ratio', 0):.2f}, "
                f"multi-session={ep.get('multi_session_pct', 0):.1%}"
            )
        eq = extra.get("entry_quality", {})
        if eq:
            lines.append(
                f"Entry quality: fast_deaths={eq.get('fast_death_pct', 0):.1%}, "
                f"evening={eq.get('evening_trade_pct', 0):.1%} (avgR={eq.get('evening_avg_r', 0):+.3f})"
            )
        return lines

    def decide_phase_action(
        self,
        phase: int,
        metrics: dict[str, float],
        state: PhaseState,
        greedy_result,
        gate_result,
        current_weights: dict[str, float] | None,
        analysis,
        max_scoring_retries: int,
        max_diagnostic_retries: int,
    ) -> PhaseDecision | None:
        return None  # Use default framework logic

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior = state.get_phase_metrics(phase - 1) if phase > 1 else None
        return gate_criteria_for_phase(phase, m, prior)


def _metrics_from_dict(metrics: dict[str, float]) -> VdubusMetrics:
    from .scoring import VdubusMetrics

    fields = VdubusMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            val = metrics[key]
            # Ensure int fields get ints
            if field_info.type == "int" or (hasattr(field_info, 'default') and isinstance(field_info.default, int)):
                val = int(val)
            payload[key] = val
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return VdubusMetrics(**payload)
