from __future__ import annotations

import multiprocessing as mp
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    deserialize_experiments,
    greedy_result_from_state,
    mutation_signature,
    resolve_worker_processes,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion

from .phase_candidates import (
    BASE_MUTATIONS,
    PHASE_FOCUS,
    R5_BASE_MUTATIONS,
    R5_PHASE_FOCUS,
    V2R1_BASE_MUTATIONS,
    V2R1_PHASE_CANDIDATES,
    V2R1_PHASE_FOCUS,
    V2R2_BASE_MUTATIONS,
    V2R2_PHASE_CANDIDATES,
    V2R2_PHASE_FOCUS,
    V2R3_BASE_MUTATIONS,
    V2R3_PHASE_CANDIDATES,
    V2R3_PHASE_FOCUS,
    V2R4_BASE_MUTATIONS,
    V2R4_PHASE_CANDIDATES,
    V2R4_PHASE_FOCUS,
    V3R1_BASE_MUTATIONS,
    V3R1_PHASE_CANDIDATES,
    V3R1_PHASE_FOCUS,
    V4R1_BASE_MUTATIONS,
    V4R1_PHASE_CANDIDATES,
    V4R1_PHASE_FOCUS,
    get_phase_candidate_lookup,
    get_phase_candidates,
    get_r5_phase_candidate_lookup,
    get_r5_phase_candidates,
    get_v2r1_phase_candidate_lookup,
    get_v2r1_phase_candidates,
    get_v2r2_phase_candidate_lookup,
    get_v2r2_phase_candidates,
    get_v2r3_phase_candidate_lookup,
    get_v2r3_phase_candidates,
    get_v2r4_phase_candidate_lookup,
    get_v2r4_phase_candidates,
    get_v3r1_phase_candidate_lookup,
    get_v3r1_phase_candidates,
    get_v4r1_phase_candidate_lookup,
    get_v4r1_phase_candidates,
)
from .phase_scoring import (
    R5_PHASE_HARD_REJECTS_BY_PROFILE,
    R5_ULTIMATE_TARGETS,
    V2R1_PHASE_HARD_REJECTS_BY_PROFILE,
    V2R1_ULTIMATE_TARGETS,
    V2R2_PHASE_HARD_REJECTS_BY_PROFILE,
    V2R2_ULTIMATE_TARGETS,
    V2R3_PHASE_HARD_REJECTS_BY_PROFILE,
    V2R3_ULTIMATE_TARGETS,
    V2R4_PHASE_HARD_REJECTS_BY_PROFILE,
    V2R4_ULTIMATE_TARGETS,
    V3R1_PHASE_HARD_REJECTS_BY_PROFILE,
    V3R1_ULTIMATE_TARGETS,
    V4R1_PHASE_HARD_REJECTS_BY_PROFILE,
    V4R1_ULTIMATE_TARGETS,
    enrich_phase_score_metrics,
    get_phase_scoring_weights,
    get_r5_phase_scoring_weights,
    get_v2r1_phase_scoring_weights,
    get_v2r2_phase_scoring_weights,
    get_v2r3_phase_scoring_weights,
    get_v2r4_phase_scoring_weights,
    get_v3r1_phase_scoring_weights,
    get_v4r1_phase_scoring_weights,
    merge_pullback_metrics,
)

MAINLINE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 60, "max_dd_pct": 0.06, "min_pf": 1.75},
    2: {"min_trades": 55, "max_dd_pct": 0.06, "min_pf": 1.80},
    3: {"min_trades": 62, "max_dd_pct": 0.06, "min_pf": 1.75},
    4: {"min_trades": 58, "max_dd_pct": 0.06, "min_pf": 1.85},
    5: {"min_trades": 58, "max_dd_pct": 0.06, "min_pf": 1.90},
}

AGGRESSIVE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 55, "max_dd_pct": 0.07, "min_pf": 1.65},
    2: {"min_trades": 50, "max_dd_pct": 0.07, "min_pf": 1.70},
    3: {"min_trades": 60, "max_dd_pct": 0.07, "min_pf": 1.65},
    4: {"min_trades": 55, "max_dd_pct": 0.07, "min_pf": 1.75},
    5: {"min_trades": 55, "max_dd_pct": 0.065, "min_pf": 1.80},
}

PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": MAINLINE_PHASE_HARD_REJECTS,
    "aggressive": AGGRESSIVE_PHASE_HARD_REJECTS,
}

ULTIMATE_TARGETS = {
    "avg_r": 0.20,
    "expected_total_r": 22.0,
    "profit_factor": 1.90,
    "sharpe": 1.40,
    "max_drawdown_pct": 0.06,
    "managed_exit_share": 0.60,
    "eod_flatten_inverse": 0.60,
    "total_trades": 120.0,
}


def _min_gate(name: str, target: float, actual: float) -> GateCriterion:
    return GateCriterion(name, float(target), float(actual), float(actual) >= float(target))


def _max_gate(name: str, target: float, actual: float) -> GateCriterion:
    return GateCriterion(f"max_{name}", float(target), float(actual), float(actual) <= float(target))


def _merge_snapshot_gate_metrics(
    metrics: dict[str, float],
    snapshot: dict[str, Any] | None,
) -> dict[str, float]:
    if not snapshot:
        return metrics

    enriched = dict(metrics)
    carry_audit = ((snapshot.get("exit_replacement") or {}).get("carry_audit") or {})
    actually_carried = float(carry_audit.get("actually_carried", 0) or 0.0)
    carried_underperformed_flatten = float(carry_audit.get("carried_underperformed_flatten", 0) or 0.0)
    enriched["carried_underperformed_flatten_share"] = (
        carried_underperformed_flatten / actually_carried if actually_carried > 0 else 0.0
    )
    return enriched


def select_pullback_branch(mainline_metrics: dict[str, float], aggressive_metrics: dict[str, float]) -> dict[str, Any]:
    main = enrich_phase_score_metrics(mainline_metrics)
    aggressive = enrich_phase_score_metrics(aggressive_metrics)
    main_expected_total_r = float(main.get("expected_total_r", 0.0))
    aggressive_expected_total_r = float(aggressive.get("expected_total_r", 0.0))
    aggressive_qualifies = (
        aggressive_expected_total_r >= main_expected_total_r * 1.10
        and float(aggressive.get("profit_factor", 0.0)) >= 2.0
        and float(aggressive.get("max_drawdown_pct", 1.0)) <= 0.07
    )
    if aggressive_qualifies:
        return {
            "selected_profile": "aggressive",
            "reason": (
                "Aggressive branch cleared the 10% expected-total-R hurdle "
                "while still holding PF >= 2.0 and max DD <= 7%."
            ),
        }
    within_five_pct = main_expected_total_r >= aggressive_expected_total_r * 0.95
    mainline_has_better_risk = (
        float(main.get("profit_factor", 0.0)) >= float(aggressive.get("profit_factor", 0.0))
        and float(main.get("max_drawdown_pct", 1.0)) <= float(aggressive.get("max_drawdown_pct", 1.0))
    )
    if within_five_pct and mainline_has_better_risk:
        reason = (
            "Mainline stayed within 5% of aggressive expected-total-R "
            "while retaining better PF/DD."
        )
    else:
        reason = (
            "Aggressive failed the selection hurdle, so the mainline branch remains the default choice."
        )
    return {"selected_profile": "mainline", "reason": reason}


class _PoolBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        initial_equity: float,
        phase: int,
        hard_rejects: dict[str, float] | None,
        scoring_weights: dict[str, float] | None,
        max_workers: int | None,
        round_name: str = "r4",
    ):
        from .worker import init_worker

        self._pool = mp.Pool(
            processes=resolve_worker_processes(max_workers),
            initializer=init_worker,
            initargs=(
                str(data_dir),
                start_date,
                end_date,
                initial_equity,
                phase,
                hard_rejects,
                scoring_weights,
                round_name,
            ),
        )

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [(candidate.name, candidate.mutations, current_mutations) for candidate in candidates]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


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
        round_name: str = "r4",
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
            round_name,
        )

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        return [score_candidate((candidate.name, candidate.mutations, current_mutations)) for candidate in candidates]

    def close(self) -> None:
        return None


class IARICPullbackPlugin:
    name = "iaric_pullback"
    num_phases = 5
    initial_mutations = BASE_MUTATIONS
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        *,
        start_date: str = "2024-01-01",
        end_date: str = "2026-03-01",
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        num_phases: int = 5,
        profile: str = "mainline",
        round_name: str = "r4",
    ):
        self._round_name = round_name
        if round_name == "v4r1":
            phase_focus = V4R1_PHASE_FOCUS
        elif round_name == "v3r1":
            phase_focus = V3R1_PHASE_FOCUS
        elif round_name == "v2r4":
            phase_focus = V2R4_PHASE_FOCUS
        elif round_name == "v2r3":
            phase_focus = V2R3_PHASE_FOCUS
        elif round_name == "v2r2":
            phase_focus = V2R2_PHASE_FOCUS
        elif round_name == "v2r1":
            phase_focus = V2R1_PHASE_FOCUS
        elif round_name == "r5":
            phase_focus = R5_PHASE_FOCUS
        else:
            phase_focus = PHASE_FOCUS
        if not 1 <= num_phases <= max(phase_focus):
            raise ValueError(
                f"IARICPullbackPlugin supports between 1 and {max(phase_focus)} phases, got {num_phases}."
            )
        profile_key = str(profile or "mainline").lower()
        if round_name == "v4r1":
            hard_rejects_map = V4R1_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "v3r1":
            hard_rejects_map = V3R1_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "v2r4":
            hard_rejects_map = V2R4_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "v2r3":
            hard_rejects_map = V2R3_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "v2r2":
            hard_rejects_map = V2R2_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "v2r1":
            hard_rejects_map = V2R1_PHASE_HARD_REJECTS_BY_PROFILE
        elif round_name == "r5":
            hard_rejects_map = R5_PHASE_HARD_REJECTS_BY_PROFILE
        else:
            hard_rejects_map = PHASE_HARD_REJECTS_BY_PROFILE
        if profile_key not in hard_rejects_map:
            raise ValueError(f"Unsupported IARIC pullback profile: {profile}")
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self.profile = profile_key
        self.name = f"iaric_pullback_{self._round_name}_{self.profile}"
        self._cached_replay = None
        self._baseline_metrics_cache: dict[str, float] | None = None
        self._last_context: dict[str, Any] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._config_cache: dict[tuple, dict[str, Any]] = {}

        if round_name == "v4r1":
            self.initial_mutations = V4R1_BASE_MUTATIONS
            self.ultimate_targets = dict(V4R1_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v4r1_phase_scoring_weights(self.profile)
        elif round_name == "v3r1":
            self.initial_mutations = V3R1_BASE_MUTATIONS
            self.ultimate_targets = dict(V3R1_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v3r1_phase_scoring_weights(self.profile)
        elif round_name == "v2r4":
            self.initial_mutations = V2R4_BASE_MUTATIONS
            self.ultimate_targets = dict(V2R4_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v2r4_phase_scoring_weights(self.profile)
        elif round_name == "v2r3":
            self.initial_mutations = V2R3_BASE_MUTATIONS
            self.ultimate_targets = dict(V2R3_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v2r3_phase_scoring_weights(self.profile)
        elif round_name == "v2r2":
            self.initial_mutations = V2R2_BASE_MUTATIONS
            self.ultimate_targets = dict(V2R2_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v2r2_phase_scoring_weights(self.profile)
        elif round_name == "v2r1":
            self.initial_mutations = V2R1_BASE_MUTATIONS
            self.ultimate_targets = dict(V2R1_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_v2r1_phase_scoring_weights(self.profile)
        elif round_name == "r5":
            self.initial_mutations = R5_BASE_MUTATIONS
            self.ultimate_targets = dict(R5_ULTIMATE_TARGETS)
            self._phase_hard_rejects = hard_rejects_map[self.profile]
            self._phase_scoring_weights = get_r5_phase_scoring_weights(self.profile)
        else:
            self.initial_mutations = BASE_MUTATIONS
            self.ultimate_targets = dict(ULTIMATE_TARGETS)
            self._phase_hard_rejects = PHASE_HARD_REJECTS_BY_PROFILE[self.profile]
            self._phase_scoring_weights = get_phase_scoring_weights(self.profile)

    def _baseline_metrics(self) -> dict[str, float]:
        if self._baseline_metrics_cache is None:
            self._baseline_metrics_cache = self._run_config(self.initial_mutations, store_context=False)["metrics"]
        return self._baseline_metrics_cache

    def _reference_metrics(self, phase: int, state: PhaseState) -> dict[str, float]:
        if phase > 1:
            previous = state.get_phase_metrics(phase - 1)
            if previous:
                return previous
        return self._baseline_metrics()

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        if self._round_name == "v4r1":
            phase_focus = V4R1_PHASE_FOCUS
        elif self._round_name == "v3r1":
            phase_focus = V3R1_PHASE_FOCUS
        elif self._round_name == "v2r4":
            phase_focus = V2R4_PHASE_FOCUS
        elif self._round_name == "v2r3":
            phase_focus = V2R3_PHASE_FOCUS
        elif self._round_name == "v2r2":
            phase_focus = V2R2_PHASE_FOCUS
        elif self._round_name == "v2r1":
            phase_focus = V2R1_PHASE_FOCUS
        elif self._round_name == "r5":
            phase_focus = R5_PHASE_FOCUS
        else:
            phase_focus = PHASE_FOCUS
        focus, focus_metrics = phase_focus[phase]
        prior = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior.get("suggested_experiments", []))
        suggested_tuples = [(experiment.name, experiment.mutations) for experiment in suggested] or None
        if self._round_name == "v4r1":
            candidate_tuples = get_v4r1_phase_candidates(
                phase, profile=self.profile, suggested_experiments=suggested_tuples,
                accepted_mutations=dict(state.cumulative_mutations) if phase > 1 else None,
            )
        elif self._round_name == "v3r1":
            candidate_tuples = get_v3r1_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        elif self._round_name == "v2r4":
            candidate_tuples = get_v2r4_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        elif self._round_name == "v2r3":
            candidate_tuples = get_v2r3_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        elif self._round_name == "v2r2":
            candidate_tuples = get_v2r2_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        elif self._round_name == "v2r1":
            candidate_tuples = get_v2r1_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        elif self._round_name == "r5":
            candidate_tuples = get_r5_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        else:
            candidate_tuples = get_phase_candidates(phase, profile=self.profile, suggested_experiments=suggested_tuples)
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in candidate_tuples
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=self._phase_scoring_weights.get(phase, {}),
            hard_rejects=self._phase_hard_rejects.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.0,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=24,
            prune_threshold=0.0,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        # Pre-seed baseline metrics into the cache so greedy round 1 gets a free hit
        baseline_key = mutation_signature(cumulative_mutations)
        if baseline_key not in self._metrics_cache:
            base_metrics = self._baseline_metrics() if cumulative_mutations == self.initial_mutations else None
            if base_metrics:
                self._metrics_cache[baseline_key] = dict(base_metrics)

        evaluation_key = mutation_signature(
            {"phase": phase, "scoring_weights": scoring_weights or {}, "hard_rejects": hard_rejects or {}}
        )

        def local_factory():
            return _LocalBatchEvaluator(
                self.data_dir,
                self.start_date,
                self.end_date,
                self.initial_equity,
                phase,
                hard_rejects,
                scoring_weights,
                self._round_name,
            )

        if self.max_workers == 1 or not _supports_spawn():
            evaluator = local_factory()
        else:
            evaluator = ResilientBatchEvaluator(
                preferred_factory=lambda: _PoolBatchEvaluator(
                    self.data_dir,
                    self.start_date,
                    self.end_date,
                    self.initial_equity,
                    phase,
                    hard_rejects,
                    scoring_weights,
                    self.max_workers,
                    self._round_name,
                ),
                fallback_factory=local_factory,
                description=f"{self.name} phase {phase} evaluator",
            )
        return CachedBatchEvaluator(
            evaluator,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        cached = self._metrics_cache.get(mutation_signature(mutations))
        if cached:
            return dict(cached)
        return self._run_config(mutations, store_context=True)["metrics"]

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        if self._round_name == "v4r1":
            phase_focus = V4R1_PHASE_FOCUS
        elif self._round_name == "v3r1":
            phase_focus = V3R1_PHASE_FOCUS
        elif self._round_name == "v2r4":
            phase_focus = V2R4_PHASE_FOCUS
        elif self._round_name == "v2r3":
            phase_focus = V2R3_PHASE_FOCUS
        elif self._round_name == "v2r2":
            phase_focus = V2R2_PHASE_FOCUS
        elif self._round_name == "v2r1":
            phase_focus = V2R1_PHASE_FOCUS
        elif self._round_name == "r5":
            phase_focus = R5_PHASE_FOCUS
        else:
            phase_focus = PHASE_FOCUS
        return _build_phase_snapshot(phase, phase_focus[phase][0], metrics, greedy_result, round_label=self._round_name.upper())

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic

        snapshot = self.run_phase_diagnostics(phase, state, metrics, greedy_result)
        trades = self._last_context.get("trades")
        replay = self._last_context.get("replay")
        if not trades or replay is None:
            return snapshot
        return snapshot + "\n\n" + pullback_full_diagnostic(
            trades,
            metrics=metrics,
            replay=replay,
            daily_selections=self._last_context.get("daily_selections"),
            candidate_ledger=self._last_context.get("candidate_ledger"),
            funnel_counters=self._last_context.get("funnel_counters"),
            rejection_log=self._last_context.get("rejection_log"),
            shadow_outcomes=self._last_context.get("shadow_outcomes"),
            selection_attribution=self._last_context.get("selection_attribution"),
            fsm_log=self._last_context.get("fsm_log"),
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        from backtests.stock.analysis.iaric_pullback_diagnostics import compute_pullback_diagnostic_snapshot
        from backtests.stock.analysis.iaric_pullback_round_diagnostics import build_pullback_round_comparison_report

        final_ctx = self._run_config(state.cumulative_mutations, store_context=True, collect_diagnostics=True)
        base_ctx = self._run_config(self.initial_mutations, store_context=False, collect_diagnostics=True)
        phase4_mutations = state.phase_results.get(4, {}).get("final_mutations") if state.phase_results.get(4) else None
        phase4_ctx = self._run_config(phase4_mutations, store_context=False, collect_diagnostics=True) if phase4_mutations else None
        phase5_mutations = state.phase_results.get(5, {}).get("final_mutations") if state.phase_results.get(5) else None
        phase5_ctx = self._run_config(phase5_mutations, store_context=False, collect_diagnostics=True) if phase5_mutations else (final_ctx if self.num_phases == 5 else None)
        final_metrics = final_ctx["metrics"]
        base_metrics = base_ctx["metrics"]
        final_trades = final_ctx["trades"]
        base_trades = base_ctx["trades"]
        final_snap = compute_pullback_diagnostic_snapshot(
            final_trades,
            metrics=final_metrics,
            replay=final_ctx.get("replay"),
            candidate_ledger=final_ctx.get("candidate_ledger"),
            funnel_counters=final_ctx.get("funnel_counters"),
            rejection_log=final_ctx.get("rejection_log"),
            shadow_outcomes=final_ctx.get("shadow_outcomes"),
            selection_attribution=final_ctx.get("selection_attribution"),
            fsm_log=final_ctx.get("fsm_log"),
        )
        base_snap = compute_pullback_diagnostic_snapshot(
            base_trades,
            metrics=base_metrics,
            replay=base_ctx.get("replay"),
            candidate_ledger=base_ctx.get("candidate_ledger"),
            funnel_counters=base_ctx.get("funnel_counters"),
            rejection_log=base_ctx.get("rejection_log"),
            shadow_outcomes=base_ctx.get("shadow_outcomes"),
            selection_attribution=base_ctx.get("selection_attribution"),
            fsm_log=base_ctx.get("fsm_log"),
        )
        ablations = self._run_ablation_suite(state, final_metrics=final_metrics)
        temporal = self._run_temporal_walkforward(state.cumulative_mutations)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=final_metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, final_metrics, final_greedy)

        dimension_reports = {
            "signal_extraction": "\n".join([
                f"Rebased baseline reproduced at {int(base_metrics['total_trades'])} trades, avg_r {base_metrics['avg_r']:+.3f}, PF {base_metrics['profit_factor']:.2f}; final bundle is {int(final_metrics['total_trades'])} trades, avg_r {final_metrics['avg_r']:+.3f}, PF {final_metrics['profit_factor']:.2f}.",
                f"RSI depth edge {base_metrics['rsi_depth_edge']:+.3f} -> {final_metrics['rsi_depth_edge']:+.3f}; trend-band edge {base_metrics['trend_band_edge']:+.3f} -> {final_metrics['trend_band_edge']:+.3f}.",
                f"Rejected-shadow avg_r {base_snap['shadow']['shadow']['avg_r']:+.3f} -> {final_snap['shadow']['shadow']['avg_r']:+.3f}; accept rate {base_snap['funnel']['accept_rate']:.1%} -> {final_snap['funnel']['accept_rate']:.1%}.",
                "SMA-distance deltas:",
                *_bucket_lines(
                    base_trades,
                    final_trades,
                    [
                        ("0-2%", lambda t: _meta_float(t, 'entry_sma_dist_pct') < 2.0),
                        ("2-5%", lambda t: 2.0 <= _meta_float(t, 'entry_sma_dist_pct') < 5.0),
                        ("5-10%", lambda t: 5.0 <= _meta_float(t, 'entry_sma_dist_pct') < 10.0),
                        ("10%+", lambda t: _meta_float(t, 'entry_sma_dist_pct') >= 10.0),
                    ],
                ),
            ]),
            "signal_discrimination": "\n".join([
                f"Late-rank edge {base_metrics['late_rank_edge']:+.3f} -> {final_metrics['late_rank_edge']:+.3f}; mean entry rank {base_metrics['mean_entry_rank']:.2f} -> {final_metrics['mean_entry_rank']:.2f}.",
                f"Crowded-day entered avg_r {base_snap['selection']['entered_avg_r']:+.3f} -> {final_snap['selection']['entered_avg_r']:+.3f}; skipped shadow avg_r {base_snap['selection']['skipped_avg_shadow_r']:+.3f} -> {final_snap['selection']['skipped_avg_shadow_r']:+.3f}.",
                "Rank-bucket deltas:",
                *_bucket_lines(
                    base_trades,
                    final_trades,
                    [
                        ("Rank 1-3", lambda t: 1 <= _meta_int(t, 'entry_rank') <= 3),
                        ("Rank 4-8", lambda t: 4 <= _meta_int(t, 'entry_rank') <= 8),
                        ("Rank 9+", lambda t: _meta_int(t, 'entry_rank') >= 9),
                    ],
                ),
                "CDD-bucket deltas:",
                *_bucket_lines(
                    base_trades,
                    final_trades,
                    [
                        ("CDD 0-1", lambda t: _meta_int(t, 'entry_cdd') <= 1),
                        ("CDD 2-3", lambda t: 2 <= _meta_int(t, 'entry_cdd') <= 3),
                        ("CDD 4-5", lambda t: 4 <= _meta_int(t, 'entry_cdd') <= 5),
                        ("CDD 6+", lambda t: _meta_int(t, 'entry_cdd') >= 6),
                    ],
                ),
            ]),
            "entry_mechanism": "\n".join([
                f"Sharpe {base_metrics['sharpe']:.2f} -> {final_metrics['sharpe']:.2f}; max DD {base_metrics['max_drawdown_pct']:.1%} -> {final_metrics['max_drawdown_pct']:.1%}.",
                f"Best timing variant: baseline {_best_variant(base_snap['entry_timing'])}; final {_best_variant(final_snap['entry_timing'])}.",
                "Weekday mix deltas:",
                *_weekday_lines(base_trades, final_trades),
            ]),
            "trade_management": "\n".join([
                f"Managed exits {base_metrics['managed_exit_share']:.1%} -> {final_metrics['managed_exit_share']:.1%}; carry share {base_metrics['carry_trade_share']:.1%} -> {final_metrics['carry_trade_share']:.1%}; carry avg_r {base_metrics['carry_avg_r']:+.3f} -> {final_metrics['carry_avg_r']:+.3f}.",
                f"Carry funnel quality-gated count {base_snap['carry_funnel']['flow_ok']} -> {final_snap['carry_funnel']['flow_ok']}; profitable-at-close EOD count {base_snap['carry_funnel']['profitable']} -> {final_snap['carry_funnel']['profitable']}.",
                f"Stop-hit cost {base_metrics['stop_hit_total_r']:+.2f}R -> {final_metrics['stop_hit_total_r']:+.2f}R; avg stop-hit R {base_metrics['stop_hit_avg_r']:+.3f} -> {final_metrics['stop_hit_avg_r']:+.3f}.",
                "Exit-mix deltas:",
                *_exit_mix_lines(base_trades, final_trades),
            ]),
            "exit_mechanism": "\n".join([
                f"EOD flatten share {base_metrics['eod_flatten_share']:.1%} -> {final_metrics['eod_flatten_share']:.1%}; positive EOD share {base_metrics['positive_eod_share']:.1%} -> {final_metrics['positive_eod_share']:.1%}.",
                f"Economic exit frontier best variant: baseline {_best_variant(base_snap['exit_frontier'])}; final {_best_variant(final_snap['exit_frontier'])}.",
                f"Final target check: avg_r {final_metrics['avg_r']:+.3f} ({_pass(final_metrics['avg_r'] >= self.ultimate_targets['avg_r'])}), PF {final_metrics['profit_factor']:.2f} ({_pass(final_metrics['profit_factor'] >= self.ultimate_targets['profit_factor'])}), sharpe {final_metrics['sharpe']:.2f} ({_pass(final_metrics['sharpe'] >= self.ultimate_targets['sharpe'])}), DD {final_metrics['max_drawdown_pct']:.1%} ({_pass(final_metrics['max_drawdown_pct'] <= self.ultimate_targets['max_drawdown_pct'])}), managed exits {final_metrics['managed_exit_share']:.1%} ({_pass(final_metrics['managed_exit_share'] >= self.ultimate_targets['managed_exit_share'])}), EOD flatten {final_metrics['eod_flatten_share']:.1%} ({_pass(1.0 - final_metrics['eod_flatten_share'] >= self.ultimate_targets['eod_flatten_inverse'])}), trades {int(final_metrics['total_trades'])} ({_pass(final_metrics['total_trades'] >= self.ultimate_targets['total_trades'])}).",
            ]),
        }
        extra_sections = {
            "robustness": "\n".join([
                "Kept-feature ablations:",
                *(ablations or ["  (no kept features to ablate)"]),
                "",
                "4-slice temporal walk-forward:",
                *(temporal or ["  (insufficient date range for temporal slicing)"]),
            ]),
            "round_comparison": build_pullback_round_comparison_report(
                base_ctx,
                final_ctx,
                phase4_ctx=phase4_ctx,
                phase5_ctx=phase5_ctx,
            ),
        }
        overall_verdict = (
            f"Signal extraction {'is' if final_metrics['rsi_depth_edge'] > 0 and final_metrics['trend_band_edge'] > 0 else 'is not yet'} "
            f"capturing alpha in the intended pullback cohorts, while discrimination quality is framed by late-rank edge "
            f"{final_metrics['late_rank_edge']:+.3f} and the rejected-shadow comparison in the final diagnostics. "
            f"Entry, management, and exit quality currently resolve to avg_r {final_metrics['avg_r']:+.3f}, PF {final_metrics['profit_factor']:.2f}, "
            f"managed exits {final_metrics['managed_exit_share']:.1%}, and EOD flatten {final_metrics['eod_flatten_share']:.1%}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
            extra_sections=extra_sections,
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        gaps: list[str] = []
        if phase == 1:
            if metrics.get("eod_flatten_share", 0.0) > 0.20:
                gaps.append("EOD flatten share is still too high; protection mechanisms need to convert more EOD exits.")
            if metrics.get("protection_candidate_share", 0.0) > 0.15:
                gaps.append("Too many losing trades still show >0.25R of MFE before turning negative.")
        elif phase == 2:
            if metrics.get("signal_score_edge", 0.0) < 0.08:
                gaps.append("Daily signal score is not separating high-alpha pullbacks from weak ones.")
            if metrics.get("crowded_day_missed_alpha_days", 0.0) > 125.0:
                gaps.append("Crowded-day ranking is still skipping too much positive shadow alpha.")
        elif phase == 3:
            if metrics.get("route_diversity", 0.0) < 0.67:
                gaps.append("Route diversification is still insufficient; need at least 2 active routes.")
            if metrics.get("routes_with_min_10_trades", 0.0) < 2.0:
                gaps.append("Route architecture is still too concentrated; at least two routes need real trade counts.")
        elif phase == 4:
            if metrics.get("carry_avg_r", 0.0) < 0.80:
                gaps.append("Carry quality is below target; calibration needs tightening.")
            if metrics.get("managed_exit_share", 0.0) < 0.65:
                gaps.append("Managed exit share regressed during carry calibration.")
        elif phase == 5:
            if metrics.get("profit_factor", 0.0) < 2.00:
                gaps.append("Risk-adjusted quality is still below the robustness floor.")
            if metrics.get("max_drawdown_pct", 1.0) > 0.07:
                gaps.append("Drawdown is still above the robustness ceiling.")
            if metrics.get("eod_flatten_share", 0.0) > 0.25:
                gaps.append("EOD flatten is still above the round target.")
        return gaps

    def suggest_experiments(self, phase: int, metrics: dict[str, float], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del state
        if self._round_name == "v4r1":
            lookup = get_v4r1_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "v3r1":
            lookup = get_v3r1_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "v2r4":
            lookup = get_v2r4_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "v2r3":
            lookup = get_v2r3_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "v2r2":
            lookup = get_v2r2_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "v2r1":
            lookup = get_v2r1_phase_candidate_lookup(phase, profile=self.profile)
        elif self._round_name == "r5":
            lookup = get_r5_phase_candidate_lookup(phase, profile=self.profile)
        else:
            lookup = get_phase_candidate_lookup(phase, profile=self.profile)
        suggestions: list[str] = []
        if phase == 1:
            suggestions.extend(["protect_030_after_050", "breakeven_050", "quick_exit_050_stale8"])
            if self.profile == "aggressive":
                suggestions.append("combo_aggressive_protect")
        elif phase == 2:
            if metrics.get("signal_score_edge", 0.0) < 0.05:
                suggestions.extend(["hybrid_alpha_floor52", "quality_hybrid_floor50"])
            suggestions.extend(["floor_59", "cdd_min3_max5", "combo_signal_focus"])
            if self.profile == "aggressive":
                suggestions.append("quality_hybrid_floor48_rescue52")
        elif phase == 3:
            suggestions.extend(["open_scored_balanced", "reclaim_selective", "combo_dual_route"])
            if self.profile == "aggressive":
                suggestions.append("open_scored_wide")
        elif phase == 4:
            suggestions.extend(["carry_frontier_combo", "carry_score_55", "carry_open_high_quality"])
        elif phase == 5:
            suggestions.extend(["thu_full_risk", "atr_stop_150", "reserve_2"])
            if self.profile == "aggressive":
                suggestions.extend(["weekday_full_risk", "sector_cap_6"])
        if not suggestions and weaknesses:
            suggestions.extend(list(lookup.keys())[:2])
        unique: list[Experiment] = []
        seen: set[str] = set()
        for name in suggestions:
            if name in seen or name not in lookup:
                continue
            seen.add(name)
            unique.append(Experiment(name=name, mutations=lookup[name]))
        return unique

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        del phase, state, greedy_result
        return {
            "signal_model": {
                "signal_score_edge": metrics["signal_score_edge"],
                "crowded_day_discrimination": metrics["crowded_day_discrimination"],
                "crowded_day_missed_alpha_inverse": metrics["crowded_day_missed_alpha_inverse"],
                "crowded_day_missed_alpha_days": metrics["crowded_day_missed_alpha_days"],
                "expected_total_r": metrics["expected_total_r"],
            },
            "route_shape": {
                "route_score_monotonicity": metrics["route_score_monotonicity"],
                "route_middle_bucket_deficit": metrics["route_middle_bucket_deficit"],
                "delayed_confirm_share": metrics["delayed_confirm_share"],
                "routes_with_min_10_trades": metrics["routes_with_min_10_trades"],
                "route_diversity": metrics["route_diversity"],
            },
            "exit_capture": {
                "managed_exit_share": metrics["managed_exit_share"],
                "eod_flatten_share": metrics["eod_flatten_share"],
                "carry_avg_r": metrics["carry_avg_r"],
                "binary_carried_share": metrics["binary_carried_share"],
                "carried_underperformed_flatten_share": metrics.get("carried_underperformed_flatten_share", 0.0),
                "protection_candidate_share": metrics["protection_candidate_share"],
                "stop_hit_total_r": metrics["stop_hit_total_r"],
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        return [
            "Signal model: " + ", ".join(f"{k}={v:+.3f}" for k, v in extra.get("signal_model", {}).items()),
            "Route shape: " + ", ".join(
                f"{k}={v:.1%}"
                if any(token in k for token in ("share", "diversity", "monotonicity"))
                else f"{k}={v:+.3f}"
                for k, v in extra.get("route_shape", {}).items()
            ),
            "Exit capture: " + ", ".join(
                f"{k}={v:.1%}" if "share" in k else f"{k}={v:+.3f}" if "avg_r" in k else f"{k}={v:+.2f}"
                for k, v in extra.get("exit_capture", {}).items()
            ),
        ]

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        hard = self._phase_hard_rejects.get(phase, {})
        criteria = [
            GateCriterion("hard_total_trades", float(hard["min_trades"]), float(metrics["total_trades"]), float(metrics["total_trades"]) >= float(hard["min_trades"])),
            GateCriterion("hard_profit_factor", float(hard["min_pf"]), float(metrics["profit_factor"]), float(metrics["profit_factor"]) >= float(hard["min_pf"])),
            GateCriterion("hard_max_drawdown_pct", float(hard["max_dd_pct"]), float(metrics["max_drawdown_pct"]), float(metrics["max_drawdown_pct"]) <= float(hard["max_dd_pct"])),
        ]
        ref = self._reference_metrics(phase, state)

        if phase == 1:
            managed_target = max(0.55, float(ref.get("managed_exit_share", 0.0)))
            eod_target = float(ref.get("eod_flatten_share", metrics["eod_flatten_share"]))
            protection_target = float(ref.get("protection_candidate_share", 0.0)) * (2.0 / 3.0)
            criteria.extend(
                [
                    _min_gate("avg_r", 0.19, metrics["avg_r"]),
                    _min_gate("managed_exit_share", managed_target, metrics["managed_exit_share"]),
                    _max_gate("eod_flatten_share", eod_target, metrics["eod_flatten_share"]),
                    _max_gate("protection_candidate_share", protection_target, metrics["protection_candidate_share"]),
                ]
            )
        elif phase == 2:
            signal_edge_target = max(float(ref.get("signal_score_edge", 0.0)) + 0.06, 0.06)
            criteria.extend(
                [
                    _min_gate("avg_r", 0.18, metrics["avg_r"]),
                    _min_gate("signal_score_edge", signal_edge_target, metrics["signal_score_edge"]),
                    _max_gate("crowded_day_missed_alpha_days", 125.0, metrics["crowded_day_missed_alpha_days"]),
                ]
            )
        elif phase == 3:
            expected_total_r_target = float(ref.get("expected_total_r", 0.0)) * 1.02
            criteria.extend(
                [
                    _min_gate("avg_r", 0.17, metrics["avg_r"]),
                    _min_gate("expected_total_r", expected_total_r_target, metrics["expected_total_r"]),
                    _min_gate("route_diversity", 0.67, metrics["route_diversity"]),
                ]
            )
        elif phase == 4:
            managed_target = float(ref.get("managed_exit_share", 0.0))
            carry_avg_r_target = float(ref.get("carry_avg_r", 0.0)) * 0.95
            criteria.extend(
                [
                    _min_gate("avg_r", 0.17, metrics["avg_r"]),
                    _min_gate("managed_exit_share", managed_target, metrics["managed_exit_share"]),
                    _min_gate("carry_avg_r", carry_avg_r_target, metrics["carry_avg_r"]),
                ]
            )
        elif phase == 5:
            expected_total_r_target = float(ref.get("expected_total_r", 0.0))
            criteria.extend(
                [
                    _min_gate("avg_r", 0.17, metrics["avg_r"]),
                    _min_gate("expected_total_r", expected_total_r_target, metrics["expected_total_r"]),
                    _min_gate("profit_factor", 2.00, metrics["profit_factor"]),
                    _max_gate("drawdown_pct", 0.07, metrics["max_drawdown_pct"]),
                ]
            )
        return criteria

    def _ensure_replay(self):
        if self._cached_replay is None:
            from backtests.stock.engine.research_replay import ResearchReplayEngine

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
        from backtests.stock._aliases import install
        from backtests.stock.analysis.iaric_pullback_diagnostics import compute_pullback_diagnostic_snapshot
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.auto.scoring import extract_metrics
        from backtests.stock.config_iaric import IARICBacktestConfig
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        install()
        effective_start = start_date or self.start_date
        effective_end = end_date or self.end_date
        diagnostics_enabled = bool(collect_diagnostics or store_context)
        sig = mutation_signature(mutations)
        cache_key = (sig, effective_start, effective_end, diagnostics_enabled)

        # Check config cache -- a diagnostics-enabled result satisfies non-diagnostic requests
        cached = self._config_cache.get(cache_key)
        if cached is None and not diagnostics_enabled:
            cached = self._config_cache.get((sig, effective_start, effective_end, True))
        if cached is not None:
            if store_context:
                self._last_context = cached
            return cached

        replay = self._ensure_replay()
        config = mutate_iaric_config(
            IARICBacktestConfig(
                start_date=effective_start,
                end_date=effective_end,
                initial_equity=self.initial_equity,
                tier=3,
                data_dir=self.data_dir,
            ),
            mutations,
        )
        result = IARICPullbackEngine(config, replay, collect_diagnostics=diagnostics_enabled).run()
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, self.initial_equity)
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
        diagnostic_snapshot = compute_pullback_diagnostic_snapshot(
            result.trades,
            metrics=metrics,
            replay=replay,
            daily_selections=result.daily_selections,
            candidate_ledger=result.candidate_ledger,
            funnel_counters=result.funnel_counters,
            rejection_log=result.rejection_log,
            shadow_outcomes=result.shadow_outcomes,
            selection_attribution=result.selection_attribution,
            fsm_log=result.fsm_log,
        )
        metrics = _merge_snapshot_gate_metrics(metrics, diagnostic_snapshot)
        context = {
            "metrics": metrics,
            "trades": result.trades,
            "replay": replay,
            "daily_selections": result.daily_selections,
            "candidate_ledger": result.candidate_ledger,
            "funnel_counters": result.funnel_counters,
            "rejection_log": result.rejection_log,
            "shadow_outcomes": result.shadow_outcomes,
            "selection_attribution": result.selection_attribution,
            "fsm_log": result.fsm_log,
            "config": config,
            "diagnostic_snapshot": diagnostic_snapshot,
        }

        # Populate caches
        self._config_cache[cache_key] = context
        if diagnostics_enabled:
            self._config_cache[(sig, effective_start, effective_end, False)] = context
        self._metrics_cache[sig] = dict(metrics)
        if store_context:
            self._last_context = context
        return context

    def _run_ablation_suite(self, state: PhaseState, *, final_metrics: dict[str, float] | None = None) -> list[str]:
        sequence = _resolve_kept_sequence(state, profile=self.profile, round_name=self._round_name)
        if not sequence:
            return []
        if final_metrics is None:
            final_metrics = self._run_config(state.cumulative_mutations, store_context=False)["metrics"]
        lines: list[str] = []
        for idx, experiment in enumerate(sequence):
            mutations = dict(self.initial_mutations)
            for keep_idx, keep_experiment in enumerate(sequence):
                if keep_idx != idx:
                    mutations.update(keep_experiment.mutations)
            metrics = self._run_config(mutations, store_context=False)["metrics"]
            lines.append(
                f"  {experiment.name}: avg_r {metrics['avg_r'] - final_metrics['avg_r']:+.3f}, PF {metrics['profit_factor'] - final_metrics['profit_factor']:+.2f}, sharpe {metrics['sharpe'] - final_metrics['sharpe']:+.2f}, eod {metrics['eod_flatten_share'] - final_metrics['eod_flatten_share']:+.1%}"
            )
        return lines

    def _run_temporal_walkforward(self, mutations: dict[str, Any]) -> list[str]:
        trade_dates = self._ensure_replay().tradable_dates(date.fromisoformat(self.start_date), date.fromisoformat(self.end_date))
        lines: list[str] = []
        for idx, chunk in enumerate([list(chunk) for chunk in np.array_split(trade_dates, 4) if len(chunk)], start=1):
            metrics = self._run_config(
                mutations,
                start_date=chunk[0].isoformat(),
                end_date=chunk[-1].isoformat(),
                store_context=False,
            )["metrics"]
            lines.append(
                f"  Slice {idx} {chunk[0]}->{chunk[-1]}: trades={int(metrics['total_trades'])}, avg_r={metrics['avg_r']:+.3f}, PF={metrics['profit_factor']:.2f}, sharpe={metrics['sharpe']:.2f}, DD={metrics['max_drawdown_pct']:.1%}"
            )
        return lines


def _build_phase_snapshot(phase: int, focus: str, metrics: dict[str, float], greedy_result, *, round_label: str = "R4") -> str:
    return "\n".join([
        "=" * 70,
        f"IARIC T3 {round_label} PHASE {phase} SNAPSHOT",
        "=" * 70,
        f"Focus: {focus}",
        f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f} with {greedy_result.accepted_count} accepted mutations.",
        "",
        (
            f"Core: trades={int(metrics['total_trades'])}, avg_r={metrics['avg_r']:+.3f}, "
            f"exp_total_r={metrics['expected_total_r']:+.2f}, pf={metrics['profit_factor']:.2f}, "
            f"sharpe={metrics['sharpe']:.2f}, dd={metrics['max_drawdown_pct']:.1%}"
        ),
        (
            f"Edges: rsi_depth={metrics['rsi_depth_edge']:+.3f}, trend_band={metrics['trend_band_edge']:+.3f}, "
            f"late_rank={metrics['late_rank_edge']:+.3f}"
        ),
        (
            f"Exit capture: managed={metrics['managed_exit_share']:.1%}, carry={metrics['carry_trade_share']:.1%}, "
            f"carry_avg_r={metrics['carry_avg_r']:+.3f}, eod={metrics['eod_flatten_share']:.1%}, "
            f"stop_cost={metrics['stop_hit_total_r']:+.2f}R"
        ),
    ])


def _resolve_kept_sequence(state: PhaseState, *, profile: str = "mainline", round_name: str = "r4") -> list[Experiment]:
    sequence: list[Experiment] = []
    if round_name == "v4r1":
        lookup_fn = get_v4r1_phase_candidate_lookup
    elif round_name == "v3r1":
        lookup_fn = get_v3r1_phase_candidate_lookup
    elif round_name == "v2r4":
        lookup_fn = get_v2r4_phase_candidate_lookup
    elif round_name == "v2r3":
        lookup_fn = get_v2r3_phase_candidate_lookup
    elif round_name == "v2r2":
        lookup_fn = get_v2r2_phase_candidate_lookup
    elif round_name == "v2r1":
        lookup_fn = get_v2r1_phase_candidate_lookup
    elif round_name == "r5":
        lookup_fn = get_r5_phase_candidate_lookup
    else:
        lookup_fn = get_phase_candidate_lookup
    for phase in sorted(state.phase_results):
        phase_result = state.phase_results.get(phase, {})
        suggested = deserialize_experiments(phase_result.get("suggested_experiments", []))
        lookup = lookup_fn(
            phase,
            profile=profile,
            suggested_experiments=[(experiment.name, experiment.mutations) for experiment in suggested] or None,
        )
        for name in phase_result.get("kept_features", []):
            if name in lookup:
                sequence.append(Experiment(name=name, mutations=lookup[name]))
    return sequence


def _bucket_lines(base_trades, final_trades, buckets) -> list[str]:
    lines: list[str] = []
    for label, predicate in buckets:
        base_group = [trade for trade in base_trades if predicate(trade)]
        final_group = [trade for trade in final_trades if predicate(trade)]
        lines.append(
            f"  {label}: avg_r {_avg_r(base_group):+.3f}->{_avg_r(final_group):+.3f} ({_avg_r(final_group) - _avg_r(base_group):+.3f}), n {len(base_group)}->{len(final_group)}"
        )
    return lines


def _weekday_lines(base_trades, final_trades) -> list[str]:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return [
        f"  {label}: avg_r {_avg_r([t for t in base_trades if t.entry_time.weekday() == idx]):+.3f}->{_avg_r([t for t in final_trades if t.entry_time.weekday() == idx]):+.3f}, n {sum(1 for t in base_trades if t.entry_time.weekday() == idx)}->{sum(1 for t in final_trades if t.entry_time.weekday() == idx)}"
        for idx, label in enumerate(labels)
    ]


def _exit_mix_lines(base_trades, final_trades) -> list[str]:
    reasons = sorted({
        *(trade.exit_reason or "UNKNOWN" for trade in base_trades),
        *(trade.exit_reason or "UNKNOWN" for trade in final_trades),
    })
    return [
        f"  {reason}: share {_share(sum(1 for t in base_trades if (t.exit_reason or 'UNKNOWN') == reason), len(base_trades)):.1%}->{_share(sum(1 for t in final_trades if (t.exit_reason or 'UNKNOWN') == reason), len(final_trades)):.1%}, avg_r {_avg_r([t for t in base_trades if (t.exit_reason or 'UNKNOWN') == reason]):+.3f}->{_avg_r([t for t in final_trades if (t.exit_reason or 'UNKNOWN') == reason]):+.3f}"
        for reason in reasons
    ]


def _avg_r(trades) -> float:
    if not trades:
        return 0.0
    return float(np.mean([float(trade.r_multiple) for trade in trades]))


def _share(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _meta_float(trade, key: str, default: float = 0.0) -> float:
    value = trade.metadata.get(key, default) if getattr(trade, "metadata", None) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _meta_int(trade, key: str, default: int = 0) -> int:
    value = trade.metadata.get(key, default) if getattr(trade, "metadata", None) else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _pass(condition: bool) -> str:
    return "PASS" if condition else "MISS"


def _best_variant(rows: list[dict[str, float]] | None) -> str:
    if not rows:
        return "n/a"
    best = max(rows, key=lambda item: float(item.get("avg_r", 0.0)))
    return f"{best.get('label', '?')} @ {float(best.get('avg_r', 0.0)):+.3f}R"


def _supports_spawn() -> bool:
    if sys.platform != "win32":
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", "")
    return bool(main_path) and not str(main_path).startswith("<")
