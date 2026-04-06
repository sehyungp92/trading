from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import build_end_of_round_report
from .greedy_optimizer import run_greedy
from .phase_analyzer import analyze_phase
from .phase_gates import evaluate_gate
from .phase_logging import PhaseLogger
from .phase_state import PhaseState, _utc_now_iso, load_phase_state, save_phase_state
from .plugin import StrategyPlugin
from .types import GateResult, GreedyResult, PhaseAnalysis


def _to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _analysis_action_allowed(
    analysis: PhaseAnalysis,
    state: PhaseState,
    phase: int,
    *,
    max_scoring_retries: int,
    max_diagnostic_retries: int,
) -> bool:
    if analysis.recommendation == "improve_scoring":
        return state.scoring_retries.get(phase, 0) < max_scoring_retries
    if analysis.recommendation == "improve_diagnostics":
        return state.diagnostic_retries.get(phase, 0) < max_diagnostic_retries
    return True


class PhaseRunner:
    def __init__(
        self,
        plugin: StrategyPlugin,
        output_dir: Path,
        round_name: str = "",
        *,
        max_rounds: int | None = None,
        min_delta: float = 0.001,
        max_retries: int = 2,
        max_diagnostic_retries: int = 1,
    ):
        self.plugin = plugin
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_name = round_name
        self.max_rounds = max_rounds
        self.min_delta = min_delta
        self.max_retries = max_retries
        self.max_diagnostic_retries = max_diagnostic_retries
        self.state_path = self.output_dir / "phase_state.json"
        self.phase_logger = PhaseLogger(self.output_dir, round_name=round_name)

    def load_state(self) -> PhaseState:
        state = load_phase_state(self.state_path)
        initial_mutations = getattr(self.plugin, "initial_mutations", None)
        if initial_mutations and not state.cumulative_mutations and not state.phase_results:
            state.cumulative_mutations = dict(initial_mutations)
        if self.round_name and not state.round_name:
            state.round_name = self.round_name
        return state

    def run_phase(self, phase: int, state: PhaseState | None = None) -> PhaseState:
        state = state or self.load_state()
        state = self._prepare_state_for_phase(state, phase)
        phase_log = self.phase_logger.get_phase_logger(phase)
        phase_base_mutations = dict(state.cumulative_mutations)
        state.start_phase(phase)
        save_phase_state(state, self.state_path)

        spec = self.plugin.get_phase_spec(phase, state)
        phase_candidates = _dedupe_experiments(spec.candidates)
        scoring_weights = spec.scoring_weights
        force_all_diagnostics = False
        greedy_result: GreedyResult | None = None
        metrics: dict[str, float] | None = None
        gate_result: GateResult | None = None

        phase_log.info("Starting phase %d with %d candidates", phase, len(phase_candidates))
        self.phase_logger.log_activity(
            phase,
            "phase_start",
            {
                "focus": spec.focus,
                "candidate_count": len(phase_candidates),
                "timestamp": _utc_now_iso(),
            },
        )

        while True:
            if greedy_result is None:
                self.phase_logger.log_activity(
                    phase,
                    "greedy_start",
                    {
                        "candidate_count": len(phase_candidates),
                        "scoring_retry": state.scoring_retries.get(phase, 0),
                        "diagnostic_retry": state.diagnostic_retries.get(phase, 0),
                    },
                )
                evaluate_batch = self.plugin.create_evaluate_batch(
                    phase,
                    phase_base_mutations,
                    scoring_weights=scoring_weights,
                    hard_rejects=spec.hard_rejects,
                )
                checkpoint_path = self.output_dir / f"phase_{phase}_greedy_checkpoint.json"
                greedy_result = run_greedy(
                    phase_candidates,
                    phase_base_mutations,
                    evaluate_batch,
                    max_rounds=self.max_rounds or spec.max_rounds or len(phase_candidates),
                    min_delta=self.min_delta,
                    prune_threshold=spec.prune_threshold if spec.prune_threshold is not None else 0.05,
                    checkpoint_path=checkpoint_path,
                    checkpoint_context={
                        "phase": phase,
                        "scoring_weights": scoring_weights or {},
                        "hard_rejects": spec.hard_rejects,
                    },
                    logger=phase_log,
                )
                self.phase_logger.save_phase_output(phase, "greedy_raw", _to_dict(greedy_result))
                try:
                    metrics = self.plugin.compute_final_metrics(greedy_result.final_mutations)
                except Exception:
                    phase_log.exception("compute_final_metrics failed for phase %d", phase)
                    raise
                if metrics is None:
                    raise ValueError(f"compute_final_metrics returned None for phase {phase}")
                greedy_result.final_metrics = metrics
                self.phase_logger.save_phase_output(phase, "greedy", _to_dict(greedy_result))
                self.phase_logger.log_activity(
                    phase,
                    "greedy_complete",
                    {
                        "base_score": greedy_result.base_score,
                        "final_score": greedy_result.final_score,
                        "accepted_count": greedy_result.accepted_count,
                    },
                )

                criteria = spec.gate_criteria_fn(metrics)
                gate_result = evaluate_gate(criteria, greedy_result)
                state.record_gate(phase, _gate_to_dict(gate_result))
                save_phase_state(state, self.state_path)
                self.phase_logger.log_activity(
                    phase,
                    "gate_check",
                    {
                        "passed": gate_result.passed,
                        "failure_category": gate_result.failure_category,
                    },
                )

            assert metrics is not None
            assert gate_result is not None
            assert greedy_result is not None

            try:
                diagnostics_text = (
                    self.plugin.run_enhanced_diagnostics(phase, state, metrics, greedy_result)
                    if force_all_diagnostics
                    else self.plugin.run_phase_diagnostics(phase, state, metrics, greedy_result)
                )
            except Exception:
                phase_log.exception("Diagnostics failed for phase %d", phase)
                diagnostics_text = f"[DIAGNOSTICS FAILED] See phase_{phase}.log for traceback."
            diagnostics_kind = "diagnostics_enhanced" if force_all_diagnostics else "diagnostics"
            self.phase_logger.save_phase_output(phase, diagnostics_kind, diagnostics_text)
            self.phase_logger.log_activity(
                phase,
                "diagnostics_run",
                {"enhanced": force_all_diagnostics},
            )

            policy = spec.analysis_policy
            if force_all_diagnostics:
                policy = replace(policy, diagnostic_gap_fn=lambda current_phase, current_metrics: [])
            analysis = analyze_phase(
                phase,
                greedy_result,
                metrics,
                state,
                gate_result,
                ultimate_targets=self.plugin.ultimate_targets,
                policy=policy,
                current_weights=scoring_weights,
                max_scoring_retries=self.max_retries,
                max_diagnostic_retries=self.max_diagnostic_retries,
            )
            if not _analysis_action_allowed(
                analysis,
                state,
                phase,
                max_scoring_retries=self.max_retries,
                max_diagnostic_retries=self.max_diagnostic_retries,
            ):
                analysis.recommendation = "advance"
                if analysis.suggested_experiments:
                    analysis.recommendation_reason = "Retry budget exhausted; carry suggested experiments into the next phase."
                else:
                    analysis.recommendation_reason = "Retry budget exhausted; advance with current best mutations."
            self.phase_logger.save_phase_output(phase, "analysis", _analysis_to_dict(analysis))
            self.phase_logger.log_activity(
                phase,
                "analysis_complete",
                {
                    "recommendation": analysis.recommendation,
                    "reason": analysis.recommendation_reason,
                },
            )

            if analysis.recommendation == "improve_scoring" and state.scoring_retries.get(phase, 0) < self.max_retries:
                state.increment_retry(phase)
                state.increment_scoring_retry(phase)
                save_phase_state(state, self.state_path)
                scoring_weights = analysis.scoring_weight_overrides or scoring_weights
                if analysis.suggested_experiments:
                    phase_candidates = _dedupe_experiments([*phase_candidates, *analysis.suggested_experiments])
                greedy_result = None
                metrics = None
                gate_result = None
                force_all_diagnostics = False
                self.phase_logger.log_activity(
                    phase,
                    "decision_improve_scoring",
                    {
                        "retry": state.scoring_retries.get(phase, 0),
                        "weight_overrides": scoring_weights or {},
                        "candidate_count": len(phase_candidates),
                    },
                )
                continue

            if analysis.recommendation == "improve_diagnostics" and state.diagnostic_retries.get(phase, 0) < self.max_diagnostic_retries:
                state.increment_retry(phase)
                state.increment_diagnostic_retry(phase)
                save_phase_state(state, self.state_path)
                force_all_diagnostics = True
                self.phase_logger.log_activity(
                    phase,
                    "decision_improve_diagnostics",
                    {"retry": state.diagnostic_retries.get(phase, 0)},
                )
                continue

            phase_new_mutations = {
                key: value
                for key, value in greedy_result.final_mutations.items()
                if phase_base_mutations.get(key) != value
            }
            phase_result = {
                "focus": spec.focus,
                "base_mutations": dict(phase_base_mutations),
                "final_mutations": dict(greedy_result.final_mutations),
                "base_score": greedy_result.base_score,
                "final_score": greedy_result.final_score,
                "kept_features": greedy_result.kept_features,
                "rounds": [_to_dict(round_result) for round_result in greedy_result.rounds],
                "final_metrics": metrics,
                "accepted_count": greedy_result.accepted_count,
                "elapsed_seconds": greedy_result.elapsed_seconds,
                "suggested_experiments": [_to_dict(experiment) for experiment in analysis.suggested_experiments],
                "analysis": _analysis_to_dict(analysis),
                "new_mutations": phase_new_mutations,
            }
            state.advance_phase(phase, phase_new_mutations, phase_result)
            state.record_gate(phase, _gate_to_dict(gate_result))
            save_phase_state(state, self.state_path)
            self.phase_logger.update_progress(phase, _progress_summary(state, phase))
            self.phase_logger.log_activity(
                phase,
                "decision_advance",
                {
                    "reason": analysis.recommendation_reason,
                    "gate_passed": gate_result.passed,
                    "new_mutation_count": len(phase_new_mutations),
                },
            )
            phase_log.info(
                "Phase %d complete: %.4f -> %.4f (%d accepted)",
                phase,
                greedy_result.base_score,
                greedy_result.final_score,
                greedy_result.accepted_count,
            )
            if phase == self.plugin.num_phases:
                self.run_end_of_round(state)
            return state

    def run_all_phases(self, start_phase: int | None = None) -> PhaseState:
        state = self.load_state()
        if self.round_name:
            state.round_name = self.round_name
        save_phase_state(state, self.state_path)

        backup_label = self.round_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.phase_logger.backup_state(self.state_path, backup_label)

        if start_phase is None:
            start_phase = max(state.completed_phases) + 1 if state.completed_phases else 1

        try:
            for phase in range(start_phase, self.plugin.num_phases + 1):
                state = self.run_phase(phase, state)

            report_path = self.output_dir / "round_evaluation.txt"
            if not report_path.exists():
                self.run_end_of_round(state)
        finally:
            # Allow plugins with persistent pools to clean up
            close_pool = getattr(self.plugin, "close_pool", None)
            if callable(close_pool):
                close_pool()
        return state

    def _prepare_state_for_phase(self, state: PhaseState, phase: int) -> PhaseState:
        stale_phases = _phases_at_or_after(state, phase)
        if stale_phases:
            rerun_label = f"pre_phase_{phase}_rerun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.phase_logger.backup_state(self.state_path, rerun_label)
            self.phase_logger.clear_generated_outputs(phase)

            for container in (
                state.phase_results,
                state.phase_gate_results,
                state.retry_count,
                state.scoring_retries,
                state.diagnostic_retries,
                state.phase_timestamps,
            ):
                for stale_phase in stale_phases:
                    container.pop(stale_phase, None)

            state.completed_phases = [completed for completed in state.completed_phases if completed < phase]
            state.current_phase = max(state.completed_phases, default=0)
            state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
            self.phase_logger.prune_progress(set(state.completed_phases), current_phase=state.current_phase)
            save_phase_state(state, self.state_path)
            self.phase_logger.log_activity(
                phase,
                "phase_reset",
                {
                    "discarded_phases": stale_phases,
                    "resuming_from_phase": phase,
                },
            )
        else:
            state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
        return state

    def _base_mutations_for_phase(self, state: PhaseState, phase: int) -> dict[str, Any]:
        base_mutations: dict[str, Any] = {}
        initial_mutations = getattr(self.plugin, "initial_mutations", None)
        if initial_mutations:
            base_mutations.update(initial_mutations)
        base_mutations.update(_mutations_through_phase(state, phase - 1))
        return base_mutations

    def run_end_of_round(self, state: PhaseState) -> str:
        artifacts = self.plugin.build_end_of_round_artifacts(state)
        (self.output_dir / "round_final_diagnostics.txt").write_text(artifacts.final_diagnostics_text, encoding="utf-8")
        report = build_end_of_round_report(self.plugin.name, state, artifacts)
        (self.output_dir / "round_evaluation.txt").write_text(report, encoding="utf-8")
        save_phase_state(state, self.state_path)
        self.phase_logger.log_activity(
            max(state.completed_phases) if state.completed_phases else 0,
            "end_of_round",
            {"completed_phases": state.completed_phases},
        )
        return report


def _gate_to_dict(gate_result: GateResult) -> dict:
    return {
        "passed": gate_result.passed,
        "criteria": [_to_dict(criterion) for criterion in gate_result.criteria],
        "failure_category": gate_result.failure_category,
        "recommendations": list(gate_result.recommendations),
    }


def _analysis_to_dict(analysis: PhaseAnalysis) -> dict:
    return {
        "phase": analysis.phase,
        "goal_progress": analysis.goal_progress,
        "strengths": analysis.strengths,
        "weaknesses": analysis.weaknesses,
        "scoring_assessment": analysis.scoring_assessment,
        "diagnostic_gaps": analysis.diagnostic_gaps,
        "suggested_experiments": [_to_dict(experiment) for experiment in analysis.suggested_experiments],
        "recommendation": analysis.recommendation,
        "recommendation_reason": analysis.recommendation_reason,
        "report": analysis.report,
        "scoring_weight_overrides": analysis.scoring_weight_overrides,
        "extra": analysis.extra,
    }


def _progress_summary(state: PhaseState, phase: int) -> dict:
    result = state.phase_results.get(phase, {})
    gate = state.phase_gate_results.get(phase, {})
    return {
        "status": "completed",
        "updated_at": _utc_now_iso(),
        "completed_phases": state.completed_phases,
        "current_phase": state.current_phase,
        "total_mutations": len(state.cumulative_mutations),
        "base_score": result.get("base_score", 0.0),
        "final_score": result.get("final_score", 0.0),
        "kept_features": result.get("kept_features", []),
        "gate_passed": gate.get("passed", False),
        "failure_category": gate.get("failure_category"),
        "scoring_retries": state.scoring_retries.get(phase, 0),
        "diagnostic_retries": state.diagnostic_retries.get(phase, 0),
    }


def _mutations_through_phase(state: PhaseState, phase: int) -> dict[str, Any]:
    """Accumulate mutations from phases 1..phase using new_mutations (preferred).

    Falls back to computing delta from final_mutations - base_mutations for
    legacy state files that lack new_mutations.
    """
    if phase <= 0:
        return {}

    mutations: dict[str, Any] = {}
    for phase_num in sorted(state.phase_results):
        if phase_num > phase:
            break
        result = state.phase_results[phase_num]
        new = result.get("new_mutations")
        if new is not None:
            mutations.update(new)
        else:
            final = result.get("final_mutations", {})
            base = result.get("base_mutations", {})
            for key, value in final.items():
                if base.get(key) != value:
                    mutations[key] = value
    return mutations


def _phases_at_or_after(state: PhaseState, phase: int) -> list[int]:
    phases = set()
    for container in (
        state.phase_results,
        state.phase_gate_results,
        state.retry_count,
        state.scoring_retries,
        state.diagnostic_retries,
        state.phase_timestamps,
    ):
        phases.update(key for key in container if key >= phase)
    phases.update(completed for completed in state.completed_phases if completed >= phase)
    if state.current_phase >= phase:
        phases.add(state.current_phase)
    return sorted(phases)


def _dedupe_experiments(experiments: list) -> list:
    deduped = []
    seen: set[str] = set()
    for experiment in experiments:
        if experiment.name in seen:
            continue
        seen.add(experiment.name)
        deduped.append(experiment)
    return deduped
