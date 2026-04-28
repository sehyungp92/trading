from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics
    from .scoring import DownturnCompositeScore

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.cache_keys import build_cache_key
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

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: {
        "correction_pnl": 0.30,
        "regime_purity": 0.18,
        "edge": 0.15,
        "risk": 0.15,
        "capture": 0.10,
        "calmar": 0.07,
        "frequency": 0.05,
    },
    2: {
        "correction_pnl": 0.28,
        "edge": 0.22,
        "capture": 0.22,
        "risk": 0.12,
        "regime_purity": 0.06,
        "calmar": 0.07,
        "frequency": 0.03,
    },
    3: {
        "risk": 0.25,
        "correction_pnl": 0.22,
        "edge": 0.18,
        "capture": 0.15,
        "calmar": 0.10,
        "regime_purity": 0.08,
        "frequency": 0.02,
    },
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 10, "max_dd_pct": 0.35, "min_correction_pnl_pct": 0.0},
    2: {"min_trades": 15, "max_dd_pct": 0.32, "min_correction_pnl_pct": 0.0},
    3: {"min_trades": 15, "max_dd_pct": 0.30, "min_pf": 1.0},
}

PHASE_FOCUS = {
    1: ("Regime Purification", ["correction_pnl_pct", "total_trades", "max_dd_pct"]),
    2: ("Capture Optimization", ["exit_efficiency", "profit_factor", "correction_pnl_pct"]),
    3: ("Risk Tuning", ["calmar", "max_dd_pct", "profit_factor"]),
}

ULTIMATE_TARGETS = {
    "correction_pnl_pct": 60.0,
    "profit_factor": 2.0,
    "net_return_pct": 50.0,
    "max_dd_pct": 20.0,
    "calmar": 2.0,
    "exit_efficiency": 0.35,
    "total_trades": 60.0,
}


def score_phase_metrics(
    phase: int,
    metrics: DownturnMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> DownturnCompositeScore:
    from .scoring import DownturnCompositeScore, composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    if metrics.total_trades < rejects.get("min_trades", 10):
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades})")
    if metrics.max_dd_pct > rejects.get("max_dd_pct", 0.35):
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_dd_pct:.2%})")
    if "min_pf" in rejects and metrics.profit_factor < rejects["min_pf"]:
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f})")
    if "min_correction_pnl_pct" in rejects and metrics.correction_pnl_pct < rejects["min_correction_pnl_pct"]:
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_corr_pnl ({metrics.correction_pnl_pct:.2f}%)")

    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {key: value / total for key, value in base.items()} if total > 0 else base
    return composite_score(metrics, weights)


class _SharedPoolBatchEvaluator:
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
            (candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects)
            for candidate in candidates
        ]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        pass

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


class DownturnPlugin:
    name = "downturn"
    num_phases = 3
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 3,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(
                f"DownturnPlugin supports between 1 and {max(PHASE_FOCUS)} phases, got {num_phases}."
            )
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._cached_bundle = None
        self._last_context: dict[str, Any] = {}
        self._pool: mp.Pool | None = None
        self._final_metrics_cache: dict[str, dict[str, Any]] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint: str = ""

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                state.cumulative_mutations,
                suggested_experiments=[(experiment.name, experiment.mutations) for experiment in suggested] or None,
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
                min_effective_score_delta_pct=0.0,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
                decide_action_fn=self.decide_phase_action,
            ),
            max_rounds=50,
            prune_threshold=0.05,
            reject_streak_limit=1,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        evaluation_key = build_cache_key(
            "downturn.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            self._ensure_pool()
            return _SharedPoolBatchEvaluator(
                self._pool,
                phase,
                scoring_weights,
                hard_rejects,
                self._destroy_pool,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"downturn phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def _replay_bundle(self):
        from backtests.momentum.auto.downturn.worker import load_worker_data

        bundle = load_worker_data("NQ", self.data_dir)
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._final_metrics_cache.clear()
            self._last_context = {}
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    def _ensure_pool(self) -> None:
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
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def close_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        from backtests.momentum.config_downturn import DownturnBacktestConfig
        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.downturn_engine import DownturnEngine
        from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
        from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config

        replay_bundle = self._replay_bundle()
        metrics_sig = mutation_signature(mutations)
        if self._last_context.get("mutation_signature") == metrics_sig:
            return dict(self._last_context["metrics"])

        cache_key = build_cache_key(
            "downturn.final_metrics",
            source_fingerprint=replay_bundle.cache_source_fingerprint,
            mutations=mutations,
            extra={"initial_equity": self.initial_equity},
        )
        cached = self._final_metrics_cache.get(cache_key)
        if cached is not None:
            self._last_context = cached["context"]
            return dict(cached["metrics"])

        config = mutate_downturn_config(
            DownturnBacktestConfig(initial_equity=self.initial_equity, data_dir=self.data_dir),
            mutations,
        )
        engine = DownturnEngine("NQ", config)
        result = engine.run(**replay_engine_kwargs(replay_bundle))
        metrics = compute_downturn_metrics(result, replay_bundle.data["daily"])
        metrics_dict = asdict(metrics)
        self._last_context = {
            "mutation_signature": metrics_sig,
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": dict(metrics_dict),
            "trades": result.trades,
            "cache_key": cache_key,
        }
        self._metrics_cache[metrics_sig] = dict(metrics_dict)
        self._final_metrics_cache[cache_key] = {
            "metrics": dict(metrics_dict),
            "context": self._last_context,
        }
        return metrics_dict

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            asdict(state),
            self._last_context.get("trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            asdict(state),
            self._last_context.get("trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        metrics_obj = _metrics_from_dict(metrics)
        extra = self.build_analysis_extra(self.num_phases, metrics, state, None)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        extraction = (
            f"Correction PnL is {metrics_obj.correction_pnl_pct:.1f}% with coverage {metrics_obj.correction_coverage:.1%}. "
            f"Bear capture ratio is {metrics_obj.bear_capture_ratio:.1%}."
        )
        discrimination = (
            f"Signal-to-entry ratio is {metrics_obj.signal_to_entry_ratio:.2f}. "
            f"Engine health summary: {', '.join(f'{k}={v}' for k, v in extra['engine_health'].items())}."
        )
        entry = (
            f"Total trades reached {metrics_obj.total_trades}; reversal/breakdown/fade split is "
            f"{metrics_obj.reversal_trades}/{metrics_obj.breakdown_trades}/{metrics_obj.fade_trades}."
        )
        management = (
            f"Max drawdown is {metrics_obj.max_dd_pct:.1%}, calmar is {metrics_obj.calmar:.2f}."
        )
        exits = (
            f"Exit efficiency is {metrics_obj.exit_efficiency:.2f}; average MFE capture is {metrics_obj.avg_mfe_capture:.2f}. "
            f"Median hold is {metrics_obj.median_hold_5m:.1f} bars."
        )
        correction_attr = extra.get("correction_attribution", {})
        regime_purity_ratio = correction_attr.get("ratio", 0.0)
        overall_verdict = (
            f"Correction-capture specialist: correction PnL {metrics_obj.correction_pnl_pct:.1f}% "
            f"({'meets' if metrics_obj.correction_pnl_pct >= 60.0 else 'below'} target of 60%). "
            f"Regime purity ratio {regime_purity_ratio:.2f}. "
            f"PF={metrics_obj.profit_factor:.2f}, DD={metrics_obj.max_dd_pct:.1%}, Calmar={metrics_obj.calmar:.2f}. "
            f"Exit efficiency {metrics_obj.exit_efficiency:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports={
                "signal_extraction": extraction,
                "signal_discrimination": discrimination,
                "entry_mechanism": entry,
                "trade_management": management,
                "exit_mechanism": exits,
            },
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
        metrics_obj = _metrics_from_dict(metrics)
        engine_health = _assess_engine_health(metrics_obj)
        weakness_text = " ".join(weaknesses).lower()
        seen = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name in seen:
                return
            seen.add(name)
            suggestions.append(Experiment(name=name, mutations=mutations))

        for engine, status in engine_health.items():
            if status == "harmful":
                add(f"ablate_{engine}", {f"flags.{engine}_engine": False})
            elif status == "insufficient_data":
                if engine == "reversal":
                    add("rev_relax_div_threshold_0.08", {"param_overrides.divergence_mag_threshold": 0.08})
                    add("rev_relax_trend_gate", {"flags.reversal_trend_weakness_gate": False})
                    add("rev_relax_extension_gate", {"flags.reversal_extension_gate": False})
                    add("rev_relax_corridor_cap", {"flags.reversal_corridor_cap": False})
                elif engine == "breakdown":
                    add("bd_relax_containment_0.60", {"param_overrides.box_containment_min": 0.60})
                    add("bd_no_chop_filter", {"flags.breakdown_chop_filter": False})
                    add("bd_relax_displacement_0.50", {"param_overrides.displacement_quantile": 0.50})
                elif engine == "fade":
                    add("fade_no_bear_required", {"flags.fade_bear_regime_required": False})
                    add("fade_no_momentum_confirm", {"flags.fade_momentum_confirm": False})
            elif status == "underperforming":
                if engine == "reversal":
                    add("rev_relax_div_threshold_0.10", {"param_overrides.divergence_mag_threshold": 0.10})
                    add("rev_relax_trend_gate", {"flags.reversal_trend_weakness_gate": False})
                elif engine == "breakdown":
                    add("bd_relax_containment_0.70", {"param_overrides.box_containment_min": 0.70})
                    add("bd_relax_displacement_0.55", {"param_overrides.displacement_quantile": 0.55})
                elif engine == "fade":
                    add("fade_widen_cap_0.40", {"param_overrides.vwap_cap_core": 0.40})
                    add("fade_no_bear_required", {"flags.fade_bear_regime_required": False})

        # Regime purity suggestions
        correction_attr = _compute_correction_attribution(self._last_context.get("trades"))
        if correction_attr.get("non_correction_pnl", 0.0) < -500:
            add("r2_counter_mult_0", {"param_overrides.regime_mult_counter": 0.0})
            add("r2_counter_neutral_zero", {
                "param_overrides.regime_mult_counter": 0.0,
                "param_overrides.regime_mult_neutral": 0.0,
            })

        if metrics_obj.correction_pnl_pct < 20.0:
            add("struct_corr_override", {"flags.correction_regime_override": True})
            add("r2_block_corr_override", {
                "flags.block_counter_regime": True,
                "flags.correction_regime_override": True,
            })

        if metrics_obj.max_dd_pct > 0.25:
            add("lower_risk_pct_0.008", {"param_overrides.base_risk_pct": 0.008})
            add("reduce_counter_mult_0.10", {"param_overrides.regime_mult_counter": 0.10})

        if metrics_obj.exit_efficiency < 0.20:
            add("faster_be_move", {"param_overrides.be_trigger_r": 0.5})
            add("tighter_chandelier_8", {"param_overrides.chandelier_lookback": 8})

        if "tp2/tp3 never hit" in weakness_text:
            add("tp2_lower_2.0R", {"param_overrides.tp2_r_aligned": 2.0})
            add("chandelier_wider_20", {"param_overrides.chandelier_lookback": 20})

        unique: dict[str, Experiment] = {}
        for experiment in suggestions:
            unique.setdefault(experiment.name, experiment)
        return list(unique.values())

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        weights = dict(current_weights or PHASE_WEIGHTS.get(phase) or {})
        if not weights:
            return None

        for criterion in gate_result.criteria:
            if criterion.passed:
                continue
            name = criterion.name.removeprefix("hard_")
            if name in {"correction_pnl_pct", "total_trades"}:
                weights["correction_pnl"] = weights.get("correction_pnl", 0.25) * 1.20
            if name in {"exit_efficiency"}:
                weights["capture"] = weights.get("capture", 0.15) * 1.25
            if name in {"profit_factor"}:
                weights["edge"] = weights.get("edge", 0.20) * 1.25
            if name in {"max_dd_pct", "calmar"}:
                weights["risk"] = weights.get("risk", 0.18) * 1.25

        weakness_text = " ".join(analysis.weaknesses).lower()
        if "exit efficiency" in weakness_text:
            weights["capture"] = weights.get("capture", 0.15) * 1.10
        if "regime purity" in weakness_text or "counter" in weakness_text:
            weights["regime_purity"] = weights.get("regime_purity", 0.10) * 1.15
        if "drawdown" in weakness_text:
            weights["risk"] = weights.get("risk", 0.18) * 1.10
            weights["calmar"] = weights.get("calmar", 0.07) * 1.15

        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()} if total > 0 else weights

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        metrics_obj = _metrics_from_dict(metrics)
        trades = self._last_context.get("trades")
        return {
            "engine_health": _assess_engine_health(metrics_obj),
            "correction_attribution": _compute_correction_attribution(trades),
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines = []
        engine_health = extra.get("engine_health", {})
        if engine_health:
            lines.append("Engine health: " + ", ".join(f"{engine}={status}" for engine, status in engine_health.items()))
        correction = extra.get("correction_attribution", {})
        if correction:
            lines.append(
                "Correction attribution: "
                f"corr_pnl={correction.get('correction_pnl', 0):.0f}, "
                f"non_corr_pnl={correction.get('non_correction_pnl', 0):.0f}, "
                f"ratio={correction.get('ratio', 0):.2f}"
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
        scoring_retries = state.scoring_retries.get(phase, 0)
        correction = analysis.extra.get("correction_attribution", {})
        if (
            correction.get("correction_pnl", 0.0) < 0.0
            and correction.get("non_correction_pnl", 0.0) > 0.0
            and scoring_retries < max_scoring_retries
        ):
            return PhaseDecision(
                action="improve_scoring",
                reason=(
                    "Strategy is profitable outside correction windows but loses during corrections; "
                    "reweight scoring toward correction capture and regime purity."
                ),
                scoring_assessment_override="MISALIGNED",
                scoring_weight_overrides=_correction_weight_overrides(phase, current_weights),
            )
        return None

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        metric_obj = _metrics_from_dict(metrics)
        criteria = [
            GateCriterion("hard_min_trades", 10.0, float(metric_obj.total_trades), metric_obj.total_trades >= 10),
            GateCriterion("hard_max_dd_pct", 0.30, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.30),
            GateCriterion("hard_correction_pnl_pct", 0.0, metric_obj.correction_pnl_pct, metric_obj.correction_pnl_pct >= 0.0),
        ]

        if phase == 1:
            criteria.extend(
                [
                    GateCriterion("correction_pnl_pct", 40.0, metric_obj.correction_pnl_pct, metric_obj.correction_pnl_pct >= 40.0),
                    GateCriterion("total_trades", 20.0, float(metric_obj.total_trades), metric_obj.total_trades >= 20),
                ]
            )
            return criteria

        if phase == 2:
            criteria.extend(
                [
                    GateCriterion("profit_factor", 1.5, metric_obj.profit_factor, metric_obj.profit_factor >= 1.5),
                    GateCriterion("exit_efficiency", 0.25, metric_obj.exit_efficiency, metric_obj.exit_efficiency >= 0.25),
                    GateCriterion("correction_pnl_pct", 50.0, metric_obj.correction_pnl_pct, metric_obj.correction_pnl_pct >= 50.0),
                ]
            )
            return criteria

        if phase == 3:
            criteria.extend(
                [
                    GateCriterion("max_dd_pct", 0.25, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.25),
                    GateCriterion("profit_factor", 1.3, metric_obj.profit_factor, metric_obj.profit_factor >= 1.3),
                    GateCriterion("calmar", 0.5, metric_obj.calmar, metric_obj.calmar >= 0.5),
                ]
            )
            # No-regression check against Phase 2
            prior_metrics = state.get_phase_metrics(2) or {}
            if prior_metrics:
                for key in ["correction_pnl_pct", "profit_factor", "exit_efficiency"]:
                    target = float(_payload_metric(prior_metrics, key)) * 0.90
                    actual = float(_payload_metric(metrics, key))
                    criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
            return criteria

        return criteria


def _metrics_from_dict(metrics: dict[str, float]) -> DownturnMetrics:
    from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics

    fields = DownturnMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            payload[key] = metrics[key]
        elif key == "correction_pnl_pct" and "correction_alpha_pct" in metrics:
            payload[key] = metrics["correction_alpha_pct"]
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return DownturnMetrics(**payload)


def _payload_metric(metrics: dict[str, float], key: str) -> float:
    if key in metrics:
        return float(metrics[key])
    if key == "correction_pnl_pct" and "correction_alpha_pct" in metrics:
        return float(metrics["correction_alpha_pct"])
    return 0.0


def _assess_engine_health(metrics: DownturnMetrics) -> dict[str, str]:
    health = {}
    for tag, trades, wr, avg_r in [
        ("reversal", metrics.reversal_trades, metrics.reversal_wr, metrics.reversal_avg_r),
        ("breakdown", metrics.breakdown_trades, metrics.breakdown_wr, metrics.breakdown_avg_r),
        ("fade", metrics.fade_trades, metrics.fade_wr, metrics.fade_avg_r),
    ]:
        if trades < 3:
            health[tag] = "insufficient_data"
        elif avg_r < -0.5:
            health[tag] = "harmful"
        elif wr < 0.35 and avg_r < 0:
            health[tag] = "underperforming"
        else:
            health[tag] = "healthy"
    return health


def _compute_correction_attribution(all_trades: list | None) -> dict[str, float]:
    if not all_trades:
        return {"correction_pnl": 0.0, "non_correction_pnl": 0.0, "ratio": 0.0}

    correction_pnl = sum(trade.pnl for trade in all_trades if trade.in_correction_window)
    non_correction_pnl = sum(trade.pnl for trade in all_trades if not trade.in_correction_window)
    total = correction_pnl + non_correction_pnl
    ratio = correction_pnl / total if total else 0.0
    return {
        "correction_pnl": correction_pnl,
        "non_correction_pnl": non_correction_pnl,
        "ratio": ratio,
    }


def _correction_weight_overrides(phase: int, current_weights: dict[str, float] | None) -> dict[str, float]:
    weights = dict(current_weights or PHASE_WEIGHTS.get(phase) or {})
    if not weights:
        return {}

    weights["correction_pnl"] = max(weights.get("correction_pnl", 0.0), 0.35)
    weights["regime_purity"] = max(weights.get("regime_purity", 0.0), 0.15)
    weights["edge"] = max(weights.get("edge", 0.0), 0.15)
    weights["risk"] = max(weights.get("risk", 0.0), 0.12)
    weights["capture"] = min(weights.get("capture", 0.0), 0.12)

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights
