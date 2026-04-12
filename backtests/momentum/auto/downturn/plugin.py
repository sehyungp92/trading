from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics
    from .scoring import DownturnCompositeScore

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    deserialize_experiments,
    greedy_result_from_state,
    greedy_result_to_dict,
    resolve_worker_processes,
    seen_experiment_names,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

from .phase_candidates import get_phase_candidates

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: {
        "coverage": 0.40,
        "capture": 0.10,
        "net_profit": 0.10,
        "edge": 0.10,
        "risk": 0.20,
        "hold": 0.10,
    },
    2: {
        "coverage": 0.15,
        "capture": 0.35,
        "net_profit": 0.15,
        "edge": 0.10,
        "risk": 0.05,
        "hold": 0.20,
    },
    3: {
        "coverage": 0.20,
        "capture": 0.20,
        "net_profit": 0.15,
        "edge": 0.15,
        "risk": 0.20,
        "hold": 0.10,
    },
    4: {
        "coverage": 0.25,
        "capture": 0.20,
        "net_profit": 0.15,
        "edge": 0.15,
        "risk": 0.15,
        "hold": 0.10,
    },
    5: {
        "coverage": 0.10,
        "capture": 0.35,
        "net_profit": 0.20,
        "edge": 0.10,
        "risk": 0.15,
        "hold": 0.10,
    },
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"max_dd_pct": 0.45, "min_trades": 5},
    2: {"max_dd_pct": 0.40, "min_trades": 8},
    3: {"max_dd_pct": 0.35, "min_trades": 10, "min_pf": 0.8},
    4: {"max_dd_pct": 0.30, "min_trades": 10, "min_pf": 1.0, "min_sharpe": 0.2},
    5: {"max_dd_pct": 0.40, "min_trades": 10, "min_pf": 1.0, "min_sharpe": 0.2},
}

PHASE_FOCUS = {
    1: ("Signal Detection", ["signal_to_entry_ratio", "correction_alpha_pct", "total_trades"]),
    2: ("Capture", ["exit_efficiency", "profit_factor", "correction_alpha_pct"]),
    3: ("Risk Control", ["calmar", "max_dd_pct", "sharpe"]),
    4: ("Fine-tuning", ["calmar", "net_return_pct", "correction_alpha_pct"]),
    5: ("Exit Management", ["exit_efficiency", "net_return_pct", "profit_factor"]),
}

ULTIMATE_TARGETS = {
    "correction_alpha_pct": 25.0,
    "profit_factor": 2.0,
    "net_return_pct": 40.0,
    "max_dd_pct": 15.0,
    "calmar": 2.0,
    "sharpe": 0.8,
    "exit_efficiency": 0.40,
    "signal_to_entry_ratio": 0.25,
    "total_trades": 120.0,
}


def score_phase_metrics(
    phase: int,
    metrics: DownturnMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> DownturnCompositeScore:
    from .scoring import DownturnCompositeScore, composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    if metrics.total_trades < rejects.get("min_trades", 8):
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades})")
    if metrics.max_dd_pct > rejects.get("max_dd_pct", 0.35):
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_dd_pct:.2%})")
    if "min_pf" in rejects and metrics.profit_factor < rejects["min_pf"]:
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f})")
    if "min_sharpe" in rejects and metrics.sharpe < rejects["min_sharpe"]:
        return DownturnCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_sharpe ({metrics.sharpe:.2f})")

    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {key: value / total for key, value in base.items()} if total > 0 else base
    return composite_score(metrics, weights)


class _PoolBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        max_workers: int | None,
    ):
        from .worker import init_worker

        processes = resolve_worker_processes(max_workers)
        self._pool = mp.Pool(
            processes=processes,
            initializer=init_worker,
            initargs=(str(data_dir), initial_equity, phase, scoring_weights, hard_rejects),
        )

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [(candidate.name, candidate.mutations, current_mutations) for candidate in candidates]
        return self._pool.map(score_candidate, args)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        self._pool.terminate()
        self._pool.join()


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
        init_worker(str(self._data_dir), self._initial_equity, self._phase, self._scoring_weights, self._hard_rejects)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate
        return [score_candidate((c.name, c.mutations, current_mutations)) for c in candidates]

    def close(self) -> None:
        pass


class DownturnPlugin:
    name = "downturn"
    num_phases = 5
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 100_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 5,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(
                f"DownturnPlugin supports between 1 and {max(PHASE_FOCUS)} phases, got {num_phases}."
            )
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._cached_data: dict[str, Any] | None = None
        self._last_context: dict[str, Any] = {}

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
            return _PoolBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects, self.max_workers,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"downturn phase {phase}")
        return CachedBatchEvaluator(raw)

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        from backtests.momentum._aliases import install

        install()

        from backtest.config_downturn import DownturnBacktestConfig
        from backtest.engine.downturn_engine import DownturnEngine
        from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
        from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
        from backtests.momentum.auto.downturn.worker import load_worker_data

        if self._cached_data is None:
            self._cached_data = load_worker_data("NQ", self.data_dir)

        config = mutate_downturn_config(
            DownturnBacktestConfig(initial_equity=self.initial_equity, data_dir=self.data_dir),
            mutations,
        )
        engine = DownturnEngine("NQ", config)
        result = engine.run(**self._cached_data)
        metrics = compute_downturn_metrics(result, self._cached_data["daily"])
        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "trades": result.trades,
        }
        return asdict(metrics)

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
            f"Correction alpha is {metrics_obj.correction_alpha_pct:.1f}% with coverage {metrics_obj.correction_coverage:.1%}. "
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
            f"Max drawdown is {metrics_obj.max_dd_pct:.1%}, calmar is {metrics_obj.calmar:.2f}, sharpe is {metrics_obj.sharpe:.2f}."
        )
        exits = (
            f"Exit efficiency is {metrics_obj.exit_efficiency:.2f}; average MFE capture is {metrics_obj.avg_mfe_capture:.2f}. "
            f"Median hold is {metrics_obj.median_hold_5m:.1f} bars."
        )
        overall_verdict = (
            f"Signal extraction {'is' if metrics_obj.correction_alpha_pct >= 25.0 else 'is not yet'} capturing correction alpha at the target level "
            f"({metrics_obj.correction_alpha_pct:.1f}%). "
            f"Discrimination quality is anchored by signal-to-entry {metrics_obj.signal_to_entry_ratio:.2f} and engine health "
            f"{', '.join(f'{engine}={status}' for engine, status in extra['engine_health'].items())}. "
            f"Final trade-management and exit quality should be judged from the full diagnostics with DD {metrics_obj.max_dd_pct:.1%}, "
            f"calmar {metrics_obj.calmar:.2f}, and exit efficiency {metrics_obj.exit_efficiency:.2f}."
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

        if "low trade frequency" in weakness_text or metrics_obj.total_trades < 50:
            add("relax_dead_zones", {"flags.use_dead_zones": False})
            add("relax_entry_windows", {"flags.use_entry_windows": False})
            add("relax_friction_gate", {"flags.friction_gate": False})
            add("wider_regime_neutral", {"param_overrides.regime_mult_neutral": 0.80})

        if metrics_obj.correction_alpha_pct < 5.0:
            add("regime_faster_ema_10", {"param_overrides.ema_fast_period": 10})
            add("regime_adx_trending_20", {"param_overrides.adx_trending_threshold": 20})
            add("regime_sma200_150", {"param_overrides.sma200_period": 150})

        if "tp2/tp3 never hit" in weakness_text:
            add("tp2_lower_2.0R", {"param_overrides.tp2_r_aligned": 2.0})
            add("tp3_lower_3.5R", {"param_overrides.tp3_r_aligned": 3.5})
            add("tp1_higher_2.0R", {"param_overrides.tp1_r_aligned": 2.0})
            add("chandelier_wider_20", {"param_overrides.chandelier_lookback": 20})

        if "low tp1 hit rate" in weakness_text:
            add("wider_stop_mult", {"param_overrides.climax_mult": 3.0})
            add("longer_stale_fade_36", {"param_overrides.stale_bars_fade": 36})

        if metrics_obj.exit_efficiency < 0.15:
            add("faster_be_move", {"param_overrides.tp1_r_aligned": 1.0})
            add("tighter_chandelier_8", {"param_overrides.chandelier_lookback": 8})

        if metrics_obj.max_dd_pct > 0.25:
            add("lower_risk_pct_0.008", {"param_overrides.base_risk_pct": 0.008})
            add("reduce_counter_mult_0.20", {"param_overrides.regime_mult_counter": 0.20})
            add("circuit_breaker_tighter", {"param_overrides.daily_circuit_breaker": 0.02})

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
            if name in {"signal_to_entry", "signal_to_entry_ratio", "correction_alpha_pct", "total_trades"}:
                weights["coverage"] *= 1.20
            if name in {"exit_efficiency"}:
                weights["capture"] *= 1.25
            if name in {"profit_factor"}:
                weights["edge"] *= 1.25
            if name in {"net_return_pct"}:
                weights["net_profit"] *= 1.20
            if name in {"max_dd_pct", "calmar", "sharpe"}:
                weights["risk"] *= 1.25

        weakness_text = " ".join(analysis.weaknesses).lower()
        if "tp2/tp3" in weakness_text or "exit efficiency" in weakness_text:
            weights["capture"] *= 1.10
            weights["hold"] *= 1.10
        if "trade frequency" in weakness_text:
            weights["coverage"] *= 1.10
        if "drawdown" in weakness_text:
            weights["risk"] *= 1.10

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
                    "reweight scoring toward correction capture and signal discrimination."
                ),
                scoring_assessment_override="MISALIGNED",
                scoring_weight_overrides=_correction_weight_overrides(phase, current_weights),
            )
        return None

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        metric_obj = _metrics_from_dict(metrics)
        criteria = [
            GateCriterion("hard_min_trades", 8.0, float(metric_obj.total_trades), metric_obj.total_trades >= 8),
            GateCriterion("hard_max_dd_pct", 0.30, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.30),
            GateCriterion("hard_correction_alpha_pct", 0.0, metric_obj.correction_alpha_pct, metric_obj.correction_alpha_pct >= 0.0),
        ]

        if phase == 1:
            criteria.extend(
                [
                    GateCriterion("signal_to_entry_ratio", 0.15, metric_obj.signal_to_entry_ratio, metric_obj.signal_to_entry_ratio >= 0.15),
                    GateCriterion("total_trades", 15.0, float(metric_obj.total_trades), metric_obj.total_trades >= 15),
                    GateCriterion("correction_alpha_pct", 5.0, metric_obj.correction_alpha_pct, metric_obj.correction_alpha_pct >= 5.0),
                ]
            )
            return criteria

        if phase == 2:
            criteria.extend(
                [
                    GateCriterion("exit_efficiency", 0.20, metric_obj.exit_efficiency, metric_obj.exit_efficiency >= 0.20),
                    GateCriterion("profit_factor", 1.3, metric_obj.profit_factor, metric_obj.profit_factor >= 1.3),
                    GateCriterion("correction_alpha_pct", 10.0, metric_obj.correction_alpha_pct, metric_obj.correction_alpha_pct >= 10.0),
                ]
            )
            return criteria

        if phase == 3:
            criteria.extend(
                [
                    GateCriterion("calmar", 1.0, metric_obj.calmar, metric_obj.calmar >= 1.0),
                    GateCriterion("max_dd_pct", 0.22, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.22),
                    GateCriterion("sharpe", 0.6, metric_obj.sharpe, metric_obj.sharpe >= 0.6),
                ]
            )
            return criteria

        prior_metrics = state.get_phase_metrics(phase - 1) or {}
        if prior_metrics:
            for key in ["calmar", "profit_factor", "sharpe", "correction_alpha_pct"]:
                target = float(prior_metrics.get(key, 0.0)) * 0.90
                actual = float(metrics.get(key, 0.0))
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion(f"phase{phase}_pass", 0.0, 1.0, True))
        return criteria


def _metrics_from_dict(metrics: dict[str, float]) -> DownturnMetrics:
    from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics

    fields = DownturnMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            payload[key] = metrics[key]
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return DownturnMetrics(**payload)


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

    weights["coverage"] = max(weights.get("coverage", 0.0), 0.35)
    weights["edge"] = max(weights.get("edge", 0.0), 0.18)
    weights["risk"] = max(weights.get("risk", 0.0), 0.15)
    weights["capture"] = min(weights.get("capture", 0.0), 0.20)
    weights["hold"] = min(weights.get("hold", 0.0), 0.10)

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights
