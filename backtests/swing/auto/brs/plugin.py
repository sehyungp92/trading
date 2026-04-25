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
    resolve_worker_processes,
    seen_experiment_names,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

from .phase_candidates import get_phase_candidates
from .scoring import BRSMetrics, BRSCompositeScore, composite_score

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,
    2: {
        "net_profit": 0.20,
        "pf": 0.13,
        "calmar": 0.10,
        "inv_dd": 0.07,
        "bear_alpha": 0.18,
        "frequency": 0.20,
        "detection_quality": 0.12,
    },
    3: {
        "net_profit": 0.25,
        "pf": 0.14,
        "calmar": 0.12,
        "inv_dd": 0.08,
        "bear_alpha": 0.18,
        "frequency": 0.15,
        "detection_quality": 0.08,
    },
    4: {
        "net_profit": 0.18,
        "pf": 0.14,
        "calmar": 0.14,
        "inv_dd": 0.14,
        "bear_alpha": 0.14,
        "frequency": 0.18,
        "detection_quality": 0.08,
    },
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"max_dd_pct": 0.30, "min_trades": 10},
    2: {"max_dd_pct": 0.30, "min_trades": 10},
    3: {"max_dd_pct": 0.25, "min_trades": 10},
    4: {"max_dd_pct": 0.20, "min_trades": 15},
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("REGIME + ENTRY", ["profit_factor", "bear_alpha_pct", "total_trades"]),
    2: ("SIGNAL_SELECT", ["profit_factor", "net_return_pct", "bear_alpha_pct"]),
    3: ("EXIT + VOLATILITY", ["profit_factor", "net_return_pct", "max_dd_pct"]),
    4: ("SIZING + FINETUNE", ["calmar", "max_dd_pct", "net_return_pct"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 100.0,
    "profit_factor": 3.0,
    "max_dd_pct": 0.05,
    "calmar": 10.0,
    "bear_alpha_pct": 20.0,
    "total_trades": 80.0,
}

_GATE_TO_SCORING = {
    "net_return_pct": "net_profit",
    "profit_factor": "pf",
    "bear_pf": "pf",
    "bear_trade_wr": "pf",
    "total_trades": "frequency",
    "calmar": "calmar",
    "sharpe": "calmar",
    "max_dd_pct": "inv_dd",
    "bear_alpha_pct": "bear_alpha",
    "exit_efficiency": "pf",
    "bear_capture_ratio": "bear_alpha",
    "regime_f1": "detection_quality",
}


def score_phase_metrics(
    phase: int,
    metrics: BRSMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> BRSCompositeScore:
    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    min_trades = int(rejects.get("min_trades", 10))
    max_dd = float(rejects.get("max_dd_pct", 0.30))

    if metrics.total_trades < min_trades:
        return BRSCompositeScore(rejected=True, reject_reason=f"Phase {phase}: too few trades {metrics.total_trades} < {min_trades}")
    if metrics.max_dd_pct > max_dd:
        return BRSCompositeScore(rejected=True, reject_reason=f"Phase {phase}: DD {metrics.max_dd_pct:.1%} > {max_dd:.0%}")

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


class BRSPlugin:
    name = "brs"
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
            raise ValueError(f"BRSPlugin supports between 1 and 4 phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._last_context: dict[str, Any] = {}

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
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
                min_effective_score_delta_pct=0.01,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
                decide_action_fn=self.decide_phase_action,
            ),
            max_rounds=20,
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

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"BRS phase {phase}")
        return CachedBatchEvaluator(raw)

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        from backtests.swing._aliases import install

        install()

        from backtest.config_brs import BRSConfig
        from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_synchronized
        from backtests.swing.analysis.brs_diagnostics import compute_brs_diagnostics
        from backtests.swing.auto.brs.config_mutator import mutate_brs_config
        from backtests.swing.auto.brs.scoring import extract_brs_metrics

        config = mutate_brs_config(
            BRSConfig(initial_equity=self.initial_equity, data_dir=self.data_dir),
            mutations,
        )
        data = load_brs_data(config)
        result = run_brs_synchronized(data, config)
        metrics = extract_brs_metrics(result, self.initial_equity)
        diagnostics = compute_brs_diagnostics(
            result.symbol_results,
            self.initial_equity,
            combined_equity=result.combined_equity,
            combined_timestamps=result.combined_timestamps,
        )

        all_trades = []
        crisis_state_logs = []
        for symbol_result in result.symbol_results.values():
            all_trades.extend(symbol_result.trades)
            crisis_state_logs.extend(getattr(symbol_result, "crisis_state_log", []))

        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "data": data,
            "result": result,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "all_trades": all_trades,
            "crisis_state_logs": crisis_state_logs,
        }
        return asdict(metrics)

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
            crisis_state_logs=self._last_context.get("crisis_state_logs"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
            force_all_modules=True,
            crisis_state_logs=self._last_context.get("crisis_state_logs"),
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        metrics_obj = _metrics_from_dict(metrics)
        diagnostics = self._last_context.get("diagnostics")
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        regime_lines = []
        if diagnostics and diagnostics.regime_metrics:
            top_regimes = ", ".join(
                f"{entry.regime} PF={entry.profit_factor:.2f} ({entry.trade_count} trades)"
                for entry in diagnostics.regime_metrics[:3]
            )
            regime_lines.append(f"Top regime buckets: {top_regimes}.")
        regime_lines.append(
            f"Bear alpha is {metrics_obj.bear_alpha_pct:.1f}% with crisis coverage {metrics_obj.crisis_coverage:.1%}."
        )

        discrimination_lines = [
            f"Profit factor is {metrics_obj.profit_factor:.2f}; bear PF is {metrics_obj.bear_pf:.2f}.",
            f"Regime F1 proxy is {metrics_obj.regime_f1:.2f} and bear trade win rate is {metrics_obj.bear_trade_wr:.1f}%.",
        ]

        entry_lines = []
        if diagnostics and diagnostics.entry_type_metrics:
            top_entries = ", ".join(
                f"{entry.entry_type} PF={entry.profit_factor:.2f}"
                for entry in diagnostics.entry_type_metrics[:3]
            )
            entry_lines.append(f"Entry-type leaders: {top_entries}.")
        entry_lines.append(f"Total trades reached {metrics_obj.total_trades} with detection latency {metrics_obj.detection_latency_days:.1f} days.")

        management_lines = [
            f"Max drawdown is {metrics_obj.max_dd_pct:.1%}, calmar is {metrics_obj.calmar:.2f}, sharpe is {metrics_obj.sharpe:.2f}.",
            f"Bear capture ratio is {metrics_obj.bear_capture_ratio:.2f} with frequency support from {metrics_obj.total_trades} trades.",
        ]

        exit_lines = [
            f"Exit efficiency is {metrics_obj.exit_efficiency:.2f}.",
            f"Bias latency is {metrics_obj.bias_latency_days:.1f} days, which frames how quickly exits adapt to downturn conditions.",
        ]

        overall_verdict = (
            f"Alpha capture {'is' if metrics_obj.bear_alpha_pct >= 20 else 'is not yet'} strong enough "
            f"({metrics_obj.bear_alpha_pct:.1f}% bear alpha). "
            f"Signal discrimination {'is' if metrics_obj.bear_pf >= 2.0 and metrics_obj.regime_f1 >= 0.65 else 'still is not'} "
            f"consistently filtering low-quality downturn signals. "
            f"Entry, management, and exit quality should be judged against the final diagnostics report, with "
            f"PF {metrics_obj.profit_factor:.2f}, calmar {metrics_obj.calmar:.2f}, and exit efficiency {metrics_obj.exit_efficiency:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports={
                "signal_extraction": " ".join(regime_lines),
                "signal_discrimination": " ".join(discrimination_lines),
                "entry_mechanism": " ".join(entry_lines),
                "trade_management": " ".join(management_lines),
                "exit_mechanism": " ".join(exit_lines),
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
        tested = seen_experiment_names(state)

        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name not in tested:
                tested.add(name)
                suggestions.append(Experiment(name=name, mutations=mutations))

        if metrics_obj.regime_f1 < 0.65:
            if phase <= 2:
                add("sug_adx_lower", {"adx_strong": 25})
                add("sug_adx_higher", {"adx_strong": 32})
                add("sug_ema_slow_longer", {"ema_slow_period": 55})
                add("sug_bear_min_2", {"regime_bear_min_conditions": 2})

        if metrics_obj.bear_trade_wr < 50:
            if phase <= 2:
                add("sug_lh_lookback_3", {"lh_swing_lookback": 3})
                add("sug_lh_lookback_7", {"lh_swing_lookback": 7})
                add("sug_v_rev_tight", {"v_reversal_max_rally": 1.0})
                add("sug_v_rev_loose", {"v_reversal_max_rally": 2.0})
                add("sug_bt_vol_tight", {"bt_volume_mult": 1.5})
            else:
                add("sug_vf_clamp_tight", {"param_overrides.vf_clamp_max": 1.2})

        if metrics_obj.max_dd_pct > 0.20:
            if phase <= 2:
                add("sug_tight_cat", {"catastrophic_cap_r": 1.75})
                add("sug_tight_cat_v2", {"catastrophic_cap_r": 1.50})
                add("sug_lh_stop_tight", {"lh_max_stop_atr": 2.0})
            else:
                add("sug_tight_heat", {"heat_cap_r": 1.25})
                add("sug_crisis_heat", {"crisis_heat_cap_r": 0.75})

        if metrics_obj.exit_efficiency < 0.35:
            if phase <= 2:
                add("sug_fast_be", {"be_trigger_r": 1.0})
                add("sug_tight_trail", {"trail_trigger_r": 1.5})
                add("sug_stale_early_tight", {"stale_early_bars": 25})
            else:
                add("sug_stale_short_40", {"stale_bars_short": 40})
                add("sug_time_decay_fast", {"time_decay_hours": 240})

        if metrics_obj.bear_pf < 1.8:
            if phase <= 2:
                add("sug_adx_strong_up", {"adx_strong": 30})
                add("sug_lh_arm_wide", {"lh_arm_bars": 60})
                add("sug_bd_quality_tight", {"bd_close_quality": 0.20})
            else:
                add("sug_stale_bars_40", {"stale_bars_short": 40})

        if metrics_obj.sharpe < 0.6:
            if phase <= 2:
                add("sug_vf_floor_up", {"param_overrides.vf_clamp_min": 0.45})
                add("sug_tod_narrow", {"tod_filter_start": 870, "tod_filter_end": 900})
            else:
                add("sug_max_concurrent_2", {"max_concurrent": 2})

        if metrics_obj.calmar < 1.0 and phase >= 3:
            add("sug_crisis_cap_tight", {"crisis_heat_cap_r": 0.60})
            add("sug_risk_low_qqq", {"symbol_configs.QQQ.base_risk_pct": 0.003})

        if metrics_obj.bear_alpha_pct < 20:
            if phase <= 2:
                add("sug_quality_gate_50", {"min_quality_score": 0.50})
                add("sug_fast_crash", {"fast_crash_enabled": True})
            else:
                add("sug_heat_cap_wide", {"heat_cap_r": 2.0})

        if metrics_obj.bear_capture_ratio < 0.40:
            if phase <= 2:
                add("sug_lh_arm_short", {"lh_arm_bars": 20})
                add("sug_bd_arm_wide", {"bd_arm_bars": 24})
                add("sug_tod_off", {"tod_filter_enabled": False})
            else:
                add("sug_concurrent_4", {"max_concurrent": 4})

        if metrics_obj.total_trades < 25 and phase <= 2:
            add("sug_lh_arm_wider", {"lh_arm_bars": 60})
            add("sug_lh_stop_wider", {"lh_max_stop_atr": 3.0})
            add("sug_v_rev_wider", {"v_reversal_max_rally": 2.5})
            add("sug_tod_wide", {"tod_filter_start": 810, "tod_filter_end": 960})
            add("sug_reenable_s1", {"disable_s1": False})

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
        return {key: value / total for key, value in base_weights.items()} if total > 0 else base_weights

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        """Crisis-window PnL attribution (analogous to downturn's correction attribution)."""
        all_trades = self._last_context.get("all_trades", [])
        crisis_logs = self._last_context.get("crisis_state_logs", [])
        crisis_pnl = sum(getattr(t, "pnl", 0.0) for t in all_trades if getattr(t, "in_crisis_window", False))
        non_crisis_pnl = sum(getattr(t, "pnl", 0.0) for t in all_trades if not getattr(t, "in_crisis_window", False))
        total_pnl = crisis_pnl + non_crisis_pnl
        return {
            "crisis_attribution": {
                "crisis_pnl": crisis_pnl,
                "non_crisis_pnl": non_crisis_pnl,
                "ratio": crisis_pnl / total_pnl if total_pnl else 0.0,
            },
            "crisis_state_log_count": len(crisis_logs),
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines = []
        crisis = extra.get("crisis_attribution", {})
        if crisis:
            lines.append(
                f"Crisis attribution: crisis_pnl={crisis.get('crisis_pnl', 0):.0f}, "
                f"non_crisis_pnl={crisis.get('non_crisis_pnl', 0):.0f}, "
                f"ratio={crisis.get('ratio', 0):.2f}"
            )
        log_count = extra.get("crisis_state_log_count", 0)
        if log_count:
            lines.append(f"Crisis state log entries: {log_count}")
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
        crisis = analysis.extra.get("crisis_attribution", {})
        if (
            crisis.get("crisis_pnl", 0.0) < 0.0
            and crisis.get("non_crisis_pnl", 0.0) > 0.0
            and scoring_retries < max_scoring_retries
        ):
            weights = dict(current_weights or PHASE_WEIGHTS.get(phase) or {})
            if weights:
                weights["bear_alpha"] = max(weights.get("bear_alpha", 0.0), 0.25)
                weights["detection_quality"] = max(weights.get("detection_quality", 0.0), 0.15)
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()} if total > 0 else weights
            return PhaseDecision(
                action="improve_scoring",
                reason=(
                    "Strategy is profitable outside crisis windows but loses during crises; "
                    "reweight scoring toward bear_alpha and detection_quality."
                ),
                scoring_assessment_override="MISALIGNED",
                scoring_weight_overrides=weights or None,
            )
        return None

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        metric_obj = _metrics_from_dict(metrics)
        criteria: list[GateCriterion] = []

        if phase in (1, 2):
            criteria.extend(
                [
                    GateCriterion("hard_min_trades", 15.0, float(metric_obj.total_trades), metric_obj.total_trades >= 15),
                    GateCriterion("hard_max_dd_pct", 0.30, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.30),
                    GateCriterion("profit_factor", 1.3, metric_obj.profit_factor, metric_obj.profit_factor >= 1.3),
                    GateCriterion("total_trades", 20.0, float(metric_obj.total_trades), metric_obj.total_trades >= 20),
                    GateCriterion("net_return_pct", 5.0, metric_obj.net_return_pct, metric_obj.net_return_pct >= 5.0),
                ]
            )
            return criteria

        if phase == 3:
            criteria.extend(
                [
                    GateCriterion("hard_min_trades", 15.0, float(metric_obj.total_trades), metric_obj.total_trades >= 15),
                    GateCriterion("hard_max_dd_pct", 0.25, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.25),
                    GateCriterion("profit_factor", 1.5, metric_obj.profit_factor, metric_obj.profit_factor >= 1.5),
                    GateCriterion("net_return_pct", 10.0, metric_obj.net_return_pct, metric_obj.net_return_pct >= 10.0),
                    GateCriterion("max_dd_pct", 0.22, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.22),
                ]
            )
            return criteria

        prior_metrics = state.get_phase_metrics(3) or {}
        criteria.extend(
            [
                GateCriterion("hard_max_dd_pct", 0.20, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.20),
                GateCriterion("hard_sharpe", 0.4, metric_obj.sharpe, metric_obj.sharpe >= 0.4),
                GateCriterion("calmar", 1.0, metric_obj.calmar, metric_obj.calmar >= 1.0),
                GateCriterion("max_dd_pct", 0.18, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= 0.18),
                GateCriterion("net_return_pct", 15.0, metric_obj.net_return_pct, metric_obj.net_return_pct >= 15.0),
            ]
        )
        if prior_metrics:
            prior_sharpe = float(prior_metrics.get("sharpe", 0.0))
            prior_calmar = float(prior_metrics.get("calmar", 0.0))
            prior_dd = float(prior_metrics.get("max_dd_pct", 0.0))
            if prior_sharpe > 0:
                criteria.append(GateCriterion("no_regression_sharpe", prior_sharpe * 0.90, metric_obj.sharpe, metric_obj.sharpe >= prior_sharpe * 0.90))
            if prior_calmar > 0:
                criteria.append(GateCriterion("no_regression_calmar", prior_calmar * 0.90, metric_obj.calmar, metric_obj.calmar >= prior_calmar * 0.90))
            if prior_dd > 0:
                criteria.append(GateCriterion("no_regression_max_dd_pct", prior_dd * 1.10, metric_obj.max_dd_pct, metric_obj.max_dd_pct <= prior_dd * 1.10))
        return criteria


def _metrics_from_dict(metrics: dict[str, float]) -> BRSMetrics:
    fields = BRSMetrics.__dataclass_fields__
    return BRSMetrics(**{key: metrics.get(key, 0.0) for key in fields})
