from __future__ import annotations

import hashlib
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from backtests.diagnostic_snapshot import build_group_snapshot
from backtests.shared.auto.cache_keys import build_cache_key
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

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .scoring import PHASE_HARD_REJECTS, PHASE_WEIGHTS, score_phase_metrics

ULTIMATE_TARGETS = {
    "net_return_pct": 200.0,
    "profit_factor": 2.0,
    "max_drawdown_pct": 0.25,
    "total_trades": 80.0,
    "calmar": 3.0,
}


class _PoolBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        max_workers: int | None,
    ) -> None:
        from .worker import init_worker

        self._pool = mp.Pool(
            processes=resolve_worker_processes(max_workers),
            initializer=init_worker,
            initargs=(str(data_dir), initial_equity),
        )
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [
            (candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects)
            for candidate in candidates
        ]
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
    ) -> None:
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
            score_candidate((candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class AKCHelixPlugin:
    name = "helix"
    num_phases = 3
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 3,
    ) -> None:
        if not 1 <= num_phases <= 3:
            raise ValueError(f"AKCHelixPlugin supports between 1 and 3 phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._last_context: dict[str, Any] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        self._cache_source_fingerprint = self._data_fingerprint()

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
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
        evaluation_key = build_cache_key(
            "momentum.helix.evaluation",
            source_fingerprint=self._cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            return _PoolBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                scoring_weights,
                hard_rejects,
                self.max_workers,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                scoring_weights,
                hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"AKC Helix phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        context = self._run_config(mutations, store_context=True, collect_diagnostics=False)
        return dict(context["metrics"])

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        return "\n".join([
            "=" * 72,
            f"AKC HELIX PHASE {phase} SNAPSHOT",
            "=" * 72,
            f"Focus: {PHASE_FOCUS[phase][0]}",
            f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f}",
            f"Trades={int(metrics.get('total_trades', 0))} PF={metrics.get('profit_factor', 0.0):.2f} "
            f"Net={metrics.get('net_profit', 0.0):+.0f} DD={metrics.get('max_drawdown_pct', 0.0):.1%} "
            f"Calmar={metrics.get('calmar', 0.0):.2f}",
        ])

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del phase, state, metrics
        context = self._run_config(greedy_result.final_mutations, store_context=True, collect_diagnostics=True)
        return self._build_diagnostics_text(context, title="AKC HELIX FULL DIAGNOSTICS")

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final_phase = max(state.completed_phases) if state.completed_phases else self.num_phases
        context = self._run_config(state.cumulative_mutations, store_context=True, collect_diagnostics=True)
        metrics = context["metrics"]
        greedy_result_from_state(state, phase=final_phase, final_metrics=metrics)
        diagnostics_text = self._build_diagnostics_text(context, title="AKC HELIX ROUND DIAGNOSTICS")
        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics_text,
            dimension_reports={
                "signal_extraction": (
                    f"Trades={int(metrics.get('total_trades', 0))}, setups={context['result'].setups_detected}, "
                    f"entries_filled={context['result'].entries_filled}."
                ),
                "signal_discrimination": (
                    f"Profit factor {metrics.get('profit_factor', 0.0):.2f}, "
                    f"win rate {metrics.get('win_rate', 0.0):.1%}, "
                    f"score {context['score'].total:.4f}."
                ),
                "entry_mechanism": (
                    f"Entries placed {context['result'].entries_placed}, "
                    f"gates blocked {context['result'].gates_blocked}."
                ),
                "trade_management": (
                    f"Net profit ${metrics.get('net_profit', 0.0):,.0f}, "
                    f"calmar {metrics.get('calmar', 0.0):.2f}, "
                    f"drawdown {metrics.get('max_drawdown_pct', 0.0):.1%}."
                ),
                "exit_mechanism": (
                    f"Avg hold {metrics.get('avg_hold_hours', 0.0):.1f}h, "
                    f"expectancy ${metrics.get('expectancy_dollar', 0.0):,.0f}."
                ),
            },
            overall_verdict=(
                f"AKC Helix finished round scoring at {context['score'].total:.4f} with "
                f"{int(metrics.get('total_trades', 0))} trades, PF {metrics.get('profit_factor', 0.0):.2f}, "
                f"net profit ${metrics.get('net_profit', 0.0):,.0f}, and DD {metrics.get('max_drawdown_pct', 0.0):.1%}."
            ),
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        gaps: list[str] = []
        if metrics.get("total_trades", 0) < 30:
            gaps.append("Trade frequency still is below the preferred Helix activity floor.")
        if metrics.get("profit_factor", 0.0) < 1.5:
            gaps.append("Profit factor still is not strong enough for a mature Helix baseline.")
        if metrics.get("max_drawdown_pct", 0.0) > 0.25:
            gaps.append("Drawdown still is too high relative to the current round target.")
        if phase >= 2 and metrics.get("calmar", 0.0) < 1.5:
            gaps.append("Risk-adjusted return still is lagging the later-phase objective.")
        return gaps

    def _run_config(
        self,
        mutations: dict[str, Any],
        *,
        store_context: bool,
        collect_diagnostics: bool,
    ) -> dict[str, Any]:
        from backtests.momentum.analysis.helix_diagnostics import helix_full_diagnostic
        from backtests.momentum.auto.config_mutator import mutate_helix_config
        from backtests.momentum.auto.scoring import extract_metrics
        from backtests.momentum.cli import _load_helix_data_cached
        from backtests.momentum.config_helix import Helix4BacktestConfig
        from backtests.momentum.engine.helix_engine import Helix4Engine

        cache_key = build_cache_key(
            "momentum.helix.final_metrics",
            source_fingerprint=self._cache_source_fingerprint,
            mutations=mutations,
            extra={
                "initial_equity": self.initial_equity,
                "collect_diagnostics": collect_diagnostics,
            },
        )
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            if store_context:
                self._last_context = cached
            return cached

        data = _load_helix_data_cached("NQ", self.data_dir)
        config = mutate_helix_config(
            Helix4BacktestConfig(
                initial_equity=self.initial_equity,
                fixed_qty=10,
                point_value=2.0,
                data_dir=self.data_dir,
                track_signals=collect_diagnostics,
                track_shadows=collect_diagnostics,
            ),
            mutations,
        )
        result = Helix4Engine(symbol="NQ", bt_config=config).run(
            data["minute_bars"],
            data["hourly"],
            data["four_hour"],
            data["daily"],
            data["hourly_idx_map"],
            data["four_hour_idx_map"],
            data["daily_idx_map"],
        )
        metrics = extract_metrics(
            result.trades,
            result.equity_curve,
            _timestamps_to_numeric(result.timestamps),
            self.initial_equity,
        )
        score = score_phase_metrics(
            self.num_phases,
            metrics,
            self.initial_equity,
            equity_curve=result.equity_curve,
        )
        metrics_dict = asdict(metrics)
        context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics_dict,
            "score": score,
            "snapshot": build_group_snapshot(
                "Momentum Helix Strength / Weakness Snapshot",
                result.trades,
                [
                    ("setup class", lambda trade: getattr(trade, "setup_class", None)),
                    ("session", lambda trade: getattr(trade, "session_at_entry", None)),
                    ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
                ],
                min_count=5,
            ),
            "full_diagnostic": helix_full_diagnostic(
                result.trades,
                setup_log=getattr(result, "setup_log", None),
                gate_log=getattr(result, "gate_log", None),
                entry_tracking=getattr(result, "entry_tracking", None),
                equity_curve=result.equity_curve,
                timestamps=result.timestamps,
            ) if collect_diagnostics else "",
        }
        self._context_cache[cache_key] = context
        self._metrics_cache[mutation_signature(mutations)] = dict(metrics_dict)
        if store_context:
            self._last_context = context
        return context

    def _build_diagnostics_text(self, context: dict[str, Any], *, title: str) -> str:
        metrics = context["metrics"]
        score = context["score"]
        result = context["result"]
        header = "\n".join([
            "=" * 72,
            f"  {title}",
            "=" * 72,
            f"  Mutation count: {len(context['mutations'])}",
            f"  Composite score: {score.total:.4f}",
            "",
            "  PERFORMANCE SUMMARY",
            "  " + "-" * 55,
            f"    Total trades:        {int(metrics.get('total_trades', 0))}",
            f"    Win rate:            {metrics.get('win_rate', 0.0):.1%}",
            f"    Profit factor:       {metrics.get('profit_factor', 0.0):.2f}",
            f"    Net profit:          ${metrics.get('net_profit', 0.0):,.0f}",
            f"    Max drawdown:        {metrics.get('max_drawdown_pct', 0.0):.1%}",
            f"    Sharpe ratio:        {metrics.get('sharpe', 0.0):.2f}",
            f"    Sortino ratio:       {metrics.get('sortino', 0.0):.2f}",
            f"    Calmar ratio:        {metrics.get('calmar', 0.0):.2f}",
            f"    Expectancy ($/trade): ${metrics.get('expectancy_dollar', 0.0):,.0f}",
            f"    Avg hold (hours):    {metrics.get('avg_hold_hours', 0.0):.1f}",
            "",
            "  FUNNEL",
            "  " + "-" * 55,
            f"    Setups detected:     {getattr(result, 'setups_detected', 0)}",
            f"    Gates blocked:       {getattr(result, 'gates_blocked', 0)}",
            f"    Entries placed:      {getattr(result, 'entries_placed', 0)}",
            f"    Entries filled:      {getattr(result, 'entries_filled', 0)}",
        ])
        sections = [header, context["snapshot"]]
        if context.get("full_diagnostic"):
            sections.append(context["full_diagnostic"])
        shadow_summary = getattr(result, "shadow_summary", "")
        if shadow_summary:
            sections.append(shadow_summary)
        return "\n\n".join(sections)

    def _data_fingerprint(self) -> str:
        parts: list[str] = []
        for suffix in ("NQ_5m.parquet", "NQ_1d.parquet"):
            path = self.data_dir / suffix
            if path.exists():
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                parts.append(f"{path.name}:missing")
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        criteria = [
            GateCriterion("hard_min_trades", float(PHASE_HARD_REJECTS[phase]["min_trades"]), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(PHASE_HARD_REJECTS[phase]["min_trades"])),
            GateCriterion("hard_profit_factor", float(PHASE_HARD_REJECTS[phase]["min_pf"]), float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= float(PHASE_HARD_REJECTS[phase]["min_pf"])),
            GateCriterion("hard_max_drawdown_pct", float(PHASE_HARD_REJECTS[phase]["max_dd_pct"]), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(PHASE_HARD_REJECTS[phase]["max_dd_pct"])),
        ]
        if phase == 1:
            criteria.append(GateCriterion("profit_factor", 1.20, float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= 1.20))
            return criteria

        prior = state.get_phase_metrics(phase - 1) or {}
        if prior:
            for name in ("net_profit", "profit_factor", "calmar"):
                target = float(prior.get(name, 0.0)) * 0.95
                actual = float(metrics.get(name, 0.0))
                criteria.append(GateCriterion(f"no_regress_{name}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion("phase_progress", 0.0, 1.0, True))
        return criteria


def _timestamps_to_numeric(timestamps) -> np.ndarray:
    if timestamps is None or len(timestamps) == 0:
        return np.array([])
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        return np.array([dt.timestamp() for dt in timestamps], dtype=float)
    return np.asarray(timestamps)
