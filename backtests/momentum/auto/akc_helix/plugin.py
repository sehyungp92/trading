from __future__ import annotations

import hashlib
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
    SharedPoolBatchEvaluator,
    create_process_pool,
    deserialize_experiments,
    greedy_result_from_state,
    mutation_signature,
    shutdown_process_pool,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .rebase_candidates import REBASE_BASE_MUTATIONS, REBASE_PHASE_FOCUS, get_rebase_candidates
from .scoring import (
    PHASE_HARD_REJECTS,
    PHASE_WEIGHTS,
    REBASE_CALMAR_TARGET,
    REBASE_EXPECTANCY_TARGET_USD,
    REBASE_HARD_REJECTS,
    REBASE_MAX_DD_CLIP,
    REBASE_NET_PROFIT_TARGET_USD,
    REBASE_PF_TARGET,
    REBASE_TRADE_TARGETS,
    REBASE_WEIGHTS,
    score_phase_metrics,
)

ULTIMATE_TARGETS = {
    "net_profit": 150_000.0,
    "expectancy_dollar": 1_650.0,
    "total_trades": 100.0,
    "profit_factor": 4.00,
    "max_drawdown_pct": 0.20,
}

PHASE_ABSOLUTE_FLOORS = {
    1: {"trades": 84.0, "net_profit": 140_000.0, "expectancy_dollar": 1_650.0, "profit_factor": 4.10, "max_drawdown_pct": 0.18},
    2: {"trades": 84.0, "net_profit": 140_000.0, "expectancy_dollar": 1_640.0, "profit_factor": 4.05, "max_drawdown_pct": 0.18},
    3: {"trades": 84.0, "net_profit": 140_000.0, "expectancy_dollar": 1_650.0, "profit_factor": 4.10, "max_drawdown_pct": 0.18},
    4: {"trades": 85.0, "net_profit": 139_000.0, "expectancy_dollar": 1_635.0, "profit_factor": 4.00, "max_drawdown_pct": 0.19},
    5: {"trades": 86.0, "net_profit": 140_000.0, "expectancy_dollar": 1_640.0, "profit_factor": 4.05, "max_drawdown_pct": 0.19},
}

PHASE_RETENTION_RULES = {
    1: {"trade_mult": 1.00, "net_mult": 0.995, "expectancy_mult": 0.985, "pf_mult": 0.985, "calmar_mult": 0.975, "dd_mult": 1.05},
    2: {"trade_mult": 1.00, "net_mult": 0.995, "expectancy_mult": 0.982, "pf_mult": 0.982, "calmar_mult": 0.970, "dd_mult": 1.06},
    3: {"trade_mult": 1.00, "net_mult": 0.995, "expectancy_mult": 0.985, "pf_mult": 0.985, "calmar_mult": 0.975, "dd_mult": 1.05},
    4: {"trade_mult": 1.00, "net_mult": 0.992, "expectancy_mult": 0.980, "pf_mult": 0.980, "calmar_mult": 0.970, "dd_mult": 1.06},
    5: {"trade_mult": 1.00, "net_mult": 0.995, "expectancy_mult": 0.985, "pf_mult": 0.985, "calmar_mult": 0.975, "dd_mult": 1.04},
}


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
    num_phases = max(PHASE_FOCUS)
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = max(PHASE_FOCUS),
    ) -> None:
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(
                f"AKCHelixPlugin supports between 1 and {max(PHASE_FOCUS)} phases, got {num_phases}."
            )
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
        reference_metrics = self._reference_metrics(state, phase)
        phase_hard_rejects = self._phase_hard_rejects(phase, reference_metrics)
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
            hard_rejects=phase_hard_rejects,
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.006 if phase == self.num_phases else 0.005,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
            ),
            max_rounds=10 if phase == self.num_phases else 20,
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
            from .worker import init_worker, score_candidate

            pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(str(self.data_dir), self.initial_equity),
            )
            return SharedPoolBatchEvaluator(
                pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (
                        candidate.name,
                        candidate.mutations,
                        current_mutations,
                        phase,
                        scoring_weights,
                        hard_rejects,
                    )
                    for candidate in candidates
                ],
                on_close=lambda pool=pool: shutdown_process_pool(pool),
                on_terminate=lambda pool=pool: shutdown_process_pool(pool, force=True),
                description=f"AKC Helix phase {phase}",
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
        del phase
        gaps: list[str] = []
        if metrics.get("total_trades", 0) < 100:
            gaps.append("Trade frequency still is below the 100-trade structural expansion target.")
        if metrics.get("net_profit", 0.0) < 136_000.0:
            gaps.append("Net profit still is below the level needed to justify broader structural expansion.")
        if metrics.get("expectancy_dollar", 0.0) < 1_550.0:
            gaps.append("Per-trade expectancy still is too weak for a quality-preserving structural expansion round.")
        if metrics.get("profit_factor", 0.0) < 3.85:
            gaps.append("Profit factor still is too weak for a quality-preserving structural expansion round.")
        if metrics.get("max_drawdown_pct", 0.0) > 0.20:
            gaps.append("Drawdown still is too high relative to the quality-preserving expansion target.")
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
            apply_hard_rejects=False,
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

    def _reference_metrics(self, state: PhaseState, phase: int) -> dict[str, float]:
        if phase > 1:
            prior = state.get_phase_metrics(phase - 1)
            if prior:
                return dict(prior)
        baseline_mutations = dict(state.cumulative_mutations or self.initial_mutations or {})
        return self._metrics_for_mutations(baseline_mutations)

    def _metrics_for_mutations(self, mutations: dict[str, Any]) -> dict[str, float]:
        signature = mutation_signature(mutations)
        cached = self._metrics_cache.get(signature)
        if cached is not None:
            return dict(cached)
        return self.compute_final_metrics(mutations)

    def _phase_hard_rejects(self, phase: int, reference_metrics: dict[str, float] | None) -> dict[str, float]:
        rejects = dict(PHASE_HARD_REJECTS.get(phase, {}))
        if reference_metrics:
            baseline_trades = int(float(reference_metrics.get("total_trades", 0.0)))
            rejects["min_trades"] = min(int(rejects.get("min_trades", 0)), baseline_trades)
        return rejects

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        reference_metrics = self._reference_metrics(state, phase)
        phase_rejects = self._phase_hard_rejects(phase, reference_metrics)
        criteria = [
            GateCriterion("hard_min_trades", float(phase_rejects["min_trades"]), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(phase_rejects["min_trades"])),
            GateCriterion("hard_profit_factor", float(phase_rejects["min_pf"]), float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= float(phase_rejects["min_pf"])),
            GateCriterion("hard_max_drawdown_pct", float(phase_rejects["max_dd_pct"]), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(phase_rejects["max_dd_pct"])),
        ]
        floors = dict(PHASE_ABSOLUTE_FLOORS[phase])
        if reference_metrics:
            floors["trades"] = min(float(floors["trades"]), float(reference_metrics.get("total_trades", 0.0)))
        criteria.append(GateCriterion("floor_total_trades", float(floors["trades"]), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(floors["trades"])))
        criteria.append(GateCriterion("floor_net_profit", float(floors["net_profit"]), float(metrics.get("net_profit", 0.0)), float(metrics.get("net_profit", 0.0)) >= float(floors["net_profit"])))
        criteria.append(GateCriterion("floor_expectancy_dollar", float(floors["expectancy_dollar"]), float(metrics.get("expectancy_dollar", 0.0)), float(metrics.get("expectancy_dollar", 0.0)) >= float(floors["expectancy_dollar"])))
        criteria.append(GateCriterion("floor_profit_factor", float(floors["profit_factor"]), float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= float(floors["profit_factor"])))
        criteria.append(GateCriterion("floor_max_drawdown_pct", float(floors["max_drawdown_pct"]), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(floors["max_drawdown_pct"])))

        if reference_metrics:
            rules = PHASE_RETENTION_RULES[phase]
            net_target = max(float(floors["net_profit"]), float(reference_metrics.get("net_profit", 0.0)) * float(rules["net_mult"]))
            trade_target = max(float(floors["trades"]), float(reference_metrics.get("total_trades", 0.0)) * float(rules["trade_mult"]))
            exp_target = max(float(floors["expectancy_dollar"]), float(reference_metrics.get("expectancy_dollar", 0.0)) * float(rules["expectancy_mult"]))
            pf_target = max(float(floors["profit_factor"]), float(reference_metrics.get("profit_factor", 0.0)) * float(rules["pf_mult"]))
            calmar_target = float(reference_metrics.get("calmar", 0.0)) * float(rules["calmar_mult"])
            dd_target = min(float(floors["max_drawdown_pct"]), float(reference_metrics.get("max_drawdown_pct", 0.0)) * float(rules["dd_mult"]))

            criteria.append(GateCriterion("no_regress_net_profit", net_target, float(metrics.get("net_profit", 0.0)), float(metrics.get("net_profit", 0.0)) >= net_target))
            criteria.append(GateCriterion("no_regress_total_trades", trade_target, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= trade_target))
            criteria.append(GateCriterion("no_regress_expectancy_dollar", exp_target, float(metrics.get("expectancy_dollar", 0.0)), float(metrics.get("expectancy_dollar", 0.0)) >= exp_target))
            criteria.append(GateCriterion("no_regress_profit_factor", pf_target, float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= pf_target))
            criteria.append(GateCriterion("no_regress_calmar", calmar_target, float(metrics.get("calmar", 0.0)), float(metrics.get("calmar", 0.0)) >= calmar_target))
            criteria.append(GateCriterion("no_regress_max_drawdown_pct", dd_target, float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= dd_target))
        else:
            criteria.append(GateCriterion("phase_progress", 0.0, 1.0, True))
        return criteria


REBASE_ULTIMATE_TARGETS = {
    "net_profit": 130_000.0,
    "expectancy_dollar": 700.0,
    "total_trades": 150.0,
    "profit_factor": 2.80,
    "max_drawdown_pct": 0.35,
}

REBASE_PHASE_ABSOLUTE_FLOORS = {
    1: {"trades": 95, "net_profit": 55_000, "expectancy_dollar": 400, "profit_factor": 2.10, "max_drawdown_pct": 0.34},
    2: {"trades": 105, "net_profit": 65_000, "expectancy_dollar": 450, "profit_factor": 2.00, "max_drawdown_pct": 0.34},
    3: {"trades": 110, "net_profit": 70_000, "expectancy_dollar": 475, "profit_factor": 2.00, "max_drawdown_pct": 0.35},
    4: {"trades": 115, "net_profit": 80_000, "expectancy_dollar": 500, "profit_factor": 2.00, "max_drawdown_pct": 0.35},
    5: {"trades": 115, "net_profit": 80_000, "expectancy_dollar": 500, "profit_factor": 2.00, "max_drawdown_pct": 0.34},
    6: {"trades": 120, "net_profit": 90_000, "expectancy_dollar": 550, "profit_factor": 2.10, "max_drawdown_pct": 0.34},
    7: {"trades": 125, "net_profit": 100_000, "expectancy_dollar": 600, "profit_factor": 2.20, "max_drawdown_pct": 0.34},
}

REBASE_PHASE_RETENTION_RULES = {
    1: {"trade_mult": 1.55, "net_mult": 0.80, "expectancy_mult": 0.45, "pf_mult": 0.60, "calmar_mult": 0.60, "dd_mult": 1.40},
    2: {"trade_mult": 1.00, "net_mult": 0.80, "expectancy_mult": 0.55, "pf_mult": 0.70, "calmar_mult": 0.65, "dd_mult": 1.40},
    3: {"trade_mult": 1.00, "net_mult": 0.82, "expectancy_mult": 0.58, "pf_mult": 0.72, "calmar_mult": 0.68, "dd_mult": 1.35},
    4: {"trade_mult": 1.00, "net_mult": 0.88, "expectancy_mult": 0.65, "pf_mult": 0.76, "calmar_mult": 0.72, "dd_mult": 1.30},
    5: {"trade_mult": 0.98, "net_mult": 0.90, "expectancy_mult": 0.72, "pf_mult": 0.82, "calmar_mult": 0.76, "dd_mult": 1.25},
    6: {"trade_mult": 0.98, "net_mult": 0.92, "expectancy_mult": 0.72, "pf_mult": 0.82, "calmar_mult": 0.78, "dd_mult": 1.20},
    7: {"trade_mult": 0.98, "net_mult": 0.94, "expectancy_mult": 0.78, "pf_mult": 0.86, "calmar_mult": 0.82, "dd_mult": 1.12},
}


class AKCHelixRebasePlugin:
    """Rebase plugin for re-establishing R1 baseline on corrected NQ daily data."""

    name = "helix_rebase"
    num_phases = max(REBASE_PHASE_FOCUS)
    initial_mutations: dict[str, Any] = dict(REBASE_BASE_MUTATIONS)
    ultimate_targets = REBASE_ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = max(REBASE_PHASE_FOCUS),
    ) -> None:
        if not 1 <= num_phases <= max(REBASE_PHASE_FOCUS):
            raise ValueError(
                f"AKCHelixRebasePlugin supports between 1 and {max(REBASE_PHASE_FOCUS)} phases, got {num_phases}."
            )
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._last_context: dict[str, Any] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        self._cache_source_fingerprint = self._data_fingerprint()

    def _target_overrides_for_phase(self, phase: int) -> dict[str, float]:
        return {
            "net_profit_target": REBASE_NET_PROFIT_TARGET_USD,
            "expectancy_target": REBASE_EXPECTANCY_TARGET_USD,
            "pf_target": REBASE_PF_TARGET,
            "calmar_target": REBASE_CALMAR_TARGET,
            "max_dd_clip": REBASE_MAX_DD_CLIP,
            "trade_target": float(REBASE_TRADE_TARGETS.get(phase, 75.0)),
        }

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = REBASE_PHASE_FOCUS[phase]
        prior = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior.get("suggested_experiments", []))
        reference_metrics = self._reference_metrics(state, phase)
        phase_hard_rejects = self._phase_hard_rejects(phase, reference_metrics)
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_rebase_candidates(
                phase,
                suggested_experiments=[(experiment.name, experiment.mutations) for experiment in suggested] or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=REBASE_WEIGHTS.get(phase),
            hard_rejects=phase_hard_rejects,
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.008 if phase == self.num_phases else 0.006,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
            ),
            max_rounds=15 if phase == self.num_phases else 20,
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
        target_overrides = self._target_overrides_for_phase(phase)
        evaluation_key = build_cache_key(
            "momentum.helix_rebase.evaluation",
            source_fingerprint=self._cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
                "target_overrides": target_overrides,
            },
        )

        def make_parallel():
            from .worker import init_worker, score_candidate

            pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(str(self.data_dir), self.initial_equity),
            )
            return SharedPoolBatchEvaluator(
                pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (
                        candidate.name,
                        candidate.mutations,
                        current_mutations,
                        phase,
                        scoring_weights,
                        hard_rejects,
                        target_overrides,
                    )
                    for candidate in candidates
                ],
                on_close=lambda pool=pool: shutdown_process_pool(pool),
                on_terminate=lambda pool=pool: shutdown_process_pool(pool, force=True),
                description=f"AKC Helix rebase phase {phase}",
            )

        def make_sequential():
            return _RebaseSequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                scoring_weights,
                hard_rejects,
                target_overrides,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"AKC Helix rebase phase {phase}")
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
            f"AKC HELIX REBASE PHASE {phase} SNAPSHOT",
            "=" * 72,
            f"Focus: {REBASE_PHASE_FOCUS[phase][0]}",
            f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f}",
            f"Trades={int(metrics.get('total_trades', 0))} PF={metrics.get('profit_factor', 0.0):.2f} "
            f"Net={metrics.get('net_profit', 0.0):+.0f} DD={metrics.get('max_drawdown_pct', 0.0):.1%} "
            f"Calmar={metrics.get('calmar', 0.0):.2f}",
        ])

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del phase, state, metrics
        context = self._run_config(greedy_result.final_mutations, store_context=True, collect_diagnostics=True)
        return self._build_diagnostics_text(context, title="AKC HELIX REBASE FULL DIAGNOSTICS")

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final_phase = max(state.completed_phases) if state.completed_phases else self.num_phases
        context = self._run_config(state.cumulative_mutations, store_context=True, collect_diagnostics=True)
        metrics = context["metrics"]
        greedy_result_from_state(state, phase=final_phase, final_metrics=metrics)
        diagnostics_text = self._build_diagnostics_text(context, title="AKC HELIX REBASE ROUND DIAGNOSTICS")
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
                f"AKC Helix rebase finished round scoring at {context['score'].total:.4f} with "
                f"{int(metrics.get('total_trades', 0))} trades, PF {metrics.get('profit_factor', 0.0):.2f}, "
                f"net profit ${metrics.get('net_profit', 0.0):,.0f}, and DD {metrics.get('max_drawdown_pct', 0.0):.1%}."
            ),
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        del phase
        gaps: list[str] = []
        if metrics.get("total_trades", 0) < 150:
            gaps.append("Trade frequency still is below the 150-trade expansion target.")
        if metrics.get("net_profit", 0.0) < 130_000.0:
            gaps.append("Net profit still is below the $130K frequency-expansion target.")
        if metrics.get("expectancy_dollar", 0.0) < 700.0:
            gaps.append("Per-trade expectancy still is below the $700 target.")
        if metrics.get("profit_factor", 0.0) < 2.80:
            gaps.append("Profit factor still is below the 2.80 quality target.")
        if metrics.get("max_drawdown_pct", 0.0) > 0.35:
            gaps.append("Drawdown still is above the 35% expansion cap.")
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
            "momentum.helix_rebase.final_metrics",
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
        target_ov = self._target_overrides_for_phase(self.num_phases)
        score = score_phase_metrics(
            self.num_phases,
            metrics,
            self.initial_equity,
            equity_curve=result.equity_curve,
            apply_hard_rejects=False,
            weight_overrides=REBASE_WEIGHTS.get(self.num_phases),
            target_overrides=target_ov,
        )
        metrics_dict = asdict(metrics)
        context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics_dict,
            "score": score,
            "snapshot": build_group_snapshot(
                "Momentum Helix Rebase Strength / Weakness Snapshot",
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

    def _reference_metrics(self, state: PhaseState, phase: int) -> dict[str, float]:
        if phase > 1:
            prior = state.get_phase_metrics(phase - 1)
            if prior:
                return dict(prior)
        baseline_mutations = dict(state.cumulative_mutations or self.initial_mutations or {})
        return self._metrics_for_mutations(baseline_mutations)

    def _metrics_for_mutations(self, mutations: dict[str, Any]) -> dict[str, float]:
        signature = mutation_signature(mutations)
        cached = self._metrics_cache.get(signature)
        if cached is not None:
            return dict(cached)
        return self.compute_final_metrics(mutations)

    def _phase_hard_rejects(self, phase: int, reference_metrics: dict[str, float] | None) -> dict[str, float]:
        rejects = dict(REBASE_HARD_REJECTS.get(phase, {}))
        if reference_metrics:
            baseline_trades = int(float(reference_metrics.get("total_trades", 0.0)))
            rejects["min_trades"] = min(int(rejects.get("min_trades", 0)), baseline_trades)
        return rejects

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        reference_metrics = self._reference_metrics(state, phase)
        phase_rejects = self._phase_hard_rejects(phase, reference_metrics)
        criteria = [
            GateCriterion("hard_min_trades", float(phase_rejects["min_trades"]), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(phase_rejects["min_trades"])),
            GateCriterion("hard_profit_factor", float(phase_rejects["min_pf"]), float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= float(phase_rejects["min_pf"])),
            GateCriterion("hard_max_drawdown_pct", float(phase_rejects["max_dd_pct"]), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(phase_rejects["max_dd_pct"])),
        ]
        floors = dict(REBASE_PHASE_ABSOLUTE_FLOORS[phase])
        if reference_metrics:
            floors["trades"] = min(float(floors["trades"]), float(reference_metrics.get("total_trades", 0.0)))
        criteria.append(GateCriterion("floor_total_trades", float(floors["trades"]), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(floors["trades"])))
        criteria.append(GateCriterion("floor_net_profit", float(floors["net_profit"]), float(metrics.get("net_profit", 0.0)), float(metrics.get("net_profit", 0.0)) >= float(floors["net_profit"])))
        criteria.append(GateCriterion("floor_expectancy_dollar", float(floors["expectancy_dollar"]), float(metrics.get("expectancy_dollar", 0.0)), float(metrics.get("expectancy_dollar", 0.0)) >= float(floors["expectancy_dollar"])))
        criteria.append(GateCriterion("floor_profit_factor", float(floors["profit_factor"]), float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= float(floors["profit_factor"])))
        criteria.append(GateCriterion("floor_max_drawdown_pct", float(floors["max_drawdown_pct"]), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(floors["max_drawdown_pct"])))

        if reference_metrics:
            rules = REBASE_PHASE_RETENTION_RULES[phase]
            net_target = max(float(floors["net_profit"]), float(reference_metrics.get("net_profit", 0.0)) * float(rules["net_mult"]))
            trade_target = max(float(floors["trades"]), float(reference_metrics.get("total_trades", 0.0)) * float(rules["trade_mult"]))
            exp_target = max(float(floors["expectancy_dollar"]), float(reference_metrics.get("expectancy_dollar", 0.0)) * float(rules["expectancy_mult"]))
            pf_target = max(float(floors["profit_factor"]), float(reference_metrics.get("profit_factor", 0.0)) * float(rules["pf_mult"]))
            calmar_target = float(reference_metrics.get("calmar", 0.0)) * float(rules["calmar_mult"])
            dd_target = min(float(floors["max_drawdown_pct"]), float(reference_metrics.get("max_drawdown_pct", 0.0)) * float(rules["dd_mult"]))

            criteria.append(GateCriterion("no_regress_net_profit", net_target, float(metrics.get("net_profit", 0.0)), float(metrics.get("net_profit", 0.0)) >= net_target))
            criteria.append(GateCriterion("no_regress_total_trades", trade_target, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= trade_target))
            criteria.append(GateCriterion("no_regress_expectancy_dollar", exp_target, float(metrics.get("expectancy_dollar", 0.0)), float(metrics.get("expectancy_dollar", 0.0)) >= exp_target))
            criteria.append(GateCriterion("no_regress_profit_factor", pf_target, float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= pf_target))
            criteria.append(GateCriterion("no_regress_calmar", calmar_target, float(metrics.get("calmar", 0.0)), float(metrics.get("calmar", 0.0)) >= calmar_target))
            criteria.append(GateCriterion("no_regress_max_drawdown_pct", dd_target, float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= dd_target))
        else:
            criteria.append(GateCriterion("phase_progress", 0.0, 1.0, True))
        return criteria


class _RebaseSequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        target_overrides: dict[str, float] | None,
    ) -> None:
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._target_overrides = target_overrides
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
            score_candidate((candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects, self._target_overrides))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


def _timestamps_to_numeric(timestamps) -> np.ndarray:
    if timestamps is None or len(timestamps) == 0:
        return np.array([])
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        return np.array([dt.timestamp() for dt in timestamps], dtype=float)
    return np.asarray(timestamps)
