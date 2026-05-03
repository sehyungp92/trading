from __future__ import annotations

import hashlib
import json
from collections import Counter
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
from .scoring import PHASE_HARD_REJECTS, PHASE_WEIGHTS, score_phase_metrics

ULTIMATE_TARGETS = {
    "net_profit": 7_000.0,
    "expectancy_dollar": 47.0,
    "profit_factor": 4.0,
    "max_drawdown_pct": 0.09,
    "total_trades": 140.0,
    "trades_per_month": 2.25,
    "edge_velocity": 105.0,
    "avg_mfe_r": 0.30,
    "winner_capture_ratio": 0.58,
}


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        symbols: tuple[str, ...],
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        fixed_qty: int | None,
    ) -> None:
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._symbols = symbols
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._fixed_qty = fixed_qty
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker

        init_worker(str(self._data_dir), self._initial_equity, ",".join(self._symbols), self._fixed_qty)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate

        return [
            score_candidate(
                (
                    candidate.name,
                    candidate.mutations,
                    current_mutations,
                    self._phase,
                    self._scoring_weights,
                    self._hard_rejects,
                )
            )
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class BreakoutPlugin:
    name = "breakout"
    num_phases = 4
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        symbols: list[str] | tuple[str, ...] | None = None,
        num_phases: int = 4,
        fixed_qty: int | None = None,
    ) -> None:
        if not 1 <= num_phases <= 4:
            raise ValueError(f"BreakoutPlugin supports between 1 and 4 phases, got {num_phases}.")
        resolved_symbols = tuple(symbol.strip().upper() for symbol in (symbols or ("QQQ", "GLD")) if symbol.strip())
        if not resolved_symbols:
            raise ValueError("BreakoutPlugin requires at least one symbol.")

        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.symbols = resolved_symbols
        self.num_phases = num_phases
        self.fixed_qty = fixed_qty
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
                current_mutations=dict(state.cumulative_mutations),
                suggested_experiments=[(experiment.name, experiment.mutations) for experiment in suggested] or None,
            )
        ]
        hard_rejects = self._phase_hard_rejects(phase, state)
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=hard_rejects,
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
        del cumulative_mutations
        evaluation_key = build_cache_key(
            "swing.breakout.evaluation",
            source_fingerprint=self._cache_source_fingerprint,
            extra={
                "phase": phase,
                "symbols": list(self.symbols),
                "evaluation_mode": "synchronized",
                "fixed_qty": self.fixed_qty,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            from .worker import init_worker, score_candidate

            pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(str(self.data_dir), self.initial_equity, ",".join(self.symbols), self.fixed_qty),
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
                description=f"Breakout phase {phase}",
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                self.symbols,
                phase,
                scoring_weights,
                hard_rejects,
                self.fixed_qty,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"Breakout phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        context = self._run_config(
            mutations,
            use_synchronized=True,
            store_context=True,
            collect_diagnostics=False,
        )
        return dict(context["metrics"])

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        del state
        return "\n".join([
            "=" * 72,
            f"BREAKOUT PHASE {phase} SNAPSHOT",
            "=" * 72,
            f"Focus: {PHASE_FOCUS[phase][0]}",
            f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f}",
            f"Trades={int(metrics.get('total_trades', 0))} TPM={metrics.get('trades_per_month', 0.0):.2f} "
            f"Net={metrics.get('net_profit', 0.0):+.0f} Exp$={metrics.get('expectancy_dollar', 0.0):.0f} "
            f"PF={metrics.get('profit_factor', 0.0):.2f} DD={metrics.get('max_drawdown_pct', 0.0):.1%}",
        ])

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        del phase, state, metrics
        context = self._run_config(
            greedy_result.final_mutations,
            use_synchronized=True,
            store_context=True,
            collect_diagnostics=True,
        )
        return self._build_diagnostics_text(context, title="BREAKOUT FULL DIAGNOSTICS")

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final_phase = max(state.completed_phases) if state.completed_phases else self.num_phases
        context = self._run_config(
            state.cumulative_mutations,
            use_synchronized=True,
            store_context=True,
            collect_diagnostics=True,
        )
        metrics = context["metrics"]
        greedy_result_from_state(state, phase=final_phase, final_metrics=metrics)
        diagnostics_text = self._build_diagnostics_text(context, title="BREAKOUT ROUND DIAGNOSTICS")
        entry_type_counts = Counter(getattr(trade, "entry_type", "?") for trade in context["all_trades"])
        exit_reason_counts = Counter(getattr(trade, "exit_reason", "?") for trade in context["all_trades"])

        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics_text,
            dimension_reports={
                "signal_extraction": (
                    f"Symbols active: {len(context['result'].symbol_results)}. "
                    f"Total trades {int(metrics.get('total_trades', 0))}. "
                    f"Entry types: {self._format_counter(entry_type_counts, top_n=4)}."
                ),
                "signal_discrimination": (
                    f"Profit factor {metrics.get('profit_factor', 0.0):.2f}, "
                    f"win rate {metrics.get('win_rate', 0.0):.1%}, "
                    f"score {context['score'].total:.4f}."
                ),
                "entry_mechanism": (
                    f"Most common entry types: {self._format_counter(entry_type_counts, top_n=3)}. "
                    f"Average trades/month {metrics.get('trades_per_month', 0.0):.1f}."
                ),
                "trade_management": (
                    f"Net profit ${metrics.get('net_profit', 0.0):,.0f}, "
                    f"calmar {metrics.get('calmar', 0.0):.2f}, "
                    f"drawdown {metrics.get('max_drawdown_pct', 0.0):.1%}, "
                    f"avg heat {context['result'].heat_stats.avg_heat_pct:.1%}."
                ),
                "exit_mechanism": (
                    f"Common exits: {self._format_counter(exit_reason_counts, top_n=4)}. "
                    f"Avg hold {metrics.get('avg_hold_hours', 0.0):.1f}h."
                ),
            },
            overall_verdict=(
                f"Breakout finished round scoring at {context['score'].total:.4f} with "
                f"{int(metrics.get('total_trades', 0))} trades, PF {metrics.get('profit_factor', 0.0):.2f}, "
                f"net profit ${metrics.get('net_profit', 0.0):,.0f}, and DD {metrics.get('max_drawdown_pct', 0.0):.1%}."
            ),
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        gaps: list[str] = []
        if metrics.get("total_trades", 0) < 140:
            gaps.append("Trade count still below the round-5 activity floor.")
        if metrics.get("trades_per_month", 0.0) < 2.27:
            gaps.append("Trading frequency still below the round-5 target.")
        if metrics.get("net_profit", 0.0) < 6_000.0:
            gaps.append("Net profit is still below the no-tradeoff round-5 floor.")
        if metrics.get("expectancy_dollar", 0.0) < 42.70:
            gaps.append("Per-trade expectancy is still too weak for the round-5 return goal.")
        if metrics.get("edge_velocity", 0.0) < 98.0:
            gaps.append("Expected return velocity remains below the round-5 target.")
        if metrics.get("avg_mfe_r", 0.0) < 0.27:
            gaps.append("Average MFE remains shallow, suggesting too many low-ceiling entries.")
        if metrics.get("winner_capture_ratio", 0.0) < 0.55:
            gaps.append("Winner capture is still leaving too much favorable excursion on the table.")
        if metrics.get("max_drawdown_pct", 0.0) > 0.09:
            gaps.append("Drawdown is above the round-5 risk budget.")
        return gaps

    def run_preflight_ablations(self, output_dir: Path, baseline_mutations: dict[str, Any]) -> None:
        json_path = output_dir / "phase_0_ablation.json"
        text_path = output_dir / "phase_0_ablation.txt"
        replay_json_path = output_dir / "round_4_replay_baseline.json"
        replay_text_path = output_dir / "round_4_replay_baseline.txt"
        if json_path.exists() and text_path.exists() and replay_json_path.exists() and replay_text_path.exists():
            return

        ablations = [
            ("baseline", {}),
            ("early_standard_off", {"param_overrides.ENTRY_C_EARLY_ENABLE": False}),
            ("fresh_market_off", {"param_overrides.ENTRY_C_FRESH_ENABLE": False}),
            ("fresh_stop_off", {"param_overrides.ENTRY_C_FRESH_STOP_ENABLE": False}),
            (
                "late_demotion_off",
                {
                    "param_overrides.C_STANDARD_LATE_RISK_MULT": 1.0,
                    "param_overrides.C_STANDARD_HIGH_DISP_RISK_MULT": 1.0,
                    "param_overrides.C_CONTINUATION_RISK_MULT": 1.0,
                },
            ),
            ("fast_branch_exit_off", {"param_overrides.FAST_BRANCH_MANAGEMENT_ENABLE": False}),
        ]

        rows: list[dict[str, Any]] = []
        baseline_metrics: dict[str, float] | None = None
        for name, delta in ablations:
            mutations = dict(baseline_mutations)
            mutations.update(delta)
            context = self._run_config(
                mutations,
                use_synchronized=True,
                store_context=False,
                collect_diagnostics=False,
            )
            metrics = dict(context["metrics"])
            if baseline_metrics is None:
                baseline_metrics = metrics
            rows.append({
                "name": name,
                "mutations": mutations,
                "metrics": metrics,
            })

        assert baseline_metrics is not None
        for row in rows:
            metrics = row["metrics"]
            row["delta"] = {
                "net_profit": float(metrics.get("net_profit", 0.0) - baseline_metrics.get("net_profit", 0.0)),
                "total_trades": float(metrics.get("total_trades", 0.0) - baseline_metrics.get("total_trades", 0.0)),
                "trades_per_month": float(metrics.get("trades_per_month", 0.0) - baseline_metrics.get("trades_per_month", 0.0)),
                "expectancy_dollar": float(metrics.get("expectancy_dollar", 0.0) - baseline_metrics.get("expectancy_dollar", 0.0)),
                "edge_velocity": float(metrics.get("edge_velocity", 0.0) - baseline_metrics.get("edge_velocity", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) - baseline_metrics.get("max_drawdown_pct", 0.0)),
            }

        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        lines = [
            "=" * 72,
            "  BREAKOUT ROUND 5 PREFLIGHT ABLATION",
            "=" * 72,
        ]
        for row in rows:
            metrics = row["metrics"]
            delta = row["delta"]
            lines.extend([
                "",
                f"[{row['name']}]",
                (
                    f"Trades={int(metrics.get('total_trades', 0))} "
                    f"TPM={metrics.get('trades_per_month', 0.0):.2f} "
                    f"Net=${metrics.get('net_profit', 0.0):,.0f} "
                    f"Exp$={metrics.get('expectancy_dollar', 0.0):.2f} "
                    f"PF={metrics.get('profit_factor', 0.0):.2f} "
                    f"DD={metrics.get('max_drawdown_pct', 0.0):.1%} "
                    f"EdgeV={metrics.get('edge_velocity', 0.0):.2f}"
                ),
                (
                    f"Delta: trades={delta['total_trades']:+.0f} "
                    f"tpm={delta['trades_per_month']:+.2f} "
                    f"net={delta['net_profit']:+.0f} "
                    f"exp={delta['expectancy_dollar']:+.2f} "
                    f"edgeV={delta['edge_velocity']:+.2f} "
                    f"dd={delta['max_drawdown_pct']:+.2%}"
                ),
            ])
        text_path.write_text("\n".join(lines), encoding="utf-8")

        replay_payload = {
            "label": "current_code_round_4_replay_baseline",
            "comparison_policy": "Future breakout rounds compare against this current-code replay, not stale stored round artifacts.",
            "mutations": dict(baseline_mutations),
            "metrics": baseline_metrics,
            "fixed_qty": self.fixed_qty,
            "symbols": list(self.symbols),
        }
        replay_json_path.write_text(json.dumps(replay_payload, indent=2), encoding="utf-8")
        replay_lines = [
            "=" * 72,
            "  BREAKOUT CURRENT-CODE ROUND 4 REPLAY BASELINE",
            "=" * 72,
            "Future round-5 candidates are gated against this replay-clean baseline.",
            f"Fixed quantity override: {self.fixed_qty}",
            f"Symbols: {', '.join(self.symbols)}",
            "",
            (
                f"Trades={int(baseline_metrics.get('total_trades', 0))} "
                f"TPM={baseline_metrics.get('trades_per_month', 0.0):.2f} "
                f"Net=${baseline_metrics.get('net_profit', 0.0):,.0f} "
                f"Exp$={baseline_metrics.get('expectancy_dollar', 0.0):.2f} "
                f"PF={baseline_metrics.get('profit_factor', 0.0):.2f} "
                f"DD={baseline_metrics.get('max_drawdown_pct', 0.0):.1%} "
                f"EdgeV={baseline_metrics.get('edge_velocity', 0.0):.2f}"
            ),
        ]
        replay_text_path.write_text("\n".join(replay_lines), encoding="utf-8")

    def _run_config(
        self,
        mutations: dict[str, Any],
        *,
        use_synchronized: bool,
        store_context: bool,
        collect_diagnostics: bool,
    ) -> dict[str, Any]:
        from backtests.swing.auto.config_mutator import mutate_breakout_config
        from backtests.swing.auto.scoring import extract_metrics
        from backtests.swing.analysis.breakout_diagnostics import breakout_full_diagnostic
        from backtests.swing.analysis.breakout_filter_attribution import breakout_filter_attribution_report
        from backtests.swing.config_breakout import BreakoutBacktestConfig
        from backtests.swing.engine.breakout_portfolio_engine import (
            load_breakout_data,
            run_breakout_independent,
            run_breakout_synchronized,
        )

        cache_key = build_cache_key(
            "swing.breakout.final_metrics",
            source_fingerprint=self._cache_source_fingerprint,
            mutations=mutations,
            extra={
                "symbols": list(self.symbols),
                "initial_equity": self.initial_equity,
                "mode": "sync" if use_synchronized else "independent",
                "collect_diagnostics": collect_diagnostics,
                "fixed_qty": self.fixed_qty,
            },
        )
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            if store_context:
                self._last_context = cached
            return cached

        data = load_breakout_data(list(self.symbols), self.data_dir)
        config = mutate_breakout_config(
            BreakoutBacktestConfig(
                symbols=list(self.symbols),
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                fixed_qty=self.fixed_qty,
                track_signals=collect_diagnostics,
                track_shadows=collect_diagnostics,
            ),
            mutations,
        )
        runner = run_breakout_synchronized if use_synchronized else run_breakout_independent
        result = runner(data, config)
        all_trades = self._collect_trades(result)
        all_signal_events = self._collect_signal_events(result)
        all_branch_shadows = self._collect_branch_shadows(result)
        metrics = extract_metrics(
            all_trades,
            result.combined_equity,
            _timestamps_to_numeric(result.combined_timestamps),
            self.initial_equity,
        )
        metrics.avg_mfe_r = _avg_mfe_r(all_trades)
        metrics.winner_capture_ratio = _winner_capture_ratio(all_trades)
        score = score_phase_metrics(
            self.num_phases,
            metrics,
            self.initial_equity,
            equity_curve=result.combined_equity,
        )
        metrics_dict = asdict(metrics)
        metrics_dict["edge_velocity"] = float(metrics.expectancy_dollar * metrics.trades_per_month)
        context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics_dict,
            "score": score,
            "all_trades": all_trades,
            "all_signal_events": all_signal_events,
            "all_branch_shadows": all_branch_shadows,
            "snapshot": build_group_snapshot(
                "Breakout Strength / Weakness Snapshot",
                all_trades,
                [
                    ("symbol", lambda trade: getattr(trade, "symbol", None)),
                    ("entry type", lambda trade: getattr(trade, "entry_type", None)),
                    ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
                ],
                min_count=3,
            ),
            "full_diagnostic": breakout_full_diagnostic(all_trades) if collect_diagnostics else "",
            "filter_diagnostic": (
                breakout_filter_attribution_report(
                    all_signal_events,
                    all_trades,
                    branch_shadows=all_branch_shadows,
                )
                if collect_diagnostics
                else ""
            ),
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
        sections = [
            "\n".join([
                "=" * 72,
                f"  {title}",
                "=" * 72,
                f"  Symbols: {', '.join(self.symbols)}",
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
                "  PORTFOLIO HEAT",
                "  " + "-" * 55,
                f"    Avg heat:            {result.heat_stats.avg_heat_pct:.1%}",
                f"    Max heat:            {result.heat_stats.max_heat_pct:.1%}",
                f"    Time at limit:       {result.heat_stats.pct_time_at_limit:.1f}%",
            ]),
            self._per_symbol_summary(context),
            context["snapshot"],
        ]
        if context.get("full_diagnostic"):
            sections.append(context["full_diagnostic"])
        if context.get("filter_diagnostic"):
            sections.append(context["filter_diagnostic"])
        return "\n\n".join(section for section in sections if section)

    def _per_symbol_summary(self, context: dict[str, Any]) -> str:
        result = context["result"]
        lines = [
            "  PER-SYMBOL SUMMARY",
            "  " + "-" * 55,
        ]
        for symbol in sorted(result.symbol_results):
            symbol_result = result.symbol_results[symbol]
            trades = symbol_result.trades
            if not trades:
                lines.append(f"    {symbol}: no trades")
                continue
            win_rate = sum(1 for trade in trades if getattr(trade, "r_multiple", 0.0) > 0) / len(trades)
            pnl = sum(_trade_net_pnl(trade) for trade in trades)
            pf = _profit_factor(trades)
            lines.append(
                f"    {symbol}: trades={len(trades)}, win_rate={win_rate:.1%}, "
                f"pf={pf:.2f}, pnl=${pnl:,.0f}"
            )
        return "\n".join(lines)

    def _collect_trades(self, result) -> list:
        trades: list = []
        for symbol, symbol_result in result.symbol_results.items():
            for trade in symbol_result.trades:
                if not getattr(trade, "symbol", ""):
                    trade.symbol = symbol
                trades.append(trade)
        trades.sort(key=_trade_sort_key)
        return trades

    def _collect_signal_events(self, result) -> list:
        signal_events: list = []
        for symbol, symbol_result in result.symbol_results.items():
            for event in getattr(symbol_result, "signal_events", []):
                if not getattr(event, "symbol", ""):
                    event.symbol = symbol
                signal_events.append(event)
        signal_events.sort(key=lambda event: (getattr(event, "timestamp", None) is None, getattr(event, "timestamp", None), getattr(event, "symbol", "")))
        return signal_events

    def _collect_branch_shadows(self, result) -> list:
        branch_shadows: list = []
        for symbol, symbol_result in result.symbol_results.items():
            for shadow in getattr(symbol_result, "branch_shadows", []):
                if not getattr(shadow, "symbol", ""):
                    shadow.symbol = symbol
                branch_shadows.append(shadow)
        branch_shadows.sort(key=lambda shadow: (getattr(shadow, "timestamp", None) is None, getattr(shadow, "timestamp", None), getattr(shadow, "symbol", "")))
        return branch_shadows

    def _data_fingerprint(self) -> str:
        parts: list[str] = []
        # Data files
        for symbol in self.symbols:
            for suffix in ("_1h.parquet", "_1d.parquet"):
                path = self.data_dir / f"{symbol}{suffix}"
                if path.exists():
                    stat = path.stat()
                    parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
                else:
                    parts.append(f"{path.name}:missing")
        # Source code files (lesson 10: source-fingerprinted caches)
        import strategies.swing.breakout.allocator as _alloc_mod
        import strategies.swing.breakout.config as _cfg_mod
        import strategies.swing.breakout.models as _models_mod
        import strategies.swing.breakout.signals as _sig_mod
        import strategies.swing.breakout.stops as _stops_mod
        from backtests.swing.analysis import breakout_diagnostics as _diag_mod
        from backtests.swing.analysis import breakout_filter_attribution as _filter_mod
        from backtests.swing.engine import breakout_engine as _bt_engine_mod
        from . import scoring as _score_mod
        from . import phase_candidates as _pc_mod
        for mod in (
            _cfg_mod,
            _models_mod,
            _sig_mod,
            _alloc_mod,
            _stops_mod,
            _bt_engine_mod,
            _diag_mod,
            _filter_mod,
            _score_mod,
            _pc_mod,
        ):
            src = getattr(mod, "__file__", None)
            if src:
                src_path = Path(src)
                if src_path.exists():
                    stat = src_path.stat()
                    parts.append(f"{src_path.name}:{stat.st_mtime_ns}")
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def _phase_hard_rejects(self, phase: int, state: PhaseState) -> dict[str, float]:
        baseline = dict(state.get_phase_metrics(phase - 1) or {}) if phase > 1 else {}
        if not baseline:
            baseline = self.compute_final_metrics(dict(state.cumulative_mutations))

        defaults = dict(PHASE_HARD_REJECTS.get(phase, {}))
        return {
            "min_trades": max(float(defaults.get("min_trades", 0.0)), float(baseline.get("total_trades", 0.0))),
            "max_dd_pct": max(0.02, float(baseline.get("max_drawdown_pct", defaults.get("max_dd_pct", 0.20))) + 0.005),
            "min_pf": max(float(defaults.get("min_pf", 1.05)), float(baseline.get("profit_factor", 0.0)) * 0.95),
            "min_net_profit": max(float(defaults.get("min_net_profit", 0.0)), float(baseline.get("net_profit", 0.0))),
            "min_expectancy_dollar": max(
                float(defaults.get("min_expectancy_dollar", 0.0)),
                float(baseline.get("expectancy_dollar", 0.0)),
            ),
            "min_tpm": max(float(defaults.get("min_tpm", 0.0)), float(baseline.get("trades_per_month", 0.0))),
            "min_edge_velocity": max(
                float(defaults.get("min_edge_velocity", 0.0)),
                float(baseline.get("edge_velocity", 0.0)),
            ),
        }

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        hard_rejects = self._phase_hard_rejects(phase, state)
        criteria = [
            GateCriterion(
                "hard_min_trades",
                float(hard_rejects["min_trades"]),
                float(metrics.get("total_trades", 0.0)),
                float(metrics.get("total_trades", 0.0)) >= float(hard_rejects["min_trades"]),
            ),
            GateCriterion(
                "hard_profit_factor",
                float(hard_rejects["min_pf"]),
                float(metrics.get("profit_factor", 0.0)),
                float(metrics.get("profit_factor", 0.0)) >= float(hard_rejects["min_pf"]),
            ),
            GateCriterion(
                "hard_net_profit",
                float(hard_rejects["min_net_profit"]),
                float(metrics.get("net_profit", 0.0)),
                float(metrics.get("net_profit", 0.0))
                >= float(hard_rejects["min_net_profit"]),
            ),
            GateCriterion(
                "hard_max_drawdown_pct",
                float(hard_rejects["max_dd_pct"]),
                float(metrics.get("max_drawdown_pct", 0.0)),
                float(metrics.get("max_drawdown_pct", 0.0)) <= float(hard_rejects["max_dd_pct"]),
            ),
            GateCriterion(
                "hard_expectancy_dollar",
                float(hard_rejects["min_expectancy_dollar"]),
                float(metrics.get("expectancy_dollar", 0.0)),
                float(metrics.get("expectancy_dollar", 0.0))
                >= float(hard_rejects["min_expectancy_dollar"]),
            ),
            GateCriterion(
                "hard_edge_velocity",
                float(hard_rejects["min_edge_velocity"]),
                float(metrics.get("edge_velocity", 0.0)),
                float(metrics.get("edge_velocity", 0.0))
                >= float(hard_rejects["min_edge_velocity"]),
            ),
        ]
        if "min_tpm" in hard_rejects:
            criteria.append(
                GateCriterion(
                    "hard_trades_per_month",
                    float(hard_rejects["min_tpm"]),
                    float(metrics.get("trades_per_month", 0.0)),
                    float(metrics.get("trades_per_month", 0.0))
                    >= float(hard_rejects["min_tpm"]),
                )
            )
        if phase == 1:
            return criteria

        prior = state.get_phase_metrics(phase - 1) or {}
        if prior:
            regression_targets = (
                ("net_profit", 1.00),
                ("expectancy_dollar", 1.00),
                ("trades_per_month", 1.00),
                ("edge_velocity", 1.00),
            )
            for name, floor_mult in regression_targets:
                target = float(prior.get(name, 0.0)) * floor_mult
                actual = float(metrics.get(name, 0.0))
                criteria.append(GateCriterion(f"no_regress_{name}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion("phase_progress", 0.0, 1.0, True))
        return criteria

    @staticmethod
    def _format_counter(counter: Counter, *, top_n: int) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{name}={count}" for name, count in counter.most_common(top_n))


def _profit_factor(trades: list) -> float:
    gross_profit = sum(
        _trade_net_pnl(trade)
        for trade in trades
        if _trade_net_pnl(trade) > 0
    )
    gross_loss = abs(sum(
        _trade_net_pnl(trade)
        for trade in trades
        if _trade_net_pnl(trade) < 0
    ))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _trade_net_pnl(trade) -> float:
    return float(getattr(trade, "pnl_dollars", 0.0)) - float(getattr(trade, "commission", 0.0) or 0.0)


def _timestamps_to_numeric(timestamps) -> np.ndarray:
    if timestamps is None or len(timestamps) == 0:
        return np.array([])
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        return np.array([dt.timestamp() for dt in timestamps], dtype=float)
    if hasattr(first, "item"):
        first = first.item()
        if hasattr(first, "timestamp"):
            return np.array([ts.item().timestamp() for ts in timestamps], dtype=float)
    return np.asarray(timestamps)


def _avg_mfe_r(trades: list) -> float:
    if not trades:
        return 0.0
    values = [float(getattr(trade, "mfe_r", 0.0) or 0.0) for trade in trades]
    return float(np.mean(values)) if values else 0.0


def _winner_capture_ratio(trades: list) -> float:
    captures = []
    for trade in trades:
        r_multiple = float(getattr(trade, "r_multiple", 0.0) or 0.0)
        mfe_r = float(getattr(trade, "mfe_r", 0.0) or 0.0)
        if r_multiple > 0 and mfe_r > 0:
            captures.append(r_multiple / mfe_r)
    return float(np.mean(captures)) if captures else 0.0


def _trade_sort_key(trade) -> str:
    dt = getattr(trade, "entry_time", None) or getattr(trade, "exit_time", None)
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
