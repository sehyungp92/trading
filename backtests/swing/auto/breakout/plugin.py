from __future__ import annotations

import hashlib
import multiprocessing as mp
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
    deserialize_experiments,
    greedy_result_from_state,
    mutation_signature,
    resolve_worker_processes,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .scoring import PHASE_HARD_REJECTS, PHASE_WEIGHTS, score_phase_metrics

ULTIMATE_TARGETS = {
    "net_return_pct": 100.0,
    "profit_factor": 2.0,
    "max_drawdown_pct": 0.15,
    "total_trades": 50.0,
    "calmar": 2.0,
}


class _PoolBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        symbols: tuple[str, ...],
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        max_workers: int | None,
    ) -> None:
        from .worker import init_worker

        self._pool = mp.Pool(
            processes=resolve_worker_processes(max_workers),
            initializer=init_worker,
            initargs=(str(data_dir), initial_equity, ",".join(symbols)),
        )
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        args = [
            (
                candidate.name,
                candidate.mutations,
                current_mutations,
                self._phase,
                self._scoring_weights,
                self._hard_rejects,
            )
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
        symbols: tuple[str, ...],
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ) -> None:
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._symbols = symbols
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker

        init_worker(str(self._data_dir), self._initial_equity, ",".join(self._symbols))
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
    num_phases = 3
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        symbols: list[str] | tuple[str, ...] | None = None,
        num_phases: int = 3,
    ) -> None:
        if not 1 <= num_phases <= 3:
            raise ValueError(f"BreakoutPlugin supports between 1 and 3 phases, got {num_phases}.")
        resolved_symbols = tuple(symbol.strip().upper() for symbol in (symbols or ("QQQ", "GLD")) if symbol.strip())
        if not resolved_symbols:
            raise ValueError("BreakoutPlugin requires at least one symbol.")

        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.symbols = resolved_symbols
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
        del cumulative_mutations
        evaluation_key = build_cache_key(
            "swing.breakout.evaluation",
            source_fingerprint=self._cache_source_fingerprint,
            extra={
                "phase": phase,
                "symbols": list(self.symbols),
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            return _PoolBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                self.symbols,
                phase,
                scoring_weights,
                hard_rejects,
                self.max_workers,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                self.symbols,
                phase,
                scoring_weights,
                hard_rejects,
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
            use_synchronized=False,
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
            f"Trades={int(metrics.get('total_trades', 0))} PF={metrics.get('profit_factor', 0.0):.2f} "
            f"Net={metrics.get('net_profit', 0.0):+.0f} DD={metrics.get('max_drawdown_pct', 0.0):.1%} "
            f"Calmar={metrics.get('calmar', 0.0):.2f}",
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
        if metrics.get("total_trades", 0) < 20:
            gaps.append("Trade count still is below the preferred breakout activity floor.")
        if metrics.get("profit_factor", 0.0) < 1.4:
            gaps.append("Profit factor still is not strong enough for a mature breakout baseline.")
        if metrics.get("max_drawdown_pct", 0.0) > 0.18:
            gaps.append("Drawdown still is too high relative to the current round target.")
        if phase >= 2 and metrics.get("calmar", 0.0) < 1.5:
            gaps.append("Risk-adjusted return still is lagging the later-phase objective.")
        return gaps

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
                fixed_qty=10,
                track_signals=collect_diagnostics,
                track_shadows=collect_diagnostics,
            ),
            mutations,
        )
        runner = run_breakout_synchronized if use_synchronized else run_breakout_independent
        result = runner(data, config)
        all_trades = self._collect_trades(result)
        metrics = extract_metrics(
            all_trades,
            result.combined_equity,
            _timestamps_to_numeric(result.combined_timestamps),
            self.initial_equity,
        )
        score = score_phase_metrics(
            self.num_phases,
            metrics,
            self.initial_equity,
            equity_curve=result.combined_equity,
        )
        metrics_dict = asdict(metrics)
        context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics_dict,
            "score": score,
            "all_trades": all_trades,
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

    def _data_fingerprint(self) -> str:
        parts: list[str] = []
        for symbol in self.symbols:
            for suffix in ("_1h.parquet", "_1d.parquet"):
                path = self.data_dir / f"{symbol}{suffix}"
                if path.exists():
                    stat = path.stat()
                    parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
                else:
                    parts.append(f"{path.name}:missing")
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        criteria = [
            GateCriterion(
                "hard_min_trades",
                float(PHASE_HARD_REJECTS[phase]["min_trades"]),
                float(metrics.get("total_trades", 0.0)),
                float(metrics.get("total_trades", 0.0)) >= float(PHASE_HARD_REJECTS[phase]["min_trades"]),
            ),
            GateCriterion(
                "hard_profit_factor",
                float(PHASE_HARD_REJECTS[phase]["min_pf"]),
                float(metrics.get("profit_factor", 0.0)),
                float(metrics.get("profit_factor", 0.0)) >= float(PHASE_HARD_REJECTS[phase]["min_pf"]),
            ),
            GateCriterion(
                "hard_max_drawdown_pct",
                float(PHASE_HARD_REJECTS[phase]["max_dd_pct"]),
                float(metrics.get("max_drawdown_pct", 0.0)),
                float(metrics.get("max_drawdown_pct", 0.0)) <= float(PHASE_HARD_REJECTS[phase]["max_dd_pct"]),
            ),
        ]
        if phase == 1:
            criteria.append(
                GateCriterion(
                    "profit_factor",
                    1.20,
                    float(metrics.get("profit_factor", 0.0)),
                    float(metrics.get("profit_factor", 0.0)) >= 1.20,
                )
            )
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


def _trade_sort_key(trade) -> str:
    dt = getattr(trade, "entry_time", None) or getattr(trade, "exit_time", None)
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
