from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.momentum.analysis.regime_attribution import module_attribution
from backtests.momentum.analysis.regime_diagnostics import generate_regime_diagnostics
from backtests.momentum.config_regime import NqRegimeBacktestConfig
from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest
from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_tree, stable_signature
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    mutation_signature,
    shutdown_process_pool,
)
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_gates import gate_criteria_for_phase
from .scoring import PHASE_HARD_REJECTS, PHASE_WEIGHTS
from .worker import init_worker, mutate_config, score_candidate


class _SequentialBatchEvaluator:
    def __init__(self, phase: int, scoring_weights, hard_rejects) -> None:
        self.phase = phase
        self.scoring_weights = scoring_weights
        self.hard_rejects = hard_rejects

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        return [
            score_candidate((candidate.name, candidate.mutations, current_mutations, self.phase, self.scoring_weights, self.hard_rejects))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class NqRegimePlugin:
    name = "nq_regime"
    num_phases = 7
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = {
        "module_liquidity_reversion_trades": 120.0,
        "module_liquidity_reversion_total_r_per_month": 4.00,
        "module_liquidity_reversion_avg_r": 1.40,
        "module_liquidity_reversion_mfe_capture": 0.80,
        "routing_liquidity_reversion_request_to_fill_rate": 0.45,
        "max_drawdown_pct": 0.12,
    }

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 1,
        *,
        symbols=None,
        analysis_symbol: str = "NQ",
        trade_symbol: str = "MNQ",
        num_phases: int = 7,
    ) -> None:
        del symbols
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.analysis_symbol = analysis_symbol.upper()
        self.trade_symbol = trade_symbol.upper()
        self.num_phases = num_phases
        self._bundle: ReplayBundle | None = None
        self._fingerprint = ""
        self._fingerprint_parts: dict[str, str] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._context_cache: dict[str, Any] = {}
        self._last_context: dict[str, Any] = {}
        self._pool = None

    def _replay_bundle(self) -> ReplayBundle:
        fingerprint_parts = self._source_fingerprint_parts()
        fingerprint = stable_signature(fingerprint_parts)
        if self._bundle is None or fingerprint != self._fingerprint:
            self._evaluation_cache.clear()
            self._metrics_cache.clear()
            self._context_cache.clear()
            cfg = self._base_config()
            self._bundle = ReplayBundle(load_nq_regime_data(cfg), str(self.data_dir), fingerprint)
            self._fingerprint = fingerprint
            self._fingerprint_parts = fingerprint_parts
        return self._bundle

    def source_fingerprint(self) -> str:
        self._replay_bundle()
        return self._fingerprint

    def source_fingerprint_parts(self) -> dict[str, str]:
        self._replay_bundle()
        return dict(self._fingerprint_parts)

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        candidates = [Experiment(name, muts) for name, muts in get_phase_candidates(phase, state.cumulative_mutations)]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(focus_metrics=focus_metrics),
            max_rounds=16,
            prune_threshold=0.05,
        )

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights=None, hard_rejects=None):
        del cumulative_mutations
        bundle = self._replay_bundle()
        init_worker(str(self.data_dir), self.initial_equity, self.analysis_symbol, self.trade_symbol)
        signature_prefix = build_cache_key(
            "momentum.nq_regime.evaluation",
            source_fingerprint=bundle.cache_source_fingerprint,
            extra={
                "phase": phase,
                "analysis_symbol": self.analysis_symbol,
                "trade_symbol": self.trade_symbol,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )
        if (self.max_workers or 1) <= 1:
            delegate = _SequentialBatchEvaluator(phase, scoring_weights, hard_rejects)
        else:
            delegate = ResilientBatchEvaluator(
                preferred_factory=lambda: self._pool_evaluator(phase, scoring_weights, hard_rejects),
                fallback_factory=lambda: _SequentialBatchEvaluator(phase, scoring_weights, hard_rejects),
                description=f"NQ_REGIME phase {phase}",
            )
        return CachedBatchEvaluator(
            delegate,
            cache=self._evaluation_cache,
            signature_prefix=signature_prefix,
            metrics_cache=self._metrics_cache,
        )

    def close_pool(self) -> None:
        shutdown_process_pool(self._pool)
        self._pool = None

    def terminate_pool(self) -> None:
        shutdown_process_pool(self._pool, force=True)
        self._pool = None

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        return dict(self._run_config(mutations)["metrics"])

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        context = self._run_config(greedy_result.final_mutations)
        summary = (
            f"NQ_REGIME phase {phase}: score {greedy_result.base_score:.4f}->{greedy_result.final_score:.4f}, "
            f"trades={metrics.get('total_trades', 0):.0f}, PF={metrics.get('profit_factor', 0):.2f}, "
            f"DD={metrics.get('max_drawdown_pct', 0):.1%}."
        )
        return summary + "\n\n" + _format_diagnostics(context)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del phase, state, metrics
        context = self._run_config(greedy_result.final_mutations)
        return _format_diagnostics(context)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        context = self._run_config(state.cumulative_mutations)
        attribution = module_attribution(context["trades"])
        return EndOfRoundArtifacts(
            final_diagnostics_text=_format_diagnostics(context),
            dimension_reports={
                "component_edges": _format_component_dimension(attribution),
                "regime_routing": (
                    f"Decisions={context['metrics'].get('routing_decisions', 0):.0f}, "
                    f"selected={context['metrics'].get('routing_selected_rate', 0):.1%}, "
                    f"avg blocked={context['metrics'].get('routing_avg_blocked', 0):.2f}"
                ),
                "candidate_funnel": _format_candidate_funnel_dimension(context["metrics"]),
                "risk": f"DD={context['metrics'].get('max_drawdown_pct', 0):.1%}, total R={context['metrics'].get('total_r', 0):.2f}",
            },
            overall_verdict=f"NQ_REGIME finished with net ${context['metrics'].get('net_profit', 0.0):.2f}.",
        )

    def _run_config(self, mutations: dict[str, Any]) -> dict[str, Any]:
        key = mutation_signature(mutations)
        if key in self._context_cache:
            return self._context_cache[key]
        bundle = self._replay_bundle()
        cfg = mutate_config(self._base_config(), mutations)
        result = run_nq_regime_backtest(bundle.data, cfg)
        context = {"result": result, "trades": result.trades, "metrics": dict(result.metrics)}
        self._context_cache[key] = context
        self._metrics_cache[key] = dict(result.metrics)
        self._last_context = context
        return context

    def _base_config(self) -> NqRegimeBacktestConfig:
        return NqRegimeBacktestConfig(
            data_dir=self.data_dir,
            initial_equity=self.initial_equity,
            analysis_symbol=self.analysis_symbol,
            trade_symbol=self.trade_symbol,
        )

    def _source_fingerprint_parts(self) -> dict[str, str]:
        repo_root = Path(__file__).resolve().parents[4]
        round_package = Path(__file__).resolve().parent
        return {
            "data_raw": fingerprint_tree(self.data_dir, patterns=("*.parquet", "*.csv")),
            "strategy_nq_regime": fingerprint_tree(repo_root / "strategies" / "momentum" / "nq_regime", patterns=("*.py",)),
            "backtest_regime_engine": fingerprint_tree(repo_root / "backtests" / "momentum" / "engine", patterns=("regime_engine.py",)),
            "backtest_regime_config": fingerprint_tree(repo_root / "backtests" / "momentum", patterns=("config_regime.py",), recursive=False),
            "round4c_auto_package": fingerprint_tree(round_package, patterns=("*.py",)),
            "shared_auto_runner": fingerprint_tree(repo_root / "backtests" / "shared" / "auto", patterns=("*.py",)),
        }

    def _pool_evaluator(self, phase: int, scoring_weights, hard_rejects) -> SharedPoolBatchEvaluator:
        if self._pool is None:
            self._pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(str(self.data_dir), self.initial_equity, self.analysis_symbol, self.trade_symbol),
                description="NQ_REGIME auto",
            )
        return SharedPoolBatchEvaluator(
            self._pool,
            worker_fn=score_candidate,
            build_args=lambda candidates, current: [
                (candidate.name, candidate.mutations, current, phase, scoring_weights, hard_rejects)
                for candidate in candidates
            ],
            on_terminate=self.terminate_pool,
            description=f"NQ_REGIME phase {phase}",
            heartbeat_seconds=120.0,
            per_candidate_timeout_seconds=1200.0,
            minimum_timeout_seconds=1200.0,
        )

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        return gate_criteria_for_phase(phase, metrics)


def _format_diagnostics(context: dict[str, Any]) -> str:
    result = context["result"]
    return generate_regime_diagnostics(
        list(context["trades"]),
        dict(context["metrics"]),
        signal_events=list(result.signal_events),
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
    )


def _format_component_dimension(attribution: dict[str, dict[str, float]]) -> str:
    if not attribution:
        return "No trades yet."
    rows = []
    for module, stats in sorted(attribution.items()):
        rows.append(
            f"{module}: N={stats.get('trades', 0):.0f}, PF={stats.get('profit_factor', 0):.2f}, "
            f"avgR={stats.get('avg_r', 0):+.3f}, MFE={stats.get('avg_mfe_r', 0):+.2f}"
        )
    return "; ".join(rows)


def _format_candidate_funnel_dimension(metrics: dict[str, float]) -> str:
    rows = []
    for module in ("second_wind", "structural_expansion", "liquidity_reversion"):
        candidates = metrics.get(f"routing_{module}_candidates", 0.0)
        selected = metrics.get(f"routing_{module}_selected", 0.0)
        executed = metrics.get(f"module_{module}_trades", 0.0)
        select_rate = metrics.get(f"routing_{module}_select_rate_when_candidate", 0.0)
        exec_rate = executed / selected if selected else 0.0
        rows.append(
            f"{module}: cand={candidates:.0f}, sel={selected:.0f} ({select_rate:.1%}), "
            f"exec={executed:.0f} ({exec_rate:.1%}/sel)"
        )
    return "; ".join(rows)
