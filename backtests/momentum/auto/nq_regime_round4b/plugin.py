from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.momentum.analysis.regime_attribution import module_attribution
from backtests.momentum.analysis.regime_diagnostics import generate_regime_diagnostics
from backtests.momentum.config_regime import NqRegimeBacktestConfig
from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest
from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_tree
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import CachedBatchEvaluator, SharedPoolBatchEvaluator, create_process_pool, mutation_signature, shutdown_process_pool
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


class NqRegimeRound4bPlugin:
    name = "nq_regime_round4b"
    num_phases = 6
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = {
        "module_structural_expansion_trades": 60.0,
        "module_structural_expansion_total_r": 8.0,
        "module_structural_expansion_avg_r": 0.10,
        "module_structural_expansion_profit_factor": 1.35,
        "module_structural_expansion_mfe_capture": 0.55,
        "routing_structural_expansion_selected": 220.0,
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
        num_phases: int = 6,
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
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._context_cache: dict[str, Any] = {}
        self._last_context: dict[str, Any] = {}
        self._pool = None

    def _replay_bundle(self) -> ReplayBundle:
        repo_root = Path(__file__).resolve().parents[4]
        fingerprint = ":".join(
            [
                fingerprint_tree(self.data_dir, patterns=("*.parquet", "*.csv")),
                fingerprint_tree(repo_root / "strategies" / "momentum" / "nq_regime", patterns=("*.py",)),
                fingerprint_tree(repo_root / "backtests" / "momentum" / "engine", patterns=("regime_engine.py",)),
                fingerprint_tree(repo_root / "backtests" / "momentum", patterns=("config_regime.py",), recursive=False),
            ]
        )
        if self._bundle is None or fingerprint != self._fingerprint:
            self._evaluation_cache.clear()
            self._metrics_cache.clear()
            self._context_cache.clear()
            cfg = self._base_config()
            self._bundle = ReplayBundle(load_nq_regime_data(cfg), str(self.data_dir), fingerprint)
            self._fingerprint = fingerprint
        return self._bundle

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
            max_rounds=8,
            prune_threshold=0.05,
            reject_streak_limit=1,
        )

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights=None, hard_rejects=None):
        del cumulative_mutations
        bundle = self._replay_bundle()
        init_worker(str(self.data_dir), self.initial_equity, self.analysis_symbol, self.trade_symbol)
        signature_prefix = build_cache_key(
            "momentum.nq_regime.round4b.evaluation",
            source_fingerprint=bundle.cache_source_fingerprint,
            extra={
                "phase": phase,
                "analysis_symbol": self.analysis_symbol,
                "trade_symbol": self.trade_symbol,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )
        delegate = (
            _SequentialBatchEvaluator(phase, scoring_weights, hard_rejects)
            if (self.max_workers or 1) <= 1
            else self._pool_evaluator(phase, scoring_weights, hard_rejects)
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

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        return dict(self._run_config(mutations)["metrics"])

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        context = self._run_config(greedy_result.final_mutations)
        summary = (
            f"NQ_REGIME round_4b phase {phase}: score {greedy_result.base_score:.4f}->{greedy_result.final_score:.4f}, "
            f"trades={metrics.get('total_trades', 0):.0f}, "
            f"structural={metrics.get('module_structural_expansion_trades', 0):.0f}, "
            f"struct_avgR={metrics.get('module_structural_expansion_avg_r', 0):+.3f}, "
            f"PF={metrics.get('profit_factor', 0):.2f}, DD={metrics.get('max_drawdown_pct', 0):.1%}."
        )
        return summary + "\n\n" + _format_diagnostics(context)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del phase, state, metrics
        context = self._run_config(greedy_result.final_mutations)
        return _format_diagnostics(context)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        context = self._run_config(state.cumulative_mutations)
        attribution = module_attribution(context["trades"])
        metrics = context["metrics"]
        return EndOfRoundArtifacts(
            final_diagnostics_text=_format_diagnostics(context),
            dimension_reports={
                "structural_nq_2": _format_structural_summary(metrics),
                "component_edges": _format_component_dimension(attribution),
                "candidate_funnel": _format_candidate_funnel_dimension(metrics),
                "risk": f"DD={metrics.get('max_drawdown_pct', 0):.1%}, total R={metrics.get('total_r', 0):.2f}",
            },
            overall_verdict=f"NQ_REGIME round_4b finished with net ${metrics.get('net_profit', 0.0):.2f}.",
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

    def _pool_evaluator(self, phase: int, scoring_weights, hard_rejects) -> SharedPoolBatchEvaluator:
        if self._pool is None:
            self._pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(str(self.data_dir), self.initial_equity, self.analysis_symbol, self.trade_symbol),
                description="NQ_REGIME round_4b auto",
            )
        return SharedPoolBatchEvaluator(
            self._pool,
            worker_fn=score_candidate,
            build_args=lambda candidates, current: [
                (candidate.name, candidate.mutations, current, phase, scoring_weights, hard_rejects)
                for candidate in candidates
            ],
            on_terminate=self.close_pool,
            description=f"NQ_REGIME round_4b phase {phase}",
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


def _format_structural_summary(metrics: dict[str, float]) -> str:
    selected = metrics.get("routing_structural_expansion_selected", 0.0)
    trades = metrics.get("module_structural_expansion_trades", 0.0)
    conversion = trades / selected if selected else 0.0
    return (
        f"N={trades:.0f}, avgR={metrics.get('module_structural_expansion_avg_r', 0):+.3f}, "
        f"totalR={metrics.get('module_structural_expansion_total_r', 0):+.2f}, "
        f"PF={metrics.get('module_structural_expansion_profit_factor', 0):.2f}, "
        f"MFE capture={metrics.get('module_structural_expansion_mfe_capture', 0):.1%}, "
        f"+MFE loser={metrics.get('module_structural_expansion_positive_mfe_loser_rate', 0):.1%}, "
        f"selected-to-fill={conversion:.1%}"
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
    for module in ("structural_expansion", "liquidity_reversion", "second_wind"):
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
