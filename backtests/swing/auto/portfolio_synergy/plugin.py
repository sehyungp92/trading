from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key
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
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

from .evaluator import evaluate_portfolio, load_evaluation_bundle, load_evaluation_data
from .phase_candidates import (
    PHASE_FOCUS,
    PHASE_GATES,
    ROUND_NAME,
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
)
from .scoring import SCORE_COMPONENTS, score_portfolio_metrics

logger = logging.getLogger(__name__)


FOCUS_METRICS: dict[int, list[str]] = {
    1: ["active_strategy_count", "active_trades_per_month", "max_drawdown_pct"],
    2: ["net_return_pct", "active_trades_per_month", "total_r_per_month"],
    3: ["net_return_pct", "max_strategy_trade_share", "max_drawdown_pct"],
    4: ["overlay_return_pct", "idle_cash_deployment_share", "overlay_score_drag_pct"],
    5: ["active_trades_per_month", "profit_factor", "expectancy_r"],
    6: ["trade_capture_ratio", "positive_alpha_block_rate", "max_strategy_trade_share"],
    7: ["net_return_pct", "active_trades_per_month", "max_drawdown_pct", "positive_years"],
}


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._data_dir = Path(data_dir)
        self._initial_equity = float(initial_equity)
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._unified_data = None

    def _ensure_data(self) -> None:
        if self._unified_data is not None:
            return
        self._unified_data, _ = load_evaluation_data(self._data_dir, self._initial_equity)

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_data()
        scored: list[ScoredCandidate] = []
        for candidate in candidates:
            try:
                merged = dict(current_mutations or {})
                merged.update(candidate.mutations or {})
                metrics = evaluate_portfolio(
                    merged,
                    data_dir=self._data_dir,
                    initial_equity=self._initial_equity,
                    unified_data=self._unified_data,
                )
                score = score_portfolio_metrics(
                    metrics,
                    scoring_weights=self._scoring_weights,
                    hard_rejects=self._hard_rejects,
                )
                scored.append(
                    ScoredCandidate(
                        name=candidate.name,
                        score=score.total,
                        rejected=score.rejected,
                        reject_reason=score.reject_reason,
                        metrics=_with_score_metrics(metrics, score),
                    )
                )
            except Exception as exc:
                scored.append(ScoredCandidate(candidate.name, 0.0, True, str(exc), {}))
        return scored

    def close(self) -> None:
        pass


class PortfolioSynergyPlugin:
    name = "portfolio_synergy"
    num_phases = 7
    ultimate_targets = {
        "active_strategy_count": ROUND_TARGETS["min_active_strategies"],
        "active_trades_per_month": ROUND_TARGETS["min_active_trades_per_month"],
        "total_r_per_month": ROUND_TARGETS["min_total_r_per_month"],
        "profit_factor": ROUND_TARGETS["min_profit_factor"],
        "max_drawdown_pct": ROUND_TARGETS["target_max_drawdown_pct"],
        "trade_capture_ratio": 0.82,
        "positive_years": 4.0,
    }

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 25_000.0,
        max_workers: int | None = 2,
    ):
        self.data_dir = Path(data_dir)
        self.initial_equity = float(initial_equity)
        self.max_workers = 2 if max_workers is None else min(2, max(1, int(max_workers)))
        self.initial_mutations: dict[str, Any] | None = dict(SEED_PORTFOLIO_CONFIG)
        self._pool: mp.pool.Pool | None = None
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cached_unified_data = None
        self._cached_bundle = None
        self._cache_source_fingerprint = ""

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        candidates = [
            Experiment(name=item["name"], mutations=item["mutations"])
            for item in get_phase_candidates(phase)
        ]
        return PhaseSpec(
            focus=PHASE_FOCUS[phase],
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics),
            scoring_weights=dict(SCORE_WEIGHTS),
            hard_rejects=self._hard_rejects_for_phase(phase),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=FOCUS_METRICS[phase],
                min_effective_score_delta_pct=0.003,
                diagnostic_gap_fn=self._diagnostic_gaps,
            ),
            max_rounds=10,
            prune_threshold=0.0,
            reject_streak_limit=2,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        if scoring_weights and len(scoring_weights) > 7:
            raise ValueError("Portfolio synergy scoring cannot use more than 7 components.")
        evaluation_key = build_cache_key(
            "swing.portfolio_synergy.evaluation",
            source_fingerprint=self._ensure_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "score_components": list(SCORE_COMPONENTS),
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
                "initial_equity": self.initial_equity,
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (
                        candidate.name,
                        candidate.mutations,
                        current_mutations,
                        scoring_weights,
                        hard_rejects,
                    )
                    for candidate in candidates
                ],
                on_terminate=self._destroy_pool,
                description=f"portfolio synergy phase {phase}",
                logger=logger,
                heartbeat_seconds=60.0,
                per_candidate_timeout_seconds=900.0,
                minimum_timeout_seconds=900.0,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                scoring_weights,
                hard_rejects,
            )

        raw = ResilientBatchEvaluator(
            make_parallel,
            make_sequential,
            description=f"portfolio synergy phase {phase}",
            logger=logger,
        )
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None:
            score = score_portfolio_metrics(cached, scoring_weights=SCORE_WEIGHTS)
            metrics = _with_score_metrics(cached, score)
            self._metrics_cache[sig] = dict(metrics)
            return metrics
        self._ensure_local_data()
        metrics = evaluate_portfolio(
            mutations,
            data_dir=self.data_dir,
            initial_equity=self.initial_equity,
            unified_data=self._cached_unified_data,
        )
        score = score_portfolio_metrics(metrics, scoring_weights=SCORE_WEIGHTS)
        metrics = _with_score_metrics(metrics, score)
        self._metrics_cache[sig] = dict(metrics)
        return metrics

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        return self._format_diagnostics(f"PHASE {phase} DIAGNOSTICS", metrics, greedy_result)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        diagnostics = self._format_diagnostics("FINAL PORTFOLIO SYNERGY DIAGNOSTICS", metrics, None)
        verdict = "PASS" if self._final_gate_passed(metrics) else "REVIEW"
        score_section = "\n".join(
            f"- {key}: {metrics.get(f'score_{key}', 0.0):.4f} (weight {SCORE_WEIGHTS[key]:.2f})"
            for key in SCORE_COMPONENTS
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics,
            dimension_reports={"score_components": score_section},
            overall_verdict=verdict,
            extra_sections={
                "Score Components": score_section,
                "Execution Note": (
                    "All-five evaluation replays ATRSS, AKC Helix, BRS, Swing Breakout, "
                    "and overlay through the unified swing portfolio engine."
                ),
            },
        )

    def close_pool(self) -> None:
        shutdown_process_pool(self._pool)
        self._pool = None

    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity),
            logger=logger,
            description="portfolio synergy evaluation",
        )

    def _destroy_pool(self) -> None:
        shutdown_process_pool(self._pool, force=True)
        self._pool = None

    def _ensure_local_data(self) -> None:
        self._cached_unified_data = self._ensure_bundle().data

    def _ensure_bundle(self):
        bundle = load_evaluation_bundle(self.data_dir, self.initial_equity)
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._destroy_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        self._cached_unified_data = bundle.data
        return bundle

    def _hard_rejects_for_phase(self, phase: int) -> dict[str, float]:
        gate = PHASE_GATES[phase]
        return {
            "max_drawdown_pct": gate.get("hard_max_drawdown_pct", ROUND_TARGETS["hard_max_drawdown_pct"]),
            "min_active_strategies": gate.get("min_active_strategies", 5.0),
        }

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        return [_criterion(name, target, metrics) for name, target in PHASE_GATES[phase].items()]

    def _diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        gaps: list[str] = []
        if metrics.get("active_strategy_count", 0.0) < 5:
            gaps.append("Not all five sleeves produced active contribution.")
        if metrics.get("max_drawdown_pct", 0.0) > ROUND_TARGETS["target_max_drawdown_pct"]:
            gaps.append("Drawdown is above the controlled-aggressive target; inspect equity clustering.")
        if metrics.get("overlay_score_drag_pct", 0.0) > 0.03:
            gaps.append("Overlay is dragging net score; inspect overlay cap and QQQ/GLD split.")
        if metrics.get("positive_alpha_block_rate", 0.0) > 0.18:
            gaps.append("Heat or routing blocks are high enough to suppress frequency.")
        return gaps

    def _format_diagnostics(self, title: str, metrics: dict[str, float], greedy_result) -> str:
        lines = [
            "=" * 78,
            title,
            "=" * 78,
            f"Round: {ROUND_NAME}",
            f"Initial equity: ${self.initial_equity:,.0f}",
            f"Max workers: {self.max_workers}",
            f"Score components: {len(SCORE_COMPONENTS)} ({', '.join(SCORE_COMPONENTS)})",
            "",
            "Headline:",
            f"  Final equity: ${metrics.get('final_equity', 0.0):,.2f}",
            f"  Net PnL: ${metrics.get('net_pnl', 0.0):+,.2f}",
            f"  Net return: {metrics.get('net_return_pct', 0.0):+.2%}",
            f"  Active trades/month: {metrics.get('active_trades_per_month', 0.0):.2f}",
            f"  Total R/month: {metrics.get('total_r_per_month', 0.0):.2f}",
            f"  Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
            f"  Win rate: {metrics.get('win_rate', 0.0):.2%}",
            f"  Max DD: {metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"  Sharpe: {metrics.get('sharpe', 0.0):.2f}",
            f"  Sortino: {metrics.get('sortino', 0.0):.2f}",
            f"  Calmar: {metrics.get('calmar', 0.0):.2f}",
            f"  Risk basis: {metrics.get('risk_basis', 'realized_delta_plus_overlay_mtm')}",
            f"  Realized-only Max DD: {metrics.get('max_drawdown_pct_realized', metrics.get('max_drawdown_pct', 0.0)):.2%}",
            f"  Realized-only Calmar: {metrics.get('calmar_realized', metrics.get('calmar', 0.0)):.2f}",
            f"  Active sleeves: {metrics.get('active_strategy_count', 0.0):.0f}/5",
            f"  Portfolio entry opportunities fired: {metrics.get('entry_signals_fired', 0.0):.0f}",
            f"  Portfolio entries accepted: {metrics.get('entries_accepted_by_portfolio', 0.0):.0f}",
            f"  Portfolio entries blocked: {metrics.get('entries_blocked_by_portfolio', 0.0):.0f}",
            "",
            "Sleeve contribution:",
        ]
        for strategy in STRATEGY_ORDER:
            lines.append(
                f"  {strategy:<20} trades={metrics.get(f'trades_{strategy}', 0.0):>6.0f} "
                f"pnl=${metrics.get(f'pnl_{strategy}', 0.0):>+11,.2f}"
            )
        lines.extend(
            [
                "",
                "Routing/risk:",
                f"  Max strategy trade share: {metrics.get('max_strategy_trade_share', 0.0):.2%}",
                f"  Trade capture ratio: {metrics.get('trade_capture_ratio', 0.0):.2%}",
                f"  Entry accept rate: {metrics.get('entry_accept_rate', 0.0):.2%}",
                f"  Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
                f"  Overlay max equity cap: {metrics.get('overlay_max_equity_pct', 0.0):.0%}",
                f"  Overlay net PnL: ${metrics.get('overlay_pnl_net', metrics.get('overlay_pnl', 0.0)):+,.2f}",
                f"  Overlay transaction costs: ${metrics.get('overlay_commission', 0.0):,.2f}",
                "",
                "Score:",
            ]
        )
        for key in SCORE_COMPONENTS:
            lines.append(
                f"  {key:<22} component={metrics.get(f'score_{key}', 0.0):.4f} "
                f"weight={SCORE_WEIGHTS[key]:.2f}"
            )
        lines.append(f"  {'total':<22} {metrics.get('score_total', 0.0):.4f}")
        if greedy_result is not None:
            lines.extend(
                [
                    "",
                    "Greedy:",
                    f"  Base score: {greedy_result.base_score:.4f}",
                    f"  Final score: {greedy_result.final_score:.4f}",
                    f"  Accepted: {greedy_result.accepted_count}",
                    f"  Kept: {', '.join(greedy_result.kept_features) if greedy_result.kept_features else 'none'}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _final_gate_passed(self, metrics: dict[str, float]) -> bool:
        return all(criterion.passed for criterion in self._gate_criteria(7, metrics))


def _criterion(name: str, target: float, metrics: dict[str, float]) -> GateCriterion:
    metric_name = _metric_for_gate(name)
    actual = float(metrics.get(metric_name, 0.0) or 0.0)
    lower_is_better = name.startswith(("max_", "hard_", "target_")) or "drawdown" in name or "drag" in name
    passed = actual <= target if lower_is_better else actual >= target
    return GateCriterion(name=name, target=float(target), actual=actual, passed=passed)


def _with_score_metrics(metrics: dict[str, float], score) -> dict[str, float]:
    return {
        **metrics,
        **{f"score_{key}": value for key, value in score.components.items()},
        "score_total": score.total,
    }


def _metric_for_gate(name: str) -> str:
    mapping = {
        "min_active_strategies": "active_strategy_count",
        "min_active_trades_per_month": "active_trades_per_month",
        "min_total_r_per_month": "total_r_per_month",
        "min_profit_factor": "profit_factor",
        "target_max_drawdown_pct": "max_drawdown_pct",
        "hard_max_drawdown_pct": "max_drawdown_pct",
        "max_single_strategy_risk_share": "max_strategy_trade_share",
        "max_overlay_equity_pct": "overlay_max_equity_pct",
        "max_overlay_score_drag_pct": "overlay_score_drag_pct",
        "min_idle_cash_deployment_share": "idle_cash_deployment_share",
        "max_strategy_quality_regression_pct": "strategy_quality_regression_pct",
        "max_positive_alpha_block_rate": "positive_alpha_block_rate",
        "max_symbol_directional_heat_R": "max_symbol_directional_heat_R",
        "min_trade_capture_ratio": "trade_capture_ratio",
        "min_positive_years": "positive_years",
    }
    return mapping.get(name, name)
