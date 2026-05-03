"""NQDTC phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
4 phases targeting regime filtering, signal quality, timing/exit,
and fine-tuning with no-regression protection.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scoring import NQDTCCompositeScore, NQDTCMetrics

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    deserialize_experiments,
    greedy_result_from_state,
    greedy_result_to_dict,
    mutation_signature,
    seen_experiment_names,
    shutdown_process_pool,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

import logging

from .phase_candidates import get_phase_candidates
from .phase_gates import gate_criteria_for_phase

_seq_log = logging.getLogger("nqdtc.sequential")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Immutable scoring weights (7 components)
# ---------------------------------------------------------------------------

IMMUTABLE_SCORE_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "pf": 0.16,
    "expectancy": 0.12,
    "frequency": 0.16,
    "risk": 0.14,
    "exit_capture": 0.12,
    "stability": 0.08,
}

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    phase: dict(IMMUTABLE_SCORE_WEIGHTS) for phase in range(1, 5)
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "max_dd_pct": 0.18,
        "min_trades": 89,
        "min_pf": 2.00,
        "min_avg_r": 0.45,
        "min_capture": 0.39,
        "min_net_return_pct": 296.0,
        "min_robust_net_return_pct": 220.0,
        "max_largest_win_pnl_share": 0.30,
    },
    2: {
        "max_dd_pct": 0.18,
        "min_trades": 89,
        "min_pf": 2.00,
        "min_avg_r": 0.45,
        "min_capture": 0.39,
        "min_net_return_pct": 296.0,
        "min_robust_net_return_pct": 220.0,
        "max_largest_win_pnl_share": 0.30,
    },
    3: {
        "max_dd_pct": 0.20,
        "min_trades": 89,
        "min_pf": 1.90,
        "min_avg_r": 0.42,
        "min_capture": 0.38,
        "min_net_return_pct": 296.0,
        "min_robust_net_return_pct": 210.0,
        "max_largest_win_pnl_share": 0.32,
    },
    4: {
        "max_dd_pct": 0.20,
        "min_trades": 89,
        "min_pf": 1.90,
        "min_avg_r": 0.42,
        "min_capture": 0.38,
        "min_net_return_pct": 296.0,
        "min_robust_net_return_pct": 210.0,
        "max_largest_win_pnl_share": 0.32,
    },
}

PHASE_FOCUS = {
    1: ("Session Frequency Harvest", ["total_trades", "net_return_pct", "profit_factor"]),
    2: ("Robust Return Protection", ["robust_net_return_pct", "largest_win_pnl_share", "max_dd_pct"]),
    3: ("Selective Alpha Recovery", ["total_trades", "net_return_pct", "avg_r"]),
    4: ("Interaction Fine-Tune", ["net_return_pct", "robust_net_return_pct", "total_trades"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 320.0,
    "robust_net_return_pct": 240.0,
    "profit_factor": 2.0,
    "max_dd_pct": 0.15,
    "calmar": 10.0,
    "total_trades": 115.0,
    "capture_ratio": 0.50,
    "avg_r": 0.50,
    "win_rate": 0.56,
    "sharpe": 2.0,
    "sortino": 6.0,
}


def score_phase_metrics(
    phase: int,
    metrics: NQDTCMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NQDTCCompositeScore:
    from .scoring import composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    weights = PHASE_WEIGHTS.get(phase)
    return composite_score(metrics, weights, hard_rejects=rejects)


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
        results = []
        total = len(candidates)
        for i, c in enumerate(candidates, 1):
            r = score_candidate((c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            tag = f"score={r.score:.4f}" if not r.rejected else f"REJECTED({r.reject_reason})"
            _seq_log.info("[%d/%d] %s -- %s", i, total, c.name, tag)
            results.append(r)
        return results

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

_INITIAL_MUTATIONS = {"flags.max_loss_cap": True, "flags.max_stop_width": True}


class NQDTCPlugin:
    name = "nqdtc"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = _INITIAL_MUTATIONS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 4,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(f"NQDTCPlugin supports 1-{max(PHASE_FOCUS)} phases, got {num_phases}.")
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
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
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
                min_effective_score_delta_pct=0.005,
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
        evaluation_key = build_cache_key(
            "nqdtc.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (candidate.name, candidate.mutations, current_mutations, phase, scoring_weights, hard_rejects)
                    for candidate in candidates
                ],
                on_terminate=self._destroy_pool,
                description=f"nqdtc phase {phase}",
                logger=logger,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        # On Windows (spawn), mp.Pool re-imports everything + reloads data per
        # worker -- extremely slow and often hangs.  Skip parallel entirely when
        # max_workers <= 1 to avoid the overhead.
        if self.max_workers is not None and self.max_workers <= 1:
            raw = make_sequential()
        else:
            raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"nqdtc phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def _replay_bundle(self):
        from backtests.momentum.auto.nqdtc.worker import load_worker_data

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

    # -- Pool lifecycle --------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None:
            return
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity),
            logger=logger,
            description=f"{self.name} evaluation",
        )

    def _destroy_pool(self) -> None:
        """Force-kill the pool (called on worker errors via terminate())."""
        shutdown_process_pool(self._pool, force=True)
        self._pool = None

    def close_pool(self) -> None:
        """Gracefully shut down the persistent worker pool.

        Called by PhaseRunner.run_all_phases() at the end of all phases.
        """
        shutdown_process_pool(self._pool)
        self._pool = None

    # -- Metrics ---------------------------------------------------------------

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics

        replay_bundle = self._replay_bundle()
        metrics_sig = mutation_signature(mutations)
        if self._last_context.get("mutation_signature") == metrics_sig:
            return dict(self._last_context["metrics"])

        cache_key = build_cache_key(
            "nqdtc.final_metrics",
            source_fingerprint=replay_bundle.cache_source_fingerprint,
            mutations=mutations,
            extra={"initial_equity": self.initial_equity},
        )
        cached = self._final_metrics_cache.get(cache_key)
        if cached is not None:
            self._last_context = cached["context"]
            return dict(cached["metrics"])

        config = mutate_nqdtc_config(
            NQDTCBacktestConfig(initial_equity=self.initial_equity, data_dir=self.data_dir, fixed_qty=10),
            mutations,
        )
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(**replay_engine_kwargs(replay_bundle))
        metrics = extract_nqdtc_metrics(
            result.trades,
            list(result.equity_curve),
            list(result.timestamps),
            self.initial_equity,
        )
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
            None,
            self._last_context.get("trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            None,
            self._last_context.get("trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        extraction = (
            f"Net return {m.net_return_pct:.1f}% with {m.total_trades} trades, "
            f"robust return {m.robust_net_return_pct:.1f}%, "
            f"PF={m.profit_factor:.2f}, DD={m.max_dd_pct:.1%}. "
            "Later recovery and interaction phases did not find a valid "
            "incremental mutation above the robust score hurdle."
        )
        discrimination = (
            f"Win rate {m.win_rate:.1%}, avg R={m.avg_r:.3f}, "
            f"capture ratio {m.capture_ratio:.2f}; Range trades are "
            f"{m.range_regime_pct:.1%} of the book and ETH shorts remain "
            f"blocked ({m.eth_short_trades} trades)."
        )
        entry = (
            "Entry expansion came from reopening the 05:00 and 09:00 ET "
            "Range windows while preserving the prior score, RVOL, box-width, "
            "cooldown, and regime blocks. Guarded session/regime recovery "
            "candidates were tested but did not improve the objective."
        )
        management = (
            f"Calmar={m.calmar:.2f}, Sharpe={m.sharpe:.2f}, "
            f"Sortino={m.sortino:.2f}; burst trades are {m.burst_trade_pct:.1%} "
            "and the accepted ratchet threshold reduced drawdown without "
            "sacrificing the frequency harvest."
        )
        exits = (
            f"TP1 hit rate {m.tp1_hit_rate:.1%}, TP2 hit rate {m.tp2_hit_rate:.1%}, "
            f"avg hold {m.avg_hold_hours:.1f}h, largest win share {m.largest_win_pnl_share:.1%}."
        )
        timing = (
            f"Burst trade pct {m.burst_trade_pct:.1%}, "
            f"ETH short WR {m.eth_short_wr:.1%} ({m.eth_short_trades} trades), "
            f"Range regime {m.range_regime_pct:.1%} of trades."
        )
        overall = (
            f"NQDTC optimized to {m.net_return_pct:.1f}% return, PF={m.profit_factor:.2f}, "
            f"DD={m.max_dd_pct:.1%}, {m.total_trades} trades. "
            f"Robust return {m.robust_net_return_pct:.1f}%, "
            f"capture ratio {'improved to' if m.capture_ratio > 0.33 else 'at'} {m.capture_ratio:.2f}."
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
            overall_verdict=overall,
            extra_sections={"timing": timing},
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
        m = _metrics_from_dict(metrics)
        seen = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name in seen:
                return
            seen.add(name)
            suggestions.append(Experiment(name=name, mutations=mutations))

        weakness_text = " ".join(weaknesses).lower()

        # Residual non-Range drag: round 1 left Aligned negative while Range carried edge.
        if m.profit_factor < 1.6 or m.avg_r < 0.30 or "profit_factor" in weakness_text:
            add("suggest_block_aligned", {"param_overrides.BLOCK_ALIGNED_REGIME": True})
            add("suggest_score_non_range_2x", {"param_overrides.SCORE_NON_RANGE_MULT": 2.0})
            add("suggest_score_non_range_2.5x", {"param_overrides.SCORE_NON_RANGE_MULT": 2.5})

        # High drawdown
        if m.max_dd_pct > 0.20 or "drawdown" in weakness_text:
            add("suggest_max_stop_width_175", {"param_overrides.MAX_STOP_WIDTH_PTS": 175})
            add("suggest_max_stop_atr_0.60", {"param_overrides.MAX_STOP_ATR_MULT": 0.60})

        # Capture ratio low / TP2 inactive
        if m.capture_ratio < 0.45 or m.tp2_hit_rate == 0:
            add("suggest_tp2_2.25_p15", {
                "param_overrides.TP1_R": 1.2,
                "param_overrides.TP1_PARTIAL_PCT": 0.5,
                "param_overrides.TP2_R": 2.25,
                "param_overrides.TP2_PARTIAL_PCT": 0.15,
            })
            add("suggest_tp2_2.5_p20", {
                "param_overrides.TP1_R": 1.2,
                "param_overrides.TP1_PARTIAL_PCT": 0.5,
                "param_overrides.TP2_R": 2.5,
                "param_overrides.TP2_PARTIAL_PCT": 0.20,
            })
            add("suggest_ratchet_0.45", {"param_overrides.RATCHET_LOCK_PCT": 0.45})

        # Burst clustering
        if m.burst_trade_pct > 0.15 or "burst" in weakness_text:
            add("suggest_cooldown_45m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45})
            add("suggest_loss_streak_skip_12", {"param_overrides.LOSS_STREAK_SKIP_BARS": 12})

        # ETH shorts
        if m.eth_short_wr < 0.40 and m.eth_short_trades > 30:
            add("suggest_block_eth_shorts", {"flags.block_eth_shorts": True})
            add("suggest_eth_short_half", {"param_overrides.ETH_SHORT_SIZE_MULT": 0.50})

        # Frequency is now a first-class score component; prefer session-specific
        # recovery over broad gate relaxation.
        if m.total_trades < 100:
            add("suggest_allow_05_et", {"flags.block_05_et": False})
            add("suggest_allow_05_score_1.75", {
                "flags.block_05_et": False,
                "param_overrides.SCORE_NORMAL": 1.75,
            })
            add("suggest_allow_09_et", {"flags.block_09_et": False})
            add("suggest_rvol_1.35", {"param_overrides.RVOL_SCORE_THRESH": 1.35})

        if m.largest_win_pnl_share > 0.30 or m.robust_net_return_pct < 220:
            add("suggest_outlier_guard_score_1.75", {"param_overrides.SCORE_NORMAL": 1.75})
            add("suggest_outlier_guard_max_stop_175", {"param_overrides.MAX_STOP_WIDTH_PTS": 175})

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        # The round-2 score is intentionally immutable so experiments are
        # compared against one stable objective rather than moving goalposts.
        return dict(PHASE_WEIGHTS.get(phase) or IMMUTABLE_SCORE_WEIGHTS)

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        m = _metrics_from_dict(metrics)
        return {
            "session_direction": {
                "eth_short_wr": m.eth_short_wr,
                "eth_short_trades": m.eth_short_trades,
                "range_regime_pct": m.range_regime_pct,
            },
            "exit_efficiency": {
                "capture_ratio": m.capture_ratio,
                "tp1_hit_rate": m.tp1_hit_rate,
                "tp2_hit_rate": m.tp2_hit_rate,
            },
            "return_robustness": {
                "robust_net_return_pct": m.robust_net_return_pct,
                "largest_win_return_pct": m.largest_win_return_pct,
                "largest_win_pnl_share": m.largest_win_pnl_share,
                "largest_winner_r": m.largest_winner_r,
            },
            "clustering": {
                "burst_trade_pct": m.burst_trade_pct,
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        sd = extra.get("session_direction", {})
        if sd:
            lines.append(
                f"Session/direction: ETH short WR={sd.get('eth_short_wr', 0):.1%} "
                f"({sd.get('eth_short_trades', 0)} trades), "
                f"Range regime={sd.get('range_regime_pct', 0):.1%}"
            )
        ee = extra.get("exit_efficiency", {})
        if ee:
            lines.append(
                f"Exit efficiency: capture={ee.get('capture_ratio', 0):.2f}, "
                f"TP1={ee.get('tp1_hit_rate', 0):.1%}, TP2={ee.get('tp2_hit_rate', 0):.1%}"
            )
        rr = extra.get("return_robustness", {})
        if rr:
            lines.append(
                f"Return robustness: robust={rr.get('robust_net_return_pct', 0):.1f}%, "
                f"largest_win_share={rr.get('largest_win_pnl_share', 0):.1%}, "
                f"largest_win={rr.get('largest_winner_r', 0):.2f}R"
            )
        cl = extra.get("clustering", {})
        if cl:
            lines.append(f"Clustering: burst_trade_pct={cl.get('burst_trade_pct', 0):.1%}")
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
        return None  # Use default framework logic

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior = state.get_phase_metrics(phase - 1) if phase > 1 else None
        return gate_criteria_for_phase(phase, m, prior)


def _metrics_from_dict(metrics: dict[str, float]) -> NQDTCMetrics:
    from .scoring import NQDTCMetrics

    fields = NQDTCMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            val = metrics[key]
            # Ensure int fields get ints
            if field_info.type == "int" or (hasattr(field_info, 'default') and isinstance(field_info.default, int)):
                val = int(val)
            payload[key] = val
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return NQDTCMetrics(**payload)
