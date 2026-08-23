from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import numpy as np

from backtests.stock.auto.scoring import extract_metrics
from backtests.shared.auto.types import ScoredCandidate
from strategies.stock.iaric.core.lanes import lane_id_for_route

from .phase_scoring import (
    enrich_phase_score_metrics,
    merge_pullback_metrics,
    score_pullback_phase,
    score_v2r1_pullback_phase,
    score_v2r2_pullback_phase,
    score_v2r3_pullback_phase,
    score_v2r4_pullback_phase,
    score_v3r1_pullback_phase,
    score_v4r1_pullback_phase,
    score_v5r1_pullback_phase,
    score_v5r2_pullback_phase,
    score_v6r1_pullback_phase,
)

_worker_replay = None
_worker_config = None
_worker_equity: float = 0.0
_worker_phase: int = 0
_worker_hard_rejects: dict | None = None
_worker_scoring_weights: dict | None = None
_worker_round_name: str = "r4"


def init_worker(
    data_dir_str: str,
    start_date: str,
    end_date: str,
    equity: float,
    phase: int = 0,
    hard_rejects: dict | None = None,
    scoring_weights: dict | None = None,
    round_name: str = "r4",
    bundle_path_str: str | None = None,
    require_bundle: bool | None = None,
) -> None:
    global _worker_replay, _worker_config, _worker_equity, _worker_phase
    global _worker_hard_rejects, _worker_scoring_weights, _worker_round_name

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = Path(data_dir_str)
    if bundle_path_str is None and require_bundle is None:
        _worker_replay = load_research_replay_bundle(data_dir).data
    else:
        _worker_replay = load_research_replay_bundle(
            data_dir,
            bundle_path=Path(bundle_path_str) if bundle_path_str else None,
            require_bundle=require_bundle,
        ).data
    _worker_equity = equity
    _worker_phase = phase
    _worker_hard_rejects = hard_rejects or {}
    _worker_scoring_weights = scoring_weights or {}
    _worker_round_name = round_name
    _worker_config = IARICBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=equity,
        tier=3,
        data_dir=data_dir,
    )


def score_candidate(args: tuple[str, dict, dict]) -> ScoredCandidate:
    name, candidate_muts, base_muts = args

    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_iaric_config(_worker_config, all_muts)
        result = IARICPullbackEngine(config, _worker_replay, collect_diagnostics=True).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )
        avg_r = float(np.mean([float(t.r_multiple) for t in result.trades])) if result.trades else 0.0
        merged_metrics = merge_pullback_metrics(
            perf,
            result.trades,
            candidate_ledger=result.candidate_ledger,
            selection_attribution=result.selection_attribution,
        )
        reject_reason = _phase_reject_reason(
            perf,
            _worker_hard_rejects,
            avg_r=avg_r,
            phase_metrics=merged_metrics,
        )
        if reject_reason:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=reject_reason,
                metrics=merged_metrics,
            )
        if _worker_round_name == "v6r1":
            score_fn = score_v6r1_pullback_phase
        elif _worker_round_name == "v5r2":
            score_fn = score_v5r2_pullback_phase
        elif _worker_round_name == "v5r1":
            score_fn = score_v5r1_pullback_phase
        elif _worker_round_name == "v4r1":
            score_fn = score_v4r1_pullback_phase
        elif _worker_round_name == "v3r1":
            score_fn = score_v3r1_pullback_phase
        elif _worker_round_name == "v2r4":
            score_fn = score_v2r4_pullback_phase
        elif _worker_round_name == "v2r3":
            score_fn = score_v2r3_pullback_phase
        elif _worker_round_name == "v2r2":
            score_fn = score_v2r2_pullback_phase
        elif _worker_round_name == "v2r1":
            score_fn = score_v2r1_pullback_phase
        else:
            score_fn = score_pullback_phase
        score = score_fn(_worker_phase, merged_metrics, _worker_scoring_weights)

        return ScoredCandidate(
            name=name,
            score=score,
            metrics=merged_metrics,
        )

    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())


def evaluate_candidate_metrics(args: tuple[str, dict, dict]) -> dict:
    """Evaluate one recovery candidate without retaining heavyweight diagnostics.

    This worker entry point is intentionally separate from ``score_candidate``.
    Baseline recovery needs raw, execution-corrected economics for many archived
    configurations and chronological folds; phase-specific rejection and scoring
    would make those results incomparable.  Each spawned worker still reuses one
    source-fingerprinted replay bundle and the normal IARIC mutator/engine path.
    """

    name, candidate_muts, base_muts = args
    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)
        config = mutate_iaric_config(_worker_config, all_muts)
        result = IARICPullbackEngine(config, _worker_replay, collect_diagnostics=False).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
        return {
            "name": name,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))
            },
            "error": "",
        }
    except Exception:
        return {"name": name, "metrics": {}, "error": traceback.format_exc()}


def evaluate_candidate_diagnostics(args: tuple[str, dict, dict]) -> dict:
    """Evaluate raw economics with a diagnostics-complete candidate ledger.

    Entry-opportunity scoring cannot be reconstructed from executed trades:
    it needs rejected as well as selected nightly candidates.  This worker
    retains diagnostics while the engine runs but returns only the compact
    numeric metric payload, keeping structural screens comparable without
    serializing the full ledger across process boundaries.
    """

    name, candidate_muts, base_muts = args
    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)
        config = mutate_iaric_config(_worker_config, all_muts)
        result = IARICPullbackEngine(config, _worker_replay, collect_diagnostics=True).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
        return {
            "name": name,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))
            },
            "error": "",
        }
    except Exception:
        return {"name": name, "metrics": {}, "error": traceback.format_exc()}


def _compact_trade_attribution(trades: list) -> list[dict]:
    legacy_score_component_keys = (
        "daily_signal",
        "reclaim",
        "volume",
        "vwap_hold",
        "cpr",
        "speed",
        "quality_adjustment",
    )
    compact = []
    for trade in trades:
        metadata = trade.metadata or {}
        # Preserve every component emitted by the shared entry decision.  The
        # former hard-coded legacy list made structural reversion components
        # (for example residual_dislocation and reversion_room) look like zero
        # in optimizer diagnostics even though they affected the live/replay
        # decision.  Legacy keys remain present for schema compatibility.
        score_component_keys = sorted(
            set(legacy_score_component_keys)
            | {
                str(key).removeprefix("entry_score_component_")
                for key in metadata
                if str(key).startswith("entry_score_component_")
            }
        )
        entry_bar_index = metadata.get("entry_bar_index")
        if entry_bar_index is None:
            entry_bar_index = trade.fill_bar_index
        entry_bar_index = int(entry_bar_index)
        signal_bar_index = metadata.get("signal_bar_index")
        if signal_bar_index is None:
            signal_bar_index = trade.signal_bar_index
        signal_bar_index = int(signal_bar_index)
        route = str(
            metadata.get("entry_route_family")
            or metadata.get("route_family")
            or trade.entry_type
        )
        # Hybrid trades predate TradeRecord's signal/fill fields.  The
        # executable OPEN_SCORED contract is completed bar N -> open N+1,
        # so recover the observer-only signal index from the stored fill
        # index when older records leave signal_bar_index at -1.
        if signal_bar_index < 0 and route == "OPEN_SCORED_ENTRY" and entry_bar_index > 0:
            signal_bar_index = entry_bar_index - 1
        if route in {"OPEN_SCORED_RETEST", "OPEN_SCORED_RETRACE_LIMIT"}:
            accepted_bar_index = metadata.get("accepted_bar_index")
            if accepted_bar_index is not None:
                signal_bar_index = int(accepted_bar_index)
        compact.append(
            {
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "route": route,
                "lane": str(
                    metadata.get("entry_lane_id")
                    or lane_id_for_route(
                        route,
                        rescue_candidate=bool(metadata.get("rescue_flow_candidate", False)),
                    )
                ),
                "exit_reason": trade.exit_reason,
                "r": float(trade.r_multiple),
                "pnl_net": float(trade.pnl_net),
                "entry_price": float(trade.entry_price),
                "risk_per_share": float(trade.risk_per_share),
                "signal_bar_index": signal_bar_index,
                "entry_bar_index": entry_bar_index,
                "opportunity_event_id": str(metadata.get("opportunity_event_id", "")),
                "reversion_anchor": float(metadata.get("reversion_anchor", 0.0) or 0.0),
                "structural_stop_anchor": float(
                    metadata.get("structural_stop_anchor", 0.0) or 0.0
                ),
                "initial_remaining_room_atr": float(
                    metadata.get("initial_remaining_room_atr", 0.0) or 0.0
                ),
                "prospective_reward_risk": float(
                    metadata.get("prospective_reward_risk", 0.0) or 0.0
                ),
                "daily_signal_score": float(metadata.get("daily_signal_score", 0.0) or 0.0),
                "route_score": float(metadata.get("route_score", metadata.get("intraday_score", 0.0)) or 0.0),
                "daily_signal_rank_pct": float(metadata.get("daily_signal_rank_pct", 100.0) or 100.0),
                "mfe_r": float(metadata.get("mfe_r", 0.0) or 0.0),
                "mae_r": float(metadata.get("mae_r", 0.0) or 0.0),
                "score_components": {
                    key: float(metadata.get(f"entry_score_component_{key}", 0.0) or 0.0)
                    for key in score_component_keys
                },
            }
        )
    return compact


def _evaluate_candidate_attribution(
    args: tuple[str, dict, dict],
    *,
    collect_diagnostics: bool,
) -> dict:
    """Evaluate a candidate and return compact causal trade attribution.

    Structural Phase 0 experiments need score monotonicity and entry/stop
    geometry from the executable replay, not post-hoc price substitution.  The
    payload is deliberately compact so spawned workers do not return the full
    candidate ledger or other heavyweight diagnostics.
    """

    name, candidate_muts, base_muts = args
    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)
        config = mutate_iaric_config(_worker_config, all_muts)
        result = IARICPullbackEngine(
            config,
            _worker_replay,
            collect_diagnostics=collect_diagnostics,
        ).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
        return {
            "name": name,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))
            },
            "trade_attribution": _compact_trade_attribution(result.trades),
            "funnel_counters": {
                str(key): int(value)
                for key, value in (result.funnel_counters or {}).items()
                if isinstance(value, (int, np.integer))
            },
            "error": "",
        }
    except Exception:
        return {
            "name": name,
            "metrics": {},
            "trade_attribution": [],
            "funnel_counters": {},
            "error": traceback.format_exc(),
        }


def evaluate_candidate_attribution(args: tuple[str, dict, dict]) -> dict:
    """Full signal-opportunity diagnostics for discrimination decisions."""

    return _evaluate_candidate_attribution(args, collect_diagnostics=True)


def evaluate_candidate_execution_attribution(args: tuple[str, dict, dict]) -> dict:
    """Execution-only structural screen with identical trade semantics.

    This deliberately omits rejected-opportunity diagnostics.  It is suitable
    only for Phase 0 activation/composition generation; every retained parent
    is re-evaluated by :func:`evaluate_candidate_attribution` before entry or
    promotion decisions.
    """

    return _evaluate_candidate_attribution(args, collect_diagnostics=False)


def _phase_reject_reason(
    metrics,
    hard_rejects: dict | None,
    *,
    avg_r: float | None = None,
    phase_metrics: dict | None = None,
) -> str:
    rejects = hard_rejects or {}

    min_trades = int(rejects.get("min_trades", 0))
    if metrics.total_trades < min_trades:
        return f"phase{_worker_phase}_too_few_trades ({metrics.total_trades} < {min_trades})"

    max_dd = rejects.get("max_dd_pct")
    if max_dd is not None and metrics.max_drawdown_pct > float(max_dd):
        return f"phase{_worker_phase}_max_dd ({metrics.max_drawdown_pct:.2%} > {float(max_dd):.2%})"

    min_pf = rejects.get("min_pf")
    if min_pf is not None and metrics.profit_factor < float(min_pf):
        return f"phase{_worker_phase}_low_pf ({metrics.profit_factor:.2f} < {float(min_pf):.2f})"

    min_net_profit = rejects.get("min_net_profit")
    if min_net_profit is not None and metrics.net_profit < float(min_net_profit):
        return f"phase{_worker_phase}_low_net_profit ({metrics.net_profit:.2f} < {float(min_net_profit):.2f})"

    min_sharpe = rejects.get("min_sharpe")
    if min_sharpe is not None and metrics.sharpe < float(min_sharpe):
        return f"phase{_worker_phase}_low_sharpe ({metrics.sharpe:.2f} < {float(min_sharpe):.2f})"

    min_expectancy = rejects.get("min_expectancy")
    if min_expectancy is not None and metrics.expectancy < float(min_expectancy):
        return f"phase{_worker_phase}_low_expectancy ({metrics.expectancy:.3f} < {float(min_expectancy):.3f})"

    _avg_r = avg_r if avg_r is not None else getattr(metrics, "avg_r", 0.0)

    min_avg_r_thresh = rejects.get("min_avg_r")
    if min_avg_r_thresh is not None and _avg_r < float(min_avg_r_thresh):
        return f"phase{_worker_phase}_low_avg_r ({_avg_r:.4f} < {float(min_avg_r_thresh):.4f})"

    min_expected_total_r = rejects.get("min_expected_total_r")
    if min_expected_total_r is not None:
        actual_etr = _avg_r * metrics.total_trades
        if actual_etr < float(min_expected_total_r):
            return f"phase{_worker_phase}_low_expected_total_r ({actual_etr:.2f} < {float(min_expected_total_r):.2f})"

    enriched = phase_metrics or {}
    min_robust_avg_r = rejects.get("min_robust_avg_r")
    robust_avg_r = float(enriched.get("robust_avg_r", 0.0))
    if min_robust_avg_r is not None and robust_avg_r < float(min_robust_avg_r):
        return f"phase{_worker_phase}_low_robust_avg_r ({robust_avg_r:.4f} < {float(min_robust_avg_r):.4f})"

    min_discrimination = rejects.get("min_discrimination_lift_r")
    discrimination = float(enriched.get("entry_realized_discrimination_lift_r", 0.0))
    if min_discrimination is not None and discrimination < float(min_discrimination):
        return f"phase{_worker_phase}_low_discrimination ({discrimination:.4f} < {float(min_discrimination):.4f})"

    return ""
