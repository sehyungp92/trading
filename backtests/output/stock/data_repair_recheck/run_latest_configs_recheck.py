from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRADING_REQUIRE_FROZEN_DATA", "false")

from backtests.shared.auto.phase_state import load_phase_state
from backtests.shared.auto.plugin_utils import greedy_result_from_state
from backtests.shared.auto.round_manager import canonicalize_metrics
from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis
from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from backtests.stock.analysis.iaric_pullback_diagnostics import (
    compute_pullback_diagnostic_snapshot,
    pullback_full_diagnostic,
)
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin
from backtests.stock.auto.alcb.phase_scoring import merge_alcb_metrics
from backtests.stock.auto.alcb.run_final_diagnostics import FLOOR_NAME_MAP
from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.config_mutator import mutate_alcb_config, mutate_iaric_config
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin, _merge_snapshot_gate_metrics
from backtests.stock.auto.iaric.phase_scoring import enrich_phase_score_metrics, merge_pullback_metrics
from backtests.stock.auto.scoring import extract_metrics
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.data.replay_cache import load_research_replay_bundle
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine


DATA_DIR = REPO_ROOT / "backtests" / "stock" / "data" / "raw"
BASE_OUTPUT = REPO_ROOT / "backtests" / "output" / "stock"
RUN_ROOT = BASE_OUTPUT / "data_repair_recheck"

RUNS = (
    {
        "strategy": "iaric",
        "round_dir": BASE_OUTPUT / "iaric" / "round_1",
        "baseline_summary": BASE_OUTPUT / "iaric" / "round_1" / "run_summary.json",
        "start": "2024-03-25",
        "end": "2026-03-01",
        "label": "common_start_saved_end",
        "full_diagnostics": True,
    },
    {
        "strategy": "alcb",
        "round_dir": BASE_OUTPUT / "alcb" / "round_2",
        "baseline_summary": BASE_OUTPUT / "alcb" / "round_2" / "run_summary.json",
        "start": "2024-03-25",
        "end": "2026-03-01",
        "label": "common_start_saved_end",
        "full_diagnostics": True,
    },
)

METRIC_KEYS = (
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "gross_profit",
    "gross_loss",
    "net_profit",
    "profit_factor",
    "expectancy",
    "expectancy_dollar",
    "avg_r",
    "expected_total_r",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown_pct",
    "max_drawdown_dollar",
    "avg_hold_hours",
    "trades_per_month",
    "total_commissions",
    "tail_loss_r",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return str(value)


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(metrics.get(key)) for key in METRIC_KEYS if key in metrics}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    payload = {
        "strategy": trade.strategy,
        "symbol": trade.symbol,
        "direction": getattr(trade.direction, "name", str(trade.direction)),
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "pnl": trade.pnl,
        "pnl_net": trade.pnl_net,
        "r_multiple": trade.r_multiple,
        "risk_per_share": trade.risk_per_share,
        "commission": trade.commission,
        "slippage": trade.slippage,
        "entry_type": trade.entry_type,
        "exit_reason": trade.exit_reason,
        "sector": trade.sector,
        "regime_tier": trade.regime_tier,
        "hold_bars": trade.hold_bars,
        "max_favorable": trade.max_favorable,
        "max_adverse": trade.max_adverse,
        "signal_time": trade.signal_time,
        "fill_time": trade.fill_time,
        "signal_bar_index": trade.signal_bar_index,
        "fill_bar_index": trade.fill_bar_index,
        "reentry_sequence": trade.reentry_sequence,
        "metadata": trade.metadata,
    }
    return _json_safe(payload)


def _monthly_summary(trades: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        buckets[trade.exit_time.strftime("%Y-%m")].append(trade)
    rows = []
    for month in sorted(buckets):
        group = buckets[month]
        total_r = sum(float(t.r_multiple) for t in group)
        pnl = sum(float(t.pnl_net) for t in group)
        wins = sum(1 for t in group if float(t.pnl_net) > 0.0)
        rows.append(
            {
                "month": month,
                "trades": len(group),
                "win_rate": wins / len(group) if group else 0.0,
                "avg_r": total_r / len(group) if group else 0.0,
                "total_r": total_r,
                "pnl_net": pnl,
            }
        )
    return rows


def _symbol_summary(trades: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        buckets[str(trade.symbol)].append(trade)
    rows = []
    for symbol in sorted(buckets):
        group = buckets[symbol]
        total_r = sum(float(t.r_multiple) for t in group)
        pnl = sum(float(t.pnl_net) for t in group)
        wins = sum(1 for t in group if float(t.pnl_net) > 0.0)
        rows.append(
            {
                "symbol": symbol,
                "trades": len(group),
                "win_rate": wins / len(group) if group else 0.0,
                "avg_r": total_r / len(group) if group else 0.0,
                "total_r": total_r,
                "pnl_net": pnl,
            }
        )
    rows.sort(key=lambda item: (-abs(float(item["pnl_net"])), item["symbol"]))
    return rows


def _exit_summary(trades: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        buckets[str(trade.exit_reason or "UNKNOWN")].append(trade)
    rows = []
    total = len(trades)
    for reason in sorted(buckets):
        group = buckets[reason]
        total_r = sum(float(t.r_multiple) for t in group)
        pnl = sum(float(t.pnl_net) for t in group)
        wins = sum(1 for t in group if float(t.pnl_net) > 0.0)
        rows.append(
            {
                "exit_reason": reason,
                "trades": len(group),
                "share": len(group) / total if total else 0.0,
                "win_rate": wins / len(group) if group else 0.0,
                "avg_r": total_r / len(group) if group else 0.0,
                "total_r": total_r,
                "pnl_net": pnl,
            }
        )
    rows.sort(key=lambda item: (-item["trades"], item["exit_reason"]))
    return rows


def _trade_identity_set(trades: list[Any]) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(t.symbol),
            t.entry_time.isoformat(),
            t.exit_time.isoformat(),
            str(t.exit_reason or ""),
        )
        for t in trades
    }


def _diff_metrics(new_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        if key not in new_metrics and key not in baseline_metrics:
            continue
        new = new_metrics.get(key)
        old = baseline_metrics.get(key)
        row = {"baseline": _json_safe(old), "new": _json_safe(new), "delta": None, "delta_pct": None}
        try:
            new_f = float(new)
            old_f = float(old)
            row["delta"] = new_f - old_f
            row["delta_pct"] = ((new_f - old_f) / old_f) if old_f else None
        except (TypeError, ValueError):
            pass
        diff[key] = row
    return diff


def _hydrate_alcb_final_phase_runtime_context(plugin: ALCBP16Plugin, state) -> int:
    final_phase = max(state.completed_phases) if state.completed_phases else plugin.num_phases
    plugin._replay_bundle()
    phase_result = state.phase_results.get(final_phase, {})
    phase_gate = state.phase_gate_results.get(final_phase, {})
    plugin._phase_runtime_context[final_phase] = {
        "base_metrics": dict(phase_result.get("final_metrics", {})),
        "hard_rejects": {
            FLOOR_NAME_MAP.get(criterion["name"], criterion["name"]): criterion["target"]
            for criterion in phase_gate.get("criteria", [])
            if isinstance(criterion, dict) and "name" in criterion and "target" in criterion
        },
    }
    return final_phase


def _make_plugin(strategy: str, state, start: str, end: str):
    if strategy == "iaric":
        return IARICPullbackPlugin(
            DATA_DIR,
            start_date=start,
            end_date=end,
            initial_equity=10_000.0,
            max_workers=1,
            num_phases=max(state.completed_phases) if state.completed_phases else 1,
            profile="mainline",
            round_name=state.round_name or "live_aligned_ablation_perturbation",
        )
    if strategy == "alcb":
        return ALCBP16Plugin(
            DATA_DIR,
            start_date=start,
            end_date=end,
            initial_equity=10_000.0,
            max_workers=1,
        )
    raise ValueError(strategy)


def _run_strategy_context(
    strategy: str,
    mutations: dict[str, Any],
    *,
    start: str,
    end: str,
    full_diagnostics: bool,
) -> dict[str, Any]:
    replay_bundle = load_research_replay_bundle(DATA_DIR, require_bundle=False)
    replay = replay_bundle.data
    if strategy == "iaric":
        config = mutate_iaric_config(
            IARICBacktestConfig(
                start_date=start,
                end_date=end,
                initial_equity=10_000.0,
                tier=3,
                data_dir=DATA_DIR,
            ),
            mutations,
        )
        result = IARICPullbackEngine(config, replay, collect_diagnostics=full_diagnostics).run()
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
        diagnostic_snapshot = compute_pullback_diagnostic_snapshot(
            result.trades,
            metrics=metrics,
            replay=replay,
            daily_selections=result.daily_selections,
            candidate_ledger=result.candidate_ledger,
            funnel_counters=result.funnel_counters,
            rejection_log=result.rejection_log,
            shadow_outcomes=result.shadow_outcomes,
            selection_attribution=result.selection_attribution,
            fsm_log=result.fsm_log,
        )
        metrics = _merge_snapshot_gate_metrics(metrics, diagnostic_snapshot)
        return {
            "metrics": metrics,
            "trades": result.trades,
            "replay": replay,
            "daily_selections": result.daily_selections,
            "candidate_ledger": result.candidate_ledger,
            "funnel_counters": result.funnel_counters,
            "rejection_log": result.rejection_log,
            "shadow_outcomes": result.shadow_outcomes,
            "selection_attribution": result.selection_attribution,
            "fsm_log": result.fsm_log,
            "config": config,
            "diagnostic_snapshot": diagnostic_snapshot,
            "cache_source_fingerprint": replay_bundle.cache_source_fingerprint,
            "metrics_cache_key": None,
        }
    if strategy == "alcb":
        hydrated = hydrate_time_mutations(mutations)
        config = mutate_alcb_config(
            ALCBBacktestConfig(
                start_date=start,
                end_date=end,
                initial_equity=10_000.0,
                tier=2,
                data_dir=DATA_DIR,
            ),
            hydrated,
        )
        engine = ALCBIntradayEngine(config, replay)
        shadow_tracker = ALCBShadowTracker() if full_diagnostics else None
        if shadow_tracker is not None:
            engine.shadow_tracker = shadow_tracker
        result = engine.run()
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
        metrics = merge_alcb_metrics(perf, result.trades)
        return {
            "metrics": metrics,
            "trades": result.trades,
            "replay": replay,
            "daily_selections": result.daily_selections,
            "shadow_tracker": shadow_tracker,
            "config": config,
            "cache_source_fingerprint": replay_bundle.cache_source_fingerprint,
            "metrics_cache_key": None,
        }
    raise ValueError(strategy)


def _render_full_diagnostics(strategy: str, plugin: Any, final_phase: int, state: Any, metrics: dict[str, Any], final_greedy: Any, ctx: dict[str, Any]) -> str:
    snapshot = plugin.run_phase_diagnostics(final_phase, state, metrics, final_greedy)
    if strategy == "iaric":
        return snapshot + "\n\n" + pullback_full_diagnostic(
            ctx["trades"],
            metrics=metrics,
            replay=ctx.get("replay"),
            daily_selections=ctx.get("daily_selections"),
            candidate_ledger=ctx.get("candidate_ledger"),
            funnel_counters=ctx.get("funnel_counters"),
            rejection_log=ctx.get("rejection_log"),
            shadow_outcomes=ctx.get("shadow_outcomes"),
            selection_attribution=ctx.get("selection_attribution"),
            fsm_log=ctx.get("fsm_log"),
            diagnostic_snapshot=ctx.get("diagnostic_snapshot"),
        )
    max_positions = int(ctx["config"].param_overrides.get("max_positions", 10))
    return "\n\n".join(
        [
            snapshot,
            alcb_full_diagnostic(
                ctx["trades"],
                shadow_tracker=ctx.get("shadow_tracker"),
                daily_selections=ctx.get("daily_selections"),
            ),
            qe_replacement_analysis(ctx["trades"], max_positions=max_positions),
        ]
    )


def _run_one(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    strategy = spec["strategy"]
    label = spec["label"]
    round_dir = Path(spec["round_dir"])
    phase_state_path = round_dir / "phase_state.json"
    state = load_phase_state(phase_state_path)
    baseline = _load_json(Path(spec["baseline_summary"]))
    plugin = _make_plugin(strategy, state, spec["start"], spec["end"])
    if strategy == "alcb":
        final_phase = _hydrate_alcb_final_phase_runtime_context(plugin, state)
    else:
        final_phase = max(state.completed_phases) if state.completed_phases else 1

    started = time.monotonic()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] running {strategy}/{label} {spec['start']}->{spec['end']}", flush=True)
    full_diagnostics = bool(spec.get("full_diagnostics"))
    final_ctx = _run_strategy_context(
        strategy,
        state.cumulative_mutations,
        start=spec["start"],
        end=spec["end"],
        full_diagnostics=full_diagnostics,
    )
    metrics = dict(final_ctx["metrics"])
    trades = list(final_ctx["trades"])
    final_greedy = greedy_result_from_state(state, phase=final_phase, final_metrics=metrics)
    if full_diagnostics:
        diagnostics_text = _render_full_diagnostics(strategy, plugin, final_phase, state, metrics, final_greedy, final_ctx)
    else:
        diagnostics_text = plugin.run_phase_diagnostics(final_phase, state, metrics, final_greedy)
    elapsed = time.monotonic() - started

    prefix = f"{strategy}_{label}"
    (out_dir / f"{prefix}_round_final_diagnostics.txt").write_text(diagnostics_text, encoding="utf-8")
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(_metric_subset(metrics), indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_trades.json").write_text(json.dumps([_trade_to_dict(t) for t in trades], indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_monthly.json").write_text(json.dumps(_monthly_summary(trades), indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_symbols.json").write_text(json.dumps(_symbol_summary(trades), indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_exits.json").write_text(json.dumps(_exit_summary(trades), indent=2), encoding="utf-8")

    first_entry = min((t.entry_time for t in trades), default=None)
    last_exit = max((t.exit_time for t in trades), default=None)
    entry_dates = Counter(t.entry_time.date().isoformat() for t in trades)
    result = {
        "strategy": strategy,
        "label": label,
        "requested_start": spec["start"],
        "requested_end": spec["end"],
        "phase_state": str(phase_state_path),
        "completed_phases": list(state.completed_phases),
        "round_name": state.round_name,
        "mutation_count": len(state.cumulative_mutations),
        "cache_source_fingerprint": final_ctx.get("cache_source_fingerprint"),
        "metrics_cache_key": final_ctx.get("metrics_cache_key"),
        "elapsed_seconds": elapsed,
        "full_diagnostics": full_diagnostics,
        "first_trade_entry": first_entry.isoformat() if first_entry else None,
        "last_trade_exit": last_exit.isoformat() if last_exit else None,
        "trade_entry_date_count": len(entry_dates),
        "metrics": _metric_subset(metrics),
        "headline_metrics": canonicalize_metrics(metrics),
        "baseline_headline_metrics": baseline.get("headline_metrics", {}),
        "baseline_final_metrics_subset": _metric_subset(baseline.get("final_metrics", {})),
        "diff_vs_baseline_final_metrics": _diff_metrics(metrics, baseline.get("final_metrics", {})),
        "artifacts": {
            "diagnostics": str(out_dir / f"{prefix}_round_final_diagnostics.txt"),
            "metrics": str(out_dir / f"{prefix}_metrics.json"),
            "trades": str(out_dir / f"{prefix}_trades.json"),
            "monthly": str(out_dir / f"{prefix}_monthly.json"),
            "symbols": str(out_dir / f"{prefix}_symbols.json"),
            "exits": str(out_dir / f"{prefix}_exits.json"),
        },
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] done {strategy}/{label}: "
        f"trades={int(metrics.get('total_trades', 0))} net={float(metrics.get('net_profit', 0.0)):+.2f} "
        f"PF={float(metrics.get('profit_factor', 0.0)):.3f} DD={float(metrics.get('max_drawdown_pct', 0.0)):.2%} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def main() -> None:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUN_ROOT / f"latest_configs_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    error: dict[str, Any] | None = None
    try:
        for spec in RUNS:
            results.append(_run_one(spec, out_dir))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        by_strategy: dict[str, dict[str, Any]] = defaultdict(dict)
        for result in results:
            by_strategy[result["strategy"]][result["label"]] = result
        comparisons: dict[str, Any] = {}
        for strategy, strategy_runs in by_strategy.items():
            saved = strategy_runs.get("saved_window")
            updated = strategy_runs.get("proper_updated_window")
            corrected = strategy_runs.get("common_start_saved_end")
            if saved and updated:
                saved_trades = _load_json(Path(saved["artifacts"]["trades"]))
                updated_trades = _load_json(Path(updated["artifacts"]["trades"]))
                saved_ids = {
                    (t["symbol"], t["entry_time"], t["exit_time"], t.get("exit_reason") or "")
                    for t in saved_trades
                }
                updated_ids = {
                    (t["symbol"], t["entry_time"], t["exit_time"], t.get("exit_reason") or "")
                    for t in updated_trades
                }
                comparisons[strategy] = {
                    "proper_updated_vs_saved_window_metrics": _diff_metrics(
                        updated["metrics"],
                        saved["metrics"],
                    ),
                    "same_trade_identity_count": len(saved_ids & updated_ids),
                    "saved_window_trade_count": len(saved_ids),
                    "proper_updated_window_trade_count": len(updated_ids),
                    "added_trade_identity_count": len(updated_ids - saved_ids),
                    "removed_trade_identity_count": len(saved_ids - updated_ids),
                }
            if corrected:
                comparisons[strategy] = {
                    **comparisons.get(strategy, {}),
                    "common_start_saved_end_vs_existing_baseline_metrics": corrected.get(
                        "diff_vs_baseline_final_metrics", {}
                    ),
                }
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out_dir),
            "data_dir": str(DATA_DIR),
            "runs": results,
            "comparisons": comparisons,
            "error": error,
        }
        (out_dir / "recheck_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
        print(f"OUTPUT_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()
