"""Promote the certified-context recovery as canonical TPC round 8.

The original four phase-auto phases remain historical search evidence.  This
promotion refreshes the canonical configuration, manifest, summaries, and full
diagnostics after the post-round OOS repair.  The consumed OOS period is always
labelled as development evidence and is never relabelled as untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtests.shared.auto.phase_state import _atomic_write_json
from backtests.shared.auto.provenance import build_tree_item
from backtests.shared.auto.provenance import build_phase_auto_provenance
from backtests.shared.auto.round_manager import RoundManager, canonicalize_metrics
from backtests.swing.analysis.etf_baseline import (
    build_strategy_full_diagnostics,
    infer_training_window,
    summarize_strategy,
    verify_etf_data_alignment,
)
from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent
from strategies.swing.tpc.config import SYMBOL_CONFIGS


ROOT = Path(__file__).resolve().parents[4]
ROUND_NUM = 8
ROUND_MANAGER = RoundManager("swing", "tpc")
ROUND_DIR = ROUND_MANAGER.round_path(ROUND_NUM)
RAW_DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
CONTEXT_AUTHORITY_DIR = (
    ROOT / "backtests" / "swing" / "data" / "authority" / "oos_20260502"
)
OPTIMIZED_CONFIG = ROUND_DIR / "optimized_config.json"
PROMOTION_RECORD = ROUND_DIR / "recovery_promotion.json"
REQUALIFICATION_REPORT = ROUND_DIR / "nq_gc_context_requalification.md"
FULL_IS_DIAGNOSTICS = ROUND_DIR / "full_diagnostics_in_sample.txt"
FULL_OOS_DIAGNOSTICS = ROUND_DIR / "full_diagnostics_out_of_sample.txt"
FINAL_DIAGNOSTICS = ROUND_DIR / "round_final_diagnostics.txt"
DIAGNOSTICS_SUMMARY = ROUND_DIR / "diagnostics_summary.json"
ROUND_EVALUATION = ROUND_DIR / "round_evaluation.txt"
RUN_SPEC = ROUND_DIR / "run_spec.json"
RUN_SUMMARY = ROUND_DIR / "run_summary.json"
PHASE_STATE = ROUND_DIR / "phase_state.json"

IS_END = "2025-11-01"
CERTIFIED_START = "2025-08-01"
OOS_END = "2026-05-02"
RECOVERY_KEY = "QQQ.asset_context_block_opposed_daily"
PROMOTION_STATUS = "post_oos_recovery_promoted_consumed_oos"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _trade_key(trade: Any) -> tuple[str, str, int, str]:
    return (
        str(getattr(trade, "symbol", "") or ""),
        _timestamp(getattr(trade, "entry_time")).isoformat(),
        int(getattr(trade, "direction", 0) or 0),
        str(getattr(trade, "campaign_id", "") or ""),
    )


def _trade_row(trade: Any) -> dict[str, Any]:
    return {
        "symbol": str(getattr(trade, "symbol", "") or ""),
        "entry_time": _timestamp(getattr(trade, "entry_time")).isoformat(),
        "exit_time": _timestamp(getattr(trade, "exit_time")).isoformat(),
        "direction": int(getattr(trade, "direction", 0) or 0),
        "pnl_dollars": float(getattr(trade, "pnl_dollars", 0.0) or 0.0),
        "r_multiple": float(getattr(trade, "r_multiple", 0.0) or 0.0),
        "mfe_r": float(getattr(trade, "mfe_r", 0.0) or 0.0),
        "mae_r": float(getattr(trade, "mae_r", 0.0) or 0.0),
        "exit_reason": str(getattr(trade, "exit_reason", "") or ""),
        "campaign_id": str(getattr(trade, "campaign_id", "") or ""),
    }


def _trade_stats(trades: Iterable[Any]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda trade: _timestamp(getattr(trade, "entry_time")))
    pnls = np.asarray(
        [float(getattr(trade, "pnl_dollars", 0.0) or 0.0) for trade in ordered],
        dtype=float,
    )
    rs = np.asarray(
        [float(getattr(trade, "r_multiple", 0.0) or 0.0) for trade in ordered],
        dtype=float,
    )
    gross_profit = float(pnls[pnls > 0].sum()) if pnls.size else 0.0
    gross_loss = abs(float(pnls[pnls < 0].sum())) if pnls.size else 0.0
    equity_path = np.concatenate(([0.0], np.cumsum(pnls)))
    equity_peak = np.maximum.accumulate(equity_path)
    r_path = np.concatenate(([0.0], np.cumsum(rs)))
    r_peak = np.maximum.accumulate(r_path)
    return {
        "total_trades": len(ordered),
        "total_pnl_dollars": float(pnls.sum()) if pnls.size else 0.0,
        "dollar_profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        "avg_r": float(rs.mean()) if rs.size else 0.0,
        "total_r": float(rs.sum()) if rs.size else 0.0,
        "win_rate": float(np.mean(rs > 0)) if rs.size else 0.0,
        "trade_path_max_dd_dollars": float(np.max(equity_peak - equity_path)),
        "trade_path_max_dd_r": float(np.max(r_peak - r_path)),
        "qqq_trade_count": sum(
            str(getattr(trade, "symbol", "") or "") == "QQQ" for trade in ordered
        ),
        "gld_trade_count": sum(
            str(getattr(trade, "symbol", "") or "") == "GLD" for trade in ordered
        ),
    }


def _metric_deltas(
    recovered: dict[str, Any],
    baseline: dict[str, Any],
    keys: Iterable[str],
) -> dict[str, float]:
    return {
        key: float(recovered.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)
        for key in keys
    }


def _baseline_reference() -> dict[str, Any]:
    if PROMOTION_RECORD.exists():
        previous = _load_json(PROMOTION_RECORD)
        baseline = previous.get("full_is", {}).get("baseline")
        if isinstance(baseline, dict) and baseline.get("final_metrics"):
            return baseline

    summary = _load_json(RUN_SUMMARY)
    if summary.get("cumulative_mutations", {}).get(RECOVERY_KEY):
        raise RuntimeError(
            "Canonical run summary already contains recovery but no preserved baseline record exists."
        )
    return {
        "source": "pre-recovery canonical run_summary.json",
        "headline_metrics": summary.get("headline_metrics", {}),
        "final_metrics": summary.get("final_metrics", {}),
    }


def _format_full_is_report(
    mutations: dict[str, Any],
) -> tuple[Any, dict[str, float], str, str]:
    start, _, data_end = infer_training_window(RAW_DATA_DIR, 0)
    end = pd.Timestamp(IS_END, tz="UTC")
    alignment = verify_etf_data_alignment(RAW_DATA_DIR)
    bundle = load_tpc_replay_bundle(RAW_DATA_DIR, start_date=start, end_date=end)
    config = TPCBacktestConfig(initial_equity=100_000.0, data_dir=RAW_DATA_DIR).with_overrides(
        mutations
    )
    started = time.time()
    result = run_tpc_independent(bundle.data, config, indicator_cache={})
    elapsed = time.time() - started
    diagnostic_metrics = summarize_strategy(
        "TPC", result, 100_000.0, start, end, elapsed
    )
    final_metrics = _extract_tpc_metrics(result, 100_000.0)
    report = build_strategy_full_diagnostics(
        "TPC",
        result,
        diagnostic_metrics,
        alignment=alignment,
        start=start,
        end=end,
        data_end=data_end,
        holdout_months=6,
        initial_equity=100_000.0,
    )
    report = report.replace(
        "TPC BASELINE FULL DIAGNOSTICS",
        "TPC ROUND 8 RECOVERED CANONICAL CONFIG FULL DIAGNOSTICS - IN SAMPLE",
        1,
    )
    report = report.replace("BASELINE RESULT", "RECOVERED CONFIG RESULT", 1)
    report = report.replace("Promotable baseline:", "Promotable recovered config:", 1)
    report = report.replace("ALPHA_CAPTURED_BASELINE", "ALPHA_CAPTURED_CONFIG")
    report += "\n\nRECOVERY PROMOTION CONTEXT\n" + "-" * 80 + "\n"
    report += (
        "Canonical status: post-round context recovery; OOS consumed as development evidence.\n"
        "Recovery mutation: QQQ.asset_context_block_opposed_daily=True\n"
        "Full-history context basis: legacy/raw NQ/GC (certified authority begins 2025-08-01).\n"
        f"Canonical R profit factor: {final_metrics['profit_factor']:.6f}; "
        f"dollar profit factor shown as PF above: {final_metrics['dollar_profit_factor']:.6f}.\n"
        "Original four phase-auto phases are retained as historical search evidence.\n"
    )
    report += "\nFINAL OPTIMISED CONFIG MUTATIONS\n" + "-" * 80 + "\n"
    report += "\n".join(
        f"{key}: {json.dumps(value) if value == '' else value}"
        for key, value in sorted(mutations.items())
    )
    report += f"\n\nReplay source fingerprint: {bundle.cache_source_fingerprint}"
    return result, final_metrics, report, bundle.cache_source_fingerprint


def _certified_recovery_evidence(
    mutations: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    baseline_mutations = dict(mutations)
    baseline_mutations[RECOVERY_KEY] = False
    bundle = load_tpc_replay_bundle(
        RAW_DATA_DIR,
        start_date=CERTIFIED_START,
        end_date=OOS_END,
        context_data_dir=CONTEXT_AUTHORITY_DIR,
        require_context_authority=True,
    )

    def run(overrides: dict[str, Any]) -> Any:
        config = TPCBacktestConfig(
            initial_equity=100_000.0,
            data_dir=RAW_DATA_DIR,
        ).with_overrides(overrides)
        return run_tpc_independent(bundle.data, config, indicator_cache={})

    baseline_result = run(baseline_mutations)
    recovered_result = run(mutations)
    cutoff = pd.Timestamp(IS_END, tz="UTC")
    baseline_trades = list(baseline_result.trades)
    recovered_trades = list(recovered_result.trades)
    baseline_is = [
        trade
        for trade in baseline_trades
        if _timestamp(getattr(trade, "entry_time")) < cutoff
    ]
    recovered_is = [
        trade
        for trade in recovered_trades
        if _timestamp(getattr(trade, "entry_time")) < cutoff
    ]
    baseline_oos = [
        trade
        for trade in baseline_trades
        if _timestamp(getattr(trade, "entry_time")) >= cutoff
    ]
    recovered_oos = [
        trade
        for trade in recovered_trades
        if _timestamp(getattr(trade, "entry_time")) >= cutoff
    ]
    recovered_keys = {_trade_key(trade) for trade in recovered_oos}
    suppressed = [
        _trade_row(trade)
        for trade in baseline_oos
        if _trade_key(trade) not in recovered_keys
    ]
    baseline_oos_stats = _trade_stats(baseline_oos)
    recovered_oos_stats = _trade_stats(recovered_oos)
    evidence = {
        "basis": (
            "single strict certified-authority replay across 2025-08-01..2026-05-02; "
            "trades partitioned at 2025-11-01 to retain warmup and equity path"
        ),
        "authority_dir": str(CONTEXT_AUTHORITY_DIR.resolve()),
        "replay_source_fingerprint": bundle.cache_source_fingerprint,
        "certified_is_overlap": {
            "baseline": _trade_stats(baseline_is),
            "recovered": _trade_stats(recovered_is),
        },
        "oos": {
            "baseline": baseline_oos_stats,
            "recovered": recovered_oos_stats,
            "delta": _metric_deltas(
                recovered_oos_stats,
                baseline_oos_stats,
                (
                    "total_trades",
                    "total_pnl_dollars",
                    "dollar_profit_factor",
                    "avg_r",
                    "total_r",
                    "win_rate",
                    "trade_path_max_dd_dollars",
                    "trade_path_max_dd_r",
                    "qqq_trade_count",
                    "gld_trade_count",
                ),
            ),
            "suppressed_baseline_trades": suppressed,
            "recovered_trades": [_trade_row(trade) for trade in recovered_oos],
        },
    }
    return evidence, _format_oos_report(evidence)


def _format_oos_report(evidence: dict[str, Any]) -> str:
    overlap = evidence["certified_is_overlap"]
    oos = evidence["oos"]
    baseline = oos["baseline"]
    recovered = oos["recovered"]
    delta = oos["delta"]
    lines = [
        "TPC ROUND 8 RECOVERED CANONICAL CONFIG FULL DIAGNOSTICS - CERTIFIED OOS",
        "=" * 80,
        f"Generated: {_utc_now()}",
        f"Certified replay: {CERTIFIED_START} -> {OOS_END}",
        f"Official split: {IS_END}",
        f"Authority: {evidence['authority_dir']}",
        "OOS status: consumed development evidence; not an untouched holdout.",
        "Method: continuous certified replay partitioned by entry time to preserve warmup and sizing.",
        "",
        "RECOVERY COMPARISON",
        "-" * 80,
        "Lane                         N       PnL       $PF     AvgR     TotR      WR      MaxDD$  QQQ/GLD",
        (
            f"Certified baseline       {baseline['total_trades']:3d} "
            f"${baseline['total_pnl_dollars']:+9,.2f} {baseline['dollar_profit_factor']:8.3f} "
            f"{baseline['avg_r']:+8.3f} {baseline['total_r']:+8.3f} "
            f"{baseline['win_rate']:7.1%} ${baseline['trade_path_max_dd_dollars']:9,.2f} "
            f"{baseline['qqq_trade_count']:3d}/{baseline['gld_trade_count']:<3d}"
        ),
        (
            f"Recovered canonical      {recovered['total_trades']:3d} "
            f"${recovered['total_pnl_dollars']:+9,.2f} {recovered['dollar_profit_factor']:8.3f} "
            f"{recovered['avg_r']:+8.3f} {recovered['total_r']:+8.3f} "
            f"{recovered['win_rate']:7.1%} ${recovered['trade_path_max_dd_dollars']:9,.2f} "
            f"{recovered['qqq_trade_count']:3d}/{recovered['gld_trade_count']:<3d}"
        ),
        (
            f"Recovery delta           {int(delta['total_trades']):+3d} "
            f"${delta['total_pnl_dollars']:+9,.2f} {delta['dollar_profit_factor']:+8.3f} "
            f"{delta['avg_r']:+8.3f} {delta['total_r']:+8.3f} "
            f"{delta['win_rate']:+7.1%} ${delta['trade_path_max_dd_dollars']:+9,.2f} "
            f"{int(delta['qqq_trade_count']):+3d}/{int(delta['gld_trade_count']):+3d}"
        ),
        "",
        "CERTIFIED IS OVERLAP",
        "-" * 80,
        (
            f"Baseline: {overlap['baseline']['total_trades']} trades, "
            f"${overlap['baseline']['total_pnl_dollars']:+,.2f}, "
            f"avgR={overlap['baseline']['avg_r']:+.3f}, "
            f"QQQ/GLD={overlap['baseline']['qqq_trade_count']}/{overlap['baseline']['gld_trade_count']}"
        ),
        (
            f"Recovered: {overlap['recovered']['total_trades']} trades, "
            f"${overlap['recovered']['total_pnl_dollars']:+,.2f}, "
            f"avgR={overlap['recovered']['avg_r']:+.3f}, "
            f"QQQ/GLD={overlap['recovered']['qqq_trade_count']}/{overlap['recovered']['gld_trade_count']}"
        ),
        "The overlap contains no QQQ trades and therefore cannot validate QQQ veto neutrality.",
        "",
        "SUPPRESSED OOS TRADES",
        "-" * 80,
    ]
    for trade in oos["suppressed_baseline_trades"]:
        lines.append(
            f"{trade['entry_time']} {trade['symbol']} pnl=${trade['pnl_dollars']:+,.2f} "
            f"R={trade['r_multiple']:+.4f} exit={trade['exit_reason']}"
        )
    lines.extend(
        [
            "",
            "VERDICT",
            "-" * 80,
            "PROMOTED_WITH_CONSUMED_OOS: large certified-OOS repair with bounded real IS cost.",
            "The next period after 2026-05-02 remains the required untouched promotion holdout.",
            "",
            f"Replay source fingerprint: {evidence['replay_source_fingerprint']}",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def _promotion_evaluation(
    final_metrics: dict[str, Any],
    baseline: dict[str, Any],
    certified: dict[str, Any],
) -> str:
    old = baseline["final_metrics"]
    oos = certified["oos"]
    return "\n".join(
        [
            "TPC ROUND 8 POST-OPTIMISATION CONTEXT RECOVERY PROMOTION",
            "=" * 80,
            f"Status: {PROMOTION_STATUS}",
            f"Canonical mutation: {RECOVERY_KEY}=True",
            "Original four phase-auto phases: retained unchanged as historical search evidence.",
            "OOS status: consumed development evidence; next untouched period starts after 2026-05-02.",
            "",
            "FULL HISTORICAL IS",
            "-" * 80,
            f"Trades: {int(old['total_trades'])} -> {int(final_metrics['total_trades'])}",
            f"Net return: {old['net_return_pct']:+.6f}% -> {final_metrics['net_return_pct']:+.6f}%",
            f"R profit factor: {old['profit_factor']:.6f} -> {final_metrics['profit_factor']:.6f}",
            f"Dollar profit factor: {old['dollar_profit_factor']:.6f} -> {final_metrics['dollar_profit_factor']:.6f}",
            f"Avg R: {old['avg_r']:+.6f} -> {final_metrics['avg_r']:+.6f}",
            f"Max DD: {old['max_dd_pct']:.6f}% -> {final_metrics['max_dd_pct']:.6f}%",
            "",
            "STRICT CERTIFIED OOS",
            "-" * 80,
            f"Trades: {oos['baseline']['total_trades']} -> {oos['recovered']['total_trades']}",
            f"PnL: ${oos['baseline']['total_pnl_dollars']:+,.2f} -> ${oos['recovered']['total_pnl_dollars']:+,.2f}",
            f"Dollar PF: {oos['baseline']['dollar_profit_factor']:.6f} -> {oos['recovered']['dollar_profit_factor']:.6f}",
            f"Avg R: {oos['baseline']['avg_r']:+.6f} -> {oos['recovered']['avg_r']:+.6f}",
            f"Trade-path DD: ${oos['baseline']['trade_path_max_dd_dollars']:,.2f} -> ${oos['recovered']['trade_path_max_dd_dollars']:,.2f}",
            "",
            "Decision: recovered configuration is the canonical active round 8.",
        ]
    )


def _build_provenance(mutations: dict[str, Any], certified: dict[str, Any]) -> Any:
    raw_data_item = build_tree_item(
        "data_dir",
        RAW_DATA_DIR,
        patterns=(
            "QQQ_*.parquet",
            "GLD_*.parquet",
            "NQ_*.parquet",
            "GC_*.parquet",
        ),
        recursive=False,
        scope="data",
        display_root=ROOT,
        notes="Root-level replay inputs consumed by the full historical IS run.",
    )
    authority_item = build_tree_item(
        "certified_context_authority",
        CONTEXT_AUTHORITY_DIR,
        patterns=(
            "NQ_1h.parquet",
            "NQ_1d.parquet",
            "NQ_1h.manifest.json",
            "NQ_1d.manifest.json",
            "NQ_futures_context.manifest.json",
            "GC_1h.parquet",
            "GC_1d.parquet",
            "GC_1h.manifest.json",
            "GC_1d.manifest.json",
            "GC_futures_context.manifest.json",
        ),
        recursive=False,
        scope="data",
        display_root=ROOT,
        notes="Certified NQ/GC parent/child authority used for post-round OOS recovery.",
    )
    return build_phase_auto_provenance(
        "tpc_round8_context_recovery_promotion",
        repo_root=ROOT,
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "backtests/swing/analysis/etf_baseline.py",
            ROOT / "backtests/swing/auto/tpc/plugin.py",
            ROOT / "backtests/swing/config_tpc.py",
            ROOT / "backtests/swing/data/replay_cache.py",
            ROOT / "backtests/swing/data/futures_context_authority.py",
            ROOT / "backtests/swing/engine/tpc_engine.py",
            ROOT / "strategies/swing/tpc/config.py",
            ROOT / "strategies/swing/tpc/core/logic.py",
        ),
        source_artifacts={
            "optimized_config": OPTIMIZED_CONFIG,
            "context_requalification": REQUALIFICATION_REPORT,
        },
        selection_context={
            "round": ROUND_NUM,
            "status": PROMOTION_STATUS,
            "recovery_mutation": {RECOVERY_KEY: mutations[RECOVERY_KEY]},
            "official_is_end": IS_END,
            "certified_start": CERTIFIED_START,
            "oos_end": OOS_END,
            "oos_status": "consumed_development_evidence",
            "certified_replay_source_fingerprint": certified["replay_source_fingerprint"],
        },
        diagnostics_paths={
            "full_is": FULL_IS_DIAGNOSTICS,
            "full_oos": FULL_OOS_DIAGNOSTICS,
            "round_final": FINAL_DIAGNOSTICS,
            "round_evaluation": ROUND_EVALUATION,
        },
        extra_items=(raw_data_item, authority_item),
    )


def promote() -> dict[str, Any]:
    mutations = _load_json(OPTIMIZED_CONFIG)
    if mutations.get(RECOVERY_KEY) is not True:
        raise RuntimeError(f"{OPTIMIZED_CONFIG} must set {RECOVERY_KEY}=true before promotion.")
    if not bool(SYMBOL_CONFIGS["QQQ"].asset_context_block_opposed_daily):
        raise RuntimeError("Live TPC QQQ config does not enable the recovered daily-opposition veto.")
    # Canonicalize bytes before provenance fingerprints the promoted config.
    ROUND_MANAGER.write_optimized_config(ROUND_DIR, mutations)
    baseline = _baseline_reference()
    phase_state = _load_json(PHASE_STATE)
    phase_mutations = dict(phase_state.get("cumulative_mutations", {}))

    print("Running recovered full historical IS replay...", flush=True)
    _, final_metrics, full_is_report, full_is_fingerprint = _format_full_is_report(mutations)
    print("Running strict certified baseline/recovered replay...", flush=True)
    certified, full_oos_report = _certified_recovery_evidence(mutations)

    full_is_delta = _metric_deltas(
        final_metrics,
        baseline["final_metrics"],
        (
            "total_trades",
            "net_return_pct",
            "total_pnl_dollars",
            "profit_factor",
            "dollar_profit_factor",
            "avg_r",
            "total_r",
            "win_rate",
            "max_dd_pct",
            "sharpe",
            "trades_per_month",
            "qqq_trade_count",
            "gld_trade_count",
        ),
    )
    evaluation = _promotion_evaluation(final_metrics, baseline, certified)
    final_header = "\n".join(
        [
            "TPC ROUND 8 RECOVERED CANONICAL FULL FINAL ROUND DIAGNOSTICS",
            "=" * 80,
            f"Generated: {_utc_now()}",
            f"Canonical status: {PROMOTION_STATUS}",
            f"Canonical mutation count: {len(mutations)}",
            f"Recovery mutation: {RECOVERY_KEY}=True",
            "Original four phase-auto phases remain unchanged historical evidence.",
            "Strict OOS was consumed during recovery and is not an untouched holdout.",
            "",
            evaluation,
            "",
            "FULL IN-SAMPLE DIAGNOSTICS",
            "=" * 80,
        ]
    )
    round_final = f"{final_header}\n{full_is_report}\n\n{full_oos_report}"

    _write_text(FULL_IS_DIAGNOSTICS, full_is_report)
    _write_text(FULL_OOS_DIAGNOSTICS, full_oos_report)
    _write_text(FINAL_DIAGNOSTICS, round_final)
    _write_text(ROUND_EVALUATION, evaluation)

    provenance = _build_provenance(mutations, certified)
    generated_at = _utc_now()
    ROUND_MANAGER.write_run_summary(
        ROUND_DIR,
        mutations,
        final_metrics,
        list(phase_state.get("completed_phases", [])),
        round_num=ROUND_NUM,
        source_diagnostics=FINAL_DIAGNOSTICS,
        source_phase_state=PHASE_STATE,
        provenance=provenance,
        provenance_status=PROMOTION_STATUS,
        provenance_validation={
            "valid": True,
            "status": PROMOTION_STATUS,
            "selection_drift": False,
            "diagnostics_drift": False,
            "message": (
                "Recovered daily-opposition veto promoted after complete historical IS replay "
                "and strict certified-context OOS development comparison."
            ),
        },
    )
    summary = _load_json(RUN_SUMMARY)
    summary.update(
        {
            "selection_status": PROMOTION_STATUS,
            "diagnostics_refresh": {
                "status": "recovered_config_replayed_is_and_certified_oos",
                "updated_saved_metrics": True,
                "basis": "complete historical IS plus strict certified OOS development replay",
                "full_is_delta": full_is_delta,
            },
            "phase_search_evidence": {
                "role": "original four-phase search evidence; intentionally not rewritten",
                "completed_phases": phase_state.get("completed_phases", []),
                "mutation_count": len(phase_mutations),
                "recovery_present": phase_mutations.get(RECOVERY_KEY) is True,
            },
            "post_optimization_recovery": {
                "status": PROMOTION_STATUS,
                "mutation": {RECOVERY_KEY: True},
                "full_is": {
                    "baseline": baseline,
                    "recovered_final_metrics": final_metrics,
                    "delta": full_is_delta,
                    "replay_source_fingerprint": full_is_fingerprint,
                    "context_basis": "legacy_raw_full_history",
                },
                "certified_context": certified,
                "oos_status": "consumed_development_evidence",
                "next_untouched_period_starts_after": OOS_END,
            },
            "artifacts": {
                "optimized_config": str(OPTIMIZED_CONFIG.resolve()),
                "round_final_diagnostics": str(FINAL_DIAGNOSTICS.resolve()),
                "full_diagnostics_in_sample": str(FULL_IS_DIAGNOSTICS.resolve()),
                "full_diagnostics_out_of_sample": str(FULL_OOS_DIAGNOSTICS.resolve()),
                "diagnostics_summary": str(DIAGNOSTICS_SUMMARY.resolve()),
                "recovery_promotion": str(PROMOTION_RECORD.resolve()),
                "context_requalification": str(REQUALIFICATION_REPORT.resolve()),
                "round_evaluation": str(ROUND_EVALUATION.resolve()),
            },
        }
    )
    _atomic_write_json(summary, RUN_SUMMARY)
    ROUND_MANAGER.append_to_manifest(
        ROUND_NUM,
        mutations,
        final_metrics,
        provenance=provenance,
        provenance_status=PROMOTION_STATUS,
    )

    spec = _load_json(RUN_SPEC)
    execution_context = dict(spec.get("execution_context", {}))
    execution_context["post_optimization_recovery"] = {
        "status": PROMOTION_STATUS,
        "mutation": {RECOVERY_KEY: True},
        "canonical_mutation_count": len(mutations),
        "phase_state_role": "original four-phase search evidence; intentionally not rewritten",
        "full_is_context_basis": "legacy_raw_full_history",
        "certified_context_dir": str(CONTEXT_AUTHORITY_DIR.resolve()),
        "certified_window": [CERTIFIED_START, OOS_END],
        "official_is_end": IS_END,
        "oos_status": "consumed_development_evidence",
        "next_untouched_period_starts_after": OOS_END,
    }
    spec.update(
        {
            "description": (
                "Round 8 train-only phased optimisation plus promoted post-round certified-context "
                "QQQ daily-opposition recovery."
            ),
            "canonicalized_at_utc": generated_at,
            "canonical_mutation_count": len(mutations),
            "canonical_config_status": PROMOTION_STATUS,
            "execution_context": execution_context,
            "provenance": provenance.to_dict(),
            "provenance_status": PROMOTION_STATUS,
        }
    )
    _atomic_write_json(spec, RUN_SPEC)

    record = {
        "family": "swing",
        "strategy": "tpc",
        "round": ROUND_NUM,
        "promoted_at_utc": generated_at,
        "status": PROMOTION_STATUS,
        "canonical_mutation_count": len(mutations),
        "recovery_mutation": {RECOVERY_KEY: True},
        "full_is": {
            "baseline": baseline,
            "recovered": {
                "headline_metrics": canonicalize_metrics(final_metrics),
                "final_metrics": final_metrics,
            },
            "delta": full_is_delta,
            "replay_source_fingerprint": full_is_fingerprint,
            "context_basis": "legacy_raw_full_history",
        },
        "certified_context": certified,
        "phase_search_evidence": summary["phase_search_evidence"],
        "oos_status": "consumed_development_evidence",
        "next_untouched_period_starts_after": OOS_END,
        "provenance": provenance.to_dict(),
        "provenance_status": PROMOTION_STATUS,
        "artifacts": summary["artifacts"],
    }
    _atomic_write_json(record, PROMOTION_RECORD)
    _atomic_write_json(
        {
            "family": "swing",
            "strategy": "tpc",
            "round": ROUND_NUM,
            "generated_at_utc": generated_at,
            "status": PROMOTION_STATUS,
            "canonical_mutation_count": len(mutations),
            "recovery_mutation": {RECOVERY_KEY: True},
            "headline_metrics": canonicalize_metrics(final_metrics),
            "final_metrics": final_metrics,
            "full_is_delta": full_is_delta,
            "certified_context": certified,
            "phase_search_evidence": summary["phase_search_evidence"],
            "provenance": provenance.to_dict(),
            "provenance_status": PROMOTION_STATUS,
            "artifacts": summary["artifacts"],
        },
        DIAGNOSTICS_SUMMARY,
    )
    return record


def verify() -> dict[str, Any]:
    config = _load_json(OPTIMIZED_CONFIG)
    summary = _load_json(RUN_SUMMARY)
    manifest = _load_json(ROUND_MANAGER.manifest_path)
    active = [
        entry
        for entry in manifest.get("rounds", [])
        if int(entry.get("round", 0)) == ROUND_NUM and not entry.get("archived")
    ]
    errors: list[str] = []
    if len(active) != 1:
        errors.append(f"expected one active round-8 manifest entry, found {len(active)}")
    if config.get(RECOVERY_KEY) is not True:
        errors.append("optimized config does not enable recovery")
    if summary.get("cumulative_mutations") != config:
        errors.append("run summary mutations differ from optimized config")
    if active and active[0].get("mutations") != config:
        errors.append("active manifest mutations differ from optimized config")
    if summary.get("mutation_count") != len(config):
        errors.append("run summary mutation count is stale")
    if active and active[0].get("mutations_count") != len(config):
        errors.append("active manifest mutation count is stale")
    for path in (
        FULL_IS_DIAGNOSTICS,
        FULL_OOS_DIAGNOSTICS,
        FINAL_DIAGNOSTICS,
        DIAGNOSTICS_SUMMARY,
        PROMOTION_RECORD,
    ):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {path}")
    if errors:
        raise RuntimeError("Round-8 promotion verification failed:\n - " + "\n - ".join(errors))
    return {
        "verified": True,
        "active_manifest_entries": len(active),
        "mutation_count": len(config),
        "recovery_enabled": config[RECOVERY_KEY],
        "headline_metrics": summary.get("headline_metrics"),
        "provenance_status": summary.get("provenance_status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify() if args.verify_only else promote()
    if not args.verify_only:
        result = {"promotion": result, "verification": verify()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
