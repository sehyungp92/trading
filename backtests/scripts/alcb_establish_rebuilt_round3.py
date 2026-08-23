"""Recoverably establish the validated ALCB lineage rebuild as Round 3.

This is intentionally separate from candidate selection.  It requires a
complete lineage-rebuild result, replays the selected configuration once more,
builds all round artifacts in staging, archives the current Round 3, and then
atomically installs the replacement.  The resulting round is a provisional
research round, not a production authorization.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.scripts.alcb_materialize_round3 import (  # noqa: E402
    _artifact_manifest,
    _assert_metrics_match,
    _group,
    _run_window,
    _sha256,
)
from backtests.scripts.alcb_round2_oos_robustness import (  # noqa: E402
    DATA_DIR,
    INITIAL_EQUITY,
    _config_fingerprint,
    _diagnostics_consistent,
    _trade_diagnostics,
    _trade_to_dict,
    _write_json,
)
from backtests.shared.auto.round_manager import RoundManager  # noqa: E402
from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic  # noqa: E402
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis  # noqa: E402
from backtests.stock.auto.alcb.run_baseline_recovery import (  # noqa: E402
    END_DATE,
    START_DATE,
    _code_fingerprint,
    _compact_metrics,
    _source_fingerprint,
)
from backtests.stock.auto.alcb.run_round3_lineage_rebuild import (  # noqa: E402
    ARCHIVED_ROUND3_PATCH,
    BASELINE_CONFIG,
    DEFAULT_OUTPUT as REBUILD_DIR,
)
from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations  # noqa: E402
from backtests.stock.auto.config_mutator import mutate_alcb_config  # noqa: E402
from backtests.stock.config_alcb import ALCBBacktestConfig  # noqa: E402
from strategies.stock.alcb.config import StrategySettings  # noqa: E402


STRATEGY_DIR = REPO_ROOT / "backtests/output/stock/alcb"
ROUND2_DIR = STRATEGY_DIR / "round_2"
ROUND3_DIR = STRATEGY_DIR / "round_3"
ROUND_MANAGER = RoundManager("stock", "alcb")
RESULTS_PATH = REBUILD_DIR / "lineage_rebuild_results.json"
RECOMMENDED_CONFIG = REBUILD_DIR / "recommended_config.json"
RECOMMENDED_PATCH = REBUILD_DIR / "recommended_patch.json"
PROVENANCE_STATUS = (
    "provisional_development_selected_projected_rth_consumed_lineage_"
    "fresh_lockbox_required"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reconcile-live-settings", action="store_true")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normalize_materialization_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Match the evaluator's explicit AvgR alias without changing economics."""

    normalized = dict(metrics)
    normalized.setdefault("avg_r", float(normalized.get("expectancy", 0.0)))
    return normalized


def _manifest_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def _live_settings_mismatches(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    backtest_config = mutate_alcb_config(
        ALCBBacktestConfig(), hydrate_time_mutations(config)
    )
    candidate = StrategySettings(**backtest_config.param_overrides)
    live = StrategySettings()
    mismatches: dict[str, dict[str, Any]] = {}
    for key in config:
        if not key.startswith("param_overrides."):
            continue
        field = key.split(".", 1)[1]
        candidate_value = getattr(candidate, field)
        live_value = getattr(live, field)
        comparable_candidate = (
            tuple(candidate_value)
            if isinstance(candidate_value, (list, tuple))
            else candidate_value
        )
        comparable_live = (
            tuple(live_value) if isinstance(live_value, (list, tuple)) else live_value
        )
        if comparable_candidate != comparable_live:
            mismatches[field] = {
                "candidate": _manifest_value(candidate_value),
                "live": _manifest_value(live_value),
            }
    return mismatches


def _reconcile_live_settings_contract() -> int:
    config = _load(ROUND3_DIR / "optimized_config.json")
    mismatches = _live_settings_mismatches(config)
    manifest = ROUND_MANAGER.load_manifest()
    active = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if len(active) != 1:
        raise RuntimeError(f"Expected one active Round 3, found {len(active)}.")
    if active[0].get("selected_candidate") != "bundle__rvol_late_quality":
        raise RuntimeError("Active Round 3 is not the expected rebuilt candidate.")
    if active[0].get("mutations") != config:
        raise RuntimeError("Active manifest config does not match optimized_config.json.")

    live_fields = {
        "live_settings_sync_required": bool(mismatches),
        "live_settings_mismatches": mismatches,
    }
    active[0].update(live_fields)
    _write_json(ROUND_MANAGER.manifest_path, manifest)

    for name in (
        "phase_state.json",
        "progress.json",
        "final_optimization_summary.json",
        "run_summary.json",
        "validation.json",
    ):
        path = ROUND3_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing rebuilt Round 3 artifact: {path}")
        payload = _load(path)
        payload.update(live_fields)
        _write_json(path, payload)

    archive_round = Path(active[0]["archive_of_replaced_round3"])
    shutil.copy2(
        ROUND_MANAGER.manifest_path,
        archive_round.parent / "rounds_manifest_after_switch.json",
    )
    receipt_path = ROUND3_DIR / "manifest_update_receipt.json"
    receipt = _load(receipt_path)
    receipt["updated_at_utc"] = datetime.now(timezone.utc)
    receipt["manifest_sha256"] = _sha256(ROUND_MANAGER.manifest_path)
    _write_json(receipt_path, receipt)
    _write_json(ROUND3_DIR / "artifact_manifest.json", _artifact_manifest(ROUND3_DIR))
    print(
        json.dumps(
            {
                "round": 3,
                "live_settings_sync_required": bool(mismatches),
                "live_settings_mismatches": mismatches,
            },
            indent=2,
        )
    )
    return 0


def _validate_recommendation(
    results: dict[str, Any], config: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    if results.get("status") != "complete":
        raise RuntimeError("Lineage rebuild is not complete.")
    if results.get("consumed_oos_accessed") is not False:
        raise RuntimeError("Lineage rebuild accessed the consumed OOS interval.")
    if results.get("development_window") != {"start": START_DATE, "end": END_DATE}:
        raise RuntimeError("Lineage rebuild used an unexpected development window.")
    if results.get("source_fingerprint") != _source_fingerprint():
        raise RuntimeError("Data fingerprint drifted after candidate selection.")
    if results.get("code_fingerprint") != _code_fingerprint():
        raise RuntimeError("Economic code fingerprint drifted after candidate selection.")

    baseline = _load(BASELINE_CONFIG)
    expected = dict(baseline)
    expected.update(patch)
    if config != dict(sorted(expected.items())) and config != expected:
        raise RuntimeError("Recommended config is not combined baseline plus selected patch.")
    if not set(patch).issubset(ARCHIVED_ROUND3_PATCH):
        raise RuntimeError("Recommendation contains a mutation outside the saved Round 3 lineage.")
    if any(ARCHIVED_ROUND3_PATCH[key] != value for key, value in patch.items()):
        raise RuntimeError("Recommendation changes an archived Round 3 mutation value.")
    required_trail = {
        "param_overrides.adaptive_trail_late_activate_r": 0.18,
        "param_overrides.adaptive_trail_start_bars": 30,
        "param_overrides.adaptive_trail_tighten_bars": 30,
    }
    if any(config.get(key) != value for key, value in required_trail.items()):
        raise RuntimeError("Recommended config lost the validated combined-trail baseline.")

    selected_id = results.get("decision", {}).get("selected_candidate")
    matches = [row for row in results.get("finalists", []) if row.get("id") == selected_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one selected finalist, found {len(matches)}.")
    selected = matches[0]
    if selected.get("mutations") != config or selected.get("patch", {}) != patch:
        raise RuntimeError("Selected finalist does not match recommendation files.")
    if selected_id != "baseline__combined_trail" and not selected.get("passes_final_gate"):
        raise RuntimeError("Selected lineage addition did not pass the final incremental gate.")
    return selected


def _comparison(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    bm = baseline["metrics"]
    sm = selected["metrics"]
    keys = (
        "total_trades",
        "trades_per_month",
        "win_rate",
        "avg_r",
        "expected_total_r",
        "net_profit",
        "profit_factor",
        "max_drawdown_pct",
        "sharpe",
    )
    return {
        key: {
            "combined_trail_baseline": bm.get(key),
            "new_round3": sm.get(key),
            "delta": float(sm.get(key, 0.0)) - float(bm.get(key, 0.0)),
        }
        for key in keys
    }


def _evaluation_text(
    selected: dict[str, Any], comparison: dict[str, Any], patch: dict[str, Any]
) -> str:
    metrics = selected["metrics"]
    lines = [
        "ALCB ROUND 3 - COMBINED-TRAIL LINEAGE REBUILD",
        "=" * 78,
        f"Selected candidate: {selected['id']}",
        f"Incremental patch: {json.dumps(patch, sort_keys=True)}",
        f"Provenance status: {PROVENANCE_STATUS}",
        "Production deployment approved: NO",
        "Selection replay interval: 2024-03-25 through 2026-03-01 only.",
        "Consumed post-2026-03-01 OOS accessed by this rebuild: NO",
        "Lineage contamination acknowledged: YES; candidate ideas originated in prior consumed-OOS research.",
        "Required next gate: accepted direct-RTH replay plus a genuinely unseen lockbox.",
        "",
        "COMBINED TRAIL BASELINE -> NEW ROUND 3",
        "-" * 78,
        (
            f"  R {comparison['expected_total_r']['combined_trail_baseline']:+.2f} -> "
            f"{metrics['expected_total_r']:+.2f}; AvgR "
            f"{comparison['avg_r']['combined_trail_baseline']:+.4f} -> {metrics['avg_r']:+.4f}; "
            f"PF {comparison['profit_factor']['combined_trail_baseline']:.3f} -> "
            f"{metrics['profit_factor']:.3f}; DD "
            f"{comparison['max_drawdown_pct']['combined_trail_baseline']:.2%} -> "
            f"{metrics['max_drawdown_pct']:.2%}; TPM "
            f"{comparison['trades_per_month']['combined_trail_baseline']:.1f} -> "
            f"{metrics['trades_per_month']:.1f}"
        ),
        "",
        "VALIDATION",
        "-" * 78,
        f"  Fold wins versus combined baseline: {selected.get('fold_wins_vs_baseline', 0)}/4",
        f"  7.5 bps R: {selected['costs']['7.5']['expected_total_r']:+.2f}",
        f"  10.0 bps R: {selected['costs']['10.0']['expected_total_r']:+.2f}",
    ]
    return "\n".join(lines) + "\n"


def _copy_selection_evidence(stage: Path, results_path: Path) -> None:
    evidence = stage / "selection_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    sources = [
        results_path,
        results_path.parent / "candidate_catalog.json",
        results_path.parent / "lineage_rebuild_report.md",
        results_path.parent / "recommended_config.json",
        results_path.parent / "recommended_patch.json",
        BASELINE_CONFIG,
        BASELINE_CONFIG.parent / "trail_combination_report.md",
        REPO_ROOT / "backtests/stock/auto/alcb/run_round3_lineage_rebuild.py",
        Path(__file__).resolve(),
    ]
    copied = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Missing selection evidence: {source}")
        target = evidence / source.name
        if target.exists():
            target = evidence / f"{source.parent.name}_{source.name}"
        shutil.copy2(source, target)
        copied.append(
            {
                "source": str(source.resolve()),
                "artifact": str(target.relative_to(stage)).replace("\\", "/"),
                "sha256": _sha256(target),
            }
        )
    _write_json(
        evidence / "evidence_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc),
            "consumed_oos_accessed_by_rebuild": False,
            "lineage_contamination_acknowledged": True,
            "files": copied,
        },
    )


def main() -> int:
    args = _parse_args()
    if args.reconcile_live_settings:
        return _reconcile_live_settings_contract()
    if not args.execute:
        raise SystemExit("Pass --execute to recoverably replace Round 3.")
    results_path = args.results.resolve()
    results = _load(results_path)
    config = _load(results_path.parent / RECOMMENDED_CONFIG.name)
    patch = _load(results_path.parent / RECOMMENDED_PATCH.name)
    selected = _validate_recommendation(results, config, patch)

    manifest_before = ROUND_MANAGER.load_manifest()
    active_before = [
        row
        for row in manifest_before.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if len(active_before) != 1:
        raise RuntimeError(f"Expected one active Round 3, found {len(active_before)}.")
    if not ROUND3_DIR.is_dir():
        raise FileNotFoundError(f"Current Round 3 directory is missing: {ROUND3_DIR}")

    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    print(f"fresh selected replay: {START_DATE} -> {END_DATE}", flush=True)
    context, provenance = _run_window(config, START_DATE, END_DATE)
    metrics = _normalize_materialization_metrics(context["metrics"])
    _assert_metrics_match(metrics, selected["metrics"], label="development")
    trades = list(context["trades"])
    rows = [_trade_to_dict(trade) for trade in trades]
    trade_diagnostics = _trade_diagnostics(trades)
    if not _diagnostics_consistent(trade_diagnostics, metrics):
        raise RuntimeError("Fresh trade diagnostics do not reconcile to metrics.")

    baseline = results["baseline"]
    comparison = _comparison(baseline, selected)
    evaluation = _evaluation_text(selected, comparison, patch)
    max_positions = int(context["config"].param_overrides.get("max_positions", 10))
    full_diagnostics = "\n\n".join(
        [
            evaluation.rstrip(),
            alcb_full_diagnostic(
                trades,
                shadow_tracker=None,
                daily_selections=context.get("daily_selections"),
            ),
            qe_replacement_analysis(trades, max_positions=max_positions),
        ]
    ) + "\n"
    generated_at = datetime.now(timezone.utc)
    config_sha = _config_fingerprint(config)
    round2 = _load(ROUND2_DIR / "optimized_config.json")
    reference_changes = {
        key: {"round2_reference": round2.get(key), "new_round3": value}
        for key, value in config.items()
        if round2.get(key) != value
    }
    live_settings_mismatches = _live_settings_mismatches(config)
    pre_switch_manifest_bytes = ROUND_MANAGER.manifest_path.read_bytes()

    with tempfile.TemporaryDirectory(prefix=".rebuilt_round3_staging_", dir=STRATEGY_DIR) as temp:
        stage = Path(temp)
        _copy_selection_evidence(stage, results_path)
        _write_json(stage / "optimized_config.json", config)
        _write_json(stage / "final_metrics.json", metrics)
        _write_json(stage / "final_trades.json", rows)
        _write_json(stage / "final_monthly.json", _group(rows, "month", lambda row: str(row["exit_time"])[:7]))
        _write_json(stage / "final_symbols.json", _group(rows, "symbol", lambda row: str(row.get("symbol") or "UNKNOWN")))
        _write_json(stage / "final_exits.json", _group(rows, "exit_reason", lambda row: str(row.get("exit_reason") or "UNKNOWN")))
        _write_json(stage / "diagnostics_summary.json", {"development": trade_diagnostics})
        _write_json(stage / "candidate_comparison.json", comparison)
        _write_json(stage / "fold_validation.json", selected["validation"])
        _write_json(stage / "cost_validation.json", selected["costs"])
        (stage / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")
        (stage / "round_final_diagnostics.txt").write_text(full_diagnostics, encoding="utf-8")

        phase_state = {
            "round": 3,
            "round_name": "round_3_combined_trail_lineage_rebuild",
            "current_phase": 1,
            "completed_phases": [1],
            "cumulative_mutations": config,
            "phase_results": {
                "1": {
                    "focus": "Combined-trail baseline plus targeted saved Round 3 lineage evaluation",
                    "candidate_count": results["candidate_count"],
                    "finalist_count": results["finalist_count"],
                    "selected_candidate": selected["id"],
                    "selected_patch": patch,
                    "final_metrics": metrics,
                }
            },
            "provenance_status": PROVENANCE_STATUS,
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "reference_settings_sync_required": bool(reference_changes),
            "reference_settings_changes": reference_changes,
            "live_settings_sync_required": bool(live_settings_mismatches),
            "live_settings_mismatches": live_settings_mismatches,
        }
        _write_json(stage / "phase_state.json", phase_state)
        _write_json(
            stage / "progress.json",
            {
                "status": "complete",
                "round": 3,
                "completed_at_utc": generated_at,
                "selected_candidate": selected["id"],
                "selected_patch": patch,
                "provenance_status": PROVENANCE_STATUS,
                "production_deployment_approved": False,
                "live_settings_sync_required": bool(live_settings_mismatches),
                "live_settings_mismatches": live_settings_mismatches,
            },
        )
        _write_json(
            stage / "final_optimization_summary.json",
            {
                "materialized_at_utc": generated_at,
                "selected_candidate": selected["id"],
                "selected_patch": patch,
                "optimized_config_sha256": config_sha,
                "mutation_count": len(config),
                "combined_trail_baseline": _compact_metrics(baseline["metrics"]),
                "new_round3": _compact_metrics(metrics),
                "comparison": comparison,
                "fold_validation": selected["validation"],
                "cost_validation": selected["costs"],
                "data_authority": "projected_rth_diagnostic_only",
                "consumed_oos_accessed_by_rebuild": False,
                "lineage_contamination_acknowledged": True,
                "promotion_authorized": False,
                "production_deployment_approved": False,
                "live_settings_sync_required": bool(live_settings_mismatches),
                "live_settings_mismatches": live_settings_mismatches,
                "required_revalidation": results["decision"]["required_revalidation"],
                "archive_of_replaced_round3": None,
            },
        )
        ROUND_MANAGER.write_run_spec(
            stage,
            3,
            "alcb",
            description="Combined-trail baseline plus validated saved Round 3 lineage mutations",
            baseline_mutations=_load(BASELINE_CONFIG),
            baseline_source=BASELINE_CONFIG,
            execution_context={
                "data_dir": str(DATA_DIR.resolve()),
                "initial_equity": INITIAL_EQUITY,
                "start": START_DATE,
                "end": END_DATE,
                "consumed_oos_accessed": False,
                "lineage_contamination_acknowledged": True,
                "research_dir": str(results_path.parent),
            },
            provenance=provenance,
            provenance_status=PROVENANCE_STATUS,
            overwrite=True,
        )
        ROUND_MANAGER.write_run_summary(
            stage,
            config,
            metrics,
            [1],
            round_num=3,
            source_diagnostics=ROUND3_DIR / "round_final_diagnostics.txt",
            source_phase_state=ROUND3_DIR / "phase_state.json",
            provenance=provenance,
            provenance_status=PROVENANCE_STATUS,
            provenance_validation={
                "valid": True,
                "status": "fresh_materialization_matches_lineage_rebuild",
                "selection_drift": False,
                "diagnostics_drift": False,
                "message": "Fresh development replay matches selection metrics; external promotion gates remain.",
            },
        )
        run_summary_path = stage / "run_summary.json"
        run_summary = _load(run_summary_path)
        run_summary.update(
            {
                "selected_candidate": selected["id"],
                "selected_patch": patch,
                "combined_trail_baseline_metrics": baseline["metrics"],
                "fold_validation": selected["validation"],
                "cost_validation": selected["costs"],
                "optimized_config_sha256": config_sha,
                "consumed_oos_accessed_by_rebuild": False,
                "lineage_contamination_acknowledged": True,
                "promotion_authorized": False,
                "production_deployment_approved": False,
                "live_settings_sync_required": bool(live_settings_mismatches),
                "live_settings_mismatches": live_settings_mismatches,
                "required_revalidation": results["decision"]["required_revalidation"],
                "archive_of_replaced_round3": None,
            }
        )
        _write_json(run_summary_path, run_summary)
        _write_json(stage / "artifact_manifest.json", _artifact_manifest(stage))

        print("archiving current Round 3", flush=True)
        archive_dir = ROUND_MANAGER.archive_rounds(
            [3], reason="replace_with_combined_trail_lineage_rebuild"
        )
        archived_round3 = archive_dir / "round_3"
        if not archived_round3.is_dir():
            raise RuntimeError(f"Expected archived Round 3 at {archived_round3}")
        (archive_dir / "rounds_manifest_pre_switch.json").write_bytes(pre_switch_manifest_bytes)
        _write_json(archive_dir / "archived_round3_snapshot_manifest.json", _artifact_manifest(archived_round3))
        shutil.move(str(stage), str(ROUND3_DIR))

    manifest_path = ROUND_MANAGER.append_to_manifest(
        3,
        config,
        metrics,
        provenance=provenance,
        provenance_status=PROVENANCE_STATUS,
        round_metadata={
            "round_dir": str(ROUND3_DIR.resolve()),
            "baseline_round": 2,
            "research_baseline": str(BASELINE_CONFIG.resolve()),
            "selected_candidate": selected["id"],
            "selected_patch": patch,
            "optimized_config_sha256": config_sha,
            "expected_total_r": metrics["expected_total_r"],
            "net_profit": metrics["net_profit"],
            "expectancy": metrics["expectancy"],
            "trades_per_month": metrics["trades_per_month"],
            "sortino": metrics["sortino"],
            "tail_loss_r": metrics["tail_loss_r"],
            "fold_validation": selected["validation"],
            "cost_validation": selected["costs"],
            "data_authority": "projected_rth_diagnostic_only",
            "consumed_oos_accessed_by_rebuild": False,
            "lineage_contamination_acknowledged": True,
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "live_settings_sync_required": bool(live_settings_mismatches),
            "live_settings_mismatches": live_settings_mismatches,
            "required_revalidation": results["decision"]["required_revalidation"],
            "archive_of_replaced_round3": str(archived_round3.resolve()),
        },
    )
    manifest = _load(manifest_path)
    active = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if len(active) != 1 or active[0].get("mutations") != config:
        raise RuntimeError("Manifest activation validation failed.")
    archived = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and row.get("archived")
    ]
    if not archived:
        raise RuntimeError("Archived Round 3 manifest lineage was lost.")
    archived[-1]["archive_dir"] = str(archived_round3.resolve())
    archived[-1]["superseded_by_candidate"] = selected["id"]
    _write_json(manifest_path, manifest)
    shutil.copy2(manifest_path, archive_dir / "rounds_manifest_after_switch.json")

    for name in ("run_summary.json", "final_optimization_summary.json"):
        path = ROUND3_DIR / name
        payload = _load(path)
        payload["archive_of_replaced_round3"] = str(archived_round3.resolve())
        _write_json(path, payload)
    validation = {
        "validated_at_utc": datetime.now(timezone.utc),
        "single_active_round3_manifest_entry": len(active) == 1,
        "manifest_config_matches": active[0].get("mutations") == config,
        "optimized_config_matches": _load(ROUND3_DIR / "optimized_config.json") == config,
        "optimized_config_sha256_matches": _config_fingerprint(config) == config_sha,
        "fresh_metrics_match_selection": True,
        "trade_count_matches": len(_load(ROUND3_DIR / "final_trades.json")) == int(metrics["total_trades"]),
        "archive_exists": archived_round3.is_dir(),
        "archived_lineage_preserved": bool(archived),
        "archive_dir": str(archive_dir.resolve()),
        "live_settings_sync_required": bool(live_settings_mismatches),
        "live_settings_mismatches": live_settings_mismatches,
    }
    if not all(value for key, value in validation.items() if isinstance(value, bool)):
        raise RuntimeError(f"Round 3 validation failed: {validation}")
    _write_json(ROUND3_DIR / "validation.json", validation)
    _write_json(
        ROUND3_DIR / "manifest_update_receipt.json",
        {
            "updated_at_utc": datetime.now(timezone.utc),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "archive_dir": str(archive_dir.resolve()),
            "archived_round3": str(archived_round3.resolve()),
        },
    )
    _write_json(ROUND3_DIR / "artifact_manifest.json", _artifact_manifest(ROUND3_DIR))
    print(
        json.dumps(
            {
                "round_dir": str(ROUND3_DIR.resolve()),
                "archived_round3": str(archived_round3.resolve()),
                "selected_candidate": selected["id"],
                "selected_patch": patch,
                "expected_total_r": metrics["expected_total_r"],
                "production_deployment_approved": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
