"""Archive the targeted ALCB Round 3 and activate the first-OOS optimum.

All economic runs and diagnostics are built in staging before the current
Round 3 is recoverably archived.  The replacement remains provisional because
the repaired projected-RTH data is diagnostic-only and OOS was used to select
the candidate.
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
    CORE_METRICS,
    DATA_DIR,
    INITIAL_EQUITY,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    _config_fingerprint,
    _diagnostics_consistent,
    _json_safe,
    _trade_diagnostics,
    _trade_to_dict,
    _write_json,
)
from backtests.shared.auto.round_manager import RoundManager  # noqa: E402
from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic  # noqa: E402
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis  # noqa: E402


STRATEGY_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "alcb"
ROUND2_DIR = STRATEGY_DIR / "round_2"
ROUND3_DIR = STRATEGY_DIR / "round_3"
RESEARCH_DIR = ROUND2_DIR / "oos_ablation_perturbation_verified_20260816"
FIRST_OOS_NAME = "perturb__rvol_threshold__1p1"
REPLACED_NAME = "r110__entry1330_late_score5_failure_m010"
ROUND_MANAGER = RoundManager("stock", "alcb")
PROVENANCE_STATUS = (
    "provisional_initial_oos_recommendation_consumed_oos_"
    "direct_rth_revalidation_required"
)
LIVE_SETTINGS_MISMATCHES = {
    "rvol_threshold": {"candidate": 1.1, "live": 1.4},
}
EVIDENCE_EXCLUDE = {"background_stdout.log", "background_stderr.log"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _find_named(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result for {name!r}, found {len(matches)}.")
    return matches[0]


def _window_metrics(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key: row[f"{prefix}_{key}"]
        for key in CORE_METRICS
        if f"{prefix}_{key}" in row
    }


def _comparison(
    baseline_is: dict[str, Any],
    baseline_oos: dict[str, Any],
    selected_is: dict[str, Any],
    selected_oos: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for window, baseline, selected in (
        ("is", baseline_is, selected_is),
        ("oos", baseline_oos, selected_oos),
    ):
        output[window] = {}
        for key in CORE_METRICS:
            if key not in baseline or key not in selected:
                continue
            old = baseline[key]
            new = selected[key]
            try:
                delta = float(new) - float(old)
                relative = delta / abs(float(old)) if abs(float(old)) > 1e-12 else None
            except (TypeError, ValueError):
                delta = None
                relative = None
            output[window][key] = {
                "round2_control": old,
                "first_oos_candidate": new,
                "delta": delta,
                "delta_pct": relative,
            }
    return output


def _evaluation_text(
    baseline_is: dict[str, Any],
    baseline_oos: dict[str, Any],
    selected_is: dict[str, Any],
    selected_oos: dict[str, Any],
    completion: dict[str, Any],
) -> str:
    lines = [
        "ALCB ROUND 3 - FIRST OOS OPTIMAL CANDIDATE",
        "=" * 78,
        f"Selected candidate: {FIRST_OOS_NAME}",
        "Selection point: initial OOS audit before the additional targeted experiments.",
        "Cumulative delta from Round 2: param_overrides.rvol_threshold 1.4 -> 1.1",
        f"Provenance status: {PROVENANCE_STATUS}",
        "Production deployment approved: NO",
        "Reason: repaired projected-RTH data is diagnostic-only and OOS was consumed.",
        "Required next gate: accepted frozen direct-RTH lockbox revalidation.",
        "",
        "INITIAL AUDIT COVERAGE",
        "-" * 78,
        f"  Candidate catalog: {completion['candidate_count']}",
        f"  OOS results: {completion['oos_result_count']}",
        f"  IS validations: {completion['is_result_count']}",
        "",
        "ROUND 2 CONTROL -> FIRST OOS CANDIDATE",
        "-" * 78,
        (
            f"  IS:  R {baseline_is['expected_total_r']:+.2f} -> "
            f"{selected_is['expected_total_r']:+.2f}; trades "
            f"{baseline_is['total_trades']} -> {selected_is['total_trades']}; "
            f"TPM {baseline_is['trades_per_month']:.1f} -> "
            f"{selected_is['trades_per_month']:.1f}; PF "
            f"{baseline_is['profit_factor']:.2f} -> {selected_is['profit_factor']:.2f}; "
            f"WR {baseline_is['win_rate']:.1%} -> {selected_is['win_rate']:.1%}; "
            f"DD {baseline_is['max_drawdown_pct']:.2%} -> "
            f"{selected_is['max_drawdown_pct']:.2%}; Sharpe "
            f"{baseline_is['sharpe']:.2f} -> {selected_is['sharpe']:.2f}"
        ),
        (
            f"  OOS: R {baseline_oos['expected_total_r']:+.2f} -> "
            f"{selected_oos['expected_total_r']:+.2f}; trades "
            f"{baseline_oos['total_trades']} -> {selected_oos['total_trades']}; "
            f"TPM {baseline_oos['trades_per_month']:.1f} -> "
            f"{selected_oos['trades_per_month']:.1f}; PF "
            f"{baseline_oos['profit_factor']:.2f} -> {selected_oos['profit_factor']:.2f}; "
            f"WR {baseline_oos['win_rate']:.1%} -> {selected_oos['win_rate']:.1%}; "
            f"DD {baseline_oos['max_drawdown_pct']:.2%} -> "
            f"{selected_oos['max_drawdown_pct']:.2%}; Sharpe "
            f"{baseline_oos['sharpe']:.2f} -> {selected_oos['sharpe']:.2f}"
        ),
        "",
        "INTERPRETATION",
        "-" * 78,
        "  The single RVOL relaxation is the initial audit's balanced optimum. It",
        "  materially raises IS total return and frequency and modestly raises OOS",
        "  total return/frequency while improving OOS drawdown and Sharpe. The OOS",
        "  trade-off is lower win rate, expectancy per trade, and profit factor.",
        "  This is the pre-additional-experiment candidate requested for Round 3;",
        "  the later targeted winner is preserved in the strategy archive.",
    ]
    return "\n".join(lines) + "\n"


def _copy_evidence(stage: Path) -> dict[str, Any]:
    evidence = stage / "selection_evidence" / "initial_oos_audit"
    evidence.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for source in sorted(RESEARCH_DIR.iterdir()):
        if not source.is_file() or source.name in EVIDENCE_EXCLUDE:
            continue
        target = evidence / source.name
        shutil.copy2(source, target)
        copied.append(
            {
                "source": str(source.resolve()),
                "artifact": str(target.relative_to(stage)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    scripts_dir = stage / "selection_evidence" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        REPO_ROOT / "backtests" / "scripts" / "alcb_round2_oos_robustness.py",
        Path(__file__).resolve(),
    ):
        target = scripts_dir / source.name
        shutil.copy2(source, target)
        copied.append(
            {
                "source": str(source),
                "artifact": str(target.relative_to(stage)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc),
        "selection_stage": "initial_oos_audit_before_additional_experiments",
        "file_count": len(copied),
        "files": copied,
    }
    _write_json(stage / "selection_evidence" / "evidence_manifest.json", manifest)
    return manifest


def _archive_hash_errors(archived_round: Path) -> list[str]:
    manifest_path = archived_round / "artifact_manifest.json"
    if not manifest_path.exists():
        return ["missing:artifact_manifest.json"]
    manifest = _load(manifest_path)
    errors: list[str] = []
    for row in manifest.get("files", []):
        path = archived_round / row["path"]
        if not path.exists():
            errors.append(f"missing:{row['path']}")
        elif _sha256(path) != row["sha256"]:
            errors.append(f"hash:{row['path']}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acknowledge the recoverable archive and Round 3 replacement.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        raise SystemExit("Pass --execute to archive the current Round 3 and activate the first-OOS candidate.")

    manifest_before = ROUND_MANAGER.load_manifest()
    active_before = [
        row
        for row in manifest_before.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if len(active_before) != 1:
        raise RuntimeError(f"Expected one active Round 3, found {len(active_before)}.")
    if active_before[0].get("selected_candidate") != REPLACED_NAME:
        raise RuntimeError(
            "Current Round 3 is not the expected targeted winner; refusing an ambiguous archive."
        )
    if not ROUND3_DIR.is_dir():
        raise RuntimeError(f"Current Round 3 directory is missing: {ROUND3_DIR}")

    completion = _load(RESEARCH_DIR / "completion.json")
    if completion.get("candidate_count") != 263 or completion.get("recommended_name") != FIRST_OOS_NAME:
        raise RuntimeError("Initial OOS audit completion metadata does not identify the expected winner.")
    all_results = _load(RESEARCH_DIR / "all_results.json")
    selected = _find_named(all_results, FIRST_OOS_NAME)
    if not selected.get("is_guardrail_pass") or not selected.get("oos_strict_uplift"):
        raise RuntimeError("First-OOS candidate no longer passes its saved selection guardrails.")

    baseline = _load(RESEARCH_DIR / "baseline_diagnostics.json")
    baseline_is = dict(baseline["is"]["metrics"])
    baseline_oos = dict(baseline["oos"]["metrics"])
    expected_is = _window_metrics(selected, "is")
    expected_oos = _window_metrics(selected, "oos")
    round2_mutations = _load(ROUND2_DIR / "optimized_config.json")
    mutations = _load(RESEARCH_DIR / "recommended_config.json")
    expected_mutations = {**round2_mutations, "param_overrides.rvol_threshold": 1.1}
    if mutations != expected_mutations or len(mutations) != 50:
        raise RuntimeError("Recommended first-OOS configuration is not the expected 50-key Round-2 patch.")

    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    print(f"running fresh IS {IS_START}..{IS_END}", flush=True)
    is_context, is_provenance = _run_window(mutations, IS_START, IS_END)
    is_metrics = dict(is_context["metrics"])
    _assert_metrics_match(is_metrics, expected_is, label="IS")
    print(
        f"IS verified: {is_metrics['total_trades']} trades, "
        f"{is_metrics['expected_total_r']:+.2f}R",
        flush=True,
    )

    print(f"running fresh OOS {OOS_START}..{OOS_END}", flush=True)
    oos_context, oos_provenance = _run_window(mutations, OOS_START, OOS_END)
    oos_metrics = dict(oos_context["metrics"])
    _assert_metrics_match(oos_metrics, expected_oos, label="OOS")
    print(
        f"OOS verified: {oos_metrics['total_trades']} trades, "
        f"{oos_metrics['expected_total_r']:+.2f}R",
        flush=True,
    )

    is_trades = list(is_context["trades"])
    oos_trades = list(oos_context["trades"])
    is_rows = [_trade_to_dict(trade) for trade in is_trades]
    oos_rows = [_trade_to_dict(trade) for trade in oos_trades]
    is_trade_diagnostics = _trade_diagnostics(is_trades)
    oos_trade_diagnostics = _trade_diagnostics(oos_trades)
    if not _diagnostics_consistent(is_trade_diagnostics, is_metrics):
        raise RuntimeError("IS diagnostics do not reconcile to metrics.")
    if not _diagnostics_consistent(oos_trade_diagnostics, oos_metrics):
        raise RuntimeError("OOS diagnostics do not reconcile to metrics.")

    print("building full IS/OOS diagnostics in staging", flush=True)
    evaluation = _evaluation_text(
        baseline_is,
        baseline_oos,
        is_metrics,
        oos_metrics,
        completion,
    )
    max_positions = int(is_context["config"].param_overrides.get("max_positions", 10))
    is_full = "\n\n".join(
        [
            alcb_full_diagnostic(
                is_trades,
                shadow_tracker=None,
                daily_selections=is_context.get("daily_selections"),
            ),
            qe_replacement_analysis(is_trades, max_positions=max_positions),
        ]
    ) + "\n"
    oos_full = "\n\n".join(
        [
            alcb_full_diagnostic(
                oos_trades,
                shadow_tracker=None,
                daily_selections=oos_context.get("daily_selections"),
            ),
            qe_replacement_analysis(oos_trades, max_positions=max_positions),
        ]
    ) + "\n"
    combined_diagnostics = "\n\n".join(
        [
            evaluation.rstrip(),
            "#" * 78 + "\nIN-SAMPLE FULL DIAGNOSTIC\n" + "#" * 78,
            is_full.rstrip(),
            "#" * 78 + "\nOUT-OF-SAMPLE FULL DIAGNOSTIC\n" + "#" * 78,
            oos_full.rstrip(),
        ]
    ) + "\n"
    comparison = _comparison(baseline_is, baseline_oos, is_metrics, oos_metrics)
    config_sha = _config_fingerprint(mutations)
    generated_at = datetime.now(timezone.utc)
    pre_switch_manifest_bytes = ROUND_MANAGER.manifest_path.read_bytes()

    with tempfile.TemporaryDirectory(prefix=".round_3_first_oos_staging_", dir=STRATEGY_DIR) as temporary:
        stage = Path(temporary)
        evidence_manifest = _copy_evidence(stage)
        _write_json(stage / "optimized_config.json", mutations)
        _write_json(stage / "final_metrics.json", is_metrics)
        _write_json(stage / "oos_metrics.json", oos_metrics)
        _write_json(stage / "final_trades.json", is_rows)
        _write_json(stage / "oos_trades.json", oos_rows)
        for prefix, rows in (("final", is_rows), ("oos", oos_rows)):
            _write_json(
                stage / f"{prefix}_monthly.json",
                _group(rows, "month", lambda row: str(row["exit_time"])[:7]),
            )
            _write_json(
                stage / f"{prefix}_symbols.json",
                _group(rows, "symbol", lambda row: str(row.get("symbol") or "UNKNOWN")),
            )
            _write_json(
                stage / f"{prefix}_exits.json",
                _group(
                    rows,
                    "exit_reason",
                    lambda row: str(row.get("exit_reason") or "UNKNOWN"),
                ),
            )
        _write_json(
            stage / "diagnostics_summary.json",
            {
                "in_sample": is_trade_diagnostics,
                "out_of_sample": oos_trade_diagnostics,
            },
        )
        _write_json(stage / "candidate_comparison.json", comparison)
        (stage / "is_full_diagnostics.txt").write_text(is_full, encoding="utf-8")
        (stage / "oos_full_diagnostics.txt").write_text(oos_full, encoding="utf-8")
        (stage / "round_final_diagnostics.txt").write_text(combined_diagnostics, encoding="utf-8")
        (stage / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")

        phase_state = {
            "round": 3,
            "round_name": "round_3_first_oos_optimal_candidate",
            "current_phase": 1,
            "completed_phases": [1],
            "cumulative_mutations": mutations,
            "phase_results": {
                "1": {
                    "focus": "Initial OOS cumulative ablation and perturbation audit",
                    "catalog_candidates": completion["candidate_count"],
                    "oos_results": completion["oos_result_count"],
                    "is_results": completion["is_result_count"],
                    "selected_candidate": FIRST_OOS_NAME,
                    "selected_patch": {"param_overrides.rvol_threshold": 1.1},
                    "source": str(RESEARCH_DIR.resolve()),
                }
            },
            "provenance_status": PROVENANCE_STATUS,
            "promotion_authorized": False,
            "live_settings_sync_required": True,
            "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
        }
        _write_json(stage / "phase_state.json", phase_state)
        _write_json(
            stage / "progress.json",
            {
                "status": "complete",
                "round": 3,
                "completed_at_utc": generated_at,
                "selected_candidate": FIRST_OOS_NAME,
                "selection_stage": "initial_oos_audit_before_additional_experiments",
                "promotion_authorized": False,
                "provenance_status": PROVENANCE_STATUS,
                "live_settings_sync_required": True,
                "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            },
        )
        optimization_summary = {
            "materialized_at_utc": generated_at,
            "selected_candidate": FIRST_OOS_NAME,
            "selection_stage": "initial_oos_audit_before_additional_experiments",
            "selected_patch": {"param_overrides.rvol_threshold": 1.1},
            "optimized_config_sha256": config_sha,
            "mutation_count": len(mutations),
            "aggregate_result": selected,
            "baseline_is_metrics": baseline_is,
            "baseline_oos_metrics": baseline_oos,
            "comparison": comparison,
            "verified_ablation_candidate_count": completion["candidate_count"],
            "evidence_file_count": evidence_manifest["file_count"],
            "data_authority": "derived_legacy_rth_projection_diagnostic_only",
            "oos_status": "consumed_selection_window",
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "required_revalidation": "accepted_frozen_direct_rth_lockbox",
            "live_settings_sync_required": True,
            "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            "replaces_candidate": REPLACED_NAME,
            "archive_of_replaced_round3": None,
        }
        _write_json(stage / "final_optimization_summary.json", optimization_summary)

        ROUND_MANAGER.write_run_spec(
            stage,
            3,
            "alcb",
            description=(
                "First OOS optimal candidate selected by the initial cumulative "
                "ablation/perturbation audit before additional targeted experiments"
            ),
            baseline_mutations=round2_mutations,
            baseline_source=ROUND2_DIR / "optimized_config.json",
            scoring_weights={
                "expected_total_r": 0.27,
                "net_profit": 0.23,
                "expectancy": 0.14,
                "trades_per_month": 0.16,
                "profit_factor": 0.09,
                "win_rate": 0.04,
                "drawdown": 0.07,
            },
            execution_context={
                "data_dir": str(DATA_DIR.resolve()),
                "initial_equity": INITIAL_EQUITY,
                "is_start": IS_START,
                "is_end": IS_END,
                "oos_start": OOS_START,
                "oos_end": OOS_END,
                "economic_parity": "no shadow tracker",
                "research_dir": str(RESEARCH_DIR.resolve()),
                "selection_stage": "initial_oos_audit_before_additional_experiments",
                "oos_status": "consumed_selection_window",
                "live_settings_sync_required": True,
                "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            },
            provenance=is_provenance,
            provenance_status=PROVENANCE_STATUS,
            overwrite=True,
        )
        ROUND_MANAGER.write_run_summary(
            stage,
            mutations,
            is_metrics,
            [1],
            round_num=3,
            source_diagnostics=ROUND3_DIR / "round_final_diagnostics.txt",
            source_phase_state=ROUND3_DIR / "phase_state.json",
            provenance=is_provenance,
            provenance_status=PROVENANCE_STATUS,
            provenance_validation={
                "valid": True,
                "status": "fresh_materialization_matches_initial_oos_research",
                "selection_drift": False,
                "diagnostics_drift": False,
                "message": (
                    "Fresh no-shadow IS/OOS runs exactly match the initial OOS "
                    "recommendation; external direct-RTH revalidation remains required."
                ),
            },
        )
        run_summary_path = stage / "run_summary.json"
        run_summary = _load(run_summary_path)
        run_summary.update(
            {
                "selected_candidate": FIRST_OOS_NAME,
                "selection_stage": "initial_oos_audit_before_additional_experiments",
                "selected_patch": {"param_overrides.rvol_threshold": 1.1},
                "optimized_config_sha256": config_sha,
                "out_of_sample_metrics": oos_metrics,
                "out_of_sample_provenance": oos_provenance.to_dict(),
                "promotion_authorized": False,
                "production_deployment_approved": False,
                "required_revalidation": "accepted_frozen_direct_rth_lockbox",
                "live_settings_sync_required": True,
                "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
                "replaces_candidate": REPLACED_NAME,
                "archive_of_replaced_round3": None,
                "artifacts": {
                    "round_final_diagnostics": str((ROUND3_DIR / "round_final_diagnostics.txt").resolve()),
                    "is_full_diagnostics": str((ROUND3_DIR / "is_full_diagnostics.txt").resolve()),
                    "oos_full_diagnostics": str((ROUND3_DIR / "oos_full_diagnostics.txt").resolve()),
                    "diagnostics_summary": str((ROUND3_DIR / "diagnostics_summary.json").resolve()),
                    "final_metrics": str((ROUND3_DIR / "final_metrics.json").resolve()),
                    "oos_metrics": str((ROUND3_DIR / "oos_metrics.json").resolve()),
                    "final_trades": str((ROUND3_DIR / "final_trades.json").resolve()),
                    "oos_trades": str((ROUND3_DIR / "oos_trades.json").resolve()),
                    "selection_evidence": str((ROUND3_DIR / "selection_evidence").resolve()),
                },
            }
        )
        _write_json(run_summary_path, run_summary)
        _write_json(stage / "artifact_manifest.json", _artifact_manifest(stage))

        # Destructive-looking work starts only here, after every expensive and
        # diagnostic step above has succeeded.  archive_rounds is recoverable.
        print("archiving current targeted-candidate Round 3", flush=True)
        archive_dir = ROUND_MANAGER.archive_rounds(
            [3],
            reason="replace_targeted_round3_with_first_oos",
        )
        archived_round3 = archive_dir / "round_3"
        if not archived_round3.is_dir():
            raise RuntimeError(f"RoundManager did not create expected archive: {archived_round3}")
        (archive_dir / "rounds_manifest_pre_switch.json").write_bytes(pre_switch_manifest_bytes)
        shutil.move(str(stage), str(ROUND3_DIR))

    manifest_path = ROUND_MANAGER.append_to_manifest(
        3,
        mutations,
        is_metrics,
        provenance=is_provenance,
        provenance_status=PROVENANCE_STATUS,
        round_metadata={
            "round_dir": str(ROUND3_DIR.resolve()),
            "baseline_round": 2,
            "selected_candidate": FIRST_OOS_NAME,
            "selection_stage": "initial_oos_audit_before_additional_experiments",
            "selected_patch": {"param_overrides.rvol_threshold": 1.1},
            "optimized_config_sha256": config_sha,
            "expected_total_r": is_metrics["expected_total_r"],
            "net_profit": is_metrics["net_profit"],
            "expectancy": is_metrics["expectancy"],
            "trades_per_month": is_metrics["trades_per_month"],
            "sortino": is_metrics["sortino"],
            "tail_loss_r": is_metrics["tail_loss_r"],
            "out_of_sample_metrics": oos_metrics,
            "verified_ablation_candidate_count": completion["candidate_count"],
            "research_dir": str(RESEARCH_DIR.resolve()),
            "selection_evidence_dir": str((ROUND3_DIR / "selection_evidence").resolve()),
            "data_authority": "derived_legacy_rth_projection_diagnostic_only",
            "oos_status": "consumed_selection_window",
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "required_revalidation": "accepted_frozen_direct_rth_lockbox",
            "live_settings_sync_required": True,
            "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            "replaces_candidate": REPLACED_NAME,
            "archive_of_replaced_round3": str(archived_round3.resolve()),
        },
    )

    manifest = _load(manifest_path)
    active_entries = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    archived_entries = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3
        and row.get("archived")
        and row.get("selected_candidate") == REPLACED_NAME
    ]
    if len(active_entries) != 1 or active_entries[0].get("selected_candidate") != FIRST_OOS_NAME:
        raise RuntimeError("Manifest does not contain the expected active first-OOS Round 3.")
    if not archived_entries:
        raise RuntimeError("Manifest lost the archived targeted-candidate Round 3 lineage.")
    archived_entry = max(archived_entries, key=lambda row: str(row.get("archived_at_utc", "")))
    archived_entry["archive_dir"] = str(archived_round3.resolve())
    archived_entry["superseded_by_candidate"] = FIRST_OOS_NAME
    _write_json(manifest_path, manifest)
    shutil.copy2(manifest_path, archive_dir / "rounds_manifest_after_switch.json")

    for name in ("run_summary.json", "final_optimization_summary.json"):
        path = ROUND3_DIR / name
        payload = _load(path)
        payload["archive_of_replaced_round3"] = str(archived_round3.resolve())
        _write_json(path, payload)

    archive_errors = _archive_hash_errors(archived_round3)
    validation = {
        "validated_at_utc": datetime.now(timezone.utc),
        "round_dir_exists": ROUND3_DIR.is_dir(),
        "archived_round_dir_exists": archived_round3.is_dir(),
        "archived_artifact_hashes_match": not archive_errors,
        "archived_artifact_hash_errors": archive_errors,
        "single_active_round3_manifest_entry": len(active_entries) == 1,
        "active_candidate_matches": active_entries[0].get("selected_candidate") == FIRST_OOS_NAME,
        "archived_candidate_preserved": bool(archived_entries),
        "manifest_config_matches": active_entries[0].get("mutations") == mutations,
        "optimized_config_matches": _load(ROUND3_DIR / "optimized_config.json") == mutations,
        "optimized_config_sha256_matches": _config_fingerprint(
            _load(ROUND3_DIR / "optimized_config.json")
        ) == config_sha,
        "is_trade_count_matches": len(_load(ROUND3_DIR / "final_trades.json"))
        == int(is_metrics["total_trades"]),
        "oos_trade_count_matches": len(_load(ROUND3_DIR / "oos_trades.json"))
        == int(oos_metrics["total_trades"]),
        "fresh_is_matches_research": True,
        "fresh_oos_matches_research": True,
        "archive_dir": str(archive_dir.resolve()),
    }
    required_checks = [
        value
        for key, value in validation.items()
        if key.endswith(("_exists", "_match", "_matches", "_entry", "_preserved"))
    ]
    if not all(required_checks):
        raise RuntimeError(f"Replacement validation failed: {validation}")
    _write_json(ROUND3_DIR / "validation.json", validation)
    _write_json(
        ROUND3_DIR / "manifest_update_receipt.json",
        {
            "updated_at_utc": datetime.now(timezone.utc),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "archive_dir": str(archive_dir.resolve()),
            "archived_round3": str(archived_round3.resolve()),
            "pre_switch_manifest": str((archive_dir / "rounds_manifest_pre_switch.json").resolve()),
            "post_switch_manifest": str((archive_dir / "rounds_manifest_after_switch.json").resolve()),
            "round": 3,
        },
    )
    _write_json(ROUND3_DIR / "artifact_manifest.json", _artifact_manifest(ROUND3_DIR))
    print(
        json.dumps(
            {
                "round_dir": str(ROUND3_DIR.resolve()),
                "archived_round3": str(archived_round3.resolve()),
                "manifest": str(manifest_path.resolve()),
                "selected_candidate": FIRST_OOS_NAME,
                "is_expected_total_r": is_metrics["expected_total_r"],
                "oos_expected_total_r": oos_metrics["expected_total_r"],
                "promotion_authorized": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
