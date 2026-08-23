"""Materialize the fold-validated ALCB candidate as diagnostic Round 3.

The selected candidate was developed against a repaired legacy RTH projection
and the OOS window was consumed during research.  This script therefore saves
the candidate and its complete evidence without claiming production promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin  # noqa: E402


STRATEGY_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "alcb"
ROUND2_DIR = STRATEGY_DIR / "round_2"
ROUND3_DIR = STRATEGY_DIR / "round_3"
VERIFIED_DIR = ROUND2_DIR / "oos_ablation_perturbation_verified_20260816"
FOLLOWUP_DIR = VERIFIED_DIR / "targeted_followup"
WINNER = "r110__entry1330_late_score5_failure_m010"
ROUND_MANAGER = RoundManager("stock", "alcb")
PREVIOUS_PROVENANCE_STATUS = (
    "diagnostic_candidate_materialized_consumed_oos_"
    "direct_rth_revalidation_required"
)
PROVENANCE_STATUS = (
    "provisional_diagnostic_candidate_materialized_consumed_oos_"
    "direct_rth_revalidation_required"
)
LIVE_SETTINGS_MISMATCHES = {
    "entry_window_end": {"candidate": "13:30:00", "live": "12:30:00"},
    "failure_stop_to_r": {"candidate": -0.1, "live": -0.25},
    "late_entry_cutoff": {"candidate": "12:30:00", "live": "11:00:00"},
    "late_entry_score_min": {"candidate": 5, "live": 0},
    "rvol_threshold": {"candidate": 1.1, "live": 1.4},
}

EVIDENCE_EXCLUDE = {"background_stdout.log", "background_stderr.log"}
SCRIPT_EVIDENCE = (
    REPO_ROOT / "backtests" / "scripts" / "alcb_round2_oos_robustness.py",
    REPO_ROOT / "backtests" / "scripts" / "alcb_round2_targeted_followup.py",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_named(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result for {name!r}, found {len(matches)}.")
    return matches[0]


def _window_metrics(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in CORE_METRICS:
        source = f"{prefix}_{key}"
        if source in row:
            output[key] = row[source]
    return output


def _assert_metrics_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"{key}=missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, int) and not isinstance(expected_value, bool):
            if int(actual_value) != expected_value:
                mismatches.append(f"{key}: {actual_value!r} != {expected_value!r}")
            continue
        try:
            delta = abs(float(actual_value) - float(expected_value))
            tolerance = max(1e-9, abs(float(expected_value)) * 1e-10)
            if delta > tolerance:
                mismatches.append(f"{key}: {actual_value!r} != {expected_value!r}")
        except (TypeError, ValueError):
            if actual_value != expected_value:
                mismatches.append(f"{key}: {actual_value!r} != {expected_value!r}")
    if mismatches:
        raise RuntimeError(f"Fresh {label} result drifted from research: {'; '.join(mismatches)}")


def _run_window(
    mutations: dict[str, Any],
    start: str,
    end: str,
) -> tuple[dict[str, Any], Any]:
    plugin = ALCBP16Plugin(
        DATA_DIR,
        start_date=start,
        end_date=end,
        initial_equity=INITIAL_EQUITY,
        max_workers=1,
        allow_projected_rth_data=True,
    )
    try:
        # Shadow tracking changes realized economics in the current engine.  Use
        # the worker-equivalent path and diagnose the resulting trades afterward.
        context = plugin._run_config(
            mutations,
            store_context=False,
            collect_diagnostics=False,
        )
        provenance = plugin.build_provenance()
        return context, provenance
    finally:
        plugin.close_pool()


def _group(
    rows: list[dict[str, Any]],
    label: str,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        rs = [float(row.get("r_multiple", 0.0) or 0.0) for row in values]
        pnls = [float(row.get("pnl_net", 0.0) or 0.0) for row in values]
        output.append(
            {
                label: key,
                "trades": len(values),
                "share": len(values) / len(rows) if rows else 0.0,
                "win_rate": sum(value > 0.0 for value in pnls) / len(values),
                "avg_r": mean(rs),
                "total_r": sum(rs),
                "pnl_net": sum(pnls),
            }
        )
    if label == "month":
        return sorted(output, key=lambda row: row[label])
    if label == "exit_reason":
        return sorted(output, key=lambda row: (-row["trades"], row[label]))
    return sorted(output, key=lambda row: (-row["pnl_net"], row[label]))


def _comparison(
    baseline: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for window in ("is", "oos"):
        base_metrics = _window_metrics(baseline, window)
        candidate_metrics = _window_metrics(selected, window)
        output[window] = {}
        for key in CORE_METRICS:
            if key not in base_metrics or key not in candidate_metrics:
                continue
            old = base_metrics[key]
            new = candidate_metrics[key]
            try:
                delta = float(new) - float(old)
                relative = delta / abs(float(old)) if abs(float(old)) > 1e-12 else None
            except (TypeError, ValueError):
                delta = None
                relative = None
            output[window][key] = {
                "round2_control": old,
                "round3_candidate": new,
                "delta": delta,
                "delta_pct": relative,
            }
    return output


def _evaluation_text(
    selected: dict[str, Any],
    baseline: dict[str, Any],
    folds: list[dict[str, Any]],
) -> str:
    is_base = _window_metrics(baseline, "is")
    is_new = _window_metrics(selected, "is")
    oos_base = _window_metrics(baseline, "oos")
    oos_new = _window_metrics(selected, "oos")
    lines = [
        "ALCB ROUND 3 FINAL MATERIALIZATION",
        "=" * 78,
        f"Selected candidate: {WINNER}",
        f"Provenance status: {PROVENANCE_STATUS}",
        "Production deployment approved: NO",
        "Reason: legacy projected-RTH research data and consumed OOS window.",
        "Required next gate: accepted frozen direct-RTH lockbox revalidation.",
        "",
        "CUMULATIVE PATCH APPLIED TO ROUND 2",
        "-" * 78,
    ]
    for key, value in sorted(selected["patch"].items()):
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "AGGREGATE ECONOMIC-PARITY RESULTS",
            "-" * 78,
            (
                f"  IS:  R {is_base['expected_total_r']:+.2f} -> {is_new['expected_total_r']:+.2f}; "
                f"trades {is_base['total_trades']} -> {is_new['total_trades']}; "
                f"TPM {is_base['trades_per_month']:.1f} -> {is_new['trades_per_month']:.1f}; "
                f"PF {is_base['profit_factor']:.2f} -> {is_new['profit_factor']:.2f}; "
                f"WR {is_base['win_rate']:.1%} -> {is_new['win_rate']:.1%}; "
                f"DD {is_base['max_drawdown_pct']:.2%} -> {is_new['max_drawdown_pct']:.2%}"
            ),
            (
                f"  OOS: R {oos_base['expected_total_r']:+.2f} -> {oos_new['expected_total_r']:+.2f}; "
                f"trades {oos_base['total_trades']} -> {oos_new['total_trades']}; "
                f"TPM {oos_base['trades_per_month']:.1f} -> {oos_new['trades_per_month']:.1f}; "
                f"PF {oos_base['profit_factor']:.2f} -> {oos_new['profit_factor']:.2f}; "
                f"WR {oos_base['win_rate']:.1%} -> {oos_new['win_rate']:.1%}; "
                f"DD {oos_base['max_drawdown_pct']:.2%} -> {oos_new['max_drawdown_pct']:.2%}"
            ),
            "",
            "FOLD ROBUSTNESS",
            "-" * 78,
        ]
    )
    for fold in folds:
        metrics = fold["metrics"]
        lines.append(
            f"  {fold['fold']}: utility {fold['utility']:+.4f}; "
            f"delta R {fold['delta_expected_total_r']:+.2f}; "
            f"delta TPM {fold['delta_trades_per_month']:+.1f}; "
            f"PF {metrics['profit_factor']:.2f}; DD {metrics['max_drawdown_pct']:.2%}"
        )
    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 78,
            "  The candidate maximizes the requested return/frequency objective among the",
            "  fold-validated targeted set. Its uplift is broad rather than a single-tail",
            "  rescue: all three IS folds add R and frequency. The trade-off is lower win",
            "  rate and lower OOS PF/expectancy than Round 2, although OOS total R, net PnL,",
            "  frequency, drawdown, and Sharpe improve. Treat this as a saved research",
            "  candidate until an untouched accepted direct-RTH window confirms it.",
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_evidence(stage: Path) -> dict[str, Any]:
    evidence_root = stage / "selection_evidence"
    copied: list[dict[str, Any]] = []
    for source_dir, name in (
        (VERIFIED_DIR, "verified_ablation"),
        (FOLLOWUP_DIR, "targeted_followup"),
    ):
        destination = evidence_root / name
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.name in EVIDENCE_EXCLUDE:
                continue
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(
                {
                    "source": str(source.resolve()),
                    "artifact": str(target.relative_to(stage)).replace("\\", "/"),
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
    scripts_dir = evidence_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for source in SCRIPT_EVIDENCE:
        target = scripts_dir / source.name
        shutil.copy2(source, target)
        copied.append(
            {
                "source": str(source.resolve()),
                "artifact": str(target.relative_to(stage)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc),
        "file_count": len(copied),
        "files": copied,
    }
    _write_json(evidence_root / "evidence_manifest.json", manifest)
    return manifest


def _artifact_manifest(round_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(round_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(round_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc),
        "file_count": len(rows),
        "files": rows,
    }


def _replace_previous_status(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(PREVIOUS_PROVENANCE_STATUS, PROVENANCE_STATUS)
    if isinstance(value, list):
        return [_replace_previous_status(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_previous_status(item) for key, item in value.items()}
    return value


def _reconcile_existing_status() -> int:
    """Repair the provisional/live-sync contract after a completed materialization."""
    if not ROUND3_DIR.is_dir():
        raise RuntimeError(f"Round 3 does not exist: {ROUND3_DIR}")
    for name in (
        "phase_state.json",
        "progress.json",
        "run_spec.json",
        "run_summary.json",
        "final_optimization_summary.json",
    ):
        path = ROUND3_DIR / name
        payload = _replace_previous_status(_load(path))
        payload["live_settings_sync_required"] = True
        payload["live_settings_mismatches"] = LIVE_SETTINGS_MISMATCHES
        _write_json(path, payload)
    for name in ("round_evaluation.txt", "round_final_diagnostics.txt"):
        path = ROUND3_DIR / name
        text = path.read_text(encoding="utf-8").replace(
            PREVIOUS_PROVENANCE_STATUS,
            PROVENANCE_STATUS,
        )
        path.write_text(text, encoding="utf-8")

    manifest = _replace_previous_status(_load(ROUND_MANAGER.manifest_path))
    entries = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one active Round 3 manifest entry, found {len(entries)}.")
    entries[0]["provenance_status"] = PROVENANCE_STATUS
    entries[0]["live_settings_sync_required"] = True
    entries[0]["live_settings_mismatches"] = LIVE_SETTINGS_MISMATCHES
    _write_json(ROUND_MANAGER.manifest_path, manifest)

    old_receipt = _load(ROUND3_DIR / "manifest_update_receipt.json")
    backup = Path(old_receipt["backup"])
    _write_json(
        ROUND3_DIR / "manifest_update_receipt.json",
        {
            "updated_at_utc": datetime.now(timezone.utc),
            "manifest": str(ROUND_MANAGER.manifest_path.resolve()),
            "manifest_sha256": _sha256(ROUND_MANAGER.manifest_path),
            "backup": str(backup.resolve()),
            "backup_sha256": _sha256(backup),
            "round": 3,
            "reconciliation": "explicit provisional/live-settings contract",
        },
    )
    _write_json(ROUND3_DIR / "artifact_manifest.json", _artifact_manifest(ROUND3_DIR))
    print(f"reconciled provisional Round 3 status in {ROUND3_DIR}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-consumed-oos",
        action="store_true",
        help="Acknowledge that OOS was consumed and Round 3 is diagnostic-only.",
    )
    parser.add_argument(
        "--reconcile-existing-status",
        action="store_true",
        help="Reconcile an existing Round 3 to the explicit provisional/live-sync contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.reconcile_existing_status:
        return _reconcile_existing_status()
    if not args.allow_consumed_oos:
        raise SystemExit("Pass --allow-consumed-oos to save this diagnostic-only candidate.")
    if ROUND3_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing Round 3: {ROUND3_DIR}")
    active_round3 = [
        row
        for row in ROUND_MANAGER.load_manifest().get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    if active_round3:
        raise RuntimeError("The manifest already contains an active Round 3 entry.")

    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    verified_completion = _load(VERIFIED_DIR / "completion.json")
    followup_completion = _load(FOLLOWUP_DIR / "completion.json")
    if not verified_completion.get("candidate_count") == 263:
        raise RuntimeError("Verified ablation catalog is incomplete.")
    if not followup_completion.get("complete"):
        raise RuntimeError("Targeted follow-up is incomplete.")
    if followup_completion.get("leading_name") != WINNER:
        raise RuntimeError("Targeted follow-up winner changed; refusing stale materialization.")

    aggregate = _load(FOLLOWUP_DIR / "aggregate_results.json")
    selected = _find_named(aggregate, WINNER)
    baseline = _find_named(aggregate, "control__base")
    finalists = _load(FOLLOWUP_DIR / "finalists.json")
    selected_finalist = _find_named(finalists, WINNER)
    folds = list(selected_finalist["folds"])
    if len(folds) != 3 or selected_finalist.get("positive_expected_r_folds") != 3:
        raise RuntimeError("Winner no longer has positive expected-R uplift in all three folds.")

    round2_mutations = _load(ROUND2_DIR / "optimized_config.json")
    mutations = {**round2_mutations, **selected["patch"]}
    if len(round2_mutations) != 50 or len(mutations) != 52:
        raise RuntimeError(
            "Unexpected cumulative mutation count; expected 50 Round-2 keys plus "
            "two new late-entry quality keys."
        )

    print(f"running fresh IS {IS_START}..{IS_END}", flush=True)
    is_context, is_provenance = _run_window(mutations, IS_START, IS_END)
    is_metrics = dict(is_context["metrics"])
    _assert_metrics_match(is_metrics, _window_metrics(selected, "is"), label="IS")
    print(
        f"IS verified: {is_metrics['total_trades']} trades, "
        f"{is_metrics['expected_total_r']:+.2f}R",
        flush=True,
    )

    print(f"running fresh OOS {OOS_START}..{OOS_END}", flush=True)
    oos_context, oos_provenance = _run_window(mutations, OOS_START, OOS_END)
    oos_metrics = dict(oos_context["metrics"])
    _assert_metrics_match(oos_metrics, _window_metrics(selected, "oos"), label="OOS")
    print(
        f"OOS verified: {oos_metrics['total_trades']} trades, "
        f"{oos_metrics['expected_total_r']:+.2f}R",
        flush=True,
    )

    is_trades = list(is_context["trades"])
    oos_trades = list(oos_context["trades"])
    is_rows = [_trade_to_dict(trade) for trade in is_trades]
    oos_rows = [_trade_to_dict(trade) for trade in oos_trades]
    is_diagnostics = _trade_diagnostics(is_trades)
    oos_diagnostics = _trade_diagnostics(oos_trades)
    if not _diagnostics_consistent(is_diagnostics, is_metrics):
        raise RuntimeError("Fresh IS trade diagnostics do not match headline metrics.")
    if not _diagnostics_consistent(oos_diagnostics, oos_metrics):
        raise RuntimeError("Fresh OOS trade diagnostics do not match headline metrics.")

    evaluation = _evaluation_text(selected, baseline, folds)
    print("building full IS/OOS diagnostic reports", flush=True)
    max_positions = int(is_context["config"].param_overrides.get("max_positions", 10))
    full_diagnostics = "\n\n".join(
        [
            evaluation.rstrip(),
            "\n" + "#" * 78 + "\nIN-SAMPLE FULL DIAGNOSTIC\n" + "#" * 78,
            alcb_full_diagnostic(
                is_trades,
                shadow_tracker=None,
                daily_selections=is_context.get("daily_selections"),
            ),
            qe_replacement_analysis(is_trades, max_positions=max_positions),
            "\n" + "#" * 78 + "\nOUT-OF-SAMPLE FULL DIAGNOSTIC\n" + "#" * 78,
            alcb_full_diagnostic(
                oos_trades,
                shadow_tracker=None,
                daily_selections=oos_context.get("daily_selections"),
            ),
            qe_replacement_analysis(oos_trades, max_positions=max_positions),
        ]
    ) + "\n"

    generated_at = datetime.now(timezone.utc)
    comparison = _comparison(baseline, selected)
    config_sha = _config_fingerprint(mutations)
    with tempfile.TemporaryDirectory(prefix=".round_3_staging_", dir=STRATEGY_DIR) as temporary:
        stage = Path(temporary)
        evidence_manifest = _copy_evidence(stage)
        _write_json(stage / "optimized_config.json", mutations)
        _write_json(stage / "final_metrics.json", is_metrics)
        _write_json(stage / "oos_metrics.json", oos_metrics)
        _write_json(stage / "final_trades.json", is_rows)
        _write_json(stage / "oos_trades.json", oos_rows)
        for prefix, rows in (("final", is_rows), ("oos", oos_rows)):
            _write_json(stage / f"{prefix}_monthly.json", _group(rows, "month", lambda row: str(row["exit_time"])[:7]))
            _write_json(stage / f"{prefix}_symbols.json", _group(rows, "symbol", lambda row: str(row.get("symbol") or "UNKNOWN")))
            _write_json(stage / f"{prefix}_exits.json", _group(rows, "exit_reason", lambda row: str(row.get("exit_reason") or "UNKNOWN")))
        _write_json(
            stage / "diagnostics_summary.json",
            {
                "in_sample": is_diagnostics,
                "out_of_sample": oos_diagnostics,
            },
        )
        _write_json(stage / "candidate_comparison.json", comparison)
        (stage / "round_final_diagnostics.txt").write_text(full_diagnostics, encoding="utf-8")
        (stage / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")

        phase_state = {
            "round": 3,
            "round_name": "round_3_ablation_perturbation_and_targeted_followup",
            "current_phase": 2,
            "completed_phases": [1, 2],
            "cumulative_mutations": mutations,
            "phase_results": {
                "1": {
                    "focus": "Granular cumulative ablation and perturbation",
                    "catalog_candidates": verified_completion["candidate_count"],
                    "oos_results": verified_completion["oos_result_count"],
                    "is_results": verified_completion["is_result_count"],
                    "source": str(VERIFIED_DIR.resolve()),
                },
                "2": {
                    "focus": "Targeted weakness repair and fold validation",
                    "catalog_candidates": followup_completion["candidate_count"],
                    "shortlist_count": followup_completion["shortlist_count"],
                    "eligible_count": followup_completion["eligible_count"],
                    "selected_candidate": WINNER,
                    "selected_patch": selected["patch"],
                    "positive_expected_r_folds": selected_finalist["positive_expected_r_folds"],
                    "source": str(FOLLOWUP_DIR.resolve()),
                },
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
                "selected_candidate": WINNER,
                "promotion_authorized": False,
                "provenance_status": PROVENANCE_STATUS,
                "live_settings_sync_required": True,
                "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            },
        )
        optimization_summary = {
            "materialized_at_utc": generated_at,
            "selected_candidate": WINNER,
            "selected_patch": selected["patch"],
            "optimized_config_sha256": config_sha,
            "mutation_count": len(mutations),
            "aggregate_result": selected,
            "fold_validation": selected_finalist,
            "baseline_result": baseline,
            "comparison": comparison,
            "verified_ablation_candidate_count": verified_completion["candidate_count"],
            "targeted_candidate_count": followup_completion["candidate_count"],
            "evidence_file_count": evidence_manifest["file_count"],
            "data_authority": "derived_legacy_rth_projection_diagnostic_only",
            "oos_status": "consumed_development_window",
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "required_revalidation": "accepted_frozen_direct_rth_lockbox",
            "live_settings_sync_required": True,
            "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
        }
        _write_json(stage / "final_optimization_summary.json", optimization_summary)

        ROUND_MANAGER.write_run_spec(
            stage,
            3,
            "alcb",
            description=(
                "Cumulative mutation ablation/perturbation plus targeted weakness repair; "
                "diagnostic candidate materialization"
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
                "fold_stability": "required for finalists",
            },
            execution_context={
                "data_dir": str(DATA_DIR.resolve()),
                "initial_equity": INITIAL_EQUITY,
                "is_start": IS_START,
                "is_end": IS_END,
                "oos_start": OOS_START,
                "oos_end": OOS_END,
                "economic_parity": "no shadow tracker",
                "verified_research_dir": str(VERIFIED_DIR.resolve()),
                "targeted_followup_dir": str(FOLLOWUP_DIR.resolve()),
                "oos_status": "consumed_development_window",
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
            [1, 2],
            round_num=3,
            source_diagnostics=ROUND3_DIR / "round_final_diagnostics.txt",
            source_phase_state=ROUND3_DIR / "phase_state.json",
            provenance=is_provenance,
            provenance_status=PROVENANCE_STATUS,
            provenance_validation={
                "valid": True,
                "status": "fresh_materialization_matches_research",
                "selection_drift": False,
                "diagnostics_drift": False,
                "message": (
                    "Fresh no-shadow IS/OOS economic runs exactly match the targeted "
                    "research metrics; external direct-RTH revalidation remains required."
                ),
            },
        )
        run_summary_path = stage / "run_summary.json"
        run_summary = _load(run_summary_path)
        run_summary.update(
            {
                "selected_candidate": WINNER,
                "selected_patch": selected["patch"],
                "optimized_config_sha256": config_sha,
                "out_of_sample_metrics": oos_metrics,
                "out_of_sample_provenance": oos_provenance.to_dict(),
                "fold_validation": selected_finalist,
                "promotion_authorized": False,
                "production_deployment_approved": False,
                "required_revalidation": "accepted_frozen_direct_rth_lockbox",
                "live_settings_sync_required": True,
                "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
                "artifacts": {
                    "round_final_diagnostics": str((ROUND3_DIR / "round_final_diagnostics.txt").resolve()),
                    "round_evaluation": str((ROUND3_DIR / "round_evaluation.txt").resolve()),
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
        shutil.move(str(stage), str(ROUND3_DIR))

    archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_backup_dir = STRATEGY_DIR / "archive" / f"{archive_stamp}_pre_round3_materialization"
    manifest_backup_dir.mkdir(parents=True, exist_ok=False)
    manifest_backup = manifest_backup_dir / "rounds_manifest.json"
    shutil.copy2(ROUND_MANAGER.manifest_path, manifest_backup)

    manifest_path = ROUND_MANAGER.append_to_manifest(
        3,
        mutations,
        is_metrics,
        provenance=is_provenance,
        provenance_status=PROVENANCE_STATUS,
        round_metadata={
            "round_dir": str(ROUND3_DIR.resolve()),
            "baseline_round": 2,
            "selected_candidate": WINNER,
            "selected_patch": selected["patch"],
            "optimized_config_sha256": config_sha,
            "expected_total_r": is_metrics["expected_total_r"],
            "net_profit": is_metrics["net_profit"],
            "expectancy": is_metrics["expectancy"],
            "trades_per_month": is_metrics["trades_per_month"],
            "sortino": is_metrics["sortino"],
            "tail_loss_r": is_metrics["tail_loss_r"],
            "out_of_sample_metrics": oos_metrics,
            "fold_validation": {
                "positive_expected_r_folds": selected_finalist["positive_expected_r_folds"],
                "mean_fold_utility": selected_finalist["mean_fold_utility"],
                "min_fold_utility": selected_finalist["min_fold_utility"],
            },
            "verified_ablation_candidate_count": verified_completion["candidate_count"],
            "targeted_candidate_count": followup_completion["candidate_count"],
            "research_dir": str(FOLLOWUP_DIR.resolve()),
            "selection_evidence_dir": str((ROUND3_DIR / "selection_evidence").resolve()),
            "data_authority": "derived_legacy_rth_projection_diagnostic_only",
            "oos_status": "consumed_development_window",
            "promotion_authorized": False,
            "production_deployment_approved": False,
            "required_revalidation": "accepted_frozen_direct_rth_lockbox",
            "live_settings_sync_required": True,
            "live_settings_mismatches": LIVE_SETTINGS_MISMATCHES,
            "manifest_backup": str(manifest_backup.resolve()),
        },
    )

    manifest = _load(manifest_path)
    active_entries = [
        row
        for row in manifest.get("rounds", [])
        if int(row.get("round", 0)) == 3 and not row.get("archived")
    ]
    validation = {
        "validated_at_utc": datetime.now(timezone.utc),
        "round_dir_exists": ROUND3_DIR.is_dir(),
        "single_active_manifest_entry": len(active_entries) == 1,
        "manifest_config_matches": bool(active_entries and active_entries[0]["mutations"] == mutations),
        "optimized_config_matches": _load(ROUND3_DIR / "optimized_config.json") == mutations,
        "optimized_config_sha256_matches": _config_fingerprint(_load(ROUND3_DIR / "optimized_config.json")) == config_sha,
        "is_trade_count_matches": len(_load(ROUND3_DIR / "final_trades.json")) == int(is_metrics["total_trades"]),
        "oos_trade_count_matches": len(_load(ROUND3_DIR / "oos_trades.json")) == int(oos_metrics["total_trades"]),
        "fresh_is_matches_research": True,
        "fresh_oos_matches_research": True,
        "manifest_backup": str(manifest_backup.resolve()),
    }
    if not all(value for key, value in validation.items() if key.endswith(("_exists", "_entry", "_matches"))):
        raise RuntimeError(f"Round 3 validation failed: {validation}")
    _write_json(ROUND3_DIR / "validation.json", validation)
    _write_json(
        ROUND3_DIR / "manifest_update_receipt.json",
        {
            "updated_at_utc": datetime.now(timezone.utc),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "backup": str(manifest_backup.resolve()),
            "backup_sha256": _sha256(manifest_backup),
            "round": 3,
        },
    )
    _write_json(ROUND3_DIR / "artifact_manifest.json", _artifact_manifest(ROUND3_DIR))
    print(
        json.dumps(
            {
                "round_dir": str(ROUND3_DIR.resolve()),
                "manifest": str(manifest_path.resolve()),
                "selected_candidate": WINNER,
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
