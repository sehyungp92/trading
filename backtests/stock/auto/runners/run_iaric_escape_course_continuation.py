"""Complete IARIC Round 3 without collapsing onto one local structure.

This continuation deliberately is not part of the escape runner's replay-code
fingerprint.  It waits for that runner, resumes it from its source-fingerprinted
cache if it stopped before producing a final result, and then broadens only the
management/validation beam.  No strategy or replay semantics are changed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.runners import run_iaric_escape_round3 as escape
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _signature,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/iaric/round_3/escape_round"
VALIDATION_LIMIT = 8


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            import os

            os.kill(pid, 0)
            return True
        except OSError:
            return False
    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        int(pid),
    )
    if not handle:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return True


def _cache_snapshot(output: Path) -> dict[str, Any]:
    cache_path = output / "evaluation_cache.json"
    payload = _load_json(cache_path, {})
    evaluations = payload.get("evaluations", {}) if isinstance(payload, dict) else {}
    progress = _load_json(output / "progress.json", {})
    return {
        "cached_evaluations": len(evaluations) if isinstance(evaluations, dict) else 0,
        "cache_updated_at_utc": payload.get("updated_at_utc") if isinstance(payload, dict) else None,
        "primary_progress": progress if isinstance(progress, dict) else {},
    }


def _heartbeat(output: Path, status: str, **extra: Any) -> None:
    _write_json(
        output / "course_progress.json",
        {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_cache_snapshot(output),
            **extra,
        },
    )


def _primary_complete(output: Path) -> bool:
    progress = _load_json(output / "progress.json", {})
    selection = _load_json(output / "final_selection.json")
    return (
        isinstance(progress, dict)
        and progress.get("status") != "running"
        and isinstance(selection, dict)
        and isinstance(selection.get("selected"), dict)
    )


def _resume_primary(
    output: Path,
    max_workers: int,
    attempt: int,
    poll_seconds: int,
) -> int:
    stdout_path = output / "background_stdout.log"
    stderr_path = output / "background_stderr.log"
    command = [
        sys.executable,
        "-m",
        "backtests.stock.auto.runners.run_iaric_escape_round3",
        "--start-date",
        escape.START_DATE,
        "--end-date",
        escape.END_DATE,
        "--max-workers",
        str(max_workers),
        "--output-dir",
        str(output),
    ]
    _heartbeat(output, "resuming_primary", restart_attempt=attempt)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        stdout.write(
            f"[{datetime.now(timezone.utc).isoformat()}] Course watchdog resuming "
            f"the primary escape search (attempt {attempt}).\n"
        )
        stdout.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
        )
        while process.poll() is None:
            _heartbeat(
                output,
                "resuming_primary",
                restart_attempt=attempt,
                resumed_primary_pid=process.pid,
            )
            time.sleep(max(int(poll_seconds), 5))
    return int(process.returncode or 0)


def _wait_or_resume_primary(args: argparse.Namespace, output: Path) -> None:
    watched_pid = int(args.wait_for_pid)
    while watched_pid > 0 and _pid_exists(watched_pid):
        _heartbeat(output, "waiting_for_primary", watched_pid=watched_pid)
        time.sleep(max(int(args.poll_seconds), 5))

    if _primary_complete(output):
        return

    for attempt in range(1, max(int(args.max_restarts), 0) + 1):
        exit_code = _resume_primary(
            output,
            int(args.max_workers),
            attempt,
            int(args.poll_seconds),
        )
        if _primary_complete(output):
            return
        _heartbeat(
            output,
            "primary_restart_incomplete",
            restart_attempt=attempt,
            exit_code=exit_code,
        )
        time.sleep(min(15 * attempt, 45))
    raise RuntimeError(
        "Primary escape search stopped without a final result and exhausted "
        f"{args.max_restarts} cache-preserving restart attempts"
    )


def _family_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in row.get("families", [])))


def _metric_vector(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row.get("metrics", {})
    return (
        float(row.get("escape_score", -99.0)),
        float(metrics.get("expected_total_r", -1e9)),
        float(metrics.get("total_trades", 0.0)),
        float(metrics.get("profit_factor", 0.0)),
        -float(metrics.get("max_drawdown_pct", 1.0)),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = _metric_vector(left)
    right_values = _metric_vector(right)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _pareto_front(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    return [
        row
        for row in values
        if not any(
            _signature(other["mutations"]) != _signature(row["mutations"])
            and _dominates(other, row)
            for other in values
        )
    ]


def _diverse_management_parents(
    phase2: list[dict[str, Any]],
    control: dict[str, Any],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Keep the best viable entry result for each structural family set."""

    return escape._diverse_structure_shortlist(phase2, limit, control)


def _broad_validation_shortlist(
    rows: list[dict[str, Any]],
    control: dict[str, Any],
    *,
    mandatory_signatures: Iterable[str] = (),
    limit: int = VALIDATION_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Select a pre-fold beam spanning score, alpha, frequency, risk, and family."""

    viable = escape._shortlist(rows, len(rows), control)
    if not viable:
        return [], {}
    by_signature = {_signature(row["mutations"]): row for row in viable}
    reasons: dict[str, list[str]] = {}
    selected: list[dict[str, Any]] = []

    def admit(row: dict[str, Any] | None, reason: str) -> None:
        if row is None:
            return
        signature = _signature(row["mutations"])
        reasons.setdefault(signature, []).append(reason)
        if all(_signature(existing["mutations"]) != signature for existing in selected):
            selected.append(row)

    for signature in mandatory_signatures:
        admit(by_signature.get(str(signature)), "primary_validated_finalist")
    admit(max(viable, key=lambda row: float(row.get("escape_score", -99.0))), "top_score")
    admit(
        max(viable, key=lambda row: float(row["metrics"].get("expected_total_r", -1e9))),
        "top_total_r",
    )
    admit(
        max(viable, key=lambda row: float(row["metrics"].get("total_trades", 0.0))),
        "top_frequency",
    )
    admit(
        min(viable, key=lambda row: float(row["metrics"].get("max_drawdown_pct", 1.0))),
        "lowest_drawdown",
    )
    best_by_family: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in viable:
        key = _family_key(row)
        if key and (
            key not in best_by_family
            or float(row.get("escape_score", -99.0))
            > float(best_by_family[key].get("escape_score", -99.0))
        ):
            best_by_family[key] = row
    for row in sorted(
        best_by_family.values(),
        key=lambda value: float(value.get("escape_score", -99.0)),
        reverse=True,
    ):
        admit(row, "best_family_structure")
    for row in sorted(
        _pareto_front(viable),
        key=lambda value: float(value.get("escape_score", -99.0)),
        reverse=True,
    ):
        admit(row, "pareto_front")

    mandatory_count = sum(
        signature in by_signature for signature in set(str(value) for value in mandatory_signatures)
    )
    effective_limit = max(int(limit), mandatory_count)
    selected = selected[:effective_limit]
    selected_signatures = {_signature(row["mutations"]) for row in selected}
    return selected, {
        signature: values
        for signature, values in reasons.items()
        if signature in selected_signatures
    }


def _combine_by_signature(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            signature = _signature(row["mutations"])
            previous = combined.get(signature)
            if previous is None or float(row.get("escape_score", -99.0)) > float(
                previous.get("escape_score", -99.0)
            ):
                combined[signature] = row
    return list(combined.values())


def _validated_rank(row: dict[str, Any]) -> tuple[float, ...]:
    folds = row.get("folds", [])
    fold_deltas = [float(fold.get("delta_total_r", -99.0)) for fold in folds]
    metrics = row.get("metrics", {})
    return (
        1.0 if row.get("all_gates_pass") else 0.0,
        float(row.get("escape_score", -99.0)),
        float(sum(value > 0.0 for value in fold_deltas)),
        min(fold_deltas, default=-99.0),
        float(metrics.get("expected_total_r", -1e9)),
        float(metrics.get("total_trades", 0.0)),
        -float(metrics.get("max_drawdown_pct", 1.0)),
    )


def _promote(
    output: Path,
    selected: dict[str, Any],
    control: dict[str, Any],
    status: str,
    course_report: dict[str, Any],
) -> None:
    round3 = output.parent
    diagnostics = escape._diagnostics(selected, control, status)
    diagnostics += (
        "\nCOURSE-CORRECTION VERIFICATION\n"
        "  Management retained the best candidate from each viable structural family set.\n"
        "  Validation retained the primary finalists plus score, return, frequency, drawdown,\n"
        "  family-diversity and Pareto representatives; fold outcomes were not used to build\n"
        "  that shortlist. The sealed holdout remained excluded.\n"
    )
    (output / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    if not selected["all_gates_pass"]:
        return

    _write_json(round3 / "optimized_config.json", selected["mutations"])
    _write_json(
        round3 / "run_summary.json",
        {
            "status": status,
            "selected_id": selected["id"],
            "metrics": selected["metrics"],
            "aperture": selected["aperture"],
            "gates": selected["gates"],
            "holdout_accessed": False,
            "escape_round": "round_3/escape_round/final_selection.json",
            "course_correction": "round_3/escape_round/course_correction.json",
        },
    )
    (round3 / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    manifest_path = round3.parent / "rounds_manifest.json"
    manifest = _load_json(manifest_path, {})
    manifest["active_round"] = 3
    manifest.pop("pending_round_3", None)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "round": 3,
        "status": status,
        "configuration_role": "local_maximum_escape_anchor_plus_diverse_satellites",
        "mutations": selected["mutations"],
        "metrics": selected["metrics"],
        "aperture": selected["aperture"],
        "score_component_count": 7,
        "sealed_holdout": {"start": escape.HOLDOUT_START, "used": False},
        "course_correction": {
            "primary_selected_id": course_report.get("primary_selected_id"),
            "selection_changed": course_report.get("selection_changed"),
            "artifact": "round_3/escape_round/course_correction.json",
        },
        "artifacts": {
            "optimized_config": "round_3/optimized_config.json",
            "full_final_diagnostics": "round_3/round_final_diagnostics.txt",
            "selection": "round_3/escape_round/final_selection.json",
        },
    }
    rounds = list(manifest.get("rounds", []))
    active_index = next(
        (
            index
            for index, existing in enumerate(rounds)
            if int(existing.get("round", -1)) == 3 and not existing.get("archived", False)
        ),
        None,
    )
    if active_index is None:
        rounds.append(entry)
    else:
        rounds[active_index] = entry
    manifest["rounds"] = rounds
    _write_json(manifest_path, manifest)


def _run_continuation(args: argparse.Namespace, output: Path) -> None:
    run_spec = _load_json(output / "run_spec.json")
    phase0 = _load_json(output / "phase_0_route_isolation_results.json")
    phase2 = _load_json(output / "phase_2_discrimination_entry_results.json")
    phase3 = _load_json(output / "phase_3_management_results.json")
    primary_selection = _load_json(output / "final_selection.json")
    primary_finalists = _load_json(output / "validated_finalists.json", [])
    required = (run_spec, phase0, phase2, phase3, primary_selection)
    if any(value is None for value in required):
        raise FileNotFoundError("Primary escape artifacts are incomplete after the primary runner exited")
    if run_spec.get("holdout_accessed") is not False or str(run_spec.get("end_date", "")) >= escape.HOLDOUT_START:
        raise ValueError("Course continuation requires an explicitly excluded sealed holdout")
    if int(run_spec.get("max_workers", 0)) > 2 or int(args.max_workers) > 2:
        raise ValueError("IARIC course continuation is capped at max-workers=2")
    if len(escape.SCORE_SPEC) != 7:
        raise RuntimeError("IARIC escape score must remain exactly seven components")

    control = next(row for row in phase0 if row["id"] == "incumbent_control")
    diverse_parents = _diverse_management_parents(phase2, control, 3)
    if not diverse_parents:
        raise RuntimeError("No diverse viable Phase 2 parents survived for management")
    _heartbeat(
        output,
        "running_diversity_management",
        diverse_parent_ids=[row["id"] for row in diverse_parents],
    )
    eval_args = argparse.Namespace(
        start_date=str(run_spec["start_date"]),
        end_date=str(run_spec["end_date"]),
        max_workers=min(max(int(args.max_workers), 1), 2),
    )
    diverse_management = escape._evaluate(
        "phase_3b_diversity_management",
        escape._management_candidates(diverse_parents),
        args=eval_args,
        output=output,
        source_fingerprint=str(run_spec["source_fingerprint"]),
        code_fingerprint=str(run_spec["code_fingerprint"]),
        control_metrics=control["metrics"],
    )
    combined = _combine_by_signature(phase3, diverse_management)
    mandatory_signatures = [
        _signature(row["mutations"])
        for row in primary_finalists
        if isinstance(row, dict) and isinstance(row.get("mutations"), dict)
    ]
    finalists, selection_reasons = _broad_validation_shortlist(
        combined,
        control,
        mandatory_signatures=mandatory_signatures,
    )
    if not finalists:
        raise RuntimeError("Diversity continuation produced no viable validation finalists")
    finalists = [deepcopy(row) for row in finalists]
    for row in finalists:
        row.pop("folds", None)
        row.pop("gates", None)
        row.pop("all_gates_pass", None)
    _heartbeat(
        output,
        "running_broad_validation",
        validation_finalist_ids=[row["id"] for row in finalists],
    )
    escape._fold_validate(
        finalists,
        control,
        args=eval_args,
        output=output,
        source_fingerprint=str(run_spec["source_fingerprint"]),
        code_fingerprint=str(run_spec["code_fingerprint"]),
    )
    for row in finalists:
        row["gates"] = escape._gates(row, control)
        row["all_gates_pass"] = all(row["gates"].values())
    finalists.sort(key=_validated_rank, reverse=True)
    selected = finalists[0]
    status = (
        "complete_value_verified"
        if selected["all_gates_pass"]
        else "blocked_value_verification"
    )
    primary_selected = primary_selection.get("selected", {})
    course_report = {
        "status": status,
        "primary_selected_id": primary_selected.get("id"),
        "selected_id": selected["id"],
        "selection_changed": selected["id"] != primary_selected.get("id"),
        "primary_escape_score": primary_selected.get("escape_score"),
        "selected_escape_score": selected.get("escape_score"),
        "diverse_management_parent_ids": [row["id"] for row in diverse_parents],
        "management_candidates_considered": len(combined),
        "validation_finalist_ids": [row["id"] for row in finalists],
        "validation_selection_reasons_by_signature": selection_reasons,
        "holdout_accessed": False,
        "score_component_count": len(escape.SCORE_SPEC),
        "value_non_regression": (
            not primary_selected.get("all_gates_pass", False)
            or (
                selected["all_gates_pass"]
                and float(selected.get("escape_score", -99.0))
                >= float(primary_selected.get("escape_score", -99.0))
            )
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not course_report["value_non_regression"]:
        raise RuntimeError("Course correction would regress a value-verified primary selection")

    _write_json(output / "validated_finalists.json", finalists)
    _write_json(
        output / "final_selection.json",
        {"status": status, "selected": selected, "control": control},
    )
    _write_json(output / "course_correction.json", course_report)
    _promote(output, selected, control, status, course_report)
    _write_json(
        output / "progress.json",
        {
            "status": status,
            "selected_id": selected["id"],
            "all_gates_pass": selected["all_gates_pass"],
            "course_correction_complete": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    _heartbeat(
        output,
        "complete",
        selected_id=selected["id"],
        final_status=status,
        all_gates_pass=selected["all_gates_pass"],
    )


def main() -> int:
    args = _args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        _wait_or_resume_primary(args, output)
        _run_continuation(args, output)
        return 0
    except Exception as exc:
        _heartbeat(output, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
