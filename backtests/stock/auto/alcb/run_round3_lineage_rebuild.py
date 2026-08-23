"""Rebuild ALCB Round 3 from the combined-trail baseline and saved lineage.

The active Round 3 contributes RVOL 1.1.  An archived predecessor also used
an extended entry window, a late-entry quality gate, and a looser failure
stop.  This runner evaluates those exact mutations and a small, pre-declared
set of dependency checks on the pre-consumed development interval.  Nearby
RVOL values are falsification controls and are never selection-eligible.

The output is a recommendation package.  It does not modify round artifacts
or the rounds manifest; materialization is deliberately a separate step.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.alcb.run_baseline_recovery import (
    CONSUMED_START,
    END_DATE,
    FOLDS,
    MAX_WORKERS,
    START_DATE,
    _code_fingerprint,
    _compact_metrics,
    _cost_candidates,
    _cost_summary,
    _evaluate_batch,
    _fold_summary,
    _signature,
    _source_fingerprint,
    _write_json,
)
from backtests.stock.auto.alcb.run_trail_combination_validation import (
    DEFAULT_OUTPUT as TRAIL_OUTPUT,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/alcb/round3_rebuild_20260822"
BASELINE_CONFIG = TRAIL_OUTPUT / "combined_config.json"
SHARED_CACHE = (
    REPO_ROOT
    / "backtests/output/stock/alcb/representative_baseline_20260821/evaluation_cache.json"
)

ACTIVE_ROUND3_PATCH: dict[str, Any] = {
    "param_overrides.rvol_threshold": 1.1,
}
ARCHIVED_ROUND3_PATCH: dict[str, Any] = {
    "param_overrides.entry_window_end": "13:30:00",
    "param_overrides.failure_stop_to_r": -0.1,
    "param_overrides.late_entry_cutoff": "12:30:00",
    "param_overrides.late_entry_score_min": 5,
    **ACTIVE_ROUND3_PATCH,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-path", type=Path, default=SHARED_CACHE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-projected-rth-data", action="store_true")
    return parser.parse_args()


def _read_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _patched(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    config.update(patch)
    return dict(sorted(config.items()))


def _candidate(
    candidate_id: str,
    baseline: dict[str, Any],
    patch: dict[str, Any],
    *,
    family: str,
    role: str,
    selection_eligible: bool = True,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "family": family,
        "era": "round3_lineage_rebuild",
        "role": role,
        "selection_eligible": selection_eligible,
        "patch": dict(sorted(patch.items())),
        "changed_keys": sorted(patch),
        "sources": [str(BASELINE_CONFIG)],
        "mutations": _patched(baseline, patch),
    }


def _catalog(baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    base = _read_dict(BASELINE_CONFIG) if baseline is None else deepcopy(baseline)
    rvol = ACTIVE_ROUND3_PATCH
    entry = {"param_overrides.entry_window_end": "13:30:00"}
    failure = {"param_overrides.failure_stop_to_r": -0.1}
    late_cutoff = {"param_overrides.late_entry_cutoff": "12:30:00"}
    late_score = {"param_overrides.late_entry_score_min": 5}
    late_quality = {**entry, **late_cutoff, **late_score}

    specs: list[tuple[str, dict[str, Any], str, str, bool]] = [
        ("baseline__combined_trail", {}, "control", "combined trail control", True),
        ("atomic__rvol_1p1", rvol, "active_round3", "active Round 3 mutation", True),
        (
            "surface__rvol_1p2",
            {"param_overrides.rvol_threshold": 1.2},
            "surface_control",
            "RVOL stability falsification",
            False,
        ),
        (
            "surface__rvol_1p3",
            {"param_overrides.rvol_threshold": 1.3},
            "surface_control",
            "RVOL stability falsification",
            False,
        ),
        ("atomic__entry_1330", entry, "archived_atomic", "archived mutation", True),
        ("atomic__failure_m010", failure, "archived_atomic", "archived mutation", True),
        (
            "atomic__late_cutoff_1230",
            late_cutoff,
            "archived_atomic",
            "archived mutation",
            True,
        ),
        ("atomic__late_score_5", late_score, "archived_atomic", "archived mutation", True),
        ("pair__rvol_entry", {**rvol, **entry}, "targeted_pair", "dependency check", True),
        (
            "pair__rvol_failure",
            {**rvol, **failure},
            "targeted_pair",
            "dependency check",
            True,
        ),
        (
            "pair__entry_failure",
            {**entry, **failure},
            "targeted_pair",
            "dependency check",
            True,
        ),
        (
            "triple__rvol_entry_failure",
            {**rvol, **entry, **failure},
            "targeted_bundle",
            "dependency check",
            True,
        ),
        (
            "bundle__late_quality",
            late_quality,
            "logical_bundle",
            "late-entry quality mechanism",
            True,
        ),
        (
            "bundle__rvol_late_quality",
            {**rvol, **late_quality},
            "logical_bundle",
            "active plus late-quality mechanism",
            True,
        ),
        (
            "bundle__archived_without_rvol",
            {key: value for key, value in ARCHIVED_ROUND3_PATCH.items() if key not in rvol},
            "archived_bundle",
            "archived Round 3 excluding active RVOL mutation",
            True,
        ),
        (
            "bundle__full_archived_lineage",
            ARCHIVED_ROUND3_PATCH,
            "archived_bundle",
            "complete archived Round 3 patch",
            True,
        ),
    ]
    for key in sorted(ARCHIVED_ROUND3_PATCH):
        short = key.removeprefix("param_overrides.").replace(".", "_")
        patch = {
            item: value
            for item, value in ARCHIVED_ROUND3_PATCH.items()
            if item != key
        }
        if any(existing_patch == patch for _, existing_patch, _, _, _ in specs):
            continue
        specs.append(
            (
                f"loo__full_without_{short}",
                patch,
                "leave_one_out",
                f"full archived patch without {key}",
                False,
            )
        )

    rows = [
        _candidate(
            candidate_id,
            base,
            patch,
            family=family,
            role=role,
            selection_eligible=eligible,
        )
        for candidate_id, patch, family, role, eligible in specs
    ]
    signatures = [_signature(row["mutations"]) for row in rows]
    if len(signatures) != len(set(signatures)):
        duplicates = sorted(sig for sig in set(signatures) if signatures.count(sig) > 1)
        raise RuntimeError(f"Catalog contains duplicate configurations: {duplicates}")
    return rows


def _assert_no_errors(stage: str, rows: list[dict[str, Any]]) -> None:
    errors = [row for row in rows if row.get("error")]
    if errors:
        detail = "\n".join(
            f"{row.get('id')}: {str(row.get('error', '')).splitlines()[-1]}"
            for row in errors
        )
        raise RuntimeError(f"{stage} evaluation failed:\n{detail}")


def _metadata_by_signature(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        _signature(row["mutations"]): {
            key: deepcopy(value)
            for key, value in row.items()
            if key not in {"mutations", "id", "family", "era", "sources"}
        }
        for row in catalog
    }


def _enrich(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    for row in rows:
        row.update(deepcopy(metadata.get(str(row.get("signature", "")), {})))
    return rows


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    cm = candidate["metrics"]
    bm = baseline["metrics"]
    return {
        key: float(cm.get(key, 0.0)) - float(bm.get(key, 0.0))
        for key in (
            "expected_total_r",
            "net_profit",
            "avg_r",
            "profit_factor",
            "win_rate",
            "trades_per_month",
            "max_drawdown_pct",
        )
    }


def _full_incremental_gate(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    cm = candidate["metrics"]
    bm = baseline["metrics"]
    if float(cm["expected_total_r"]) < float(bm["expected_total_r"]) + 2.0:
        reasons.append("less than +2R full-period uplift")
    if float(cm["avg_r"]) < float(bm["avg_r"]) * 0.95:
        reasons.append("retained less than 95% of baseline AvgR")
    if float(cm["profit_factor"]) < float(bm["profit_factor"]) * 0.97:
        reasons.append("retained less than 97% of baseline PF")
    dd_cap = max(
        float(bm["max_drawdown_pct"]) * 1.15,
        float(bm["max_drawdown_pct"]) + 0.005,
    )
    if float(cm["max_drawdown_pct"]) > dd_cap:
        reasons.append("drawdown exceeded the fixed relative cap")
    return not reasons, reasons


def _fold_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["fold"]: item for item in row.get("validation", {}).get("folds", [])}


def _final_incremental_gate(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str]]:
    passes, reasons = _full_incremental_gate(candidate, baseline)
    del passes
    validation = candidate.get("validation", {})
    costs = candidate.get("costs", {})
    baseline_costs = baseline.get("costs", {})
    if not validation.get("robust_eligible"):
        reasons.append("chronological robustness gate failed")
    if not costs.get("seven_five_gate") or not costs.get("ten_gate"):
        reasons.append("absolute 7.5/10 bps cost gate failed")
    for cost in ("7.5", "10.0"):
        if float(costs.get(cost, {}).get("expected_total_r", -1e9)) <= float(
            baseline_costs.get(cost, {}).get("expected_total_r", 0.0)
        ):
            reasons.append(f"did not beat baseline total R at {cost} bps")

    base_folds = _fold_map(baseline)
    candidate_folds = _fold_map(candidate)
    fold_wins = sum(
        float(candidate_folds[name]["expected_total_r"])
        > float(base_folds[name]["expected_total_r"])
        for name in base_folds
        if name in candidate_folds
    )
    candidate["fold_wins_vs_baseline"] = fold_wins
    if len(candidate_folds) != len(FOLDS):
        reasons.append("not all four folds completed")
    elif fold_wins < 3:
        reasons.append("failed to beat baseline total R in at least 3/4 folds")
    return not reasons, reasons


def _rvol_surface_support(
    records: dict[str, dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for candidate_id in ("atomic__rvol_1p1", "surface__rvol_1p2", "surface__rvol_1p3"):
        row = records.get(candidate_id)
        if not row:
            continue
        passed, reasons = _final_incremental_gate(row, baseline)
        rows.append(
            {
                "id": candidate_id,
                "threshold": row["patch"]["param_overrides.rvol_threshold"],
                "passes_incremental_gate": passed,
                "gate_reasons": reasons,
                "metrics": _compact_metrics(row["metrics"]),
                "fold_wins_vs_baseline": row.get("fold_wins_vs_baseline", 0),
                "costs": row.get("costs", {}),
            }
        )
    exact = next((row for row in rows if row["threshold"] == 1.1), None)
    neighbors = [row for row in rows if row["threshold"] != 1.1]
    supported = bool(
        exact
        and exact["passes_incremental_gate"]
        and any(row["passes_incremental_gate"] for row in neighbors)
    )
    return {
        "supported": supported,
        "policy": "RVOL 1.1 requires at least one unselectable neighbor to pass the same gates.",
        "rows": rows,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = row["metrics"]
    costs = row["costs"]
    return (
        float(row["validation"].get("validated_score", -99.0)),
        float(costs.get("10.0", {}).get("expected_total_r", -1e9)),
        float(metrics.get("expected_total_r", -1e9)),
        float(metrics.get("profit_factor", 0.0)),
        -len(row.get("patch", {})),
    )


def _gate_snapshot(
    candidate: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    working = deepcopy(candidate)
    passed, reasons = _final_incremental_gate(working, control)
    return {
        "id": candidate["id"],
        "control": control["id"],
        "passes": passed,
        "reasons": reasons,
        "fold_wins_vs_control": working.get("fold_wins_vs_baseline", 0),
        "delta_vs_control": _delta(candidate, control),
    }


def _select(
    finalists: list[dict[str, Any]], baseline: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {row["id"]: row for row in finalists}
    surface = _rvol_surface_support(by_id, baseline)
    passing: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    for row in finalists:
        if row["id"] == baseline["id"]:
            continue
        passed, reasons = _final_incremental_gate(row, baseline)
        if "param_overrides.rvol_threshold" in row.get("patch", {}) and not surface["supported"]:
            passed = False
            reasons = [*reasons, "RVOL surface stability requirement failed"]
        if not row.get("selection_eligible", False):
            passed = False
            reasons = [*reasons, "diagnostic-only candidate"]
        row["passes_final_gate"] = passed
        row["final_gate_reasons"] = reasons
        assessments.append(
            {
                "id": row["id"],
                "passes": passed,
                "reasons": reasons,
                "patch": row.get("patch", {}),
                "fold_wins_vs_baseline": row.get("fold_wins_vs_baseline", 0),
            }
        )
        if passed:
            passing.append(row)

    # Follow the actual lineage rather than globally ranking incomparable
    # mutations.  First admit (or reject) the active Round 3 RVOL mutation.
    # Only then assess archived additions against that stronger control.  This
    # prevents a low-dispersion but economically weaker atomic mutation from
    # outranking the established active lineage through scorer weights alone.
    rvol = by_id["atomic__rvol_1p1"]
    active_step = _gate_snapshot(rvol, baseline)
    if not surface["supported"]:
        active_step["passes"] = False
        active_step["reasons"].append("RVOL surface stability requirement failed")

    selected = rvol if active_step["passes"] else baseline
    overlay_steps: list[dict[str, Any]] = []
    if selected is rvol:
        overlay_ids = (
            "pair__rvol_entry",
            "pair__rvol_failure",
            "bundle__rvol_late_quality",
            "bundle__full_archived_lineage",
        )
        overlay_candidates: list[dict[str, Any]] = []
        for candidate_id in overlay_ids:
            candidate = by_id[candidate_id]
            assessment = _gate_snapshot(candidate, rvol)
            overlay_steps.append(assessment)
            if assessment["passes"] and candidate.get("selection_eligible", False):
                overlay_candidates.append(candidate)
        if overlay_candidates:
            selected = max(overlay_candidates, key=_selection_key)

    # Failure-stop -0.1 is the only remaining addition in the full archived
    # patch once the RVOL + late-quality bundle is selected.  Require it to add
    # value against that exact control rather than merely against the original
    # combined trail.
    final_extension = None
    if selected["id"] == "bundle__rvol_late_quality":
        full = by_id["bundle__full_archived_lineage"]
        final_extension = _gate_snapshot(full, selected)
        if final_extension["passes"]:
            selected = full

    if selected["id"] != baseline["id"]:
        selected["passes_final_gate"] = True
        selected["final_gate_reasons"] = []
    decision = {
        "status": (
            "lineage_mutations_added_to_combined_baseline"
            if selected["id"] != baseline["id"]
            else "combined_trail_only_no_lineage_addition"
        ),
        "selected_candidate": selected["id"],
        "selected_patch": selected.get("patch", {}),
        "selected_changed_keys": selected.get("changed_keys", []),
        "passing_candidate_count": len(passing),
        "rvol_surface": surface,
        "sequential_lineage_assessment": {
            "active_round3_mutation": active_step,
            "archived_overlays_vs_active_rvol": overlay_steps,
            "failure_stop_extension_vs_selected_bundle": final_extension,
            "policy": (
                "Admit active RVOL first; assess archived additions only against "
                "the accepted RVOL control; assess failure-stop last against the "
                "accepted late-quality bundle."
            ),
        },
        "assessments": assessments,
        "production_deployment_approved": False,
        "required_revalidation": [
            "accepted frozen direct-RTH replay",
            "point-in-time or ex-ante frozen universe",
            "genuinely unseen lockbox",
            "intraday mark-to-market drawdown and gap-through-stop stress",
        ],
    }
    return selected, decision


def _render_report(payload: dict[str, Any]) -> str:
    selected = payload["decision"]["selected_candidate"]
    lines = [
        "# ALCB Round 3 lineage rebuild",
        "",
        f"Decision: **{payload['decision']['status']}**",
        "",
        f"Selected configuration: `{selected}`.",
        "",
        "The combined trail is the control. Existing Round 3 lineage mutations are admitted only when they add incremental value after chronology, quality, drawdown, cost, and dependency checks.",
        "",
        "| Candidate | Patch keys | R | AvgR | PF | TPM | DD | Fold wins | 7.5bps R | 10bps R | Final gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    assessment = {row["id"]: row for row in payload["decision"]["assessments"]}
    for row in payload["finalists"]:
        metrics = row["metrics"]
        check = assessment.get(row["id"], {})
        lines.append(
            f"| {row['id']} | {len(row.get('patch', {}))} | "
            f"{metrics['expected_total_r']:+.2f} | {metrics['avg_r']:+.4f} | "
            f"{metrics['profit_factor']:.3f} | {metrics['trades_per_month']:.1f} | "
            f"{metrics['max_drawdown_pct']:.2%} | {row.get('fold_wins_vs_baseline', 0)}/4 | "
            f"{row['costs']['7.5']['expected_total_r']:+.2f} | "
            f"{row['costs']['10.0']['expected_total_r']:+.2f} | "
            f"{'yes' if check.get('passes') else ('control' if row['id'] == 'baseline__combined_trail' else 'no')} |"
        )
    lines.extend(
        [
            "",
            "## RVOL stability",
            "",
            payload["decision"]["rvol_surface"]["policy"],
            "",
        ]
    )
    for row in payload["decision"]["rvol_surface"]["rows"]:
        lines.append(
            f"- RVOL {row['threshold']}: {row['metrics']['expected_total_r']:+.2f}R; "
            f"gate {'pass' if row['passes_incremental_gate'] else 'fail'}."
        )
    lines.extend(
        [
            "",
            "## Promotion boundary",
            "",
            "This package may establish a provisional research Round 3, but it cannot authorize production deployment. The mutations were discovered in a lineage that consumed the former OOS interval, and the replay source remains projected RTH.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if not args.allow_projected_rth_data:
        raise RuntimeError("Pass --allow-projected-rth-data for diagnostic research.")
    if args.max_workers < 1 or args.max_workers > MAX_WORKERS:
        raise ValueError(f"Use between 1 and {MAX_WORKERS} workers.")
    if END_DATE >= CONSUMED_START:
        raise RuntimeError("Development window overlaps consumed OOS.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path.resolve()
    source = _source_fingerprint()
    code = _code_fingerprint()
    catalog = _catalog()
    metadata = _metadata_by_signature(catalog)

    print("=" * 78, flush=True)
    print("ALCB ROUND 3 LINEAGE REBUILD", flush=True)
    print("=" * 78, flush=True)
    print(f"Development only: {START_DATE} -> {END_DATE}", flush=True)
    print(f"Candidates: {len(catalog)}; cache: {cache_path}", flush=True)

    full = _enrich(
        _evaluate_batch(
            catalog,
            start_date=START_DATE,
            end_date=END_DATE,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        ),
        metadata,
    )
    _assert_no_errors("full-period", full)
    by_id = {row["id"]: row for row in full}
    baseline = by_id["baseline__combined_trail"]
    for row in full:
        row["delta_vs_baseline"] = _delta(row, baseline)
        passed, reasons = _full_incremental_gate(row, baseline)
        row["passes_full_incremental_gate"] = passed
        row["full_gate_reasons"] = reasons

    mandatory = {
        "baseline__combined_trail",
        "atomic__rvol_1p1",
        "surface__rvol_1p2",
        "surface__rvol_1p3",
        "bundle__full_archived_lineage",
    }
    finalists = [
        row
        for row in full
        if row["id"] in mandatory or row["passes_full_incremental_gate"]
    ]

    fold_results: dict[str, list[dict[str, Any]]] = {}
    for name, fold_start, fold_end in FOLDS:
        fold_results[name] = _evaluate_batch(
            finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
        _assert_no_errors(name, fold_results[name])
    for row in finalists:
        row["validation"] = _fold_summary(row, fold_results)

    cost_results: dict[float, list[dict[str, Any]]] = {}
    for cost in (7.5, 10.0):
        cost_results[cost] = _evaluate_batch(
            _cost_candidates(finalists, cost),
            start_date=START_DATE,
            end_date=END_DATE,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
        _assert_no_errors(f"cost_{cost}", cost_results[cost])
    for row in finalists:
        row["costs"] = _cost_summary(row, cost_results)

    baseline = next(row for row in finalists if row["id"] == baseline["id"])
    selected, decision = _select(finalists, baseline)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "development_window": {"start": START_DATE, "end": END_DATE},
        "consumed_oos_accessed": False,
        "data_authority": "projected_rth_diagnostic_only",
        "source_fingerprint": source,
        "code_fingerprint": code,
        "candidate_count": len(catalog),
        "finalist_count": len(finalists),
        "baseline": baseline,
        "full_period_results": full,
        "finalists": finalists,
        "decision": decision,
    }
    _write_json(output_dir / "candidate_catalog.json", catalog)
    _write_json(output_dir / "lineage_rebuild_results.json", payload)
    _write_json(output_dir / "recommended_config.json", selected["mutations"])
    _write_json(output_dir / "recommended_patch.json", selected.get("patch", {}))
    (output_dir / "lineage_rebuild_report.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    print(f"Decision: {decision['status']}", flush=True)
    print(f"Selected: {selected['id']}", flush=True)
    print(
        f"Metrics: {selected['metrics']['expected_total_r']:+.2f}R, "
        f"PF {selected['metrics']['profit_factor']:.3f}, "
        f"DD {selected['metrics']['max_drawdown_pct']:.2%}",
        flush=True,
    )
    print(f"Report: {output_dir / 'lineage_rebuild_report.md'}", flush=True)


if __name__ == "__main__":
    main()
