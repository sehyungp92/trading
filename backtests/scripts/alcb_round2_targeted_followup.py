"""Provenance-locked follow-up for the verified ALCB Round-2 robustness audit.

This runner deliberately starts from the verified aggregate baseline and tests
only evidence-driven extensions around the RVOL, entry-time, and early-failure
frontiers.  It evaluates every candidate on aggregate IS and OOS, then runs a
shortlist across three non-overlapping IS folds.  It never promotes a config.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import time as clock_time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.scripts.alcb_round2_oos_robustness import (
    BASE_CONFIG_PATH,
    INITIAL_EQUITY,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    Candidate,
    _attach_is_assessment,
    _candidate_checkpoint_fingerprint,
    _config_fingerprint,
    _data_source_fingerprint,
    _diagnostics_consistent,
    _evaluate_candidates,
    _execution_code_fingerprint,
    _is_guardrail,
    _json_safe,
    _load_json,
    _metric_subset,
    _run_context,
    _strict_uplift,
    _trade_diagnostics,
    _write_json,
    utility,
)


VERIFIED_DIR = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_2"
    / "oos_ablation_perturbation_verified_20260816"
)
DEFAULT_OUTPUT = VERIFIED_DIR / "targeted_followup"

FOLDS: tuple[tuple[str, str, str], ...] = (
    ("is_early", "2024-03-25", "2024-12-31"),
    ("is_middle", "2025-01-01", "2025-08-31"),
    ("is_late", "2025-09-01", "2026-03-01"),
)


def _catalog(base: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(name: str, patch: dict[str, Any], thesis: str, category: str) -> None:
        candidates.append(Candidate(name, "followup", category, patch, thesis, lineage="verified follow-up"))

    # Reproduce the current baseline and the strongest verified incumbents.
    add("control__base", {}, "Exact evaluator parity control.", "control")
    add(
        "control__rvol110",
        {"param_overrides.rvol_threshold": 1.10},
        "Verified aggregate recommendation.",
        "incumbent",
    )
    add(
        "control__rvol110_failure_m010",
        {"param_overrides.rvol_threshold": 1.10, "param_overrides.failure_stop_to_r": -0.10},
        "Verified higher-OOS-quality RVOL pair.",
        "incumbent",
    )
    add(
        "control__entry1330_failure_m010",
        {"param_overrides.entry_window_end": clock_time(13, 30), "param_overrides.failure_stop_to_r": -0.10},
        "Verified OOS-centric frequency/exit pair.",
        "incumbent",
    )
    add(
        "control__rvol110_entry1330",
        {"param_overrides.rvol_threshold": 1.10, "param_overrides.entry_window_end": clock_time(13, 30)},
        "Verified high-frequency pair whose OOS PF needs repair.",
        "incumbent",
    )

    # Resolve the coarse/non-monotone RVOL response around 1.10.
    for value in (0.80, 0.90, 1.00, 1.05, 1.15, 1.25, 1.35):
        add(
            f"rvol_fine__{str(value).replace('.', 'p')}",
            {"param_overrides.rvol_threshold": value},
            f"Fine RVOL response at {value:.2f}.",
            "rvol_fine",
        )

    # Isolate whether the low-RVOL uplift belongs to PDH or OR entries.
    route_specs = {
        "r110__or_only": {"param_overrides.rvol_threshold": 1.10, "param_overrides.pdh_breakout_min_rvol": 1.40},
        "r110__pdh_only": {
            "param_overrides.rvol_threshold": 1.10,
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_min_rvol": 1.40,
        },
        "r110__or_floor120": {
            "param_overrides.rvol_threshold": 1.10,
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_min_rvol": 1.20,
        },
        "r110__or_floor130": {
            "param_overrides.rvol_threshold": 1.10,
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_min_rvol": 1.30,
        },
        "r110__pdh_floor120": {"param_overrides.rvol_threshold": 1.10, "param_overrides.pdh_breakout_min_rvol": 1.20},
        "r110__pdh_floor130": {"param_overrides.rvol_threshold": 1.10, "param_overrides.pdh_breakout_min_rvol": 1.30},
        "r100__or_floor120": {
            "param_overrides.rvol_threshold": 1.00,
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_min_rvol": 1.20,
        },
        "r100__pdh_floor120": {"param_overrides.rvol_threshold": 1.00, "param_overrides.pdh_breakout_min_rvol": 1.20},
    }
    for name, patch in route_specs.items():
        add(name, patch, "Route-specific low-RVOL attribution/containment.", "rvol_route")

    # Find a locally stable early-failure boundary under RVOL 1.10.
    for value in (0.0, -0.05, -0.075, -0.125, -0.15, -0.175, -0.20):
        label = str(value).replace("-", "m").replace(".", "p")
        add(
            f"r110__failure_to_{label}",
            {"param_overrides.rvol_threshold": 1.10, "param_overrides.failure_stop_to_r": value},
            f"RVOL 1.10 with failure-stop target {value:.3f}R.",
            "rvol_exit",
        )
    for bars in (6, 8, 12):
        add(
            f"r110__failure_bars{bars}",
            {"param_overrides.rvol_threshold": 1.10, "param_overrides.failure_stop_bars": bars},
            f"RVOL 1.10 with failure check at bar {bars}.",
            "rvol_exit",
        )
    add(
        "r110__failure_bars6_to_m010",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.failure_stop_bars": 6,
            "param_overrides.failure_stop_to_r": -0.10,
        },
        "Earlier failure recognition plus the verified mild stop target.",
        "rvol_exit",
    )

    # Combine the independent 28-32 bar adaptive-trail neighborhood with RVOL.
    for bars in (27, 28, 30, 32):
        add(
            f"r110__trail_start{bars}",
            {"param_overrides.rvol_threshold": 1.10, "param_overrides.adaptive_trail_start_bars": bars},
            f"RVOL 1.10 plus adaptive-trail start at bar {bars}.",
            "rvol_trail",
        )
    add(
        "r110__trail30_failure_m010",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.adaptive_trail_start_bars": 30,
            "param_overrides.failure_stop_to_r": -0.10,
        },
        "Pair winner extension with mild loser containment.",
        "rvol_trail",
    )

    # Map the missing time boundary and repair the 13:30 high-frequency pair.
    for hour, minute in ((12, 45), (13, 0), (13, 15)):
        add(
            f"r110__entry{hour:02d}{minute:02d}",
            {"param_overrides.rvol_threshold": 1.10, "param_overrides.entry_window_end": clock_time(hour, minute)},
            "RVOL 1.10 entry-window interpolation.",
            "rvol_time",
        )
    for value in (-0.05, -0.10, -0.15):
        label = str(value).replace("-", "m").replace(".", "p")
        add(
            f"r110__entry1330_failure_{label}",
            {
                "param_overrides.rvol_threshold": 1.10,
                "param_overrides.entry_window_end": clock_time(13, 30),
                "param_overrides.failure_stop_to_r": value,
            },
            "Repair high-frequency late entries with a mild early-failure target.",
            "rvol_time_exit",
        )
    add(
        "r110__entry1330_trail30",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_window_end": clock_time(13, 30),
            "param_overrides.adaptive_trail_start_bars": 30,
        },
        "Extend winners in the high-frequency configuration.",
        "rvol_time_exit",
    )
    add(
        "r110__entry1330_trail30_failure_m010",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_window_end": clock_time(13, 30),
            "param_overrides.adaptive_trail_start_bars": 30,
            "param_overrides.failure_stop_to_r": -0.10,
        },
        "Three-way high-frequency winner/loser management.",
        "rvol_time_exit",
    )

    # Late-entry quality escalation: use smooth time/RVOL logic, not dates/symbols.
    for score in (4, 5, 6):
        add(
            f"r110__entry1330_late_score{score}",
            {
                "param_overrides.rvol_threshold": 1.10,
                "param_overrides.entry_window_end": clock_time(13, 30),
                "param_overrides.late_entry_cutoff": clock_time(12, 30),
                "param_overrides.late_entry_score_min": score,
            },
            "Escalate momentum quality only after the incumbent cutoff.",
            "late_quality",
        )
    for add_per_30m in (0.05, 0.10, 0.15):
        label = str(add_per_30m).replace(".", "p")
        add(
            f"r110__entry1330_late_rvol_add{label}",
            {
                "param_overrides.rvol_threshold": 1.10,
                "param_overrides.entry_window_end": clock_time(13, 30),
                "param_overrides.orb_time_decay_start": clock_time(10, 30),
                "param_overrides.orb_late_rvol_add_per_30m": add_per_30m,
            },
            "Require progressively stronger RVOL later in the session.",
            "late_quality",
        )
    add(
        "r110__entry1330_late_rvol_add0p10_failure_m010",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_window_end": clock_time(13, 30),
            "param_overrides.orb_time_decay_start": clock_time(10, 30),
            "param_overrides.orb_late_rvol_add_per_30m": 0.10,
            "param_overrides.failure_stop_to_r": -0.10,
        },
        "Progressive late quality plus mild loser containment.",
        "late_quality",
    )
    add(
        "r110__entry1330_late_score5_failure_m010",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_window_end": clock_time(13, 30),
            "param_overrides.late_entry_cutoff": clock_time(12, 30),
            "param_overrides.late_entry_score_min": 5,
            "param_overrides.failure_stop_to_r": -0.10,
        },
        "Late score escalation plus mild loser containment.",
        "late_quality",
    )

    # Check whether a previously useful simplification composes with the frontier.
    for name, patch in {
        "r110__combined_quality_off": {
            "param_overrides.rvol_threshold": 1.10,
            "ablation.use_combined_quality_gate": False,
        },
        "r110__entry1330_combined_quality_off": {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_window_end": clock_time(13, 30),
            "ablation.use_combined_quality_gate": False,
        },
        "entry1330__failure_m010_combined_quality_off": {
            "param_overrides.entry_window_end": clock_time(13, 30),
            "param_overrides.failure_stop_to_r": -0.10,
            "ablation.use_combined_quality_gate": False,
        },
    }.items():
        add(name, patch, "Composition test for the low-value combined-quality gate.", "simplification")

    # Small-sample falsification only; never prefer this without fold support.
    add(
        "r110__block_score7",
        {
            "param_overrides.rvol_threshold": 1.10,
            "param_overrides.entry_score_blocklist": ("COMBINED_BREAKOUT:5", "*:7"),
        },
        "Falsify whether the negative score-7 holdout cohort generalizes.",
        "overfit_falsification",
    )

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate follow-up candidate name")
    return candidates


def _execution_fingerprint(base: dict[str, Any]) -> tuple[str, str]:
    parent = _config_fingerprint(
        {
            "audit_execution_version": 2,
            "base_config_fingerprint": _config_fingerprint(base),
            "code_fingerprint": _execution_code_fingerprint(),
            "data_fingerprint": _data_source_fingerprint(),
            "windows": {"is": [IS_START, IS_END], "oos": [OOS_START, OOS_END]},
            "initial_equity": INITIAL_EQUITY,
            "execution_mode": "worker_parity_no_shadow",
        }
    )
    return parent, _config_fingerprint({"parent_execution_fingerprint": parent, "followup_version": 1})


def _pareto(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            at_least = all(float(other[field]) >= float(row[field]) for field in fields)
            strictly = any(float(other[field]) > float(row[field]) for field in fields)
            if at_least and strictly:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda item: float(item["robust_score"]), reverse=True)


def _flatten(
    candidates: list[Candidate],
    oos: dict[str, dict[str, Any]],
    is_results: dict[str, dict[str, Any]],
    baseline_oos: dict[str, Any],
    baseline_is: dict[str, Any],
) -> list[dict[str, Any]]:
    catalog = {candidate.name: candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []
    for name, candidate in catalog.items():
        oos_row = oos[name]
        is_row = is_results[name]
        oos_metrics = oos_row.get("metrics", {})
        is_metrics = is_row.get("metrics", {})
        row = {
            "name": name,
            "category": candidate.category,
            "patch": candidate.patch,
            "oos_utility": utility(oos_metrics, baseline_oos),
            "is_utility": utility(is_metrics, baseline_is),
            "strict_oos_uplift": _strict_uplift(oos_metrics, baseline_oos),
            "is_guardrail_pass": _is_guardrail(is_metrics, baseline_is),
        }
        row["robust_score"] = 0.65 * row["oos_utility"] + 0.35 * row["is_utility"]
        for prefix, metrics in (("oos", oos_metrics), ("is", is_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(_json_safe(row))
    rows.sort(key=lambda item: float(item["robust_score"]), reverse=True)
    return rows


def _shortlist(rows: list[dict[str, Any]], limit: int = 12) -> list[str]:
    eligible = [
        row
        for row in rows
        if row["strict_oos_uplift"] and row["is_guardrail_pass"] and row["name"] != "control__base"
    ]
    selected: list[str] = [
        "control__rvol110",
        "control__rvol110_failure_m010",
        "control__entry1330_failure_m010",
        "control__rvol110_entry1330",
    ]
    for field in ("robust_score", "oos_utility", "oos_expected_total_r", "oos_trades_per_month", "is_expected_total_r"):
        for row in sorted(eligible, key=lambda item: float(item.get(field, -999.0)), reverse=True)[:4]:
            if row["name"] not in selected:
                selected.append(row["name"])
            if len(selected) >= limit:
                return selected
    return selected[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows]
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _load_json(BASE_CONFIG_PATH)
    parent_fingerprint, followup_fingerprint = _execution_fingerprint(base)
    verified_spec = _load_json(VERIFIED_DIR / "run_spec.json")
    if verified_spec.get("execution_fingerprint") != parent_fingerprint:
        raise RuntimeError("verified parent artifacts do not match current code/data/base execution fingerprint")

    baseline_payload = _load_json(VERIFIED_DIR / "baseline_diagnostics.json")
    baseline_is = baseline_payload["is"]["metrics"]
    baseline_oos = baseline_payload["oos"]["metrics"]
    candidates = _catalog(base)
    catalog = {candidate.name: candidate for candidate in candidates}
    _write_json(output_dir / "run_spec.json", {
        "parent_verified_dir": str(VERIFIED_DIR),
        "parent_execution_fingerprint": parent_fingerprint,
        "followup_execution_fingerprint": followup_fingerprint,
        "candidate_count": len(candidates),
        "folds": [{"name": name, "start": start, "end": end} for name, start, end in FOLDS],
        "promotion_authorized": False,
    })
    _write_json(output_dir / "candidate_catalog.json", [asdict(candidate) for candidate in candidates])

    print(f"aggregate OOS: {len(candidates)} follow-up candidates", flush=True)
    oos = _evaluate_candidates(
        candidates,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output_dir / "oos_results.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind="process",
        execution_fingerprint=followup_fingerprint,
    )
    if oos["control__base"]["metrics"] != _metric_subset(baseline_oos):
        raise RuntimeError("follow-up OOS parity control failed")

    print(f"aggregate IS: {len(candidates)} follow-up candidates", flush=True)
    is_path = output_dir / "is_results.json"
    is_results = _evaluate_candidates(
        candidates,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=is_path,
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
        execution_fingerprint=followup_fingerprint,
    )
    _attach_is_assessment(is_results, baseline_is, is_path)
    if is_results["control__base"]["metrics"] != _metric_subset(baseline_is):
        raise RuntimeError("follow-up IS parity control failed")

    rows = _flatten(candidates, oos, is_results, baseline_oos, baseline_is)
    eligible = [row for row in rows if row["strict_oos_uplift"] and row["is_guardrail_pass"] and row["name"] != "control__base"]
    for row in eligible:
        row["neg_is_drawdown"] = -float(row["is_max_drawdown_pct"])
        row["neg_oos_drawdown"] = -float(row["oos_max_drawdown_pct"])
    return_frequency_front = _pareto(
        eligible,
        ("oos_expected_total_r", "oos_trades_per_month", "is_expected_total_r", "is_trades_per_month"),
    )
    quality_front = _pareto(
        eligible,
        (
            "oos_expected_total_r",
            "oos_trades_per_month",
            "oos_profit_factor",
            "is_expected_total_r",
            "is_trades_per_month",
            "is_profit_factor",
            "neg_is_drawdown",
        ),
    )
    _write_json(output_dir / "aggregate_results.json", rows)
    _write_csv(output_dir / "aggregate_results.csv", rows)
    _write_json(output_dir / "eligible.json", eligible)
    _write_json(output_dir / "pareto_return_frequency.json", return_frequency_front)
    _write_json(output_dir / "pareto_quality.json", quality_front)

    shortlist_names = _shortlist(rows)
    shortlist_candidates = [catalog[name] for name in shortlist_names]
    print(f"temporal validation: {len(shortlist_candidates)} finalists across {len(FOLDS)} IS folds", flush=True)
    fold_baseline_path = output_dir / "fold_baselines.json"
    fold_baselines_payload = _load_json(fold_baseline_path) if fold_baseline_path.exists() else {}
    if fold_baselines_payload.get("followup_execution_fingerprint") != followup_fingerprint:
        fold_baselines_payload = {"followup_execution_fingerprint": followup_fingerprint, "folds": {}}
    fold_rows: list[dict[str, Any]] = []
    for fold_name, start, end in FOLDS:
        fold_baseline = fold_baselines_payload["folds"].get(fold_name)
        if not fold_baseline:
            context = _run_context(base, start, end)
            fold_baseline = _metric_subset(context["metrics"])
            fold_baselines_payload["folds"][fold_name] = fold_baseline
            _write_json(fold_baseline_path, fold_baselines_payload)
        fold_fingerprint = _config_fingerprint({"followup": followup_fingerprint, "fold": [fold_name, start, end]})
        fold_path = output_dir / f"fold_{fold_name}.json"
        results = _evaluate_candidates(
            shortlist_candidates,
            base,
            start=start,
            end=end,
            max_workers=args.max_workers,
            output_path=fold_path,
            baseline_metrics=fold_baseline,
            batch_size=args.batch_size,
            evaluator_kind="process",
            execution_fingerprint=fold_fingerprint,
        )
        for name in shortlist_names:
            metrics = results[name]["metrics"]
            fold_rows.append({
                "fold": fold_name,
                "name": name,
                "utility": utility(metrics, fold_baseline),
                "delta_expected_total_r": float(metrics["expected_total_r"]) - float(fold_baseline["expected_total_r"]),
                "delta_trades_per_month": float(metrics["trades_per_month"]) - float(fold_baseline["trades_per_month"]),
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "metrics": metrics,
                "baseline": fold_baseline,
            })
    _write_json(output_dir / "fold_results.json", fold_rows)

    fold_by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in shortlist_names}
    for row in fold_rows:
        fold_by_name[row["name"]].append(row)
    aggregate_by_name = {row["name"]: row for row in rows}
    finalists: list[dict[str, Any]] = []
    for name in shortlist_names:
        aggregate = aggregate_by_name[name]
        fold_values = fold_by_name[name]
        mean_fold_utility = sum(float(row["utility"]) for row in fold_values) / len(fold_values)
        min_fold_utility = min(float(row["utility"]) for row in fold_values)
        positive_r_folds = sum(float(row["delta_expected_total_r"]) > 0.0 for row in fold_values)
        final_score = (
            0.45 * float(aggregate["oos_utility"])
            + 0.25 * float(aggregate["is_utility"])
            + 0.30 * mean_fold_utility
            - 0.10 * max(0.0, -min_fold_utility)
        )
        finalists.append({
            "name": name,
            "final_score": final_score,
            "aggregate": aggregate,
            "mean_fold_utility": mean_fold_utility,
            "min_fold_utility": min_fold_utility,
            "positive_expected_r_folds": positive_r_folds,
            "folds": fold_values,
        })
    finalists.sort(key=lambda row: float(row["final_score"]), reverse=True)
    _write_json(output_dir / "finalists.json", finalists)

    diagnostic_names = [row["name"] for row in finalists[:3]]
    diagnostics_path = output_dir / "finalist_oos_diagnostics.json"
    diagnostics_payload = _load_json(diagnostics_path) if diagnostics_path.exists() else {}
    diagnostics_fingerprint = _config_fingerprint({"execution": followup_fingerprint, "names": diagnostic_names})
    if diagnostics_payload.get("fingerprint") != diagnostics_fingerprint:
        diagnostics_payload = {"fingerprint": diagnostics_fingerprint, "results": {}}
        for name in diagnostic_names:
            context = _run_context({**base, **catalog[name].patch}, OOS_START, OOS_END)
            diagnostics_payload["results"][name] = {
                "metrics": _metric_subset(context["metrics"]),
                "diagnostics": _trade_diagnostics(context["trades"]),
            }
            _write_json(diagnostics_path, diagnostics_payload)

    lines = [
        "# ALCB Round 2 targeted follow-up",
        "",
        "This is a diagnostic-only, consumed-OOS study. No configuration is promotion-authorized.",
        "",
        f"Tested {len(candidates)} evidence-driven candidates on aggregate IS/OOS and {len(shortlist_names)} finalists across three IS folds.",
        "",
        "## Leading fold-validated candidates",
        "",
        "| Candidate | Final score | OOS R | OOS TPM | OOS PF | IS R | IS TPM | IS PF | Positive IS folds | Worst fold utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for finalist in finalists:
        agg = finalist["aggregate"]
        lines.append(
            f"| {finalist['name']} | {float(finalist['final_score']):+.4f} | "
            f"{float(agg['oos_expected_total_r']):+.2f} | {float(agg['oos_trades_per_month']):.1f} | "
            f"{float(agg['oos_profit_factor']):.2f} | {float(agg['is_expected_total_r']):+.2f} | "
            f"{float(agg['is_trades_per_month']):.1f} | {float(agg['is_profit_factor']):.2f} | "
            f"{int(finalist['positive_expected_r_folds'])}/3 | {float(finalist['min_fold_utility']):+.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Prefer the Pareto set and fold stability over the scalar rank. A high OOS score is not sufficient when it comes from lower expectancy, PF collapse, or one IS subperiod.",
        "",
        "See `aggregate_results.json`, `pareto_return_frequency.json`, `pareto_quality.json`, `fold_results.json`, `finalists.json`, and `finalist_oos_diagnostics.json`.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "completion.json", {
        "complete": True,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "shortlist_count": len(shortlist_names),
        "leading_name": finalists[0]["name"] if finalists else None,
        "promotion_authorized": False,
    })
    print(f"complete: {output_dir}", flush=True)
    if finalists:
        print(f"leading fold-validated candidate: {finalists[0]['name']}", flush=True)


if __name__ == "__main__":
    main()
