"""Second-stage review of the exploratory ALCB Round-2 recommendation.

This runner deliberately reuses the already-consumed repaired-cache OOS window,
so every result remains research-only.  It tests a dense local neighborhood,
orthogonal combinations of independently robust mutations, targeted early-loss
controls, and early/late in-sample segment stability for the finalists.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from backtests.scripts.alcb_round2_oos_robustness import (
    BASE_CONFIG_PATH,
    Candidate,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    REPO_ROOT,
    _attach_is_assessment,
    _evaluate_candidates,
    _is_guardrail,
    _load_json,
    _strict_uplift,
    _write_json,
    utility,
)


DEFAULT_PRIOR_OUTPUT = (
    REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_2" / "oos_robustness_20260722"
)
DEFAULT_OUTPUT = DEFAULT_PRIOR_OUTPUT / "recommendation_review_20260722"
EARLY_IS = ("2024-03-25", "2025-03-24")
LATE_IS = ("2025-03-25", "2026-03-01")


def _label(value: Any) -> str:
    return str(value).replace("-", "m").replace(":", "").replace(".", "p")


def _candidate_catalog() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(name: str, patch: dict[str, Any], thesis: str, category: str) -> None:
        candidates.append(
            Candidate(
                name=name,
                stage="recommendation_review",
                category=category,
                patch=patch,
                thesis=thesis,
                lineage="post-audit targeted follow-up",
            )
        )

    recommended = {
        "param_overrides.rvol_threshold": 1.90,
        "param_overrides.pdh_size_mult": 0.90,
    }
    add("control__current_round2", {}, "Current Round-2 configuration.", "control")
    add(
        "control__recommended_rvol190_pdh090",
        recommended,
        "Recommendation from the first robustness audit.",
        "control",
    )

    # Dense local surface around the recommended RVOL and PDH sizing values.
    for rvol in (1.85, 1.90, 1.95):
        for pdh in (0.75, 0.80, 0.85, 0.90, 0.95):
            if rvol == 1.90 and pdh == 0.90:
                continue
            add(
                f"local__rvol{_label(rvol)}__pdh{_label(pdh)}",
                {"param_overrides.rvol_threshold": rvol, "param_overrides.pdh_size_mult": pdh},
                "Map the local RVOL x PDH-size response surface around the recommendation.",
                "local_surface",
            )

    # Add each independently robust mechanism to the recommended patch.
    for bars in (7, 8, 9, 10, 11, 12):
        add(
            f"rec__or_bars_{bars}",
            {**recommended, "param_overrides.opening_range_bars": bars},
            "Test opening-range stability on top of the recommendation.",
            "orthogonal_single",
        )
    for distance in (0.06, 0.08, 0.10, 0.14):
        add(
            f"rec__trail_distance_{_label(distance)}",
            {**recommended, "param_overrides.adaptive_trail_late_distance_r": distance},
            "Test the robust late-trail neighborhood on top of the recommendation.",
            "orthogonal_single",
        )
    for cutoff in (clock_time(12, 45), clock_time(13, 0), clock_time(13, 15)):
        add(
            f"rec__entry_{_label(cutoff)}",
            {**recommended, "param_overrides.entry_window_end": cutoff},
            "Test later entry opportunity without changing other entry gates.",
            "orthogonal_single",
        )

    composites = [
        ("rec__or8__trail08", {"param_overrides.opening_range_bars": 8, "param_overrides.adaptive_trail_late_distance_r": 0.08}),
        ("rec__or9__trail08", {"param_overrides.opening_range_bars": 9, "param_overrides.adaptive_trail_late_distance_r": 0.08}),
        ("rec__or10__trail08", {"param_overrides.opening_range_bars": 10, "param_overrides.adaptive_trail_late_distance_r": 0.08}),
        ("rec__or9__entry1300", {"param_overrides.opening_range_bars": 9, "param_overrides.entry_window_end": clock_time(13, 0)}),
        ("rec__trail08__entry1300", {"param_overrides.adaptive_trail_late_distance_r": 0.08, "param_overrides.entry_window_end": clock_time(13, 0)}),
        (
            "rec__or9__trail08__entry1300",
            {
                "param_overrides.opening_range_bars": 9,
                "param_overrides.adaptive_trail_late_distance_r": 0.08,
                "param_overrides.entry_window_end": clock_time(13, 0),
            },
        ),
    ]
    for name, patch in composites:
        add(name, {**recommended, **patch}, "Combine orthogonal robust mechanisms.", "orthogonal_composite")

    # Target the repeated 0-24 bar loss cohort without symbol/date exclusions.
    short_hold = [
        ("rec__failure8", {"param_overrides.failure_stop_bars": 8}),
        (
            "rec__failure8_mfe015_to_m020",
            {
                "param_overrides.failure_stop_bars": 8,
                "param_overrides.failure_stop_mfe_max_r": 0.15,
                "param_overrides.failure_stop_to_r": -0.20,
            },
        ),
        (
            "rec__failure6_mfe010_to_m015",
            {
                "param_overrides.failure_stop_bars": 6,
                "param_overrides.failure_stop_mfe_max_r": 0.10,
                "param_overrides.failure_stop_to_r": -0.15,
            },
        ),
        ("rec__flow_hold8", {"param_overrides.flow_reversal_min_hold_bars": 8}),
        (
            "rec__flow_hold8__trail08",
            {
                "param_overrides.flow_reversal_min_hold_bars": 8,
                "param_overrides.adaptive_trail_late_distance_r": 0.08,
            },
        ),
        (
            "rec__mfe_check12",
            {"param_overrides.mfe_conviction_check_bars": 12},
        ),
        (
            "rec__failure8__mfe_check12",
            {"param_overrides.failure_stop_bars": 8, "param_overrides.mfe_conviction_check_bars": 12},
        ),
        (
            "rec__adaptive_start22",
            {"param_overrides.adaptive_trail_start_bars": 22},
        ),
    ]
    for name, patch in short_hold:
        add(name, {**recommended, **patch}, "Target repeated early trade failures.", "short_hold_target")

    # Risk-adjusted combinations that do not require the recommendation.
    add(
        "risk__or9__trail08",
        {"param_overrides.opening_range_bars": 9, "param_overrides.adaptive_trail_late_distance_r": 0.08},
        "Combine the two strongest drawdown-improving singles.",
        "risk_adjusted",
    )
    add(
        "risk__or9__trail08__entry1300",
        {
            "param_overrides.opening_range_bars": 9,
            "param_overrides.adaptive_trail_late_distance_r": 0.08,
            "param_overrides.entry_window_end": clock_time(13, 0),
        },
        "Test a frequency-forward combination without RVOL/PDH changes.",
        "risk_adjusted",
    )

    score_gradient = {
        "OR_BREAKOUT:4": 0.90,
        "OR_BREAKOUT:5": 0.85,
        "OR_BREAKOUT:6": 1.05,
        "COMBINED_BREAKOUT:7": 1.20,
        "PDH_BREAKOUT:6": 0.65,
    }
    for name, patch in [
        ("score__rvol190__or9", {"param_overrides.opening_range_bars": 9}),
        ("score__rvol190__trail08", {"param_overrides.adaptive_trail_late_distance_r": 0.08}),
        (
            "score__rvol190__or9__trail08",
            {"param_overrides.opening_range_bars": 9, "param_overrides.adaptive_trail_late_distance_r": 0.08},
        ),
    ]:
        add(
            name,
            {
                "param_overrides.rvol_threshold": 1.90,
                "param_overrides.entry_score_size_mults": score_gradient,
                **patch,
            },
            "Use a drawdown-improving mutation to stabilize the score-gradient candidate.",
            "score_gradient_stabilization",
        )

    # Test whether OR/trail improvements can rescue the high-return RVOL 1.8 setting.
    for name, patch in [
        ("rvol180__or9", {"param_overrides.opening_range_bars": 9}),
        (
            "rvol180__or9__trail08",
            {"param_overrides.opening_range_bars": 9, "param_overrides.adaptive_trail_late_distance_r": 0.08},
        ),
        (
            "rvol180__pdh090__or9",
            {"param_overrides.pdh_size_mult": 0.90, "param_overrides.opening_range_bars": 9},
        ),
        (
            "rvol180__pdh090__or9__trail08",
            {
                "param_overrides.pdh_size_mult": 0.90,
                "param_overrides.opening_range_bars": 9,
                "param_overrides.adaptive_trail_late_distance_r": 0.08,
            },
        ),
    ]:
        add(
            name,
            {"param_overrides.rvol_threshold": 1.80, **patch},
            "Test whether robust OR/trail changes offset RVOL 1.8's historical risk expansion.",
            "aggressive_stabilization",
        )

    # The first follow-up identified RVOL 1.8 + OR 9 + a 0.08R late trail as
    # the return/frequency leader.  Map a bounded three-dimensional
    # neighborhood so that an isolated interaction is not mistaken for a
    # stable optimum.  The already-evaluated centre point is reused above.
    for rvol in (1.75, 1.80, 1.85):
        for bars in (8, 9, 10):
            for distance in (0.06, 0.08, 0.10):
                if rvol == 1.80 and bars == 9 and distance == 0.08:
                    continue
                add(
                    f"stability__rvol{_label(rvol)}__or{bars}__trail{_label(distance)}",
                    {
                        "param_overrides.rvol_threshold": rvol,
                        "param_overrides.opening_range_bars": bars,
                        "param_overrides.adaptive_trail_late_distance_r": distance,
                    },
                    "Map the local RVOL x opening-range x trail-distance stability surface.",
                    "aggressive_stability_surface",
                )

    # Isolate PDH sizing around the centre without conflating it with the
    # mechanism grid.  The base 0.75 and tested 0.90 points are reused above.
    for pdh in (0.80, 0.85, 0.95):
        add(
            f"stability__rvol180__or9__trail08__pdh{_label(pdh)}",
            {
                "param_overrides.rvol_threshold": 1.80,
                "param_overrides.opening_range_bars": 9,
                "param_overrides.adaptive_trail_late_distance_r": 0.08,
                "param_overrides.pdh_size_mult": pdh,
            },
            "Test whether the selected PDH sizing value lies on a stable response surface.",
            "pdh_sensitivity",
        )

    # Frequency and early-loss tests on the new mechanism, kept granular so
    # their contribution is identifiable.
    for cutoff in (clock_time(12, 45), clock_time(13, 0)):
        add(
            f"stability__rvol180__or9__trail08__entry{_label(cutoff)}",
            {
                "param_overrides.rvol_threshold": 1.80,
                "param_overrides.opening_range_bars": 9,
                "param_overrides.adaptive_trail_late_distance_r": 0.08,
                "param_overrides.entry_window_end": cutoff,
            },
            "Test a granular frequency extension on the stabilized RVOL candidate.",
            "frequency_extension",
        )
    add(
        "stability__rvol180__or9__trail08__flowhold8",
        {
            "param_overrides.rvol_threshold": 1.80,
            "param_overrides.opening_range_bars": 9,
            "param_overrides.adaptive_trail_late_distance_r": 0.08,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "Test the independently useful early-loss control on the stabilized RVOL candidate.",
        "early_loss_control",
    )

    # One outward ring is necessary because the bounded surface's provisional
    # leader sits at both the low-RVOL and tight-trail edges.  Restrict it to
    # OR 9/10, the adjacent robust OR plateau, and include the existing 0.08R
    # value as a control.  This is deliberately the final search ring.
    for rvol in (1.65, 1.70):
        for bars in (9, 10):
            for distance in (0.04, 0.06, 0.08):
                add(
                    f"boundary__rvol{_label(rvol)}__or{bars}__trail{_label(distance)}",
                    {
                        "param_overrides.rvol_threshold": rvol,
                        "param_overrides.opening_range_bars": bars,
                        "param_overrides.adaptive_trail_late_distance_r": distance,
                    },
                    "Resolve the low-RVOL/tight-trail boundary without broadening the search indefinitely.",
                    "boundary_resolution",
                )
    for rvol in (1.75, 1.80):
        for bars in (9, 10):
            add(
                f"boundary__rvol{_label(rvol)}__or{bars}__trail0p04",
                {
                    "param_overrides.rvol_threshold": rvol,
                    "param_overrides.opening_range_bars": bars,
                    "param_overrides.adaptive_trail_late_distance_r": 0.04,
                },
                "Test the adjacent tighter-trail boundary at the two strongest RVOL levels.",
                "boundary_resolution",
            )
    for bars in (9, 10):
        add(
            f"boundary__rvol175__or{bars}__trail06__flowhold8",
            {
                "param_overrides.rvol_threshold": 1.75,
                "param_overrides.opening_range_bars": bars,
                "param_overrides.adaptive_trail_late_distance_r": 0.06,
                "param_overrides.flow_reversal_min_hold_bars": 8,
            },
            "Test the early-loss control on the strongest stable RVOL/trail setting.",
            "early_loss_control",
        )

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate follow-up candidate name")
    return candidates


def _complexity(candidate: Candidate) -> int:
    return sum(len(value) if isinstance(value, dict) else 1 for value in candidate.patch.values())


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    objectives = (
        "is_expected_total_r",
        "is_net_profit",
        "is_trades_per_month",
        "oos_expected_total_r",
        "oos_net_profit",
        "oos_trades_per_month",
    )
    return all(float(left[key]) >= float(right[key]) for key in objectives) and any(
        float(left[key]) > float(right[key]) for key in objectives
    )


def _flatten(
    candidates: list[Candidate],
    oos: dict[str, dict[str, Any]],
    insample: dict[str, dict[str, Any]],
    baseline_oos: dict[str, Any],
    baseline_is: dict[str, Any],
) -> list[dict[str, Any]]:
    catalog = {candidate.name: candidate for candidate in candidates}
    rows = []
    for name, oos_row in oos.items():
        if name not in catalog or name not in insample:
            continue
        is_row = insample[name]
        oos_metrics, is_metrics = oos_row["metrics"], is_row["metrics"]
        oos_utility = utility(oos_metrics, baseline_oos)
        is_utility = utility(is_metrics, baseline_is)
        complexity = _complexity(catalog[name])
        row = {
            "name": name,
            "category": catalog[name].category,
            "patch": catalog[name].patch,
            "complexity": complexity,
            "oos_strict_uplift": _strict_uplift(oos_metrics, baseline_oos),
            "is_guardrail_pass": _is_guardrail(is_metrics, baseline_is),
            "oos_utility": oos_utility,
            "is_utility": is_utility,
            "weighted_65_35": 0.65 * oos_utility + 0.35 * is_utility,
            "weighted_equal": 0.50 * oos_utility + 0.50 * is_utility,
            "maximin_utility": min(oos_utility, is_utility),
            "complexity_adjusted_equal": 0.50 * oos_utility + 0.50 * is_utility - 0.004 * max(0, complexity - 2),
        }
        for prefix, metrics in (("oos", oos_metrics), ("is", is_metrics)):
            for key in (
                "total_trades",
                "win_rate",
                "expectancy",
                "expected_total_r",
                "net_profit",
                "profit_factor",
                "trades_per_month",
                "max_drawdown_pct",
            ):
                row[f"{prefix}_{key}"] = metrics[key]
        rows.append(row)
    rows.sort(key=lambda row: row["complexity_adjusted_equal"], reverse=True)
    return rows


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["oos_strict_uplift"] and row["is_guardrail_pass"]]
    return [row for row in eligible if not any(_dominates(other, row) for other in eligible if other is not row)]


def _segment_candidates(rows: list[dict[str, Any]], catalog: dict[str, Candidate], limit: int = 11) -> list[Candidate]:
    eligible = [row for row in rows if row["oos_strict_uplift"] and row["is_guardrail_pass"]]
    chosen: list[str] = ["control__recommended_rvol190_pdh090"]
    current = next(row for row in rows if row["name"] == chosen[0])
    # Never prune a strict dominator of the recommendation merely because a
    # composite-score diversity cap selected another representative first.
    for row in sorted(eligible, key=lambda item: item["complexity_adjusted_equal"], reverse=True):
        if _dominates(row, current) and row["name"] not in chosen:
            chosen.append(row["name"])
    rankers = (
        lambda row: row["weighted_65_35"],
        lambda row: row["weighted_equal"],
        lambda row: row["maximin_utility"],
        lambda row: row["is_trades_per_month"] + row["oos_trades_per_month"],
        lambda row: -(row["is_max_drawdown_pct"] + row["oos_max_drawdown_pct"]),
    )
    for ranker in rankers:
        for row in sorted(eligible, key=ranker, reverse=True):
            if row["name"] not in chosen:
                chosen.append(row["name"])
                break
    for row in eligible:
        if len(chosen) >= limit:
            break
        if row["name"] not in chosen:
            chosen.append(row["name"])
    return [Candidate("segment__baseline", "segment", "control", {}, "Segment baseline")] + [catalog[name] for name in chosen]


def _segment_assessment(
    candidates: list[Candidate], early: dict[str, dict[str, Any]], late: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for candidate in candidates:
        if candidate.name == "segment__baseline":
            continue
        row: dict[str, Any] = {"name": candidate.name, "patch": candidate.patch}
        passes = []
        for label, results in (("early", early), ("late", late)):
            baseline = results["segment__baseline"]["metrics"]
            metrics = results[candidate.name]["metrics"]
            passes.append(
                metrics["expected_total_r"] >= baseline["expected_total_r"] * 0.95
                and metrics["net_profit"] >= baseline["net_profit"] * 0.90
                and metrics["profit_factor"] >= baseline["profit_factor"] * 0.85
                and metrics["max_drawdown_pct"] <= max(
                    baseline["max_drawdown_pct"] * 1.25, baseline["max_drawdown_pct"] + 0.015
                )
            )
            for key in ("expected_total_r", "net_profit", "profit_factor", "trades_per_month", "max_drawdown_pct"):
                row[f"{label}_{key}"] = metrics[key]
                row[f"{label}_{key}_delta"] = metrics[key] - baseline[key]
        row["both_segments_pass"] = all(passes)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        value = dict(row)
        value["patch"] = json.dumps(value["patch"], sort_keys=True, default=str)
        flat.append(value)
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _render_report(
    rows: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    current_name: str,
    selected_name: str | None,
) -> str:
    by_name = {row["name"]: row for row in rows}
    current = by_name[current_name]
    dominant = [row for row in rows if row["oos_strict_uplift"] and row["is_guardrail_pass"] and _dominates(row, current)]
    dominant.sort(key=lambda row: row["complexity_adjusted_equal"], reverse=True)
    lines = [
        "# ALCB Round 2 recommendation review",
        "",
        "## Decision",
        "",
        (
            f"The original recommendation is {'dominated by at least one tested candidate' if dominant else 'not strictly dominated'} "
            "on IS/OOS total R, net PnL, and trade frequency under the predefined risk guardrails."
        ),
        "This remains diagnostic-only: the repaired legacy OOS window has been reused and no frozen direct-RTH bundle exists.",
        "",
        f"Segment-qualified selection: `{selected_name}`." if selected_name else "No follow-up candidate passed every selection layer.",
        "",
        "## Candidates that dominate the original recommendation",
        "",
        "| Candidate | IS R | IS trades/mo | IS PF | IS DD | OOS R | OOS trades/mo | OOS PF | OOS DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dominant[:15]:
        lines.append(
            f"| {row['name']} | {row['is_expected_total_r']:.2f} | {row['is_trades_per_month']:.1f} | "
            f"{row['is_profit_factor']:.2f} | {100*row['is_max_drawdown_pct']:.1f}% | {row['oos_expected_total_r']:.2f} | "
            f"{row['oos_trades_per_month']:.1f} | {row['oos_profit_factor']:.2f} | {100*row['oos_max_drawdown_pct']:.1f}% |"
        )
    if not dominant:
        lines.append("| none | | | | | | | | |")
    lines.extend(
        [
            "",
            "## Top robust candidates by equal-window, complexity-adjusted utility",
            "",
            "| Candidate | Score | IS R | IS trades/mo | IS PF | IS DD | OOS R | OOS trades/mo | OOS PF | OOS DD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    eligible = [row for row in rows if row["oos_strict_uplift"] and row["is_guardrail_pass"]]
    for row in eligible[:20]:
        lines.append(
            f"| {row['name']} | {row['complexity_adjusted_equal']:+.3f} | {row['is_expected_total_r']:.2f} | "
            f"{row['is_trades_per_month']:.1f} | {row['is_profit_factor']:.2f} | {100*row['is_max_drawdown_pct']:.1f}% | "
            f"{row['oos_expected_total_r']:.2f} | {row['oos_trades_per_month']:.1f} | {row['oos_profit_factor']:.2f} | "
            f"{100*row['oos_max_drawdown_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"The four-objective Pareto frontier contains {len(pareto)} candidate(s); see `pareto_frontier.json`.",
            "",
            "## Early/late IS stability",
            "",
            "| Finalist | Both pass | Early delta R | Early delta trades/mo | Late delta R | Late delta trades/mo |",
            "|---|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in segments:
        lines.append(
            f"| {row['name']} | {'yes' if row['both_segments_pass'] else 'no'} | "
            f"{row['early_expected_total_r_delta']:+.2f} | {row['early_trades_per_month_delta']:+.2f} | "
            f"{row['late_expected_total_r_delta']:+.2f} | {row['late_trades_per_month_delta']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A strict dominator is stronger evidence than winning one arbitrary composite score.",
            "- Smooth local RVOL/PDH neighborhoods are preferred to isolated optima.",
            "- Segment failure vetoes a candidate even when aggregate IS and OOS pass.",
            "- No result is eligible for production promotion until a fresh authoritative lockbox is available.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-output", type=Path, default=DEFAULT_PRIOR_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.allow_legacy_data:
        raise SystemExit("Pass --allow-legacy-data; no authoritative frozen bundle is available.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = args.prior_output.resolve()
    baseline_payload = _load_json(prior / "baseline_diagnostics.json")
    baseline_is = baseline_payload["is"]["metrics"]
    baseline_oos = baseline_payload["oos"]["metrics"]
    base = _load_json(BASE_CONFIG_PATH)
    candidates = _candidate_catalog()
    catalog = {candidate.name: candidate for candidate in candidates}
    _write_json(output / "candidate_catalog.json", [asdict(candidate) for candidate in candidates])
    _write_json(
        output / "run_spec.json",
        {
            "generated_at_utc": datetime.now(timezone.utc),
            "candidate_count": len(candidates),
            "windows": {"is": [IS_START, IS_END], "oos": [OOS_START, OOS_END], "early_is": EARLY_IS, "late_is": LATE_IS},
            "data_authority": "diagnostic-only repaired legacy filename cache",
            "promotion_authorized": False,
        },
    )

    oos = _evaluate_candidates(
        candidates,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output / "oos_results.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    insample_path = output / "is_results.json"
    insample = _evaluate_candidates(
        candidates,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=insample_path,
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    _attach_is_assessment(insample, baseline_is, insample_path)
    rows = _flatten(candidates, oos, insample, baseline_oos, baseline_is)
    pareto = _pareto(rows)
    _write_json(output / "all_results.json", rows)
    _write_csv(output / "all_results.csv", rows)
    _write_json(output / "pareto_frontier.json", pareto)

    segment_candidates = _segment_candidates(rows, catalog)
    _write_json(output / "segment_candidate_catalog.json", [asdict(candidate) for candidate in segment_candidates])
    early = _evaluate_candidates(
        segment_candidates,
        base,
        start=EARLY_IS[0],
        end=EARLY_IS[1],
        max_workers=args.max_workers,
        output_path=output / "early_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    late = _evaluate_candidates(
        segment_candidates,
        base,
        start=LATE_IS[0],
        end=LATE_IS[1],
        max_workers=args.max_workers,
        output_path=output / "late_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    segments = _segment_assessment(segment_candidates, early, late)
    _write_json(output / "segment_assessment.json", segments)
    segment_by_name = {row["name"]: row for row in segments}
    qualified = [
        row
        for row in rows
        if row["oos_strict_uplift"]
        and row["is_guardrail_pass"]
        and segment_by_name.get(row["name"], {}).get("both_segments_pass")
    ]
    qualified.sort(key=lambda row: row["complexity_adjusted_equal"], reverse=True)
    selected = qualified[0] if qualified else None
    current_name = "control__recommended_rvol190_pdh090"
    current = next(row for row in rows if row["name"] == current_name)
    dominant = [row for row in rows if row["oos_strict_uplift"] and row["is_guardrail_pass"] and _dominates(row, current)]
    review = {
        "original_recommendation": current,
        "dominant_candidates": dominant,
        "selected_after_segment_validation": selected,
        "segment_assessment": segments,
        "promotion_authorized": False,
        "promotion_blocker": "OOS reused for targeted search and authoritative frozen data bundle is unavailable.",
    }
    _write_json(output / "recommendation_review.json", review)
    (output / "report.md").write_text(
        _render_report(rows, pareto, segments, current_name, selected["name"] if selected else None),
        encoding="utf-8",
    )
    _write_json(
        output / "completion.json",
        {
            "completed_at_utc": datetime.now(timezone.utc),
            "candidate_count": len(candidates),
            "segment_candidate_count": len(segment_candidates),
            "selected_name": selected["name"] if selected else None,
            "promotion_authorized": False,
        },
    )
    print(f"complete: {output}", flush=True)
    print(f"selected: {selected['name'] if selected else 'none'}", flush=True)


if __name__ == "__main__":
    main()
