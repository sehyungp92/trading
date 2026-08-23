"""Validate the bounded ALCB trail interaction requested after baseline recovery.

The experiment compares the immutable Round 2 control, the two validated
single-setting challengers, and their combination.  It never accesses the
consumed post-2026-03-01 period and never mutates historical round artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    _safety_gate,
    _signature,
    _source_fingerprint,
    _write_json,
)
from backtests.stock.auto.alcb.run_representative_baseline_sequence import (
    DEFAULT_OUTPUT as BASELINE_OUTPUT,
    ROUND3_CONFIG,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT / "backtests/output/stock/alcb/trail_combination_20260822"
)
SHARED_CACHE = BASELINE_OUTPUT / "evaluation_cache.json"
BASELINE_CONFIG = BASELINE_OUTPUT / "representative_baseline_config.json"
ACTIVATION_CONFIG = BASELINE_OUTPUT / "development_challenger_config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-path", type=Path, default=SHARED_CACHE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-projected-rth-data", action="store_true")
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _candidate(candidate_id: str, family: str, mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "family": family,
        "era": "bounded_trail_interaction",
        "sources": [str(BASELINE_CONFIG), str(ACTIVATION_CONFIG)],
        "mutations": dict(sorted(mutations.items())),
    }


def _catalog() -> list[dict[str, Any]]:
    baseline = _read_config(BASELINE_CONFIG)
    activation = _read_config(ACTIVATION_CONFIG)
    existing_round3 = _read_config(ROUND3_CONFIG)
    timing = deepcopy(baseline)
    timing.update(
        {
            "param_overrides.adaptive_trail_start_bars": 30,
            "param_overrides.adaptive_trail_tighten_bars": 30,
        }
    )
    combined = deepcopy(activation)
    combined.update(
        {
            "param_overrides.adaptive_trail_start_bars": 30,
            "param_overrides.adaptive_trail_tighten_bars": 30,
        }
    )
    return [
        _candidate("control__round2_exact", "control", baseline),
        _candidate(
            "diagnostic__existing_round3_rvol_1p1",
            "oos_contaminated_round3",
            existing_round3,
        ),
        _candidate("single__trail_activate_0p18", "single_activation", activation),
        _candidate("single__trail_timing_30", "single_timing", timing),
        _candidate("combined__activation_0p18_timing_30", "combined_trail", combined),
    ]


def _by_base_signature(cost_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["base_signature"]): row
        for row in cost_rows
        if not row.get("error") and row.get("base_signature")
    }


def _assert_no_errors(stage: str, rows: list[dict[str, Any]]) -> None:
    errors = [row for row in rows if row.get("error")]
    if errors:
        detail = "\n".join(
            f"{row.get('id', 'unknown')}: {str(row.get('error', '')).splitlines()[-1]}"
            for row in errors
        )
        raise RuntimeError(f"{stage} evaluation failed:\n{detail}")


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(left.get(key, 0.0)) - float(right.get(key, 0.0))
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


def _interaction_delta(
    combined: dict[str, Any],
    activation: dict[str, Any],
    timing: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, float]:
    """Actual combination minus the additive expectation of both singles."""

    return {
        key: (
            float(combined.get(key, 0.0))
            - float(activation.get(key, 0.0))
            - float(timing.get(key, 0.0))
            + float(control.get(key, 0.0))
        )
        for key in (
            "expected_total_r",
            "net_profit",
            "avg_r",
            "profit_factor",
            "max_drawdown_pct",
        )
    }


def _promotion_assessment(
    combined: dict[str, Any],
    activation: dict[str, Any],
    timing: dict[str, Any],
    control: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Require incremental evidence over both singles, not merely the control."""

    reasons: list[str] = []
    cm = combined["metrics"]
    singles = (activation, timing)
    best_r = max(float(row["metrics"]["expected_total_r"]) for row in singles)
    best_avg_r = max(float(row["metrics"]["avg_r"]) for row in singles)
    best_pf = max(float(row["metrics"]["profit_factor"]) for row in singles)
    best_7p5 = max(
        float(row["costs"]["7.5"]["expected_total_r"]) for row in singles
    )
    best_10 = max(float(row["costs"]["10.0"]["expected_total_r"]) for row in singles)

    if not _safety_gate(cm):
        reasons.append("full-period safety gate failed")
    if not combined["validation"].get("robust_eligible"):
        reasons.append("chronological robustness gate failed")
    if not combined["costs"].get("seven_five_gate") or not combined["costs"].get("ten_gate"):
        reasons.append("7.5/10 bps absolute cost gate failed")
    if float(cm["expected_total_r"]) < best_r + 2.0:
        reasons.append("combination did not beat the best single by at least 2R")
    if float(cm["avg_r"]) < best_avg_r * 0.98:
        reasons.append("combination retained less than 98% of best-single AvgR")
    if float(cm["profit_factor"]) < best_pf * 0.98:
        reasons.append("combination retained less than 98% of best-single PF")
    if float(combined["costs"]["7.5"]["expected_total_r"]) < best_7p5:
        reasons.append("combination did not beat the best single at 7.5 bps")
    if float(combined["costs"]["10.0"]["expected_total_r"]) < best_10:
        reasons.append("combination did not beat the best single at 10 bps")

    control_folds = {
        row["fold"]: row for row in control["validation"].get("folds", [])
    }
    activation_folds = {
        row["fold"]: row for row in activation["validation"].get("folds", [])
    }
    timing_folds = {
        row["fold"]: row for row in timing["validation"].get("folds", [])
    }
    combined_folds = {
        row["fold"]: row for row in combined["validation"].get("folds", [])
    }
    best_single_wins = 0
    for fold in control_folds:
        combined_r = float(combined_folds[fold]["expected_total_r"])
        control_r = float(control_folds[fold]["expected_total_r"])
        best_single_r = max(
            float(activation_folds[fold]["expected_total_r"]),
            float(timing_folds[fold]["expected_total_r"]),
        )
        if combined_r < control_r:
            reasons.append(f"combination fell below control in {fold}")
        if combined_r >= best_single_r:
            best_single_wins += 1
    if best_single_wins < 3:
        reasons.append("combination failed to match the best single in at least 3/4 folds")
    return not reasons, reasons


def _render_report(payload: dict[str, Any]) -> str:
    rows = payload["results"]
    decision = payload["decision"]
    lines = [
        "# ALCB trail-combination validation",
        "",
        f"Decision: **{decision['status']}**",
        "",
        decision["interpretation"],
        "",
        "| Candidate | R | AvgR | PF | Win rate | Trades/mo | DD | 7.5bps R | 10bps R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['id']} | {metrics['expected_total_r']:+.2f} | "
            f"{metrics['avg_r']:+.4f} | {metrics['profit_factor']:.3f} | "
            f"{metrics['win_rate']:.1%} | {metrics['trades_per_month']:.1f} | "
            f"{metrics['max_drawdown_pct']:.2%} | "
            f"{row['costs']['7.5']['expected_total_r']:+.2f} | "
            f"{row['costs']['10.0']['expected_total_r']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interaction",
            "",
            (
                "Interaction delta is the combined result minus the additive expectation "
                "of the two single-setting improvements. Positive R indicates synergy; "
                "negative R indicates overlap or interference."
            ),
            "",
            f"- Total-R interaction: {decision['interaction_delta']['expected_total_r']:+.2f}R",
            f"- Avg-R interaction: {decision['interaction_delta']['avg_r']:+.4f}R",
            f"- PF interaction: {decision['interaction_delta']['profit_factor']:+.3f}",
            "",
            "## Policy",
            "",
            "This is development evidence only. The consumed post-2026-03-01 interval was not accessed. Even a passing combination is only a candidate for a fresh lockbox; this runner does not overwrite Round 3 or the rounds manifest.",
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
    catalog = _catalog()
    source = _source_fingerprint()
    code = _code_fingerprint()

    print("=" * 78, flush=True)
    print("ALCB BOUNDED TRAIL-COMBINATION VALIDATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Cache: {cache_path}", flush=True)
    print(f"Development only: {START_DATE} -> {END_DATE}", flush=True)

    full = _evaluate_batch(
        catalog,
        start_date=START_DATE,
        end_date=END_DATE,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source=source,
        code=code,
    )
    _assert_no_errors("full-period", full)

    fold_results: dict[str, list[dict[str, Any]]] = {}
    for name, fold_start, fold_end in FOLDS:
        fold_results[name] = _evaluate_batch(
            catalog,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
        _assert_no_errors(name, fold_results[name])
    for row in full:
        row["validation"] = _fold_summary(row, fold_results)

    cost_results: dict[float, list[dict[str, Any]]] = {}
    for cost in (7.5, 10.0):
        cost_results[cost] = _evaluate_batch(
            _cost_candidates(catalog, cost),
            start_date=START_DATE,
            end_date=END_DATE,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
        _assert_no_errors(f"cost_{cost}", cost_results[cost])
    for row in full:
        row["costs"] = _cost_summary(row, cost_results)

    by_id = {row["id"]: row for row in full}
    control = by_id["control__round2_exact"]
    activation = by_id["single__trail_activate_0p18"]
    timing = by_id["single__trail_timing_30"]
    combined = by_id["combined__activation_0p18_timing_30"]
    existing_round3 = by_id["diagnostic__existing_round3_rvol_1p1"]
    for row in full:
        row["delta_vs_control"] = _metric_delta(row["metrics"], control["metrics"])
    interaction = _interaction_delta(
        combined["metrics"], activation["metrics"], timing["metrics"], control["metrics"]
    )
    passes, reasons = _promotion_assessment(combined, activation, timing, control)
    status = (
        "fresh_lockbox_candidate_not_promoted"
        if passes
        else "do_not_adopt_combination"
    )
    interpretation = (
        "The combination adds incremental value over both single changes and may be "
        "carried as a pre-registered fresh-lockbox candidate. It is not a new Round 3 yet."
        if passes
        else "The combination does not add sufficiently robust incremental value over the "
        "single-setting challengers and should not replace the representative baseline."
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_window": {"start": START_DATE, "end": END_DATE},
        "consumed_oos_accessed": False,
        "source_fingerprint": source,
        "code_fingerprint": code,
        "results": full,
        "decision": {
            "status": status,
            "passes_incremental_gate": passes,
            "reasons": reasons,
            "interpretation": interpretation,
            "interaction_delta": interaction,
            "combined_delta_vs_control": combined["delta_vs_control"],
            "combined_delta_vs_existing_round3": _metric_delta(
                combined["metrics"], existing_round3["metrics"]
            ),
            "existing_round3_is_oos_contaminated": True,
        },
    }
    _write_json(output_dir / "candidate_catalog.json", catalog)
    _write_json(output_dir / "trail_combination_results.json", payload)
    _write_json(output_dir / "combined_config.json", combined["mutations"])
    (output_dir / "trail_combination_report.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    print(f"Decision: {status}", flush=True)
    print(
        f"Combined: {combined['metrics']['expected_total_r']:+.2f}R, "
        f"PF {combined['metrics']['profit_factor']:.3f}, "
        f"DD {combined['metrics']['max_drawdown_pct']:.2%}",
        flush=True,
    )
    print(f"Interaction delta: {interaction['expected_total_r']:+.2f}R", flush=True)
    print(f"Report: {output_dir / 'trail_combination_report.md'}", flush=True)


if __name__ == "__main__":
    main()
