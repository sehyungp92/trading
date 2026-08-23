"""Synthesize the two IARIC research streams into a finalizable Round 3.

The objective is absolute IARIC alpha capture: maximize robust total R and
frequency subject to quality and drawdown constraints.  ALCB reference families
are ignored for selection. Every evaluated executable branch is rescored with
the fixed unified score, management is re-expanded around a diverse Pareto
beam, and fresh validation replaces the earlier top-four prune. Walk-forward
reversion survivors must add positive alpha on symbol-days not already traded
before they are admitted for exact route replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.runners.analyze_stock_opportunity_atlas import (
    APERTURES,
    _bootstrap_probability_positive,
    _metrics as opportunity_metrics,
    _records_for_entry_variant,
)
from backtests.stock.auto.runners.run_iaric_branched_aperture import (
    END_DATE,
    START_DATE,
    _attribution_stats,
    _audit_candidates,
    _candidate,
    _evaluate,
    _paired_block_bootstrap,
    _signature,
    _validation,
)
from backtests.stock.auto.runners.run_iaric_unified_round3 import (
    ROUTE_READINESS,
    UNIFIED_SCORE_SPEC,
    _score,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
BRANCHED_DIR = REPO_ROOT / "backtests/output/stock/iaric/round_3"
ATLAS_DIR = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
OUTPUT_DIR = BRANCHED_DIR / "alpha_synthesis"

FINAL_VALUE_VERIFICATION_SPEC: dict[str, float] = {
    "return_led_min_total_r_uplift": 0.03,
    "return_led_min_trade_retention": 0.95,
    "frequency_led_min_trade_uplift": 0.15,
    "frequency_led_min_total_r_retention": 1.00,
    "min_unified_score_uplift": 0.005,
    "min_paired_bootstrap_probability": 0.75,
    "min_avg_r": 0.12,
    "min_profit_factor": 1.45,
    "max_drawdown_pct": 0.07,
    "max_drawdown_increase": 0.015,
    "max_single_symbol_net_share": 0.35,
    "max_changed_mutations": 15.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branched-dir", default=str(BRANCHED_DIR))
    parser.add_argument("--atlas-dir", default=str(ATLAS_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--bootstrap-simulations", type=int, default=5000)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _wait_for_pid(pid: int) -> None:
    if pid <= 0:
        return
    print(f"queued behind PID {pid}", flush=True)
    if os.name == "nt":
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue"],
            check=False,
        )
        return
    import time
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(10.0)


SOURCE_STAGE_FILES = (
    "generator_results.json",
    "filter_results.json",
    "route_isolation_results.json",
    "route_additive_results.json",
    "management_results.json",
    "validated_finalists.json",
)


def _synthesis_code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "backtests/stock/auto/runners/run_iaric_branched_aperture.py",
        REPO_ROOT / "backtests/stock/auto/runners/run_iaric_unified_round3.py",
        REPO_ROOT / "backtests/stock/auto/runners/analyze_stock_opportunity_atlas.py",
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_candidate_pool(branched_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recover every evaluated branch instead of inheriting the old top-four prune."""

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for name in SOURCE_STAGE_FILES:
        payload = _load_json(branched_dir / name)
        if not isinstance(payload, list):
            raise ValueError(f"expected candidate list: {branched_dir / name}")
        counts[name] = len(payload)
        rows.extend(payload)
    # Prefer the canonical control's identity when duplicate signatures occur.
    rows.sort(key=lambda row: row.get("id") != "control_oversold")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row.get("mutations"), dict):
            continue
        unique.setdefault(_signature(row["mutations"]), row)
    return list(unique.values()), counts


def _preliminary_viable(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if row.get("id") == "control_oversold":
        return True
    metrics = row.get("metrics", {})
    base = baseline.get("metrics", {})
    trades = float(metrics.get("total_trades", 0.0))
    total_r = float(metrics.get("expected_total_r", 0.0))
    breadth = (
        trades >= 0.75 * float(base.get("total_trades", 0.0))
        or total_r >= 0.75 * float(base.get("expected_total_r", 0.0))
    )
    return bool(
        breadth
        and float(metrics.get("avg_r", -99.0)) >= 0.05
        and float(metrics.get("profit_factor", 0.0)) >= 1.10
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.10
        and float(metrics.get("robust_avg_r", -99.0)) >= -0.10
    )


def _preliminary_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm, rm = left["metrics"], right["metrics"]
    comparisons = (
        float(left["preliminary_unified_score"]) >= float(right["preliminary_unified_score"]),
        float(lm.get("expected_total_r", 0.0)) >= float(rm.get("expected_total_r", 0.0)),
        float(lm.get("total_trades", 0.0)) >= float(rm.get("total_trades", 0.0)),
        float(lm.get("max_drawdown_pct", 1.0)) <= float(rm.get("max_drawdown_pct", 1.0)),
    )
    strict = (
        float(left["preliminary_unified_score"]) > float(right["preliminary_unified_score"]) + 1e-12
        or float(lm.get("expected_total_r", 0.0)) > float(rm.get("expected_total_r", 0.0)) + 1e-9
        or float(lm.get("total_trades", 0.0)) > float(rm.get("total_trades", 0.0)) + 1e-9
        or float(lm.get("max_drawdown_pct", 1.0)) < float(rm.get("max_drawdown_pct", 1.0)) - 1e-12
    )
    return all(comparisons) and strict


def _broad_unified_shortlist(
    rows: list[dict[str, Any]], *, cap: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = next(row for row in rows if row.get("id") == "control_oversold")
    viable: list[dict[str, Any]] = []
    for source in rows:
        if not _preliminary_viable(source, baseline):
            continue
        row = dict(source)
        score, components, raw = _score(row)
        row.update({
            "preliminary_unified_score": score,
            "preliminary_unified_components": components,
            "preliminary_unified_raw": raw,
        })
        viable.append(row)
    if not viable:
        raise RuntimeError("no candidates survived broad unified economic safety gates")

    reasons: dict[str, set[str]] = defaultdict(set)

    def preserve(samples: list[dict[str, Any]], reason: str) -> None:
        for sample in samples:
            reasons[_signature(sample["mutations"])].add(reason)

    by_score = sorted(viable, key=lambda row: float(row["preliminary_unified_score"]), reverse=True)
    preserve(by_score[:12], "top_unified_score")
    preserve(sorted(viable, key=lambda row: float(row["metrics"].get("expected_total_r", 0.0)), reverse=True)[:4], "top_total_r")
    preserve(sorted(viable, key=lambda row: float(row["metrics"].get("total_trades", 0.0)), reverse=True)[:4], "top_frequency")
    preserve(sorted(viable, key=lambda row: float(row["metrics"].get("avg_r", 0.0)), reverse=True)[:2], "top_avg_r")
    for family in sorted({str(row.get("root_family", row.get("family", "unknown"))) for row in viable}):
        family_rows = [
            row for row in viable
            if str(row.get("root_family", row.get("family", "unknown"))) == family
        ]
        preserve([max(family_rows, key=lambda row: float(row["preliminary_unified_score"]))], "best_root_family")
    frontier = [
        row for row in viable
        if not any(_preliminary_dominates(other, row) for other in viable if other is not row)
    ]
    frontier.sort(key=lambda row: float(row["preliminary_unified_score"]), reverse=True)
    preserve(frontier[:8], "pareto_frontier")
    preserve([baseline], "mandatory_control")

    selected = [row for row in by_score if _signature(row["mutations"]) in reasons]
    if len(selected) > cap:
        priority_reasons = {"mandatory_control", "best_root_family", "pareto_frontier"}
        mandatory = [
            row for row in selected
            if reasons[_signature(row["mutations"])] & priority_reasons
        ]
        # The cap is soft: never discard a root-family representative or a
        # top Pareto point merely to hit an arbitrary compute target.
        selected = mandatory + [
            row for row in selected if row not in mandatory
        ][:max(cap - len(mandatory), 0)]
    viable_families = {
        str(row.get("root_family", row.get("family", "unknown"))) for row in viable
    }
    selected_families = {
        str(row.get("root_family", row.get("family", "unknown"))) for row in selected
    }
    top_frontier_signatures = {_signature(row["mutations"]) for row in frontier[:8]}
    selected_signatures = {_signature(row["mutations"]) for row in selected}
    preservation_checks = {
        "mandatory_control": any(row.get("id") == "control_oversold" for row in selected),
        "every_viable_root_family": viable_families <= selected_families,
        "top_pareto_frontier": top_frontier_signatures <= selected_signatures,
    }
    catalog = {
        "source_candidates": len(rows),
        "broad_gate_survivors": len(viable),
        "shortlist_candidates": len(selected),
        "shortlist_soft_cap": cap,
        "pareto_candidates": len(frontier),
        "preservation_checks": preservation_checks,
        "preservation_gate_passed": all(preservation_checks.values()),
        "selection_reasons": {
            row["id"]: sorted(reasons[_signature(row["mutations"])]) for row in selected
        },
    }
    return selected, catalog


def _expanded_management_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("control", {}),
        ("secondary_size_050", {"param_overrides.pb_v2_secondary_route_sizing_mult": 0.50}),
        ("secondary_size_065", {"param_overrides.pb_v2_secondary_route_sizing_mult": 0.65}),
        ("failure_exit_4", {
            "param_overrides.pb_open_scored_stale_exit_bars": 4,
            "param_overrides.pb_open_scored_stale_exit_min_r": 0.10,
        }),
        ("failure_exit_6", {
            "param_overrides.pb_open_scored_stale_exit_bars": 6,
            "param_overrides.pb_open_scored_stale_exit_min_r": 0.15,
        }),
    )
    candidates = []
    for parent in parents:
        for name, changes in definitions:
            candidates.append(_candidate(
                parent,
                f"{parent['id']}__synthesis_management__{name}",
                changes,
                stage="synthesis_management_expansion",
                module=name,
            ))
    return candidates


def _candidate_gates(
    candidate: dict[str, Any], baseline: dict[str, Any], attribution: dict[str, Any], bootstrap: dict[str, Any],
) -> dict[str, bool]:
    metrics = candidate["metrics"]
    base = baseline["metrics"]
    validation = candidate.get("validation", {})
    folds = validation.get("folds", [])
    baseline_candidate = candidate["id"] == baseline["id"]
    trades = float(metrics.get("total_trades", 0.0))
    total_r = float(metrics.get("expected_total_r", 0.0))
    baseline_trades = float(base.get("total_trades", 0.0))
    baseline_total_r = float(base.get("expected_total_r", 0.0))
    return {
        "frequency_noninferior": baseline_candidate or trades >= 0.95 * baseline_trades,
        "total_r_noninferior": baseline_candidate or total_r >= 0.90 * baseline_total_r,
        "frequency_or_total_r_improves": baseline_candidate or (
            trades >= 1.15 * baseline_trades or total_r >= 1.03 * baseline_total_r
        ),
        "avg_r": float(metrics.get("avg_r", -99.0)) >= 0.12,
        "profit_factor": float(metrics.get("profit_factor", 0.0)) >= 1.45,
        "absolute_drawdown": float(metrics.get("max_drawdown_pct", 1.0)) <= 0.07,
        "relative_drawdown": baseline_candidate or (
            float(metrics.get("max_drawdown_pct", 1.0))
            <= float(base.get("max_drawdown_pct", 0.0)) + 0.015
        ),
        "robust_avg_r": float(metrics.get("robust_avg_r", -99.0)) >= 0.0,
        "folds": baseline_candidate or (
            sum(float(fold.get("total_r", 0.0)) > 0.0 for fold in folds) >= 2
            and bool(folds)
            and float(folds[-1].get("total_r", 0.0)) > 0.0
        ),
        "ex_top3": baseline_candidate or (
            float(attribution.get("ex_top3_total_r", 0.0)) > 0.0
            and float(attribution.get("ex_top3_profit_factor", 0.0)) >= 1.05
        ),
        "symbol_concentration": baseline_candidate or (
            float(attribution.get("max_single_symbol_net_share", math.inf)) <= 0.35
        ),
        "paired_bootstrap": baseline_candidate or float(bootstrap.get("probability_positive", 0.0)) >= 0.75,
    }


def _select_executable(
    finalists: list[dict[str, Any]], attribution_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in attribution_rows}
    baseline = next(row for row in finalists if row["id"] == "control_oversold")
    baseline_trades = by_id[baseline["id"]].get("trade_attribution", [])
    decorated = []
    for candidate in finalists:
        attribution_row = by_id[candidate["id"]]
        trades = attribution_row.get("trade_attribution", [])
        stats = _attribution_stats(trades)
        bootstrap = (
            {"probability_positive": 1.0, "observed_delta_r": 0.0}
            if candidate["id"] == baseline["id"]
            else _paired_block_bootstrap(trades, baseline_trades)
        )
        gates = _candidate_gates(candidate, baseline, stats, bootstrap)
        score, components, raw = _score(candidate)
        row = dict(candidate)
        row.update({
            "unified_score": score,
            "unified_score_components": components,
            "unified_score_raw": raw,
            "attribution": stats,
            "paired_bootstrap": bootstrap,
            "alpha_synthesis_gates": gates,
            "alpha_synthesis_eligible": all(gates.values()),
            "trade_attribution": trades,
        })
        decorated.append(row)
    eligible = [row for row in decorated if row["alpha_synthesis_eligible"]]
    selected = max(
        eligible or [row for row in decorated if row["id"] == baseline["id"]],
        key=lambda row: (
            float(row["unified_score"]),
            float(row["metrics"].get("expected_total_r", 0.0)),
            float(row["metrics"].get("total_trades", 0.0)),
        ),
    )
    return selected, sorted(decorated, key=lambda row: float(row["unified_score"]), reverse=True)


def _audit_stability(
    selected: dict[str, Any], audit_rows: list[dict[str, Any]], expected_audits: int,
) -> dict[str, Any]:
    """Require selected modules to add value and nearby settings to remain sane."""

    selected_metrics = selected["metrics"]
    selected_preliminary_score = _score({"metrics": selected_metrics})[0]
    details = []
    for row in audit_rows:
        row_score = _score({"metrics": row["metrics"]})[0]
        kind = "ablation" if str(row["id"]).startswith("audit__ablate") else "perturbation"
        if kind == "ablation":
            contribution = (
                selected_preliminary_score >= row_score + 0.002
                or float(selected_metrics.get("expected_total_r", 0.0))
                >= float(row["metrics"].get("expected_total_r", 0.0)) + 1.0
                or float(selected_metrics.get("total_trades", 0.0))
                >= float(row["metrics"].get("total_trades", 0.0)) + 3.0
            )
            noninferior = (
                float(selected_metrics.get("expected_total_r", 0.0))
                >= 0.90 * float(row["metrics"].get("expected_total_r", 0.0))
                and float(selected_metrics.get("total_trades", 0.0))
                >= 0.90 * float(row["metrics"].get("total_trades", 0.0))
                and float(selected_metrics.get("max_drawdown_pct", 1.0))
                <= float(row["metrics"].get("max_drawdown_pct", 0.0)) + 0.015
            )
            passed = contribution and noninferior
        else:
            passed = (
                float(row["metrics"].get("expected_total_r", -99.0))
                >= 0.65 * float(selected_metrics.get("expected_total_r", 0.0))
                and float(row["metrics"].get("profit_factor", 0.0)) >= 1.10
                and float(row["metrics"].get("avg_r", -99.0)) >= 0.05
                and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.10
            )
        details.append({
            "id": row["id"],
            "kind": kind,
            "passed": passed,
            "unified_score_without_fold_credit": row_score,
            "metrics": row["metrics"],
        })
    checks = {
        "all_requested_audits_completed": expected_audits > 0 and len(audit_rows) == expected_audits,
        "every_ablation_and_perturbation_passes": bool(details) and all(row["passed"] for row in details),
        "at_least_one_ablation": any(row["kind"] == "ablation" for row in details),
    }
    return {
        "selected_unified_score_without_fold_credit": selected_preliminary_score,
        "requested_audits": expected_audits,
        "completed_audits": len(audit_rows),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "details": details,
    }


def _value_creation_verification(
    selected: dict[str, Any],
    executable_rows: list[dict[str, Any]],
    cleaned_baseline: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    """Prove that Round 3 adds economic value rather than merely winning a score."""

    baseline = next(row for row in executable_rows if row["id"] == "control_oversold")
    metrics = selected["metrics"]
    base = baseline["metrics"]
    trades = float(metrics.get("total_trades", 0.0))
    base_trades = float(base.get("total_trades", 0.0))
    total_r = float(metrics.get("expected_total_r", 0.0))
    base_total_r = float(base.get("expected_total_r", 0.0))
    spec = FINAL_VALUE_VERIFICATION_SPEC
    return_led = (
        total_r >= (1.0 + spec["return_led_min_total_r_uplift"]) * base_total_r
        and trades >= spec["return_led_min_trade_retention"] * base_trades
    )
    frequency_led = (
        trades >= (1.0 + spec["frequency_led_min_trade_uplift"]) * base_trades
        and total_r >= spec["frequency_led_min_total_r_retention"] * base_total_r
    )
    folds = selected.get("validation", {}).get("folds", [])
    attribution = selected.get("attribution", {})
    bootstrap = selected.get("paired_bootstrap", {})
    changed_keys = sorted(
        key for key in set(cleaned_baseline) | set(selected["mutations"])
        if cleaned_baseline.get(key) != selected["mutations"].get(key)
    )
    checks = {
        "selected_is_not_round2_control": selected["id"] != "control_oversold",
        "economic_value_uplift": return_led or frequency_led,
        "unified_score_improves_by_at_least_0p005": (
            float(selected["unified_score"])
            >= float(baseline["unified_score"]) + spec["min_unified_score_uplift"]
        ),
        "paired_daily_delta_positive": float(bootstrap.get("observed_delta_r", 0.0)) > 0.0,
        "paired_block_bootstrap_at_least_75pct": (
            float(bootstrap.get("probability_positive", 0.0))
            >= spec["min_paired_bootstrap_probability"]
        ),
        "two_positive_folds_and_latest_positive": (
            len(folds) == 3
            and sum(float(fold.get("total_r", 0.0)) > 0.0 for fold in folds) >= 2
            and float(folds[-1].get("total_r", 0.0)) > 0.0
        ),
        "absolute_quality": (
            float(metrics.get("avg_r", -99.0)) >= spec["min_avg_r"]
            and float(metrics.get("profit_factor", 0.0)) >= spec["min_profit_factor"]
            and float(metrics.get("robust_avg_r", -99.0)) >= 0.0
        ),
        "drawdown_within_aggressive_but_bounded_limit": (
            float(metrics.get("max_drawdown_pct", 1.0)) <= spec["max_drawdown_pct"]
            and float(metrics.get("max_drawdown_pct", 1.0))
            <= float(base.get("max_drawdown_pct", 0.0)) + spec["max_drawdown_increase"]
        ),
        "alpha_not_top3_symbol_dependent": (
            float(attribution.get("ex_top3_total_r", 0.0)) > 0.0
            and float(attribution.get("ex_top3_profit_factor", 0.0)) >= 1.05
            and float(attribution.get("max_single_symbol_net_share", math.inf))
            <= spec["max_single_symbol_net_share"]
        ),
        "entry_discrimination_not_negative": (
            float(metrics.get("entry_realized_discrimination_lift_r", 0.0)) >= 0.0
        ),
        "mutation_surface_is_parsimonious": len(changed_keys) <= spec["max_changed_mutations"],
        "ablation_and_neighbour_stability": bool(stability.get("gate_passed")),
        "existing_alpha_synthesis_gates_pass": bool(selected.get("alpha_synthesis_eligible")),
    }
    return {
        "baseline_id": baseline["id"],
        "immutable_verification_spec": spec,
        "selected_id": selected["id"],
        "baseline_metrics": base,
        "selected_metrics": metrics,
        "deltas": {
            "expected_total_r": total_r - base_total_r,
            "total_trades": trades - base_trades,
            "avg_r": float(metrics.get("avg_r", 0.0)) - float(base.get("avg_r", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)) - float(base.get("profit_factor", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)) - float(base.get("max_drawdown_pct", 0.0)),
            "unified_score": float(selected["unified_score"]) - float(baseline["unified_score"]),
        },
        "economic_uplift_paths": {
            "return_led": return_led,
            "frequency_led_without_total_r_sacrifice": frequency_led,
        },
        "changed_keys": changed_keys,
        "paired_bootstrap": bootstrap,
        "stability": stability,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "boundary": "research-value gate on the pre-holdout legacy authority; not production promotion",
    }


def _credible_research2_candidate(row: dict[str, Any]) -> bool:
    """Keep positive near-misses visible even when the full route gate fails."""

    if row.get("selected_aperture") is None or row.get("selected_entry_variant") is None:
        return False
    folds = row.get("folds", {})
    middle = folds.get("middle", {})
    latest = folds.get("latest", {})
    return bool(
        int(middle.get("events", 0)) >= 20
        and int(latest.get("events", 0)) >= 20
        and float(middle.get("avg_r", 0.0)) > 0.0
        and float(latest.get("avg_r", 0.0)) > 0.0
        and float(row.get("validation_bootstrap_probability_positive", 0.0)) >= 0.80
    )


def _event_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["symbol"]), str(record["date"])


def _trade_keys(trades: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(trade["symbol"]), str(trade["entry_time"])[:10]) for trade in trades}


def _survivor_incremental_audit(
    family: str,
    walk_row: dict[str, Any],
    all_events: list[dict[str, Any]],
    incumbent_keys: set[tuple[str, str]],
    simulations: int,
) -> dict[str, Any]:
    aperture_name = str(walk_row["selected_aperture"])
    entry_variant = str(walk_row.get("selected_entry_variant", "next_bar_open"))
    horizon = str(walk_row["selected_horizon"])
    predicate = APERTURES[aperture_name]
    aperture_records = [
        record for record in all_events
        if record["family"] == family and predicate(record)
    ]
    selected = _records_for_entry_variant(aperture_records, entry_variant)
    unique = [record for record in selected if _event_key(record) not in incumbent_keys]
    folds = {
        fold: opportunity_metrics([record for record in unique if record["fold"] == fold], horizon)
        for fold in ("early", "middle", "latest")
    }
    unique_metrics = opportunity_metrics(unique, horizon)
    probability = _bootstrap_probability_positive(unique, horizon, simulations)
    unique_share = len(unique) / len(selected) if selected else 0.0
    gates = {
        "at_least_50_unique_events": len(unique) >= 50,
        "at_least_20pct_unique": unique_share >= 0.20,
        "positive_unique_avg_r": float(unique_metrics["avg_r"]) > 0.0,
        "positive_unique_total_r": float(unique_metrics["total_r"]) > 0.0,
        "middle_unique_positive": folds["middle"]["events"] >= 15 and folds["middle"]["avg_r"] > 0.0,
        "latest_unique_positive": folds["latest"]["events"] >= 15 and folds["latest"]["avg_r"] > 0.0,
        "bootstrap_probability_at_least_90pct": probability >= 0.90,
    }
    return {
        "family": family,
        "selected_aperture": aperture_name,
        "selected_entry_variant": entry_variant,
        "selected_horizon": horizon,
        "events": len(selected),
        "unique_events": len(unique),
        "unique_share": unique_share,
        "unique_metrics": unique_metrics,
        "unique_folds": folds,
        "unique_bootstrap_probability_positive": probability,
        "gates": gates,
        "admitted_for_exact_route_replay": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "event_keys": sorted([list(key) for key in {_event_key(record) for record in unique}]),
    }


def _pair_catalog(admitted: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    families = sorted(admitted)
    for left_index, left in enumerate(families):
        left_keys = {tuple(key) for key in admitted[left]["event_keys"]}
        for right in families[left_index + 1:]:
            right_keys = {tuple(key) for key in admitted[right]["event_keys"]}
            union = left_keys | right_keys
            overlap = len(left_keys & right_keys) / len(union) if union else 1.0
            pairs.append({"families": [left, right], "event_jaccard_overlap": overlap})
    eligible = [row for row in pairs if row["event_jaccard_overlap"] <= 0.50]
    eligible.sort(key=lambda row: (row["event_jaccard_overlap"], row["families"]))
    return eligible[:1]


def _render_report(result: dict[str, Any]) -> str:
    selected = result["selected_executable"]
    lines = [
        "# IARIC Round-3 Alpha Synthesis",
        "",
        f"Status: {result['status']}",
        f"Selected executable incumbent: {selected['id']}",
        f"Trades: {selected['metrics'].get('total_trades', 0):.0f}",
        f"Expected total R: {selected['metrics'].get('expected_total_r', 0):+.2f}",
        f"Average R: {selected['metrics'].get('avg_r', 0):+.3f}",
        f"Profit factor: {selected['metrics'].get('profit_factor', 0):.2f}",
        f"Max drawdown: {selected['metrics'].get('max_drawdown_pct', 0):.2%}",
        f"Search coverage gate: {'PASS' if result['search_coverage']['gate_passed'] else 'FAIL'}",
        f"Final value-verification gate: {'PASS' if result['final_outcome_verification']['gate_passed'] else 'FAIL'}",
        f"Expected total-R uplift vs control: {result['final_outcome_verification']['value_creation']['deltas']['expected_total_r']:+.2f}",
        f"Trade-count uplift vs control: {result['final_outcome_verification']['value_creation']['deltas']['total_trades']:+.0f}",
        f"Fresh unified-score finalists: {result['search_coverage']['fresh_validated_candidates']}",
        "",
        "## Structural reversion admission",
        "",
        "| Family | Entry | Events | Unique | Unique avg R | Latest avg R | Admitted? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in sorted(result["structural_audits"].items()):
        lines.append(
            f"| {family} | {row['selected_entry_variant']} | {row['events']} | {row['unique_events']} | "
            f"{row['unique_metrics']['avg_r']:+.3f} | {row['unique_folds']['latest']['avg_r']:+.3f} | "
            f"{'yes' if row['admitted_for_exact_route_replay'] else 'no'} |"
        )
    lines.extend([
        "",
        "ALCB and breakout-reference families are excluded from selection. They are magnitude/context "
        "references only. Canonical finalization is allowed only when no admitted reversion route remains "
        "outside an exact shared-core, shared-capital replay. Canonical Round 3 additionally requires a "
        "non-control selection, economic uplift without sacrificing total R, paired block-bootstrap support, "
        "fold/concentration/drawdown quality, and successful ablation plus neighbour audits.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if not 1 <= args.max_workers <= 2:
        raise ValueError("max-workers must be 1 or 2")
    if args.end_date >= "2026-03-02":
        raise ValueError("end date overlaps sealed holdout beginning 2026-03-02")
    branched_dir = Path(args.branched_dir).resolve()
    atlas_dir = Path(args.atlas_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "queue_status.json", {
        "status": "queued" if args.wait_for_pid > 0 else "starting",
        "waiting_for_pid": args.wait_for_pid,
        "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _wait_for_pid(args.wait_for_pid)
    _write_json(output_dir / "queue_status.json", {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    unified = _load_json(branched_dir / "unified/unified_selection.json")
    branched_spec = _load_json(branched_dir / "run_spec.json")
    walk = _load_json(atlas_dir / "walk_forward/walk_forward_summary.json")
    events = [
        json.loads(line)
        for line in (atlas_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if unified.get("holdout_accessed") is not False or walk.get("holdout_accessed") is not False:
        raise ValueError("upstream research does not prove holdout exclusion")
    if len(UNIFIED_SCORE_SPEC) != 7:
        raise AssertionError("alpha-synthesis score must have exactly seven components")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    # Research 1's top-four was selected by its earlier score. Recover every
    # evaluated branch, expand management around a broader unified-score entry
    # beam, then perform fresh folds and attribution. This is the key protection
    # against leaving frequency or total-R alpha behind at an old prune point.
    source_pool, source_stage_counts = _load_candidate_pool(branched_dir)
    source_baseline = next(row for row in source_pool if row.get("id") == "control_oversold")
    entry_pool = [
        row for row in source_pool
        if row.get("stage") in {"route", "route_additive"}
    ] + [source_baseline]
    entry_parents, entry_catalog = _broad_unified_shortlist(entry_pool, cap=10)
    evaluation_args = argparse.Namespace(
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
    )
    source_fingerprint = str(branched_spec["data_fingerprint"])
    code_fingerprint = _synthesis_code_fingerprint()
    management_candidates = _expanded_management_candidates(entry_parents)
    expanded_management = _evaluate(
        "synthesis_management_expansion",
        management_candidates,
        args=evaluation_args,
        output_dir=output_dir,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    combined_pool = source_pool + expanded_management
    combined_pool.sort(key=lambda row: row.get("id") != "control_oversold")
    unique_pool: dict[str, dict[str, Any]] = {}
    for row in combined_pool:
        unique_pool.setdefault(_signature(row["mutations"]), row)
    shortlist, shortlist_catalog = _broad_unified_shortlist(list(unique_pool.values()), cap=32)
    validated = _validation(
        shortlist,
        args=evaluation_args,
        output_dir=output_dir,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    attribution_rows = _evaluate(
        "synthesis_attribution",
        validated,
        args=evaluation_args,
        output_dir=output_dir,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        attribution=True,
    )
    selected, executable_rows = _select_executable(validated, attribution_rows)
    cleaned_baseline = _load_json(branched_dir / "cleaned_baseline_config.json")
    verification_candidates = _audit_candidates(selected, cleaned_baseline)
    verification_rows = (
        _evaluate(
            "value_verification_audit",
            verification_candidates,
            args=evaluation_args,
            output_dir=output_dir,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
        if verification_candidates else []
    )
    stability = _audit_stability(selected, verification_rows, len(verification_candidates))
    value_verification = _value_creation_verification(
        selected, executable_rows, cleaned_baseline, stability,
    )
    incumbent_keys = _trade_keys(selected["trade_attribution"])
    research2_candidates = {
        family: row
        for family, row in walk.get("families", {}).items()
        if family in ROUTE_READINESS
        and (bool(row.get("route_ready_for_portfolio_replay")) or _credible_research2_candidate(row))
    }
    structural_audits = {
        family: _survivor_incremental_audit(
            family,
            row,
            events,
            incumbent_keys,
            args.bootstrap_simulations,
        )
        for family, row in research2_candidates.items()
    }
    for family, audit in structural_audits.items():
        source = research2_candidates[family]
        audit["research2_evidence_class"] = (
            "full_walk_forward_survivor"
            if source.get("route_ready_for_portfolio_replay")
            else "credible_positive_near_miss"
        )
    admitted = {
        family: row for family, row in structural_audits.items()
        if row["admitted_for_exact_route_replay"]
    }
    candidate_catalog = {
        "isolated": sorted(admitted),
        "incumbent_plus_one": [
            {"incumbent": selected["id"], "route": family}
            for family in sorted(admitted)
        ],
        "incumbent_plus_low_overlap_pair": _pair_catalog(admitted),
        "cartesian_unions_allowed": False,
        "required_before_replay": {
            family: ROUTE_READINESS[family] for family in sorted(admitted)
        },
    }
    registered_reversion = set(
        walk.get("hypothesis_coverage", {}).get("implemented_reversion_families", [])
    )
    search_checks = {
        "all_research1_stage_artifacts_loaded": all(source_stage_counts.get(name, 0) > 0 for name in SOURCE_STAGE_FILES),
        "old_prune_bypassed_with_full_candidate_pool": len(source_pool) > len(_load_json(branched_dir / "validated_finalists.json")),
        "management_reexpanded_around_broad_unified_beam": len(expanded_management) >= len(entry_parents),
        "entry_beam_preserves_family_and_pareto_diversity": bool(entry_catalog["preservation_gate_passed"]),
        "validation_shortlist_preserves_family_and_pareto_diversity": bool(shortlist_catalog["preservation_gate_passed"]),
        "fresh_validation_completed_for_broad_shortlist": len(validated) == len(shortlist),
        "fresh_attribution_completed_for_broad_shortlist": len(attribution_rows) == len(validated),
        "mandatory_control_preserved": any(row.get("id") == "control_oversold" for row in validated),
        "exactly_seven_unified_score_components": len(UNIFIED_SCORE_SPEC) == 7,
        "atlas_route_registry_complete": registered_reversion <= set(ROUTE_READINESS),
        "holdout_excluded": args.end_date < "2026-03-02" and walk.get("holdout_accessed") is False,
    }
    search_coverage = {
        "scope": "bounded_exhaustive_within_registered_data_feasible_hypotheses_and_evaluated_research1_pool",
        "source_stage_counts": source_stage_counts,
        "source_unique_candidates": len(source_pool),
        "entry_parent_selection": entry_catalog,
        "management_candidates_requested": len(management_candidates),
        "management_unique_results": len(expanded_management),
        "combined_unique_candidates": len(unique_pool),
        "validation_shortlist": shortlist_catalog,
        "fresh_validated_candidates": len(validated),
        "fresh_attribution_candidates": len(attribution_rows),
        "research2_hypothesis_coverage": walk.get("hypothesis_coverage", {}),
        "checks": search_checks,
        "gate_passed": all(search_checks.values()),
        "boundary": "not a claim over unavailable event, pair, borrow, or positioning data",
    }
    research2_resolution_checks = {
        "every_full_survivor_or_credible_near_miss_audited": (
            set(research2_candidates) == set(structural_audits)
        ),
        "no_breakout_reference_used_for_selection": all(
            family in ROUTE_READINESS for family in structural_audits
        ),
        "no_incremental_route_silently_omitted": not admitted,
    }
    research2_resolution = {
        "full_or_near_miss_candidates": sorted(research2_candidates),
        "full_survivors": sorted(
            family for family, row in research2_candidates.items()
            if row.get("route_ready_for_portfolio_replay")
        ),
        "credible_positive_near_misses": sorted(
            family for family, row in research2_candidates.items()
            if not row.get("route_ready_for_portfolio_replay")
        ),
        "incremental_routes_requiring_exact_replay": sorted(admitted),
        "checks": research2_resolution_checks,
        "gate_passed": all(research2_resolution_checks.values()),
    }
    final_outcome_verification = {
        "value_creation": value_verification,
        "research2_resolution": research2_resolution,
        "search_coverage_passed": search_coverage["gate_passed"],
        "gate_passed": (
            value_verification["gate_passed"]
            and research2_resolution["gate_passed"]
            and search_coverage["gate_passed"]
        ),
    }
    candidate_catalog["search_coverage"] = search_coverage
    candidate_catalog["research2_resolution"] = research2_resolution
    finalizable = not admitted and final_outcome_verification["gate_passed"]
    status = (
        "ready_for_canonical_finalization"
        if finalizable
        else (
            "exact_structural_route_replay_required"
            if admitted
            else (
                "search_coverage_incomplete"
                if not search_coverage["gate_passed"]
                else "value_creation_not_verified"
            )
        )
    )
    result = {
        "status": status,
        "canonical_finalization_allowed": finalizable,
        "research_only": True,
        "holdout_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": "maximize robust IARIC expected total R and frequency; ALCB is reference-only",
        "unified_score_spec": UNIFIED_SCORE_SPEC,
        "score_component_count": len(UNIFIED_SCORE_SPEC),
        "code_fingerprint": code_fingerprint,
        "search_coverage": search_coverage,
        "final_outcome_verification": final_outcome_verification,
        "selected_executable": {
            key: selected[key]
            for key in (
                "id", "mutations", "metrics", "validation", "unified_score",
                "unified_score_components", "unified_score_raw", "alpha_synthesis_gates",
                "alpha_synthesis_eligible", "attribution", "paired_bootstrap",
            )
        },
        "executable_ranking": [
            {
                key: row[key]
                for key in (
                    "id", "metrics", "validation", "unified_score",
                    "unified_score_components", "alpha_synthesis_gates", "alpha_synthesis_eligible",
                )
            }
            for row in executable_rows
        ],
        "structural_audits": {
            family: {key: value for key, value in row.items() if key != "event_keys"}
            for family, row in structural_audits.items()
        },
        "admitted_structural_routes": sorted(admitted),
        "candidate_catalog": candidate_catalog,
        "reference_families_used_for_selection": [],
    }
    _write_json(output_dir / "alpha_synthesis_selection.json", result)
    _write_json(output_dir / "candidate_catalog.json", candidate_catalog)
    _write_json(output_dir / "final_value_verification.json", final_outcome_verification)
    _write_json(output_dir / "selected_executable_config.json", selected["mutations"])
    (output_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    _write_json(output_dir / "queue_status.json", {
        "status": "complete" if finalizable else "blocked",
        "result_status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(
        f"alpha synthesis complete: {status}; selected={selected['id']}; "
        f"admitted_structural_routes={len(admitted)}; holdout accessed=no",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failed_args = _parse_args()
        failed_output = Path(failed_args.output_dir).resolve()
        _write_json(failed_output / "queue_status.json", {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        raise
