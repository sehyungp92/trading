"""Build an honest ALCB research baseline and a bounded development challenger.

Round 2 is frozen as the last control that predates the consumed 2026-03-02
through 2026-05-01 selection window.  Round 3 is retained as a contaminated
diagnostic challenger, never as the representative baseline.  The runner:

* imports only the IS side of the prior effective-mutation audit;
* evaluates explicit controls, grouped simplifications, and bounded targeted
  perturbations through 2026-03-01;
* validates finalists on four chronological folds and 7.5/10 bps replays; and
* writes research artifacts without mutating historical rounds or manifests.

The local cache is a projected-RTH diagnostic source, so this workflow cannot
authorize production promotion.  A fresh lockbox and authoritative direct-RTH
bundle remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from backtests.stock.auto.alcb.run_baseline_recovery import (
    ALCB_OUTPUT,
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
    _full_rank_key,
    _safety_gate,
    _signature,
    _source_fingerprint,
    _validated_rank_key,
    _write_json,
)
from backtests.stock.data.calendar import RTH_SESSION_POLICY


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND2_CONFIG = ALCB_OUTPUT / "round_2/optimized_config.json"
ROUND3_CONFIG = ALCB_OUTPUT / "round_3/optimized_config.json"
DEFAULT_OUTPUT = ALCB_OUTPUT / "representative_baseline_20260821"
IS_AUDIT_DIR = (
    ALCB_OUTPUT / "round_2/oos_ablation_perturbation_verified_20260816"
)
READINESS_PATH = (
    REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--top-fold-finalists", type=int, default=8)
    parser.add_argument("--top-cost-finalists", type=int, default=4)
    parser.add_argument("--allow-projected-rth-data", action="store_true")
    return parser.parse_args()


def _read_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _normalized_config(path: Path) -> dict[str, Any]:
    config = _read_dict(path)
    config["intraday_session_policy"] = RTH_SESSION_POLICY
    return config


def _changed_keys(base: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    keys = set(base) | set(candidate)
    return sorted(key for key in keys if base.get(key) != candidate.get(key))


def _candidate(
    candidate_id: str,
    family: str,
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    origin: str,
    thesis: str,
    selection_eligible: bool = True,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "family": family,
        "era": origin,
        "mutations": dict(sorted(config.items())),
        "sources": [origin],
        "origin": origin,
        "thesis": thesis,
        "changed_keys": _changed_keys(baseline, config),
        "selection_eligible": selection_eligible,
    }


def _patched(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    result.update(changes)
    result["intraday_session_policy"] = RTH_SESSION_POLICY
    return result


def _catalog(round2: dict[str, Any], round3: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        _candidate(
            "control__round2_exact",
            "control",
            round2,
            round2,
            origin="pre_consumed_oos_round2",
            thesis="Immutable representative baseline selected before the consumed OOS search.",
        ),
        _candidate(
            "diagnostic__current_round3",
            "contaminated_control",
            round3,
            round2,
            origin="consumed_oos_selected_round3",
            thesis="Saved current Round 3; diagnostic only because its lineage consumed OOS.",
            selection_eligible=False,
        ),
    ]

    def add(
        candidate_id: str,
        family: str,
        changes: dict[str, Any],
        thesis: str,
        *,
        selection_eligible: bool = True,
    ) -> None:
        rows.append(
            _candidate(
                candidate_id,
                family,
                _patched(round2, changes),
                round2,
                origin="post_oos_informed_development",
                thesis=thesis,
                selection_eligible=selection_eligible,
            )
        )

    # Structural controls and grouped simplifications.  The adaptive-trail
    # removal is a negative control and cannot become a selected challenger.
    add(
        "negative__adaptive_trail_off",
        "negative_control",
        {"ablation.use_adaptive_trail": False},
        "Verify that the long-winner architecture remains indispensable.",
        selection_eligible=False,
    )
    add(
        "group__early_exit_layers_off",
        "grouped_simplification",
        {
            "param_overrides.failure_stop_bars": 0,
            "ablation.use_mfe_conviction_exit": False,
        },
        "Remove the two small early-failure overlays together.",
    )
    add(
        "group__combined_filters_off",
        "grouped_simplification",
        {
            "ablation.use_combined_quality_gate": False,
            "param_overrides.block_combined_regime_b": False,
            "param_overrides.combined_breakout_score_min": 0,
        },
        "Test whether the layered combined-breakout exclusions are redundant.",
    )
    add(
        "group__signal_sizing_flattened",
        "grouped_simplification",
        {
            "param_overrides.entry_score_size_mults": {},
            "param_overrides.entry_detail_size_mults": {},
            "param_overrides.pdh_size_mult": 1.0,
        },
        "Test whether score/detail sizing complexity adds portable value.",
    )
    add(
        "group__carry_filters_neutral",
        "grouped_simplification",
        {
            "param_overrides.carry_min_cpr": 0.0,
            "param_overrides.carry_min_r": 0.0,
            "param_overrides.fr_cpr_threshold": 0.0,
        },
        "Neutralize weak or inactive carry/CPR restrictions.",
    )
    add(
        "core__simplified_v1",
        "minimal_core",
        {
            "param_overrides.failure_stop_bars": 0,
            "ablation.use_mfe_conviction_exit": False,
            "param_overrides.carry_min_cpr": 0.0,
            "param_overrides.carry_min_r": 0.0,
            "param_overrides.fr_cpr_threshold": 0.0,
            "param_overrides.combined_breakout_score_min": 0,
        },
        "Retain entries, adaptive trailing, and risk while removing weak overlays.",
    )
    add(
        "group__opening_range_guards_off",
        "structural_ablation",
        {
            "ablation.use_or_width_min": False,
            "ablation.use_orb_entry_range_gate": False,
        },
        "Test the opening-range geometry protections as a coherent group.",
    )

    # Atomic cleanup candidates supported by the IS-only mutation audit.
    atomic = (
        ("flow_hold_0", {"param_overrides.flow_reversal_min_hold_bars": 0}),
        ("entry_blocklist_empty", {"param_overrides.entry_score_blocklist": []}),
        ("pdh_equal_size", {"param_overrides.pdh_size_mult": 1.0}),
        ("failure_stop_off", {"param_overrides.failure_stop_bars": 0}),
        ("combined_regime_unblocked", {"param_overrides.block_combined_regime_b": False}),
        ("combined_score_floor_off", {"param_overrides.combined_breakout_score_min": 0}),
        ("mfe_exit_off", {"ablation.use_mfe_conviction_exit": False}),
        ("combined_quality_off", {"ablation.use_combined_quality_gate": False}),
    )
    for name, changes in atomic:
        add(
            f"atomic__{name}",
            "atomic_simplification",
            changes,
            f"IS-supported one-setting simplification: {name}.",
        )

    # Pre-registered narrow RVOL surface.  1.1 is represented only by the
    # contaminated saved Round 3 control and cannot win development selection.
    for value in (1.2, 1.4, 1.6, 1.8, 1.9):
        add(
            f"rvol__{str(value).replace('.', 'p')}",
            "rvol_plateau",
            {"param_overrides.rvol_threshold": value},
            "Find a broad RVOL plateau rather than select a single sharp optimum.",
        )

    # Bounded exit robustness around the current 25-bar/0.04R/0.22R settings.
    for bars in (20, 30):
        add(
            f"trail_timing__{bars}",
            "winner_management",
            {
                "param_overrides.adaptive_trail_start_bars": bars,
                "param_overrides.adaptive_trail_tighten_bars": bars,
            },
            "Test whether long-winner performance persists around the trail timing.",
        )
    for value in (0.03, 0.06, 0.08):
        add(
            f"trail_distance__{str(value).replace('.', 'p')}",
            "winner_management",
            {"param_overrides.adaptive_trail_late_distance_r": value},
            "Test a bounded neighbourhood around the late-trail distance.",
        )
    for value in (0.18, 0.26):
        add(
            f"trail_activate__{str(value).replace('.', 'p')}",
            "winner_management",
            {"param_overrides.adaptive_trail_late_activate_r": value},
            "Test a bounded neighbourhood around late-trail activation.",
        )

    # Early-failure controls use only information available after entry.
    for bars in (4, 12):
        add(
            f"flow_hold__{bars}",
            "early_failure",
            {"param_overrides.flow_reversal_min_hold_bars": bars},
            "Test a broad early-flow timing alternative without symbol/date exclusions.",
        )
    for bars in (8, 12):
        add(
            f"failure_stop__{bars}",
            "early_failure",
            {"param_overrides.failure_stop_bars": bars},
            "Test whether failure-stop timing is stable around ten bars.",
        )
    for bars in (12, 20):
        add(
            f"mfe_check__{bars}",
            "early_failure",
            {"param_overrides.mfe_conviction_check_bars": bars},
            "Test whether the MFE checkpoint is stable around sixteen bars.",
        )

    # Preserve the first semantically named candidate for duplicate signatures
    # (notably the Round 2 control and RVOL 1.4).
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(_signature(row["mutations"]), row)
    return list(deduped.values())


def _metadata_by_signature(catalog: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_signature(row["mutations"]): row for row in catalog}


def _enrich(
    records: Iterable[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in records:
        enriched = dict(row)
        meta = metadata.get(str(row.get("signature")), {})
        for key in ("origin", "thesis", "changed_keys", "selection_eligible"):
            if key in meta:
                enriched[key] = meta[key]
        output.append(enriched)
    return output


def _is_ablation_evidence() -> dict[str, Any]:
    audit_path = IS_AUDIT_DIR / "literal_removal_audit.json"
    validation_path = IS_AUDIT_DIR / "is_validation.json"
    literal = json.loads(audit_path.read_text(encoding="utf-8"))
    validation = _read_dict(validation_path)
    baseline = validation["baseline_metrics"]
    rows = []
    for record in validation["results"].values():
        if record.get("stage") != "ablation":
            continue
        metrics = record.get("metrics", {})
        rows.append(
            {
                "name": record["name"],
                "atomic_key": record.get("atomic_key"),
                "patch": record.get("patch", {}),
                "is_guardrail_pass": bool(record.get("is_guardrail_pass")),
                "is_metrics": _compact_metrics(metrics),
                "is_delta_total_r": float(metrics.get("expected_total_r", 0.0))
                - float(baseline.get("expected_total_r", 0.0)),
            }
        )
    rows.sort(key=lambda row: row["is_delta_total_r"], reverse=True)
    return {
        "source_policy": "IS-only import; consumed OOS metrics are not read",
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "validation_path": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "literal_mutation_count": len(literal),
        "effective_literal_count": sum(
            bool(row.get("literal_removal_changes_effective_config")) for row in literal
        ),
        "is_ablation_result_count": len(rows),
        "baseline_metrics": _compact_metrics(baseline),
        "results": rows,
    }


def _record_for_id(records: Iterable[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    return next(row for row in records if row["id"] == candidate_id)


def _delta_metrics(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    cm = candidate["metrics"]
    bm = control["metrics"]
    return {
        key: float(cm.get(key, 0.0)) - float(bm.get(key, 0.0))
        for key in (
            "expected_total_r",
            "net_profit",
            "avg_r",
            "profit_factor",
            "trades_per_month",
            "max_drawdown_pct",
        )
    }


def _challenger_gate(candidate: dict[str, Any], control: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    metrics = candidate["metrics"]
    baseline = control["metrics"]
    validation = candidate.get("validation", {})
    costs = candidate.get("costs", {})
    control_costs = control.get("costs", {})
    if not candidate.get("selection_eligible", False):
        reasons.append("candidate is diagnostic-only or OOS-contaminated")
    if not _safety_gate(metrics):
        reasons.append("full-period safety gate failed")
    if not validation.get("robust_eligible"):
        reasons.append("chronological robustness gate failed")
    if not costs.get("seven_five_gate") or not costs.get("ten_gate"):
        reasons.append("7.5/10 bps cost gate failed")
    if float(metrics.get("expected_total_r", 0.0)) < float(baseline["expected_total_r"]) * 0.98:
        reasons.append("retained less than 98% of control total R")
    if float(metrics.get("avg_r", 0.0)) < float(baseline["avg_r"]) * 0.95:
        reasons.append("retained less than 95% of control AvgR")
    if float(metrics.get("profit_factor", 0.0)) < float(baseline["profit_factor"]) * 0.95:
        reasons.append("retained less than 95% of control PF")
    dd_cap = max(0.05, float(baseline["max_drawdown_pct"]) * 1.25)
    if float(metrics.get("max_drawdown_pct", 1.0)) > dd_cap:
        reasons.append("drawdown exceeded the relative cap")
    candidate_cost_r = float(costs.get("7.5", {}).get("expected_total_r", -1e9))
    control_cost_r = float(control_costs.get("7.5", {}).get("expected_total_r", 0.0))
    if candidate_cost_r < max(0.0, control_cost_r * 0.95):
        reasons.append("7.5 bps R retained less than 95% of control")
    return not reasons, reasons


def _parsimony_pick(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in records if row.get("challenger_eligible")]
    if not eligible:
        return None
    for row in eligible:
        row["parsimony_score"] = float(row["validation"]["validated_score"]) - (
            0.002 * len(row.get("changed_keys", []))
        )
    best_score = max(float(row["parsimony_score"]) for row in eligible)
    band = [row for row in eligible if float(row["parsimony_score"]) >= best_score - 0.01]
    return sorted(
        band,
        key=lambda row: (
            len(row.get("changed_keys", [])),
            -float(row["parsimony_score"]),
            row["id"],
        ),
    )[0]


def _fold_overview(record: dict[str, Any]) -> dict[str, Any]:
    validation = record.get("validation", {})
    fold_rows = validation.get("folds", [])
    return {
        "positive_fold_count": validation.get("positive_fold_count", 0),
        "worst_fold_avg_r": validation.get("worst_fold_avg_r"),
        "median_fold_avg_r": median(
            [float(row["avg_r"]) for row in fold_rows]
        )
        if fold_rows
        else None,
        "minimum_fold_profit_factor": validation.get("minimum_fold_profit_factor"),
        "maximum_fold_drawdown_pct": validation.get("maximum_fold_drawdown_pct"),
    }


def _render_report(manifest: dict[str, Any], ranked: list[dict[str, Any]]) -> str:
    baseline = manifest["representative_baseline"]
    challenger = manifest.get("development_challenger")
    lines = [
        "# ALCB representative baseline sequence",
        "",
        "## Decision",
        "",
        (
            f"Round 2 remains the immutable representative baseline: "
            f"{baseline['metrics']['expected_total_r']:+.2f}R, PF "
            f"{baseline['metrics']['profit_factor']:.3f}, and "
            f"{baseline['metrics']['max_drawdown_pct']:.2%} daily-close drawdown."
        ),
        "",
    ]
    if challenger:
        lines.extend(
            [
                (
                    f"The bounded development winner is `{challenger['id']}` at "
                    f"{challenger['metrics']['expected_total_r']:+.2f}R and PF "
                    f"{challenger['metrics']['profit_factor']:.3f}. It is a challenger "
                    "for a future lockbox, not a promoted baseline."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No development candidate cleared every relative chronology, quality, risk, and cost gate.",
                "",
            ]
        )
    lines.extend(
        [
            "The current data source is projected RTH and non-authoritative. No historical round or manifest was changed.",
            "",
            "## Full-period development ranking",
            "",
            "| Candidate | Family | R | AvgR | PF | Trades/mo | DD | Eligible origin |",
            "|---|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in ranked[:15]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['id']} | {row.get('family', '')} | "
            f"{metrics.get('expected_total_r', 0):+.2f} | {metrics.get('avg_r', 0):+.4f} | "
            f"{metrics.get('profit_factor', 0):.3f} | {metrics.get('trades_per_month', 0):.1f} | "
            f"{metrics.get('max_drawdown_pct', 0):.2%} | "
            f"{'yes' if row.get('selection_eligible') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Validation policy",
            "",
            "- Candidate generation and replay stop at 2026-03-01.",
            "- Current Round 3 is diagnostic-only because its candidate lineage consumed OOS.",
            "- The prior mutation dependency map is imported from IS-only validation fields.",
            "- Four chronological folds and full causal 7.5/10 bps replays are mandatory challenger gates.",
            "- A direct-RTH replay, conservative stop-gap stress, intraday mark-to-market drawdown, and a genuinely unseen lockbox remain external promotion gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if not args.allow_projected_rth_data:
        raise RuntimeError(
            "Authoritative direct-RTH data are unavailable. Pass "
            "--allow-projected-rth-data for diagnostic research output."
        )
    if args.max_workers < 1 or args.max_workers > MAX_WORKERS:
        raise ValueError(f"Use between 1 and {MAX_WORKERS} workers.")
    if END_DATE >= CONSUMED_START:
        raise RuntimeError("Development window overlaps the consumed OOS period.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "evaluation_cache.json"
    source = _source_fingerprint()
    code = _code_fingerprint()
    round2 = _normalized_config(ROUND2_CONFIG)
    round3 = _normalized_config(ROUND3_CONFIG)
    catalog = _catalog(round2, round3)
    metadata = _metadata_by_signature(catalog)
    ablation_evidence = _is_ablation_evidence()

    print("=" * 78, flush=True)
    print("ALCB REPRESENTATIVE BASELINE SEQUENCE", flush=True)
    print("=" * 78, flush=True)
    print(f"Development window: {START_DATE} -> {END_DATE}", flush=True)
    print(f"Consumed OOS excluded from replay: {CONSUMED_START} onward", flush=True)
    print(f"Candidates: {len(catalog)}; workers: {args.max_workers}", flush=True)

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
    errors = [row for row in full if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} full-period evaluations failed")
    ranked = sorted(full, key=_full_rank_key, reverse=True)

    required_ids = {
        "control__round2_exact",
        "diagnostic__current_round3",
        "core__simplified_v1",
    }
    fold_map: dict[str, dict[str, Any]] = {}
    for row in ranked:
        if row["id"] in required_ids:
            fold_map[row["signature"]] = row
    for row in ranked:
        if not row.get("selection_eligible") or row.get("family") == "negative_control":
            continue
        fold_map.setdefault(row["signature"], row)
        if len(fold_map) >= args.top_fold_finalists + 2:
            break
    fold_finalists = list(fold_map.values())

    fold_results: dict[str, list[dict[str, Any]]] = {}
    for name, fold_start, fold_end in FOLDS:
        fold_results[name] = _evaluate_batch(
            fold_finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for row in fold_finalists:
        row["validation"] = _fold_summary(row, fold_results)
    validated = sorted(fold_finalists, key=_validated_rank_key, reverse=True)

    cost_map: dict[str, dict[str, Any]] = {}
    for candidate_id in ("control__round2_exact", "diagnostic__current_round3"):
        row = next((item for item in validated if item["id"] == candidate_id), None)
        if row:
            cost_map[row["signature"]] = row
    for row in validated:
        if row.get("selection_eligible"):
            cost_map.setdefault(row["signature"], row)
        if len(cost_map) >= args.top_cost_finalists + 2:
            break
    cost_finalists = list(cost_map.values())
    cost_results: dict[float, list[dict[str, Any]]] = {}
    for cost in (7.5, 10.0):
        cost_results[cost] = _evaluate_batch(
            _cost_candidates(cost_finalists, cost),
            start_date=START_DATE,
            end_date=END_DATE,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for row in cost_finalists:
        row["costs"] = _cost_summary(row, cost_results)

    control = _record_for_id(cost_finalists, "control__round2_exact")
    for row in cost_finalists:
        eligible, reasons = _challenger_gate(row, control)
        row["challenger_eligible"] = bool(eligible and row["id"] != control["id"])
        row["challenger_gate_reasons"] = reasons
        row["delta_vs_round2"] = _delta_metrics(row, control)
    challenger = _parsimony_pick(
        [row for row in cost_finalists if row["id"] != control["id"]]
    )

    readiness = _read_dict(READINESS_PATH) if READINESS_PATH.exists() else {}
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "representative_baseline_frozen_development_challenger_only",
        "development_window": {"start": START_DATE, "end": END_DATE},
        "consumed_oos_policy": {
            "start": CONSUMED_START,
            "replayed": False,
            "round3_selection_contamination_acknowledged": True,
        },
        "data_authority": {
            "authoritative": False,
            "projected_rth_diagnostic_only": True,
            "frozen_bundle_available": readiness.get("frozen_bundle_available", False),
            "accepted_latest_count": readiness.get("accepted_latest_count", 0),
            "missing_dataset_count": readiness.get("missing_dataset_count"),
        },
        "provenance": {
            "source_fingerprint": source,
            "code_fingerprint": code,
            "orchestration_sha256": _sha256(Path(__file__)),
            "round2_config": str(ROUND2_CONFIG),
            "round2_config_sha256": _sha256(ROUND2_CONFIG),
            "round3_config": str(ROUND3_CONFIG),
            "round3_config_sha256": _sha256(ROUND3_CONFIG),
        },
        "candidate_count": len(catalog),
        "fold_finalist_count": len(fold_finalists),
        "cost_finalist_count": len(cost_finalists),
        "representative_baseline": {
            "id": control["id"],
            "signature": control["signature"],
            "metrics": _compact_metrics(control["metrics"]),
            "folds": _fold_overview(control),
            "validation": control["validation"],
            "costs": control["costs"],
            "policy": "immutable until authoritative replay and a fresh lockbox",
        },
        "development_challenger": (
            {
                "id": challenger["id"],
                "family": challenger.get("family"),
                "signature": challenger["signature"],
                "changed_keys": challenger.get("changed_keys", []),
                "metrics": _compact_metrics(challenger["metrics"]),
                "delta_vs_round2": challenger["delta_vs_round2"],
                "folds": _fold_overview(challenger),
                "validation": challenger["validation"],
                "costs": challenger["costs"],
                "policy": "future-lockbox challenger; not promoted",
            }
            if challenger
            else None
        ),
        "unresolved_promotion_gates": [
            "accepted frozen direct-RTH bundle",
            "point-in-time or ex-ante frozen universe",
            "conservative gap-through-stop replay",
            "intraday mark-to-market drawdown",
            "genuinely unseen post-research lockbox",
            "paper/live fill confirmation",
        ],
    }

    _write_json(output_dir / "representative_baseline_config.json", round2)
    if challenger:
        _write_json(
            output_dir / "development_challenger_config.json",
            challenger["mutations"],
        )
    _write_json(output_dir / "ablation_dependency_evidence.json", ablation_evidence)
    _write_json(output_dir / "candidate_catalog.json", catalog)
    _write_json(output_dir / "full_period_results.json", ranked)
    _write_json(output_dir / "fold_validated_results.json", validated)
    _write_json(output_dir / "cost_validated_results.json", cost_finalists)
    _write_json(output_dir / "sequence_manifest.json", manifest)
    (output_dir / "sequence_report.md").write_text(
        _render_report(manifest, ranked), encoding="utf-8"
    )

    bm = manifest["representative_baseline"]["metrics"]
    print("=" * 78, flush=True)
    print("REPRESENTATIVE BASELINE SEQUENCE COMPLETE", flush=True)
    print(
        f"Round 2 baseline: {bm['expected_total_r']:+.2f}R, "
        f"PF {bm['profit_factor']:.3f}, DD {bm['max_drawdown_pct']:.2%}",
        flush=True,
    )
    if challenger:
        cm = manifest["development_challenger"]["metrics"]
        print(
            f"Development challenger: {challenger['id']} "
            f"{cm['expected_total_r']:+.2f}R, PF {cm['profit_factor']:.3f}",
            flush=True,
        )
    else:
        print("Development challenger: none cleared all gates", flush=True)
    print(f"Report: {output_dir / 'sequence_report.md'}", flush=True)


if __name__ == "__main__":
    main()
