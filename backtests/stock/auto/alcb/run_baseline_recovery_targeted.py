"""Bounded structural follow-up for the corrected-RTH ALCB recovery winner.

This is intentionally small: decompose the four-gate loose-geometry bundle,
then test only four interactions whose families were independently positive in
the preceding ablation.  Promotion requires fixed full-period, fold, and cost
evidence plus a minimum economic uplift; otherwise the family winner remains
the baseline.  The consumed post-2026-03-01 period is never accessed.
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
    ALCB_OUTPUT,
    CONSUMED_START,
    DATA_DIR,
    DEFAULT_OUTPUT,
    END_DATE,
    FOLDS,
    MAX_WORKERS,
    SCORE_SPEC,
    START_DATE,
    _compact_metrics,
    _cost_candidates,
    _cost_summary,
    _evaluate_batch,
    _final_rank_key,
    _fold_summary,
    _full_rank_key,
    _report_table,
    _safety_gate,
    _signature,
    _source_fingerprint,
    _code_fingerprint,
    _validated_rank_key,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--top-fold-finalists", type=int, default=4)
    parser.add_argument("--top-cost-finalists", type=int, default=4)
    parser.add_argument("--allow-projected-rth-data", action="store_true")
    return parser.parse_args()


def _candidate(
    name: str,
    family: str,
    mutations: dict[str, Any],
    thesis: str,
) -> dict[str, Any]:
    return {
        "id": f"targeted__{name}",
        "family": family,
        "era": "bounded_targeted_followup",
        "mutations": mutations,
        "sources": ["family__loose_entry_geometry"],
        "thesis": thesis,
    }


def _catalog(loose: dict[str, Any]) -> list[dict[str, Any]]:
    # Reconstruct the corrected Round 4 parent, then disable one broad gate or
    # one coherent pair at a time.  This separates true contributors from
    # passengers in the four-change family winner.
    parent = deepcopy(loose)
    parent.update(
        {
            "ablation.use_or_width_min": True,
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.pdh_avwap_cap_pct": 0.005,
            "param_overrides.combined_avwap_cap_pct": 0.003,
        }
    )

    def patch(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(base)
        result.update(changes)
        return result

    rows = [
        _candidate(
            "loose_geometry_control",
            "control",
            deepcopy(loose),
            "Unchanged validated family winner.",
        ),
        _candidate(
            "or_width_off_only",
            "atomic_geometry",
            patch(parent, {"ablation.use_or_width_min": False}),
            "Measure only the minimum OR-width gate.",
        ),
        _candidate(
            "orb_range_off_only",
            "atomic_geometry",
            patch(parent, {"ablation.use_orb_entry_range_gate": False}),
            "Measure only the completed signal-range cap.",
        ),
        _candidate(
            "pdh_avwap_cap_off_only",
            "atomic_geometry",
            patch(parent, {"param_overrides.pdh_avwap_cap_pct": 0.0}),
            "Measure only the PDH AVWAP premium cap.",
        ),
        _candidate(
            "combined_avwap_cap_off_only",
            "atomic_geometry",
            patch(parent, {"param_overrides.combined_avwap_cap_pct": 0.0}),
            "Measure only the combined-entry AVWAP premium cap.",
        ),
        _candidate(
            "or_geometry_pair_off",
            "paired_geometry",
            patch(
                parent,
                {
                    "ablation.use_or_width_min": False,
                    "ablation.use_orb_entry_range_gate": False,
                },
            ),
            "Measure the two RTH opening-range geometry gates together.",
        ),
        _candidate(
            "avwap_caps_pair_off",
            "paired_geometry",
            patch(
                parent,
                {
                    "param_overrides.pdh_avwap_cap_pct": 0.0,
                    "param_overrides.combined_avwap_cap_pct": 0.0,
                },
            ),
            "Measure the two entry-family AVWAP caps together.",
        ),
        _candidate(
            "avwap_caps_pair_plus_rvol_1p5",
            "supported_interaction",
            patch(
                parent,
                {
                    "param_overrides.pdh_avwap_cap_pct": 0.0,
                    "param_overrides.combined_avwap_cap_pct": 0.0,
                    "param_overrides.rvol_threshold": 1.5,
                },
            ),
            "Pair the two positive AVWAP-cap removals with RVOL 1.5 while retaining both OR protections.",
        ),
        _candidate(
            "loose_plus_flow_off",
            "supported_interaction",
            patch(loose, {"ablation.use_flow_reversal_exit": False}),
            "Combine loose geometry with the independently positive flow-exit removal.",
        ),
        _candidate(
            "loose_plus_rvol_1p5",
            "supported_interaction",
            patch(loose, {"param_overrides.rvol_threshold": 1.5}),
            "Combine loose geometry with independently positive broad RVOL acceptance.",
        ),
        _candidate(
            "loose_plus_equal_risk",
            "supported_interaction",
            patch(
                loose,
                {
                    "param_overrides.entry_score_size_mults": {},
                    "param_overrides.entry_detail_size_mults": {},
                    "param_overrides.pdh_size_mult": 1.0,
                    "param_overrides.regime_mult_b": 1.0,
                    "param_overrides.momentum_size_mult_score_3": 1.0,
                    "param_overrides.momentum_size_mult_score_4": 1.0,
                    "param_overrides.momentum_size_mult_score_5": 1.0,
                    "param_overrides.momentum_size_mult_score_6": 1.0,
                    "param_overrides.momentum_size_mult_score_7_plus": 1.0,
                },
            ),
            "Combine loose geometry with independently positive equal-risk sizing.",
        ),
        _candidate(
            "loose_plus_combined_gate_off",
            "supported_interaction",
            patch(
                loose,
                {
                    "ablation.use_combined_quality_gate": False,
                    "param_overrides.block_combined_regime_b": False,
                },
            ),
            "Combine the two independently positive broad entry-acceptance families.",
        ),
    ]
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(_signature(row["mutations"]), row)
    return list(deduped.values())


def _find_by_signature(rows: list[dict[str, Any]], signature: str) -> dict[str, Any]:
    return next(row for row in rows if _signature(row["mutations"]) == signature)


def _promotable(challenger: dict[str, Any], control: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    cm = challenger["metrics"]
    bm = control["metrics"]
    cv = challenger["validation"]
    bv = control["validation"]
    cc = challenger["costs"]
    bc = control["costs"]

    if not _safety_gate(cm):
        reasons.append("full-period safety gate failed")
    if not cv.get("robust_eligible"):
        reasons.append("chronological gate failed")
    if not cc.get("seven_five_gate"):
        reasons.append("7.5 bps gate failed")
    if float(challenger.get("economic_score", -99.0)) < float(control.get("economic_score", -99.0)) + 0.01:
        reasons.append("economic score uplift < 0.01")
    if float(cv.get("validated_score", -99.0)) < float(bv.get("validated_score", -99.0)) + 0.005:
        reasons.append("validated score uplift < 0.005")
    economic_uplift = (
        float(cm.get("expected_total_r", 0.0)) >= float(bm.get("expected_total_r", 0.0)) * 1.03
        or float(cm.get("trades_per_month", 0.0)) >= float(bm.get("trades_per_month", 0.0)) * 1.05
    )
    if not economic_uplift:
        reasons.append("neither R nor frequency improved materially")
    base_7p5_r = float(bc.get("7.5", {}).get("expected_total_r", 0.0))
    challenger_7p5_r = float(cc.get("7.5", {}).get("expected_total_r", -1e9))
    if challenger_7p5_r < max(0.0, base_7p5_r * 0.95):
        reasons.append("7.5 bps R retained < 95% of control")
    if float(cm.get("max_drawdown_pct", 1.0)) > 0.065:
        reasons.append("base-cost DD exceeds 6.5% targeted cap")
    return not reasons, reasons


def _render_report(manifest: dict[str, Any], full_ranked: list[dict[str, Any]]) -> str:
    selected = manifest["selected"]
    control = manifest["control"]
    lines = [
        "# ALCB targeted recovery follow-up",
        "",
        f"Decision: **{manifest['decision']}**",
        "",
        (
            f"Selected `{selected['id']}` with {selected['metrics']['expected_total_r']:+.2f}R, "
            f"{selected['metrics']['trades_per_month']:.2f} trades/month, PF "
            f"{selected['metrics']['profit_factor']:.3f}, and DD "
            f"{selected['metrics']['max_drawdown_pct']:.2%}."
        ),
        "",
        (
            f"Control `{control['id']}` had {control['metrics']['expected_total_r']:+.2f}R, "
            f"{control['metrics']['trades_per_month']:.2f} trades/month, PF "
            f"{control['metrics']['profit_factor']:.3f}, and DD "
            f"{control['metrics']['max_drawdown_pct']:.2%}."
        ),
        "",
        "## Full-period structural screen",
        "",
        "| Rank | Candidate | Family | R | TPM | PF | DD | Score |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(full_ranked, 1):
        m = row["metrics"]
        lines.append(
            f"| {rank} | {row['id']} | {row['family']} | {m.get('expected_total_r', 0):+.2f} | "
            f"{m.get('trades_per_month', 0):.2f} | {m.get('profit_factor', 0):.3f} | "
            f"{m.get('max_drawdown_pct', 0):.2%} | {row.get('economic_score', 0):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion rule",
            "",
            "A targeted candidate could replace the family winner only if it passed full-period, "
            "four-fold, and 7.5 bps gates; improved economic score by at least 0.01 and validated "
            "score by at least 0.005; materially improved R or frequency; retained at least 95% of "
            "the control's 7.5 bps R; and kept base-cost DD at or below 6.5%.",
            "",
            "The post-2026-03-01 period was not accessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if not args.allow_projected_rth_data:
        raise RuntimeError("Pass --allow-projected-rth-data for this diagnostic-only recovery.")
    if args.max_workers < 1 or args.max_workers > MAX_WORKERS:
        raise ValueError(f"Use between 1 and {MAX_WORKERS} workers.")
    if args.end_date >= CONSUMED_START:
        raise ValueError("Targeted recovery overlaps the excluded consumed period.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = args.output_dir.resolve()
    baseline_path = output_dir / "optimized_config.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Run baseline recovery first: {baseline_path}")
    # A rerun may follow a successful targeted promotion that overwrote the
    # canonical optimized_config.  Always recover the original loose-geometry
    # control from the persisted catalog so the bounded experiment definition
    # and its control signature remain immutable.
    prior_catalog_path = output_dir / "targeted_candidate_catalog.json"
    if prior_catalog_path.exists():
        prior_catalog = json.loads(prior_catalog_path.read_text(encoding="utf-8"))
        loose_row = next(
            (row for row in prior_catalog if row.get("id") == "targeted__loose_geometry_control"),
            None,
        )
        loose = deepcopy(loose_row["mutations"]) if loose_row else json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
    else:
        loose = json.loads(baseline_path.read_text(encoding="utf-8"))
    catalog = _catalog(loose)
    cache_path = output_dir / "evaluation_cache.json"
    source = _source_fingerprint()
    code = _code_fingerprint()

    print("=" * 78)
    print("ALCB TARGETED STRUCTURAL RECOVERY FOLLOW-UP")
    print("=" * 78)
    print(f"Candidates: {len(catalog)}")
    print(f"Training only: {args.start_date} -> {args.end_date}")
    print(f"Excluded consumed period begins: {CONSUMED_START}", flush=True)

    full = _evaluate_batch(
        catalog,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source=source,
        code=code,
    )
    errors = [row for row in full if row.get("error")]
    if errors:
        _write_json(output_dir / "targeted_errors.json", errors)
        raise RuntimeError(f"{len(errors)} targeted evaluations failed")
    full_ranked = sorted(full, key=_full_rank_key, reverse=True)
    control_sig = _signature(loose)
    finalists_by_sig: dict[str, dict[str, Any]] = {}
    for row in full_ranked[: max(1, args.top_fold_finalists)]:
        finalists_by_sig.setdefault(_signature(row["mutations"]), row)
    finalists_by_sig.setdefault(control_sig, _find_by_signature(full, control_sig))
    finalists = list(finalists_by_sig.values())

    fold_results: dict[str, list[dict[str, Any]]] = {}
    for name, start, end in FOLDS:
        fold_results[name] = _evaluate_batch(
            finalists,
            start_date=start,
            end_date=end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for row in finalists:
        row["validation"] = _fold_summary(row, fold_results)
    validated = sorted(finalists, key=_validated_rank_key, reverse=True)

    cost_finalists = validated[: max(1, args.top_cost_finalists)]
    if all(_signature(row["mutations"]) != control_sig for row in cost_finalists):
        cost_finalists.append(_find_by_signature(validated, control_sig))
    cost_results: dict[float, list[dict[str, Any]]] = {}
    for cost in (7.5, 10.0):
        cost_results[cost] = _evaluate_batch(
            _cost_candidates(cost_finalists, cost),
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for row in cost_finalists:
        row["costs"] = _cost_summary(row, cost_results)
    final_ranked = sorted(cost_finalists, key=_final_rank_key, reverse=True)
    control = _find_by_signature(final_ranked, control_sig)
    challenger = final_ranked[0]
    if _signature(challenger["mutations"]) == control_sig:
        winner = control
        promoted = False
        promotion_reasons = ["control retained the best fixed final rank"]
    else:
        promoted, promotion_reasons = _promotable(challenger, control)
        winner = challenger if promoted else control

    decision = "promote_targeted_structural_candidate" if promoted else "retain_loose_geometry_family_winner"
    selected_config = dict(sorted(winner["mutations"].items()))
    _write_json(output_dir / "optimized_config.json", selected_config)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "provisional_direct_rth_revalidation_required",
        "decision": decision,
        "promoted": promoted,
        "promotion_reasons": promotion_reasons,
        "data_authority": "derived_legacy_extended_cache_filtered_to_versioned_rth_diagnostic_only",
        "data_source_fingerprint": source,
        "code_fingerprint": code,
        "training_window": {"start": args.start_date, "end": args.end_date},
        "excluded_period": {"start": CONSUMED_START, "accessed": False},
        "immutable_score": SCORE_SPEC,
        "candidate_count": len(catalog),
        "control": {
            "id": control["id"],
            "signature": control_sig,
            "metrics": _compact_metrics(control["metrics"]),
            "economic_score": control["economic_score"],
            "validation": control["validation"],
            "costs": control["costs"],
        },
        "selected": {
            "id": winner["id"],
            "family": winner["family"],
            "signature": _signature(selected_config),
            "metrics": _compact_metrics(winner["metrics"]),
            "economic_score": winner["economic_score"],
            "validation": winner["validation"],
            "costs": winner["costs"],
        },
        "promotion_policy": (
            "A bounded targeted candidate must clear fixed economics, stability, cost, "
            "material-uplift, and drawdown tests; otherwise retain the family winner."
        ),
    }
    _write_json(output_dir / "final_recovery_manifest.json", manifest)
    _write_json(output_dir / "targeted_candidate_catalog.json", catalog)
    _write_json(output_dir / "targeted_full_ranking.json", _report_table(full_ranked))
    _write_json(output_dir / "targeted_validated_finalists.json", _report_table(validated))
    _write_json(output_dir / "targeted_cost_finalists.json", _report_table(final_ranked))
    (output_dir / "targeted_recovery_report.md").write_text(
        _render_report(manifest, full_ranked), encoding="utf-8"
    )

    m = manifest["selected"]["metrics"]
    c = manifest["selected"]["costs"]
    summary = [
        "=" * 78,
        "ALCB TARGETED STRUCTURAL RECOVERY COMPLETE",
        "=" * 78,
        f"Decision: {decision}",
        f"Selected: {manifest['selected']['id']} ({manifest['selected']['family']})",
        f"Signature: {manifest['selected']['signature']}",
        (
            f"Trades={m['total_trades']:.0f} TPM={m['trades_per_month']:.2f} "
            f"R={m['expected_total_r']:+.2f} Net=${m['net_profit']:+,.2f} "
            f"AvgR={m['avg_r']:+.4f} PF={m['profit_factor']:.3f} DD={m['max_drawdown_pct']:.2%}"
        ),
        (
            f"7.5bps R={c.get('7.5', {}).get('expected_total_r', 0):+.2f} "
            f"PF={c.get('7.5', {}).get('profit_factor', 0):.3f}; "
            f"10bps R={c.get('10.0', {}).get('expected_total_r', 0):+.2f} "
            f"PF={c.get('10.0', {}).get('profit_factor', 0):.3f}"
        ),
        f"Config: {output_dir / 'optimized_config.json'}",
        "The post-2026-03-01 period was not accessed.",
    ]
    (output_dir / "final_recovery_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
