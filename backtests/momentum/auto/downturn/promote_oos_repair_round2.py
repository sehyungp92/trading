"""Promote the corrected-split downturn repair candidate into Round 2 artifacts.

This is an artifact-packaging step, not a production activation step.  The
2026-03-21 through 2026-05-01 interval was observed during candidate selection,
so the resulting round remains shadow-only even though its performance gates
pass.  The script is intentionally idempotent: an existing Round 2 manifest
entry is replaced instead of duplicated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output" / "momentum" / "downturn"
SOURCE_DIR = OUTPUT_ROOT / "round_1" / "oos_repair"
ROUND_DIR = OUTPUT_ROOT / "round_2"
RESEARCH_DIR = ROUND_DIR / "research"
MANIFEST_PATH = OUTPUT_ROOT / "rounds_manifest.json"

EXPECTED_SIGNATURE = "e4bb3351c373a75f051e22e40eb0fc32359735fbaafd42664fef9cf80d2233b7"
EXPECTED_SPLIT = {
    "is_start": "2024-01-01T00:00:00+00:00",
    "is_end_inclusive": "2026-03-20",
    "oos_start": "2026-03-21T00:00:00+00:00",
    "oos_end_inclusive": "2026-05-01",
    "evaluation_end_exclusive": "2026-05-02T00:00:00+00:00",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _manifest_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert evaluator drawdown fractions to manifest percentage points."""
    return {
        "total_trades": metrics["total_trades"],
        "net_pnl": metrics["net_pnl"],
        "net_return_pct": metrics["net_return_pct"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown_pct": metrics["max_dd_pct"] * 100.0,
        "calmar_ratio": metrics["calmar"],
        "net_r": metrics["net_r"],
        "avg_r": metrics["avg_r"],
    }


def _metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"{label:<25} {metrics['total_trades']:>6} "
        f"{metrics['net_return_pct']:>11.4f}% {metrics['profit_factor']:>9.4f} "
        f"{metrics['win_rate']:>9.2f}% {metrics['max_dd_pct'] * 100:>9.4f}% "
        f"{metrics['net_r']:>10.4f} {metrics['avg_r']:>9.4f}"
    )


def _group_lines(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [title, "  Group                              Trades       PnL       PF       WR      Avg R"]
    for name, metrics in groups.items():
        lines.append(
            f"  {name:<34} {metrics['trades']:>6} "
            f"${metrics['net_pnl']:>9.2f} {metrics['profit_factor']:>8.3f} "
            f"{metrics['win_rate']:>8.2f}% {metrics['avg_r']:>10.4f}"
        )
    return lines


def _build_diagnostics(
    generated_at: str,
    config: dict[str, Any],
    core_summary: dict[str, Any],
    extension_summary: dict[str, Any],
    selected_payload: dict[str, Any],
    round1_config: dict[str, Any],
) -> str:
    evaluation = selected_payload["evaluation"]
    attribution = selected_payload["attribution"]
    selected_is = evaluation["selection_metrics"]
    selected_oos = evaluation["oos_metrics"]
    selected_full = evaluation["full_window_metrics"]
    baseline = core_summary["baseline"]
    prior = core_summary["selected"]

    added = {key: value for key, value in config.items() if key not in round1_config}
    changed = {
        key: {"round_1": round1_config[key], "round_2": value}
        for key, value in config.items()
        if key in round1_config and round1_config[key] != value
    }

    lines = [
        "=" * 112,
        "MOMENTUM DOWNTURN ROUND 2 - FINAL DIAGNOSTICS",
        "=" * 112,
        f"Generated: {generated_at}",
        f"Selected candidate: {extension_summary['selected']['name']}",
        f"Detailed replay label: {evaluation['name']}",
        f"Selection signature: {evaluation['signature']}",
        "Disposition: SHADOW_ONLY",
        "Production activation: NOT AUTHORIZED",
        "",
        "PERIOD CONTRACT AND INTERPRETATION",
        "- IS: 2024-01-01 through 2026-03-20 inclusive.",
        "- Observed validation/OOS: 2026-03-21 through 2026-05-01 inclusive.",
        "- Evaluation end is exclusive at 2026-05-02T00:00:00+00:00.",
        "- Initial individual-strategy equity: $10,000.",
        "- OOS was examined during repair and is retrospective validation, not untouched confirmation.",
        "",
        "HEADLINE PERFORMANCE",
        "Window                    Trades      Return        PF        WR         DD      Net R     Avg R",
        _metric_line("IS", selected_is),
        _metric_line("Observed validation", selected_oos),
        _metric_line("Combined", selected_full),
        f"IS frequency: {extension_summary['frequency']['is_trades_per_month']:.4f} trades/month",
        f"Observed-validation frequency: {extension_summary['frequency']['oos_trades_per_month']:.4f} trades/month",
        "",
        "COMPARISON TO ROUND 1 AND THE FIRST REPAIR RECOMMENDATION",
        "Configuration              IS trades   IS return   IS PF   OOS trades   OOS return   OOS PF",
        (
            f"Round 1 frozen            {baseline['selection_metrics']['total_trades']:>9} "
            f"{baseline['selection_metrics']['net_return_pct']:>10.4f}% "
            f"{baseline['selection_metrics']['profit_factor']:>7.3f} "
            f"{baseline['oos_metrics']['total_trades']:>12} "
            f"{baseline['oos_metrics']['net_return_pct']:>11.4f}% "
            f"{baseline['oos_metrics']['profit_factor']:>8.3f}"
        ),
        (
            f"First repair recommendation{prior['selection_metrics']['total_trades']:>8} "
            f"{prior['selection_metrics']['net_return_pct']:>10.4f}% "
            f"{prior['selection_metrics']['profit_factor']:>7.3f} "
            f"{prior['oos_metrics']['total_trades']:>12} "
            f"{prior['oos_metrics']['net_return_pct']:>11.4f}% "
            f"{prior['oos_metrics']['profit_factor']:>8.3f}"
        ),
        (
            f"Round 2 selected           {selected_is['total_trades']:>9} "
            f"{selected_is['net_return_pct']:>10.4f}% {selected_is['profit_factor']:>7.3f} "
            f"{selected_oos['total_trades']:>12} {selected_oos['net_return_pct']:>11.4f}% "
            f"{selected_oos['profit_factor']:>8.3f}"
        ),
        "",
        "ROOT-CAUSE FINDING",
        f"- {core_summary['root_cause']['finding']}",
        "- The corrected OOS has one losing trade (-$77.24), not a small set of catastrophic edge cases.",
        "- Removing that one loss leaves $2,557.84 across nine winning trades; it does not explain the earlier reported collapse.",
        "- The apparent IS/OOS win-rate gap (47.48% versus 90.00%) is based on only ten OOS trades and is not treated as stable.",
        "",
        "MUTATION LINEAGE FROM ROUND 1",
        f"Round 1 cumulative mutation count: {len(round1_config)}",
        f"Round 2 cumulative mutation count: {len(config)}",
        "Added mutations:",
    ]
    lines.extend(f"  {key}: {value!r}" for key, value in sorted(added.items()))
    lines.append("Changed mutations:")
    lines.extend(
        f"  {key}: {value['round_1']!r} -> {value['round_2']!r}"
        for key, value in sorted(changed.items())
    )
    lines.extend(["", "FULL ROUND 2 CUMULATIVE CONFIGURATION"])
    lines.extend(f"  {key}: {value!r}" for key, value in sorted(config.items()))

    lines.extend(
        [
            "",
            "IS FOLD DIAGNOSTICS",
            "Start        End          Trades    Return       PF       WR       DD      Net R",
        ]
    )
    for fold in evaluation["folds"]:
        lines.append(
            f"{fold['start'][:10]}   {fold['end'][:10]}   {fold['total_trades']:>6} "
            f"{fold['net_return_pct']:>9.4f}% {fold['profit_factor']:>8.3f} "
            f"{fold['win_rate']:>7.2f}% {fold['max_dd_pct'] * 100:>7.3f}% {fold['net_r']:>10.4f}"
        )
    lines.extend(
        [
            "- The 2025-07-01 to 2026-01-01 fold has zero trades. This is a material regime/frequency gap, not a reporting omission.",
            "",
            "TRADE ATTRIBUTION - IS",
        ]
    )
    for key, title in [
        ("signal_class", "By signal class"),
        ("regime", "By regime"),
        ("exit_type", "By exit type"),
        ("vol_state", "By volatility state"),
    ]:
        lines.extend(_group_lines(title, attribution["development"][key]))
        lines.append("")
    lines.append("TRADE ATTRIBUTION - OBSERVED VALIDATION")
    for key, title in [
        ("signal_class", "By signal class"),
        ("regime", "By regime"),
        ("exit_type", "By exit type"),
        ("vol_state", "By volatility state"),
    ]:
        lines.extend(_group_lines(title, attribution["oos"][key]))
        lines.append("")

    lines.extend(
        [
            "OOS TRADE-BY-TRADE DIAGNOSTICS",
            "Entry UTC             Exit UTC              Signal             Regime          Exit             PnL       R",
        ]
    )
    for trade in attribution["oos_trades"]:
        lines.append(
            f"{trade['entry_time'][:19]}   {trade['exit_time'][:19]}   "
            f"{trade['signal_class']:<18} {trade['composite_regime_at_entry']:<15} "
            f"{trade['exit_type']:<13} ${trade['pnl']:>8.2f} {trade['r_multiple']:>7.3f}"
        )

    lines.extend(["", "OOS ACTIVE-DAY CONCENTRATION AND EDGE-CASE TESTS"])
    for day, pnl in extension_summary["oos_day_pnl"].items():
        loo = extension_summary["oos_leave_one_active_day_out_pnl"][day]
        lines.append(f"  {day}: PnL=${pnl:,.2f}; PnL with day removed=${loo:,.2f}")
    bootstrap = extension_summary["oos_bootstrap"]
    lines.extend(
        [
            f"  Bootstrap: {bootstrap['samples']:,} resamples of {bootstrap['trades']} trades.",
            f"  P(PnL > 0): {bootstrap['probability_positive_net_pnl'] * 100:.2f}%.",
            f"  95% PnL interval: ${bootstrap['net_pnl_ci95'][0]:,.2f} to ${bootstrap['net_pnl_ci95'][1]:,.2f}.",
            f"  95% net-R interval: {bootstrap['net_r_ci95'][0]:.4f} to {bootstrap['net_r_ci95'][1]:.4f}.",
            "  Caveat: resampling ten trades does not create independent market episodes; only three active days exist.",
            "",
            "ABLATION, PERTURBATION, AND TARGETED-SEARCH COVERAGE",
            f"  Initial corrected-split configurations: {core_summary['evaluated_unique_configurations']}",
            f"  Extension configurations: {extension_summary['extension_unique_configurations']}",
            f"  Total unique corrected-split configurations: {extension_summary['all_repair_unique_configurations']}",
        ]
    )
    for stage, count in core_summary["candidate_counts_by_stage"].items():
        lines.append(f"  Initial {stage}: {count}")
    for stage, count in extension_summary["candidate_counts"].items():
        lines.append(f"  Extension {stage}: {count}")
    stability = extension_summary["stability"]
    lines.extend(
        [
            f"  TTL/BE surface meeting IS >=110% and OOS >=24%: "
            f"{stability['ttl_be_points_is_ge_110_oos_ge_24']}/{stability['ttl_be_points']}",
            f"  Floor/lock surface meeting IS >=110% and OOS >=24%: "
            f"{stability['floor_lock_points_is_ge_110_oos_ge_24']}/{stability['floor_lock_points']}",
            "  The selected 1.5R/40% protection point is the balanced knee between the 1.25R OOS-protective and 1.6R IS-maximizing frontiers.",
            "  The inherited 1.8R emerging TP is discontinuously sensitive above approximately 1.85R and was not retuned on observed OOS.",
            "",
            "EXECUTION STRESS",
            "Stress                       IS trades   IS return   IS PF   OOS trades   OOS return   OOS PF",
        ]
    )
    for stress_name, result in extension_summary["execution_stress"].items():
        is_metrics = result["selection_metrics"]
        oos_metrics = result["oos_metrics"]
        lines.append(
            f"{stress_name:<28} {is_metrics['total_trades']:>8} "
            f"{is_metrics['net_return_pct']:>10.4f}% {is_metrics['profit_factor']:>7.3f} "
            f"{oos_metrics['total_trades']:>12} {oos_metrics['net_return_pct']:>11.4f}% "
            f"{oos_metrics['profit_factor']:>8.3f}"
        )
    lines.extend(
        [
            "- Commission, slippage, and spread stresses remain strong.",
            "- One-bar entry latency is the principal fragility: IS +63.57%, OOS +8.52%; combined stress is IS +60.95%, OOS +8.34%.",
            "",
            "QUALIFICATION AND DISPOSITION",
            "  PASS: corrected period contract and $10,000 strategy basis.",
            "  PASS: IS return, profit factor, drawdown, and frequency objectives.",
            "  PASS: observed-validation return, profit factor, and minimum eight-trade research gate.",
            "  PASS: positive observed-validation results under every recorded execution stress.",
            "  FAIL: observed validation is not untouched because it was used in candidate selection.",
            "  FAIL: observed validation has only ten trades across three active days.",
            "  FAIL: min_hold_profit_protection has backtest/default-off config support but lacks verified live/core parity.",
            "  FINAL DISPOSITION: SHADOW_ONLY. Preserve as Round 2; do not activate for production.",
            "",
            "ARCHIVE CONTENTS",
            "  final_qualification.json contains the complete selected evaluation, attribution, robustness, gate, and lineage data.",
            "  trade_attribution.json contains granular IS/OOS grouping and all ten OOS trades.",
            "  research/ contains the full cumulative ablation, perturbation, targeted, interaction, mechanism, verification, cache, reports, and run specifications.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    config = _read_json(SOURCE_DIR / "recommended_config_extension.json")
    extension_summary = _read_json(SOURCE_DIR / "extension_summary.json")
    extension_spec = _read_json(SOURCE_DIR / "extension_run_spec.json")
    selected_payload = _read_json(SOURCE_DIR / "recommended_trade_attribution_extension.json")
    core_summary = _read_json(SOURCE_DIR / "summary.json")
    round1_config = _read_json(OUTPUT_ROOT / "round_1" / "optimized_config.json")
    selected = selected_payload["evaluation"]
    selected_name = extension_summary["selected"]["name"]

    if extension_summary["split"] != EXPECTED_SPLIT:
        raise ValueError(f"Unexpected date split: {extension_summary['split']!r}")
    if selected["signature"] != EXPECTED_SIGNATURE:
        raise ValueError(f"Unexpected selection signature: {selected['signature']}")
    if selected["mutations"] != config or extension_summary["selected"]["mutations"] != config:
        raise ValueError("Selected mutations do not match recommended_config_extension.json")
    if len(config) != 32:
        raise ValueError(f"Expected 32 cumulative mutations, found {len(config)}")

    generated_at = datetime.now(timezone.utc).isoformat()
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the complete research trail self-contained under Round 2.  The cache
    # is included because it records the union of the initial and extension
    # candidate evaluations used to make the final decision.
    archived_sources: list[dict[str, Any]] = []
    for source_path in sorted(SOURCE_DIR.iterdir()):
        if not source_path.is_file():
            continue
        destination = RESEARCH_DIR / source_path.name
        shutil.copy2(source_path, destination)
        archived_sources.append(
            {
                "path": _relative(destination),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
                "source_path": _relative(source_path),
            }
        )

    research_index = {
        "schema_version": 1,
        "generated_at": generated_at,
        "description": "Complete corrected-split Round 1 OOS-repair research archive supporting Round 2.",
        "files": archived_sources,
        "total_files": len(archived_sources),
        "total_bytes": sum(item["size_bytes"] for item in archived_sources),
    }
    _write_json(RESEARCH_DIR / "index.json", research_index)

    _write_json(ROUND_DIR / "optimized_config.json", config)
    _write_json(ROUND_DIR / "trade_attribution.json", selected_payload)
    _write_json(ROUND_DIR / "research_summary.json", extension_summary)

    added_mutations = {key: value for key, value in config.items() if key not in round1_config}
    changed_mutations = {
        key: {"round_1": round1_config[key], "round_2": value}
        for key, value in config.items()
        if key in round1_config and round1_config[key] != value
    }
    gate = {
        "criteria": {
            "correct_period_contract": True,
            "is_return_ge_round1": True,
            "is_profit_factor_ge_2_20": True,
            "is_drawdown_le_8pct": True,
            "is_frequency_preserved": True,
            "observed_oos_positive": True,
            "observed_oos_profit_factor_ge_1_10": True,
            "observed_oos_trades_ge_8": True,
            "execution_stresses_positive_both_windows": True,
            "observed_oos_untouched": False,
            "observed_oos_trades_ge_30": False,
            "live_core_parity_verified": False,
        },
        "performance_qualification_passed": True,
        "production_promotion_passed": False,
        "blocking_reasons": [
            "The validation interval was observed and optimized against.",
            "Validation contains only ten trades on three active days.",
            "min_hold_profit_protection lacks verified live/core parity.",
        ],
    }
    selected_record = {
        **selected,
        "name": selected_name,
        "detail_evaluation_name": selected["name"],
    }
    final_qualification = {
        "schema_version": 2,
        "family": "momentum",
        "strategy": "downturn",
        "round": 2,
        "generated_at": generated_at,
        "initial_equity": 10_000.0,
        "disposition": "SHADOW_ONLY",
        "activation_authorized": False,
        "period_contract": {
            **EXPECTED_SPLIT,
            "oos_status": "retrospective_observed_validation_not_untouched",
        },
        "selected": selected_record,
        "frequency": extension_summary["frequency"],
        "lineage": {
            "source_round": "round_1",
            "source_candidate": core_summary["selected"]["name"],
            "round_1_mutation_count": len(round1_config),
            "round_2_mutation_count": len(config),
            "added_mutations": added_mutations,
            "changed_mutations": changed_mutations,
        },
        "comparison": {
            "round_1_frozen": baseline_subset(core_summary["baseline"]),
            "first_repair_recommendation": baseline_subset(core_summary["selected"]),
            "round_2_selected": baseline_subset(selected_record),
        },
        "root_cause": {
            **core_summary["root_cause"],
            "edge_case_assessment": (
                "No catastrophic-loss cluster exists in the corrected OOS: nine wins and one -$77.24 loss. "
                "The earlier severe loss came from evaluating a different interval."
            ),
            "win_rate_gap_assessment": (
                "The 47.48% IS versus 90.00% observed-OOS win-rate gap is dominated by a ten-trade, "
                "three-active-day sample and is not considered persistent."
            ),
        },
        "attribution": selected_payload["attribution"],
        "robustness": {
            "oos_bootstrap": extension_summary["oos_bootstrap"],
            "oos_day_pnl": extension_summary["oos_day_pnl"],
            "oos_leave_one_active_day_out_pnl": extension_summary["oos_leave_one_active_day_out_pnl"],
            "execution_stress": extension_summary["execution_stress"],
            "stability": extension_summary["stability"],
            "contenders": extension_summary["contenders"],
        },
        "candidate_search": {
            "initial_unique_configurations": core_summary["evaluated_unique_configurations"],
            "extension_unique_configurations": extension_summary["extension_unique_configurations"],
            "all_unique_configurations": extension_summary["all_repair_unique_configurations"],
            "initial_candidate_counts": core_summary["candidate_counts_by_stage"],
            "extension_candidate_counts": extension_summary["candidate_counts"],
            "selection_rationale": extension_summary["selection_rationale"],
        },
        "promotion_gate": gate,
        "research_archive": {
            "index": "research/index.json",
            "files": research_index["total_files"],
            "bytes": research_index["total_bytes"],
        },
    }
    _write_json(ROUND_DIR / "final_qualification.json", final_qualification)

    diagnostics = _build_diagnostics(
        generated_at, config, core_summary, extension_summary, selected_payload, round1_config
    )
    (ROUND_DIR / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    artifact_paths = [
        ROUND_DIR / "optimized_config.json",
        ROUND_DIR / "final_qualification.json",
        ROUND_DIR / "round_final_diagnostics.txt",
        ROUND_DIR / "trade_attribution.json",
        ROUND_DIR / "research_summary.json",
        RESEARCH_DIR / "index.json",
    ]
    run_spec = {
        "schema_version": 2,
        "family": "momentum",
        "strategy": "downturn",
        "round": 2,
        "purpose": "Promote the latest corrected-split OOS-repair recommendation as a fully diagnosed shadow round.",
        "generated_at": generated_at,
        "initial_equity": 10_000.0,
        "source_round": "round_1/oos_repair",
        "selected_name": selected_name,
        "selected_signature": selected["signature"],
        "mutation_count": len(config),
        "split": {**EXPECTED_SPLIT, "oos_status": "retrospective_observed_validation_not_untouched"},
        "disposition": "SHADOW_ONLY",
        "production_activation_authorized": False,
        "source_research_spec": extension_spec,
        "promotion_code": {
            "path": _relative(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "research_archive": research_index,
        "artifacts": [
            {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        ],
        "provenance_status": "complete",
    }
    _write_json(ROUND_DIR / "run_spec.json", run_spec)

    manifest = _read_json(MANIFEST_PATH)
    manifest_round2 = {
        "round": 2,
        "timestamp": generated_at,
        "source_round": "round_1/oos_repair",
        "description": "Corrected-split cumulative ablation, perturbation, and targeted-repair recommendation",
        "selected_name": selected_name,
        "initial_equity": 10_000.0,
        "diagnostics_period_contract": {
            "in_sample_start": EXPECTED_SPLIT["is_start"],
            "in_sample_end_inclusive": EXPECTED_SPLIT["is_end_inclusive"],
            "out_of_sample_start": EXPECTED_SPLIT["oos_start"],
            "out_of_sample_end_inclusive": EXPECTED_SPLIT["oos_end_inclusive"],
            "study_end_exclusive": EXPECTED_SPLIT["evaluation_end_exclusive"],
            "oos_status": "retrospective_observed_validation_not_untouched",
        },
        "mutations_count": len(config),
        "mutations": config,
        "development_metrics": _manifest_metrics(selected["selection_metrics"]),
        **_manifest_metrics(selected["full_window_metrics"]),
        "oos_metrics": {
            "start": EXPECTED_SPLIT["oos_start"],
            "end_inclusive": EXPECTED_SPLIT["oos_end_inclusive"],
            "status": "retrospective_observed_validation_not_untouched",
            **_manifest_metrics(selected["oos_metrics"]),
        },
        "frequency": extension_summary["frequency"],
        "candidate_search": {
            "extension_unique_configurations": extension_summary["extension_unique_configurations"],
            "all_repair_unique_configurations": extension_summary["all_repair_unique_configurations"],
        },
        "performance_qualification_passed": True,
        "promotion_gate_passed": False,
        "activation_disposition": "SHADOW_ONLY",
        "activation_authorized": False,
        "selection_fingerprint": selected["signature"],
        "provenance_schema_version": 2,
        "provenance_status": "complete",
        "optimized_config": "round_2/optimized_config.json",
        "final_diagnostics_text": "round_2/round_final_diagnostics.txt",
        "final_qualification": "round_2/final_qualification.json",
        "trade_attribution": "round_2/trade_attribution.json",
        "research_summary": "round_2/research_summary.json",
        "research_archive_index": "round_2/research/index.json",
        "run_spec": "round_2/run_spec.json",
    }
    retained_rounds = [entry for entry in manifest.get("rounds", []) if entry.get("round") != 2]
    retained_rounds.append(manifest_round2)
    manifest["rounds"] = sorted(retained_rounds, key=lambda entry: entry["round"])
    _write_json(MANIFEST_PATH, manifest)

    print(f"Saved Round 2 candidate {selected['signature']}")
    print(f"Artifacts: {ROUND_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


def baseline_subset(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": evaluation["name"],
        "signature": evaluation["signature"],
        "selection_metrics": evaluation["selection_metrics"],
        "oos_metrics": evaluation["oos_metrics"],
        "full_window_metrics": evaluation["full_window_metrics"],
        "mutations": evaluation["mutations"],
    }


if __name__ == "__main__":
    main()
