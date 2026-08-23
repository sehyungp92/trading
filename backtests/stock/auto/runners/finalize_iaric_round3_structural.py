"""Canonicalize the completed IARIC structural challenger as Round 3.

This is intentionally separate from automatic promotion: it requires an
explicit caveat-acceptance flag and records the failed automatic value gates
in every canonical artifact and in the rounds manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCORE_SPEC = {
    "incremental_total_r": {"weight": 0.30, "scale": 12.0},
    "incremental_trades": {"weight": 0.22, "scale": 40.0},
    "marginal_avg_r": {"weight": 0.12, "scale": 0.20},
    "profit_factor": {"weight": 0.07, "center": 1.0, "scale": 0.50},
    "discrimination_lift": {"weight": 0.05, "scale": 0.15},
    "inverse_drawdown": {"weight": 0.12, "center": 0.10, "scale": 0.03},
    "robust_avg_r": {"weight": 0.12, "scale": 0.15},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--accept-validation-caveats", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(trades: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(trades)
    values = [float(row.get("r", 0.0)) for row in rows]
    gross_profit = sum(max(value, 0.0) for value in values)
    gross_loss = -sum(min(value, 0.0) for value in values)
    return {
        "trades": len(rows),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0.0
            else (99.0 if gross_profit > 0.0 else 0.0)
        ),
        "win_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
        "net_profit": sum(float(row.get("pnl_net", 0.0)) for row in rows),
    }


def _grouped(trades: list[dict[str, Any]], key_fn: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade))].append(trade)
    return {key: _stats(rows) for key, rows in groups.items()}


def _metric_line(name: str, baseline: float, selected: float, *, pct: bool = False) -> str:
    delta = selected - baseline
    if pct:
        return f"  {name:<28} {baseline:>9.3%} -> {selected:>9.3%}  delta={delta:+.3%}"
    return f"  {name:<28} {baseline:>9.4f} -> {selected:>9.4f}  delta={delta:+.4f}"


def _stage_ledger(research: Path) -> list[dict[str, Any]]:
    names = (
        "phase_0a_activation_roots_results.json",
        "phase_0b_primary_interactions_results.json",
        "phase_0c1_activation_rescue_atoms_results.json",
        "phase_0c2_activation_rescue_followups_results.json",
        "phase_0d_conditional_interactions_results.json",
        "phase_0e_full_diagnostics_parents_results.json",
        "phase_1_structural_entry_results.json",
        "phase_2_lean_management_results.json",
        "validated_finalists.json",
    )
    ledger: list[dict[str, Any]] = []
    for name in names:
        path = research / name
        if not path.exists():
            continue
        rows = _load(path)
        if not isinstance(rows, list):
            continue
        best = max(
            rows,
            key=lambda row: (
                float(row.get("escape_score", -99.0)),
                float(row.get("metrics", {}).get("expected_total_r", -1e9)),
            ),
            default={},
        )
        ledger.append({
            "artifact": f"research/structural_challenger/{name}",
            "stage": name.removesuffix("_results.json").removesuffix(".json"),
            "evaluated": len(rows),
            "best_id": best.get("id"),
            "best_escape_score": best.get("escape_score"),
            "best_metrics": best.get("metrics", {}),
        })
    return ledger


def _diagnostics(
    selected: dict[str, Any],
    control: dict[str, Any],
    monthly: dict[str, dict[str, Any]],
    symbols: dict[str, dict[str, Any]],
    exits: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    ledger: list[dict[str, Any]],
    finalized_at: str,
) -> str:
    sm = selected["metrics"]
    cm = control["metrics"]
    gates = selected["gates"]
    delta_trades = int(sm["total_trades"] - cm["total_trades"])
    delta_r = float(sm["expected_total_r"] - cm["expected_total_r"])
    failed = [name for name, passed in gates.items() if not passed]
    lines = [
        "IARIC ROUND 3 — STRUCTURAL LOCAL-MAXIMUM ESCAPE FINAL DIAGNOSTICS",
        "=" * 84,
        "Status: ACTIVE ROUND 3 — EXPLICIT USER SELECTION WITH VALIDATION CAVEATS",
        f"Finalized: {finalized_at}",
        f"Selected: {selected['id']}",
        "Training authority: 2024-03-25 through 2026-03-01",
        "Sealed holdout: begins 2026-03-02; accessed=false",
        "Selection authority: explicit user direction after completed structural research",
        "Automatic promotion verdict: FAIL (recorded, not concealed)",
        "",
        "EXECUTIVE OUTCOME",
        f"  Trades                         {cm['total_trades']:.0f} -> {sm['total_trades']:.0f}  delta={delta_trades:+d} ({delta_trades / cm['total_trades']:+.2%})",
        f"  Expected total R              {cm['expected_total_r']:+.3f} -> {sm['expected_total_r']:+.3f}  delta={delta_r:+.3f}",
        _metric_line("Average R", float(cm["avg_r"]), float(sm["avg_r"])),
        _metric_line("Profit factor", float(cm["profit_factor"]), float(sm["profit_factor"])),
        _metric_line("Maximum drawdown", float(cm["max_drawdown_pct"]), float(sm["max_drawdown_pct"]), pct=True),
        _metric_line("Sharpe", float(cm["sharpe"]), float(sm["sharpe"])),
        _metric_line("CAGR", float(cm["cagr"]), float(sm["cagr"]), pct=True),
        _metric_line("Calmar", float(cm["calmar"]), float(sm["calmar"])),
        "",
        "DECISION",
        "  The candidate created economically positive incremental alpha and a broader",
        "  reversion aperture, so it is retained as the next-round optimization anchor.",
        "  It is not represented as fully validated: three automatic gates failed and",
        "  subsequent optimization must directly repair fold consistency, concentration,",
        "  and the remaining four-trade frequency shortfall without surrendering total R.",
        "",
        "AUTOMATIC VALUE / REAL-ALPHA GATES",
    ]
    observed = {
        "frequency_uplift_15pct": f"149 trades; threshold=153; shortfall=4",
        "satellite_not_single_symbol": "max positive symbol share=36.448%; threshold<=35.000%",
        "chronological_consistency": "positive delta-R folds=1/3; threshold>=2/3",
    }
    for name, passed in gates.items():
        suffix = f" — {observed[name]}" if name in observed else ""
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}{suffix}")
    lines += [
        f"  Failed gates: {', '.join(failed)}",
        "",
        "CHRONOLOGICAL FOLD VALIDATION",
        "  Fold       Dates                       delta trades   delta R   aperture trades/R/PF",
    ]
    for fold in selected.get("folds", []):
        aperture = fold["aperture"]
        lines.append(
            f"  {fold['fold']:<10} {fold['start_date']}..{fold['end_date']}"
            f" {fold['delta_trades']:+12.0f} {fold['delta_total_r']:+9.3f}"
            f"   {aperture['trades']:>3} / {aperture['total_r']:+7.3f} / {aperture['profit_factor']:.3f}"
        )
    lines += [
        "",
        "FOCUS-FAMILY MARGINAL ALPHA",
    ]
    for family, stats in selected.get("focus_family_marginal_alpha", {}).items():
        lines.append(
            f"  {family}: trades={stats['trades']:.0f}, totalR={stats['total_r']:+.3f}, PF={stats['profit_factor']:.3f}"
        )
    lines += ["", "ROUTE ATTRIBUTION"]
    for route, stats in sorted(routes.items(), key=lambda item: item[1]["total_r"], reverse=True):
        lines.append(
            f"  {route:<50} n={stats['trades']:>3} totalR={stats['total_r']:+8.3f} avgR={stats['avg_r']:+.4f} PF={stats['profit_factor']:.3f}"
        )
    lines += [
        "",
        "SIGNAL EXTRACTION / DISCRIMINATION",
        _metric_line("Opportunity recall", float(cm["entry_opportunity_recall"]), float(sm["entry_opportunity_recall"])),
        _metric_line("Entry potential total R", float(cm["entry_potential_total_r"]), float(sm["entry_potential_total_r"])),
        _metric_line("Oracle potential R", float(cm["entry_oracle_potential_r"]), float(sm["entry_oracle_potential_r"])),
        _metric_line("Realized discrimination", float(cm["entry_realized_discrimination_lift_r"]), float(sm["entry_realized_discrimination_lift_r"])),
        _metric_line("Rejected potential avg R", float(cm["entry_rejected_potential_avg_r"]), float(sm["entry_rejected_potential_avg_r"])),
        _metric_line("Robust average R", float(cm["robust_avg_r"]), float(sm["robust_avg_r"])),
        _metric_line("Candidates per entered day", float(cm["mean_n_candidates"]), float(sm["mean_n_candidates"])),
        "  Interpretation: recall and potential alpha increased, but realized discrimination",
        "  softened slightly. The route-specific floor/transition recovered genuine alpha;",
        "  the broader opportunity set still contains material rejected and low-quality supply.",
        "",
        "TRADE MANAGEMENT / EXIT PROFILE",
        _metric_line("Average hold hours", float(cm["avg_hold_hours"]), float(sm["avg_hold_hours"])),
        _metric_line("Stop-hit share", float(cm["stop_hit_share"]), float(sm["stop_hit_share"]), pct=True),
        _metric_line("Stop-hit average R", float(cm["stop_hit_avg_r"]), float(sm["stop_hit_avg_r"])),
        _metric_line("Stop-hit total R", float(cm["stop_hit_total_r"]), float(sm["stop_hit_total_r"])),
        _metric_line("Carry trade share", float(cm["carry_trade_share"]), float(sm["carry_trade_share"]), pct=True),
        _metric_line("Carry average R", float(cm["carry_avg_r"]), float(sm["carry_avg_r"])),
        _metric_line("EOD flatten share", float(cm["eod_flatten_share"]), float(sm["eod_flatten_share"]), pct=True),
        "",
        "EXIT-REASON ATTRIBUTION",
    ]
    for reason, stats in sorted(exits.items(), key=lambda item: item[1]["trades"], reverse=True):
        lines.append(
            f"  {reason:<24} n={stats['trades']:>3} totalR={stats['total_r']:+8.3f} avgR={stats['avg_r']:+.4f} PF={stats['profit_factor']:.3f}"
        )
    lines += ["", "FUNNEL COUNTERS"]
    for key, value in sorted(selected.get("funnel_counters", {}).items()):
        lines.append(f"  {key:<34} {value}")
    lines += ["", "MONTHLY PERFORMANCE"]
    for month, stats in sorted(monthly.items()):
        lines.append(
            f"  {month}: n={stats['trades']:>3} totalR={stats['total_r']:+8.3f} avgR={stats['avg_r']:+.4f} PF={stats['profit_factor']:.3f} win={stats['win_rate']:.1%}"
        )
    lines += ["", "SYMBOL ATTRIBUTION (SORTED BY TOTAL R)"]
    total_positive = sum(max(float(stats["total_r"]), 0.0) for stats in symbols.values())
    for symbol, stats in sorted(symbols.items(), key=lambda item: item[1]["total_r"], reverse=True):
        share = max(float(stats["total_r"]), 0.0) / total_positive if total_positive else 0.0
        lines.append(
            f"  {symbol:<8} n={stats['trades']:>3} totalR={stats['total_r']:+8.3f} avgR={stats['avg_r']:+.4f} PF={stats['profit_factor']:.3f} positive-share={share:.2%}"
        )
    lines += ["", "EXPERIMENT LEDGER"]
    for stage in ledger:
        metrics = stage["best_metrics"]
        lines.append(
            f"  {stage['stage']}: evaluated={stage['evaluated']}, best={stage['best_id']}, "
            f"trades={metrics.get('total_trades', 0):.0f}, totalR={metrics.get('expected_total_r', 0):+.3f}, "
            f"PF={metrics.get('profit_factor', 0):.3f}, DD={metrics.get('max_drawdown_pct', 0):.3%}"
        )
    lines += [
        "",
        "IMMUTABLE OPTIMIZATION SCORE — EXACTLY 7 COMPONENTS",
    ]
    for name, spec in SCORE_SPEC.items():
        center = f", center={spec['center']:.4g}" if "center" in spec else ""
        lines.append(
            f"  {name}: weight={spec['weight']:.2f}, scale={spec['scale']:.4g}{center}"
        )
    lines += [
        "",
        "STRENGTHS",
        f"  - Created {delta_r:+.3f} expected R and {delta_trades:+d} trades versus the 133-trade anchor.",
        "  - Improved portfolio PF, Sharpe, CAGR and Calmar with only a 12.2 bp DD increase.",
        "  - MULTIDAY_HIGHER_LOW_RECLAIM produced positive attributed alpha rather than being",
        "    carried by an unrelated strong route.",
        "  - All three chronological folds were executed under an immutable date contract.",
        "  - Shared-core family floor, transition and sizing settings preserve live/backtest parity.",
        "",
        "WEAKNESSES / CAVEATS",
        "  - Frequency increased 12.0%, not the pre-registered 15% requirement.",
        "  - Early and middle fold incremental R were negative; most improvement arrived latest.",
        "  - The positive satellite contribution is slightly too concentrated in one symbol.",
        "  - Portfolio average R and realized discrimination declined modestly despite higher total R.",
        "  - The 149 trades remain a relatively narrow sample for a broad stock-reversion thesis.",
        "",
        "ROUND 4 STARTING REQUIREMENTS",
        "  - Use this configuration as the starting anchor; do not return to the 88/89-trade basin.",
        "  - Target orthogonal route activation and cross-fold quality, not more tuning around the",
        "    same MULTIDAY_HIGHER_LOW_RECLAIM parent.",
        "  - Require incremental routes to be positive by attributed family and to reduce symbol",
        "    concentration; preserve or improve +47.029 expected R and approximately 3.04% DD.",
        "  - Keep the sealed holdout untouched until a later explicit validation decision.",
        "",
        "SELECTED MUTATIONS",
        json.dumps(selected["mutations"], indent=2, sort_keys=True),
        "",
        "ARTIFACT PROVENANCE",
        "  Full research ledger: research/structural_challenger/",
        "  Canonical selection: final_selection.json",
        "  Canonical trades: final_trades.json",
        "  Aggregates: final_monthly.json, final_symbols.json, final_exits.json, final_routes.json",
        "  This diagnostic deliberately preserves the failed automatic gates while recording",
        "  the explicit decision to use the candidate as the active next-round anchor.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if not args.accept_validation_caveats:
        raise ValueError("Explicit --accept-validation-caveats is required")
    if len(SCORE_SPEC) != 7 or abs(sum(row["weight"] for row in SCORE_SPEC.values()) - 1.0) > 1e-12:
        raise RuntimeError("Immutable score must contain exactly seven components summing to 1")
    round_dir = Path(args.round_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    research = round_dir / "research/structural_challenger"
    source = _load(research / "final_selection.json")
    selected = source["selected"]
    control = source["control"]
    if bool(selected.get("all_gates_pass")):
        raise RuntimeError("This finalizer is only for an explicitly accepted caveated selection")
    if selected.get("validation_contract", {}).get("passed") is not True:
        raise RuntimeError("Chronological-fold integrity did not pass")
    if selected.get("validation_contract", {}).get("holdout_accessed") is not False:
        raise RuntimeError("Sealed holdout contract is not intact")
    trades = list(selected.get("trade_attribution", []))
    if len(trades) != int(selected["metrics"]["total_trades"]):
        raise RuntimeError("Selected trade ledger does not match total_trades")
    finalized_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    monthly = _grouped(trades, lambda row: str(row.get("entry_time", ""))[:7])
    symbols = _grouped(trades, lambda row: row.get("symbol", "UNKNOWN"))
    exits = _grouped(trades, lambda row: row.get("exit_reason", "UNKNOWN"))
    routes = _grouped(trades, lambda row: row.get("route", "UNKNOWN"))
    ledger = _stage_ledger(research)
    canonical_selection = dict(source)
    canonical_selection.update({
        "status": "complete_user_selected_with_validation_caveats",
        "official": True,
        "selection_authority": "explicit_user_direction_2026-08-21",
        "automatic_value_verification_passed": False,
        "finalized_at_utc": finalized_at,
    })
    _write(round_dir / "optimized_config.json", selected["mutations"])
    _write(round_dir / "final_selection.json", canonical_selection)
    _write(round_dir / "final_metrics.json", selected["metrics"])
    _write(round_dir / "final_trades.json", trades)
    _write(round_dir / "final_monthly.json", monthly)
    _write(round_dir / "final_symbols.json", symbols)
    _write(round_dir / "final_exits.json", exits)
    _write(round_dir / "final_routes.json", routes)
    _write(round_dir / "experiment_ledger.json", ledger)
    _write(round_dir / "run_spec.json", {
        "objective": "escape the narrow IARIC local maximum and create a broader reversion optimization anchor",
        "training_window": {"start": "2024-03-25", "end": "2026-03-01"},
        "sealed_holdout": {"start": "2026-03-02", "accessed": False},
        "max_workers": 2,
        "immutable_score": SCORE_SPEC,
        "score_component_count": 7,
        "source_research": "research/structural_challenger",
        "selected_id": selected["id"],
        "selection_authority": "explicit_user_direction_2026-08-21",
        "automatic_value_verification_passed": False,
        "live_backtest_parity": {
            "shared_core_family_score_floor": True,
            "shared_core_family_transition": True,
            "shared_core_aperture_sizing": True,
            "completed_bar_causality": True,
        },
    })
    diagnostics = _diagnostics(
        selected, control, monthly, symbols, exits, routes, ledger, finalized_at
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    config_hash = _sha256(round_dir / "optimized_config.json")
    diagnostics_hash = _sha256(round_dir / "round_final_diagnostics.txt")
    selection_hash = _sha256(round_dir / "final_selection.json")
    failed_gates = [name for name, passed in selected["gates"].items() if not passed]
    summary = {
        "status": "complete_user_selected_with_validation_caveats",
        "official": True,
        "active_round": 3,
        "selected_id": selected["id"],
        "selection_authority": "explicit_user_direction_2026-08-21",
        "automatic_value_verification_passed": False,
        "failed_automatic_gates": failed_gates,
        "metrics": selected["metrics"],
        "baseline_metrics": control["metrics"],
        "aperture": selected["aperture"],
        "focus_family_marginal_alpha": selected.get("focus_family_marginal_alpha", {}),
        "gates": selected["gates"],
        "validation_contract": selected["validation_contract"],
        "holdout_accessed": False,
        "hashes": {
            "optimized_config_sha256": config_hash,
            "final_selection_sha256": selection_hash,
            "round_final_diagnostics_sha256": diagnostics_hash,
        },
        "artifacts": {
            "optimized_config": "round_3/optimized_config.json",
            "final_selection": "round_3/final_selection.json",
            "full_final_diagnostics": "round_3/round_final_diagnostics.txt",
            "final_metrics": "round_3/final_metrics.json",
            "final_trades": "round_3/final_trades.json",
            "final_monthly": "round_3/final_monthly.json",
            "final_symbols": "round_3/final_symbols.json",
            "final_exits": "round_3/final_exits.json",
            "final_routes": "round_3/final_routes.json",
            "experiment_ledger": "round_3/experiment_ledger.json",
            "research": "round_3/research/structural_challenger/",
        },
        "finalized_at_utc": finalized_at,
    }
    _write(round_dir / "run_summary.json", summary)
    manifest = _load(manifest_path)
    manifest["active_round"] = 3
    manifest.pop("pending_round_3", None)
    manifest["generated_at_utc"] = finalized_at
    manifest["rounds"] = [
        row for row in manifest.get("rounds", []) if int(row.get("round", -1)) != 3
    ]
    manifest["rounds"].append({
        "round": 3,
        "status": summary["status"],
        "official": True,
        "active": True,
        "configuration_role": "structural_local_maximum_escape_anchor_for_further_optimization",
        "selection_authority": summary["selection_authority"],
        "automatic_value_verification_passed": False,
        "failed_automatic_gates": failed_gates,
        "validation_status": "chronological_folds_complete_user_accepted_caveats",
        "validation_contract": selected["validation_contract"],
        "training_window": {"start": "2024-03-25", "end": "2026-03-01"},
        "sealed_holdout": {"start": "2026-03-02", "accessed": False},
        "score_component_count": 7,
        "total_trades": int(selected["metrics"]["total_trades"]),
        "expected_total_r": selected["metrics"]["expected_total_r"],
        "avg_r": selected["metrics"]["avg_r"],
        "profit_factor": selected["metrics"]["profit_factor"],
        "max_drawdown_pct": 100.0 * selected["metrics"]["max_drawdown_pct"],
        "sharpe_ratio": selected["metrics"]["sharpe"],
        "mutations_count": len(selected["mutations"]),
        "mutations": selected["mutations"],
        "config_sha256": config_hash,
        "diagnostics_sha256": diagnostics_hash,
        "selection_sha256": selection_hash,
        "live_backtest_parity": {
            "shared_core_family_policies": True,
            "causal_completed_bar_transitions": True,
        },
        "artifacts": summary["artifacts"],
        "timestamp": finalized_at,
    })
    _write(manifest_path, manifest)
    print(json.dumps({
        "round_dir": str(round_dir),
        "manifest": str(manifest_path),
        "selected_id": selected["id"],
        "total_trades": selected["metrics"]["total_trades"],
        "failed_automatic_gates": failed_gates,
        "diagnostics_sha256": diagnostics_hash,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
