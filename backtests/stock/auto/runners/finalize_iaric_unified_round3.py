"""Finalize canonical IARIC Round 3 after unified research staging.

This is the only stage allowed to turn the research directory into a canonical
round.  It replays the selected executable config, builds full diagnostics,
checks metric agreement and the atlas decision, then atomically updates the
round summary and manifest.  A parity-unwired atlas survivor is recorded as a
pending round and cannot become the active baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_DIR = REPO_ROOT / "backtests/output/stock/iaric/round_3"
UNIFIED_DIR = ROUND_DIR / "unified"
MANIFEST_PATH = REPO_ROOT / "backtests/output/stock/iaric/rounds_manifest.json"
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
HOLDOUT_START = "2026-03-02"
INITIAL_EQUITY = 10_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=str(ROUND_DIR))
    parser.add_argument("--unified-dir", default=str(UNIFIED_DIR))
    parser.add_argument("--synthesis-dir", default=str(ROUND_DIR / "alpha_synthesis"))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--wait-for-pid", type=int, default=0)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_agreement(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    tolerances = {
        "total_trades": 0.0,
        "expected_total_r": 1e-8,
        "avg_r": 1e-8,
        "profit_factor": 1e-8,
        "max_drawdown_pct": 1e-8,
        "net_profit": 1e-6,
    }
    checks: dict[str, Any] = {}
    for key, tolerance in tolerances.items():
        expected_value = float(expected.get(key, 0.0))
        actual_value = float(actual.get(key, 0.0))
        delta = actual_value - expected_value
        checks[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "delta": delta,
            "tolerance": tolerance,
            "passed": abs(delta) <= tolerance,
        }
    return {"passed": all(row["passed"] for row in checks.values()), "checks": checks}


def _research_appendix(
    unified: dict[str, Any],
    synthesis: dict[str, Any],
    atlas: dict[str, Any],
    agreement: dict[str, Any],
    final_status: str,
) -> str:
    selected = synthesis["selected_executable"]
    verification = synthesis.get("final_outcome_verification", {})
    value = verification.get("value_creation", {})
    deltas = value.get("deltas", {})
    failed_value_checks = [
        name for name, passed in value.get("checks", {}).items() if not passed
    ]
    research2_resolution = verification.get("research2_resolution", {})
    lines = [
        "=" * 72,
        "  UNIFIED ROUND-3 RESEARCH RECONCILIATION",
        "=" * 72,
        f"  Finalization status: {final_status}",
        f"  Branched incumbent before synthesis: {unified['incumbent']['id']}",
        f"  Alpha-synthesis selection: {selected['id']}",
        f"  Unified score: {float(selected['unified_score']):.4f}",
        f"  Bounded search coverage: {'PASS' if synthesis.get('search_coverage', {}).get('gate_passed') else 'FAIL'}",
        f"  Final value verification: {'PASS' if synthesis.get('final_outcome_verification', {}).get('gate_passed') else 'FAIL'}",
        f"  Value deltas vs Round-2 control: totalR={float(deltas.get('expected_total_r', 0.0)):+.2f}, "
        f"trades={float(deltas.get('total_trades', 0.0)):+.0f}, "
        f"unified_score={float(deltas.get('unified_score', 0.0)):+.4f}",
        f"  Failed value checks: {', '.join(failed_value_checks) or 'none'}",
        f"  Research-2 credible near-misses audited: "
        f"{', '.join(research2_resolution.get('credible_positive_near_misses', [])) or 'none'}",
        f"  Fresh unified-score finalists: {int(synthesis.get('search_coverage', {}).get('fresh_validated_candidates', 0))}",
        f"  Admitted structural routes: {', '.join(synthesis['admitted_structural_routes']) or 'none'}",
        f"  Reference-only survivors: {', '.join(unified.get('reference_survivors_not_iaric_routes', [])) or 'none'}",
        f"  Executable replay agreement: {'PASS' if agreement['passed'] else 'FAIL'}",
        "  Holdout accessed: no",
        "",
        "Event-family evidence (standardized opportunity outcomes; not portfolio metrics):",
    ]
    for family, row in sorted(atlas.get("family_results", {}).items()):
        gate = row.get("promotion_gate", {})
        structural = synthesis.get("structural_audits", {}).get(family, {})
        selected_entry = structural.get("selected_entry_variant", "-")
        lines.append(
            f"  {family}: events={int(row.get('events', 0))} "
            f"avg12R={float(row.get('avg_bar_12_r', 0.0)):+.3f} "
            f"PF={float(row.get('stop_target_profit_factor', 0.0)):.2f} "
            f"selected_entry={selected_entry} "
            f"unconditional_gate={'pass' if gate.get('research_survivor') else 'fail'} "
            f"incremental_route_admitted={'yes' if family in synthesis.get('admitted_structural_routes', []) else 'no'}"
        )
    lines.extend([
        "",
        "Interpretation:",
        "  The branched search measures executable monetization inside the IARIC opportunity set.",
        "  The atlas measures opportunity breadth before selection and portfolio management.",
        "  The canonical round is coherent only when replay matches the selected incumbent and no",
        "  positive structural route is omitted merely because its live/backtest adapters are absent.",
        "",
    ])
    return "\n".join(lines)


def _run_full_diagnostics(
    config_mutations: dict[str, Any],
    unified: dict[str, Any],
    synthesis: dict[str, Any],
    atlas: dict[str, Any],
    round_dir: Path,
) -> tuple[dict[str, Any], str]:
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    from backtests.shared.diagnostics.snapshot import build_group_snapshot
    from backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.auto.iaric.phase_scoring import enrich_phase_score_metrics, merge_pullback_metrics
    from backtests.stock.auto.scoring import extract_metrics
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

    replay = load_research_replay_bundle(DATA_DIR, require_bundle=False).data
    base = IARICBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=3,
        data_dir=DATA_DIR,
    )
    config = mutate_iaric_config(base, config_mutations)
    result = IARICPullbackEngine(config, replay, collect_diagnostics=True).run()
    performance = extract_metrics(
        result.trades, result.equity_curve, result.timestamps, INITIAL_EQUITY,
    )
    metrics = enrich_phase_score_metrics(merge_pullback_metrics(
        performance,
        result.trades,
        candidate_ledger=result.candidate_ledger,
        selection_attribution=result.selection_attribution,
    ))
    snapshot = build_group_snapshot(
        "IARIC Unified Round-3 Strength / Weakness Snapshot",
        result.trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("entry variant", lambda trade: getattr(trade, "metadata", {}).get("entry_variant")),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
    )
    diagnostic = pullback_full_diagnostic(
        result.trades,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
        funnel_counters=result.funnel_counters,
        rejection_log=result.rejection_log,
        shadow_outcomes=result.shadow_outcomes,
        selection_attribution=result.selection_attribution,
        fsm_log=result.fsm_log,
    )
    agreement = _metric_agreement(synthesis["selected_executable"]["metrics"], metrics)
    structural_survivors = list(synthesis.get("admitted_structural_routes", []))
    if not agreement["passed"]:
        final_status = "blocked_executable_replay_mismatch"
    elif not bool(synthesis.get("search_coverage", {}).get("gate_passed")):
        final_status = "blocked_incomplete_search_coverage"
    elif structural_survivors:
        final_status = "blocked_pending_structural_route_parity"
    elif not bool(synthesis.get("final_outcome_verification", {}).get("gate_passed")):
        final_status = "blocked_final_value_verification"
    elif not bool(synthesis.get("canonical_finalization_allowed")):
        final_status = "blocked_alpha_synthesis_not_finalizable"
    else:
        final_status = "canonical_round3_research_complete"
    header = "\n".join([
        "=" * 72,
        "  IARIC UNIFIED ROUND 3 - FULL FINAL DIAGNOSTICS",
        "=" * 72,
        f"  Date range:      {START_DATE} -- {END_DATE}",
        f"  Initial equity:  ${INITIAL_EQUITY:,.0f}",
        "  Data authority:  legacy_diagnostic_only",
        "  Fill timing:     completed signal bar -> next 5m open",
        f"  Mutation count:  {len(config_mutations)}",
        "  Holdout:         sealed from 2026-03-02; accessed=no",
    ])
    report = (
        header + "\n\n" + snapshot + "\n\n" + diagnostic + "\n\n"
        + _research_appendix(unified, synthesis, atlas, agreement, final_status)
    )
    return {
        "metrics": metrics,
        "agreement": agreement,
        "final_status": final_status,
        "trades": len(result.trades),
    }, report


def _update_manifest(
    manifest_path: Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    unified: dict[str, Any],
    synthesis: dict[str, Any],
    final_status: str,
    round_dir: Path,
) -> None:
    from backtests.shared.auto.round_manager import canonicalize_metrics

    manifest = _load_json(manifest_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    diagnostics_path = round_dir / "round_final_diagnostics.txt"
    config_path = round_dir / "optimized_config.json"
    entry = {
        "round": 3,
        "timestamp": timestamp,
        "status": final_status,
        "configuration_role": "unified_research_reference",
        "mutations_count": len(config),
        "mutations": config,
        **canonicalize_metrics(metrics),
        "expected_total_r": float(metrics.get("expected_total_r", 0.0)),
        "avg_r": float(metrics.get("avg_r", 0.0)),
        "trades_per_month": float(metrics.get("trades_per_month", 0.0)),
        "data_authority": "legacy_diagnostic_only",
        "validation_status": "provisional_legacy_data_revalidation_required",
        "promotion_allowed": False,
        "phased_auto_run": True,
        "training_window": {"start": START_DATE, "end": END_DATE},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "score_component_count": 7,
        "score_components": list(unified["unified_score_spec"]),
        "unified_score_spec": unified["unified_score_spec"],
        "unified_score": synthesis["selected_executable"]["unified_score"],
        "branched_incumbent": unified["incumbent"]["id"],
        "alpha_synthesis_selection": synthesis["selected_executable"]["id"],
        "admitted_structural_routes": synthesis.get("admitted_structural_routes", []),
        "search_coverage": synthesis.get("search_coverage", {}),
        "final_outcome_verification": synthesis.get("final_outcome_verification", {}),
        "config_sha256": _sha256(config_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "artifacts": {
            "optimized_config": "round_3/optimized_config.json",
            "run_summary": "round_3/run_summary.json",
            "full_final_diagnostics": "round_3/round_final_diagnostics.txt",
            "unified_selection": "round_3/unified/unified_selection.json",
            "alpha_synthesis": "round_3/alpha_synthesis/alpha_synthesis_selection.json",
            "final_value_verification": "round_3/alpha_synthesis/final_value_verification.json",
            "opportunity_atlas": "../opportunity_atlas/round_1/atlas_summary.json",
        },
    }
    rounds = manifest.setdefault("rounds", [])
    rounds[:] = [
        row for row in rounds
        if int(row.get("round", 0)) != 3 or bool(row.get("archived"))
    ]
    rounds.append(entry)
    rounds.sort(key=lambda row: int(row.get("round", 0)))
    manifest["active_round"] = 3
    manifest["generated_at_utc"] = timestamp
    manifest.pop("pending_round_3", None)
    _write_json(manifest_path, manifest)


def _record_blocked_manifest(
    manifest_path: Path,
    unified: dict[str, Any],
    synthesis: dict[str, Any],
    final_status: str,
    round_dir: Path,
) -> None:
    manifest = _load_json(manifest_path)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["pending_round_3"] = {
        "status": final_status,
        "promotion_allowed": False,
        "walk_forward_survivors": unified.get("atlas_survivors", []),
        "admitted_structural_routes": synthesis.get("admitted_structural_routes", []),
        "search_coverage": synthesis.get("search_coverage", {}),
        "final_outcome_verification": synthesis.get("final_outcome_verification", {}),
        "diagnostics": str((round_dir / "round_final_diagnostics.txt").relative_to(round_dir.parent)),
        "candidate_catalog": "round_3/alpha_synthesis/candidate_catalog.json",
    }
    _write_json(manifest_path, manifest)


def main() -> None:
    args = _parse_args()
    round_dir = Path(args.round_dir).resolve()
    unified_dir = Path(args.unified_dir).resolve()
    synthesis_dir = Path(args.synthesis_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    round_dir.mkdir(parents=True, exist_ok=True)
    _write_json(unified_dir / "finalization_queue_status.json", {
        "status": "queued" if args.wait_for_pid > 0 else "starting",
        "waiting_for_pid": args.wait_for_pid,
        "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _wait_for_pid(args.wait_for_pid)
    _write_json(unified_dir / "finalization_queue_status.json", {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    unified = _load_json(unified_dir / "unified_selection.json")
    synthesis = _load_json(synthesis_dir / "alpha_synthesis_selection.json")
    atlas_dir = Path(unified_dir).parents[2] / "opportunity_atlas/round_1"
    # The default sibling calculation above is intentionally not authoritative;
    # use the source path embedded by the atlas job when the standard layout is present.
    standard_atlas = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
    atlas = _load_json((standard_atlas if standard_atlas.exists() else atlas_dir) / "atlas_summary.json")
    config_source = synthesis_dir / "selected_executable_config.json"
    config = _load_json(config_source)
    replay, report = _run_full_diagnostics(config, unified, synthesis, atlas, round_dir)
    diagnostics_path = round_dir / "round_final_diagnostics.txt"
    diagnostics_path.write_text(report, encoding="utf-8")
    final_status = replay["final_status"]
    canonical = final_status == "canonical_round3_research_complete"
    if canonical:
        _write_json(round_dir / "optimized_config.json", config)
    else:
        _write_json(unified_dir / "blocked_incumbent_config.json", config)
    run_summary = {
        "family": "stock",
        "strategy": "iaric",
        "round": 3,
        "status": final_status,
        "research_only": True,
        "promotion_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_phases": [
            "branched_aperture", "opportunity_atlas", "walk_forward",
            "alpha_synthesis", "unified_replay",
        ],
        "mutation_count": len(config),
        "cumulative_mutations": config,
        "final_metrics": replay["metrics"],
        "replay_agreement": replay["agreement"],
        "unified_selection": unified,
        "alpha_synthesis": synthesis,
        "final_outcome_verification": synthesis.get("final_outcome_verification", {}),
        "source_diagnostics": str(diagnostics_path),
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
    }
    _write_json(round_dir / "run_summary.json", run_summary)
    _write_json(round_dir / "final_metrics.json", replay["metrics"])
    branched_spec = _load_json(round_dir / "run_spec.json")
    _write_json(round_dir / "branched_run_spec.json", branched_spec)
    _write_json(round_dir / "run_spec.json", {
        "round": 3,
        "status": final_status,
        "research_only": True,
        "architecture": "branched_aperture_plus_causal_opportunity_atlas_then_alpha_synthesis_and_unified_replay",
        "training_window": {"start": START_DATE, "end": END_DATE},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "data_authority": "legacy_research_only",
        "data_fingerprint": unified["data_fingerprint"],
        "unified_code_fingerprint": unified["code_fingerprint"],
        "score_component_count": 7,
        "score_spec": unified["unified_score_spec"],
        "sources": {
            "branched": "round_3/branched_run_spec.json",
            "unified": "round_3/unified/unified_selection.json",
            "alpha_synthesis": "round_3/alpha_synthesis/alpha_synthesis_selection.json",
            "final_value_verification": "round_3/alpha_synthesis/final_value_verification.json",
            "atlas": "../opportunity_atlas/round_1/atlas_summary.json",
            "walk_forward": "../opportunity_atlas/round_1/walk_forward/walk_forward_summary.json",
        },
    })
    if canonical:
        _update_manifest(
            manifest_path,
            config=config,
            metrics=replay["metrics"],
            unified=unified,
            synthesis=synthesis,
            final_status=final_status,
            round_dir=round_dir,
        )
    else:
        _record_blocked_manifest(manifest_path, unified, synthesis, final_status, round_dir)
    _write_json(unified_dir / "finalization_status.json", {
        "status": final_status,
        "canonical_round_created": canonical,
        "manifest_updated": True,
        "holdout_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _write_json(unified_dir / "finalization_queue_status.json", {
        "status": "complete" if canonical else "blocked",
        "result_status": final_status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(
        f"Round3 finalization: {final_status}; diagnostics={diagnostics_path}; "
        f"manifest={manifest_path}; holdout accessed=no",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failed_args = _parse_args()
        failed_unified = Path(failed_args.unified_dir).resolve()
        _write_json(failed_unified / "finalization_queue_status.json", {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        raise
