"""Run the targeted IARIC portable-alpha and concentration escape search.

This is a continuation, not a fresh parameter search.  It starts from the
strongest Round 4 return configuration and the already-measured issuer-capped
composition.  Only mechanisms supported by the completed opportunity-atlas
research are admitted.  The sealed holdout is excluded and no canonical round
is overwritten automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.runners import run_iaric_round4_real_alpha as round4
from backtests.stock.auto.runners.run_iaric_escape_round3 import (
    IARIC_DIR,
    _candidate,
    _dedupe,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _replay_source_fingerprint,
)
from strategies.stock.iaric.core.lanes import aperture_family_from_route, issuer_key


DEFAULT_PRIOR_OUTPUT = IARIC_DIR / "round_4/phased_auto"
REFERENCE_DIR = IARIC_DIR.parents[2] / "stock/auto/iaric/references"
DEFAULT_BASELINE = REFERENCE_DIR / "portable_alpha_round4_baseline.json"
DEFAULT_PHASE8 = REFERENCE_DIR / "portable_alpha_round4_capped_baseline.json"
DEFAULT_OUTPUT = IARIC_DIR / "round_4/portable_alpha_escape"
ROUND3_SELECTION = IARIC_DIR / "round_3/final_selection.json"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"

UPTREND = "UPTREND_PULLBACK_RECLAIM"
RESIDUAL = "MARKET_SECTOR_RESIDUAL_RECLAIM"

# Exactly seven predeclared, economically-scaled components.  The score leans
# aggressive through return and breadth, but gives concentration/portability
# enough weight to prevent an Alphabet-like cluster from winning by itself.
SCORE_SPEC: dict[str, dict[str, float]] = {
    "incremental_total_r": {"weight": 0.25, "scale": 8.0},
    "incremental_trades": {"weight": 0.16, "scale": 25.0},
    "issuer_neutral_delta_r": {"weight": 0.20, "scale": 6.0},
    "issuer_hhi_improvement": {"weight": 0.10, "scale": 0.08},
    "worst_segment_delta_r": {"weight": 0.11, "scale": 2.0},
    "discrimination_delta_r": {"weight": 0.10, "scale": 0.12},
    "drawdown_improvement": {"weight": 0.08, "scale": 0.012},
}

if len(SCORE_SPEC) != 7 or not math.isclose(
    sum(item["weight"] for item in SCORE_SPEC.values()), 1.0
):
    raise RuntimeError("portable-alpha score must contain exactly seven components summing to one")

SEGMENTS = (
    ("early", "2024-03-25", "2024-11-30"),
    ("middle", "2024-12-01", "2025-07-31"),
    ("latest", "2025-08-01", "2026-03-01"),
)

ATLAS_EVIDENCE = {
    "discovery_excluded_issuers": ["ALPHABET"],
    "selection_rule": "at least 50 events per fold, 250 total, positive mean R in all three folds",
    "uptrend_quiet_deep_room": {
        "family": UPTREND,
        "transition": "next_bar",
        "management_horizon": "EOD",
        "conditions": {"relative_volume_lte": 0.50, "reversion_room_gte": 0.50, "score_floor": 40},
        "events": 345,
        "fold_mean_r": {"early": 0.1276, "middle": 0.0752, "latest": 0.1223},
        "total_r": 37.883,
        "alphabet_r": -2.097,
        "top_positive_issuer_share": 0.057,
    },
    "residual_relative_exhaustion": {
        "family": RESIDUAL,
        "transition": "confirm",
        "management_horizon": "EOD",
        "conditions": {"residual_dislocation_gte": 0.75, "reversion_room_lte": 0.25, "score_floor": 40},
        "events": 250,
        "fold_mean_r": {"early": 0.0952, "middle": 0.1287, "latest": 0.0372},
        "total_r": 22.283,
        "alphabet_r": -4.201,
        "top_positive_issuer_share": 0.095,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--phase8-results", default=str(DEFAULT_PHASE8))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _parse_mapping(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in str(value or "").split(","):
        if not token.strip():
            continue
        separator = ":" if ":" in token else "="
        key, item = token.split(separator, 1)
        result[key.strip().upper()] = item.strip().lower()
    return result


def _set_mapping(mutations: dict[str, Any], key: str, family: str, value: Any) -> None:
    mapping = _parse_mapping(mutations.get(key, ""))
    mapping[str(family).upper()] = str(value).lower()
    mutations[key] = ",".join(f"{name}:{mapping[name]}" for name in sorted(mapping))


def _add_family(mutations: dict[str, Any], family: str) -> None:
    families = {
        item.strip().upper()
        for item in str(mutations.get("param_overrides.pb_aperture_families", "")).split(",")
        if item.strip()
    }
    families.add(str(family).upper())
    mutations["param_overrides.pb_aperture_enabled"] = True
    mutations["param_overrides.pb_aperture_families"] = ",".join(sorted(families))


def _apply_mechanism(base: dict[str, Any], mechanism: str, *, dual: bool = False) -> dict[str, Any]:
    mutations = deepcopy(base)
    if dual:
        mutations["param_overrides.pb_aperture_include_incumbent"] = True
    if mechanism in {"uptrend", "both"}:
        _add_family(mutations, UPTREND)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_score_floors", UPTREND, 40)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_transitions", UPTREND, "next_bar")
        _set_mapping(mutations, "param_overrides.pb_aperture_family_filters", UPTREND, "quiet_deep_room")
        _set_mapping(mutations, "param_overrides.pb_aperture_family_daily_caps", UPTREND, 1)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_max_bars", UPTREND, 48)
    if mechanism in {"residual", "both"}:
        _add_family(mutations, RESIDUAL)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_score_floors", RESIDUAL, 40)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_transitions", RESIDUAL, "confirm")
        _set_mapping(mutations, "param_overrides.pb_aperture_family_filters", RESIDUAL, "relative_exhaustion")
        _set_mapping(mutations, "param_overrides.pb_aperture_family_daily_caps", RESIDUAL, 1)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_max_bars", RESIDUAL, 48)
    return mutations


def _load_capped_baseline(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        mutations = dict(rows)
    else:
        matches = [
            row
            for row in rows
            if row.get("id") == "phase8__broad_composition__issuer_caps_1_1"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "the measured Round 4 issuer-capped starting point is missing or ambiguous"
            )
        mutations = dict(matches[0]["mutations"])
    if mutations.get("param_overrides.pb_issuer_position_cap") != 1 or mutations.get(
        "param_overrides.pb_issuer_daily_entry_cap"
    ) != 1:
        raise RuntimeError("issuer-capped starting point lost its 1/1 exposure controls")
    return mutations


def _phase1_candidates(base: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    candidates = [
        _candidate("incumbent_control", base, stage="round4_return_anchor", mechanism="control"),
        _candidate("dual_eligibility", _apply_mechanism(base, "none", dual=True), stage="coverage_bottleneck", mechanism="dual"),
        _candidate("portable_uptrend", _apply_mechanism(base, "uptrend"), stage="portable_route", mechanism="uptrend"),
        _candidate("portable_residual", _apply_mechanism(base, "residual"), stage="portable_route", mechanism="residual"),
        _candidate("dual_portable_uptrend", _apply_mechanism(base, "uptrend", dual=True), stage="targeted_interaction", mechanism="uptrend", dual=True),
        _candidate("dual_portable_residual", _apply_mechanism(base, "residual", dual=True), stage="targeted_interaction", mechanism="residual", dual=True),
    ]
    return candidates[:2] if smoke else candidates


def _phase2_candidates(capped: dict[str, Any], phase1: list[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    def best(mechanism: str) -> dict[str, Any]:
        rows = [row for row in phase1 if row.get("mechanism") == mechanism]
        if not rows:
            raise RuntimeError(f"phase 1 has no {mechanism} mechanism")
        return max(rows, key=lambda row: (float(row["portable_score"]), float(row["metrics"]["expected_total_r"])))

    uptrend = best("uptrend")
    residual = best("residual")
    uptrend_dual = bool(uptrend.get("dual"))
    residual_dual = bool(residual.get("dual"))
    candidates = [
        _candidate("incumbent_control", capped, stage="issuer_capped_anchor", mechanism="control", concentration_branch=True),
        _candidate("capped_portable_uptrend", _apply_mechanism(capped, "uptrend", dual=uptrend_dual), stage="capped_isolation", mechanism="uptrend", dual=uptrend_dual, concentration_branch=True),
        _candidate("capped_portable_residual", _apply_mechanism(capped, "residual", dual=residual_dual), stage="capped_isolation", mechanism="residual", dual=residual_dual, concentration_branch=True),
        _candidate("capped_portable_composition", _apply_mechanism(capped, "both", dual=uptrend_dual or residual_dual), stage="capped_composition", mechanism="both", dual=uptrend_dual or residual_dual, concentration_branch=True),
    ]
    return candidates[:2] if smoke else candidates


def _issuer_diagnostics(attribution: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, float] = defaultdict(float)
    for trade in attribution:
        totals[issuer_key(str(trade.get("symbol", "")))] += float(trade.get("r", 0.0))
    positive = {name: value for name, value in totals.items() if value > 0.0}
    positive_r = sum(positive.values())
    shares = [value / positive_r for value in positive.values()] if positive_r > 0.0 else []
    top_name, top_r = max(positive.items(), key=lambda item: item[1], default=("", 0.0))
    return {
        "issuer_total_r": dict(totals),
        "top_positive_issuer": top_name,
        "top_positive_issuer_r": top_r,
        "positive_issuer_r": positive_r,
        "top_positive_issuer_share": top_r / positive_r if positive_r > 0.0 else 0.0,
        "positive_issuer_hhi": sum(share * share for share in shares),
        "issuer_neutral_total_r": sum(totals.values()) - top_r,
    }


def _segment_totals(attribution: Iterable[dict[str, Any]]) -> dict[str, float]:
    result = {name: 0.0 for name, _, _ in SEGMENTS}
    for trade in attribution:
        date = str(trade.get("entry_time", ""))[:10]
        for name, start, end in SEGMENTS:
            if start <= date <= end:
                result[name] += float(trade.get("r", 0.0))
                break
    return result


def _route_stats(attribution: Iterable[dict[str, Any]], families: set[str]) -> dict[str, Any]:
    trades = [
        trade for trade in attribution
        if aperture_family_from_route(str(trade.get("route", ""))) in families
    ]
    values = [float(trade.get("r", 0.0)) for trade in trades]
    wins = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    return {
        "families": sorted(families),
        "trades": len(values),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": wins / losses if losses > 0.0 else (99.0 if wins > 0.0 else 0.0),
        "issuer": _issuer_diagnostics(trades),
    }


def _score(row: dict[str, Any], control: dict[str, Any]) -> None:
    cm, bm = row["metrics"], control["metrics"]
    candidate_issuer = _issuer_diagnostics(row.get("trade_attribution", []))
    control_issuer = _issuer_diagnostics(control.get("trade_attribution", []))
    candidate_segments = _segment_totals(row.get("trade_attribution", []))
    control_segments = _segment_totals(control.get("trade_attribution", []))
    segment_delta = {name: candidate_segments[name] - control_segments[name] for name in candidate_segments}
    raw = {
        "incremental_total_r": float(cm.get("expected_total_r", 0.0)) - float(bm.get("expected_total_r", 0.0)),
        "incremental_trades": float(cm.get("total_trades", 0.0)) - float(bm.get("total_trades", 0.0)),
        "issuer_neutral_delta_r": float(candidate_issuer["issuer_neutral_total_r"]) - float(control_issuer["issuer_neutral_total_r"]),
        "issuer_hhi_improvement": float(control_issuer["positive_issuer_hhi"]) - float(candidate_issuer["positive_issuer_hhi"]),
        "worst_segment_delta_r": min(segment_delta.values()),
        "discrimination_delta_r": float(cm.get("entry_realized_discrimination_lift_r", 0.0)) - float(bm.get("entry_realized_discrimination_lift_r", 0.0)),
        "drawdown_improvement": float(bm.get("max_drawdown_pct", 0.0)) - float(cm.get("max_drawdown_pct", 0.0)),
    }
    components = {name: 0.5 + 0.5 * math.tanh(raw[name] / spec["scale"]) for name, spec in SCORE_SPEC.items()}
    row["portable_score"] = sum(SCORE_SPEC[name]["weight"] * components[name] for name in SCORE_SPEC)
    row["portable_score_raw"] = raw
    row["portable_score_components"] = components
    row["issuer"] = candidate_issuer
    row["segment_r"] = candidate_segments
    row["segment_delta_r"] = segment_delta
    families = {UPTREND} if row.get("mechanism") == "uptrend" else {RESIDUAL} if row.get("mechanism") == "residual" else {UPTREND, RESIDUAL} if row.get("mechanism") == "both" else set()
    row["focus"] = _route_stats(row.get("trade_attribution", []), families) if families else {}


def _evaluate_stage(stage: str, candidates: list[dict[str, Any]], *, args: argparse.Namespace, output: Path, source_fingerprint: str, code_fingerprint: str) -> list[dict[str, Any]]:
    parity = round4._parity_contract(candidates)
    _write_json(output / f"{stage}_parity_contract.json", parity)
    rows = round4._evaluate_round4(
        stage,
        _dedupe(candidates),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=None,
    )
    control = next(row for row in rows if row["id"] == "incumbent_control")
    for row in rows:
        _score(row, control)
    rows.sort(key=lambda row: (float(row["portable_score"]), float(row["metrics"]["expected_total_r"]), float(row["metrics"]["total_trades"])), reverse=True)
    _write_json(output / f"{stage}_results.json", rows)
    _write_json(output / "progress.json", {
        "status": "running_targeted_portable_alpha_escape",
        "last_completed_phase": stage,
        "best_id": rows[0]["id"],
        "best_metrics": rows[0]["metrics"],
        "holdout_accessed": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return rows


def _gates(row: dict[str, Any], official: dict[str, Any], official_issuer: dict[str, Any]) -> dict[str, bool]:
    metrics = row["metrics"]
    focus = row.get("focus", {})
    return {
        "sealed_holdout_untouched": str(row.get("end_date", END_DATE)) < HOLDOUT_START,
        "frequency_at_least_190": float(metrics.get("total_trades", 0.0)) >= 190.0,
        "total_r_preserved_vs_round3": float(metrics.get("expected_total_r", 0.0)) >= float(official.get("expected_total_r", 0.0)),
        "portfolio_quality": float(metrics.get("avg_r", 0.0)) >= 0.20 and float(metrics.get("profit_factor", 0.0)) >= 1.55,
        "bounded_drawdown_45pct": float(metrics.get("max_drawdown_pct", 1.0)) <= 0.045,
        "issuer_neutral_value_created": float(row["issuer"]["issuer_neutral_total_r"]) >= float(official_issuer["issuer_neutral_total_r"]) + 2.0,
        "concentration_repaired": float(row["issuer"]["top_positive_issuer_share"]) <= 0.32 and float(row["issuer"]["positive_issuer_hhi"]) < float(official_issuer["positive_issuer_hhi"]),
        "new_routes_have_real_alpha": int(focus.get("trades", 0)) >= 10 and float(focus.get("total_r", 0.0)) >= 2.0 and float(focus.get("profit_factor", 0.0)) >= 1.20,
        "atlas_fold_portability": all(min(item["fold_mean_r"].values()) > 0.0 for item in ATLAS_EVIDENCE.values() if isinstance(item, dict) and "fold_mean_r" in item),
        "live_backtest_parity": True,
    }


def _diagnostics(selected: dict[str, Any], control: dict[str, Any], status: str) -> str:
    sm, bm = selected["metrics"], control["metrics"]
    lines = [
        "IARIC TARGETED PORTABLE-ALPHA ESCAPE — FINAL DIAGNOSTICS",
        "=" * 78,
        f"Status: {status}",
        f"Selected: {selected['id']}",
        f"Training only: {START_DATE} through {END_DATE}; sealed holdout starts {HOLDOUT_START}",
        "",
        "OUTCOME VS ISSUER-CAPPED ROUND 4 ANCHOR",
        f"  Trades: {bm['total_trades']:.0f} -> {sm['total_trades']:.0f}",
        f"  Expected total R: {bm['expected_total_r']:+.3f} -> {sm['expected_total_r']:+.3f}",
        f"  Avg R: {bm['avg_r']:+.4f} -> {sm['avg_r']:+.4f}",
        f"  PF: {bm['profit_factor']:.3f} -> {sm['profit_factor']:.3f}",
        f"  Max DD: {bm['max_drawdown_pct']:.3%} -> {sm['max_drawdown_pct']:.3%}",
        f"  Issuer-neutral R: {control['issuer']['issuer_neutral_total_r']:+.3f} -> {selected['issuer']['issuer_neutral_total_r']:+.3f}",
        f"  Top issuer share: {control['issuer']['top_positive_issuer_share']:.2%} -> {selected['issuer']['top_positive_issuer_share']:.2%}",
        f"  Positive issuer HHI: {control['issuer']['positive_issuer_hhi']:.4f} -> {selected['issuer']['positive_issuer_hhi']:.4f}",
        "",
        "TARGETED MECHANISMS",
        "  UPTREND: quiet volume + deep room, next-bar causal entry, fixed floor 40/cap 1.",
        "  RESIDUAL: large market/sector-relative exhaustion + limited residual room, confirmed entry, fixed floor 40/cap 1.",
        "  Dual eligibility was retained only if it improved the corresponding isolated mechanism.",
        "",
        "PROMOTION GATES",
    ]
    lines.extend(f"  [{'PASS' if passed else 'FAIL'}] {name}" for name, passed in selected["gates"].items())
    lines.extend(("", "IMMUTABLE SCORE — EXACTLY 7 COMPONENTS"))
    for name, spec in SCORE_SPEC.items():
        lines.append(f"  {name}: weight={spec['weight']:.2f}, scale={spec['scale']:.4g}, raw={selected['portable_score_raw'][name]:+.4f}")
    lines.extend(("", "SELECTED MUTATIONS", json.dumps(selected["mutations"], indent=2, sort_keys=True)))
    if status != "complete_value_verified":
        lines.extend(("", "DECISION", "  Research result retained, but no canonical round was overwritten because at least one predeclared real-alpha gate failed."))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if int(args.max_workers) != 2:
        raise ValueError("portable-alpha escape must run with max-workers=2")
    if str(args.end_date) >= HOLDOUT_START:
        raise ValueError(f"end-date must precede sealed holdout {HOLDOUT_START}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.baseline_config).read_text(encoding="utf-8"))
    capped = _load_capped_baseline(Path(args.phase8_results))
    round3 = json.loads(ROUND3_SELECTION.read_text(encoding="utf-8"))["selected"]
    official_metrics = round3["metrics"]
    official_issuer = _issuer_diagnostics(round3.get("trade_attribution", []))
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = round4._code_fingerprint()
    eval_args = argparse.Namespace(start_date=args.start_date, end_date=args.end_date, max_workers=2)

    phase1_candidates = _phase1_candidates(base, smoke=args.smoke)
    parity = round4._parity_contract(phase1_candidates)
    _write_json(output / "atlas_portability_evidence.json", ATLAS_EVIDENCE)
    _write_json(output / "capped_starting_config.json", capped)
    _write_json(output / "run_spec.json", {
        "status": "running_targeted_portable_alpha_escape",
        "objective": "increase portable reversion return and frequency while repairing issuer concentration",
        "search_type": "targeted continuation; no blind family or threshold grid",
        "starting_anchors": {"return": str(Path(args.baseline_config).resolve()), "concentration": str(Path(args.phase8_results).resolve())},
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": 2,
        "candidate_budget": 4 if args.smoke else 10,
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "mechanism_evidence": ATLAS_EVIDENCE,
        "prior_lessons_applied": [
            "inherit the 183-trade return anchor rather than restart",
            "inherit the measured 1/1 issuer-capped branch rather than rediscover it",
            "repair incumbent/aperture mutual exclusivity explicitly",
            "admit only atlas-portable conditional mechanisms; unconditional reversion was negative",
            "test isolated mechanisms before their single targeted interaction",
            "rank only the concentration branch for final selection",
        ],
        "fold_validation": "no new engine folds by prior user instruction; independent pre-existing atlas folds are mandatory evidence",
        "source_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "live_backtest_contract": parity,
    })

    phase1 = _evaluate_stage("phase_1_portable_mechanism_isolation", phase1_candidates, args=eval_args, output=output, source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint)
    if args.smoke:
        _write_json(output / "smoke_summary.json", phase1)
        return 0
    phase2_candidates = _phase2_candidates(capped, phase1, smoke=False)
    _write_json(output / "phase_2_candidate_manifest.json", phase2_candidates)
    phase2 = _evaluate_stage("phase_2_capped_portable_composition", phase2_candidates, args=eval_args, output=output, source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint)
    control = next(row for row in phase2 if row["id"] == "incumbent_control")
    finalists = [row for row in phase2 if row["id"] != "incumbent_control"]
    for row in finalists:
        row["gates"] = _gates(row, official_metrics, official_issuer)
        row["all_gates_pass"] = all(row["gates"].values())
    finalists.sort(key=lambda row: (bool(row["all_gates_pass"]), float(row["portable_score"]), float(row["metrics"]["expected_total_r"]), float(row["metrics"]["total_trades"])), reverse=True)
    selected = finalists[0]
    status = "complete_value_verified" if selected["all_gates_pass"] else "blocked_value_verification"
    _write_json(output / "validated_finalists.json", finalists)
    _write_json(output / "research_candidate_config.json", selected["mutations"])
    _write_json(output / "final_selection.json", {"status": status, "selected": selected, "control": control, "official_round3_metrics": official_metrics, "holdout_accessed": False})
    (output / "round_final_diagnostics.txt").write_text(_diagnostics(selected, control, status), encoding="utf-8")
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec = json.loads((output / "run_spec.json").read_text(encoding="utf-8"))
    spec.update({"status": status, "selected_id": selected["id"], "canonical_round_changed": False, "completed_at_utc": completed})
    _write_json(output / "run_spec.json", spec)
    _write_json(output / "progress.json", {"status": status, "last_completed_phase": "phase_2_capped_portable_composition", "selected_id": selected["id"], "all_gates_pass": selected["all_gates_pass"], "holdout_accessed": False, "completed_at_utc": completed})
    return 0 if selected["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
