"""Continue IARIC Round 4 after its completed Phase 4 checkpoint.

Phases 1-4 are immutable inputs.  This continuation starts from the Phase 4
research candidate, evaluates bounded causal lane hypotheses, and postpones
chronological validation until the new structural phases are complete.  The
sealed holdout is excluded by construction and Round 3 is evidence/control
only: this runner can only promote into Round 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.iaric.round4_scoring import SCORE_SPEC, issuer_diagnostics
from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CONTRACT_VERSION,
    DISCOVERY_START,
    EXPERIMENT_REGISTRY,
    HOLDOUT_START,
    LOCKED_VALIDATION_END,
    PHASE_ORDER,
    assess_atlas_for_optimization,
    chronology_contract,
)
from backtests.stock.auto.runners import run_iaric_round4_real_alpha as round4
from backtests.stock.auto.runners.run_iaric_escape_round3 import (
    IARIC_DIR,
    _candidate,
    _dedupe,
    _fold_validate,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _replay_source_fingerprint,
)
from strategies.stock.iaric.core.lanes import (
    FAMILY_ARCHETYPES,
    SCORE_COMPONENTS,
    aperture_family_from_route,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = IARIC_DIR / "round_4/phased_auto_architectural_restart_v2"
DEFAULT_PHASE4_SOURCE = DEFAULT_OUTPUT / "phase_4_continuation_baseline_config.json"
DEFAULT_BASELINE = DEFAULT_OUTPUT / "phase_4_continuation_baseline_config.json"
ROUND3_EVIDENCE = IARIC_DIR / "round_3/research/alpha_escape_continuation"
START_DATE = DISCOVERY_START
END_DATE = LOCKED_VALIDATION_END

MAPPING_KEYS = frozenset({
    "param_overrides.pb_aperture_family_score_floors",
    "param_overrides.pb_aperture_family_filters",
    "param_overrides.pb_aperture_family_daily_caps",
    "param_overrides.pb_aperture_family_transitions",
    "param_overrides.pb_aperture_family_max_bars",
    "param_overrides.pb_aperture_family_hybrid_next_policies",
    "param_overrides.pb_aperture_family_score_profiles",
    "param_overrides.pb_aperture_family_management_profiles",
    "param_overrides.pb_aperture_family_max_events",
    "param_overrides.pb_reversion_lane_daily_caps",
})

FAMILY_PROFILES = {
    "FAILED_BREAKDOWN_RECLAIM": "level_reclaim",
    "GAP_EXHAUSTION_RECLAIM": "shock_exhaustion",
    "GAP_FILL_RECLAIM": "shock_exhaustion",
    "GAP_PARTIAL_RECLAIM": "shock_exhaustion",
    "OPENING_FLUSH_RECLAIM": "shock_exhaustion",
    "OPENING_RANGE_LOW_RECLAIM": "level_reclaim",
    "MARKET_SECTOR_RESIDUAL_RECLAIM": "relative_shock",
    "MULTIDAY_HIGHER_LOW_RECLAIM": "trend_pullback",
    "PRIOR_DAY_LOW_RECLAIM": "level_reclaim",
    "UPTREND_PULLBACK_RECLAIM": "trend_pullback",
    "VOLUME_CLIMAX_RECLAIM": "shock_exhaustion",
    "VWAP_DEVIATION_RECLAIM": "level_reclaim",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--round3-evidence-dir", default=str(ROUND3_EVIDENCE))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_mapping(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in str(value or "").split(","):
        if not token.strip():
            continue
        separator = ":" if ":" in token else "="
        key, item = token.split(separator, 1)
        result[key.strip().upper()] = item.strip().lower()
    return result


def _set_mapping(mutations: dict[str, Any], key: str, name: str, value: Any) -> None:
    mapping = _parse_mapping(mutations.get(key, ""))
    mapping[str(name).strip().upper()] = str(value).strip().lower()
    mutations[key] = ",".join(f"{item}:{mapping[item]}" for item in sorted(mapping))


def _families(mutations: dict[str, Any]) -> set[str]:
    return {
        value.strip().upper()
        for value in str(mutations.get("param_overrides.pb_aperture_families", "")).split(",")
        if value.strip()
    }


def _add_family(mutations: dict[str, Any], family: str) -> None:
    families = _families(mutations) | {str(family).upper()}
    mutations["param_overrides.pb_aperture_enabled"] = True
    mutations["param_overrides.pb_aperture_families"] = ",".join(sorted(families))


def _scope(*, families: Iterable[str] = (), lanes: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "families": sorted({str(value).upper() for value in families if value}),
        "lanes": sorted({str(value).upper() for value in lanes if value}),
    }


def _scope_stats(attribution: Iterable[dict[str, Any]], scope: dict[str, Any]) -> dict[str, Any]:
    families = set(scope.get("families", []))
    lanes = set(scope.get("lanes", []))
    trades = []
    for trade in attribution:
        family = aperture_family_from_route(str(trade.get("route", "")))
        lane = str(trade.get("lane", trade.get("entry_lane_id", ""))).upper()
        if (family and family in families) or (lane and lane in lanes):
            trades.append(trade)
    values = [float(trade.get("r", 0.0)) for trade in trades]
    wins = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    label = "+".join([*sorted(families), *sorted(lanes)]) or "UNSPECIFIED"
    return {
        "family": label,
        "scope": scope,
        "trades": len(values),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": wins / losses if losses > 0.0 else (99.0 if wins > 0.0 else 0.0),
        "issuer": issuer_diagnostics(trades),
    }


def _trade_identity(trade: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(trade.get("symbol", "")).upper(),
        str(trade.get("entry_time", "")),
        str(trade.get("route", "")).upper(),
    )


def _behavior_delta(row: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Parent-relative activation and cannibalization attribution."""

    candidate_trades = {
        _trade_identity(trade): trade for trade in row.get("trade_attribution", [])
    }
    parent_trades = {
        _trade_identity(trade): trade for trade in parent.get("trade_attribution", [])
    }
    added = [candidate_trades[key] for key in candidate_trades.keys() - parent_trades.keys()]
    removed = [parent_trades[key] for key in parent_trades.keys() - candidate_trades.keys()]
    changed = [
        candidate_trades[key]
        for key in candidate_trades.keys() & parent_trades.keys()
        if abs(
            float(candidate_trades[key].get("r", 0.0))
            - float(parent_trades[key].get("r", 0.0))
        ) >= 0.05
    ]
    added_r = sum(float(trade.get("r", 0.0)) for trade in added)
    removed_r = sum(float(trade.get("r", 0.0)) for trade in removed)
    net_r = float(row["metrics"].get("expected_total_r", 0.0)) - float(
        parent["metrics"].get("expected_total_r", 0.0)
    )
    added_segments = round4._r_by_chronological_segment(added)
    removed_segments = round4._r_by_chronological_segment(removed)
    net_segments = {
        name: added_segments[name] - removed_segments[name]
        for name in added_segments
    }
    return {
        "parent_id": parent.get("id"),
        "materially_active": bool(added or removed or changed),
        "added_trades": len(added),
        "removed_trades": len(removed),
        "changed_trades": len(changed),
        "added_total_r": added_r,
        "removed_total_r": removed_r,
        "net_total_r": net_r,
        "added_issuer": issuer_diagnostics(added),
        "added_segment_r": added_segments,
        "removed_segment_r": removed_segments,
        "net_segment_r": net_segments,
        "positive_net_segments": sum(value > 0.0 for value in net_segments.values()),
        "calibration_net_r": net_segments["calibration"],
    }


def _evaluate_phase(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    control: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    parity = round4._parity_contract(candidates)
    _write_json(output / f"{stage}_parity_contract.json", parity)
    rows = round4._evaluate_round4(
        stage,
        _dedupe(candidates),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    for row in rows:
        scope = row.get("focus_scope")
        if scope:
            row["focus"] = _scope_stats(row.get("trade_attribution", []), scope)
    by_id = {str(row.get("id")): row for row in rows}
    phase_control = control or by_id.get("incumbent_control")
    for row in rows:
        parent_id = str(row.get("parent_id", ""))
        if not parent_id:
            continue
        parent = next(
            (
                candidate for candidate in rows
                if str(candidate.get("parent_id", "")) == parent_id
                and str(candidate.get("stage", "")).endswith("parent_control")
            ),
            by_id.get(parent_id) or control,
        )
        if parent is not None and parent is not row:
            row["incremental_attribution"] = _behavior_delta(row, parent)
        elif phase_control is not None and row is not phase_control:
            row["incremental_attribution"] = _behavior_delta(row, phase_control)
    rows.sort(
        key=lambda row: (
            float(row["round4_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
            float(row["metrics"].get("total_trades", 0.0)),
        ),
        reverse=True,
    )
    _write_json(output / f"{stage}_results.json", rows)
    _write_json(
        output / "progress.json",
        {
            "status": "running_round4_continuation",
            "last_completed_phase": stage,
            "phase_1_to_4_frozen": True,
            "best_id": rows[0]["id"] if rows else None,
            "best_metrics": rows[0]["metrics"] if rows else {},
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return rows


def _apply_profiles(base: dict[str, Any], families: Iterable[str]) -> dict[str, Any]:
    mutations = deepcopy(base)
    for family in sorted({str(value).upper() for value in families}):
        profile = FAMILY_PROFILES[family]
        _set_mapping(
            mutations,
            "param_overrides.pb_aperture_family_score_profiles",
            family,
            profile,
        )
    return mutations


def _phase5_candidates(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    enabled = _families(baseline)
    # Score-profile mutations are intentionally absent.  The previous Phase 5
    # changed weights while retaining non-equivalent raw floors, which was an
    # implicit admission relaxation.  Profile research remains deferred until
    # an outcome-blind atlas floor preserves the same activation percentile.
    return [
        _candidate(
            "incumbent_control",
            baseline,
            stage="fixed_opportunity_atlas_and_score_integrity_control",
            focus_scope=_scope(families=enabled),
        ),
    ]


def _profile_parent_beam(rows: list[dict[str, Any]], control: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [control]
    viable = [
        row for row in rows
        if row["id"] != "incumbent_control"
        and float(row["round4_score"]) > 0.51
        and float(row["metrics"].get("avg_r", 0.0)) >= 0.20
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.50
        and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.045
    ]
    if viable:
        selected.append(max(viable, key=lambda row: float(row["round4_score"])))
    return selected


def _family_lane(
    parent: dict[str, Any],
    name: str,
    family: str,
    *,
    floor: float,
    transition: str,
    daily_cap: int = 1,
    max_bar: int | None = None,
    filter_name: str | None = None,
) -> dict[str, Any]:
    mutations = deepcopy(parent["mutations"])
    _add_family(mutations, family)
    _set_mapping(mutations, "param_overrides.pb_aperture_family_score_floors", family, floor)
    _set_mapping(mutations, "param_overrides.pb_aperture_family_transitions", family, transition)
    _set_mapping(mutations, "param_overrides.pb_aperture_family_daily_caps", family, daily_cap)
    if max_bar is not None:
        _set_mapping(mutations, "param_overrides.pb_aperture_family_max_bars", family, max_bar)
    if filter_name is not None:
        _set_mapping(mutations, "param_overrides.pb_aperture_family_filters", family, filter_name)
    lane = f"APERTURE_{FAMILY_ARCHETYPES[family]}"
    _set_mapping(mutations, "param_overrides.pb_reversion_lane_daily_caps", lane, 2)
    return _candidate(
        f"{parent['id']}__{name}",
        mutations,
        stage="isolated_additive_causal_lane",
        parent_id=parent["id"],
        focus_scope=_scope(families=(family,)),
        focus_key=family,
        structural_hypothesis=name,
        independent_family_daily_cap=daily_cap,
        independent_lane_daily_cap=2,
    )


def _rescue_lane(parent: dict[str, Any]) -> dict[str, Any]:
    mutations = deepcopy(parent["mutations"])
    mutations.update({
        "param_overrides.pb_rescue_event_lane_enabled": True,
        "param_overrides.pb_rescue_event_daily_score_min": 60.0,
        "param_overrides.pb_rescue_event_entry_score_min": 65.0,
        "param_overrides.pb_rescue_event_trigger_policy": "oversold_or_multi",
    })
    _set_mapping(mutations, "param_overrides.pb_reversion_lane_daily_caps", "RESCUE_EVENT", 1)
    return _candidate(
        f"{parent['id']}__rescue_event_oversold_or_multi",
        mutations,
        stage="isolated_additive_causal_lane",
        parent_id=parent["id"],
        focus_scope=_scope(lanes=("RESCUE_EVENT",)),
        focus_key="RESCUE_EVENT",
        structural_hypothesis="separate causal rescue-event lane; completed bar then next-bar fill",
        independent_lane_daily_cap=1,
    )


def _phase6_candidates(
    parents: list[dict[str, Any]],
    atlas_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        candidates.append(
            _candidate(
                f"{parent['id']}__phase6_parent_control",
                parent["mutations"],
                stage="phase6_parent_control",
                parent_id=parent["id"],
                focus_scope=parent.get("focus_scope", _scope(families=_families(parent["mutations"]))),
            )
        )
        specs = (
            ("gap_exhaustion_p90_nextbar", "GAP_EXHAUSTION_RECLAIM", "next_bar", 12, None),
            ("gap_partial_p90_nextbar", "GAP_PARTIAL_RECLAIM", "next_bar", 24, None),
            ("opening_flush_p90_nextbar", "OPENING_FLUSH_RECLAIM", "next_bar", 24, None),
            ("opening_range_low_p90_nextbar", "OPENING_RANGE_LOW_RECLAIM", "next_bar", 24, None),
            ("residual_shock_p90_nextbar", "MARKET_SECTOR_RESIDUAL_RECLAIM", "next_bar", 24, "residual_reclaim"),
            ("vwap_deviation_p90_confirm", "VWAP_DEVIATION_RECLAIM", "confirm", 24, None),
            ("volume_climax_p90_confirm", "VOLUME_CLIMAX_RECLAIM", "confirm", 12, "participation"),
        )
        for name, family, transition, max_bar, filter_name in specs:
            integrity = (
                (atlas_summary or {}).get("family_results", {})
                .get(family, {})
                .get("score_integrity", {})
            )
            if atlas_summary is not None and not bool(integrity.get("activation_ready")):
                continue
            candidates.append(_family_lane(
                parent,
                name,
                family,
                floor=round4._atlas_family_floor(atlas_summary, family),
                transition=transition,
                max_bar=max_bar,
                filter_name=filter_name,
            ))
    return _dedupe(candidates)


def _rearm_candidate(
    parent: dict[str, Any], candidate_id: str, families: Iterable[str]
) -> dict[str, Any]:
    mutations = deepcopy(parent["mutations"])
    focus = sorted({str(family).upper() for family in families})
    for family in focus:
        _set_mapping(mutations, "param_overrides.pb_aperture_family_max_events", family, 2)
        _set_mapping(mutations, "param_overrides.pb_aperture_family_daily_caps", family, 2)
    mutations["param_overrides.pb_aperture_rearm_cooldown_bars"] = 12
    _set_mapping(
        mutations,
        "param_overrides.pb_reversion_lane_daily_caps",
        "APERTURE_LEVEL_RECLAIM",
        4,
    )
    return _candidate(
        f"{parent['id']}__{candidate_id}",
        mutations,
        stage="causal_second_dislocation_delivery",
        parent_id=parent["id"],
        focus_scope=_scope(families=focus),
        focus_key="+".join(focus),
        event_cap_per_family=2,
        family_daily_cap=2,
        lane_daily_cap=4,
        rearm_cooldown_bars=12,
        structural_hypothesis="admit one separately identified post-cooldown dislocation episode",
    )


def _phase7_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        candidates.extend([
            _candidate(
                f"{parent['id']}__phase7_parent_control",
                parent["mutations"],
                stage="phase7_parent_control",
                parent_id=parent["id"],
                focus_scope=_scope(families=("PRIOR_DAY_LOW_RECLAIM", "FAILED_BREAKDOWN_RECLAIM")),
            ),
            _rearm_candidate(parent, "pdl_second_event_cd12", ("PRIOR_DAY_LOW_RECLAIM",)),
            _rearm_candidate(parent, "failed_breakdown_second_event_cd12", ("FAILED_BREAKDOWN_RECLAIM",)),
            _rearm_candidate(
                parent,
                "pdl_failed_breakdown_second_events_cd12",
                ("PRIOR_DAY_LOW_RECLAIM", "FAILED_BREAKDOWN_RECLAIM"),
            ),
        ])
    return _dedupe(candidates)


def _positive_structural(row: dict[str, Any], control: dict[str, Any]) -> bool:
    incremental = row.get("incremental_attribution") or _behavior_delta(row, control)
    added_issuer = incremental.get("added_issuer", {})
    added_trades = int(incremental.get("added_trades", 0))
    return (
        row["id"] != "incumbent_control"
        and bool(incremental.get("materially_active"))
        and added_trades >= 20
        and float(incremental.get("added_total_r", 0.0)) > 0.0
        and float(incremental.get("net_total_r", -99.0)) >= 0.0
        and int(incremental.get("positive_net_segments", 0)) >= 2
        and float(incremental.get("calibration_net_r", -99.0)) >= 0.0
        and float(added_issuer.get("top_positive_issuer_share", 1.0)) <= 0.35
        and float(row["metrics"].get("avg_r", 0.0)) >= 0.18
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.40
        and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.045
    )


def _mutation_delta(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row["mutations"].items()
        if base.get(key) != value
    }


def _merge_rows(base: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = deepcopy(base)
    base_families = _families(base)
    merged_families = set(base_families)
    for row in rows:
        for key, value in _mutation_delta(base, row).items():
            if key == "param_overrides.pb_aperture_families":
                merged_families.update(
                    item.strip().upper() for item in str(value).split(",") if item.strip()
                )
            elif key in MAPPING_KEYS:
                base_map = _parse_mapping(base.get(key, ""))
                for item, mapped_value in _parse_mapping(value).items():
                    if base_map.get(item) != mapped_value:
                        _set_mapping(result, key, item, mapped_value)
            else:
                result[key] = value
    if merged_families != base_families:
        result["param_overrides.pb_aperture_families"] = ",".join(sorted(merged_families))
    return result


def _combined_scope(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    families: set[str] = set()
    lanes: set[str] = set()
    for row in rows:
        scope = row.get("focus_scope", {})
        families.update(scope.get("families", []))
        lanes.update(scope.get("lanes", []))
    return _scope(families=families, lanes=lanes)


def _phase8_candidates(
    baseline: dict[str, Any],
    isolated_rows: list[dict[str, Any]],
    rearm_rows: list[dict[str, Any]],
    control: dict[str, Any],
) -> list[dict[str, Any]]:
    isolated = sorted(
        (row for row in isolated_rows if _positive_structural(row, control) and row.get("focus_key")),
        key=lambda row: float(row["round4_score"]),
        reverse=True,
    )
    distinct: list[dict[str, Any]] = []
    seen_focus: set[str] = set()
    for row in isolated:
        focus_key = str(row.get("focus_key", ""))
        if focus_key not in seen_focus:
            seen_focus.add(focus_key)
            distinct.append(row)
        if len(distinct) >= 3:
            break
    rearm = sorted(
        (row for row in rearm_rows if _positive_structural(row, control) and "second_event" in row["id"]),
        key=lambda row: float(row["round4_score"]),
        reverse=True,
    )
    candidates: list[dict[str, Any]] = [
        _candidate(
            "incumbent_control",
            baseline,
            stage="phase8_round4_phase4_control",
            focus_scope=_scope(families=_families(baseline)),
        )
    ]
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    if distinct:
        sources.append(("best_isolated_lane", [distinct[0]]))
    if len(distinct) >= 2:
        sources.append(("top_two_positive_lanes", distinct[:2]))
    if rearm:
        sources.append(("best_second_event", [rearm[0]]))
    if distinct and rearm:
        sources.append(("best_lane_plus_second_event", [distinct[0], rearm[0]]))
    if len(distinct) >= 2 and rearm:
        sources.append(("two_lanes_plus_second_event", [*distinct[:2], rearm[0]]))

    for name, rows in sources:
        mutations = _merge_rows(baseline, rows)
        candidates.append(_candidate(
            f"phase8__{name}",
            mutations,
            stage="evidence_backed_capped_composition",
            source_ids=[row["id"] for row in rows],
            focus_scope=_combined_scope(rows),
            structural_hypothesis="compose only independently positive and explicitly capped causal lanes",
        ))

    broad = sources[-1][1] if sources else []
    if broad:
        broad_mutations = _merge_rows(baseline, broad)
        issuer_mutations = deepcopy(broad_mutations)
        issuer_mutations.update({
            "param_overrides.pb_issuer_position_cap": 1,
            "param_overrides.pb_issuer_daily_entry_cap": 1,
        })
        candidates.append(_candidate(
            "phase8__broad_composition__issuer_caps_1_1",
            issuer_mutations,
            stage="evidence_backed_capped_composition",
            source_ids=[row["id"] for row in broad],
            focus_scope=_combined_scope(broad),
            structural_hypothesis="reduce share-class and same-issuer concentration without altering signal extraction",
        ))
        shock_families = sorted(
            family for family in _families(broad_mutations)
            if FAMILY_ARCHETYPES.get(family) == "SHOCK_EXHAUSTION"
        )
        if shock_families:
            managed = deepcopy(broad_mutations)
            for family in shock_families:
                _set_mapping(
                    managed,
                    "param_overrides.pb_aperture_family_management_profiles",
                    family,
                    "fast_snapback",
                )
            candidates.append(_candidate(
                "phase8__broad_composition__shock_fast_snapback",
                managed,
                stage="evidence_backed_capped_composition",
                source_ids=[row["id"] for row in broad],
                focus_scope=_combined_scope(broad),
                structural_hypothesis="manage shock exhaustion separately from slower level/trend reversion",
            ))
    return _dedupe(candidates)[:7]


def _phase9_ablation_candidates(
    baseline: dict[str, Any],
    phase8_rows: list[dict[str, Any]],
    source_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build literal leave-one-lane-out tests for true multi-lane compositions."""

    lookup = {str(row.get("id")): row for row in source_rows}
    compositions = [
        row for row in phase8_rows
        if len(row.get("source_ids", [])) >= 2
        and bool((row.get("incremental_attribution") or {}).get("materially_active"))
    ]
    compositions.sort(key=lambda row: float(row.get("round4_score", 0.0)), reverse=True)
    candidates = [
        _candidate(
            "incumbent_control",
            baseline,
            stage="phase9_round4_control",
            focus_scope=_scope(families=_families(baseline)),
        )
    ]
    for composition in compositions[:2]:
        source_ids = [str(value) for value in composition.get("source_ids", [])]
        sources = [lookup[value] for value in source_ids if value in lookup]
        if len(sources) != len(source_ids):
            raise RuntimeError(f"composition {composition['id']} has unresolved source rows")
        parent_id = str(composition["id"])
        candidates.append(_candidate(
            f"{parent_id}__ablation_control",
            composition["mutations"],
            stage="phase9_parent_control",
            parent_id=parent_id,
            source_ids=source_ids,
            focus_scope=composition.get("focus_scope", _combined_scope(sources)),
        ))
        for removed in sources:
            retained = [row for row in sources if row["id"] != removed["id"]]
            candidates.append(_candidate(
                f"{parent_id}__without__{removed['id']}",
                _merge_rows(baseline, retained),
                stage="literal_lane_ablation",
                parent_id=parent_id,
                removed_source_id=removed["id"],
                source_ids=[row["id"] for row in retained],
                focus_scope=composition.get("focus_scope", _combined_scope(sources)),
            ))
    return _dedupe(candidates)


def _ablation_survivors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parent_id"):
            grouped[str(row["parent_id"])].append(row)
    survivors: list[dict[str, Any]] = []
    for parent_id, group in grouped.items():
        control = next(
            (row for row in group if str(row.get("stage", "")) == "phase9_parent_control"),
            None,
        )
        ablations = [row for row in group if row.get("removed_source_id")]
        if control is None or len(ablations) < 2:
            continue
        control_r = float(control["metrics"].get("expected_total_r", 0.0))
        checks = {
            str(row["removed_source_id"]): (
                bool((row.get("incremental_attribution") or {}).get("materially_active"))
                and control_r >= float(row["metrics"].get("expected_total_r", 0.0))
            )
            for row in ablations
        }
        control["literal_ablation_checks"] = checks
        control["literal_ablation_pass"] = bool(checks) and all(checks.values())
        if control["literal_ablation_pass"]:
            survivors.append(control)
    return survivors


def _phase10_management_candidates(
    baseline: dict[str, Any],
    parents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        _candidate(
            "incumbent_control",
            baseline,
            stage="phase10_round4_control",
            focus_scope=_scope(families=_families(baseline)),
        )
    ]
    for parent in parents[:2]:
        parent_id = str(parent["id"])
        candidates.append(_candidate(
            f"{parent_id}__management_control",
            parent["mutations"],
            stage="phase10_parent_control",
            parent_id=parent_id,
            focus_scope=parent.get("focus_scope", _scope(families=_families(parent["mutations"]))),
        ))
        typed = deepcopy(parent["mutations"])
        typed["param_overrides.pb_aperture_anchor_exit_enabled"] = True
        for family in sorted(_families(typed)):
            archetype = FAMILY_ARCHETYPES.get(family)
            profile = "fast_snapback" if archetype in {"SHOCK_EXHAUSTION", "RELATIVE_SHOCK"} else "tail_capture"
            _set_mapping(
                typed,
                "param_overrides.pb_aperture_family_management_profiles",
                family,
                profile,
            )
        candidates.append(_candidate(
            f"{parent_id}__typed_anchor_management",
            typed,
            stage="family_typed_anchor_management",
            parent_id=parent_id,
            focus_scope=parent.get("focus_scope", _scope(families=_families(typed))),
            structural_hypothesis="family target owns normalization exit; archetype owns post-entry time and tail policy",
        ))
    return _dedupe(candidates)


def _validation_shortlist(
    rows: list[dict[str, Any]], control: dict[str, Any], limit: int = 3
) -> list[dict[str, Any]]:
    control_signature = _signature(control["mutations"])
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = _signature(row["mutations"])
        if signature != control_signature:
            incumbent = unique.get(signature)
            if incumbent is None or float(row["round4_score"]) > float(incumbent["round4_score"]):
                unique[signature] = row
    pool = list(unique.values())
    if not pool:
        raise RuntimeError("Round 4 continuation produced no signature-distinct finalist")
    base = control["metrics"]
    selectors = (
        lambda row: float(row["round4_score"]),
        lambda row: (
            float(row["metrics"].get("total_trades", 0.0))
            if float(row["metrics"].get("expected_total_r", -99.0))
            >= float(base.get("expected_total_r", 0.0)) + 2.0
            and float(row["metrics"].get("profit_factor", 0.0)) >= 1.45
            and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.045
            else -1e9
        ),
        lambda row: (
            float(row["metrics"].get("expected_total_r", -99.0))
            if float(row["metrics"].get("total_trades", 0.0))
            >= float(base.get("total_trades", 0.0))
            else -1e9
        ),
    )
    selected: list[dict[str, Any]] = []
    for selector in selectors:
        row = max(pool, key=selector)
        if _signature(row["mutations"]) not in {_signature(item["mutations"]) for item in selected}:
            selected.append(row)
    return selected[:limit]


def _freeze_phase4(output: Path, baseline_path: Path) -> dict[str, Any]:
    required = (
        "phase_1_signal_supply_results.json",
        "phase_2_discrimination_results.json",
        "phase_3_entry_results.json",
        "phase_4_management_exit_results.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Round 4 Phase 1-4 checkpoint is incomplete: {missing}")
    checkpoint = output / "phase_4_checkpoint_selection.json"
    selection_source = checkpoint if checkpoint.exists() else output / "final_selection.json"
    if not selection_source.exists():
        raise RuntimeError("Round 4 Phase 4 selection checkpoint is missing")
    prior = json.loads(selection_source.read_text(encoding="utf-8"))
    selected = prior.get("selected", {})
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if _signature(selected.get("mutations", {})) != _signature(baseline):
        raise RuntimeError("Round 4 continuation baseline is not the Phase 4 selected research candidate")
    frozen = {
        "phase_1_to_4_frozen": True,
        "phase_4_selected_id": selected.get("id"),
        "phase_4_status": prior.get("status"),
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": _sha256(baseline_path),
        "artifact_sha256": {name: _sha256(output / name) for name in required},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not checkpoint.exists():
        _write_json(checkpoint, prior)
    frozen["phase_4_checkpoint_sha256"] = _sha256(checkpoint)
    _write_json(output / "phase_1_to_4_frozen_contract.json", frozen)
    return frozen


def _evidence_manifest(evidence: Path, output: Path) -> dict[str, Any]:
    names = (
        "run_spec.json",
        "progress.json",
        "phase_0_positive_route_synthesis_results.json",
        "phase_1_breadth_repair_atoms_results.json",
        "phase_2_evidence_backed_sleeve_synthesis_results.json",
        "evaluation_cache.json",
    )
    available = [name for name in names if (evidence / name).exists()]
    manifest = {
        "role": "evidence_and_incumbent_control_only",
        "source": str(evidence.resolve()),
        "available_artifact_sha256": {name: _sha256(evidence / name) for name in available},
        "cache_metrics_imported": False,
        "candidate_metrics_require_round4_replay": True,
        "reason": "Round 3 baseline and code fingerprints are not authoritative for the Phase 4 continuation",
        "imported_hypotheses": [
            "gap-exhaustion floor 75 showed positive standalone executable attribution",
            "positive sleeves should be composed only after isolation",
            "family score archetypes and causal second-dislocation rearm remained unexplored",
        ],
    }
    _write_json(output / "phase_5_round3_evidence_manifest.json", manifest)
    return manifest


def _diagnostics(selected: dict[str, Any], control: dict[str, Any], status: str) -> str:
    sm, bm = selected["metrics"], control["metrics"]
    raw = selected["round4_score_raw"]
    delta_trades = float(sm.get("total_trades", 0.0)) - float(bm.get("total_trades", 0.0))
    delta_r = float(sm.get("expected_total_r", 0.0)) - float(bm.get("expected_total_r", 0.0))
    lines = [
        "IARIC ROUND 4 CONTINUATION — PHASES 5-11 FINAL DIAGNOSTICS",
        "=" * 76,
        f"Status: {status}",
        f"Selected: {selected['id']}",
        "Phases 1-4: frozen; Phase 4 research candidate is the mutable baseline",
        f"Training: {START_DATE} through {END_DATE}; sealed holdout begins {HOLDOUT_START}",
        "",
        "OUTCOME VS PHASE 4 CHECKPOINT",
        f"  Trades: {bm.get('total_trades', 0):.0f} -> {sm.get('total_trades', 0):.0f} ({delta_trades:+.0f})",
        f"  Expected total R: {bm.get('expected_total_r', 0):+.3f} -> {sm.get('expected_total_r', 0):+.3f} ({delta_r:+.3f})",
        f"  Avg R: {bm.get('avg_r', 0):+.4f} -> {sm.get('avg_r', 0):+.4f}",
        f"  PF: {bm.get('profit_factor', 0):.3f} -> {sm.get('profit_factor', 0):.3f}",
        f"  Max DD: {bm.get('max_drawdown_pct', 0):.3%} -> {sm.get('max_drawdown_pct', 0):.3%}",
        f"  Focus scope: {selected['focus']['family']}, n={selected['focus']['trades']}, totalR={selected['focus']['total_r']:+.3f}",
        "",
        "PROMOTION GATES",
    ]
    lines.extend(f"  [{'PASS' if passed else 'FAIL'}] {name}" for name, passed in selected["gates"].items())
    lines.extend(("", "CHRONOLOGICAL FOLDS"))
    for fold in selected.get("folds", []):
        lines.append(
            f"  {fold['fold']}: deltaR={fold['delta_total_r']:+.3f}, delta trades={fold['delta_trades']:+.0f}"
        )
    lines.extend(("", "IMMUTABLE ROUND 4 SCORE — EXACTLY 7 COMPONENTS"))
    for name, spec in SCORE_SPEC.items():
        lines.append(f"  {name}: weight={spec['weight']:.2f}, scale={spec['scale']:.4g}")
    lines.extend(("", "SELECTED MUTATIONS", json.dumps(selected["mutations"], indent=2, sort_keys=True)))
    if status != "complete_value_verified":
        lines.extend((
            "",
            "PROMOTION DECISION",
            "  The candidate remains a research result; canonical Round 4/live configuration",
            "  was not changed because at least one predeclared gate failed.",
        ))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if int(args.max_workers) != 2:
        raise ValueError("Round 4 continuation must run with max-workers=2")
    if str(args.end_date) >= HOLDOUT_START:
        raise ValueError(f"end-date must precede sealed holdout {HOLDOUT_START}")
    if len(SCORE_SPEC) != 7 or len(SCORE_COMPONENTS) != 7:
        raise RuntimeError("both optimization and causal event scores must contain exactly seven components")

    output = Path(args.output_dir).resolve()
    baseline_path = Path(args.baseline_config).resolve()
    evidence_path = Path(args.round3_evidence_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atlas_path = output / "phase_0_opportunity_atlas" / "atlas_summary.json"
    if not atlas_path.exists():
        _write_json(output / "representative_continuation_blocker.json", {
            "status": "blocked_missing_authority_preflight",
            "blockers": [f"missing authority/atlas summary: {atlas_path}"],
            "holdout_accessed": False,
        })
        return 2
    atlas_preflight = json.loads(atlas_path.read_text(encoding="utf-8"))
    representative = assess_atlas_for_optimization(atlas_preflight)
    if not representative["passed"]:
        _write_json(output / "representative_continuation_blocker.json", {
            "status": "blocked_representative_pipeline_contract",
            "representative_contract_version": CONTRACT_VERSION,
            "assessment": representative,
            "blockers": representative["blockers"],
            "holdout_accessed": False,
        })
        return 2

    # Baseline hashing, replay loading and evidence scans are deliberately
    # downstream of the inexpensive direct-entrypoint authority gate.
    if not baseline_path.exists() and baseline_path == DEFAULT_BASELINE.resolve():
        source = output / DEFAULT_PHASE4_SOURCE.name
        if not source.exists():
            raise RuntimeError("Round 4 Phase 4 research candidate is missing")
        _write_json(
            baseline_path,
            json.loads(source.read_text(encoding="utf-8")),
        )
    frozen = _freeze_phase4(output, baseline_path)
    evidence = _evidence_manifest(evidence_path, output)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = round4._code_fingerprint()
    atlas_summary = round4._load_atlas_summary(atlas_path, source_fingerprint)
    _write_json(output / "phase_5_score_integrity_manifest.json", {
        "atlas": str(atlas_path),
        "code_fingerprint": atlas_summary.get("code_fingerprint"),
        "family_score_integrity": {
            family: row.get("score_integrity", {})
            for family, row in atlas_summary.get("family_results", {}).items()
        },
        "outcome_used_for_floor_calibration": False,
        "profile_weight_experiments_enabled": False,
        "inert_rescue_lane_enabled": False,
    })
    eval_args = argparse.Namespace(
        start_date=str(args.start_date),
        end_date=CALIBRATION_END,
        max_workers=int(args.max_workers),
    )

    phase5_candidates = _phase5_candidates(baseline)
    parity = round4._parity_contract(phase5_candidates)
    _write_json(output / "continuation_parity_contract.json", parity)
    _write_json(output / "run_spec.json", {
        "status": "running_round4_continuation",
        "objective": "widen causal reversion definition and delivery while preserving discrimination, attribution and bounded risk",
        "round": 4,
        "continued_after_completed_phase": 4,
        "baseline": str(baseline_path),
        "baseline_sha256": frozen["baseline_sha256"],
        "chronology": chronology_contract(),
        "representative_contract_version": CONTRACT_VERSION,
        "selection_window": {"start": args.start_date, "end": CALIBRATION_END},
        "locked_validation_used_for_candidate_ranking": False,
        "max_workers": 2,
        "phase_order": list(PHASE_ORDER),
        "experiment_registry": EXPERIMENT_REGISTRY,
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "causal_event_score_component_count": len(SCORE_COMPONENTS),
        "phase_1_to_4_contract": frozen,
        "round3_evidence_contract": evidence,
        "source_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "anti_overfit_contract": {
            "fixed_mechanism_presets_not_threshold_grids": True,
            "standalone_lane_attribution_precedes_composition": True,
            "behavioral_activation_is_parent_relative": True,
            "incumbent_trade_cannibalization_is_charged_to_lane": True,
            "each_additive_lane_is_independently_capped": True,
            "only_one_predeclared_rearm_cooldown": 12,
            "round4_score_unchanged_after_phase4": True,
            "controls_carried_each_phase": True,
            "fixed_opportunity_atlas_denominator": True,
            "literal_leave_one_lane_out_ablation": True,
            "failed_gates_block_promotion": True,
        },
        "live_backtest_contract": parity,
    })
    _write_json(output / "progress.json", {
        "status": "running_round4_continuation",
        "last_completed_phase": "phase_4_trade_management_and_exit_checkpoint",
        "current_phase": "phase_5_fixed_opportunity_atlas_and_score_integrity",
        "phase_1_to_4_frozen": True,
        "holdout_accessed": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    phase5 = _evaluate_phase(
        "phase_5_fixed_opportunity_atlas_and_score_integrity",
        phase5_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=None,
    )
    control = next(row for row in phase5 if row["id"] == "incumbent_control")
    if args.smoke:
        _write_json(output / "phase_5_smoke_summary.json", phase5)
        return 0

    parents = _profile_parent_beam(phase5, control)
    _write_json(output / "phase_5_survivors.json", parents)
    phase6 = _evaluate_phase(
        "phase_6_isolated_additive_causal_lanes",
        _phase6_candidates(parents, atlas_summary),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    phase7 = _evaluate_phase(
        "phase_7_causal_second_dislocation_delivery",
        _phase7_candidates(parents),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    phase8_candidates = _phase8_candidates(baseline, phase6, phase7, control)
    _write_json(output / "phase_8_candidate_manifest.json", phase8_candidates)
    phase8 = _evaluate_phase(
        "phase_8_evidence_backed_capped_composition",
        phase8_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )

    phase9_candidates = _phase9_ablation_candidates(
        baseline,
        phase8,
        [*phase6, *phase7],
    )
    _write_json(output / "phase_9_candidate_manifest.json", phase9_candidates)
    phase9 = _evaluate_phase(
        "phase_9_literal_lane_ablation_and_simplification",
        phase9_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    ablation_survivors = _ablation_survivors(phase9)
    _write_json(output / "phase_9_ablation_survivors.json", ablation_survivors)
    standalone_parents = sorted(
        (
            row for row in phase8
            if row["id"] != "incumbent_control"
            and len(row.get("source_ids", [])) == 1
            and _positive_structural(row, control)
        ),
        key=lambda row: float(row["round4_score"]),
        reverse=True,
    )
    management_parents = ablation_survivors or standalone_parents[:2]
    phase10 = _evaluate_phase(
        "phase_10_family_typed_management",
        _phase10_management_candidates(baseline, management_parents),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )

    finalist_pool = [*phase6, *phase7, *standalone_parents, *ablation_survivors, *phase10]
    finalists = _validation_shortlist(finalist_pool, control)
    _write_json(output / "phase_11_validation_shortlist.json", finalists)
    _fold_validate(
        finalists,
        control,
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    for row in finalists:
        row["gates"] = round4._gates(row, control)
        row["all_gates_pass"] = all(row["gates"].values())
    finalists.sort(
        key=lambda row: (
            bool(row["all_gates_pass"]),
            float(row["round4_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
            float(row["metrics"].get("total_trades", 0.0)),
        ),
        reverse=True,
    )
    selected = finalists[0]
    status = "complete_value_verified" if selected["all_gates_pass"] else "blocked_value_verification"
    _write_json(output / "phase_11_validated_finalists.json", finalists)
    _write_json(output / "validated_finalists.json", finalists)
    _write_json(output / "research_candidate_config.json", selected["mutations"])
    _write_json(output / "final_selection.json", {
        "status": status,
        "continued_after_completed_phase": 4,
        "selected": selected,
        "control": control,
        "holdout_accessed": False,
    })
    diagnostics = _diagnostics(selected, control, status)
    (output / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    if selected["all_gates_pass"]:
        round_dir = IARIC_DIR / "round_4"
        _write_json(round_dir / "optimized_config.json", selected["mutations"])
        _write_json(round_dir / "run_summary.json", {
            "status": status,
            "selected_id": selected["id"],
            "metrics": selected["metrics"],
            "gates": selected["gates"],
            "holdout_accessed": False,
            "score_component_count": len(SCORE_SPEC),
            "selection": "round_4/phased_auto/final_selection.json",
            "continued_after_completed_phase": 4,
        })
        (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_spec = json.loads((output / "run_spec.json").read_text(encoding="utf-8"))
    run_spec.update({
        "status": status,
        "selected_id": selected["id"],
        "canonical_round_changed": bool(selected["all_gates_pass"]),
        "completed_at_utc": completed,
    })
    _write_json(output / "run_spec.json", run_spec)
    _write_json(output / "progress.json", {
        "status": status,
        "last_completed_phase": "phase_11_chronological_validation_and_promotion",
        "selected_id": selected["id"],
        "all_gates_pass": selected["all_gates_pass"],
        "phase_1_to_4_frozen": True,
        "holdout_accessed": False,
        "completed_at_utc": completed,
    })
    return 0 if selected["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
