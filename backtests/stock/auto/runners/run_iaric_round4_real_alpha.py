"""Restart the bounded, causal IARIC Round 4 search after architecture repair.

The restart preserves the latest optimized Phase 4 lineage, migrates it onto
the repaired causal score, issuer-risk and entry/exit contracts, and then
re-runs the experimental sequence required by the strategy implementation
lessons.  The sealed holdout is excluded by construction and a failed gate can
never update the live/canonical configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.iaric.round4_scoring import (
    SCORE_SPEC,
    fixed_atlas_recall,
    issuer_diagnostics,
    score_candidate,
)
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
from backtests.stock.auto.runners.run_iaric_escape_round3 import (
    EVALUATION_OWNED_FIELDS,
    IARIC_DIR,
    _candidate,
    _candidate_metadata,
    _evaluate,
    _family_transition_mutation,
    _fold_validate,
    _replay_candidate,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _replay_source_fingerprint,
    _signature,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core.lanes import SCORE_COMPONENTS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = IARIC_DIR / "round_4/phased_auto/phase_4_continuation_baseline_config.json"
DEFAULT_BASELINE_REFERENCE = IARIC_DIR / "round_4/phased_auto/final_selection.json"
DEFAULT_BASELINE_METRICS = IARIC_DIR / "round_4/phased_auto_architectural_restart_v2/baseline_metrics.json"
DEFAULT_OUTPUT = IARIC_DIR / "round_4/phased_auto_architectural_restart_v2"
START_DATE = DISCOVERY_START
END_DATE = LOCKED_VALIDATION_END

BASE_FAMILIES = {
    "FAILED_BREAKDOWN_RECLAIM",
    "MULTIDAY_HIGHER_LOW_RECLAIM",
    "PRIOR_DAY_LOW_RECLAIM",
    "UPTREND_PULLBACK_RECLAIM",
}

FAMILY_SCORE_PROFILES = {
    "FAILED_BREAKDOWN_RECLAIM": "level_reclaim",
    "GAP_EXHAUSTION_RECLAIM": "shock_exhaustion",
    "GAP_PARTIAL_RECLAIM": "shock_exhaustion",
    "OPENING_FLUSH_RECLAIM": "shock_exhaustion",
    "OPENING_RANGE_LOW_RECLAIM": "level_reclaim",
    "MARKET_SECTOR_RESIDUAL_RECLAIM": "relative_shock",
    "MULTIDAY_HIGHER_LOW_RECLAIM": "trend_pullback",
    "PRIOR_DAY_LOW_RECLAIM": "level_reclaim",
    "UPTREND_PULLBACK_RECLAIM": "trend_pullback",
    "VWAP_DEVIATION_RECLAIM": "level_reclaim",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--baseline-reference", default=str(DEFAULT_BASELINE_REFERENCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--opportunity-atlas-summary", default="")
    parser.add_argument("--require-representative-inputs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _code_fingerprint() -> str:
    # Fingerprint only code that can change replay economics. Candidate
    # orchestration and post-hoc scoring are deliberately excluded: mutation
    # signatures identify the tested behavior, while an orchestration-only
    # resume fix must not invalidate hours of completed causal replays.
    paths = (
        REPO_ROOT / "backtests/stock/auto/iaric/worker.py",
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_intraday_hybrid_engine.py",
        REPO_ROOT / "strategies/stock/iaric/config.py",
        REPO_ROOT / "strategies/stock/iaric/core/lanes.py",
        REPO_ROOT / "strategies/stock/iaric/core/logic.py",
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
        REPO_ROOT / "strategies/stock/iaric/core/residual.py",
        REPO_ROOT / "strategies/stock/iaric/engine.py",
        REPO_ROOT / "strategies/stock/iaric/exits.py",
        REPO_ROOT / "strategies/stock/iaric/entry_request.py",
        REPO_ROOT / "strategies/stock/iaric/models.py",
        REPO_ROOT / "strategies/stock/iaric/research.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parse_map(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in str(value or "").split(","):
        if not token.strip():
            continue
        separator = ":" if ":" in token else "="
        key, item = token.split(separator, 1)
        result[key.strip().upper()] = item.strip()
    return result


def _set_map(mutations: dict[str, Any], key: str, family: str, value: Any) -> None:
    mapping = _parse_map(mutations.get(key, ""))
    mapping[str(family).upper()] = str(value)
    mutations[key] = ",".join(f"{name}:{mapping[name]}" for name in sorted(mapping))


def _add_family(base: dict[str, Any], family: str) -> dict[str, Any]:
    mutations = deepcopy(base)
    key = "param_overrides.pb_aperture_families"
    families = {value.strip().upper() for value in str(mutations.get(key, "")).split(",") if value.strip()}
    families.add(str(family).upper())
    mutations[key] = ",".join(sorted(families))
    mutations["param_overrides.pb_aperture_enabled"] = True
    return mutations


def _architectural_baseline(latest: dict[str, Any]) -> dict[str, Any]:
    """Apply integrity invariants without changing incumbent selectivity.

    The latest optimized family floors and score profile remain authoritative.
    New score geometry must first prove baseline incidence; it must never be
    "rescaled" by assigning a lower threshold from intuition alone.
    """

    migrated = deepcopy(latest)
    migrated.update({
        "param_overrides.pb_aperture_min_remaining_room_atr": 0.10,
        "param_overrides.pb_aperture_min_prospective_rr": 0.60,
        "param_overrides.pb_aperture_anchor_exit_enabled": False,
        "param_overrides.pb_issuer_event_dedupe_enabled": True,
        "param_overrides.pb_issuer_position_cap": 1,
        "param_overrides.pb_issuer_daily_entry_cap": 2,
    })
    return migrated


def _structural_preset(
    base: dict[str, Any],
    family: str,
    *,
    floor: float,
    transition: str,
    filter_name: str | None = None,
    daily_cap: int | None = None,
    max_bar: int | None = None,
) -> dict[str, Any]:
    mutations = _add_family(base, family)
    _set_map(mutations, "param_overrides.pb_aperture_family_score_floors", family, floor)
    _set_map(mutations, "param_overrides.pb_aperture_family_transitions", family, transition)
    # Lane isolation uses the immutable balanced score.  Profile weights are
    # a separate experiment and may only be introduced with an outcome-blind
    # percentile-equivalent floor from the fixed opportunity atlas.
    if filter_name is not None:
        _set_map(mutations, "param_overrides.pb_aperture_family_filters", family, filter_name)
    if daily_cap is not None:
        _set_map(mutations, "param_overrides.pb_aperture_family_daily_caps", family, daily_cap)
    if max_bar is not None:
        _set_map(mutations, "param_overrides.pb_aperture_family_max_bars", family, max_bar)
    return mutations


def _atlas_family_floor(
    atlas_summary: dict[str, Any] | None,
    family: str,
    *,
    profile: str = "balanced",
    fallback: float = 65.0,
) -> float:
    if atlas_summary is None:
        return float(fallback)
    integrity = (
        atlas_summary.get("family_results", {})
        .get(family, {})
        .get("score_integrity", {})
    )
    quantiles = integrity.get("profile_score_quantiles", {}).get(profile, {})
    value = float(quantiles.get("p90", fallback))
    return round(min(max(value, 0.0), 100.0), 2)


def _supply_candidates(
    baseline: dict[str, Any],
    *,
    atlas_summary: dict[str, Any] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Pre-specified mechanisms aimed at unused supply, not a threshold grid."""

    specs = (
        (
            "gap_exhaustion_isolated",
            "GAP_EXHAUSTION_RECLAIM",
            dict(floor=65, transition="next_bar", daily_cap=1, max_bar=12),
        ),
        (
            "partial_gap_reversion_isolated",
            "GAP_PARTIAL_RECLAIM",
            dict(floor=65, transition="next_bar", daily_cap=1, max_bar=24),
        ),
        (
            "opening_flush_isolated",
            "OPENING_FLUSH_RECLAIM",
            dict(floor=65, transition="next_bar", daily_cap=1, max_bar=24),
        ),
        (
            "opening_range_low_isolated",
            "OPENING_RANGE_LOW_RECLAIM",
            dict(floor=65, transition="next_bar", daily_cap=1, max_bar=24),
        ),
        (
            "residual_dislocation_reclaim",
            "MARKET_SECTOR_RESIDUAL_RECLAIM",
            dict(
                floor=65,
                transition="next_bar",
                filter_name="residual_reclaim",
                daily_cap=1,
                max_bar=24,
            ),
        ),
        (
            "vwap_room_reclaim",
            "VWAP_DEVIATION_RECLAIM",
            dict(
                floor=65,
                transition="next_bar",
                daily_cap=1,
                max_bar=24,
            ),
        ),
        (
            "volume_climax_partial_reclaim",
            "VOLUME_CLIMAX_RECLAIM",
            dict(
                floor=65,
                transition="next_bar",
                filter_name="participation",
                daily_cap=1,
                max_bar=12,
            ),
        ),
    )
    if smoke:
        specs = specs[:1]
    candidates = [_candidate("incumbent_control", baseline, stage="baseline", families=sorted(BASE_FAMILIES))]
    for name, family, kwargs in specs:
        integrity = (
            (atlas_summary or {}).get("family_results", {})
            .get(family, {})
            .get("score_integrity", {})
        )
        if atlas_summary is not None and not bool(integrity.get("activation_ready")):
            continue
        kwargs = dict(kwargs)
        kwargs["floor"] = _atlas_family_floor(
            atlas_summary,
            family,
            fallback=float(kwargs["floor"]),
        )
        candidates.append(
            _candidate(
                name,
                _structural_preset(baseline, family, **kwargs),
                stage="signal_supply",
                families=sorted(BASE_FAMILIES | {family}),
                focus_family=family,
                parent_id="incumbent_control",
            )
        )
    return candidates


def _load_atlas_summary(path: Path, source_fingerprint: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete_research_only":
        raise RuntimeError(f"opportunity atlas is incomplete: {path}")
    if bool(payload.get("holdout_accessed")):
        raise RuntimeError("opportunity atlas accessed the sealed holdout")
    if str(payload.get("data_fingerprint", "")) != str(source_fingerprint):
        raise RuntimeError("opportunity atlas data fingerprint does not match replay source")
    family_results = payload.get("family_results", {})
    if not family_results:
        raise RuntimeError("opportunity atlas contains no family integrity results")
    return payload


def _baseline_incidence_contract(
    control: dict[str, Any],
    reference_control: dict[str, Any],
) -> dict[str, Any]:
    """Outcome-blind guard against silent baseline aperture expansion."""

    observed_total = int(control.get("metrics", {}).get("total_trades", 0))
    expected_total = int(reference_control.get("metrics", {}).get("total_trades", 0))
    observed_aperture = int(control.get("aperture", {}).get("trades", 0))
    expected_aperture = int(reference_control.get("aperture", {}).get("trades", 0))
    observed_open = int(control.get("metrics", {}).get("open_scored_trades", 0))
    expected_open = int(reference_control.get("metrics", {}).get("open_scored_trades", 0))
    observed_routes = control.get("aperture", {}).get("routes", {})
    expected_routes = reference_control.get("aperture", {}).get("routes", {})
    route_checks = {
        route: int(observed_routes.get(route, {}).get("trades", 0))
        <= max(2 * int(expected.get("trades", 0)), int(expected.get("trades", 0)) + 5)
        for route, expected in expected_routes.items()
    }
    funnels = control.get("funnel_counters", {})
    detected = sum(
        int(funnels.get(f"lane__{lane}__event_detected", 0))
        for lane in ("aperture_level_reclaim", "aperture_trend_pullback")
    )
    score_rejected = sum(
        int(funnels.get(f"lane__{lane}__score_rejected", 0))
        for lane in ("aperture_level_reclaim", "aperture_trend_pullback")
    )
    aperture_ready = int(funnels.get("aperture_ready", 0))
    reference_funnels = reference_control.get("funnel_counters", {})
    reference_detected = sum(
        int(reference_funnels.get(f"lane__{lane}__event_detected", 0))
        for lane in ("aperture_level_reclaim", "aperture_trend_pullback")
    )
    reference_ready = int(reference_funnels.get("aperture_ready", expected_aperture))
    reference_admission_rate = reference_ready / reference_detected if reference_detected else 0.0
    observed_admission_rate = aperture_ready / detected if detected else 1.0
    checks = {
        "total_trade_incidence_preserved": (
            max(math.floor(0.50 * expected_total), 1)
            <= observed_total
            <= max(math.ceil(1.50 * expected_total), expected_total + 25)
        ),
        "aperture_incidence_preserved": (
            max(math.floor(0.50 * expected_aperture), 1)
            <= observed_aperture
            <= max(math.ceil(1.50 * expected_aperture), expected_aperture + 15)
        ),
        "open_scored_incidence_preserved": abs(observed_open - expected_open) <= 25,
        "family_routes_not_silently_expanded": all(route_checks.values()),
        "admission_remains_discriminatory": (
            detected > 0
            and observed_admission_rate <= max(0.05, 2.0 * reference_admission_rate)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "route_checks": route_checks,
        "expected": {
            "total_trades": expected_total,
            "aperture_trades": expected_aperture,
            "open_scored_trades": expected_open,
        },
        "observed": {
            "total_trades": observed_total,
            "aperture_trades": observed_aperture,
            "open_scored_trades": observed_open,
            "event_detected": detected,
            "score_rejected": score_rejected,
            "score_rejection_rate": score_rejected / detected if detected else 0.0,
            "aperture_ready": aperture_ready,
            "admission_rate": observed_admission_rate,
        },
        "reference_admission_rate": reference_admission_rate,
        "basis": "trade/event incidence only; no return or outcome was used",
    }


def _focus_stats(attribution: Iterable[dict[str, Any]], family: str | None) -> dict[str, Any]:
    prefix = f"APERTURE_{str(family or '').upper()}"
    trades = [trade for trade in attribution if str(trade.get("route", "")).upper().startswith(prefix)]
    values = [float(trade.get("r", 0.0)) for trade in trades]
    wins = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    issuer = issuer_diagnostics(trades)
    return {
        "family": family,
        "trades": len(values),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0),
        "issuer": issuer,
    }


def _trade_identity(trade: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(trade.get("symbol", "")).upper(),
        str(trade.get("entry_time", "")),
        str(trade.get("route", "")).upper(),
    )


def _r_by_chronological_segment(trades: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals = {"discovery": 0.0, "calibration": 0.0}
    for trade in trades:
        value = str(trade.get("entry_time", ""))[:10]
        if not value:
            continue
        if value <= "2024-11-30":
            totals["discovery"] += float(trade.get("r", 0.0))
        elif value <= CALIBRATION_END:
            totals["calibration"] += float(trade.get("r", 0.0))
    return totals


def _incremental_attribution(row: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    candidate = {_trade_identity(trade): trade for trade in row.get("trade_attribution", [])}
    control = {_trade_identity(trade): trade for trade in parent.get("trade_attribution", [])}
    added = [candidate[key] for key in candidate.keys() - control.keys()]
    removed = [control[key] for key in control.keys() - candidate.keys()]
    changed = sum(
        abs(float(candidate[key].get("r", 0.0)) - float(control[key].get("r", 0.0))) >= 0.05
        for key in candidate.keys() & control.keys()
    )
    added_segments = _r_by_chronological_segment(added)
    removed_segments = _r_by_chronological_segment(removed)
    net_segments = {
        name: added_segments[name] - removed_segments[name]
        for name in added_segments
    }
    return {
        "parent_id": parent.get("id"),
        "materially_active": bool(added or removed or changed),
        "added_trades": len(added),
        "removed_trades": len(removed),
        "changed_trades": changed,
        "added_total_r": sum(float(trade.get("r", 0.0)) for trade in added),
        "removed_total_r": sum(float(trade.get("r", 0.0)) for trade in removed),
        "net_total_r": float(row["metrics"].get("expected_total_r", 0.0))
        - float(parent["metrics"].get("expected_total_r", 0.0)),
        "added_issuer": issuer_diagnostics(added),
        "added_segment_r": added_segments,
        "removed_segment_r": removed_segments,
        "net_segment_r": net_segments,
        "positive_net_segments": sum(value > 0.0 for value in net_segments.values()),
        "calibration_net_r": net_segments["calibration"],
    }


def _evaluate_round4(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    control: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if str(args.end_date) > CALIBRATION_END:
        raise ValueError(
            "candidate ranking cannot access the locked internal-validation period"
        )
    rows = _evaluate(
        stage,
        candidates,
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=None if control is None else control["metrics"],
        cache_filename="round4_evaluation_cache.json",
    )
    anchor = control or next(row for row in rows if row["id"] == "incumbent_control")
    for row in rows:
        score, components, raw, audit = score_candidate(
            row["metrics"],
            list(row.get("trade_attribution", [])),
            anchor["metrics"],
            list(anchor.get("trade_attribution", [])),
        )
        row["round4_score"] = score
        row["round4_score_components"] = components
        row["round4_score_raw"] = raw
        row["round4_score_audit"] = audit
        row["issuer"] = audit["candidate_issuer"]
        row["focus"] = _focus_stats(row.get("trade_attribution", []), row.get("focus_family"))
    by_id = {str(row.get("id")): row for row in rows}
    for row in rows:
        parent_id = str(row.get("parent_id", ""))
        parent = None
        if parent_id:
            parent = next(
                (
                    candidate for candidate in rows
                    if str(candidate.get("parent_id", "")) == parent_id
                    and str(candidate.get("id", "")).endswith("control")
                ),
                by_id.get(parent_id),
            )
        if parent is None and row is not anchor:
            parent = anchor
        if parent is not None and parent is not row:
            row["incremental_attribution"] = _incremental_attribution(row, parent)
    rows.sort(
        key=lambda row: (
            float(row["round4_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
            float(row["metrics"].get("total_trades", 0.0)),
        ),
        reverse=True,
    )
    _write_json(output / f"{stage}_results.json", rows)
    return rows


def _shortlist(rows: list[dict[str, Any]], control: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    base_dd = float(control["metrics"].get("max_drawdown_pct", 0.0))
    viable = [
        row
        for row in rows
        if row["id"] != "incumbent_control"
        and bool(row.get("incremental_attribution", {}).get("materially_active"))
        and int(row.get("incremental_attribution", {}).get("added_trades", 0)) >= 20
        and float(row.get("incremental_attribution", {}).get("added_total_r", 0.0)) > 0.0
        and float(row.get("incremental_attribution", {}).get("net_total_r", -99.0)) >= 0.0
        and int(row.get("incremental_attribution", {}).get("positive_net_segments", 0)) >= 2
        and float(row.get("incremental_attribution", {}).get("calibration_net_r", -99.0)) >= 0.0
        and float(
            row.get("incremental_attribution", {})
            .get("added_issuer", {})
            .get("top_positive_issuer_share", 1.0)
        ) <= 0.35
        and int(row.get("focus", {}).get("trades", 0)) >= 20
        and float(row.get("focus", {}).get("total_r", 0.0)) > 0.0
        and float(row["metrics"].get("avg_r", 0.0)) >= 0.18
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.40
        and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= max(0.045, base_dd + 0.012)
    ]
    return viable[:limit]


def _child(parent: dict[str, Any], suffix: str, delta: dict[str, Any], stage: str) -> dict[str, Any]:
    mutations = dict(parent["mutations"])
    mutations.update(delta)
    return _candidate(
        f"{parent['id']}__{suffix}",
        mutations,
        stage=stage,
        families=list(parent.get("families", [])),
        focus_family=parent.get("focus_family"),
        parent_id=parent["id"],
    )


def _discrimination_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        candidates.extend(
            (
                _child(parent, "discrimination_control", {}, "discrimination"),
                _child(
                    parent,
                    "aperture_room15_rr75",
                    {
                        "param_overrides.pb_aperture_min_remaining_room_atr": 0.15,
                        "param_overrides.pb_aperture_min_prospective_rr": 0.75,
                    },
                    "discrimination",
                ),
                _child(
                    parent,
                    "aperture_room25_rr100",
                    {
                        "param_overrides.pb_aperture_min_remaining_room_atr": 0.25,
                        "param_overrides.pb_aperture_min_prospective_rr": 1.00,
                    },
                    "discrimination",
                ),
                _child(
                    parent,
                    "aperture_room10_rr100",
                    {
                        "param_overrides.pb_aperture_min_remaining_room_atr": 0.10,
                        "param_overrides.pb_aperture_min_prospective_rr": 1.00,
                    },
                    "discrimination",
                ),
            )
        )
        family = str(parent.get("focus_family", ""))
        relaxed = dict(parent["mutations"])
        _set_map(relaxed, "param_overrides.pb_aperture_family_daily_caps", family, 2)
        candidates.append(
            _candidate(
                f"{parent['id']}__focus_cap2",
                relaxed,
                stage="discrimination",
                families=list(parent.get("families", [])),
                focus_family=family,
                parent_id=parent["id"],
            )
        )
    return candidates


def _entry_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        family = str(parent.get("focus_family", ""))
        candidates.append(_child(parent, "entry_control", {}, "entry"))
        for transition in ("confirm", "retrace"):
            candidates.append(
                _candidate(
                    f"{parent['id']}__focus_{transition}",
                    _family_transition_mutation(parent["mutations"], family, transition),
                    stage="entry",
                    families=list(parent.get("families", [])),
                    focus_family=family,
                    parent_id=parent["id"],
                )
            )
        hybrid = _family_transition_mutation(
            parent["mutations"], family, "quality_hybrid"
        )
        hybrid_policy = (
            "residual_reclaim"
            if family == "MARKET_SECTOR_RESIDUAL_RECLAIM"
            else "room_reclaim"
        )
        _set_map(
            hybrid,
            "param_overrides.pb_aperture_family_hybrid_next_policies",
            family,
            hybrid_policy,
        )
        candidates.append(_candidate(
            f"{parent['id']}__focus_quality_hybrid",
            hybrid,
            stage="entry",
            families=list(parent.get("families", [])),
            focus_family=family,
            parent_id=parent["id"],
        ))
        candidates.append(
            _child(
                parent,
                "open_confirmed_retest",
                {
                    "param_overrides.pb_open_scored_transition": "confirmed_retest",
                    "param_overrides.pb_open_scored_retest_window_bars": 6,
                    "param_overrides.pb_open_scored_retest_retrace_frac": 0.35,
                },
                "entry",
            )
        )
        candidates.append(
            _child(
                parent,
                "open_reclaim_or_limit",
                {
                    "param_overrides.pb_open_scored_transition": "reclaim_or_limit",
                    "param_overrides.pb_open_scored_limit_anchor": "daily_atr",
                    "param_overrides.pb_open_scored_limit_atr_frac": 0.25,
                    "param_overrides.pb_open_scored_limit_arm_bar": 3,
                },
                "entry",
            )
        )
    return candidates


def _management_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = (
        ("management_control", {}),
        (
            "earlier_progress_protection",
            {
                "param_overrides.pb_v2_mfe_stage1_trigger": 0.35,
                "param_overrides.pb_v2_mfe_stage1_stop_r": -0.10,
                "param_overrides.pb_v2_mfe_stage2_trigger": 0.55,
            },
        ),
        (
            "small_partial75",
            {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.75,
                "param_overrides.pb_v2_partial_profit_fraction": 0.25,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.0,
            },
        ),
        (
            "wider_tail_trail",
            {"param_overrides.pb_v2_mfe_stage3_trail_atr": 1.0},
        ),
        (
            "aperture_tail_capture",
            {
                "param_overrides.pb_aperture_max_hold_days": 5,
                "param_overrides.pb_aperture_rsi_exit": 62.0,
                "param_overrides.pb_aperture_quick_exit_loss_r": 0.50,
                "param_overrides.pb_aperture_stale_exit_bars": 0,
            },
        ),
        ("aggressive_bounded_size", {"param_overrides.pb_aperture_sizing_mult": 0.85}),
    )
    return [
        _child(parent, name, delta, "management")
        for parent in parents
        for name, delta in changes
    ]


def _parity_contract(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    settings_fields = {field.name for field in fields(StrategySettings)}
    missing: set[str] = set()
    noncausal: list[str] = []
    for candidate in candidates:
        for key, value in candidate["mutations"].items():
            if not key.startswith("param_overrides.pb_"):
                continue
            field_name = key.removeprefix("param_overrides.")
            if field_name not in settings_fields:
                missing.add(field_name)
            if field_name == "pb_open_scored_fill_timing" and value != "next_5m_open":
                noncausal.append(f"{candidate['id']}:{value}")
    live_source = (REPO_ROOT / "strategies/stock/iaric/engine.py").read_text(encoding="utf-8")
    replay_source = (
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_intraday_hybrid_engine.py"
    ).read_text(encoding="utf-8")
    shared_call = "iaric_core_logic.advance_aperture_route"
    residual_call = "causal_relative_dislocation_atr"
    issuer_call = "issuer_batch_arbitration"
    result = {
        "passed": (
            not missing
            and not noncausal
            and shared_call in live_source
            and shared_call in replay_source
            and residual_call in live_source
            and residual_call in replay_source
            and issuer_call in live_source
            and issuer_call in replay_source
        ),
        "typed_settings_missing": sorted(missing),
        "noncausal_fill_contracts": noncausal,
        "shared_entry_decision_owner": "strategies.stock.iaric.core.logic.advance_aperture_route",
        "live_adapter_uses_shared_owner": shared_call in live_source,
        "backtest_adapter_uses_shared_owner": shared_call in replay_source,
        "causal_residual_live_replay_parity": (
            residual_call in live_source and residual_call in replay_source
        ),
        "issuer_arbitration_live_replay_parity": (
            issuer_call in live_source and issuer_call in replay_source
        ),
        "signal_fill_contract": "completed bar t may fill no earlier than t+1; pre-existing limits only",
    }
    if not result["passed"]:
        raise RuntimeError(f"live/backtest parity contract failed: {result}")
    return result


def _freeze_baseline(control: dict[str, Any], baseline_path: Path, output: Path, args: argparse.Namespace) -> None:
    observed = control["metrics"]
    comparisons: dict[str, Any] = {}
    passed = True
    baseline_metrics_path = output / "baseline_metrics.json"
    if args.start_date == START_DATE and args.end_date == END_DATE and baseline_metrics_path.exists():
        expected = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        for key, tolerance in {
            "total_trades": 0.0,
            "expected_total_r": 1e-6,
            "avg_r": 1e-8,
            "profit_factor": 1e-8,
            "max_drawdown_pct": 1e-8,
        }.items():
            delta = abs(float(observed.get(key, 0.0)) - float(expected.get(key, 0.0)))
            comparisons[key] = {
                "expected": expected.get(key),
                "observed": observed.get(key),
                "absolute_delta": delta,
                "tolerance": tolerance,
                "passed": delta <= tolerance,
            }
            passed = passed and delta <= tolerance
    payload = {
        "passed": passed,
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "comparisons": comparisons,
    }
    _write_json(output / "baseline_contract.json", payload)
    if not passed:
        raise RuntimeError("Round 3 baseline drifted; Round 4 search stopped before optimization")


def _gates(row: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    metrics = row["metrics"]
    base = control["metrics"]
    focus = row["focus"]
    folds = row.get("folds", [])
    locked_fold = next((fold for fold in folds if fold.get("fold") == "latest"), None)
    delta_n = float(metrics.get("total_trades", 0.0)) - float(
        base.get("total_trades", 0.0)
    )
    delta_r = float(metrics.get("expected_total_r", 0.0)) - float(
        base.get("expected_total_r", 0.0)
    )
    score_audit = row.get("round4_score_audit", {})
    issuer_neutral_delta_r = (
        float((score_audit.get("candidate_issuer") or {}).get("issuer_neutral_total_r", 0.0))
        - float((score_audit.get("control_issuer") or {}).get("issuer_neutral_total_r", 0.0))
    )
    base_issuer_share = float(control["issuer"]["top_positive_issuer_share"])
    return {
        "fold_and_holdout_integrity": bool(row.get("validation_contract", {}).get("passed"))
        and not bool(row.get("validation_contract", {}).get("holdout_accessed")),
        "total_r_created_5r": delta_r >= 5.0,
        "frequency_uplift_15pct": float(metrics.get("total_trades", 0.0))
        >= 1.15 * float(base.get("total_trades", 0.0)),
        "marginal_expectancy_008r": delta_n > 0.0 and delta_r / delta_n >= 0.08,
        "portfolio_quality": float(metrics.get("avg_r", 0.0)) >= 0.20
        and float(metrics.get("profit_factor", 0.0)) >= 1.55,
        "bounded_drawdown_45pct": float(metrics.get("max_drawdown_pct", 1.0)) <= 0.045,
        "fixed_atlas_recall_at_least_25pct": fixed_atlas_recall(metrics, base) >= 0.25,
        "discrimination_at_least_030r": float(
            metrics.get("entry_realized_discrimination_lift_r", 0.0)
        )
        >= 0.30,
        "issuer_neutral_alpha_2r": issuer_neutral_delta_r >= 2.0,
        "issuer_concentration_reduced": float(row["issuer"]["top_positive_issuer_share"])
        <= min(0.45, base_issuer_share - 0.02),
        "focus_route_real_alpha": int(focus["trades"]) >= 50
        and float(focus["total_r"]) >= 1.0
        and float(focus["profit_factor"]) >= 1.20,
        "focus_route_not_one_issuer": float(focus["issuer"]["top_positive_issuer_share"])
        <= 0.50,
        "chronological_consistency": len(folds) == 3
        and sum(float(fold["delta_total_r"]) > 0.0 for fold in folds) >= 2
        and min((float(fold["delta_total_r"]) for fold in folds), default=-99.0) >= -2.0,
        "locked_internal_validation_positive": locked_fold is not None
        and float(locked_fold.get("delta_total_r", -99.0)) > 0.0,
    }


def _diagnostics(selected: dict[str, Any], control: dict[str, Any], status: str) -> str:
    sm, bm = selected["metrics"], control["metrics"]
    raw = selected["round4_score_raw"]
    score_audit = selected.get("round4_score_audit", {})
    issuer_neutral_delta = (
        float((score_audit.get("candidate_issuer") or {}).get("issuer_neutral_total_r", 0.0))
        - float((score_audit.get("control_issuer") or {}).get("issuer_neutral_total_r", 0.0))
    )
    delta_trades = float(sm.get("total_trades", 0.0)) - float(bm.get("total_trades", 0.0))
    delta_r = float(sm.get("expected_total_r", 0.0)) - float(bm.get("expected_total_r", 0.0))
    lines = [
        "IARIC ROUND 4 — REAL-ALPHA PHASED AUTO FINAL DIAGNOSTICS",
        "=" * 76,
        f"Status: {status}",
        f"Selected research candidate: {selected['id']}",
        f"Training only: {START_DATE} through {END_DATE}; sealed holdout begins {HOLDOUT_START}",
        "",
        "OUTCOME",
        f"  Trades: {bm.get('total_trades', 0):.0f} -> {sm.get('total_trades', 0):.0f} ({delta_trades:+.0f})",
        f"  Expected total R: {bm.get('expected_total_r', 0):+.3f} -> {sm.get('expected_total_r', 0):+.3f} ({delta_r:+.3f})",
        f"  Avg R: {bm.get('avg_r', 0):+.4f} -> {sm.get('avg_r', 0):+.4f}",
        f"  PF: {bm.get('profit_factor', 0):.3f} -> {sm.get('profit_factor', 0):.3f}",
        f"  Max DD: {bm.get('max_drawdown_pct', 0):.3%} -> {sm.get('max_drawdown_pct', 0):.3%}",
        f"  Recall: {bm.get('entry_opportunity_recall', 0):.3%} -> {sm.get('entry_opportunity_recall', 0):.3%}",
        f"  Realized discrimination: {bm.get('entry_realized_discrimination_lift_r', 0):+.3f}R -> {sm.get('entry_realized_discrimination_lift_r', 0):+.3f}R",
        f"  Issuer-neutral delta R: {issuer_neutral_delta:+.3f}",
        f"  Focus route: {selected['focus']['family']}, n={selected['focus']['trades']}, totalR={selected['focus']['total_r']:+.3f}, PF={selected['focus']['profit_factor']:.3f}",
        "",
        "PROMOTION GATES",
    ]
    lines.extend(f"  [{'PASS' if passed else 'FAIL'}] {name}" for name, passed in selected["gates"].items())
    lines.extend(("", "CHRONOLOGICAL FOLDS"))
    for fold in selected.get("folds", []):
        lines.append(
            f"  {fold['fold']}: deltaR={fold['delta_total_r']:+.3f}, delta trades={fold['delta_trades']:+.0f}"
        )
    lines.extend(("", "IMMUTABLE SCORE — EXACTLY 7 COMPONENTS"))
    for name, spec in SCORE_SPEC.items():
        lines.append(
            f"  {name}: weight={spec['weight']:.2f}, scale={spec['scale']:.4g}, raw={raw[name]:+.4f}"
        )
    lines.extend(("", "SELECTED MUTATIONS", json.dumps(selected["mutations"], indent=2, sort_keys=True)))
    if status != "complete_value_verified":
        lines.extend(
            (
                "",
                "PROMOTION DECISION",
                "  The research candidate is retained for diagnosis, but the canonical Round 3/live",
                "  configuration remains unchanged because at least one pre-declared gate failed.",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"end-date must precede sealed holdout {HOLDOUT_START}")
    if int(args.max_workers) != 2:
        raise ValueError("Round 4 must run with max-workers=2")
    if len(SCORE_SPEC) != 7 or len(SCORE_COMPONENTS) != 7:
        raise ValueError("Round 4 requires exactly seven optimizer and causal signal components")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atlas_path = Path(args.opportunity_atlas_summary) if args.opportunity_atlas_summary else (
        output / "phase_0_opportunity_atlas" / "atlas_summary.json"
    )
    if not atlas_path.exists():
        _write_json(output / "representative_baseline_blocker.json", {
            "status": "blocked_missing_authority_preflight",
            "blockers": [f"missing authority/atlas summary: {atlas_path.resolve()}"],
            "holdout_accessed": False,
        })
        return 2
    atlas_preflight = json.loads(atlas_path.read_text(encoding="utf-8"))
    representative = assess_atlas_for_optimization(atlas_preflight)
    if not representative["passed"]:
        _write_json(output / "representative_baseline_blocker.json", {
            "status": "blocked_representative_pipeline_contract",
            "representative_contract_version": CONTRACT_VERSION,
            "assessment": representative,
            "input_authority": atlas_preflight.get("input_authority", {}),
            "blockers": representative["blockers"],
            "holdout_accessed": False,
        })
        return 2

    # Only incur replay loading and migration after the inexpensive authority,
    # chronology and mechanism-pipeline checks pass.
    baseline_path = Path(args.baseline_config)
    latest_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = _architectural_baseline(latest_baseline)
    migrated_baseline_path = output / "architectural_baseline_config.json"
    _write_json(migrated_baseline_path, baseline)
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    atlas_summary = _load_atlas_summary(atlas_path, source_fingerprint)
    selection_args = argparse.Namespace(**vars(args))
    selection_args.end_date = CALIBRATION_END

    phase1_candidates = _supply_candidates(
        baseline,
        atlas_summary=atlas_summary,
        smoke=args.smoke,
    )
    parity = _parity_contract(phase1_candidates)
    _write_json(output / "parity_contract.json", parity)
    _write_json(
        output / "run_spec.json",
        {
            "objective": "increase executable reversion alpha and frequency without fitting a threshold grid",
            "baseline": str(baseline_path.resolve()),
            "architectural_baseline": str(migrated_baseline_path.resolve()),
            "baseline_migration": {
                "latest_optimized_lineage_preserved": True,
                "incumbent_score_floors_and_profiles_preserved": True,
                "score_floor_rescaling_from_intuition_forbidden": True,
                "issuer_event_identity_invariant": True,
                "positive_room_and_prospective_rr_invariants": True,
                "incumbent_tail_management_preserved": True,
            },
            "chronology": chronology_contract(),
            "representative_contract_version": CONTRACT_VERSION,
            "max_workers": 2,
            "opportunity_atlas_summary": str(atlas_path.resolve()),
            "opportunity_atlas_code_fingerprint": atlas_summary.get("code_fingerprint"),
            "outcome_blind_family_floors": {
                candidate.get("focus_family"): _parse_map(
                    candidate["mutations"].get("param_overrides.pb_aperture_family_score_floors", "")
                ).get(str(candidate.get("focus_family", "")), "")
                for candidate in phase1_candidates
                if candidate.get("focus_family")
            },
            "score_spec": SCORE_SPEC,
            "score_component_count": len(SCORE_SPEC),
            "causal_event_score_component_count": len(SCORE_COMPONENTS),
            "phase_order": list(PHASE_ORDER),
            "experiment_registry": EXPERIMENT_REGISTRY,
            "selection_end_date": CALIBRATION_END,
            "locked_validation_used_for_candidate_ranking": False,
            "anti_overfit_contract": {
                "bounded_mechanism_presets": True,
                "no_learned_score_scales": True,
                "controls_carried_each_phase": True,
                "issuer_share_classes_combined": True,
                "fixed_opportunity_atlas_denominator": True,
                "parent_relative_activation_and_cannibalization": True,
                "failed_gates_block_promotion": True,
            },
            "parity_contract": parity,
            "source_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
        },
    )

    reference_payload = json.loads(Path(args.baseline_reference).read_text(encoding="utf-8"))
    reference_control = reference_payload.get("control", reference_payload)
    phase0 = _evaluate_round4(
        "phase_0_repaired_baseline_incidence",
        [_candidate(
            "incumbent_control",
            baseline,
            stage="repaired_baseline_incidence",
            families=sorted(BASE_FAMILIES),
        )],
        args=selection_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=None,
    )
    phase0_control = phase0[0]
    incidence_contract = _baseline_incidence_contract(phase0_control, reference_control)
    _write_json(output / "baseline_incidence_contract.json", incidence_contract)
    if not incidence_contract["passed"]:
        raise RuntimeError(
            "repaired baseline changed incumbent signal incidence; structural search blocked"
        )

    phase1 = _evaluate_round4(
        "phase_1_signal_supply",
        phase1_candidates,
        args=selection_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=None,
    )
    control = next(row for row in phase1 if row["id"] == "incumbent_control")
    _freeze_baseline(control, migrated_baseline_path, output, selection_args)
    baseline_metrics_path = output / "baseline_metrics.json"
    if not baseline_metrics_path.exists():
        _write_json(baseline_metrics_path, control["metrics"])
    if args.smoke:
        _write_json(output / "smoke_summary.json", phase1)
        return 0

    parents1 = _shortlist(phase1, control, 2)
    if not parents1:
        raise RuntimeError("No added signal family produced positive executable focus-route alpha")
    _write_json(output / "phase_1_survivors.json", parents1)

    phase2 = _evaluate_round4(
        "phase_2_discrimination",
        _discrimination_candidates(parents1),
        args=selection_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    parents2 = _shortlist(phase2, control, 2)
    if not parents2:
        raise RuntimeError("Discrimination phase lost every positive parent control")

    phase3 = _evaluate_round4(
        "phase_3_entry",
        _entry_candidates(parents2),
        args=selection_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    parents3 = _shortlist(phase3, control, 2)
    if not parents3:
        raise RuntimeError("Entry phase lost every positive parent control")

    phase4 = _evaluate_round4(
        "phase_4_management_exit",
        _management_candidates(parents3),
        args=selection_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control=control,
    )
    finalists = _shortlist(phase4, control, 3)
    if not finalists:
        raise RuntimeError("Management phase lost every positive parent control")
    selected = max(
        finalists,
        key=lambda row: (
            float(row["round4_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
        ),
    )
    checkpoint = {
        "status": "phase4_checkpoint_complete",
        "selected": selected,
        "control": control,
        "holdout_accessed": False,
        "promotion_eligible": False,
        "reason": "chronological validation and promotion occur only after phases 5-11",
    }
    _write_json(output / "phase_4_checkpoint_selection.json", checkpoint)
    _write_json(output / "phase_4_continuation_baseline_config.json", selected["mutations"])
    _write_json(
        output / "progress.json",
        {
            "status": "phase4_checkpoint_complete",
            "selected_id": selected["id"],
            "last_completed_phase": "phase_4_trade_management_and_exit_checkpoint",
            "next_phase": "phase_5_fixed_family_archetype_calibration",
            "holdout_accessed": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
