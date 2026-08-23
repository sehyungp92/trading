"""Continue IARIC Round 3 beyond the narrow 149-trade local optimum.

This is a cache-preserving synthesis and breadth-repair search.  It starts from
the completed 133/149-trade value anchors, recombines independently positive
reversion routes, repairs the retained 164/192-trade breadth parents, and only
promotes a replacement Round 3 after the enabled full-period, attribution,
concentration, risk, and live/replay-contract gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.runners import run_iaric_escape_round3 as escape
from strategies.stock.iaric.core.logic import aperture_family_from_route


REPO_ROOT = Path(__file__).resolve().parents[4]
IARIC_DIR = REPO_ROOT / "backtests/output/stock/iaric"
ROUND_DIR = IARIC_DIR / "round_3"
RESEARCH_DIR = ROUND_DIR / "research/structural_challenger"
OUTPUT_DIR = ROUND_DIR / "research/alpha_escape_continuation"
MANIFEST_PATH = IARIC_DIR / "rounds_manifest.json"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
HOLDOUT_START = "2026-03-02"
FREQUENCY_TARGET_TRADES = 180
MINIMUM_FREQUENCY_ESCAPE_FRACTION = 0.10

# Exactly seven immutable components.  The centers express the intended
# aggressive-but-not-reckless frontier; they are not fitted to candidate
# outcomes.  Frequency and absolute total R jointly receive 56% of the score.
SCORE_SPEC: dict[str, dict[str, float]] = {
    "absolute_total_r": {"weight": 0.28, "center": 47.0, "scale": 12.0},
    "absolute_trades": {"weight": 0.28, "center": 180.0, "scale": 40.0},
    "average_r": {"weight": 0.10, "center": 0.18, "scale": 0.12},
    "profit_factor": {"weight": 0.08, "center": 1.60, "scale": 0.40},
    "inverse_drawdown": {"weight": 0.12, "center": 0.045, "scale": 0.020},
    "positive_route_breadth": {"weight": 0.07, "center": 3.0, "scale": 1.5},
    "robust_average_r": {"weight": 0.07, "center": 0.10, "scale": 0.12},
}

MAPPING_KEYS = {
    "floor": "param_overrides.pb_aperture_family_score_floors",
    "filter": "param_overrides.pb_aperture_family_filters",
    "cap": "param_overrides.pb_aperture_family_daily_caps",
    "transition": "param_overrides.pb_aperture_family_transitions",
    "max_bar": "param_overrides.pb_aperture_family_max_bars",
    "hybrid_policy": "param_overrides.pb_aperture_family_hybrid_next_policies",
    "score_profile": "param_overrides.pb_aperture_family_score_profiles",
    "max_events": "param_overrides.pb_aperture_family_max_events",
}
NEW_FILTERS = {"deep_reclaim", "residual_reclaim", "room_reclaim"}
ORCHESTRATION_COMPATIBLE_CODE_FINGERPRINTS = {
    # Attempt 3 added the opt-in quality-hybrid entry and explicit no-fold
    # orchestration.  The strategy/core files are unchanged by the adaptive
    # phase redesign below, so all 343 completed decisions remain identical.
    "61e864e67f23234503807cc2c6c9e0d84b16d97937bea4e24c0946212e4db3e5",
    # Attempt 4 completed candidate decisions under the original single-rank
    # finalizer. Splitting exploration from deployment ranking below cannot
    # change a replay decision, so its cache is safe to retain on restart.
    "c511acd8a2ac406b01f2a6615e5be9db29087a0a4009bf074e325bc13d6d0523",
}
FAMILY_SCORE_PROFILES = {
    "GAP_EXHAUSTION_RECLAIM": "shock_exhaustion",
    "GAP_FILL_RECLAIM": "shock_exhaustion",
    "FAILED_BREAKDOWN_RECLAIM": "level_reclaim",
    "MULTIDAY_HIGHER_LOW_RECLAIM": "trend_pullback",
    "PRIOR_DAY_LOW_RECLAIM": "level_reclaim",
    "UPTREND_PULLBACK_RECLAIM": "trend_pullback",
    "VOLUME_CLIMAX_RECLAIM": "shock_exhaustion",
    "VWAP_DEVIATION_RECLAIM": "level_reclaim",
}
PRIMARY_PROFILES: dict[str, dict[str, Any]] = {
    "mhl": {
        "family": "MULTIDAY_HIGHER_LOW_RECLAIM",
        "floor": 65,
        "transition": "next_bar",
    },
    "pdl": {
        "family": "PRIOR_DAY_LOW_RECLAIM",
        "transition": "next_bar",
    },
    "gap": {
        "family": "GAP_FILL_RECLAIM",
        "floor": 65,
    },
    "vwap_confirm": {
        "family": "VWAP_DEVIATION_RECLAIM",
        "floor": 75,
        "transition": "confirm",
    },
    "vwap_next": {
        "family": "VWAP_DEVIATION_RECLAIM",
        "floor": 75,
        "transition": "next_bar",
    },
    "vwap_hybrid_deep": {
        "family": "VWAP_DEVIATION_RECLAIM",
        "floor": 75,
        "transition": "quality_hybrid",
        "hybrid_policy": "deep_reclaim",
    },
    "vwap_hybrid_residual": {
        "family": "VWAP_DEVIATION_RECLAIM",
        "floor": 75,
        "transition": "quality_hybrid",
        "hybrid_policy": "residual_reclaim",
    },
    "vwap_hybrid_room": {
        "family": "VWAP_DEVIATION_RECLAIM",
        "floor": 75,
        "transition": "quality_hybrid",
        "hybrid_policy": "room_reclaim",
    },
    "volume": {
        "family": "VOLUME_CLIMAX_RECLAIM",
        "floor": 75,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=str(ROUND_DIR))
    parser.add_argument("--research-dir", default=str(RESEARCH_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument(
        "--skip-fold-validation",
        action="store_true",
        help="Skip chronological fold evaluations while retaining full-period structural/economic gates.",
    )
    return parser.parse_args()


def _load(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        if default is not None:
            return default
        raise


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(escape._code_fingerprint().encode("ascii"))
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "strategies/stock/iaric/core/logic.py",
        REPO_ROOT / "strategies/stock/iaric/config.py",
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
        REPO_ROOT / "strategies/stock/iaric/engine.py",
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_intraday_hybrid_engine.py",
        REPO_ROOT / "backtests/stock/auto/iaric/worker.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_count(output: Path) -> int:
    total = 0
    for name in ("evaluation_cache.json", "structural_screen_cache.json"):
        payload = _load(output / name, {})
        total += len(payload.get("evaluations", {})) if isinstance(payload, dict) else 0
    return total


def _progress(output: Path, status: str, **extra: Any) -> None:
    _write(output / "queue_status.json", {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cached_evaluations": _cache_count(output),
        **extra,
    })


def _prepare_compatible_cache(
    research: Path,
    output: Path,
    *,
    source_fingerprint: str,
    code_fingerprint: str,
) -> dict[str, Any]:
    """Re-key only rows proven decision-identical under opt-in extensions."""

    report: dict[str, Any] = {"source": str(research), "caches": {}}
    for name in ("evaluation_cache.json", "structural_screen_cache.json"):
        target = output / name
        if target.exists():
            payload = _load(target)
        else:
            source = research / name
            if not source.exists():
                continue
            shutil.copy2(source, target)
            payload = _load(target)
        if payload.get("source_fingerprint") != source_fingerprint:
            raise ValueError(f"{name} source fingerprint is incompatible")
        old_code = str(payload.get("code_fingerprint", ""))
        evaluations = dict(payload.get("evaluations", {}))
        if old_code != code_fingerprint:
            orchestration_only = old_code in ORCHESTRATION_COMPATIBLE_CODE_FINGERPRINTS
            migrated: dict[str, Any] = {}
            for key, row in evaluations.items():
                mutations = dict(row.get("mutations", {}))
                transition_text = str(mutations.get(MAPPING_KEYS["transition"], ""))
                if (
                    MAPPING_KEYS["hybrid_policy"] in mutations
                    or "quality_hybrid" in transition_text
                ) and not orchestration_only:
                    raise ValueError(
                        f"{name} already contains quality-hybrid rows; compatibility is unproven"
                    )
                parts = key.split("|")
                if len(parts) != 5 or parts[1] != old_code:
                    raise ValueError(f"unexpected cache namespace in {name}: {key}")
                parts[1] = code_fingerprint
                migrated["|".join(parts)] = row
            backup = output / f"{name}.pre_alpha_escape_extension.json"
            if not backup.exists():
                shutil.copy2(target, backup)
            payload["evaluations"] = migrated
            payload["code_fingerprint"] = code_fingerprint
            payload["compatibility_migration"] = {
                "from": old_code,
                "to": code_fingerprint,
                "proof": (
                    "runner-only adaptive phase redesign; strategy/core decision files are "
                    "unchanged, so quality-hybrid and all earlier cached rows are decision-identical"
                    if orchestration_only
                    else "quality_hybrid is opt-in; every migrated row lacks that transition "
                    "and its policy. The fold-skip control is orchestration-only and does "
                    "not change any cached full-period candidate decision."
                ),
            }
            _write(target, payload)
        report["caches"][name] = {
            "evaluations": len(payload.get("evaluations", {})),
            "from_code_fingerprint": old_code,
            "to_code_fingerprint": code_fingerprint,
        }
    return report


def _mapping(raw: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "="
        key, value = token.split(separator, 1)
        result[key.strip().upper()] = value.strip().lower()
    return result


def _set_mapping(
    mutations: dict[str, Any], key: str, family: str, value: Any | None,
) -> dict[str, Any]:
    result = deepcopy(mutations)
    mappings = _mapping(result.get(key, ""))
    family_key = str(family).upper()
    if value is None:
        mappings.pop(family_key, None)
    else:
        mappings[family_key] = str(value).lower()
    if mappings:
        result[key] = ",".join(f"{name}:{item}" for name, item in sorted(mappings.items()))
    else:
        result.pop(key, None)
    return result


def _families(mutations: dict[str, Any]) -> set[str]:
    return {
        value.strip().upper()
        for value in str(mutations.get("param_overrides.pb_aperture_families", "")).split(",")
        if value.strip()
    }


def _apply_profiles(
    mutations: dict[str, Any], names: Iterable[str], *, candidate_size: float | None = None,
) -> dict[str, Any]:
    result = deepcopy(mutations)
    families = _families(result)
    for name in names:
        profile = PRIMARY_PROFILES[name]
        family = str(profile["family"])
        families.add(family)
        for policy in (
            "floor", "filter", "cap", "transition", "max_bar", "hybrid_policy"
        ):
            if policy in profile:
                result = _set_mapping(result, MAPPING_KEYS[policy], family, profile[policy])
    result.update({
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_max_symbols": 120,
        "param_overrides.pb_aperture_families": ",".join(sorted(families)),
    })
    if candidate_size is not None:
        result["param_overrides.pb_aperture_sizing_mult"] = float(candidate_size)
    return result


def _candidate(candidate_id: str, mutations: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return escape._candidate(candidate_id, mutations, **meta)


def _family_stats(row: dict[str, Any], family: str) -> dict[str, float]:
    values = [
        float(trade.get("r", 0.0))
        for trade in row.get("trade_attribution", [])
        if aperture_family_from_route(str(trade.get("route", ""))) == family
    ]
    wins = sum(max(value, 0.0) for value in values)
    losses = -sum(min(value, 0.0) for value in values)
    return {
        "trades": float(len(values)),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": wins / losses if losses else (99.0 if wins else 0.0),
    }


def _positive_route_count(row: dict[str, Any]) -> int:
    return sum(
        int(stats.get("trades", 0)) >= 3 and float(stats.get("total_r", 0.0)) > 0.0
        for stats in row.get("aperture", {}).get("routes", {}).values()
    )


def _score(row: dict[str, Any]) -> tuple[float, dict[str, float], dict[str, float]]:
    metrics = row["metrics"]
    raw = {
        "absolute_total_r": float(metrics.get("expected_total_r", 0.0)),
        "absolute_trades": float(metrics.get("total_trades", 0.0)),
        "average_r": float(metrics.get("avg_r", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "inverse_drawdown": float(metrics.get("max_drawdown_pct", 1.0)),
        "positive_route_breadth": float(_positive_route_count(row)),
        "robust_average_r": float(metrics.get("robust_avg_r", 0.0)),
    }
    components: dict[str, float] = {}
    for name, spec in SCORE_SPEC.items():
        if name == "inverse_drawdown":
            z_value = (spec["center"] - raw[name]) / spec["scale"]
        else:
            z_value = (raw[name] - spec["center"]) / spec["scale"]
        components[name] = 0.5 + 0.5 * math.tanh(z_value)
    score = sum(SCORE_SPEC[name]["weight"] * components[name] for name in SCORE_SPEC)
    return score, components, raw


def _rescore(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        score, components, raw = _score(row)
        row["alpha_escape_score"] = score
        row["alpha_escape_score_components"] = components
        row["alpha_escape_score_raw"] = raw
        row["positive_route_count"] = _positive_route_count(row)
    return sorted(rows, key=lambda row: (
        float(row["alpha_escape_score"]),
        float(row["metrics"].get("expected_total_r", -1e9)),
        float(row["metrics"].get("total_trades", 0.0)),
        -float(row["metrics"].get("max_drawdown_pct", 1.0)),
    ), reverse=True)


def _minimum_frequency_escape(current: dict[str, Any]) -> int:
    """Require a material step out of the incumbent, without fitting to 180."""

    current_trades = int(current.get("metrics", {}).get("total_trades", 0))
    return math.ceil(current_trades * (1.0 + MINIMUM_FREQUENCY_ESCAPE_FRACTION))


def _deployment_value_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Rank deployable value separately from the breadth-seeking search score.

    Frequency remains economically important, but it is a secondary outcome:
    a wider candidate must first convert its opportunities into total R while
    retaining the portfolio quality/risk floors. The immutable seven-part
    score continues to drive exploration and Pareto parent preservation.
    """

    metrics = row.get("metrics", {})
    quality_floor = (
        float(metrics.get("avg_r", 0.0)) >= 0.18
        and float(metrics.get("profit_factor", 0.0)) >= 1.60
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.045
        and float(metrics.get("robust_avg_r", 0.0)) >= 0.0
    )
    return (
        quality_floor,
        float(metrics.get("expected_total_r", -1e9)),
        float(metrics.get("profit_factor", 0.0)),
        -float(metrics.get("max_drawdown_pct", 1.0)),
        int(metrics.get("total_trades", 0)),
        float(metrics.get("avg_r", 0.0)),
        float(row.get("alpha_escape_score", -99.0)),
    )


def _promotion_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Promotion is gate-first and value-first; exploration score is last."""

    return (
        bool(row.get("all_gates_pass")),
        *_deployment_value_key(row),
    )


def _evaluate_stage(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    control_metrics: dict[str, Any],
    incumbent_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows = escape._evaluate(
        stage,
        candidates,
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=control_metrics,
    )
    rows = _rescore(rows)
    _write(output / f"{stage}_results.json", rows)
    global_rows = _rescore([*list(incumbent_rows), *rows])
    exploration_best = global_rows[0] if global_rows else None
    deployment_best = max(global_rows, key=_deployment_value_key) if global_rows else None
    _progress(
        output,
        "running",
        last_completed_stage=stage,
        evaluated=len(rows),
        stage_best_id=rows[0]["id"] if rows else None,
        stage_best_metrics=rows[0]["metrics"] if rows else {},
        # Keep the legacy fields as exploration aliases for existing monitors.
        best_id=exploration_best["id"] if exploration_best else None,
        best_metrics=exploration_best["metrics"] if exploration_best else {},
        exploration_best_id=exploration_best["id"] if exploration_best else None,
        exploration_best_metrics=exploration_best["metrics"] if exploration_best else {},
        deployment_best_id=deployment_best["id"] if deployment_best else None,
        deployment_best_metrics=deployment_best["metrics"] if deployment_best else {},
    )
    return rows


def _required_row(rows: Iterable[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    match = next((row for row in rows if row.get("id") == candidate_id), None)
    if match is None:
        raise RuntimeError(f"required continuation row is missing: {candidate_id}")
    return match


def _add_gap_exhaustion_floor75(mutations: dict[str, Any]) -> dict[str, Any]:
    result = _apply_profiles(mutations, (), candidate_size=0.55)
    family = "GAP_EXHAUSTION_RECLAIM"
    families = _families(result) | {family}
    result["param_overrides.pb_aperture_families"] = ",".join(sorted(families))
    return _set_mapping(result, MAPPING_KEYS["floor"], family, 75)


def _evidence_synthesis_candidates(
    core_rows: list[dict[str, Any]], atom_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compose only sleeves with positive executable route evidence."""

    leader174 = _required_row(core_rows, "improved133__union__mhl_pdl_vwap_next_gap")
    return159 = _required_row(core_rows, "improved133__union__mhl_pdl_vwap_confirm_gap")
    no_vwap160 = _required_row(core_rows, "improved133__union__mhl_pdl_gap")
    breadth196 = _required_row(core_rows, "course164__union__mhl_pdl_vwap_next")
    repair139 = _required_row(
        atom_rows, "course164__repair__gap_exhaustion_reclaim__floor_75"
    )
    gap_evidence = _family_stats(repair139, "GAP_EXHAUSTION_RECLAIM")
    if (
        gap_evidence["trades"] < 5
        or gap_evidence["total_r"] <= 0.0
        or gap_evidence["profit_factor"] < 1.10
    ):
        raise RuntimeError("gap-exhaustion floor-75 sleeve lacks positive executable evidence")

    def gap_candidate(parent: dict[str, Any]) -> dict[str, Any]:
        return _candidate(
            parent["id"] + "__add_gap_exhaustion_floor75",
            _add_gap_exhaustion_floor75(parent["mutations"]),
            stage="evidence_backed_sleeve_synthesis",
            parent_id=parent["id"],
            target_family="GAP_EXHAUSTION_RECLAIM",
            activation_evidence=gap_evidence,
            hypothesis="add a positive filtered sleeve without reopening its negative tail",
        )

    def quality_stack(profiles: tuple[str, ...]) -> dict[str, Any]:
        return _candidate(
            repair139["id"] + "__positive_stack__" + "_".join(profiles),
            _apply_profiles(repair139["mutations"], profiles, candidate_size=0.55),
            stage="quality_parent_positive_sleeve_synthesis",
            parent_id=repair139["id"],
            profiles=list(profiles),
            structural_sources=["quality_repair_parent", "independently_positive_sleeves"],
        )

    # Every two-candidate wave spans a different parent role.  This preserves
    # the quality sleeve even if the broad-parent branch is stopped early.
    candidates = [
        gap_candidate(leader174),
        quality_stack(("mhl", "pdl", "gap")),
        gap_candidate(breadth196),
        gap_candidate(no_vwap160),
        gap_candidate(return159),
        quality_stack(("mhl", "pdl")),
    ]
    return escape._dedupe(candidates)


def _apply_family_score_profiles(mutations: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(mutations)
    for family in sorted(_families(result)):
        profile = FAMILY_SCORE_PROFILES.get(family)
        if profile:
            result = _set_mapping(
                result, MAPPING_KEYS["score_profile"], family, profile
            )
    return result


def _structural_parent_beam(rows: Iterable[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Preserve broad leaders and high-value sleeves below the old 150-trade gate."""

    viable = [
        row for row in rows
        if float(row.get("metrics", {}).get("expected_total_r", -99.0)) >= 43.0
        and float(row.get("metrics", {}).get("profit_factor", 0.0)) >= 1.58
        and float(row.get("metrics", {}).get("max_drawdown_pct", 1.0)) <= 0.045
        and int(row.get("metrics", {}).get("total_trades", 0)) >= 130
    ]
    if not viable:
        return []
    selectors = (
        lambda row: float(row.get("alpha_escape_score", -99.0)),
        lambda row: float(row["metrics"].get("avg_r", -99.0)),
        lambda row: float(row["metrics"]["expected_total_r"]),
        lambda row: (
            float(row["metrics"]["total_trades"])
            if float(row["metrics"]["expected_total_r"]) >= 47.0
            else -1e9
        ),
    )
    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for selector in selectors:
        row = max(viable, key=selector)
        signature = escape._signature(row["mutations"])
        if signature not in signatures:
            signatures.add(signature)
            selected.append(row)
    return selected[:limit]


def _profile_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in _structural_parent_beam(rows, 3):
        mutations = _apply_family_score_profiles(parent["mutations"])
        if escape._signature(mutations) == escape._signature(parent["mutations"]):
            continue
        candidates.append(_candidate(
            parent["id"] + "__family_score_profiles",
            mutations,
            stage="seven_component_family_score_calibration",
            parent_id=parent["id"],
            score_component_count=7,
            structural_sources=["fixed_economic_family_archetypes", "completed_bar_components"],
        ))
    return escape._dedupe(candidates)


def _quality_policy_passes(components: dict[str, Any], policy: str) -> bool:
    reclaim = float(components.get("reclaim", 0.0))
    close_quality = float(components.get("close_quality", 0.0))
    if policy == "deep_reclaim":
        return (
            float(components.get("dislocation", 0.0)) >= 0.40
            and reclaim >= 0.35
            and close_quality >= 0.55
        )
    if policy == "residual_reclaim":
        return (
            float(components.get("residual_dislocation", 0.0)) >= 0.35
            and reclaim >= 0.35
            and close_quality >= 0.55
        )
    if policy == "room_reclaim":
        return (
            float(components.get("reversion_room", 0.0)) >= 0.30
            and reclaim >= 0.35
            and close_quality >= 0.55
        )
    raise ValueError(f"unknown activation policy {policy!r}")


def _hybrid_activation_report(parent: dict[str, Any], family: str) -> dict[str, Any]:
    trades = [
        trade for trade in parent.get("trade_attribution", [])
        if aperture_family_from_route(str(trade.get("route", ""))) == family
    ]
    policies: dict[str, Any] = {}
    fingerprints: dict[tuple[bool, ...], list[str]] = defaultdict(list)
    for policy in ("deep_reclaim", "residual_reclaim", "room_reclaim"):
        mask = tuple(
            _quality_policy_passes(dict(trade.get("score_components", {})), policy)
            for trade in trades
        )
        fingerprints[mask].append(policy)
        passed = sum(mask)
        policies[policy] = {
            "observed": len(mask),
            "passed": passed,
            "pass_rate": passed / len(mask) if mask else 0.0,
            "non_degenerate": bool(mask) and 0 < passed < len(mask),
        }
    return {
        "parent_id": parent.get("id"),
        "family": family,
        "policies": policies,
        "distinct_policy_masks": len(fingerprints),
        "duplicate_policy_groups": [values for values in fingerprints.values() if len(values) > 1],
        "residual_component_available": any(
            float(dict(trade.get("score_components", {})).get("residual_dislocation", 0.0)) > 0.0
            for trade in trades
        ),
        "decision": "exclude_degenerate_and_residual_dependent_policies_from_next_phase",
    }


def _rearm_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    beam = _structural_parent_beam(rows, 1)
    if not beam:
        return []
    parent = beam[0]
    candidates: list[dict[str, Any]] = []
    for family in ("FAILED_BREAKDOWN_RECLAIM", "PRIOR_DAY_LOW_RECLAIM"):
        stats = _family_stats(parent, family)
        if family not in _families(parent["mutations"]):
            continue
        if stats["trades"] < 8 or stats["total_r"] <= 0.0 or stats["profit_factor"] < 1.10:
            continue
        mutations = _set_mapping(
            parent["mutations"], MAPPING_KEYS["max_events"], family, 2
        )
        mutations["param_overrides.pb_aperture_rearm_cooldown_bars"] = 12
        candidates.append(_candidate(
            parent["id"] + f"__rearm__{family.lower()}",
            mutations,
            stage="causal_second_dislocation_atoms",
            parent_id=parent["id"],
            target_family=family,
            activation_evidence=stats,
            hypothesis="add a separated second episode without lowering first-event quality",
        ))
    return escape._dedupe(candidates)


def _combined_rearm_candidate(
    rearm_rows: Iterable[dict[str, Any]], parent_pool: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(rearm_rows)
    if len(rows) != 2 or len({row.get("parent_id") for row in rows}) != 1:
        return []
    parent_id = str(rows[0]["parent_id"])
    parent = next((row for row in parent_pool if row.get("id") == parent_id), None)
    if parent is None:
        return []
    parent_metrics = parent["metrics"]
    useful = all(
        int(row["metrics"]["total_trades"]) >= int(parent_metrics["total_trades"]) + 2
        and float(row["metrics"]["expected_total_r"]) >= float(parent_metrics["expected_total_r"]) + 0.25
        and float(row["metrics"]["profit_factor"]) >= float(parent_metrics["profit_factor"]) - 0.08
        for row in rows
    )
    if not useful:
        return []
    mutations = deepcopy(parent["mutations"])
    for family in ("FAILED_BREAKDOWN_RECLAIM", "PRIOR_DAY_LOW_RECLAIM"):
        mutations = _set_mapping(mutations, MAPPING_KEYS["max_events"], family, 2)
    mutations["param_overrides.pb_aperture_rearm_cooldown_bars"] = 12
    return [_candidate(
        parent_id + "__rearm__failed_breakdown_and_prior_low",
        mutations,
        stage="causal_second_dislocation_interaction",
        parent_id=parent_id,
        structural_sources=[row["id"] for row in rows],
    )]


def _behavior_signature(row: dict[str, Any]) -> str:
    decisions = sorted(
        (
            str(trade.get("symbol", "")),
            str(trade.get("entry_time", "")),
            str(trade.get("route", "")),
        )
        for trade in row.get("trade_attribution", [])
    )
    return hashlib.sha256(
        json.dumps(decisions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _materially_improves(
    row: dict[str, Any], references: Iterable[dict[str, Any]],
) -> bool:
    metrics = row["metrics"]
    if (
        float(metrics.get("profit_factor", 0.0)) < 1.58
        or float(metrics.get("max_drawdown_pct", 1.0)) > 0.045
        or float(metrics.get("expected_total_r", 0.0)) < 43.0
    ):
        return False
    for reference in _structural_parent_beam(references, 3):
        base = reference["metrics"]
        return_gain = float(metrics["expected_total_r"]) - float(base["expected_total_r"])
        trade_gain = int(metrics["total_trades"]) - int(base["total_trades"])
        score_gain = float(row.get("alpha_escape_score", 0.0)) - float(
            reference.get("alpha_escape_score", 0.0)
        )
        if return_gain >= 0.75 and trade_gain >= -10:
            return True
        if trade_gain >= 6 and return_gain >= -1.0:
            return True
        if score_gain >= 0.008 and (return_gain >= 0.25 or trade_gain >= 4):
            return True
    return False


def _evaluate_adaptive_stage(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    control_metrics: dict[str, Any],
    reference_rows: list[dict[str, Any]],
    batch_size: int = 2,
    stale_wave_limit: int = 2,
) -> list[dict[str, Any]]:
    """Evaluate bounded waves and stop after repeated non-material behavior."""

    candidates = escape._dedupe(candidates)
    rows: list[dict[str, Any]] = []
    frontier = _rescore(list(reference_rows))
    seen_behaviors = {_behavior_signature(row) for row in frontier}
    stale_waves = 0
    audit: dict[str, Any] = {
        "stage": stage,
        "planned": len(candidates),
        "batch_size": int(batch_size),
        "stale_wave_limit": int(stale_wave_limit),
        "waves": [],
        "residual_dependent_policies_excluded": True,
    }
    for offset in range(0, len(candidates), batch_size):
        wave_number = offset // batch_size + 1
        wave_candidates = candidates[offset : offset + batch_size]
        wave_rows = _evaluate_stage(
            f"{stage}_wave_{wave_number}",
            wave_candidates,
            args=args,
            output=output,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
            control_metrics=control_metrics,
            incumbent_rows=[*frontier, *rows],
        )
        novel = [row for row in wave_rows if _behavior_signature(row) not in seen_behaviors]
        material = [row for row in novel if _materially_improves(row, frontier)]
        for row in novel:
            seen_behaviors.add(_behavior_signature(row))
        rows.extend(wave_rows)
        audit["waves"].append({
            "wave": wave_number,
            "evaluated": len(wave_rows),
            "behaviorally_novel": len(novel),
            "material_improvements": [row["id"] for row in material],
        })
        if material:
            stale_waves = 0
        else:
            stale_waves += 1
        frontier = _rescore([*frontier, *novel])
        if stale_waves >= stale_wave_limit:
            audit["stopped_early"] = True
            audit["stop_reason"] = (
                f"{stale_waves} consecutive waves without a behaviorally novel material improvement"
            )
            break
    audit.setdefault("stopped_early", False)
    audit["evaluated"] = len(rows)
    audit["skipped"] = len(candidates) - len(rows)
    rows = _rescore(rows)
    _write(output / f"{stage}_results.json", rows)
    _write(output / f"{stage}_adaptive_audit.json", audit)
    return rows


def _find_row(path: Path, predicate: Any) -> dict[str, Any]:
    rows = _load(path)
    match = next((row for row in rows if predicate(row)), None)
    if match is None:
        raise RuntimeError(f"required preserved research row is missing: {path}")
    return match


def _anchors(round_dir: Path, research: Path) -> dict[str, dict[str, Any]]:
    canonical = _load(round_dir / "final_selection.json")
    current = canonical.get("selected", canonical)
    improved = _find_row(
        research / "phase_0_starting_control_full_results.json",
        lambda row: row.get("id") == "improved_start_control",
    )
    course = _find_row(
        research / "phase_0e_full_diagnostics_parents_results.json",
        lambda row: row.get("id") == "course_final_control",
    )
    breadth = _find_row(
        research / "phase_1_structural_entry_results.json",
        lambda row: int(row.get("metrics", {}).get("total_trades", 0)) == 192,
    )
    return {"current149": current, "improved133": improved, "course164": course, "breadth192": breadth}


def _core_synthesis_candidates(anchors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    improved = anchors["improved133"]["mutations"]
    course = anchors["course164"]["mutations"]
    candidates = [
        _candidate(name, row["mutations"], stage="preserved_control", anchor_role=name)
        for name, row in anchors.items()
    ]
    singles = ("mhl", "pdl", "gap", "vwap_confirm", "vwap_next", "volume")
    for name in singles:
        candidates.append(_candidate(
            f"improved133__{name}",
            _apply_profiles(improved, [name], candidate_size=0.55),
            stage="positive_route_control",
            profiles=[name],
        ))
    unions = (
        ("mhl", "pdl"),
        ("mhl", "vwap_next"),
        ("mhl", "gap"),
        ("pdl", "vwap_next"),
        ("mhl", "pdl", "vwap_next"),
        ("mhl", "pdl", "gap"),
        ("mhl", "pdl", "vwap_confirm", "gap"),
        ("mhl", "pdl", "vwap_next", "gap"),
    )
    for names in unions:
        candidates.append(_candidate(
            "improved133__union__" + "_".join(names),
            _apply_profiles(improved, names, candidate_size=0.55),
            stage="positive_route_recombination",
            profiles=list(names),
        ))
    course_unions = (
        ("mhl",),
        ("pdl",),
        ("vwap_next",),
        ("mhl", "pdl"),
        ("mhl", "pdl", "vwap_next"),
    )
    for names in course_unions:
        candidates.append(_candidate(
            "course164__union__" + "_".join(names),
            _apply_profiles(course, names, candidate_size=0.55),
            stage="breadth_parent_recombination",
            profiles=list(names),
        ))
    return escape._dedupe(candidates)


def _repair_candidate(
    parent_name: str,
    parent: dict[str, Any],
    family: str,
    kind: str,
    value: Any,
) -> dict[str, Any]:
    mutations = _set_mapping(parent["mutations"], MAPPING_KEYS[kind], family, value)
    return _candidate(
        f"{parent_name}__repair__{family.lower()}__{kind}_{value}",
        mutations,
        stage="breadth_repair_atom",
        parent_id=parent_name,
        target_family=family,
        repair_kind=kind,
        repair_value=value,
        repair_delta={kind: value},
    )


def _repair_atoms(anchors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("breadth192", "PRIOR_DAY_LOW_RECLAIM", anchors["breadth192"]),
        ("course164", "GAP_EXHAUSTION_RECLAIM", anchors["course164"]),
    )
    candidates: list[dict[str, Any]] = []
    for parent_name, family, parent in definitions:
        if parent_name == "course164":
            candidates.append(_repair_candidate(parent_name, parent, family, "floor", 75))
        for policy in sorted(NEW_FILTERS):
            candidates.append(_repair_candidate(parent_name, parent, family, "filter", policy))
        for max_bar in (6, 12, 24):
            candidates.append(_repair_candidate(parent_name, parent, family, "max_bar", max_bar))
        for cap in (1, 2):
            candidates.append(_repair_candidate(parent_name, parent, family, "cap", cap))
    return escape._dedupe(candidates)


def _hybrid_entry_candidates(core_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bridge the confirm return leader and next-bar breadth leader causally."""

    parent_ids = {
        "improved133__union__mhl_pdl_vwap_confirm_gap",
        "improved133__union__mhl_pdl_vwap_next_gap",
    }
    parents = [row for row in core_rows if row.get("id") in parent_ids]
    if len(parents) != 2:
        raise RuntimeError("both VWAP transition parents are required for hybrid synthesis")
    # Normalizing either parent to the same hybrid policy produces the same
    # immutable mutation signature. Start from the higher-total-R confirm
    # parent and deduplicate defensively.
    parent = max(parents, key=lambda row: float(row["metrics"]["expected_total_r"]))
    candidates: list[dict[str, Any]] = []
    for profile in (
        "vwap_hybrid_deep",
        "vwap_hybrid_residual",
        "vwap_hybrid_room",
    ):
        candidates.append(_candidate(
            parent["id"] + "__" + profile,
            _apply_profiles(parent["mutations"], [profile], candidate_size=0.55),
            stage="vwap_quality_hybrid_entry",
            parent_id=parent["id"],
            profiles=[profile],
            structural_sources=[
                "confirm_return_leader",
                "next_bar_breadth_leader",
                "causal_completed_bar_quality_router",
            ],
        ))
    return escape._dedupe(candidates)


def _repair_utility(row: dict[str, Any], parent: dict[str, Any], family: str) -> float:
    route = _family_stats(row, family)
    base_route = _family_stats(parent, family)
    metrics, base = row["metrics"], parent["metrics"]
    return (
        (route["total_r"] - base_route["total_r"])
        + 0.20 * (float(metrics["expected_total_r"]) - float(base["expected_total_r"]))
        + 0.03 * (float(metrics["total_trades"]) - float(base["total_trades"]))
        - 40.0 * max(float(metrics["max_drawdown_pct"]) - 0.045, 0.0)
    )


def _repair_followups(
    atom_rows: list[dict[str, Any]], anchors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent_name, family in (
        ("breadth192", "PRIOR_DAY_LOW_RECLAIM"),
        ("course164", "GAP_EXHAUSTION_RECLAIM"),
    ):
        parent = anchors[parent_name]
        rows = [row for row in atom_rows if row.get("parent_id") == parent_name]
        best_by_kind: list[dict[str, Any]] = []
        for kind in ("floor", "filter", "max_bar", "cap"):
            choices = [row for row in rows if row.get("repair_kind") == kind]
            if choices:
                best_by_kind.append(max(choices, key=lambda row: _repair_utility(row, parent, family)))
        ranked = sorted(
            best_by_kind,
            key=lambda row: _repair_utility(row, parent, family),
            reverse=True,
        )[:3]
        for left, right in combinations(ranked, 2):
            mutations = deepcopy(parent["mutations"])
            deltas: dict[str, Any] = {}
            for source in (left, right):
                kind = str(source["repair_kind"])
                value = source["repair_value"]
                mutations = _set_mapping(mutations, MAPPING_KEYS[kind], family, value)
                deltas[kind] = value
            candidates.append(_candidate(
                f"{parent_name}__repair_joint__" + "__".join(
                    f"{kind}_{value}" for kind, value in sorted(deltas.items())
                ),
                mutations,
                stage="breadth_repair_interaction",
                parent_id=parent_name,
                target_family=family,
                repair_delta=deltas,
            ))
    return escape._dedupe(candidates)


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm, rm = left["metrics"], right["metrics"]
    comparisons = (
        float(lm["expected_total_r"]) >= float(rm["expected_total_r"]),
        float(lm["total_trades"]) >= float(rm["total_trades"]),
        float(lm["profit_factor"]) >= float(rm["profit_factor"]),
        float(lm["max_drawdown_pct"]) <= float(rm["max_drawdown_pct"]),
    )
    strict = (
        float(lm["expected_total_r"]) > float(rm["expected_total_r"]) + 1e-9
        or float(lm["total_trades"]) > float(rm["total_trades"])
        or float(lm["profit_factor"]) > float(rm["profit_factor"]) + 1e-9
        or float(lm["max_drawdown_pct"]) < float(rm["max_drawdown_pct"]) - 1e-12
    )
    return all(comparisons) and strict


def _pareto(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    viable = [
        row for row in rows
        if float(row["metrics"].get("expected_total_r", -99.0)) >= 28.0
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.35
        and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.055
        and int(row["metrics"].get("total_trades", 0)) >= 150
    ]
    frontier = [
        row for row in viable
        if not any(_dominates(other, row) for other in viable if other is not row)
    ]
    frontier.sort(key=lambda row: float(row["alpha_escape_score"]), reverse=True)
    return frontier[:limit]


def _recombine_repaired(
    all_rows: list[dict[str, Any]], anchors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parents = _pareto(all_rows, 4)
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        existing = _families(parent["mutations"])
        stacks = (
            ("mhl", "pdl", "vwap_next", "gap"),
            ("mhl", "pdl", "vwap_confirm", "gap"),
        )
        for stack in stacks:
            required = [
                name for name in stack
                if PRIMARY_PROFILES[name]["family"] not in existing
            ]
            if not required:
                continue
            mutations = _apply_profiles(parent["mutations"], required, candidate_size=0.55)
            candidates.append(_candidate(
                parent["id"] + "__positive_stack__" + "_".join(required),
                mutations,
                stage="repaired_parent_positive_route_synthesis",
                parent_id=parent["id"],
                profiles=required,
            ))
    # The best 133-anchor synthesis is also allowed to add a repaired
    # gap-exhaustion sleeve, which tests the inverse direction of the search.
    core_frontier = _pareto(
        [row for row in all_rows if str(row.get("id", "")).startswith("improved133")],
        2,
    )
    for parent in core_frontier:
        for policy in ("deep_reclaim", "residual_reclaim", "room_reclaim"):
            mutations = _apply_profiles(parent["mutations"], [], candidate_size=0.55)
            families = _families(mutations) | {"GAP_EXHAUSTION_RECLAIM"}
            mutations["param_overrides.pb_aperture_families"] = ",".join(sorted(families))
            mutations = _set_mapping(
                mutations, MAPPING_KEYS["filter"], "GAP_EXHAUSTION_RECLAIM", policy
            )
            mutations = _set_mapping(
                mutations, MAPPING_KEYS["max_bar"], "GAP_EXHAUSTION_RECLAIM", 12
            )
            candidates.append(_candidate(
                parent["id"] + f"__gap_exhaustion_repair__{policy}",
                mutations,
                stage="core_to_breadth_bridge",
                parent_id=parent["id"],
                target_family="GAP_EXHAUSTION_RECLAIM",
            ))
    return escape._dedupe(candidates)


def _management_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = _pareto(rows, 4)
    candidates: list[dict[str, Any]] = []
    changes = (
        ("management_control", {}),
        ("size70", {"param_overrides.pb_aperture_sizing_mult": 0.70}),
        ("stale4", {"param_overrides.pb_aperture_stale_exit_bars": 4}),
        ("size70_stale4", {
            "param_overrides.pb_aperture_sizing_mult": 0.70,
            "param_overrides.pb_aperture_stale_exit_bars": 4,
        }),
    )
    for parent in parents:
        for name, delta in changes:
            mutations = deepcopy(parent["mutations"])
            mutations.update(delta)
            candidates.append(_candidate(
                parent["id"] + "__" + name,
                mutations,
                stage="lean_management",
                parent_id=parent["id"],
            ))
    return escape._dedupe(candidates)


def _validation_shortlist(
    rows: list[dict[str, Any]], current: dict[str, Any], improved: dict[str, Any],
) -> list[dict[str, Any]]:
    control_signature = escape._signature(improved["mutations"])
    unique: dict[str, dict[str, Any]] = {
        escape._signature(row["mutations"]): row
        for row in rows
        if escape._signature(row["mutations"]) != control_signature
    }
    pool = list(unique.values())
    selected = [current]
    frequency_escape_floor = _minimum_frequency_escape(current)
    selectors = (
        lambda row: float(row["alpha_escape_score"]),
        lambda row: (
            float(row["metrics"]["total_trades"])
            if float(row["metrics"]["expected_total_r"]) >= 42.0
            and float(row["metrics"]["profit_factor"]) >= 1.45
            and float(row["metrics"]["max_drawdown_pct"]) <= 0.045
            else -1e9
        ),
        lambda row: (
            float(row["metrics"]["expected_total_r"])
            if int(row["metrics"]["total_trades"]) >= frequency_escape_floor
            else -1e9
        ),
    )
    for selector in selectors:
        candidate = max(pool, key=selector)
        if escape._signature(candidate["mutations"]) not in {
            escape._signature(row["mutations"]) for row in selected
        }:
            selected.append(candidate)
    return selected[:4]


def _symbol_concentration(row: dict[str, Any]) -> float:
    totals: dict[str, float] = defaultdict(float)
    for trade in row.get("trade_attribution", []):
        totals[str(trade.get("symbol", ""))] += float(trade.get("r", 0.0))
    positive = sum(max(value, 0.0) for value in totals.values())
    return max((max(value, 0.0) for value in totals.values()), default=0.0) / positive if positive else 1.0


def _gates(
    row: dict[str, Any], current: dict[str, Any], improved: dict[str, Any],
    *, fold_validation_enabled: bool = True,
) -> dict[str, bool]:
    metrics = row["metrics"]
    folds = row.get("folds", [])
    aperture = row.get("aperture", {})
    fold_deltas = [float(fold.get("delta_total_r", -99.0)) for fold in folds]
    minimum_frequency_escape = _minimum_frequency_escape(current)
    gates = {
        "sealed_holdout_excluded": row.get("validation_contract", {}).get("holdout_accessed") is False,
        "material_frequency_escape": int(metrics.get("total_trades", 0)) >= minimum_frequency_escape,
        "current_total_r_preserved": float(metrics.get("expected_total_r", 0.0)) >= float(current["metrics"]["expected_total_r"]),
        "portfolio_avg_r": float(metrics.get("avg_r", 0.0)) >= 0.18,
        "portfolio_pf": float(metrics.get("profit_factor", 0.0)) >= 1.60,
        "bounded_drawdown": float(metrics.get("max_drawdown_pct", 1.0)) <= 0.045,
        "robust_average_r": float(metrics.get("robust_avg_r", 0.0)) >= 0.0,
        "aperture_positive": int(aperture.get("trades", 0)) >= 30 and float(aperture.get("total_r", 0.0)) > 0.0 and float(aperture.get("profit_factor", 0.0)) >= 1.10,
        "three_positive_routes": _positive_route_count(row) >= 3,
        "symbol_concentration": _symbol_concentration(row) <= 0.35,
        "value_vs_133_anchor": float(metrics.get("expected_total_r", 0.0)) >= float(improved["metrics"]["expected_total_r"]) + 2.0,
    }
    if fold_validation_enabled:
        gates["fold_integrity"] = bool(row.get("validation_contract", {}).get("passed"))
        gates["chronological_consistency"] = (
            len(fold_deltas) == 3
            and sum(value > 0.0 for value in fold_deltas) >= 2
            and min(fold_deltas) >= -3.0
        )
    else:
        gates["fold_validation_skipped_by_user_request"] = True
    return gates


def _stats(trades: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(trades)
    values = [float(row.get("r", 0.0)) for row in rows]
    wins = sum(max(value, 0.0) for value in values)
    losses = -sum(min(value, 0.0) for value in values)
    return {
        "trades": len(rows),
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": wins / losses if losses else (99.0 if wins else 0.0),
        "win_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
        "net_profit": sum(float(row.get("pnl_net", 0.0)) for row in rows),
    }


def _grouped(trades: list[dict[str, Any]], key: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(key(trade))].append(trade)
    return {name: _stats(values) for name, values in sorted(groups.items())}


def _diagnostics(
    selected: dict[str, Any], current: dict[str, Any], improved: dict[str, Any],
    stage_ledger: list[dict[str, Any]], finalized_at: str,
    *, fold_validation_enabled: bool,
) -> str:
    metrics = selected["metrics"]
    current_metrics = current["metrics"]
    trades = list(selected.get("trade_attribution", []))
    monthly = _grouped(trades, lambda row: str(row.get("entry_time", ""))[:7])
    symbols = _grouped(trades, lambda row: row.get("symbol", "UNKNOWN"))
    routes = _grouped(trades, lambda row: row.get("route", "UNKNOWN"))
    exits = _grouped(trades, lambda row: row.get("exit_reason", "UNKNOWN"))
    lines = [
        "IARIC ROUND 3 — ALPHA ESCAPE SYNTHESIS FINAL DIAGNOSTICS",
        "=" * 84,
        (
            "Status: ACTIVE ROUND 3 — AUTOMATIC REAL-ALPHA PROMOTION GATES PASSED"
            if fold_validation_enabled
            else "Status: ACTIVE ROUND 3 — FULL-PERIOD ALPHA GATES PASSED; "
            "FOLD VALIDATION SKIPPED BY USER REQUEST"
        ),
        f"Finalized: {finalized_at}",
        f"Selected: {selected['id']}",
        f"Training authority: {START_DATE} through {END_DATE}",
        f"Sealed holdout: begins {HOLDOUT_START}; accessed=false",
        "",
        "EXECUTIVE OUTCOME",
        f"  Trades: {current_metrics['total_trades']:.0f} -> {metrics['total_trades']:.0f} ({metrics['total_trades']-current_metrics['total_trades']:+.0f})",
        f"  Expected total R: {current_metrics['expected_total_r']:+.3f} -> {metrics['expected_total_r']:+.3f} ({metrics['expected_total_r']-current_metrics['expected_total_r']:+.3f})",
        f"  Average R: {current_metrics['avg_r']:+.4f} -> {metrics['avg_r']:+.4f}",
        f"  Profit factor: {current_metrics['profit_factor']:.3f} -> {metrics['profit_factor']:.3f}",
        f"  Maximum drawdown: {current_metrics['max_drawdown_pct']:.3%} -> {metrics['max_drawdown_pct']:.3%}",
        f"  Positive aperture routes: {_positive_route_count(selected)}",
        "",
        "PROMOTION GATES",
    ]
    for name, passed in selected["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if fold_validation_enabled:
        lines += ["", "CHRONOLOGICAL FOLDS VS 133-TRADE VALUE ANCHOR"]
        for fold in selected.get("folds", []):
            lines.append(
                f"  {fold['fold']}: {fold['start_date']}..{fold['end_date']} "
                f"delta trades={fold['delta_trades']:+.0f}, delta R={fold['delta_total_r']:+.3f}, "
                f"aperture={fold['aperture']['trades']}/{fold['aperture']['total_r']:+.3f}R"
            )
    else:
        lines += [
            "",
            "CHRONOLOGICAL FOLD VALIDATION",
            "  SKIPPED by explicit user request. No fold-integrity or chronological-consistency",
            "  claim is made; the sealed holdout remains untouched.",
        ]
    lines += ["", "ROUTE ATTRIBUTION"]
    for name, values in sorted(routes.items(), key=lambda item: item[1]["total_r"], reverse=True):
        lines.append(
            f"  {name:<52} n={values['trades']:>3} totalR={values['total_r']:+8.3f} "
            f"avgR={values['avg_r']:+.4f} PF={values['profit_factor']:.3f}"
        )
    lines += ["", "SIGNAL EXTRACTION / DISCRIMINATION"]
    for name in (
        "entry_opportunity_recall", "entry_potential_total_r", "entry_oracle_potential_r",
        "entry_realized_discrimination_lift_r", "entry_rejected_potential_avg_r", "robust_avg_r",
    ):
        lines.append(
            f"  {name:<38} {float(current_metrics.get(name, 0.0)):>10.4f} -> {float(metrics.get(name, 0.0)):>10.4f}"
        )
    lines += ["", "TRADE MANAGEMENT / EXIT PROFILE"]
    for name in (
        "avg_hold_hours", "stop_hit_share", "stop_hit_avg_r", "stop_hit_total_r",
        "carry_trade_share", "carry_avg_r", "eod_flatten_share",
    ):
        lines.append(
            f"  {name:<38} {float(current_metrics.get(name, 0.0)):>10.4f} -> {float(metrics.get(name, 0.0)):>10.4f}"
        )
    lines += ["", "EXIT ATTRIBUTION"]
    for name, values in sorted(exits.items(), key=lambda item: item[1]["trades"], reverse=True):
        lines.append(
            f"  {name:<26} n={values['trades']:>3} totalR={values['total_r']:+8.3f} avgR={values['avg_r']:+.4f}"
        )
    lines += ["", "MONTHLY PERFORMANCE"]
    for name, values in monthly.items():
        lines.append(
            f"  {name}: n={values['trades']:>3} totalR={values['total_r']:+8.3f} avgR={values['avg_r']:+.4f} PF={values['profit_factor']:.3f}"
        )
    lines += ["", "SYMBOL ATTRIBUTION"]
    positive = sum(max(float(values["total_r"]), 0.0) for values in symbols.values())
    for name, values in sorted(symbols.items(), key=lambda item: item[1]["total_r"], reverse=True):
        share = max(float(values["total_r"]), 0.0) / positive if positive else 0.0
        lines.append(
            f"  {name:<8} n={values['trades']:>3} totalR={values['total_r']:+8.3f} avgR={values['avg_r']:+.4f} positive-share={share:.2%}"
        )
    lines += ["", "EXPERIMENT LEDGER"]
    for row in stage_ledger:
        best = row.get("best_metrics", {})
        lines.append(
            f"  {row['stage']}: evaluated={row['evaluated']}, best={row['best_id']}, "
            f"trades={best.get('total_trades', 0):.0f}, R={best.get('expected_total_r', 0):+.3f}, "
            f"PF={best.get('profit_factor', 0):.3f}, DD={best.get('max_drawdown_pct', 0):.3%}"
        )
    lines += ["", "IMMUTABLE OPTIMIZATION SCORE — EXACTLY 7 COMPONENTS"]
    for name, spec in SCORE_SPEC.items():
        lines.append(
            f"  {name}: weight={spec['weight']:.2f}, center={spec['center']:.4g}, scale={spec['scale']:.4g}"
        )
    lines += [
        "",
        "STRUCTURAL INTERPRETATION",
        "  This round is a synthesis, not a restart. It preserves the 133/149-trade anchors,",
        "  recombines executable positive sleeves across breadth and quality parents, then tests",
        "  fixed seven-component family score profiles and causal second-dislocation re-arming.",
        "  Degenerate policy splits and residual-dependent filters were excluded before replay.",
        "  Two-candidate waves stop automatically after repeated non-material decision streams.",
        "  The selected candidate passed absolute frequency, total-R, PF, drawdown, route-breadth,",
        (
            "  concentration, and chronological consistency gates before replacing the prior Round 3."
            if fold_validation_enabled
            else "  and concentration gates before replacing the prior Round 3. Fold validation was skipped."
        ),
        "",
        "SELECTED MUTATIONS",
        json.dumps(selected["mutations"], indent=2, sort_keys=True),
        "",
        f"133-trade starting value anchor expected R: {improved['metrics']['expected_total_r']:+.3f}",
        "Full continuation research: research/alpha_escape_continuation/",
    ]
    return "\n".join(lines) + "\n"


def _finalize(
    selected: dict[str, Any], current: dict[str, Any], improved: dict[str, Any],
    *, round_dir: Path, manifest_path: Path, stage_ledger: list[dict[str, Any]],
    fold_validation_enabled: bool,
) -> None:
    if not selected.get("all_gates_pass"):
        raise RuntimeError("automatic finalization requires every promotion gate")
    trades = list(selected.get("trade_attribution", []))
    if len(trades) != int(selected["metrics"]["total_trades"]):
        raise RuntimeError("trade attribution does not match selected total_trades")
    finalized_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    monthly = _grouped(trades, lambda row: str(row.get("entry_time", ""))[:7])
    symbols = _grouped(trades, lambda row: row.get("symbol", "UNKNOWN"))
    exits = _grouped(trades, lambda row: row.get("exit_reason", "UNKNOWN"))
    routes = _grouped(trades, lambda row: row.get("route", "UNKNOWN"))
    final_status = (
        "complete_automatic_alpha_escape_verified"
        if fold_validation_enabled
        else "complete_alpha_escape_selected_without_fold_validation"
    )
    validation_status = (
        "chronological_and_structural_gates_passed"
        if fold_validation_enabled
        else "full_period_structural_and_economic_gates_passed_fold_validation_skipped_by_user_request"
    )
    selection = {
        "status": final_status,
        "official": True,
        "selected": selected,
        "control": current,
        "starting_value_anchor": improved,
        "automatic_value_verification_passed": fold_validation_enabled,
        "full_period_value_checks_passed": True,
        "fold_validation_status": "passed" if fold_validation_enabled else "skipped_by_user_request",
        "finalized_at_utc": finalized_at,
    }
    _write(round_dir / "optimized_config.json", selected["mutations"])
    _write(round_dir / "final_selection.json", selection)
    _write(round_dir / "final_metrics.json", selected["metrics"])
    _write(round_dir / "final_trades.json", trades)
    _write(round_dir / "final_monthly.json", monthly)
    _write(round_dir / "final_symbols.json", symbols)
    _write(round_dir / "final_exits.json", exits)
    _write(round_dir / "final_routes.json", routes)
    _write(round_dir / "experiment_ledger.json", stage_ledger)
    _write(round_dir / "run_spec.json", {
        "objective": "escape the narrow IARIC local maximum while maximizing real total R and trade frequency",
        "training_window": {"start": START_DATE, "end": END_DATE},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": 2,
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "source_research": "research/alpha_escape_continuation",
        "selected_id": selected["id"],
        "automatic_value_verification_passed": fold_validation_enabled,
        "full_period_value_checks_passed": True,
        "fold_validation_status": "passed" if fold_validation_enabled else "skipped_by_user_request",
        "live_backtest_parity": {
            "shared_core_family_filters": True,
            "shared_core_family_max_bars": True,
            "shared_core_family_transitions": True,
            "causal_completed_bars": True,
        },
    })
    diagnostics = _diagnostics(
        selected,
        current,
        improved,
        stage_ledger,
        finalized_at,
        fold_validation_enabled=fold_validation_enabled,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    hashes = {
        "optimized_config_sha256": _sha256(round_dir / "optimized_config.json"),
        "final_selection_sha256": _sha256(round_dir / "final_selection.json"),
        "round_final_diagnostics_sha256": _sha256(round_dir / "round_final_diagnostics.txt"),
    }
    artifacts = {
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
        "research": "round_3/research/alpha_escape_continuation/",
    }
    summary = {
        "status": selection["status"],
        "official": True,
        "active_round": 3,
        "selected_id": selected["id"],
        "automatic_value_verification_passed": fold_validation_enabled,
        "full_period_value_checks_passed": True,
        "fold_validation_status": "passed" if fold_validation_enabled else "skipped_by_user_request",
        "metrics": selected["metrics"],
        "baseline_metrics": current["metrics"],
        "gates": selected["gates"],
        "validation_contract": selected["validation_contract"],
        "holdout_accessed": False,
        "hashes": hashes,
        "artifacts": artifacts,
        "finalized_at_utc": finalized_at,
    }
    _write(round_dir / "run_summary.json", summary)
    manifest = _load(manifest_path)
    manifest["active_round"] = 3
    manifest.pop("pending_round_3", None)
    manifest["generated_at_utc"] = finalized_at
    manifest["rounds"] = [
        row for row in manifest.get("rounds", []) if int(row.get("round", -1)) != 3
    ] + [{
        "round": 3,
        "status": selection["status"],
        "official": True,
        "active": True,
        "configuration_role": (
            "verified_broad_reversion_alpha_escape_anchor"
            if fold_validation_enabled
            else "full_period_broad_reversion_alpha_escape_anchor_unvalidated_by_folds"
        ),
        "automatic_value_verification_passed": fold_validation_enabled,
        "full_period_value_checks_passed": True,
        "validation_status": validation_status,
        "validation_contract": selected["validation_contract"],
        "training_window": {"start": START_DATE, "end": END_DATE},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "score_component_count": len(SCORE_SPEC),
        "total_trades": int(selected["metrics"]["total_trades"]),
        "expected_total_r": selected["metrics"]["expected_total_r"],
        "avg_r": selected["metrics"]["avg_r"],
        "profit_factor": selected["metrics"]["profit_factor"],
        "max_drawdown_pct": 100.0 * selected["metrics"]["max_drawdown_pct"],
        "mutations_count": len(selected["mutations"]),
        "mutations": selected["mutations"],
        "config_sha256": hashes["optimized_config_sha256"],
        "diagnostics_sha256": hashes["round_final_diagnostics_sha256"],
        "selection_sha256": hashes["final_selection_sha256"],
        "live_backtest_parity": {
            "shared_core_family_policies": True,
            "causal_completed_bar_transitions": True,
        },
        "artifacts": artifacts,
        "timestamp": finalized_at,
    }]
    _write(manifest_path, manifest)


def main() -> int:
    args = _args()
    if not 1 <= int(args.max_workers) <= 2:
        raise ValueError("max-workers must be 1 or 2")
    if str(args.end_date) >= HOLDOUT_START:
        raise ValueError("end date overlaps the sealed holdout")
    if len(SCORE_SPEC) != 7 or abs(sum(spec["weight"] for spec in SCORE_SPEC.values()) - 1.0) > 1e-12:
        raise RuntimeError("immutable optimization score must have exactly seven components summing to one")
    round_dir = Path(args.round_dir).resolve()
    research = Path(args.research_dir).resolve()
    output = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _progress(output, "starting", max_workers=args.max_workers)
    source_fingerprint = escape._replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    cache_report = _prepare_compatible_cache(
        research,
        output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    anchors = _anchors(round_dir, research)
    fold_validation_enabled = not bool(args.skip_fold_validation)
    eval_args = argparse.Namespace(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        max_workers=int(args.max_workers),
    )
    run_spec = {
        "status": "running",
        "objective": "escape the narrow IARIC local maximum by synthesizing positive routes and repairing high-frequency breadth parents",
        "starting_anchors": {
            name: row["metrics"] for name, row in anchors.items()
        },
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": int(args.max_workers),
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "source_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "cache_compatibility": cache_report,
        "promotion_targets": {
            "minimum_trades": _minimum_frequency_escape(anchors["current149"]),
            "frequency_aspiration_trades": FREQUENCY_TARGET_TRADES,
            "minimum_expected_total_r": anchors["current149"]["metrics"]["expected_total_r"],
            "minimum_profit_factor": 1.60,
            "maximum_drawdown_pct": 0.045,
            "positive_delta_r_folds": (
                "at least 2 of 3 vs 133-trade value anchor"
                if fold_validation_enabled
                else "skipped_by_user_request"
            ),
        },
        "fold_validation": {
            "enabled": fold_validation_enabled,
            "status": "pending" if fold_validation_enabled else "skipped_by_user_request",
        },
        "adaptive_search_controls": {
            "maximum_new_full_period_evaluations": 12,
            "batch_size": 2,
            "evidence_stage_stale_wave_limit": 2,
            "secondary_stage_stale_wave_limit": 1,
            "behavioral_decision_stream_deduplication": True,
            "material_improvement_required_to_extend": True,
            "separate_exploration_and_deployment_rankings": True,
            "family_preserved_quality_parent_minimum_trades": 130,
            "old_global_parent_minimum_trades": 150,
            "residual_dependent_policies": "excluded_until_causal_live_replay_input_exists",
        },
        "live_backtest_contract": {
            "shared_core_family_filters": True,
            "shared_core_family_max_bars": True,
            "completed_bar_transitions": True,
            "holdout_excluded": True,
        },
    }
    _write(output / "run_spec.json", run_spec)
    stages: list[tuple[str, list[dict[str, Any]]]] = []

    core_rows = _evaluate_stage(
        "phase_0_positive_route_synthesis",
        _core_synthesis_candidates(anchors),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        incumbent_rows=list(anchors.values()),
    )
    stages.append(("phase_0_positive_route_synthesis", core_rows))

    atom_rows = _evaluate_stage(
        "phase_1_breadth_repair_atoms",
        _repair_atoms(anchors),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        incumbent_rows=core_rows,
    )
    stages.append(("phase_1_breadth_repair_atoms", atom_rows))

    hybrid_rows = _evaluate_stage(
        "phase_1b_vwap_quality_hybrid_entry",
        _hybrid_entry_candidates(core_rows),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        incumbent_rows=[*core_rows, *atom_rows],
    )
    stages.append(("phase_1b_vwap_quality_hybrid_entry", hybrid_rows))

    hybrid_parent = _required_row(
        core_rows, "improved133__union__mhl_pdl_vwap_next_gap"
    )
    _write(
        output / "phase_1b_hybrid_activation_report.json",
        _hybrid_activation_report(hybrid_parent, "VWAP_DEVIATION_RECLAIM"),
    )

    starting_frontier = [*core_rows, *atom_rows, *hybrid_rows]
    evidence_rows = _evaluate_adaptive_stage(
        "phase_2_evidence_backed_sleeve_synthesis",
        _evidence_synthesis_candidates(core_rows, atom_rows),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        reference_rows=starting_frontier,
        batch_size=2,
        stale_wave_limit=2,
    )
    stages.append(("phase_2_evidence_backed_sleeve_synthesis", evidence_rows))

    profile_parent_pool = [*starting_frontier, *evidence_rows]
    profile_rows = _evaluate_adaptive_stage(
        "phase_2b_family_score_profile_calibration",
        _profile_candidates(profile_parent_pool),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        reference_rows=profile_parent_pool,
        batch_size=2,
        stale_wave_limit=1,
    )
    stages.append(("phase_2b_family_score_profile_calibration", profile_rows))

    rearm_parent_pool = [*profile_parent_pool, *profile_rows]
    rearm_rows = _evaluate_adaptive_stage(
        "phase_2c_causal_second_dislocation_atoms",
        _rearm_candidates(rearm_parent_pool),
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        reference_rows=rearm_parent_pool,
        batch_size=2,
        stale_wave_limit=1,
    )
    stages.append(("phase_2c_causal_second_dislocation_atoms", rearm_rows))

    interaction_candidates = _combined_rearm_candidate(rearm_rows, rearm_parent_pool)
    interaction_rows = _evaluate_adaptive_stage(
        "phase_2d_causal_second_dislocation_interaction",
        interaction_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=anchors["improved133"]["metrics"],
        reference_rows=[*rearm_parent_pool, *rearm_rows],
        batch_size=1,
        stale_wave_limit=1,
    ) if interaction_candidates else []
    stages.append(("phase_2d_causal_second_dislocation_interaction", interaction_rows))

    full_pool = [
        *starting_frontier,
        *evidence_rows,
        *profile_rows,
        *rearm_rows,
        *interaction_rows,
    ]

    shortlist = _validation_shortlist(
        full_pool, anchors["current149"], anchors["improved133"]
    )
    _write(output / "validation_shortlist.json", shortlist)
    if fold_validation_enabled:
        escape._fold_validate(
            shortlist,
            anchors["improved133"],
            args=eval_args,
            output=output,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
    else:
        for row in shortlist:
            row.pop("folds", None)
            row["validation_contract"] = {
                "passed": None,
                "fold_validation_performed": False,
                "fold_validation_status": "skipped_by_user_request",
                "holdout_accessed": False,
                "sealed_holdout_start": HOLDOUT_START,
            }
    shortlist = _rescore(shortlist)
    for row in shortlist:
        gates = _gates(
            row,
            anchors["current149"],
            anchors["improved133"],
            fold_validation_enabled=fold_validation_enabled,
        )
        row["gates"] = gates
        row["all_gates_pass"] = all(gates.values())
        row["symbol_concentration"] = _symbol_concentration(row)
        row["frequency_target_180_met"] = (
            int(row["metrics"].get("total_trades", 0)) >= FREQUENCY_TARGET_TRADES
        )
    validated = sorted(shortlist, key=_promotion_rank_key, reverse=True)
    _write(output / "validated_finalists.json", validated)
    eligible = [row for row in validated if row["all_gates_pass"]]
    selected = eligible[0] if eligible else validated[0]
    stage_ledger = []
    for name, rows in stages:
        best = rows[0] if rows else {}
        stage_ledger.append({
            "stage": name,
            "evaluated": len(rows),
            "best_id": best.get("id"),
            "best_metrics": best.get("metrics", {}),
        })
    result = {
        "status": (
            "promotion_verified"
            if eligible and fold_validation_enabled
            else "promotion_selected_without_fold_validation"
            if eligible
            else "no_candidate_passed_all_promotion_gates"
        ),
        "canonical_round_changed": bool(eligible and not args.no_finalize),
        "selected": selected,
        "current_round3": anchors["current149"],
        "starting_value_anchor": anchors["improved133"],
        "validated_finalists": validated,
        "ranking_contract": {
            "exploration": "immutable_seven_component_alpha_escape_score",
            "promotion": "gates_then_total_r_then_pf_then_inverse_drawdown_then_trades",
            "frequency_target_trades": FREQUENCY_TARGET_TRADES,
            "minimum_frequency_escape_trades": _minimum_frequency_escape(anchors["current149"]),
        },
        "stage_ledger": stage_ledger,
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "holdout_accessed": False,
        "fold_validation_status": "passed" if fold_validation_enabled else "skipped_by_user_request",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write(output / "final_selection.json", result)
    if eligible and not args.no_finalize:
        _finalize(
            selected,
            anchors["current149"],
            anchors["improved133"],
            round_dir=round_dir,
            manifest_path=manifest,
            stage_ledger=stage_ledger,
            fold_validation_enabled=fold_validation_enabled,
        )
    run_spec["status"] = result["status"]
    run_spec["completed_at_utc"] = result["completed_at_utc"]
    run_spec["canonical_round_changed"] = result["canonical_round_changed"]
    _write(output / "run_spec.json", run_spec)
    _progress(
        output,
        "complete",
        result_status=result["status"],
        selected_id=selected["id"],
        selected_metrics=selected["metrics"],
        canonical_round_changed=result["canonical_round_changed"],
    )
    print(
        f"alpha escape continuation complete: {result['status']}; "
        f"selected={selected['id']}; trades={selected['metrics']['total_trades']}; "
        f"holdout accessed=no",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failed = _args()
        failed_output = Path(failed.output_dir).resolve()
        failed_output.mkdir(parents=True, exist_ok=True)
        _progress(
            failed_output,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
