"""Run the activation-first IARIC structural challenger.

The challenger reopens composition generation across every implemented
reversion family.  It preserves family coverage, frequency and the Pareto
frontier instead of pruning solely by a scalar leader, then spends entry and
management replays only around viable structural roots. Weak routes receive
the pre-registered family floor/filter/cap controls before they can be pruned,
so a negative standalone route can still become useful conditionally without
forcing a global low-quality union. The sealed holdout is excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backtests.stock.auto.runners import run_iaric_escape_round3 as escape
from backtests.stock.auto.runners.run_iaric_escape_course_continuation import (
    _broad_validation_shortlist,
    _validated_rank,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _signature,
    _write_json,
)
from backtests.stock.auto.iaric.worker import evaluate_candidate_execution_attribution
from strategies.stock.iaric.core.logic import aperture_family_from_route


REPO_ROOT = Path(__file__).resolve().parents[4]
IARIC_DIR = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_ESCAPE = IARIC_DIR / "round_3/escape_round"
DEFAULT_OUTPUT = IARIC_DIR / "round_3/structural_challenger"
HOLDOUT_START = "2026-03-02"
FAMILY_POLICY_KEYS = (
    "param_overrides.pb_aperture_family_score_floors",
    "param_overrides.pb_aperture_family_filters",
    "param_overrides.pb_aperture_family_daily_caps",
    "param_overrides.pb_aperture_family_transitions",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escape-dir", default=str(DEFAULT_ESCAPE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _cache_count(output: Path) -> int:
    total = 0
    for name in ("evaluation_cache.json", "structural_screen_cache.json"):
        payload = _load_json(output / name, {})
        if isinstance(payload, dict):
            total += len(payload.get("evaluations", {}))
    return total


def _progress(output: Path, status: str, **extra: Any) -> None:
    _write_json(
        output / "queue_status.json",
        {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cached_evaluations": _cache_count(output),
            **extra,
        },
    )


def _structural_code_fingerprint() -> str:
    """Fingerprint the executable search as well as the shared replay path."""

    digest = hashlib.sha256()
    digest.update(escape._code_fingerprint().encode("ascii"))
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "backtests/stock/auto/runners/run_iaric_repaired_baseline_recovery.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rekey_compatible_cache(
    target: Path,
    *,
    source_fingerprint: str,
    code_fingerprint: str,
    backup: Path,
) -> None:
    """Preserve decision-identical work, failing closed on policy results."""

    payload = _load_json(target, {})
    if payload.get("source_fingerprint") != source_fingerprint:
        raise ValueError(f"{target.name} has an incompatible source fingerprint")
    old_code = str(payload.get("code_fingerprint", ""))
    if old_code == code_fingerprint:
        return
    evaluations = dict(payload.get("evaluations", {}))
    for key, row in evaluations.items():
        mutations = dict(row.get("mutations", {})) if isinstance(row, dict) else {}
        if any(policy in mutations for policy in FAMILY_POLICY_KEYS[:3]):
            raise ValueError(
                "Cannot migrate a cache containing family-policy evaluations across code fingerprints"
            )
        parts = key.split("|")
        if len(parts) != 5 or parts[1] != old_code:
            raise ValueError(f"Unexpected structural cache namespace: {key}")
    if not backup.exists():
        shutil.copy2(target, backup)
    migrated: dict[str, Any] = {}
    for key, row in evaluations.items():
        parts = key.split("|")
        parts[1] = code_fingerprint
        migrated["|".join(parts)] = row
    payload["evaluations"] = migrated
    payload["code_fingerprint"] = code_fingerprint
    payload["family_policy_compatibility_migration"] = {
        "from": old_code,
        "to": code_fingerprint,
        "reason": (
            "new family policies are opt-in; every migrated mutation lacks the new settings"
        ),
    }
    _write_json(target, payload)


def _prepare_cache(
    escape_dir: Path,
    output: Path,
    *,
    source_fingerprint: str,
    code_fingerprint: str,
) -> None:
    """Seed and re-key only decision-identical cached evaluations.

    Orchestration can change without discarding hours of completed replay.
    Once any cache contains an opt-in family-policy mutation, however, a code
    fingerprint change fails closed and requires explicit compatibility proof.
    """

    target = output / "evaluation_cache.json"
    if not target.exists():
        source = escape_dir / "evaluation_cache.json"
        if not source.exists():
            raise FileNotFoundError(source)
        temporary = output / ".evaluation_cache.seed.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(target)
    _rekey_compatible_cache(
        target,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        backup=output / "evaluation_cache.pre_family_policy_extension.json",
    )

    screen = output / "structural_screen_cache.json"
    if screen.exists():
        _rekey_compatible_cache(
            screen,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
            backup=output / "structural_screen_cache.pre_starting_anchor_fix.json",
        )


def _seed_screen_cache_from_full(output: Path) -> dict[str, Any]:
    """Reuse full diagnostics as a strict execution-result superset.

    Trade execution must be diagnostics-pure. Every overlapping row is checked
    for exact economic and trade-count parity before missing full rows are
    copied into the cheaper structural-screen namespace. Scoring-specific
    diagnostic fields are normalized separately by
    :func:`_execution_screen_score_metrics`.
    """

    full_path = output / "evaluation_cache.json"
    screen_path = output / "structural_screen_cache.json"
    full = _load_json(full_path, {})
    screen = _load_json(screen_path, None)
    if not isinstance(full, dict):
        raise ValueError("Structural caches are not valid JSON objects")
    if screen is None:
        screen = {
            "source_fingerprint": full.get("source_fingerprint"),
            "code_fingerprint": full.get("code_fingerprint"),
            "evaluations": {},
        }
    if not isinstance(screen, dict):
        raise ValueError("Structural caches are not valid JSON objects")
    if (
        full.get("source_fingerprint") != screen.get("source_fingerprint")
        or full.get("code_fingerprint") != screen.get("code_fingerprint")
    ):
        raise ValueError("Full and structural-screen cache namespaces differ")
    full_rows = dict(full.get("evaluations", {}))
    screen_rows = dict(screen.get("evaluations", {}))
    parity_fields = ("total_trades", "expected_total_r", "profit_factor", "max_drawdown_pct")
    verified_overlap = 0
    for key in set(full_rows).intersection(screen_rows):
        left = full_rows[key]
        right = screen_rows[key]
        if any(
            abs(float(left.get("metrics", {}).get(field, 0.0)) - float(right.get("metrics", {}).get(field, 0.0)))
            > 1e-12
            for field in parity_fields
        ) or len(left.get("trade_attribution", [])) != len(right.get("trade_attribution", [])):
            raise RuntimeError(f"Diagnostics changed execution for cached signature {key.rsplit('|', 1)[-1]}")
        verified_overlap += 1
    reused = 0
    for key, row in full_rows.items():
        if key in screen_rows or row.get("error"):
            continue
        copied = deepcopy(row)
        copied["screen_cache_provenance"] = "full_diagnostics_execution_superset"
        screen_rows[key] = copied
        reused += 1
    screen["evaluations"] = screen_rows
    screen["full_diagnostics_reuse"] = {
        "verified_overlap": verified_overlap,
        "reused": reused,
        "economic_parity_fields": list(parity_fields),
        "reason": "diagnostics-pure execution; scoring diagnostics normalized at screen time",
    }
    _write_json(screen_path, screen)
    report = {
        "passed": True,
        "verified_overlap": verified_overlap,
        "reused": reused,
        "full_cache_evaluations": len(full_rows),
        "screen_cache_evaluations": len(screen_rows),
    }
    _write_json(output / "cache_reuse_report.json", report)
    return report


def _execution_screen_score_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize full/fast rows to the same trade-only Phase 0 score scope."""

    metrics = dict(row.get("metrics", {}))
    trades = list(row.get("trade_attribution", []))
    metrics["entry_realized_discrimination_lift_r"] = float(metrics.get("avg_r", 0.0))
    dates = sorted(
        {
            datetime.fromisoformat(str(trade["entry_time"])).date()
            for trade in trades
            if trade.get("entry_time")
        }
    )
    fold_avg_r: list[float] = []
    for chunk_array in (
        np.array_split(np.asarray(dates, dtype=object), min(4, len(dates)))
        if dates
        else []
    ):
        chunk = set(chunk_array.tolist())
        values = [
            float(trade.get("r", 0.0))
            for trade in trades
            if trade.get("entry_time")
            and datetime.fromisoformat(str(trade["entry_time"])).date() in chunk
        ]
        fold_avg_r.append(sum(values) / len(values) if values else 0.0)
    metrics["robust_avg_r"] = (
        float(np.percentile(fold_avg_r, 25))
        if fold_avg_r
        else float(metrics.get("avg_r", 0.0))
    )
    return metrics


def _candidate(candidate_id: str, mutations: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return {"id": candidate_id, "mutations": dict(sorted(mutations.items())), **meta}


def _combine_by_signature(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            signature = _signature(row["mutations"])
            previous = combined.get(signature)
            if previous is None or float(row.get("escape_score", -99.0)) > float(
                previous.get("escape_score", -99.0)
            ):
                combined[signature] = row
    return list(combined.values())


def _mapping_mutation(
    base: dict[str, Any],
    setting: str,
    family: str,
    value: str | int | float,
) -> dict[str, Any]:
    mappings: dict[str, str] = {}
    raw = str(base.get(setting, "") or "")
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "="
        key, current = token.split(separator, 1)
        mappings[key.strip().upper()] = current.strip().lower()
    mappings[str(family).strip().upper()] = str(value).strip().lower()
    mutations = dict(base)
    mutations[setting] = ",".join(
        f"{key}:{mappings[key]}" for key in sorted(mappings)
    )
    return mutations


def _root_candidates(
    catalog: dict[str, Any],
    baseline: dict[str, Any],
    course_selected: dict[str, Any],
) -> list[dict[str, Any]]:
    leader_pair = list(catalog["current_leader_pair"])
    improved_start = escape._common_aperture(baseline, leader_pair)
    rows = [
        _candidate("incumbent_control", baseline, stage="control", families=[], focus_families=[]),
        _candidate(
            "improved_start_control",
            improved_start,
            stage="improved_start_control",
            families=leader_pair,
            focus_families=[],
            mandatory_improved_start=True,
        ),
        _candidate(
            "course_final_control",
            course_selected["mutations"],
            stage="course_control",
            families=course_selected.get("families", []),
            focus_families=[],
            mandatory_course_control=True,
        ),
    ]
    for root in catalog["root_candidates"]:
        mutations = escape._common_aperture(baseline, root["families"])
        mutations["param_overrides.pb_aperture_event_score_min"] = float(root["event_score_min"])
        rows.append(
            _candidate(
                root["id"],
                mutations,
                stage="structural_root",
                families=root["families"],
                focus_families=root.get("focus_families", []),
                structural_sources=root.get("sources", []),
            )
        )
    return escape._dedupe(rows)


def _activation_root_candidates(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Controls and all single-family roots establish activation before interaction."""

    return [
        row
        for row in roots
        if row["id"] in {
            "incumbent_control",
            "improved_start_control",
            "course_final_control",
        }
        or len(row.get("families", [])) == 1
    ]


def _primary_interaction_ids(
    catalog: dict[str, Any],
    overlap: dict[str, Any],
) -> set[str]:
    """Choose broad, orthogonal first-pass contexts without scalar pruning.

    Every weak family is tested with the leader pair, its least-overlapping
    positive anchor, and the quota-balanced weak/weak interaction set. The
    remaining anchor contexts are conditionally opened only after executable
    activation, which reduces multiple testing without assuming a route weak
    alone must be weak in composition.
    """

    selected = {
        row["id"]
        for row in catalog["root_candidates"]
        if "leader_pair_plus_orthogonal_family" in row.get("sources", [])
        or "weak_alone_orthogonal_pair" in row.get("sources", [])
    }
    matrix = overlap.get("symbol_day_jaccard", {})
    for family in catalog["weak_or_dormant_families"]:
        choices = [
            row
            for row in catalog["root_candidates"]
            if "positive_anchor_plus_weak_family" in row.get("sources", [])
            and family in row.get("focus_families", [])
        ]
        if not choices:
            continue

        def overlap_key(row: dict[str, Any]) -> tuple[float, str]:
            anchor = next(value for value in row["families"] if value != family)
            return float(matrix.get(family, {}).get(anchor, 1.0)), str(row["id"])

        selected.add(min(choices, key=overlap_key)["id"])
    return selected


def _best_alternative_transition(activation: dict[str, Any], family: str) -> str:
    default = (
        "retrace"
        if family == "PRIOR_DAY_LOW_RECLAIM"
        else "confirm"
        if family == "MULTIDAY_HIGHER_LOW_RECLAIM"
        else "next_bar"
    )
    transition_rows = activation.get(family, {}).get("atlas", {}).get("transitions", {})

    def robust_key(transition: str) -> tuple[float, float, float, str]:
        folds = list(transition_rows.get(transition, {}).values())
        averages = [float(row.get("avg_r", -99.0)) for row in folds]
        return (
            float(sum(value > 0.0 for value in averages)),
            min(averages, default=-99.0),
            sum(averages) / max(len(averages), 1),
            transition,
        )

    alternatives = [value for value in ("next_bar", "confirm", "retrace") if value != default]
    return max(alternatives, key=robust_key)


def _focus_family_stats(row: dict[str, Any], family: str) -> dict[str, float]:
    trades = [
        trade
        for trade in row.get("trade_attribution", [])
        if aperture_family_from_route(str(trade.get("route", ""))) == family
    ]
    total_r = sum(float(trade.get("r", 0.0)) for trade in trades)
    gross_profit = sum(max(float(trade.get("r", 0.0)), 0.0) for trade in trades)
    gross_loss = -sum(min(float(trade.get("r", 0.0)), 0.0) for trade in trades)
    return {
        "trades": float(len(trades)),
        "total_r": total_r,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (99.0 if gross_profit else 0.0),
    }


def _family_activated(rows: Iterable[dict[str, Any]], family: str) -> bool:
    return any(_focus_family_stats(row, family)["trades"] >= 3 for row in rows)


def _family_positive(rows: Iterable[dict[str, Any]], family: str) -> bool:
    for row in rows:
        stats = _focus_family_stats(row, family)
        if stats["trades"] >= 3 and stats["total_r"] > 0.0:
            return True
    return False


def _best_family_context(
    rows: Iterable[dict[str, Any]],
    family: str,
) -> dict[str, Any] | None:
    contexts = [row for row in rows if family in row.get("focus_families", [])]
    if not contexts:
        return None
    return max(
        contexts,
        key=lambda row: (
            _focus_family_stats(row, family)["trades"],
            _focus_family_stats(row, family)["total_r"],
            float(row.get("escape_score", -99.0)),
        ),
    )


def _activation_rescue_candidates(
    interaction_rows: list[dict[str, Any]],
    activation: dict[str, Any],
    weak_families: Iterable[str],
) -> list[dict[str, Any]]:
    """Run only the first-line, role-specific rescue atoms.

    Combinations and secondary cap values are withheld until an atom changes
    executable focus-family economics. This reduces both runtime and the
    multiple-testing burden without dropping any pre-registered policy family.
    """

    candidates: list[dict[str, Any]] = []
    for family in weak_families:
        contexts = [row for row in interaction_rows if family in row.get("focus_families", [])]
        if not contexts or _family_positive(contexts, family):
            continue
        parent = _best_family_context(contexts, family)
        if parent is None:
            continue
        role = str(activation.get(family, {}).get("role", ""))
        transition = _best_alternative_transition(activation, family)
        common = {
            "stage": "activation_rescue_stage1",
            "families": parent.get("families", []),
            "focus_families": [family],
            "root_id": parent.get("root_id", parent["id"]),
            "policy_family": family,
            "structural_sources": ["pre_registered_family_policy_rescue"],
        }
        if role == "dormant_or_transition_blocked":
            floor_mutations = _mapping_mutation(
                parent["mutations"],
                "param_overrides.pb_aperture_family_score_floors",
                family,
                65,
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_floor65",
                floor_mutations,
                rescue_atom="floor65",
                **common,
            ))
            transition_mutations = escape._family_transition_mutation(
                parent["mutations"], family, transition
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_{transition}",
                transition_mutations,
                rescue_atom=f"transition:{transition}",
                **common,
            ))
        elif role == "high_supply_negative_standalone":
            floor_mutations = _mapping_mutation(
                parent["mutations"],
                "param_overrides.pb_aperture_family_score_floors",
                family,
                75,
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_floor75",
                floor_mutations,
                rescue_atom="floor75",
                **common,
            ))
            for policy in ("geometry", "participation"):
                mutations = _mapping_mutation(
                    parent["mutations"],
                    "param_overrides.pb_aperture_family_filters",
                    family,
                    policy,
                )
                candidates.append(_candidate(
                    parent["id"] + f"__{family.lower()}_{policy}",
                    mutations,
                    rescue_atom=f"filter:{policy}",
                    **common,
                ))
            cap1 = _mapping_mutation(
                parent["mutations"],
                "param_overrides.pb_aperture_family_daily_caps",
                family,
                1,
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_cap1",
                cap1,
                rescue_atom="daily_cap:1",
                **common,
            ))
    return escape._dedupe(candidates)


def _activation_rescue_followup_candidates(
    interaction_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    activation: dict[str, Any],
    weak_families: Iterable[str],
) -> list[dict[str, Any]]:
    """Open at most two evidence-triggered rescue follow-ups per family."""

    candidates: list[dict[str, Any]] = []
    evidence = [*interaction_rows, *stage1_rows]
    for family in weak_families:
        if _family_positive(evidence, family):
            continue
        parent = _best_family_context(interaction_rows, family)
        if parent is None:
            continue
        base_stats = _focus_family_stats(parent, family)
        atoms = [row for row in stage1_rows if row.get("policy_family") == family]
        role = str(activation.get(family, {}).get("role", ""))
        common = {
            "stage": "activation_rescue_followup",
            "families": parent.get("families", []),
            "focus_families": [family],
            "root_id": parent.get("root_id", parent["id"]),
            "policy_family": family,
            "structural_sources": ["activation_triggered_rescue_followup"],
        }
        if role == "dormant_or_transition_blocked":
            floor = next((row for row in atoms if row.get("rescue_atom") == "floor65"), None)
            transition = next(
                (row for row in atoms if str(row.get("rescue_atom", "")).startswith("transition:")),
                None,
            )
            changed = any(
                row is not None
                and (
                    _focus_family_stats(row, family)["trades"] != base_stats["trades"]
                    or abs(_focus_family_stats(row, family)["total_r"] - base_stats["total_r"]) > 1e-12
                )
                for row in (floor, transition)
            )
            if changed and floor is not None and transition is not None:
                transition_name = str(transition["rescue_atom"]).split(":", 1)[1]
                trigger_gain = max(
                    _focus_family_stats(floor, family)["total_r"],
                    _focus_family_stats(transition, family)["total_r"],
                ) - base_stats["total_r"]
                joint = escape._family_transition_mutation(
                    floor["mutations"], family, transition_name
                )
                candidates.append(_candidate(
                    parent["id"] + f"__{family.lower()}_floor65_{transition_name}",
                    joint,
                    rescue_atom="floor65+transition",
                    rescue_trigger_gain_r=trigger_gain,
                    **common,
                ))
        elif role == "high_supply_negative_standalone":
            floor = next((row for row in atoms if row.get("rescue_atom") == "floor75"), None)
            filters = [
                row for row in atoms
                if str(row.get("rescue_atom", "")).startswith("filter:")
            ]
            best_filter = max(
                filters,
                key=lambda row: _focus_family_stats(row, family)["total_r"],
                default=None,
            )
            atom_improved = any(
                _focus_family_stats(row, family)["total_r"] > base_stats["total_r"]
                for row in (floor, best_filter)
                if row is not None
            )
            if atom_improved and floor is not None and best_filter is not None:
                policy = str(best_filter["rescue_atom"]).split(":", 1)[1]
                joint = _mapping_mutation(
                    floor["mutations"],
                    "param_overrides.pb_aperture_family_filters",
                    family,
                    policy,
                )
                candidates.append(_candidate(
                    parent["id"] + f"__{family.lower()}_floor75_{policy}",
                    joint,
                    rescue_atom="floor75+best_filter",
                    rescue_trigger_gain_r=max(
                        _focus_family_stats(floor, family)["total_r"],
                        _focus_family_stats(best_filter, family)["total_r"],
                    ) - base_stats["total_r"],
                    **common,
                ))
            cap1 = next((row for row in atoms if row.get("rescue_atom") == "daily_cap:1"), None)
            if cap1 is not None:
                cap_stats = _focus_family_stats(cap1, family)
                if cap_stats["total_r"] > base_stats["total_r"] and cap_stats["trades"] < base_stats["trades"]:
                    cap2 = _mapping_mutation(
                        parent["mutations"],
                        "param_overrides.pb_aperture_family_daily_caps",
                        family,
                        2,
                    )
                    candidates.append(_candidate(
                        parent["id"] + f"__{family.lower()}_cap2",
                        cap2,
                        rescue_atom="daily_cap:2",
                        rescue_trigger_gain_r=cap_stats["total_r"] - base_stats["total_r"],
                        **common,
                    ))
    deduped = escape._dedupe(candidates)
    ranked = sorted(
        deduped,
        key=lambda row: float(row.get("rescue_trigger_gain_r", -1e9)),
        reverse=True,
    )
    # Preserve one evidence-triggered continuation per family before spending
    # any of the strict eight-evaluation budget on a second continuation.
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in ranked:
        family = str(row.get("policy_family", ""))
        if family and family not in seen_families:
            selected.append(row)
            seen_families.add(family)
    for row in ranked:
        if len(selected) >= 8:
            break
        if all(_signature(row["mutations"]) != _signature(existing["mutations"]) for existing in selected):
            selected.append(row)
    return selected[:8]


def _policy_overrides(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.get("mutations", {}).items()
        if key in FAMILY_POLICY_KEYS
    }


def _conditional_interaction_candidates(
    roots: list[dict[str, Any]],
    primary_ids: set[str],
    evidence: list[dict[str, Any]],
    weak_families: Iterable[str],
    anchor_priority: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Open one evidence-selected secondary context per positive family.

    A family that is already positive is carried forward in its simplest
    positive form.  A rescue policy is inherited only when no unfiltered
    context made the family positive; evaluating both forms would duplicate a
    branch without adding an independent hypothesis. Activated-but-negative
    families stop here: their pre-registered primary contexts and rescue atoms
    already tested the hypothesis, so more anchors would add multiplicity
    without evidence of marginal alpha.
    """

    candidates: list[dict[str, Any]] = []
    anchor_rank = {
        family: index for index, family in enumerate(anchor_priority)
    }
    for family in weak_families:
        family_evidence = [row for row in evidence if family in row.get("focus_families", [])]
        if not _family_positive(family_evidence, family):
            continue
        positive = [
            row
            for row in family_evidence
            if _focus_family_stats(row, family)["trades"] >= 3
            and _focus_family_stats(row, family)["total_r"] > 0.0
        ]
        unfiltered_positive = [row for row in positive if not _policy_overrides(row)]
        best_policy: dict[str, Any] = {}
        if not unfiltered_positive and positive:
            best_policy = _policy_overrides(
                max(positive, key=lambda row: _focus_family_stats(row, family)["total_r"])
            )
        eligible_roots = [
            root
            for root in roots
            if root["id"] not in primary_ids
            and len(root.get("families", [])) > 1
            and family in root.get("focus_families", [])
        ]
        if not eligible_roots:
            continue

        def context_rank(root: dict[str, Any]) -> tuple[int, int, str]:
            partners = [
                value for value in root.get("families", []) if value != family
            ]
            return (
                min((anchor_rank.get(value, len(anchor_rank)) for value in partners), default=len(anchor_rank)),
                len(root.get("families", [])),
                str(root["id"]),
            )

        # Exactly one secondary context verifies whether the positive marginal
        # family survives behind the strongest not-yet-tested anchor.
        root = min(eligible_roots, key=context_rank)
        if best_policy:
            mutations = dict(root["mutations"])
            mutations.update(best_policy)
            candidates.append(_candidate(
                root["id"] + f"__{family.lower()}_inherited_policy",
                mutations,
                stage="conditional_interaction",
                families=root.get("families", []),
                focus_families=root.get("focus_families", []),
                root_id=root["id"],
                structural_sources=["activation_supported_secondary_context"],
            ))
        else:
            candidates.append(root)
    return escape._dedupe(candidates)


def _viable(rows: Iterable[dict[str, Any]], control: dict[str, Any]) -> list[dict[str, Any]]:
    base_dd = float(control["metrics"].get("max_drawdown_pct", 1.0))
    result = []
    for row in rows:
        metrics = row.get("metrics", {})
        aperture = row.get("aperture", {})
        mandatory = bool(row.get("mandatory_course_control"))
        passed = (
            float(metrics.get("avg_r", -99.0)) >= 0.08
            and float(metrics.get("profit_factor", 0.0)) >= 1.15
            and float(metrics.get("max_drawdown_pct", 1.0)) <= max(base_dd + 0.020, 0.08)
            and int(aperture.get("trades", 0)) >= 3
            and float(aperture.get("total_r", 0.0)) > 0.0
        )
        if passed or mandatory:
            result.append(row)
    return sorted(result, key=lambda row: float(row.get("escape_score", -99.0)), reverse=True)


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def vector(row: dict[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        return (
            float(row.get("escape_score", -99.0)),
            float(metrics.get("expected_total_r", -1e9)),
            float(metrics.get("total_trades", 0.0)),
            float(metrics.get("profit_factor", 0.0)),
            -float(metrics.get("max_drawdown_pct", 1.0)),
        )

    result = []
    for row in rows:
        current = vector(row)
        dominated = False
        for other in rows:
            if _signature(other["mutations"]) == _signature(row["mutations"]):
                continue
            challenger = vector(other)
            if all(left >= right for left, right in zip(challenger, current)) and any(
                left > right for left, right in zip(challenger, current)
            ):
                dominated = True
                break
        if not dominated:
            result.append(row)
    return result


def _structural_beam(
    rows: list[dict[str, Any]],
    control: dict[str, Any],
    weak_families: Iterable[str],
    *,
    soft_limit: int,
    require_focus_positive: bool = False,
    family_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    viable = _viable(rows, control)
    if not viable:
        return [], {}
    weak_families = list(weak_families)
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}

    def admit(row: dict[str, Any] | None, reason: str) -> None:
        if row is None:
            return
        signature = _signature(row["mutations"])
        reasons.setdefault(signature, []).append(reason)
        if all(_signature(existing["mutations"]) != signature for existing in selected):
            selected.append(row)

    admit(
        next((row for row in viable if row.get("mandatory_improved_start")), None),
        "improved_start_control",
    )
    admit(next((row for row in viable if row.get("mandatory_course_control")), None), "course_control")

    # Coverage is deliberately constrained to families whose own attributed
    # trades created positive marginal alpha.  Whole-composition quality can
    # otherwise smuggle a losing route into the beam behind a strong anchor.
    representatives: list[tuple[str, dict[str, Any]]] = []
    for family in weak_families:
        membership_key = "focus_families" if require_focus_positive else "families"
        matches = [row for row in viable if family in row.get(membership_key, [])]
        if require_focus_positive:
            matches = [
                row
                for row in matches
                if _focus_family_stats(row, family)["trades"] >= 3
                and _focus_family_stats(row, family)["total_r"] > 0.0
            ]
        if matches:
            representatives.append((family, max(
                matches,
                key=lambda row: (
                    _focus_family_stats(row, family)["total_r"],
                    _focus_family_stats(row, family)["trades"],
                    _focus_family_stats(row, family)["profit_factor"],
                    float(row.get("escape_score", -99.0)),
                ),
            )))

    # Preserve complementary marginal-alpha expressions: return, frequency,
    # and discrimination each get first refusal before filling by family R.
    family_order: list[tuple[str, dict[str, Any]]] = []
    if representatives:
        selectors = (
            lambda item: _focus_family_stats(item[1], item[0])["total_r"],
            lambda item: _focus_family_stats(item[1], item[0])["trades"],
            lambda item: _focus_family_stats(item[1], item[0])["profit_factor"],
        )
        for selector in selectors:
            choice = max(representatives, key=selector)
            if choice[0] not in {family for family, _ in family_order}:
                family_order.append(choice)
        for item in sorted(
            representatives,
            key=lambda value: _focus_family_stats(value[1], value[0])["total_r"],
            reverse=True,
        ):
            if item[0] not in {family for family, _ in family_order}:
                family_order.append(item)
    if family_limit is not None:
        family_order = family_order[: max(int(family_limit), 0)]
    for family, row in family_order:
        admit(row, f"positive_focus_family:{family}" if require_focus_positive else f"best_weak_family:{family}")

    def optional_parent(row: dict[str, Any]) -> bool:
        if row.get("mandatory_improved_start") or row.get("mandatory_course_control"):
            return True
        if not require_focus_positive:
            return True
        return any(
            family in row.get("focus_families", [])
            and _focus_family_stats(row, family)["trades"] >= 3
            and _focus_family_stats(row, family)["total_r"] > 0.0
            for family in weak_families
        )

    eligible = [row for row in viable if optional_parent(row)]
    if eligible:
        admit(max(eligible, key=lambda row: float(row.get("escape_score", -99.0))), "top_score")
        admit(max(eligible, key=lambda row: float(row["metrics"].get("expected_total_r", -1e9))), "top_total_r")
        admit(max(eligible, key=lambda row: float(row["metrics"].get("total_trades", 0.0))), "top_frequency")
        admit(max(eligible, key=lambda row: float(row["metrics"].get("profit_factor", 0.0))), "top_profit_factor")
        admit(min(eligible, key=lambda row: float(row["metrics"].get("max_drawdown_pct", 1.0))), "lowest_drawdown")
    for row in sorted(_pareto(eligible), key=lambda value: float(value.get("escape_score", -99.0)), reverse=True):
        admit(row, "pareto_front")

    selected = selected[: max(int(soft_limit), 0)]
    retained = {_signature(row["mutations"]) for row in selected}
    return selected, {key: value for key, value in reasons.items() if key in retained}


def _entry_candidates(
    parents: list[dict[str, Any]],
    activation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Test atomic, family-scoped discrimination without contaminating siblings."""

    activation = activation or {}
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        base = parent["mutations"]
        families = list(parent.get("families", []))
        focus = list(parent.get("focus_families", [])) or families
        common = {
            "stage": "structural_entry",
            "families": families,
            "focus_families": focus,
            "root_id": parent.get("root_id", parent["id"]),
        }
        candidates.append(_candidate(parent["id"] + "__entry_control", base, **common))
        if parent.get("mandatory_improved_start") or parent.get("mandatory_course_control"):
            # These are frozen comparison/starting controls. Their structural
            # expansions have their own focus-family parents; re-mining the
            # already optimized anchor families adds multiplicity, not a new
            # reversion hypothesis.
            continue
        for family in sorted(set(focus)):
            role = str(activation.get(family, {}).get("role", ""))
            family_filters = base.get(
                "param_overrides.pb_aperture_family_filters", {}
            )
            family_caps = base.get(
                "param_overrides.pb_aperture_family_daily_caps", {}
            )
            quality_protected = (
                isinstance(family_filters, dict) and family in family_filters
            ) or (isinstance(family_caps, dict) and family in family_caps)
            # One floor hypothesis per parent. A negative high-supply family
            # must first prove discrimination at 75 unless a rescue policy is
            # already protecting it; dormant routes use the breadth floor 65.
            floor = 75 if role == "high_supply_negative_standalone" and not quality_protected else 65
            mutations = _mapping_mutation(
                base,
                "param_overrides.pb_aperture_family_score_floors",
                family,
                floor,
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_floor{floor}",
                mutations,
                policy_atom=f"family_floor:{floor}",
                policy_family=family,
                **common,
            ))
            transition = _best_alternative_transition(activation, family)
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_{transition}",
                escape._family_transition_mutation(base, family, transition),
                policy_atom=f"transition:{transition}",
                policy_family=family,
                **common,
            ))
    return escape._dedupe(candidates)


def _entry_interaction_candidates(
    parents: list[dict[str, Any]],
    activation: dict[str, Any],
    atomic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Test pre-registered interactions only after breadth actually activates.

    These interactions directly address the observed global-floor/union
    non-linearity. The breadth floor must add focus-family trades and the
    alternative transition must independently change that family's economics
    before their single joint branch is opened. Rescue filters/caps are not
    repeated here.
    """

    candidates: list[dict[str, Any]] = []
    for parent in parents:
        base = parent["mutations"]
        families = list(parent.get("families", []))
        focus = list(parent.get("focus_families", [])) or families
        for family in sorted(set(focus)):
            role = str(activation.get(family, {}).get("role", ""))
            if role not in {
                "dormant_or_transition_blocked",
                "high_supply_negative_standalone",
            }:
                continue
            root_id = parent.get("root_id", parent["id"])
            matching = [
                row
                for row in atomic_rows
                if row.get("root_id") == root_id
                and family in row.get("focus_families", [])
            ]
            control = next(
                (
                    row for row in matching
                    if not row.get("policy_atom")
                    and _signature(row["mutations"]) == _signature(base)
                ),
                None,
            )
            family_filters = base.get(
                "param_overrides.pb_aperture_family_filters", {}
            )
            family_caps = base.get(
                "param_overrides.pb_aperture_family_daily_caps", {}
            )
            quality_protected = (
                isinstance(family_filters, dict) and family in family_filters
            ) or (isinstance(family_caps, dict) and family in family_caps)
            floor = 75 if role == "high_supply_negative_standalone" and not quality_protected else 65
            expected_floor = _mapping_mutation(
                base,
                "param_overrides.pb_aperture_family_score_floors",
                family,
                floor,
            )
            floor_row = next(
                (
                    row
                    for row in matching
                    if row.get("policy_family") == family
                    and row.get("policy_atom") == f"family_floor:{floor}"
                    and _signature(row["mutations"]) == _signature(expected_floor)
                ),
                None,
            )
            transition = _best_alternative_transition(activation, family)
            expected_transition = escape._family_transition_mutation(
                base, family, transition
            )
            transition_row = next(
                (
                    row
                    for row in matching
                    if row.get("policy_family") == family
                    and row.get("policy_atom") == f"transition:{transition}"
                    and _signature(row["mutations"]) == _signature(expected_transition)
                ),
                None,
            )
            if control is None or floor_row is None or transition_row is None:
                continue
            control_stats = _focus_family_stats(control, family)
            floor_stats = _focus_family_stats(floor_row, family)
            if floor_stats["trades"] <= control_stats["trades"]:
                continue
            transition_stats = _focus_family_stats(transition_row, family)
            transition_changed = (
                transition_stats["trades"] != control_stats["trades"]
                or abs(transition_stats["total_r"] - control_stats["total_r"]) > 1e-12
            )
            if not transition_changed:
                continue
            common = {
                "stage": "structural_entry_interaction",
                "families": families,
                "focus_families": [family],
                "root_id": root_id,
                "activation_evidence": {
                    "control_focus_trades": int(control_stats["trades"]),
                    f"floor{floor}_focus_trades": int(floor_stats["trades"]),
                    "transition_focus_trades": int(transition_stats["trades"]),
                },
                "structural_sources": ["activation_triggered_pre_registered_breadth_quality_interaction"],
            }
            floor_mutations = _mapping_mutation(
                base,
                "param_overrides.pb_aperture_family_score_floors",
                family,
                floor,
            )
            candidates.append(_candidate(
                parent["id"] + f"__{family.lower()}_floor{floor}_{transition}",
                escape._family_transition_mutation(floor_mutations, family, transition),
                **common,
            ))
    return escape._dedupe(candidates)


def _lean_management_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = (
        ("management_control", {}),
        ("size70", {"param_overrides.pb_aperture_sizing_mult": 0.70}),
        ("stale4", {"param_overrides.pb_aperture_stale_exit_bars": 4}),
    )
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        for name, delta in changes:
            mutations = dict(parent["mutations"])
            mutations.update(delta)
            candidates.append(
                _candidate(
                    parent["id"] + "__" + name,
                    mutations,
                    stage="structural_management",
                    families=parent.get("families", []),
                    focus_families=parent.get("focus_families", []),
                    root_id=parent.get("root_id", parent["id"]),
                )
            )
    return escape._dedupe(candidates)


def _management_parent_beam(
    rows: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Retain four complementary novel structures for management refinement."""

    eligible = [
        row
        for row in rows
        if not row.get("mandatory_improved_start")
        and not row.get("mandatory_course_control")
    ]
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}

    def admit(row: dict[str, Any] | None, reason: str) -> None:
        if row is None:
            return
        signature = _signature(row["mutations"])
        reasons.setdefault(signature, []).append(reason)
        if all(_signature(existing["mutations"]) != signature for existing in selected):
            selected.append(row)

    if eligible:
        admit(max(eligible, key=lambda row: float(row.get("escape_score", -99.0))), "top_score")
        admit(max(eligible, key=lambda row: float(row["metrics"].get("expected_total_r", -1e9))), "top_total_r")
        admit(max(eligible, key=lambda row: float(row["metrics"].get("total_trades", 0.0))), "top_frequency")
        admit(min(eligible, key=lambda row: float(row["metrics"].get("max_drawdown_pct", 1.0))), "lowest_drawdown")
        for row in sorted(
            _pareto(eligible),
            key=lambda value: float(value.get("escape_score", -99.0)),
            reverse=True,
        ):
            admit(row, "pareto_fill")
    selected = selected[: max(int(limit), 0)]
    retained = {_signature(row["mutations"]) for row in selected}
    return selected, {key: value for key, value in reasons.items() if key in retained}


def _mark_canonical_round_pending(output: Path) -> None:
    """Quarantine the prior false fold promotion before new research begins."""

    round3 = output.parent
    previous_summary = _load_json(round3 / "run_summary.json", {})
    invalidation_path = output / "invalidated_escape_promotion.json"
    if not invalidation_path.exists():
        _write_json(
            invalidation_path,
            {
                "status": "invalidated_fold_result_overwrite",
                "invalidated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "previous_selected_id": previous_summary.get("selected_id"),
                "reason": (
                    "full-period finalist fields overwrote chronological-fold replay results"
                ),
                "holdout_accessed": False,
            },
        )
    _write_json(
        round3 / "run_summary.json",
        {
            "status": "running_structural_challenger_pending_valid_folds",
            "previous_invalid_selected_id": previous_summary.get("selected_id"),
            "official": False,
            "holdout_accessed": False,
            "structural_challenger": "round_3/structural_challenger/queue_status.json",
        },
    )
    (round3 / "round_final_diagnostics.txt").write_text(
        "IARIC ROUND 3 — PENDING STRUCTURAL CHALLENGER\n"
        "=" * 56
        + "\nThe prior promotion was invalidated because full-period finalist fields "
        "overwrote chronological fold results. No Round 3 configuration is official "
        "until the fold-integrity gate passes. The sealed holdout remains unused.\n",
        encoding="utf-8",
    )
    manifest_path = round3.parent / "rounds_manifest.json"
    manifest = _load_json(manifest_path, {})
    rounds = list(manifest.get("rounds", []))
    for row in rounds:
        if int(row.get("round", -1)) == 3 and not row.get("archived", False):
            row["status"] = "invalidated_fold_result_overwrite"
            row["promotion_allowed"] = False
            row["official"] = False
    prior_rounds = [
        int(row.get("round", -1))
        for row in rounds
        if not row.get("archived", False) and int(row.get("round", -1)) < 3
    ]
    if prior_rounds:
        manifest["active_round"] = max(prior_rounds)
    manifest["pending_round_3"] = {
        "status": "running_structural_challenger_pending_valid_folds",
        "artifact": "round_3/structural_challenger/queue_status.json",
        "holdout_accessed": False,
    }
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["rounds"] = rounds
    _write_json(manifest_path, manifest)


def _promote(
    output: Path,
    selected: dict[str, Any],
    control: dict[str, Any],
    status: str,
) -> None:
    diagnostics = escape._diagnostics(selected, control, status)
    diagnostics += (
        "\nSTRUCTURAL CHALLENGER\n"
        "  Every implemented reversion family was retained at root activation.\n"
        "  Primary interactions covered the leader, least-overlap positive anchor, and\n"
        "  quota-balanced weak/weak pairs. Secondary contexts opened only after positive\n"
        "  executable focus-family marginal alpha. Pre-registered family floors,\n"
        "  geometry/participation filters,\n"
        "  daily caps, and causal transitions separated breadth from quality without adding\n"
        "  score components. Fold dates/results were immutable and the holdout stayed sealed.\n"
    )
    (output / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    if not selected["all_gates_pass"]:
        return
    if not bool(selected.get("validation_contract", {}).get("passed")):
        raise RuntimeError("Refusing to promote without an exact chronological-fold contract")
    round3 = output.parent
    _write_json(round3 / "optimized_config.json", selected["mutations"])
    _write_json(
        round3 / "run_summary.json",
        {
            "status": status,
            "selected_id": selected["id"],
            "metrics": selected["metrics"],
            "aperture": selected["aperture"],
            "gates": selected["gates"],
            "validation_contract": selected["validation_contract"],
            "official": True,
            "holdout_accessed": False,
            "structural_challenger": "round_3/structural_challenger/final_selection.json",
        },
    )
    (round3 / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    manifest_path = round3.parent / "rounds_manifest.json"
    manifest = _load_json(manifest_path, {})
    manifest["active_round"] = 3
    manifest.pop("pending_round_3", None)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "round": 3,
        "status": status,
        "configuration_role": "structural_local_maximum_escape_anchor_plus_conditional_reversion_routes",
        "mutations": selected["mutations"],
        "metrics": selected["metrics"],
        "aperture": selected["aperture"],
        "score_component_count": 7,
        "official": True,
        "validation_contract": selected["validation_contract"],
        "live_backtest_parity": {
            "family_policies_shared_core": True,
            "completed_bar_policy_unchanged": True,
            "causal_transitions_only": True,
        },
        "sealed_holdout": {"start": HOLDOUT_START, "used": False},
        "artifacts": {
            "optimized_config": "round_3/optimized_config.json",
            "full_final_diagnostics": "round_3/round_final_diagnostics.txt",
            "selection": "round_3/structural_challenger/final_selection.json",
        },
    }
    rounds = list(manifest.get("rounds", []))
    active = next(
        (
            index for index, row in enumerate(rounds)
            if int(row.get("round", -1)) == 3 and not row.get("archived", False)
        ),
        None,
    )
    if active is None:
        rounds.append(entry)
    else:
        rounds[active] = entry
    manifest["rounds"] = rounds
    _write_json(manifest_path, manifest)


def main() -> int:
    args = _args()
    escape_dir = Path(args.escape_dir).resolve()
    output = Path(args.output_dir).resolve()
    if int(args.max_workers) > 2:
        raise ValueError("Structural challenger is capped at max-workers=2")
    specification = _load_json(output / "run_spec.json")
    catalog = _load_json(output / "candidate_catalog.json")
    activation = _load_json(output / "activation_map.json")
    overlap = _load_json(output / "overlap_matrix.json")
    escape_spec = _load_json(escape_dir / "run_spec.json")
    course_selection = _load_json(escape_dir / "final_selection.json")
    phase0 = _load_json(escape_dir / "phase_0_route_isolation_results.json")
    composition_center = _load_json(escape_dir / "phase_1a_composition_center_results.json")
    if any(
        value is None
        for value in (
            specification,
            catalog,
            activation,
            overlap,
            escape_spec,
            course_selection,
            phase0,
            composition_center,
        )
    ):
        raise FileNotFoundError("Structural challenger inputs are incomplete")
    if specification.get("sealed_holdout", {}).get("accessed") is not False:
        raise ValueError("Structural catalog does not prove holdout exclusion")
    if str(escape_spec.get("end_date", "")) >= HOLDOUT_START:
        raise ValueError("Structural challenger end date overlaps the sealed holdout")
    if len(escape.SCORE_SPEC) != 7:
        raise RuntimeError("Structural challenger score must remain exactly seven components")
    output.mkdir(parents=True, exist_ok=True)
    current_source_fingerprint = escape._replay_source_fingerprint()
    current_code_fingerprint = _structural_code_fingerprint()
    if str(escape_spec["source_fingerprint"]) != current_source_fingerprint:
        raise RuntimeError("Replay source changed after structural preparation")
    _prepare_cache(
        escape_dir,
        output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
    )
    cache_reuse = _seed_screen_cache_from_full(output)
    _mark_canonical_round_pending(output)
    specification["schema_version"] = 2
    specification["status"] = "running_activation_first_structural_challenger"
    specification["runtime_source_fingerprint"] = current_source_fingerprint
    specification["runtime_code_fingerprint"] = current_code_fingerprint
    specification["live_backtest_contract"].update({
        "family_score_floors_shared_core": True,
        "family_filters_shared_core": True,
        "family_daily_caps_live_replay_aligned": True,
        "chronological_fold_result_fields_immutable": True,
    })
    specification["adaptive_execution"] = {
        "activation_roots_first": True,
        "baseline_roles": {
            "frozen_no_drift_comparator": "89-trade incumbent",
            "optimization_start_and_value_gate": (
                "133-trade FAILED_BREAKDOWN_RECLAIM+UPTREND_PULLBACK_RECLAIM"
            ),
        },
        "structural_screen": (
            "execution-only generation followed by mandatory full-diagnostics "
            "replay of every retained family representative"
        ),
        "primary_interactions": (
            "leader-plus-weak, least-overlap-anchor-plus-weak, and quota-balanced weak pairs"
        ),
        "secondary_interactions": (
            "opened only after positive executable focus-family marginal alpha"
        ),
        "rescue_budget": {
            "stage_1_role_aware_maximum": 24,
            "stage_2_evidence_triggered_maximum": 8,
        },
        "structural_parent_beam": (
            "133-trade start + course reference + up to four diverse positive-focus parents"
        ),
        "pre_registered_family_policies": {
            "score_floors": [65, 75],
            "filters": ["geometry", "participation"],
            "daily_caps": [1, 2],
        },
        "score_component_count": len(escape.SCORE_SPEC),
        "cache_reuse": cache_reuse,
    }
    _write_json(output / "run_spec.json", specification)

    frozen_control = next(row for row in phase0 if row["id"] == "incumbent_control")
    baseline = frozen_control["mutations"]
    course_selected = course_selection["selected"]
    eval_args = argparse.Namespace(
        start_date=str(escape_spec["start_date"]),
        end_date=str(escape_spec["end_date"]),
        max_workers=min(max(int(args.max_workers), 1), 2),
    )

    all_root_candidates = _root_candidates(catalog, baseline, course_selected)
    improved_candidate = next(
        row for row in all_root_candidates if row["id"] == "improved_start_control"
    )
    improved_signature = _signature(improved_candidate["mutations"])
    archived_start = next(
        (
            row
            for row in composition_center
            if _signature(row["mutations"]) == improved_signature
        ),
        None,
    )
    if archived_start is None:
        raise RuntimeError("The archived 133-trade improved starting point is missing")
    activation_candidates = _activation_root_candidates(all_root_candidates)
    structural_screen_kwargs = {
        "evaluation_fn": evaluate_candidate_execution_attribution,
        "cache_filename": "structural_screen_cache.json",
        "score_metrics_fn": _execution_screen_score_metrics,
    }
    # Re-establish both controls from the current fingerprint. These are
    # signature cache hits when previous work is compatible; no broad search
    # is repeated. The 89-trade row verifies no drift, while all optimization
    # scores and value gates use the 133-trade row.
    full_start = escape._evaluate(
        "phase_0_starting_control_full",
        [improved_candidate],
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=archived_start["metrics"],
    )[0]
    fast_start = escape._evaluate(
        "phase_0_starting_control_screen",
        [improved_candidate],
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=archived_start["metrics"],
        **structural_screen_kwargs,
    )[0]
    _progress(
        output,
        "running_activation_roots",
        requested=len(activation_candidates),
        total_catalog_roots=len(all_root_candidates),
    )
    activation_roots = escape._evaluate(
        "phase_0a_activation_roots",
        activation_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=fast_start["metrics"],
        **structural_screen_kwargs,
    )
    frozen_replay = next(row for row in activation_roots if row["id"] == "incumbent_control")
    control = next(row for row in activation_roots if row["id"] == "improved_start_control")
    parity_metrics = ("total_trades", "expected_total_r", "profit_factor", "max_drawdown_pct")
    frozen_deltas = {
        key: float(frozen_replay["metrics"].get(key, 0.0))
        - float(frozen_control["metrics"].get(key, 0.0))
        for key in parity_metrics
    }
    full_start_deltas = {
        key: float(full_start["metrics"].get(key, 0.0))
        - float(archived_start["metrics"].get(key, 0.0))
        for key in parity_metrics
    }
    fast_start_deltas = {
        key: float(control["metrics"].get(key, 0.0))
        - float(archived_start["metrics"].get(key, 0.0))
        for key in parity_metrics
    }
    parity_passed = all(
        abs(value) <= 1e-12
        for value in (*frozen_deltas.values(), *full_start_deltas.values(), *fast_start_deltas.values())
    )
    _write_json(
        output / "baseline_parity_report.json",
        {
            "passed": parity_passed,
            "frozen_comparator": {
                "mutation_signature_match": _signature(frozen_replay["mutations"])
                == _signature(frozen_control["mutations"]),
                "metric_deltas": frozen_deltas,
                "authoritative_total_trades": frozen_replay["metrics"].get("total_trades"),
            },
            "improved_start": {
                "mutation_signature_match": _signature(control["mutations"])
                == improved_signature,
                "full_diagnostics_metric_deltas": full_start_deltas,
                "execution_screen_metric_deltas": fast_start_deltas,
                "authoritative_total_trades": full_start["metrics"].get("total_trades"),
                "expected_total_r": full_start["metrics"].get("expected_total_r"),
                "profit_factor": full_start["metrics"].get("profit_factor"),
                "max_drawdown_pct": full_start["metrics"].get("max_drawdown_pct"),
            },
        },
    )
    if not parity_passed:
        raise RuntimeError("Frozen or improved structural control drifted from its archive")

    primary_ids = _primary_interaction_ids(catalog, overlap)
    primary_candidates = [row for row in all_root_candidates if row["id"] in primary_ids]
    _progress(
        output,
        "running_primary_interactions",
        requested=len(primary_candidates),
        family_coverage=len(catalog["weak_or_dormant_families"]),
    )
    primary_interactions = escape._evaluate(
        "phase_0b_primary_interactions",
        primary_candidates,
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=control["metrics"],
        **structural_screen_kwargs,
    )

    rescue_stage1_candidates = _activation_rescue_candidates(
        primary_interactions,
        activation,
        catalog["weak_or_dormant_families"],
    )
    rescue_stage1_rows: list[dict[str, Any]] = []
    if rescue_stage1_candidates:
        _progress(
            output,
            "running_activation_rescue_atoms",
            requested=len(rescue_stage1_candidates),
        )
        rescue_stage1_rows = escape._evaluate(
            "phase_0c1_activation_rescue_atoms",
            rescue_stage1_candidates,
            args=eval_args,
            output=output,
            source_fingerprint=current_source_fingerprint,
            code_fingerprint=current_code_fingerprint,
            control_metrics=control["metrics"],
            **structural_screen_kwargs,
        )
    rescue_followup_candidates = _activation_rescue_followup_candidates(
        primary_interactions,
        rescue_stage1_rows,
        activation,
        catalog["weak_or_dormant_families"],
    )
    rescue_followup_rows: list[dict[str, Any]] = []
    if rescue_followup_candidates:
        _progress(
            output,
            "running_activation_rescue_followups",
            requested=len(rescue_followup_candidates),
        )
        rescue_followup_rows = escape._evaluate(
            "phase_0c2_activation_rescue_followups",
            rescue_followup_candidates,
            args=eval_args,
            output=output,
            source_fingerprint=current_source_fingerprint,
            code_fingerprint=current_code_fingerprint,
            control_metrics=control["metrics"],
            **structural_screen_kwargs,
        )
    rescue_rows = _combine_by_signature(
        rescue_stage1_rows,
        rescue_followup_rows,
    )

    root_evidence = _combine_by_signature(primary_interactions, rescue_rows)
    conditional_candidates = _conditional_interaction_candidates(
        all_root_candidates,
        primary_ids,
        root_evidence,
        catalog["weak_or_dormant_families"],
        catalog["positive_anchors"],
    )
    conditional_rows: list[dict[str, Any]] = []
    if conditional_candidates:
        _progress(
            output,
            "running_activation_supported_interactions",
            requested=len(conditional_candidates),
        )
        conditional_rows = escape._evaluate(
            "phase_0d_conditional_interactions",
            conditional_candidates,
            args=eval_args,
            output=output,
            source_fingerprint=current_source_fingerprint,
            code_fingerprint=current_code_fingerprint,
            control_metrics=control["metrics"],
            **structural_screen_kwargs,
        )

    roots = _combine_by_signature(
        activation_roots,
        primary_interactions,
        rescue_rows,
        conditional_rows,
    )
    _write_json(output / "phase_0_structural_roots_results.json", roots)
    screen_parents, screen_reasons = _structural_beam(
        roots,
        control,
        catalog["weak_or_dormant_families"],
        soft_limit=int(catalog["adaptive_followups"]["root_parent_soft_limit"]),
        require_focus_positive=True,
        family_limit=4,
    )
    if not screen_parents:
        raise RuntimeError("No structural root created positive executable conditional alpha")
    _write_json(
        output / "phase_0_screen_parent_selection.json",
        {"parents": screen_parents, "reasons": screen_reasons},
    )

    # The fast screen has no authority over discrimination or promotion.  Run
    # every coverage-preserved representative through the full candidate
    # ledger before entry refinement so negative and low-quality rejected
    # signals remain part of the decision.
    _progress(
        output,
        "running_full_diagnostics_parent_verification",
        parents=len(screen_parents),
    )
    diagnostic_parents = escape._evaluate(
        "phase_0e_full_diagnostics_parents",
        [escape._replay_candidate(row) for row in screen_parents],
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=full_start["metrics"],
    )
    root_parents, root_reasons = _structural_beam(
        diagnostic_parents,
        full_start,
        catalog["weak_or_dormant_families"],
        soft_limit=int(catalog["adaptive_followups"]["root_parent_soft_limit"]),
        require_focus_positive=True,
        family_limit=4,
    )
    if not root_parents:
        raise RuntimeError("Full diagnostics rejected every structural parent")
    _write_json(
        output / "phase_0_parent_selection.json",
        {"parents": root_parents, "reasons": root_reasons},
    )

    _progress(output, "running_structural_entry_atoms", parents=len(root_parents))
    entry_atomic = escape._evaluate(
        "phase_1a_structural_entry_atoms",
        _entry_candidates(root_parents, activation),
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=full_start["metrics"],
    )
    interaction_candidates = _entry_interaction_candidates(
        root_parents,
        activation,
        entry_atomic,
    )
    entry_interactions: list[dict[str, Any]] = []
    if interaction_candidates:
        _progress(
            output,
            "running_structural_entry_interactions",
            requested=len(interaction_candidates),
        )
        entry_interactions = escape._evaluate(
            "phase_1b_structural_entry_interactions",
            interaction_candidates,
            args=eval_args,
            output=output,
            source_fingerprint=current_source_fingerprint,
            code_fingerprint=current_code_fingerprint,
            control_metrics=full_start["metrics"],
        )
    entry = _combine_by_signature(entry_atomic, entry_interactions)
    _write_json(output / "phase_1_structural_entry_results.json", entry)
    entry_parents, entry_reasons = _structural_beam(
        entry,
        full_start,
        catalog["weak_or_dormant_families"],
        soft_limit=int(catalog["adaptive_followups"]["root_parent_soft_limit"]),
        require_focus_positive=True,
        family_limit=4,
    )
    if not entry_parents:
        raise RuntimeError("Structural entry refinement lost every viable root")
    _write_json(output / "phase_1_parent_selection.json", {"parents": entry_parents, "reasons": entry_reasons})

    management_parents, management_parent_reasons = _management_parent_beam(
        entry_parents,
        limit=4,
    )
    if not management_parents:
        raise RuntimeError("No novel positive-focus parent survived for management")
    _write_json(
        output / "phase_1_management_parent_selection.json",
        {"parents": management_parents, "reasons": management_parent_reasons},
    )
    _progress(output, "running_lean_management", parents=len(management_parents))
    management = escape._evaluate(
        "phase_2_lean_management",
        _lean_management_candidates(management_parents),
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
        control_metrics=full_start["metrics"],
    )
    course_signature = _signature(course_selected["mutations"])
    start_signature = _signature(full_start["mutations"])
    management_challengers = [
        row
        for row in management
        if _signature(row["mutations"]) != start_signature
    ]
    finalists, finalist_reasons = _broad_validation_shortlist(
        management_challengers,
        full_start,
        mandatory_signatures=[course_signature],
        limit=int(catalog["adaptive_followups"]["validation_limit"]),
    )
    # The course control may dedupe out of management when it was not a viable
    # structural parent. Preserve it explicitly so the challenger cannot win by
    # excluding the incumbent comparison.
    if all(_signature(row["mutations"]) != course_signature for row in finalists):
        course_row = next(
            (
                row
                for row in diagnostic_parents
                if _signature(row["mutations"]) == course_signature
            ),
            None,
        )
        if course_row is not None:
            finalists.insert(0, course_row)
    finalists = escape._dedupe(finalists)[
        : int(catalog["adaptive_followups"]["validation_limit"])
    ]
    if not finalists:
        raise RuntimeError("Structural challenger produced no validation finalists")
    _write_json(
        output / "validation_shortlist.json",
        {"finalists": finalists, "selection_reasons": finalist_reasons},
    )
    finalists = [deepcopy(row) for row in finalists]
    for row in finalists:
        row.pop("folds", None)
        row.pop("gates", None)
        row.pop("all_gates_pass", None)

    _progress(output, "running_chronological_validation", finalists=len(finalists))
    escape._fold_validate(
        finalists,
        full_start,
        args=eval_args,
        output=output,
        source_fingerprint=current_source_fingerprint,
        code_fingerprint=current_code_fingerprint,
    )
    for row in finalists:
        focus_families = sorted(set(row.get("focus_families", [])))
        focus_audit = {
            family: _focus_family_stats(row, family)
            for family in focus_families
        }
        row["focus_family_marginal_alpha"] = focus_audit
        row["gates"] = escape._gates(row, full_start)
        row["gates"]["positive_focus_family_marginal_alpha"] = bool(focus_audit) and all(
            stats["trades"] >= 3 and stats["total_r"] > 0.0
            for stats in focus_audit.values()
        )
        row["all_gates_pass"] = all(row["gates"].values())
    finalists.sort(key=_validated_rank, reverse=True)
    selected = finalists[0]
    status = "complete_value_verified" if selected["all_gates_pass"] else "blocked_value_verification"
    _write_json(output / "validated_finalists.json", finalists)
    _write_json(
        output / "final_selection.json",
        {
            "status": status,
            "selected": selected,
            "control": full_start,
            "frozen_no_drift_comparator": frozen_control,
        },
    )
    _promote(output, selected, full_start, status)
    _progress(
        output,
        "complete",
        result_status=status,
        selected_id=selected["id"],
        all_gates_pass=selected["all_gates_pass"],
        holdout_accessed=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
