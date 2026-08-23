"""Prepare a pre-holdout structural challenger without running replays.

The preparation is intentionally safe to run beside the active escape search:
it reads source-fingerprinted attribution and the opportunity atlas, writes to a
separate output directory, and never imports or mutates strategy state.  The
resulting catalog is frozen before the challenger sees validation outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
IARIC_DIR = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_ESCAPE = IARIC_DIR / "round_3/escape_round"
DEFAULT_ATLAS = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
DEFAULT_OUTPUT = IARIC_DIR / "round_3/structural_challenger"
HOLDOUT_START = "2026-03-02"
FAMILIES = (
    "GAP_EXHAUSTION_RECLAIM",
    "GAP_FILL_RECLAIM",
    "OPENING_FLUSH_RECLAIM",
    "OPENING_RANGE_LOW_RECLAIM",
    "PRIOR_DAY_LOW_RECLAIM",
    "VWAP_DEVIATION_RECLAIM",
    "FAILED_BREAKDOWN_RECLAIM",
    "MARKET_SECTOR_RESIDUAL_RECLAIM",
    "MULTIDAY_HIGHER_LOW_RECLAIM",
    "UPTREND_PULLBACK_RECLAIM",
    "VOLUME_CLIMAX_RECLAIM",
)
TRANSITIONS = {
    "next_bar": "next_bar_open",
    "confirm": "one_bar_confirmation",
    "retrace": "resting_25pct_retrace",
}
SCORE_THRESHOLDS = (60.0, 70.0, 75.0)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escape-dir", default=str(DEFAULT_ESCAPE))
    parser.add_argument("--atlas-dir", default=str(DEFAULT_ATLAS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.preparing.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(*values: str) -> int:
    digest = hashlib.blake2b("|".join(values).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _outcome(record: dict[str, Any], variant: str) -> dict[str, Any] | None:
    if variant == "next_bar_open":
        candidate = record.get("entry_variants", {}).get(variant)
        return candidate if isinstance(candidate, dict) else record
    candidate = record.get("entry_variants", {}).get(variant)
    return candidate if isinstance(candidate, dict) else None


def _empty_metric() -> dict[str, Any]:
    return {"events": 0, "total_r": 0.0, "wins_r": 0.0, "losses_r": 0.0}


def _add_metric(metric: dict[str, Any], value: float) -> None:
    metric["events"] += 1
    metric["total_r"] += value
    if value > 0:
        metric["wins_r"] += value
    elif value < 0:
        metric["losses_r"] += value


def _finish_metric(metric: dict[str, Any]) -> dict[str, Any]:
    events = int(metric["events"])
    losses = abs(float(metric["losses_r"]))
    wins = float(metric["wins_r"])
    return {
        "events": events,
        "total_r": float(metric["total_r"]),
        "avg_r": float(metric["total_r"]) / events if events else 0.0,
        "profit_factor": wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0),
    }


def _atlas_activation(events_path: Path) -> tuple[dict[str, Any], dict[str, set[int]], dict[str, set[int]]]:
    counters: dict[str, Any] = {}
    exact_keys = {family: set() for family in FAMILIES}
    symbol_day_keys = {family: set() for family in FAMILIES}
    for family in FAMILIES:
        counters[family] = {
            "events": 0,
            "fold_events": defaultdict(int),
            "score_thresholds": {
                str(int(threshold)): {
                    fold: _empty_metric() for fold in ("early", "middle", "latest")
                }
                for threshold in SCORE_THRESHOLDS
            },
            "transitions": {
                transition: {
                    fold: _empty_metric() for fold in ("early", "middle", "latest")
                }
                for transition in TRANSITIONS
            },
        }

    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            family = str(record.get("family", ""))
            if family not in counters:
                continue
            fold = str(record.get("fold", ""))
            if fold not in ("early", "middle", "latest"):
                continue
            stats = counters[family]
            stats["events"] += 1
            stats["fold_events"][fold] += 1
            exact_keys[family].add(
                _stable_key(
                    str(record.get("date", "")),
                    str(record.get("symbol", "")),
                    str(record.get("signal_time", "")),
                )
            )
            symbol_day_keys[family].add(
                _stable_key(str(record.get("date", "")), str(record.get("symbol", "")))
            )
            score = float(record.get("score", 0.0))
            next_open = _outcome(record, "next_bar_open")
            if next_open is not None:
                value = float(next_open.get("horizon_r", {}).get("bar_12", 0.0))
                for threshold in SCORE_THRESHOLDS:
                    if score >= threshold:
                        _add_metric(stats["score_thresholds"][str(int(threshold))][fold], value)
            for transition, variant in TRANSITIONS.items():
                candidate = _outcome(record, variant)
                if candidate is None:
                    continue
                value = float(candidate.get("horizon_r", {}).get("bar_12", 0.0))
                _add_metric(stats["transitions"][transition][fold], value)

    finished: dict[str, Any] = {}
    for family, stats in counters.items():
        finished[family] = {
            "events": int(stats["events"]),
            "fold_events": dict(stats["fold_events"]),
            "score_thresholds": {
                threshold: {
                    fold: _finish_metric(metric) for fold, metric in folds.items()
                }
                for threshold, folds in stats["score_thresholds"].items()
            },
            "transitions": {
                transition: {
                    fold: _finish_metric(metric) for fold, metric in folds.items()
                }
                for transition, folds in stats["transitions"].items()
            },
        }
    return finished, exact_keys, symbol_day_keys


def _jaccard(left: set[int], right: set[int]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _overlap_report(
    exact: dict[str, set[int]], symbol_day: dict[str, set[int]],
) -> dict[str, Any]:
    exact_matrix: dict[str, dict[str, float]] = {}
    day_matrix: dict[str, dict[str, float]] = {}
    for left in FAMILIES:
        exact_matrix[left] = {}
        day_matrix[left] = {}
        for right in FAMILIES:
            exact_matrix[left][right] = _jaccard(exact[left], exact[right])
            day_matrix[left][right] = _jaccard(symbol_day[left], symbol_day[right])
    return {
        "definition": {
            "exact_event": "Jaccard(date, symbol, causal signal timestamp)",
            "symbol_day": "Jaccard(date, symbol); measures capital-timing overlap",
        },
        "exact_event_jaccard": exact_matrix,
        "symbol_day_jaccard": day_matrix,
    }


def _isolation_map(escape_dir: Path) -> tuple[dict[str, Any], list[str]]:
    rows = _load_json(escape_dir / "phase_0_route_isolation_results.json")
    result: dict[str, Any] = {}
    for row in rows:
        families = row.get("families", [])
        if len(families) != 1:
            continue
        family = str(families[0])
        aperture = row.get("aperture", {})
        funnel = row.get("funnel_counters", {})
        result[family] = {
            "escape_score": float(row.get("escape_score", 0.0)),
            "portfolio_trades": int(row.get("metrics", {}).get("total_trades", 0)),
            "portfolio_total_r": float(row.get("metrics", {}).get("expected_total_r", 0.0)),
            "portfolio_profit_factor": float(row.get("metrics", {}).get("profit_factor", 0.0)),
            "portfolio_max_drawdown_pct": float(row.get("metrics", {}).get("max_drawdown_pct", 1.0)),
            "route_ready": int(funnel.get("aperture_ready", 0) or 0),
            "route_trades": int(aperture.get("trades", 0) or 0),
            "route_total_r": float(aperture.get("total_r", 0.0)),
            "route_avg_r": float(aperture.get("avg_r", 0.0)),
            "route_profit_factor": float(aperture.get("profit_factor", 0.0)),
        }
    strong = sorted(
        (
            family for family, row in result.items()
            if row["route_trades"] >= 3
            and row["route_total_r"] > 0
            and row["route_profit_factor"] >= 1.05
        ),
        key=lambda family: result[family]["escape_score"],
        reverse=True,
    )
    return result, strong


def _role(row: dict[str, Any]) -> str:
    if row["route_trades"] >= 3 and row["route_total_r"] > 0 and row["route_profit_factor"] >= 1.05:
        return "positive_executable_anchor"
    if row["route_trades"] < 10:
        return "dormant_or_transition_blocked"
    if row["route_trades"] >= 25:
        return "high_supply_negative_standalone"
    return "low_quality_standalone"


def _pair_key(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(str(value) for value in values)))


def _catalog(
    isolation: dict[str, Any],
    strong: list[str],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    if not strong:
        raise RuntimeError("Structural challenger requires at least one positive executable anchor")
    leader_pair = _pair_key(("FAILED_BREAKDOWN_RECLAIM", "UPTREND_PULLBACK_RECLAIM"))
    anchors = [family for family in strong if family in FAMILIES]
    weak = [family for family in FAMILIES if family not in anchors]
    day_matrix = overlap["symbol_day_jaccard"]
    all_weak_pairs = [
        _pair_key((left, right))
        for index, left in enumerate(weak)
        for right in weak[index + 1 :]
    ]

    def pair_rank(pair: tuple[str, ...]) -> tuple[float, int, tuple[str, ...]]:
        return (
            float(day_matrix[pair[0]][pair[1]]),
            -(int(isolation[pair[0]]["route_ready"]) + int(isolation[pair[1]]["route_ready"])),
            pair,
        )

    role_by_family = {family: _role(isolation[family]) for family in weak}
    dormant = "dormant_or_transition_blocked"
    high_supply = "high_supply_negative_standalone"
    pair_buckets = {
        "dormant_dormant": sorted(
            (
                pair for pair in all_weak_pairs
                if role_by_family[pair[0]] == dormant and role_by_family[pair[1]] == dormant
            ),
            key=pair_rank,
        ),
        "dormant_high_supply": sorted(
            (
                pair for pair in all_weak_pairs
                if {role_by_family[pair[0]], role_by_family[pair[1]]} == {dormant, high_supply}
            ),
            key=pair_rank,
        ),
        "high_supply_high_supply": sorted(
            (
                pair for pair in all_weak_pairs
                if role_by_family[pair[0]] == high_supply and role_by_family[pair[1]] == high_supply
            ),
            key=pair_rank,
        ),
    }
    weak_pairs = (
        pair_buckets["dormant_dormant"][:2]
        + pair_buckets["dormant_high_supply"][:3]
        + pair_buckets["high_supply_high_supply"][:3]
    )
    if len(weak_pairs) < 8:
        for pair in sorted(all_weak_pairs, key=pair_rank):
            if pair not in weak_pairs:
                weak_pairs.append(pair)
            if len(weak_pairs) >= 8:
                break

    candidates: dict[tuple[str, ...], dict[str, Any]] = {}

    def add(families: Iterable[str], source: str, focus: Iterable[str]) -> None:
        family_set = _pair_key(families)
        if not family_set:
            return
        record = candidates.setdefault(
            family_set,
            {
                "id": "root__" + "_".join(value.lower() for value in family_set),
                "families": list(family_set),
                "event_score_min": 70.0,
                "focus_families": sorted(set(str(value) for value in focus)),
                "sources": [],
            },
        )
        record["sources"].append(source)
        record["sources"] = sorted(set(record["sources"]))

    for family in FAMILIES:
        add((family,), "mandatory_single_family_activation", (family,))
    for anchor in anchors:
        for family in weak:
            add((anchor, family), "positive_anchor_plus_weak_family", (family,))
    for family in weak:
        add((*leader_pair, family), "leader_pair_plus_orthogonal_family", (family,))
    for pair in weak_pairs:
        add(pair, "weak_alone_orthogonal_pair", pair)

    return {
        "schema_version": 1,
        "frozen_before_challenger_validation": True,
        "generation_policy": {
            "single_family_activation": "retain every implemented reversion family regardless of standalone quality",
            "anchor_interactions": "pair every positive executable anchor with every weak/dormant family",
            "leader_expansion": "add every weak/dormant family to the current two-family leader",
            "weak_interactions": "retain eight lowest symbol-day-overlap weak-family pairs; tie-break by executable supply",
            "weak_interaction_quotas": {
                "dormant_dormant": 2,
                "dormant_high_supply": 3,
                "high_supply_high_supply": 3,
            },
            "outcome_usage": "outcomes classify anchors but do not rank weak-family interactions",
        },
        "positive_anchors": anchors,
        "current_leader_pair": list(leader_pair),
        "weak_or_dormant_families": weak,
        "weak_orthogonal_pairs": [list(pair) for pair in weak_pairs],
        "root_candidates": sorted(candidates.values(), key=lambda row: row["id"]),
        "adaptive_followups": {
            # The 133-trade improved start, the prior course reference, and up
            # to four diverse representatives with positive focus-family
            # marginal alpha. This preserves hypotheses without carrying
            # losing routes behind a strong anchor.
            "root_parent_soft_limit": 6,
            "preserve_best_per_root_family": True,
            "preserve_pareto_front": True,
            "preserve_top_frequency": True,
            "entry_floor_variants": [65.0, 75.0],
            "family_transition_variants": ["next_bar", "confirm", "retrace"],
            "transition_scope": "focus families only",
            "management_variants": [
                "control",
                "param_overrides.pb_aperture_sizing_mult=0.70",
                "param_overrides.pb_aperture_stale_exit_bars=4",
            ],
            "validation_limit": 5,
        },
    }


def _report(
    activation: dict[str, Any], catalog: dict[str, Any], escape_dir: Path,
) -> str:
    lines = [
        "# IARIC structural challenger — parallel preparation",
        "",
        "This is a pre-holdout, shared-core-compatible challenger to the narrow escape-round beam.",
        "It does not treat unconditional reversion as alpha; it searches for executable conditional",
        "cohorts while preserving every implemented family and low-overlap interactions.",
        "",
        "## Activation diagnosis",
        "",
        "| Family | Role | Atlas events | Ready | Trades | Route R | Route PF |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = activation[family]
        lines.append(
            f"| {family} | {row['role']} | {row['atlas']['events']} | "
            f"{row['isolation']['route_ready']} | {row['isolation']['route_trades']} | "
            f"{row['isolation']['route_total_r']:+.2f} | {row['isolation']['route_profit_factor']:.2f} |"
        )
    lines += [
        "",
        "## Frozen search allocation",
        "",
        f"- Root candidates: {len(catalog['root_candidates'])}",
        f"- Positive executable anchors: {', '.join(catalog['positive_anchors'])}",
        f"- Weak/dormant families retained: {len(catalog['weak_or_dormant_families'])}",
        "- Later phases preserve family coverage, frequency, and the Pareto frontier rather than only the top scalar score.",
        "- Management excludes settings already proven behaviorally inert in the active escape round.",
        "- Replays remain capped at two workers and cannot start until the active continuation releases them.",
        "",
        "## Integrity boundary",
        "",
        f"- Active escape source: `{escape_dir}`",
        f"- Training ends: 2026-03-01; sealed holdout begins: {HOLDOUT_START}.",
        "- Existing route composition only; no live/backtest decision logic is changed by preparation.",
        "- Any future route implementation must enter the typed shared core and pass completed-bar, causal-fill, adapter-equivalence, and replay/live parity tests.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    escape_dir = Path(args.escape_dir).resolve()
    atlas_dir = Path(args.atlas_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_spec = _load_json(escape_dir / "run_spec.json")
    if run_spec.get("holdout_accessed") is not False or str(run_spec.get("end_date", "")) >= HOLDOUT_START:
        raise ValueError("Structural preparation requires explicit sealed-holdout exclusion")

    events_path = atlas_dir / "events.jsonl"
    atlas, exact, symbol_day = _atlas_activation(events_path)
    overlap = _overlap_report(exact, symbol_day)
    isolation, strong = _isolation_map(escape_dir)
    activation = {
        family: {
            "role": _role(isolation[family]),
            "atlas": atlas[family],
            "isolation": isolation[family],
            "exact_overlap_with_current_leader": {
                leader: overlap["exact_event_jaccard"][family][leader]
                for leader in ("FAILED_BREAKDOWN_RECLAIM", "UPTREND_PULLBACK_RECLAIM")
            },
            "symbol_day_overlap_with_current_leader": {
                leader: overlap["symbol_day_jaccard"][family][leader]
                for leader in ("FAILED_BREAKDOWN_RECLAIM", "UPTREND_PULLBACK_RECLAIM")
            },
        }
        for family in FAMILIES
    }
    catalog = _catalog(isolation, strong, overlap)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    specification = {
        "schema_version": 1,
        "status": "prepared_waiting_for_active_continuation",
        "objective": "escape the narrow local optimum by increasing real conditional reversion alpha and executable frequency",
        "training_window": {"start": run_spec["start_date"], "end": run_spec["end_date"]},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": 2,
        "score": {"component_count": 7, "source": "escape_round/run_spec.json", "immutable": True},
        "live_backtest_contract": {
            "existing_shared_core_routes_only": True,
            "completed_bar_policy_unchanged": True,
            "causal_entry_transitions_only": ["next_bar", "confirm", "retrace"],
            "fingerprinted_strategy_files_modified_by_preparation": False,
        },
        "inputs": {
            "escape_run_spec": str(escape_dir / "run_spec.json"),
            "escape_code_fingerprint": run_spec["code_fingerprint"],
            "escape_source_fingerprint": run_spec["source_fingerprint"],
            "atlas_events": str(events_path),
            "atlas_events_sha256": _file_sha256(events_path),
            "route_isolation_sha256": _file_sha256(escape_dir / "phase_0_route_isolation_results.json"),
        },
        "generated_at_utc": generated,
    }
    deferred_extension = {
        "status": "pre_registered_deferred_until_active_fingerprint_released",
        "purpose": "allow broad route supply without forcing one global floor/filter onto economically different reversion families",
        "activation_trigger": (
            "execute only if the existing-route structural challenger confirms that a weak/dormant "
            "family adds opportunities but loses value under the shared global discriminator"
        ),
        "shared_core_changes": [
            {
                "setting": "pb_aperture_family_score_floors",
                "format": "FAMILY:65|70|75",
                "test_values": [65.0, 70.0, 75.0],
                "reason": "separate breadth for dormant families from quality floors for high-supply negative families",
            },
            {
                "setting": "pb_aperture_family_filters",
                "fixed_policies": {
                    "geometry": {"score_min": 40.0, "reclaim_component_min": 0.40, "close_quality_min": 0.60},
                    "participation": {"score_min": 40.0, "relative_volume_component_min": 0.25},
                },
                "reason": "reuse pre-registered atlas policies rather than mine new small-sample thresholds",
            },
            {
                "setting": "pb_aperture_family_daily_caps",
                "test_values": [1, 2],
                "reason": "prevent a broad negative family from crowding out distinct positive routes under shared capital",
            },
        ],
        "implementation_contract": {
            "decision_owner": "strategies/stock/iaric/core/logic.py",
            "typed_configuration": "strategies/stock/iaric/config.py",
            "live_and_replay_call_same_policy": True,
            "completed_bar_policy_unchanged": True,
            "causal_fill_policy_unchanged": True,
            "neutral_actions_unchanged": True,
            "score_component_count": 7,
            "required_tests": [
                "family policy parser rejects unknown family/filter/value",
                "default empty mapping reproduces frozen decisions exactly",
                "live/replay adapters produce identical family admission decisions",
                "signal-bar close cannot fill before the next bar unless a resting order pre-existed",
                "per-family caps are deterministic under symbol-order permutations",
                "snapshot/hydration preserves family policy state",
            ],
        },
        "anti_overfit_rules": [
            "use only the frozen floors and atlas geometry/participation policies above",
            "test one family-policy change at a time before combinations",
            "preserve every root-family control and the current course finalist",
            "require positive incremental R, positive marginal expectancy, bounded drawdown, and chronological consistency",
            "do not access the sealed holdout",
        ],
    }
    _write_json(output / "activation_map.json", activation)
    _write_json(output / "overlap_matrix.json", overlap)
    _write_json(output / "candidate_catalog.json", catalog)
    _write_json(output / "run_spec.json", specification)
    _write_json(output / "deferred_shared_core_extension.json", deferred_extension)
    _write_json(
        output / "queue_status.json",
        {
            "status": "prepared_waiting_for_active_continuation",
            "root_candidates": len(catalog["root_candidates"]),
            "holdout_accessed": False,
            "updated_at_utc": generated,
        },
    )
    (output / "preparation_report.md").write_text(
        _report(activation, catalog, escape_dir), encoding="utf-8"
    )
    print(
        f"prepared {len(catalog['root_candidates'])} structural roots at {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
