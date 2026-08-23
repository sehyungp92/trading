"""Focused corrected-split extension of the completed Round-2 IARIC search.

The completed broad sweep is re-scored from its cached trades using IS
2024-03-25..2026-03-01 and OOS 2026-03-02..2026-05-01.  Only controls, a
shortlist, and genuinely new targeted combinations receive exact continuous
corrected-window replays.  The March-May OOS has already informed research,
so every recommendation remains research-only pending later untouched data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from backtests.scripts import iaric_round2_residual_oos_ablation as base
from strategies.stock.iaric.config import StrategySettings


OLD_ROOT = base.ROUND_ROOT / "oos_ablation_perturbation_20260823"
DEFAULT_OUTPUT = base.ROUND_ROOT / "corrected_split_targeted_extension_20260823"
SHARED_PREPARED_CACHE = OLD_ROOT / "prepared_feature_cache"


def _cached_path(candidate: Mapping[str, Any], window: str) -> Path:
    signature = str(candidate["settings_sha256"])[:16]
    return OLD_ROOT / "cache" / f"{signature}__{window}__20bps.json"


def _catalog() -> list[dict[str, Any]]:
    rows = list(base._read_json(OLD_ROOT / "candidate_catalog.json"))
    targeted = OLD_ROOT / "phase_3_targeted_candidate_catalog.json"
    if targeted.is_file():
        rows.extend(base._read_json(targeted))
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_name.setdefault(str(row["name"]), row)
    return list(by_name.values())


def _corrected_cached_metrics(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    oos_path = _cached_path(candidate, "oos")
    if not oos_path.is_file():
        return None
    old_oos = base._result_from_cache(base._read_json(oos_path))
    corrected_oos = [
        trade
        for trade in old_oos.trades
        if base.OOS_START <= trade.entry_date <= base.OOS_END
    ]
    payload: dict[str, Any] = {
        "name": candidate["name"],
        "group": candidate["group"],
        "family": candidate["family"],
        "patch": candidate["patch"],
        "settings_sha256": candidate["settings_sha256"],
        "oos": base._period_metrics(
            corrected_oos,
            start=base.OOS_START,
            end=base.OOS_END,
        ),
        "early_oos": base._period_metrics(
            corrected_oos,
            start=base.OOS_START,
            end=base.EARLY_OOS_END,
        ),
        "latest_oos": base._period_metrics(
            corrected_oos,
            start=base.LATEST_OOS_START,
            end=base.OOS_END,
        ),
        "oos_trades": [base._trade_payload(trade) for trade in corrected_oos],
        "is_available": False,
    }
    is_path = _cached_path(candidate, "is")
    if is_path.is_file():
        old_is = base._result_from_cache(base._read_json(is_path))
        corrected_is = [
            *old_is.trades,
            *[
                trade
                for trade in old_oos.trades
                if date(2025, 8, 1) <= trade.entry_date <= base.IS_END
            ],
        ]
        corrected_is.sort(key=lambda trade: (trade.entry_date, trade.entry_time, trade.symbol))
        payload["is"] = base._period_metrics(
            corrected_is,
            start=base.IS_START,
            end=base.IS_END,
        )
        payload["is_trades"] = [
            base._trade_payload(trade) for trade in corrected_is
        ]
        payload["is_available"] = True
    return payload


def _rescored_gate(row: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    if not row.get("is_available"):
        return {"passed": False, "failed_gates": ["corrected_is_not_previously_replayed"]}
    gates = {
        "oos_total_r_improves_5pct": row["oos"]["total_r"] >= control["oos"]["total_r"] * 1.05,
        "oos_average_r_improves": row["oos"]["average_r"] > control["oos"]["average_r"],
        "oos_frequency_retains_90pct": row["oos"]["trades_per_month"] >= control["oos"]["trades_per_month"] * 0.90,
        "early_oos_not_worse_by_1r": (
            row["early_oos"]["total_r"]
            >= control["early_oos"]["total_r"] - 1.0
        ),
        "latest_oos_positive": row["latest_oos"]["total_r"] > 0.0,
        "is_total_r_retains_90pct": row["is"]["total_r"] >= control["is"]["total_r"] * 0.90,
        "is_average_r_retains_90pct": row["is"]["average_r"] >= control["is"]["average_r"] * 0.90,
        "is_frequency_retains_90pct": row["is"]["trades_per_month"] >= control["is"]["trades_per_month"] * 0.90,
        "is_drawdown_not_materially_worse": row["is"]["close_to_close_trade_drawdown_pct"] <= control["is"]["close_to_close_trade_drawdown_pct"] * 1.10 + 0.005,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _merge_patches(*patches: Mapping[str, Any]) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for patch in patches:
        for field, value in patch.items():
            if field in merged and merged[field] != value:
                return None
            merged[field] = value
    return merged


def _new_candidate(
    current: StrategySettings,
    patch: Mapping[str, Any],
    *,
    family: str,
    index: int,
) -> dict[str, Any]:
    signature = base._sha(base._jsonable(dict(patch)))[:10]
    return base._candidate(
        f"corrected_target_{family}_{index:03d}_{signature}",
        "corrected_targeted_extension",
        current,
        patch,
        family=family,
        note="Corrected-split targeted mutation not present in the completed broad sweep.",
    )


def _targeted_candidates(
    current: StrategySettings,
    catalog: list[dict[str, Any]],
    rescored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control = next(row for row in rescored if row["name"] == "current")
    by_name = {row["name"]: row for row in rescored}
    atomic = [
        row
        for row in catalog
        if row.get("group") == "single_perturbation"
        and row["name"] in by_name
        and by_name[row["name"]]["oos"]["trades"]
        >= control["oos"]["trades"] * 0.75
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in atomic:
        by_family[str(row["family"])].append(row)
    levers: list[dict[str, Any]] = []
    for family, rows in by_family.items():
        rows.sort(
            key=lambda row: by_name[row["name"]]["oos"]["total_r"],
            reverse=True,
        )
        for row in rows[:2]:
            levers.append(
                {
                    "family": family,
                    "patch": dict(row["patch"]),
                    "uplift": (
                        by_name[row["name"]]["oos"]["total_r"]
                        - control["oos"]["total_r"]
                    ),
                }
            )
    levers.sort(key=lambda row: row["uplift"], reverse=True)
    levers = levers[:24]
    incumbent_patch = {
        "daily_residual_minimum_z": 1.10,
        "daily_residual_minimum_score": 20.0,
    }
    proposals: list[tuple[str, dict[str, Any], float]] = []
    for lever in levers:
        patch = _merge_patches(incumbent_patch, lever["patch"])
        if patch is not None:
            proposals.append(("incumbent_plus", patch, float(lever["uplift"])))
    for left_index, left in enumerate(levers):
        for right in levers[left_index + 1 :]:
            if left["family"] == right["family"]:
                continue
            patch = _merge_patches(left["patch"], right["patch"])
            if patch is not None:
                proposals.append(
                    (
                        "cross_family_pair",
                        patch,
                        float(left["uplift"]) + float(right["uplift"]),
                    )
                )
            triple = _merge_patches(incumbent_patch, left["patch"], right["patch"])
            if triple is not None:
                proposals.append(
                    (
                        "incumbent_triple",
                        triple,
                        float(left["uplift"]) + float(right["uplift"]),
                    )
                )
    local = [
        {"daily_residual_minimum_z": value, "daily_residual_minimum_score": score}
        for value in (1.075, 1.125)
        for score in (17.5, 22.5)
    ]
    local.extend(
        _merge_patches(incumbent_patch, {field: value})
        for field, values in (
            ("daily_residual_catastrophic_stop_residual_r", (6.5, 7.5)),
            ("daily_residual_minimum_failed_continuation_r", (0.15, 0.25, 0.35)),
            ("daily_residual_minimum_market_trend_z_20d", (-0.875, -0.625)),
            ("daily_residual_minimum_sector_return_5d", (-0.125, -0.075)),
            ("daily_residual_max_positions", (11, 13)),
            ("daily_residual_maximum_holding_sessions", (9, 11)),
        )
        for value in values
    )
    proposals.extend(("local_response", patch, 0.0) for patch in local if patch)
    proposals.sort(key=lambda row: row[2], reverse=True)
    existing_signatures = {str(row["settings_sha256"]) for row in catalog}
    seen = set(existing_signatures)
    output: list[dict[str, Any]] = []
    for family, patch, _score in proposals:
        candidate = _new_candidate(
            current,
            patch,
            family=family,
            index=len(output) + 1,
        )
        signature = str(candidate["settings_sha256"])
        if signature in seen:
            continue
        seen.add(signature)
        output.append(candidate)
        if len(output) >= 120:
            break
    return output


def _shortlist(
    catalog: list[dict[str, Any]],
    rescored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in catalog}
    control = next(row for row in rescored if row["name"] == "current")
    frequency_floor = control["oos"]["trades"] * 0.75
    ranked_oos = sorted(
        (row for row in rescored if row["oos"]["trades"] >= frequency_floor),
        key=lambda row: (row["oos"]["total_r"], row["oos"]["trades"]),
        reverse=True,
    )
    joint = sorted(
        (row for row in rescored if row.get("corrected_gate", {}).get("passed")),
        key=lambda row: row["oos"]["total_r"],
        reverse=True,
    )
    names = [
        "current",
        "target_z110_score20",
        *[row["name"] for row in ranked_oos[:12]],
        *[row["name"] for row in joint[:12]],
    ]
    return [by_name[name] for name in dict.fromkeys(names) if name in by_name]


def _exact_gate(
    oos: Mapping[str, Any],
    is_row: Mapping[str, Any],
    control_oos: Mapping[str, Any],
    control_is: Mapping[str, Any],
) -> dict[str, Any]:
    return base._eligibility(oos, is_row, control_oos, control_is)


def run(output: Path, workers: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    catalog = _catalog()
    rescored = [
        row
        for candidate in catalog
        if (row := _corrected_cached_metrics(candidate)) is not None
    ]
    control_rescored = next(row for row in rescored if row["name"] == "current")
    for row in rescored:
        row["corrected_gate"] = _rescored_gate(row, control_rescored)
    base._write_json(output / "existing_candidates_corrected_rescoring.json", rescored)
    _candidate_row, baseline_lineage = base._load_round2_baseline(base.CURRENT_CANDIDATE)
    current = StrategySettings(**baseline_lineage["settings"])
    shortlist = _shortlist(catalog, rescored)
    targeted = _targeted_candidates(current, catalog, rescored)
    exact_candidates = [*shortlist, *targeted]
    by_signature: dict[str, dict[str, Any]] = {}
    for row in exact_candidates:
        by_signature.setdefault(str(row["settings_sha256"]), row)
    exact_candidates = list(by_signature.values())
    base._write_json(output / "exact_candidate_catalog.json", exact_candidates)
    base._write_json(
        output / "run_spec.json",
        {
            "contract": "corrected_split_shortlist_and_new_targeted_extension_v1",
            "is": [base.IS_START, base.IS_END],
            "oos": [base.OOS_START, base.OOS_END],
            "existing_candidates_rescored": len(rescored),
            "exact_existing_shortlist": len(shortlist),
            "genuinely_new_targeted_candidates": len(targeted),
            "workers": workers,
            "shared_prepared_cache": str(SHARED_PREPARED_CACHE.resolve()),
            "started_at_utc": datetime.now(timezone.utc),
            "promotion_eligible": False,
        },
    )
    exp = base.Experiment(
        output=output,
        data_dir=base.DATA_DIR.resolve(),
        workers=workers,
        candidates=exact_candidates,
        prepared_cache_dir=SHARED_PREPARED_CACHE.resolve(),
    )
    exp.load_data()
    try:
        names = [row["name"] for row in exact_candidates]
        oos_rows = exp.evaluate_many(names, window="oos")
        is_rows = exp.evaluate_many(names, window="is")
        by_oos = {row["name"]: row for row in oos_rows}
        by_is = {row["name"]: row for row in is_rows}
        current_name = next(
            row["name"]
            for row in exact_candidates
            if row["settings_sha256"]
            == next(item for item in catalog if item["name"] == "current")["settings_sha256"]
        )
        control_oos, control_is = by_oos[current_name], by_is[current_name]
        comparison = []
        candidate_map = {row["name"]: row for row in exact_candidates}
        for name in names:
            comparison.append(
                {
                    "name": name,
                    "group": candidate_map[name]["group"],
                    "family": candidate_map[name]["family"],
                    "patch": candidate_map[name]["patch"],
                    "oos": by_oos[name]["period_metrics"],
                    "is": by_is[name]["period_metrics"]["is"],
                    "eligibility": _exact_gate(
                        by_oos[name],
                        by_is[name],
                        control_oos,
                        control_is,
                    ),
                }
            )
        eligible = sorted(
            (row for row in comparison if row["eligibility"]["passed"]),
            key=lambda row: (
                row["oos"]["oos"]["r_per_month"],
                row["oos"]["oos"]["trades_per_month"],
                row["is"]["r_per_month"],
            ),
            reverse=True,
        )
        robustness = []
        for index, row in enumerate(eligible[:10]):
            name = row["name"]
            cost30 = exp.evaluate_one(candidate_map[name], window="oos", cost_bps=30.0)
            cost40 = exp.evaluate_one(candidate_map[name], window="oos", cost_bps=40.0)
            robustness.append(
                {
                    "name": name,
                    "base": row,
                    "oos_cost30": cost30["period_metrics"],
                    "oos_cost40": cost40["period_metrics"],
                    "paired_oos_bootstrap": base._bootstrap_daily_delta(
                        control_oos["trades"],
                        by_oos[name]["trades"],
                        seed=20260824 + index,
                    ),
                }
            )
        qualified = [
            row
            for row in robustness
            if row["oos_cost30"]["oos"]["total_r"] > 0.0
            and row["oos_cost40"]["oos"]["total_r"] > 0.0
        ]
        selected = max(
            qualified,
            key=lambda row: (
                row["base"]["oos"]["oos"]["r_per_month"],
                row["base"]["oos"]["oos"]["trades_per_month"],
            ),
            default=None,
        )
        payload = {
            "contract": "corrected_split_exact_shortlist_targeted_and_cost_confirmation_v1",
            "control": {
                "name": current_name,
                "oos": control_oos["period_metrics"],
                "is": control_is["period_metrics"]["is"],
            },
            "comparison": comparison,
            "eligible_count": len(eligible),
            "finalist_robustness": robustness,
            "selected_research_candidate": selected,
            "promotion_eligible": False,
            "selection_caveat": "March-May 2026 OOS has already informed research; validate after 2026-05-01.",
        }
        base._write_json(output / "corrected_targeted_results.json", payload)
        if selected is not None:
            chosen = candidate_map[selected["name"]]
            base._write_json(
                output / "recommended_research_config.json",
                {
                    "candidate_name": chosen["name"],
                    "patch_vs_round2": chosen["patch"],
                    "settings": chosen["settings"],
                    "settings_sha256": chosen["settings_sha256"],
                    "promotion_eligible": False,
                    "validation_required": "untouched chronological data after 2026-05-01",
                },
            )
        base._write_json(
            output / "completion.json",
            {
                "status": "complete",
                "selected": selected["name"] if selected else None,
                "completed_at_utc": datetime.now(timezone.utc),
            },
        )
        return payload
    finally:
        exp.close()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not 1 <= args.workers <= 3:
        raise SystemExit("--workers must be between 1 and 3")
    result = run(args.output_dir.resolve(), args.workers)
    selected = result.get("selected_research_candidate")
    print(
        "completed corrected targeted extension: "
        + (selected["name"] if selected else "no eligible candidate"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
