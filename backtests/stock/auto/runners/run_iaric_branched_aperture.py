"""Run the pre-registered IARIC branched-aperture optimization round.

The round broadens opportunity generation before applying independent quality
filters, preserves generator diversity with a small beam, then tests causal
entry routes and management.  It never reads the holdout beginning 2026-03-02.
Legacy replay data makes the result research-only until repeated unchanged on a
frozen authoritative bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.iaric.phase_scoring import score_v6r1_pullback_phase
from backtests.stock.auto.iaric.worker import (
    evaluate_candidate_attribution,
    evaluate_candidate_diagnostics,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    DATA_DIR,
    HOLDOUT_START,
    INITIAL_EQUITY,
    _code_fingerprint as _base_code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = REPO_ROOT / "backtests/output/stock/iaric/round_2/optimized_config.json"
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/iaric/round_3"
CATALOG = REPO_ROOT / "docs/iaric-branched-aperture-catalog.json"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
MAX_WORKERS = 2
FOLDS: tuple[tuple[str, str, str], ...] = (
    ("early", "2024-03-25", "2024-11-30"),
    ("middle", "2024-12-01", "2025-07-31"),
    ("latest", "2025-08-01", "2026-03-01"),
)

# Exactly seven immutable components.  Fixed Round-1 anchors prevent the small
# candidate sample from learning its own normalization.
SCORE_SPEC: dict[str, dict[str, float]] = {
    "expected_total_r": {"weight": 0.27, "center": 21.0484961925357, "scale": 12.0},
    "net_profit": {"weight": 0.13, "center": 1307.46, "scale": 600.0},
    "total_trades": {"weight": 0.15, "center": 89.0, "scale": 45.0},
    "avg_r": {"weight": 0.10, "center": 0.236499957219503, "scale": 0.12},
    "profit_factor": {"weight": 0.08, "center": 1.6472990836043948, "scale": 0.35},
    "entry_realized_discrimination_lift_r": {"weight": 0.12, "center": 0.2604893198240668, "scale": 0.12},
    "inverse_drawdown": {"weight": 0.15, "center": 0.0632752553726446, "scale": 0.025},
}

GENERATOR_GATES = {
    "min_trades": 85,
    "min_frequency_improvement_if_below_min_trades": 0.10,
    "min_avg_r": 0.05,
    "min_profit_factor": 1.10,
    "min_expected_total_r": 12.0,
    "max_drawdown_pct": 0.10,
    "min_robust_avg_r": -0.10,
}

FINAL_GATES = {
    "min_frequency_improvement": 0.15,
    "min_avg_r": 0.12,
    "min_expected_total_r": 25.0,
    "min_profit_factor": 1.45,
    "max_drawdown_pct": 0.07,
    "max_drawdown_increase_pct_points": 0.0125,
    "min_robust_avg_r": 0.0,
    "min_ex_top3_profit_factor": 1.05,
    "max_single_symbol_net_share": 0.35,
    "min_bootstrap_improvement_probability": 0.75,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _code_fingerprint() -> str:
    digest = hashlib.sha256(_base_code_fingerprint().encode("utf-8"))
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "strategies/stock/iaric/risk.py",
        REPO_ROOT / "strategies/stock/iaric/entry_request.py",
        REPO_ROOT / "docs/iaric-branched-aperture-catalog.json",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _clean_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(payload)
    # Round 2 ablation proved hard_reject changed shadow opportunity accounting
    # but no executable trade.  Restore the real incumbent behavior.
    baseline["param_overrides.pb_flow_policy"] = "soft_penalty_rescue"
    baseline["param_overrides.pb_open_scored_fill_timing"] = "next_5m_open"
    baseline["param_overrides.pb_entry_micropressure_policy"] = "score_penalty"
    baseline["param_overrides.pb_entry_min_reversion_room_atr"] = 0.0
    baseline["param_overrides.pb_v2_secondary_route_sizing_mult"] = 1.0
    return dict(sorted(baseline.items()))


def _candidate(
    parent: dict[str, Any],
    candidate_id: str,
    changes: dict[str, Any],
    *,
    stage: str,
    module: str,
) -> dict[str, Any]:
    mutations = deepcopy(parent["mutations"])
    mutations.update(changes)
    modules = deepcopy(parent.get("modules", []))
    if changes:
        modules.append({"stage": stage, "name": module, "changes": dict(changes)})
    return {
        "id": candidate_id,
        "family": parent.get("root_family", parent["id"]),
        "root_family": parent.get("root_family", parent["id"]),
        "parent_id": parent["id"],
        "stage": stage,
        "modules": modules,
        "mutations": mutations,
        "sources": [parent["id"]],
    }


def _dedupe(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_signature: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_signature.setdefault(_signature(candidate["mutations"]), candidate)
    return list(by_signature.values())


def _score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for name, spec in SCORE_SPEC.items():
        if name == "inverse_drawdown":
            z_value = (spec["center"] - float(metrics.get("max_drawdown_pct", 0.0))) / spec["scale"]
        else:
            z_value = (float(metrics.get(name, 0.0)) - spec["center"]) / spec["scale"]
        components[name] = min(max(0.5 + 0.5 * math.tanh(z_value), 0.0), 1.0)
    score = sum(SCORE_SPEC[name]["weight"] * value for name, value in components.items())
    reference = score_v6r1_pullback_phase(5, metrics)
    if abs(score - reference) > 1e-12:
        raise AssertionError(f"immutable score drift: runner={score}, phase={reference}")
    return float(score), components


def _decorate(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {candidate["id"]: candidate for candidate in candidates}
    decorated: list[dict[str, Any]] = []
    for row in rows:
        source = metadata[row["id"]]
        row.update({
            key: deepcopy(source[key])
            for key in ("root_family", "parent_id", "stage", "modules")
        })
        score, components = _score(row.get("metrics", {}))
        row["immutable_score"] = score
        row["immutable_score_components"] = components
        decorated.append(row)
    return sorted(
        decorated,
        key=lambda row: (
            float(row.get("immutable_score", -1.0)),
            float(row.get("metrics", {}).get("expected_total_r", -1e9)),
            float(row.get("metrics", {}).get("total_trades", 0.0)),
            -float(row.get("metrics", {}).get("max_drawdown_pct", 1.0)),
        ),
        reverse=True,
    )


def _evaluate(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    attribution: bool = False,
) -> list[dict[str, Any]]:
    unique = _dedupe(candidates)
    rows = _evaluate_batch(
        unique,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / ("attribution_cache.json" if attribution else "evaluation_cache.json"),
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        evaluation_fn=evaluate_candidate_attribution if attribution else evaluate_candidate_diagnostics,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / f"{stage}_errors.json", errors)
        raise RuntimeError(f"{len(errors)} {stage} candidate evaluations failed")
    ranked = _decorate(rows, unique)
    _write_json(output_dir / f"{stage}_results.json", ranked)
    return ranked


def _generator_candidates(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    root = {
        "id": "cleaned_round2_control",
        "root_family": "cleaned_round2_control",
        "mutations": baseline,
        "modules": [],
    }
    definitions = (
        ("control_oversold", {}),
        ("multi_dislocation", {"param_overrides.pb_v2_open_scored_trigger_policy": "multi_dislocation"}),
        ("signal_floor_55", {"param_overrides.pb_v2_signal_floor": 55.0}),
        ("oversold_or_multi", {"param_overrides.pb_v2_open_scored_trigger_policy": "oversold_or_multi"}),
        ("floor55_multi_union", {
            "param_overrides.pb_v2_signal_floor": 55.0,
            "param_overrides.pb_v2_open_scored_trigger_policy": "oversold_or_multi",
        }),
        ("broad_dislocation", {"param_overrides.pb_v2_open_scored_trigger_policy": "dislocation"}),
        ("moderate_context_intraday_shock", {
            "param_overrides.pb_v2_signal_floor": 55.0,
            "param_overrides.pb_v2_open_scored_trigger_policy": "any",
            "param_overrides.pb_v2_open_scored_confirmation_policy": "band_reclaim",
        }),
    )
    result = []
    for name, changes in definitions:
        candidate = _candidate(root, name, changes, stage="generator", module=name)
        candidate["root_family"] = name
        candidate["family"] = name
        result.append(candidate)
    return result


def _generator_survives(row: dict[str, Any], baseline_trades: float) -> bool:
    if row["id"] == "control_oversold":
        return True
    metrics = row.get("metrics", {})
    trades = float(metrics.get("total_trades", 0.0))
    frequency_ok = trades >= GENERATOR_GATES["min_trades"] or trades >= baseline_trades * 1.10
    return bool(
        frequency_ok
        and float(metrics.get("avg_r", -99.0)) >= GENERATOR_GATES["min_avg_r"]
        and float(metrics.get("profit_factor", 0.0)) >= GENERATOR_GATES["min_profit_factor"]
        and float(metrics.get("expected_total_r", -99.0)) >= GENERATOR_GATES["min_expected_total_r"]
        and float(metrics.get("max_drawdown_pct", 1.0)) <= GENERATOR_GATES["max_drawdown_pct"]
        and float(metrics.get("robust_avg_r", -99.0)) >= GENERATOR_GATES["min_robust_avg_r"]
    )


def _filter_candidates(generators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("unfiltered_control", {}),
        ("block_distribute", {"param_overrides.pb_entry_micropressure_policy": "block_distribute"}),
        ("room_010", {"param_overrides.pb_entry_min_reversion_room_atr": 0.10}),
        ("block_distribute_room_010", {
            "param_overrides.pb_entry_micropressure_policy": "block_distribute",
            "param_overrides.pb_entry_min_reversion_room_atr": 0.10,
        }),
        ("reversion_event_ranker", {"param_overrides.pb_entry_score_family": "reversion_event_v1"}),
        ("event_geometry_priority", {"param_overrides.pb_open_scored_priority": "low_score"}),
    )
    result = []
    for parent in generators:
        for name, changes in definitions:
            result.append(_candidate(
                parent,
                f"{parent['id']}__filter__{name}",
                changes,
                stage="filter",
                module=name,
            ))
    return result


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm, rm = left["metrics"], right["metrics"]
    comparisons = (
        float(left["immutable_score"]) >= float(right["immutable_score"]),
        float(lm.get("expected_total_r", 0.0)) >= float(rm.get("expected_total_r", 0.0)),
        float(lm.get("total_trades", 0.0)) >= float(rm.get("total_trades", 0.0)),
        float(lm.get("max_drawdown_pct", 1.0)) <= float(rm.get("max_drawdown_pct", 1.0)),
    )
    strict = (
        float(left["immutable_score"]) > float(right["immutable_score"]) + 1e-12
        or float(lm.get("expected_total_r", 0.0)) > float(rm.get("expected_total_r", 0.0)) + 1e-9
        or float(lm.get("total_trades", 0.0)) > float(rm.get("total_trades", 0.0)) + 1e-9
        or float(lm.get("max_drawdown_pct", 1.0)) < float(rm.get("max_drawdown_pct", 1.0)) - 1e-12
    )
    return all(comparisons) and strict


def _beam(rows: list[dict[str, Any]], *, max_per_family: int = 2) -> list[dict[str, Any]]:
    beam: list[dict[str, Any]] = []
    for family in sorted({row["root_family"] for row in rows}):
        family_rows = [row for row in rows if row["root_family"] == family]
        viable = [
            row for row in family_rows
            if float(row["metrics"].get("avg_r", -99.0)) >= 0.05
            and float(row["metrics"].get("profit_factor", 0.0)) >= 1.10
            and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.10
        ]
        frontier = [
            row for row in viable
            if not any(_dominates(other, row) for other in viable if other is not row)
        ]
        frontier.sort(key=lambda row: float(row["immutable_score"]), reverse=True)
        beam.extend(frontier[:max_per_family])
    return beam


def _route_candidates(filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("next_bar_band_reclaim", {}),
        ("confirmed_retest", {"param_overrides.pb_open_scored_transition": "confirmed_retest"}),
        ("resting_retrace", {
            "param_overrides.pb_open_scored_transition": "resting_retrace",
            "param_overrides.pb_open_scored_retrace_limit_fraction": 0.35,
            "param_overrides.pb_open_scored_retrace_limit_window_bars": 12,
        }),
        ("reclaim_or_limit", {
            "param_overrides.pb_open_scored_transition": "reclaim_or_limit",
            "param_overrides.pb_open_scored_retrace_limit_fraction": 0.20,
            "param_overrides.pb_open_scored_retrace_limit_window_bars": 12,
        }),
        ("opening_reclaim", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_opening_reclaim_min_daily_signal_score": 55.0,
        }),
        ("vwap_shock_reclaim", {
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_vwap_bounce_after_bar": 12,
        }),
        ("late_second_dislocation", {
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_after_bar": 48,
            "param_overrides.pb_v2_afternoon_retest_min_score": 50.0,
        }),
    )
    result = []
    for parent in filtered:
        for name, changes in definitions:
            result.append(_candidate(
                parent,
                f"{parent['id']}__route__{name}",
                changes,
                stage="route",
                module=name,
            ))
    return result


def _diverse_top(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    viable = [
        row for row in rows
        if float(row["metrics"].get("avg_r", -99.0)) >= 0.05
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.10
        and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.10
    ]
    selected: list[dict[str, Any]] = []
    for family in sorted({row["root_family"] for row in viable}):
        best = max(
            (row for row in viable if row["root_family"] == family),
            key=lambda row: float(row["immutable_score"]),
        )
        selected.append(best)
    selected.sort(key=lambda row: float(row["immutable_score"]), reverse=True)
    if len(selected) >= limit:
        return selected[:limit]
    seen = {row["id"] for row in selected}
    for row in sorted(viable, key=lambda value: float(value["immutable_score"]), reverse=True):
        if row["id"] not in seen:
            selected.append(row)
            seen.add(row["id"])
        if len(selected) >= limit:
            break
    return selected


def _additive_route_candidates(route_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("opening_plus_vwap", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
        }),
        ("vwap_plus_late_retest", {
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        }),
        ("three_secondary_routes", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        }),
    )
    result = []
    for parent in _diverse_top(route_rows, 4):
        for name, changes in definitions:
            result.append(_candidate(
                parent,
                f"{parent['id']}__additive__{name}",
                changes,
                stage="route_additive",
                module=name,
            ))
    return result


def _management_candidates(entry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("management_control", {}),
        ("secondary_size_050", {"param_overrides.pb_v2_secondary_route_sizing_mult": 0.50}),
        ("secondary_size_065", {"param_overrides.pb_v2_secondary_route_sizing_mult": 0.65}),
        ("failure_exit_4", {
            "param_overrides.pb_open_scored_stale_exit_bars": 4,
            "param_overrides.pb_open_scored_stale_exit_min_r": 0.10,
        }),
        ("failure_exit_6", {
            "param_overrides.pb_open_scored_stale_exit_bars": 6,
            "param_overrides.pb_open_scored_stale_exit_min_r": 0.15,
        }),
    )
    result = []
    for parent in _diverse_top(entry_rows, 6):
        for name, changes in definitions:
            result.append(_candidate(
                parent,
                f"{parent['id']}__management__{name}",
                changes,
                stage="management",
                module=name,
            ))
    return result


def _validation(
    finalists: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_fingerprint: str,
    code_fingerprint: str,
) -> list[dict[str, Any]]:
    fold_results: dict[str, list[dict[str, Any]]] = {}
    for name, start_date, end_date in FOLDS:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.start_date = start_date
        fold_args.end_date = end_date
        fold_results[name] = _evaluate(
            f"validation_{name}",
            finalists,
            args=fold_args,
            output_dir=output_dir,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
    for finalist in finalists:
        signature = _signature(finalist["mutations"])
        folds = []
        for name, _, _ in FOLDS:
            match = next(row for row in fold_results[name] if row["signature"] == signature)
            metrics = match["metrics"]
            folds.append({
                "fold": name,
                "trades": float(metrics.get("total_trades", 0.0)),
                "total_r": float(metrics.get("expected_total_r", 0.0)),
                "avg_r": float(metrics.get("avg_r", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            })
        finalist["validation"] = {
            "folds": folds,
            "positive_fold_count": sum(fold["total_r"] > 0 for fold in folds),
            "latest_fold_positive": folds[-1]["total_r"] > 0,
            "worst_fold_avg_r": min(fold["avg_r"] for fold in folds),
        }
    _write_json(output_dir / "validated_finalists.json", finalists)
    return finalists


def _attribution_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    by_symbol: dict[str, dict[str, float]] = {}
    for trade in trades:
        bucket = by_symbol.setdefault(str(trade["symbol"]), {"pnl_net": 0.0, "r": 0.0})
        bucket["pnl_net"] += float(trade.get("pnl_net", 0.0))
        bucket["r"] += float(trade.get("r", 0.0))
    top_symbols = sorted(by_symbol, key=lambda symbol: by_symbol[symbol]["pnl_net"], reverse=True)[:3]
    remaining = [trade for trade in trades if str(trade["symbol"]) not in top_symbols]
    gains = sum(max(float(trade.get("r", 0.0)), 0.0) for trade in remaining)
    losses = -sum(min(float(trade.get("r", 0.0)), 0.0) for trade in remaining)
    total_net = sum(bucket["pnl_net"] for bucket in by_symbol.values())
    max_share = (
        max((bucket["pnl_net"] for bucket in by_symbol.values()), default=0.0) / total_net
        if total_net > 0 else float("inf")
    )
    return {
        "top_three_symbols": top_symbols,
        "ex_top3_total_r": sum(float(trade.get("r", 0.0)) for trade in remaining),
        "ex_top3_profit_factor": gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "max_single_symbol_net_share": max_share,
        "by_symbol": by_symbol,
    }


def _paired_block_bootstrap(
    candidate_trades: list[dict[str, Any]],
    baseline_trades: list[dict[str, Any]],
    *,
    simulations: int = 5000,
    block_days: int = 5,
) -> dict[str, Any]:
    def daily(trades: list[dict[str, Any]]) -> dict[str, float]:
        result: dict[str, float] = {}
        for trade in trades:
            day = str(trade["entry_time"])[:10]
            result[day] = result.get(day, 0.0) + float(trade.get("r", 0.0))
        return result

    candidate_daily = daily(candidate_trades)
    baseline_daily = daily(baseline_trades)
    days = sorted(set(candidate_daily) | set(baseline_daily))
    deltas = [candidate_daily.get(day, 0.0) - baseline_daily.get(day, 0.0) for day in days]
    if not deltas:
        return {"probability_positive": 0.0, "observed_delta_r": 0.0, "simulations": simulations}
    rng = random.Random(20260820)
    positive = 0
    n_days = len(deltas)
    for _ in range(simulations):
        sampled: list[float] = []
        while len(sampled) < n_days:
            start = rng.randrange(n_days)
            sampled.extend(deltas[(start + offset) % n_days] for offset in range(block_days))
        positive += sum(sampled[:n_days]) > 0.0
    return {
        "probability_positive": positive / simulations,
        "observed_delta_r": sum(deltas),
        "simulations": simulations,
        "block_days": block_days,
        "paired_days": n_days,
    }


def _final_gate_results(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    attribution: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, bool]:
    metrics = candidate["metrics"]
    base_metrics = baseline["metrics"]
    validation = candidate["validation"]
    frequency_target = math.ceil(float(base_metrics.get("total_trades", 0.0)) * 1.15)
    return {
        "frequency": float(metrics.get("total_trades", 0.0)) >= frequency_target,
        "avg_r": float(metrics.get("avg_r", -99.0)) >= FINAL_GATES["min_avg_r"],
        "expected_total_r": float(metrics.get("expected_total_r", -99.0)) >= FINAL_GATES["min_expected_total_r"],
        "profit_factor": float(metrics.get("profit_factor", 0.0)) >= FINAL_GATES["min_profit_factor"],
        "absolute_drawdown": float(metrics.get("max_drawdown_pct", 1.0)) <= FINAL_GATES["max_drawdown_pct"],
        "relative_drawdown": float(metrics.get("max_drawdown_pct", 1.0)) <= float(base_metrics.get("max_drawdown_pct", 0.0)) + FINAL_GATES["max_drawdown_increase_pct_points"],
        "robust_avg_r": float(metrics.get("robust_avg_r", -99.0)) >= FINAL_GATES["min_robust_avg_r"],
        "folds": validation["positive_fold_count"] >= 2 and validation["latest_fold_positive"],
        "ex_top3": float(attribution["ex_top3_total_r"]) > 0.0 and float(attribution["ex_top3_profit_factor"]) >= FINAL_GATES["min_ex_top3_profit_factor"],
        "symbol_concentration": float(attribution["max_single_symbol_net_share"]) <= FINAL_GATES["max_single_symbol_net_share"],
        "bootstrap": float(bootstrap["probability_positive"]) >= FINAL_GATES["min_bootstrap_improvement_probability"],
    }


def _audit_candidates(selected: dict[str, Any], cleaned_baseline: dict[str, Any]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    modules = selected.get("modules", [])
    for index, module in enumerate(modules):
        mutations = dict(selected["mutations"])
        for key in module["changes"]:
            if key in cleaned_baseline:
                mutations[key] = cleaned_baseline[key]
            else:
                mutations.pop(key, None)
        audits.append({
            "id": f"audit__ablate_{index:02d}_{module['name']}",
            "family": selected["root_family"],
            "root_family": selected["root_family"],
            "parent_id": selected["id"],
            "stage": "audit",
            "modules": modules,
            "mutations": mutations,
            "sources": [selected["id"]],
            "audit_kind": "ablation",
            "audit_module": module["name"],
        })
    numeric_neighbors = (
        ("param_overrides.pb_v2_signal_floor", (-2.5, 2.5)),
        ("param_overrides.pb_entry_min_reversion_room_atr", (-0.05, 0.05)),
        ("param_overrides.pb_open_scored_retrace_limit_fraction", (-0.05, 0.05)),
        ("param_overrides.pb_open_scored_stale_exit_bars", (-1.0, 1.0)),
        ("param_overrides.pb_v2_secondary_route_sizing_mult", (-0.15, 0.15)),
    )
    for key, offsets in numeric_neighbors:
        if key not in selected["mutations"] or selected["mutations"].get(key) == cleaned_baseline.get(key):
            continue
        for offset in offsets:
            mutations = dict(selected["mutations"])
            value = float(mutations[key]) + offset
            mutations[key] = int(round(value)) if key.endswith("stale_exit_bars") else max(value, 0.0)
            audits.append({
                "id": f"audit__perturb_{key.rsplit('.', 1)[-1]}_{offset:+g}".replace(".", "p"),
                "family": selected["root_family"],
                "root_family": selected["root_family"],
                "parent_id": selected["id"],
                "stage": "audit",
                "modules": modules,
                "mutations": mutations,
                "sources": [selected["id"]],
                "audit_kind": "perturbation",
                "audit_module": key,
            })
    return _dedupe(audits)


def _audit_passes(selected: dict[str, Any], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_score = float(selected["immutable_score"])
    selected_etr = float(selected["metrics"].get("expected_total_r", 0.0))
    details = []
    for row in audit_rows:
        kind = "ablation" if row["id"].startswith("audit__ablate") else "perturbation"
        if kind == "ablation":
            passed = (
                selected_score >= float(row["immutable_score"]) + 0.002
                or selected_etr >= float(row["metrics"].get("expected_total_r", 0.0)) + 1.0
                or float(selected["metrics"].get("total_trades", 0.0)) >= float(row["metrics"].get("total_trades", 0.0)) + 3.0
            )
        else:
            passed = (
                float(row["metrics"].get("expected_total_r", -99.0)) >= 0.65 * selected_etr
                and float(row["metrics"].get("profit_factor", 0.0)) >= 1.10
                and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.10
            )
        details.append({"id": row["id"], "kind": kind, "passed": passed})
    return {
        "all_passed": bool(details) and all(item["passed"] for item in details),
        "checks": details,
    }


def _progress(output_dir: Path, stage: str, status: str, **extra: Any) -> None:
    _write_json(output_dir / "progress.json", {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        **extra,
    })


def main() -> None:
    args = _parse_args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"end date overlaps sealed holdout beginning {HOLDOUT_START}")
    if not args.allow_legacy_data:
        raise ValueError("this workspace currently requires --allow-legacy-data; results remain research-only")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    baseline_path = Path(args.baseline_config).resolve()
    cleaned_baseline = _clean_baseline(json.loads(baseline_path.read_text(encoding="utf-8")))
    generators = _generator_candidates(cleaned_baseline)
    if args.dry_run:
        assert len(SCORE_SPEC) == 7
        assert len({candidate["id"] for candidate in generators}) == len(generators)
        print(f"dry-run ok: {len(generators)} generators, seven-component immutable score, holdout excluded")
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    catalog_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    _write_json(output_dir / "cleaned_baseline_config.json", cleaned_baseline)
    _write_json(output_dir / "run_spec.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running_research_only",
        "architecture": "generator_beam_then_independent_filters_then_routes_then_management",
        "baseline": str(baseline_path.relative_to(REPO_ROOT)),
        "baseline_cleanup": {"param_overrides.pb_flow_policy": "soft_penalty_rescue"},
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": args.max_workers,
        "data_authority": "legacy_research_only",
        "data_dir": str(DATA_DIR.relative_to(REPO_ROOT)),
        "data_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "catalog_sha256": catalog_hash,
        "score_spec": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "generator_gates": GENERATOR_GATES,
        "final_gates": FINAL_GATES,
        "deferred_not_falsely_approximated": [
            "market_sector_residual_generator_requires_authoritative sector residual feed",
            "multi_day_first_higher_low route requires a separately typed multi-session FSM",
        ],
    })

    _progress(output_dir, "generators", "running", candidates=len(generators))
    generator_rows = _evaluate(
        "generator", generators, args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    baseline_row = next(row for row in generator_rows if row["id"] == "control_oversold")
    baseline_trades = float(baseline_row["metrics"].get("total_trades", 0.0))
    generator_survivors = [row for row in generator_rows if _generator_survives(row, baseline_trades)]
    _write_json(output_dir / "generator_survivors.json", generator_survivors)

    _progress(output_dir, "filters", "running", branches=len(generator_survivors))
    filter_rows = _evaluate(
        "filter", _filter_candidates(generator_survivors), args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    filter_beam = _beam(filter_rows, max_per_family=2)
    _write_json(output_dir / "filter_beam.json", filter_beam)
    if not filter_beam:
        raise RuntimeError("no filtered generator branch survived the broad economic safety gates")

    _progress(output_dir, "route_isolation", "running", branches=len(filter_beam))
    route_rows = _evaluate(
        "route_isolation", _route_candidates(filter_beam), args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    additive_rows = _evaluate(
        "route_additive", _additive_route_candidates(route_rows), args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    entry_rows = route_rows + additive_rows

    _progress(output_dir, "management", "running", entry_branches=min(6, len(entry_rows)))
    management_rows = _evaluate(
        "management", _management_candidates(entry_rows), args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    all_final_pool = entry_rows + management_rows
    finalists = _diverse_top(all_final_pool, 4)
    if _signature(baseline_row["mutations"]) not in {_signature(row["mutations"]) for row in finalists}:
        finalists.append(baseline_row)

    _progress(output_dir, "validation", "running", finalists=len(finalists))
    finalists = _validation(
        finalists, args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
    )
    finalists.sort(key=lambda row: float(row["immutable_score"]), reverse=True)

    attribution_rows = _evaluate(
        "finalist_attribution", finalists, args=args, output_dir=output_dir,
        source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint, attribution=True,
    )
    attr_by_sig = {_signature(row["mutations"]): row for row in attribution_rows}
    baseline_attr = attr_by_sig[_signature(baseline_row["mutations"])]
    strict_candidates = []
    for finalist in finalists:
        attribution_row = attr_by_sig[_signature(finalist["mutations"])]
        stats = _attribution_stats(attribution_row.get("trade_attribution", []))
        bootstrap = _paired_block_bootstrap(
            attribution_row.get("trade_attribution", []),
            baseline_attr.get("trade_attribution", []),
        )
        finalist["attribution_gates"] = stats
        finalist["paired_block_bootstrap"] = bootstrap
        finalist["final_gate_results"] = _final_gate_results(finalist, baseline_row, stats, bootstrap)
        finalist["pre_audit_promotable"] = all(finalist["final_gate_results"].values())
        if finalist["pre_audit_promotable"]:
            strict_candidates.append(finalist)
    selected = max(strict_candidates or finalists, key=lambda row: float(row["immutable_score"]))

    _progress(output_dir, "ablation_perturbation", "running", selected=selected["id"])
    audits = _audit_candidates(selected, cleaned_baseline)
    audit_rows = (
        _evaluate(
            "ablation_perturbation", audits, args=args, output_dir=output_dir,
            source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint,
        )
        if audits else []
    )
    audit_summary = _audit_passes(selected, audit_rows) if audits else {"all_passed": False, "checks": []}
    promotable = bool(selected.get("pre_audit_promotable") and audit_summary["all_passed"])
    status = "promotable_research_candidate" if promotable else "provisional_not_promoted"
    config_name = "optimized_config.json" if promotable else "provisional_config.json"
    _write_json(output_dir / config_name, dict(sorted(selected["mutations"].items())))
    _write_json(output_dir / "final_selection.json", {
        "status": status,
        "selected": selected,
        "audit": audit_summary,
        "holdout_accessed": False,
        "research_only": True,
    })
    _progress(output_dir, "complete", status, selected=selected["id"], promotable=promotable)
    print(
        f"branched aperture complete: {status}; selected={selected['id']}; "
        f"trades={selected['metrics'].get('total_trades', 0):.0f}; "
        f"totalR={selected['metrics'].get('expected_total_r', 0):+.2f}; "
        f"PF={selected['metrics'].get('profit_factor', 0):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
