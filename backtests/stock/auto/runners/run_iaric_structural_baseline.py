"""Establish a causal, live-parity IARIC training baseline before phased auto.

The search is deliberately structural and staged: selection aperture, route
capacity/timing, then executable management.  The sealed holdout is excluded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    FOLDS,
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = REPO_ROOT / "backtests/output/stock/iaric/round_1/optimized_config.json"
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment"
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
MAX_WORKERS = 2

# Exactly seven immutable economic components.  Fold robustness is a hard
# promotion overlay, not an eighth component that can be optimized around.
SCORE_SPEC: dict[str, dict[str, float | str]] = {
    "expected_total_r": {"weight": 0.26, "transform": "tanh(x / 75)"},
    "avg_r": {"weight": 0.18, "transform": "tanh(x / 0.15)"},
    "profit_factor": {"weight": 0.14, "transform": "tanh((x - 1) / 0.50)"},
    "sharpe": {"weight": 0.11, "transform": "tanh(x / 2.0)"},
    "inverse_drawdown": {"weight": 0.13, "transform": "tanh((0.10 - x) / 0.08)"},
    "trades_per_month": {"weight": 0.10, "transform": "tanh(x / 20)"},
    "tail_resilience": {"weight": 0.08, "transform": "tanh((tail_loss_r + 1) / 0.75)"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument("--stop-after-stage", choices=("selection", "routes", "management", "validation"), default="validation")
    return parser.parse_args()


def _score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    values = {
        "expected_total_r": math.tanh(float(metrics.get("expected_total_r", 0.0)) / 75.0),
        "avg_r": math.tanh(float(metrics.get("avg_r", 0.0)) / 0.15),
        "profit_factor": math.tanh((float(metrics.get("profit_factor", 0.0)) - 1.0) / 0.50),
        "sharpe": math.tanh(float(metrics.get("sharpe", 0.0)) / 2.0),
        "inverse_drawdown": math.tanh((0.10 - float(metrics.get("max_drawdown_pct", 1.0))) / 0.08),
        "trades_per_month": math.tanh(float(metrics.get("trades_per_month", 0.0)) / 20.0),
        "tail_resilience": math.tanh((float(metrics.get("tail_loss_r", -2.0)) + 1.0) / 0.75),
    }
    return sum(float(SCORE_SPEC[key]["weight"]) * value for key, value in values.items()), values


def _full_eligible(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("total_trades", 0.0)) >= 300
        and float(metrics.get("expected_total_r", -1e9)) >= 50.0
        and float(metrics.get("avg_r", -1e9)) >= 0.06
        and float(metrics.get("profit_factor", 0.0)) >= 1.25
        and float(metrics.get("sharpe", -1e9)) >= 1.00
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.10
    )


def _decorate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        score, components = _score(row.get("metrics", {}))
        row["baseline_score"] = score
        row["baseline_score_components"] = components
        row["full_period_eligible"] = _full_eligible(row.get("metrics", {}))
    return sorted(
        rows,
        key=lambda row: (
            1 if row["full_period_eligible"] else 0,
            float(row["baseline_score"]),
            float(row.get("metrics", {}).get("expected_total_r", -1e9)),
            -float(row.get("metrics", {}).get("max_drawdown_pct", 1.0)),
        ),
        reverse=True,
    )


def _candidate(base: dict[str, Any], candidate_id: str, overrides: dict[str, Any], family: str) -> dict[str, Any]:
    mutations = deepcopy(base)
    mutations.update(overrides)
    return {"id": candidate_id, "family": family, "sources": ["structural_baseline"], "mutations": mutations}


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        result.setdefault(_signature(candidate["mutations"]), candidate)
    return list(result.values())


def _selection_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        _candidate(base, f"selection_floor_{floor}", {"param_overrides.pb_v2_signal_floor": float(floor)}, "selection_aperture")
        for floor in (40, 45, 50, 55, 60, 65, 70)
    ]
    candidates.extend(
        [
            _candidate(base, "selection_floor_55_hard_flow", {
                "param_overrides.pb_v2_signal_floor": 55.0,
                "param_overrides.pb_flow_policy": "hard_reject",
            }, "selection_flow"),
            _candidate(base, "selection_floor_55_soft_flow", {
                "param_overrides.pb_v2_signal_floor": 55.0,
                "param_overrides.pb_flow_policy": "soft_penalty",
            }, "selection_flow"),
        ]
    )
    return _dedupe(candidates)


def _route_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    # Mechanism-led decomposition.  Earlier controls established that slot
    # caps/reserves do not bind and that the momentum/quality score families
    # are economically destructive.  Keep only the threshold neighborhood and
    # route ablations needed to locate the executable source of edge.
    candidates = [
            _candidate(base, "routes_control", {}, "route_control"),
            _candidate(base, "routes_entry_score_35", {"param_overrides.pb_entry_score_min": 35.0}, "route_aperture"),
            _candidate(base, "routes_entry_score_40", {"param_overrides.pb_entry_score_min": 40.0}, "route_aperture"),
            _candidate(base, "routes_entry_score_45", {"param_overrides.pb_entry_score_min": 45.0}, "route_aperture"),
            _candidate(base, "routes_score40_confirmation_only", {
                "param_overrides.pb_entry_score_min": 40.0,
                "param_overrides.pb_v2_open_scored_enabled": False,
                "param_overrides.pb_open_scored_enabled": False,
            }, "route_ablation"),
            _candidate(base, "routes_score40_open_only", {
                "param_overrides.pb_entry_score_min": 40.0,
                "param_overrides.pb_delayed_confirm_enabled": False,
                "param_overrides.pb_opening_reclaim_enabled": False,
                "param_overrides.pb_v2_vwap_bounce_enabled": False,
                "param_overrides.pb_v2_afternoon_retest_enabled": False,
            }, "route_ablation"),
        ]
    return _dedupe(candidates)


def _management_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [_candidate(base, "management_control", {}, "management_control")]
    candidates.extend(
        [
            # The archived frontier shows overnight holds dilute expectancy;
            # this is the sole carry experiment needed for the baseline.
            _candidate(base, "management_intraday_only", {"param_overrides.pb_carry_enabled": False}, "carry_ablation"),
            _candidate(base, "management_partial_020_000", {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.20,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.00,
            }, "executable_partial"),
            _candidate(base, "management_partial_030_000", {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.30,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.00,
            }, "executable_partial"),
            _candidate(base, "management_early_protection", {
                "param_overrides.pb_v2_mfe_stage1_trigger": 0.30,
                "param_overrides.pb_v2_mfe_stage1_stop_r": -0.10,
                "param_overrides.pb_v2_mfe_stage2_trigger": 0.50,
            }, "mfe_protection"),
        ]
    )
    return _dedupe(candidates)


def _run_stage(
    name: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    cache_path: Path,
    source_fingerprint: str,
    code_fingerprint: str,
) -> list[dict[str, Any]]:
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / f"{name}_errors.json", errors)
        raise RuntimeError(f"{len(errors)} {name} evaluations failed")
    ranked = _decorate(rows)
    _write_json(output_dir / f"{name}_ranking.json", ranked)
    return ranked


def _validation(candidate: dict[str, Any], fold_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    signature = _signature(candidate["mutations"])
    folds = []
    for fold_name, _, _ in FOLDS:
        row = next((value for value in fold_rows[fold_name] if value["signature"] == signature), None)
        if row is None or row.get("error"):
            continue
        metrics = row["metrics"]
        folds.append({
            "fold": fold_name,
            "trades": float(metrics.get("total_trades", 0.0)),
            "total_r": float(metrics.get("expected_total_r", 0.0)),
            "avg_r": float(metrics.get("avg_r", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        })
    positive = sum(row["total_r"] > 0 and row["avg_r"] > 0 for row in folds)
    worst_avg_r = min((row["avg_r"] for row in folds), default=-99.0)
    max_dd = max((row["max_drawdown_pct"] for row in folds), default=1.0)
    robust = len(folds) == len(FOLDS) and positive >= 3 and worst_avg_r >= -0.05 and max_dd <= 0.12
    penalty = 0.08 * max(0, 3 - positive) + 0.10 * max(0.0, -worst_avg_r / 0.10)
    return {
        "folds": folds,
        "positive_fold_count": positive,
        "worst_fold_avg_r": worst_avg_r,
        "max_fold_drawdown_pct": max_dd,
        "robust_eligible": robust,
        "validated_score": float(candidate["baseline_score"]) - penalty,
    }


def main() -> None:
    args = _parse_args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"End date overlaps sealed holdout beginning {HOLDOUT_START}")
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.baseline_config).resolve().read_text(encoding="utf-8"))
    base.update({
        "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
        "param_overrides.pb_v2_open_scored_allow_rescue": False,
        "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
    })
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    cache_path = output_dir / "evaluation_cache.json"

    selection = _run_stage("selection", _selection_candidates(base), args=args, output_dir=output_dir, cache_path=cache_path, source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint)
    selection_winner = selection[0]
    if args.stop_after_stage == "selection":
        return
    routes = _run_stage("routes", _route_candidates(selection_winner["mutations"]), args=args, output_dir=output_dir, cache_path=cache_path, source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint)
    routes_winner = routes[0]
    if args.stop_after_stage == "routes":
        return
    management = _run_stage("management", _management_candidates(routes_winner["mutations"]), args=args, output_dir=output_dir, cache_path=cache_path, source_fingerprint=source_fingerprint, code_fingerprint=code_fingerprint)
    if args.stop_after_stage == "management":
        return

    finalists = _dedupe(selection[:2] + routes[:2] + management[:4])
    fold_results: dict[str, list[dict[str, Any]]] = {}
    for fold_name, fold_start, fold_end in FOLDS:
        fold_results[fold_name] = _evaluate_batch(
            finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
    for finalist in finalists:
        score, components = _score(finalist.get("metrics", {}))
        finalist["baseline_score"] = score
        finalist["baseline_score_components"] = components
        finalist["full_period_eligible"] = _full_eligible(finalist.get("metrics", {}))
        finalist["validation"] = _validation(finalist, fold_results)
    finalists.sort(key=lambda row: (
        1 if row["full_period_eligible"] and row["validation"]["robust_eligible"] else 0,
        float(row["validation"]["validated_score"]),
        float(row.get("metrics", {}).get("expected_total_r", -1e9)),
    ), reverse=True)
    winner = finalists[0]
    promotable = bool(winner["full_period_eligible"] and winner["validation"]["robust_eligible"])
    status = "provisional_training_baseline" if promotable else "no_promotable_baseline"
    selected_config = dict(sorted(winner["mutations"].items()))
    _write_json(output_dir / "validated_finalists.json", finalists)
    _write_json(output_dir / "optimized_config.json", selected_config)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "data_authority": "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle",
        "data_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": args.max_workers,
        "score_spec": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "promotion_gates": {
            "full_period": {"min_trades": 300, "min_total_r": 50, "min_avg_r": 0.06, "min_pf": 1.25, "min_sharpe": 1.00, "max_dd": 0.10},
            "folds": {"positive_folds": "3/4", "worst_avg_r": -0.05, "max_dd": 0.12},
        },
        "selected": {
            "id": winner["id"],
            "signature": _signature(selected_config),
            "metrics": winner["metrics"],
            "baseline_score": winner["baseline_score"],
            "validation": winner["validation"],
        },
    }
    _write_json(output_dir / "baseline_manifest.json", manifest)
    summary = [
        "IARIC STRUCTURAL BASELINE ESTABLISHMENT",
        f"Status: {status}",
        f"Selected: {winner['id']}",
        f"Trades: {winner['metrics'].get('total_trades', 0):.0f}",
        f"Total R: {winner['metrics'].get('expected_total_r', 0):+.2f}",
        f"Avg R: {winner['metrics'].get('avg_r', 0):+.4f}",
        f"PF: {winner['metrics'].get('profit_factor', 0):.3f}",
        f"Sharpe: {winner['metrics'].get('sharpe', 0):+.3f}",
        f"Max DD: {winner['metrics'].get('max_drawdown_pct', 0):.2%}",
        f"Positive folds: {winner['validation']['positive_fold_count']}/4",
        "Holdout accessed: no",
    ]
    (output_dir / "baseline_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
