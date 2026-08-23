"""Run IARIC Phase 0 as executable, causal entry-timing replays.

The historical timing diagnostic substituted prices on already-selected
trades.  This runner instead replays the complete strategy for each timing,
including signal extraction, admission, structural stop construction, sizing,
portfolio capacity, exits, costs, and the shared live/backtest next-bar rule.
The sealed holdout is excluded by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtests.stock.auto.iaric.worker import evaluate_candidate_attribution
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import (
    MAX_WORKERS,
    SCORE_SPEC,
    _score,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/post_integrity_selected_config.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/causal_entry_phase0"
)
READINESS_PATH = (
    REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
)
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _phase0_code_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _candidate(
    base: dict[str, Any],
    candidate_id: str,
    *,
    after_bar: int,
    label: str,
) -> dict[str, Any]:
    mutations = deepcopy(base)
    mutations["param_overrides.pb_v2_open_scored_after_bar"] = after_bar
    return {
        "id": candidate_id,
        "family": "causal_open_scored_timing",
        "sources": [str(DEFAULT_BASELINE.relative_to(REPO_ROOT))],
        "timing_label": label,
        "mutations": mutations,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.ptp(x_arr) <= 1e-12 or np.ptp(y_arr) <= 1e-12:
        return None
    value = float(np.corrcoef(_average_ranks(x_arr), _average_ranks(y_arr))[0, 1])
    return value if np.isfinite(value) else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _score_attribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    routed = [
        trade
        for trade in trades
        if trade.get("route") in {"OPEN_SCORED_ENTRY", "OPEN_SCORED_RETEST"}
        and np.isfinite(float(trade.get("route_score", 0.0)))
        and np.isfinite(float(trade.get("r", 0.0)))
    ]
    routed.sort(key=lambda trade: (float(trade["route_score"]), str(trade["entry_time"])))
    quartiles: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(np.arange(len(routed)), 4), start=1):
        bucket = [routed[int(idx)] for idx in indices]
        quartiles.append(
            {
                "quartile": index,
                "n": len(bucket),
                "score_min": min((float(trade["route_score"]) for trade in bucket), default=None),
                "score_max": max((float(trade["route_score"]) for trade in bucket), default=None),
                "avg_r": (
                    float(np.mean([float(trade["r"]) for trade in bucket]))
                    if bucket
                    else None
                ),
                "total_r": float(sum(float(trade["r"]) for trade in bucket)),
                "win_rate": (
                    float(np.mean([float(trade["r"]) > 0.0 for trade in bucket]))
                    if bucket
                    else None
                ),
            }
        )
    scores = [float(trade["route_score"]) for trade in routed]
    outcomes = [float(trade["r"]) for trade in routed]
    component_names = (
        "daily_signal",
        "reclaim",
        "volume",
        "vwap_hold",
        "cpr",
        "speed",
        "quality_adjustment",
    )
    component_spearman = {
        name: _spearman(
            [float((trade.get("score_components") or {}).get(name, 0.0)) for trade in routed],
            outcomes,
        )
        for name in component_names
    }
    q1_avg = quartiles[0]["avg_r"] if quartiles else None
    q4_avg = quartiles[-1]["avg_r"] if quartiles else None
    q4_minus_q1 = (
        float(q4_avg - q1_avg)
        if q1_avg is not None and q4_avg is not None
        else None
    )
    monotone_steps = 0
    comparable_steps = 0
    for left, right in zip(quartiles, quartiles[1:]):
        if left["avg_r"] is not None and right["avg_r"] is not None:
            comparable_steps += 1
            monotone_steps += int(right["avg_r"] >= left["avg_r"])
    return {
        "n": len(routed),
        "route_score_spearman_r": _spearman(scores, outcomes),
        "quartiles": quartiles,
        "q4_minus_q1_avg_r": q4_minus_q1,
        "nondecreasing_step_fraction": (
            float(monotone_steps / comparable_steps) if comparable_steps else None
        ),
        "component_spearman_r": component_spearman,
    }


def _entry_geometry(trades: list[dict[str, Any]]) -> dict[str, Any]:
    risk_pct = [
        float(trade["risk_per_share"]) / float(trade["entry_price"])
        for trade in trades
        if float(trade.get("entry_price", 0.0)) > 0.0
        and float(trade.get("risk_per_share", 0.0)) > 0.0
    ]
    mfe = [float(trade.get("mfe_r", 0.0)) for trade in trades]
    mae = [float(trade.get("mae_r", 0.0)) for trade in trades]
    signal_indices = sorted({int(trade.get("signal_bar_index", -1)) for trade in trades})
    entry_indices = sorted({int(trade.get("entry_bar_index", -1)) for trade in trades})
    return {
        "n": len(trades),
        "signal_bar_indices": signal_indices,
        "entry_bar_indices": entry_indices,
        "next_bar_contract_violations": sum(
            int(trade.get("entry_bar_index", -1))
            != int(trade.get("signal_bar_index", -1)) + 1
            for trade in trades
            if trade.get("route") in {"OPEN_SCORED_ENTRY", "OPEN_SCORED_RETEST"}
        ),
        "risk_distance_pct": {
            "median": _percentile(risk_pct, 50),
            "p25": _percentile(risk_pct, 25),
            "p75": _percentile(risk_pct, 75),
        },
        "mfe_r": {
            "median": _percentile(mfe, 50),
            "p20": _percentile(mfe, 20),
            "p80": _percentile(mfe, 80),
        },
        "mae_r": {
            "median": _percentile(mae, 50),
            "p20": _percentile(mae, 20),
            "p80": _percentile(mae, 80),
        },
        "fraction_mae_over_0_5r": (
            float(np.mean(np.asarray(mae, dtype=float) > 0.5)) if mae else None
        ),
        "fraction_mfe_over_0_5r": (
            float(np.mean(np.asarray(mfe, dtype=float) > 0.5)) if mfe else None
        ),
    }


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "trade_attribution"}


def _validate_baseline(base: dict[str, Any]) -> None:
    required = {
        "param_overrides.pb_execution_mode": "intraday_hybrid",
        "param_overrides.pb_v2_enabled": True,
        "param_overrides.pb_v2_open_scored_enabled": True,
        "param_overrides.pb_open_scored_enabled": True,
        "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
        "param_overrides.pb_carry_enabled": False,
        "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
        "param_overrides.pb_delayed_confirm_enabled": False,
        "param_overrides.pb_opening_reclaim_enabled": False,
        "param_overrides.pb_v2_vwap_bounce_enabled": False,
        "param_overrides.pb_v2_afternoon_retest_enabled": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": base.get(key)}
        for key, expected in required.items()
        if base.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Baseline violates Phase 0 isolation controls: {mismatches}")


def main() -> None:
    args = _args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"End date overlaps sealed holdout beginning {HOLDOUT_START}")
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    if not readiness.get("frozen_bundle_available") and not args.allow_legacy_data:
        raise RuntimeError(
            "Authoritative frozen replay bundle is unavailable; pass "
            "--allow-legacy-data for diagnostic-only work."
        )
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline_config).resolve()
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    _validate_baseline(base)
    candidates = [
        _candidate(base, "timing_bar0_fill_0935", after_bar=0, label="09:35 ET fill"),
        _candidate(base, "timing_bar5_fill_1000", after_bar=5, label="10:00 ET fill"),
        _candidate(base, "timing_bar11_fill_1030", after_bar=11, label="10:30 ET fill"),
    ]
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _phase0_code_fingerprint()
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "evaluation_cache.json",
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        evaluation_fn=evaluate_candidate_attribution,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} causal entry evaluations failed")

    control = next(row for row in rows if row["id"] == "timing_bar0_fill_0935")
    control_metrics = control["metrics"]
    for row in rows:
        trades = row.get("trade_attribution", [])
        row["immutable_score"], row["immutable_score_components"] = _score(row["metrics"])
        row["score_attribution"] = _score_attribution(trades)
        row["entry_geometry"] = _entry_geometry(trades)
        metrics = row["metrics"]
        row["delta_vs_control"] = {
            "trades": float(metrics.get("total_trades", 0.0))
            - float(control_metrics.get("total_trades", 0.0)),
            "total_r": float(metrics.get("expected_total_r", 0.0))
            - float(control_metrics.get("expected_total_r", 0.0)),
            "avg_r": float(metrics.get("avg_r", 0.0))
            - float(control_metrics.get("avg_r", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0))
            - float(control_metrics.get("profit_factor", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0))
            - float(control_metrics.get("sharpe", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0))
            - float(control_metrics.get("max_drawdown_pct", 0.0)),
        }
        row["timing_improvement_gate"] = bool(
            row["id"] != control["id"]
            and float(metrics.get("total_trades", 0.0)) >= 100.0
            and float(row["delta_vs_control"]["total_r"]) >= 5.0
            and float(row["delta_vs_control"]["avg_r"]) >= 0.04
            and float(metrics.get("profit_factor", 0.0)) >= 1.25
            and float(metrics.get("max_drawdown_pct", 1.0))
            <= max(0.10, float(control_metrics.get("max_drawdown_pct", 1.0)) + 0.01)
            and row["entry_geometry"]["next_bar_contract_violations"] == 0
        )
        score_attr = row["score_attribution"]
        row["score_direction_positive"] = bool(
            score_attr["route_score_spearman_r"] is not None
            and score_attr["route_score_spearman_r"] > 0.0
            and score_attr["q4_minus_q1_avg_r"] is not None
            and score_attr["q4_minus_q1_avg_r"] > 0.0
        )

    rows.sort(
        key=lambda row: (
            1 if row["timing_improvement_gate"] else 0,
            float(row["immutable_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
        ),
        reverse=True,
    )
    eligible = [row for row in rows if row["timing_improvement_gate"]]
    preferred = eligible[0] if eligible else control
    if eligible and preferred["score_direction_positive"]:
        next_decision = "validate_timing_and_untouched_score_on_chronological_folds"
    elif eligible:
        next_decision = "validate_timing_on_folds_then_rebuild_score_at_fixed_timing"
    else:
        next_decision = "timing_not_sufficient_rebuild_shared_signal_to_entry_transition"

    attribution_dir = output_dir / "trade_attribution"
    for row in rows:
        _write_json(attribution_dir / f"{row['id']}.json", row.get("trade_attribution", []))
    compact = [_compact_row(row) for row in rows]
    _write_json(output_dir / "ranking.json", compact)
    _write_json(output_dir / "preferred_config.json", dict(sorted(preferred["mutations"].items())))
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_phase0_complete",
            "data_authority": (
                "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle"
            ),
            "promotion_allowed": False,
            "promotion_blockers": readiness.get("blocking_reasons", []),
            "data_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "baseline_path": str(baseline_path.relative_to(REPO_ROOT)),
            "baseline_signature": _signature(base),
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": args.max_workers,
            "immutable_score": SCORE_SPEC,
            "score_component_count": len(SCORE_SPEC),
            "experiment_contract": {
                "signal": "completed 5-minute bar",
                "fill": "next 5-minute bar open plus executable costs",
                "full_replay": True,
                "recomputed": [
                    "signal extraction",
                    "admission",
                    "structural stop",
                    "position sizing",
                    "portfolio constraints",
                    "trade management",
                    "exits",
                ],
                "post_hoc_price_substitution": False,
                "same_open_or_moo_tested": False,
                "same_open_or_moo_reason": (
                    "The first completed-bar score is unknowable at the market open; "
                    "a real pre-open/MOO route requires a separate shared live/replay implementation."
                ),
            },
            "timing_gate": {
                "min_trades": 100,
                "min_total_r_uplift": 5.0,
                "min_avg_r_uplift": 0.04,
                "min_profit_factor": 1.25,
                "max_drawdown": "min(10%, control + 1 percentage point)",
                "next_bar_contract_violations": 0,
            },
            "preferred_candidate_id": preferred["id"],
            "preferred_candidate_passed_timing_gate": preferred["timing_improvement_gate"],
            "next_decision": next_decision,
        },
    )

    print("IARIC CAUSAL ENTRY PHASE 0", flush=True)
    for row in rows:
        metrics = row["metrics"]
        score_attr = row["score_attribution"]
        print(
            f"{row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"rho={score_attr.get('route_score_spearman_r')} "
            f"Q4-Q1={score_attr.get('q4_minus_q1_avg_r')} "
            f"gate={row['timing_improvement_gate']}",
            flush=True,
        )
    print(f"Preferred: {preferred['id']}", flush=True)
    print(f"Next decision: {next_decision}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
