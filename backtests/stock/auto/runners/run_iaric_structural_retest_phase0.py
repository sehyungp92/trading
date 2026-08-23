"""Evaluate causal OPEN_SCORED entry transitions for IARIC.

The screen is structural and bounded: unchanged immediate control, two broad
completed-bar confirmation policies, confirmed pullback/recovery, and a
resting retrace order that is active before a later bar can fill it.  All
nightly selectors, risk, management, costs, and exits remain fixed.  The
sealed holdout is never accessed and a structural winner must also survive
pre-specified chronological folds.
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

from backtests.stock.auto.iaric.worker import (
    evaluate_candidate_attribution,
    evaluate_candidate_diagnostics,
)
from backtests.stock.auto.iaric.phase_scoring import (
    V5R1_PHASE_SCORING_WEIGHTS,
    score_v5r1_pullback_phase,
)
from backtests.stock.auto.runners.run_iaric_causal_entry_phase0 import (
    _entry_geometry,
    _score_attribution,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    FOLDS,
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import (
    MAX_WORKERS,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/post_integrity_selected_config.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/structural_retest_phase0"
)
READINESS_PATH = (
    REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
)
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"

# Exactly seven immutable entry-alpha components, identical in every later
# phased-auto stage.  Opportunity terms are causal MFE diagnostics capped at
# 2R; realized and fold-robust terms prevent an attractive opportunity funnel
# from masking an untradeable execution path.
SCORE_SPEC: dict[str, dict[str, float | str]] = {
    "entry_potential_total_r": {"weight": 0.18, "transform": "baseline-centred tanh"},
    "entry_potential_avg_r": {"weight": 0.14, "transform": "baseline-centred tanh"},
    "entry_opportunity_recall": {"weight": 0.10, "transform": "baseline-centred tanh"},
    "entry_discrimination_lift_r": {"weight": 0.14, "transform": "baseline-centred tanh"},
    "expected_total_r": {"weight": 0.24, "transform": "baseline-centred tanh"},
    "robust_avg_r": {"weight": 0.12, "transform": "baseline-centred tanh"},
    "robust_high_quality_frequency": {"weight": 0.08, "transform": "baseline-centred tanh"},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        help="Optional bounded subset of predeclared structural candidates.",
    )
    return parser.parse_args()


def _runner_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _candidate(
    base: dict[str, Any],
    candidate_id: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    mutations = deepcopy(base)
    mutations.update(overrides)
    return {
        "id": candidate_id,
        "family": "shared_open_scored_transition",
        "sources": ["post_integrity_reference", "causal_entry_phase0"],
        "mutations": mutations,
    }


def _fixed_base(base: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(base)
    fixed.update(
        {
            "param_overrides.pb_v2_open_scored_after_bar": 0,
            "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
            "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
            "param_overrides.pb_carry_enabled": False,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
            "param_overrides.pb_delayed_confirm_enabled": False,
            "param_overrides.pb_opening_reclaim_enabled": False,
            "param_overrides.pb_v2_vwap_bounce_enabled": False,
            "param_overrides.pb_v2_afternoon_retest_enabled": False,
        }
    )
    return fixed


def _candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    retest_common = {
        "param_overrides.pb_open_scored_transition": "confirmed_retest",
        "param_overrides.pb_open_scored_retest_min_close_pct": 0.55,
        "param_overrides.pb_open_scored_retest_min_impulse_atr": 0.15,
        "param_overrides.pb_open_scored_retest_max_extension_atr": 0.35,
    }
    return [
        _candidate(
            base,
            "immediate_next_bar_control",
            {"param_overrides.pb_open_scored_transition": "next_bar"},
        ),
        _candidate(
            base,
            "completed_bullish_close",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_v2_open_scored_confirmation_policy": "bullish_close",
            },
        ),
        _candidate(
            base,
            "completed_bullish_vwap",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_v2_open_scored_confirmation_policy": "bullish_vwap",
            },
        ),
        _candidate(
            base,
            "causal_vwap_reclaim",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_v2_open_scored_confirmation_policy": "vwap_reclaim",
            },
        ),
        _candidate(
            base,
            "true_dislocation_only",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_v2_open_scored_trigger_policy": "dislocation",
            },
        ),
        _candidate(
            base,
            "true_multi_dislocation_only",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_v2_open_scored_trigger_policy": "multi_dislocation",
            },
        ),
        _candidate(
            base,
            "retest_20pct_3bar",
            {
                **retest_common,
                "param_overrides.pb_open_scored_retest_retrace_frac": 0.20,
                "param_overrides.pb_open_scored_retest_window_bars": 3,
            },
        ),
        _candidate(
            base,
            "retest_35pct_6bar",
            {
                **retest_common,
                "param_overrides.pb_open_scored_retest_retrace_frac": 0.35,
                "param_overrides.pb_open_scored_retest_window_bars": 6,
            },
        ),
        _candidate(
            base,
            "retest_35pct_12bar",
            {
                **retest_common,
                "param_overrides.pb_open_scored_retest_retrace_frac": 0.35,
                "param_overrides.pb_open_scored_retest_window_bars": 12,
            },
        ),
        _candidate(
            base,
            "retest_50pct_12bar",
            {
                **retest_common,
                "param_overrides.pb_open_scored_retest_retrace_frac": 0.50,
                "param_overrides.pb_open_scored_retest_window_bars": 12,
            },
        ),
        _candidate(
            base,
            "resting_retrace_20pct_12bar",
            {
                "param_overrides.pb_open_scored_transition": "resting_retrace",
                "param_overrides.pb_open_scored_retrace_limit_fraction": 0.20,
                "param_overrides.pb_open_scored_retrace_limit_window_bars": 12,
            },
        ),
        _candidate(
            base,
            "resting_retrace_35pct_12bar",
            {
                "param_overrides.pb_open_scored_transition": "resting_retrace",
                "param_overrides.pb_open_scored_retrace_limit_fraction": 0.35,
                "param_overrides.pb_open_scored_retrace_limit_window_bars": 12,
            },
        ),
        _candidate(
            base,
            "resting_retrace_50pct_12bar",
            {
                "param_overrides.pb_open_scored_transition": "resting_retrace",
                "param_overrides.pb_open_scored_retrace_limit_fraction": 0.50,
                "param_overrides.pb_open_scored_retrace_limit_window_bars": 12,
            },
        ),
    ]


def _delta(metrics: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0)) - float(control.get(key, 0.0))
        for key in (
            "total_trades",
            "expected_total_r",
            "avg_r",
            "profit_factor",
            "sharpe",
            "max_drawdown_pct",
        )
    }


def _material_gate(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    delta = _delta(metrics, control)
    return bool(
        float(metrics.get("total_trades", 0.0)) >= 80.0
        and delta["expected_total_r"] >= 5.0
        and delta["avg_r"] >= 0.05
        and float(metrics.get("profit_factor", 0.0)) >= 1.35
        and float(metrics.get("sharpe", 0.0)) >= 0.90
        and float(metrics.get("max_drawdown_pct", 1.0))
        <= max(0.08, float(control.get("max_drawdown_pct", 1.0)) + 0.005)
    )


def _decorate(rows: list[dict[str, Any]], control: dict[str, Any]) -> list[dict[str, Any]]:
    for row in rows:
        row["immutable_score"] = score_v5r1_pullback_phase(
            1,
            row["metrics"],
            V5R1_PHASE_SCORING_WEIGHTS[1],
        )
        row["immutable_score_components"] = {
            key: float(row["metrics"].get(key, 0.0))
            for key in SCORE_SPEC
        }
        row["delta_vs_control"] = _delta(row["metrics"], control["metrics"])
        row["structural_materiality_gate"] = bool(
            row["id"] != control["id"]
            and _material_gate(row["metrics"], control["metrics"])
        )
    return sorted(
        rows,
        key=lambda row: (
            1 if row["structural_materiality_gate"] else 0,
            float(row["immutable_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
        ),
        reverse=True,
    )


def _fold_gate(folds: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    metrics = [row.get("metrics", {}) for row in folds]
    avg_rs = [float(item.get("avg_r", 0.0)) for item in metrics]
    totals = [float(item.get("expected_total_r", 0.0)) for item in metrics]
    trades = [float(item.get("total_trades", 0.0)) for item in metrics]
    summary = {
        "fold_count": len(metrics),
        "positive_fold_count": sum(value > 0.0 for value in totals),
        "min_fold_trades": min(trades, default=0.0),
        "worst_fold_avg_r": min(avg_rs, default=0.0),
        "mean_fold_avg_r": sum(avg_rs) / max(len(avg_rs), 1),
    }
    passed = bool(
        len(metrics) == len(FOLDS)
        and summary["positive_fold_count"] >= 2
        and summary["min_fold_trades"] >= 15.0
        and summary["worst_fold_avg_r"] >= -0.05
        and summary["mean_fold_avg_r"] >= 0.03
    )
    return passed, summary


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
    base = _fixed_base(json.loads(baseline_path.read_text(encoding="utf-8")))
    candidates = _candidates(base)
    if args.candidate_ids:
        requested = set(args.candidate_ids)
        candidates = [row for row in candidates if row["id"] in requested]
        missing = requested.difference(row["id"] for row in candidates)
        if missing:
            raise ValueError(f"Unknown candidate ids: {sorted(missing)}")
        if "immediate_next_bar_control" not in requested:
            raise ValueError(
                "A bounded subset must include immediate_next_bar_control"
            )
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _runner_fingerprint()
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "economics_cache.json",
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        evaluation_fn=evaluate_candidate_diagnostics,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} structural transition evaluations failed")
    control = next(row for row in rows if row["id"] == "immediate_next_bar_control")
    rows = _decorate(rows, control)
    eligible = [row for row in rows if row["structural_materiality_gate"]]
    fold_results: dict[str, list[dict[str, Any]]] = {}
    fold_summaries: dict[str, dict[str, Any]] = {}
    robust_eligible: list[dict[str, Any]] = []
    for candidate in eligible:
        candidate_folds: list[dict[str, Any]] = []
        for fold_name, fold_start, fold_end in FOLDS:
            fold_rows = _evaluate_batch(
                [candidate],
                start_date=fold_start,
                end_date=fold_end,
                max_workers=min(args.max_workers, 1),
                cache_path=output_dir / "fold_cache.json",
                source_fingerprint=source_fingerprint,
                code_fingerprint=f"{code_fingerprint}:fold-v1",
                evaluation_fn=evaluate_candidate_diagnostics,
            )
            fold_row = fold_rows[0]
            if fold_row.get("error"):
                _write_json(output_dir / "fold_errors.json", fold_rows)
                raise RuntimeError(f"Fold evaluation failed for {candidate['id']}")
            candidate_folds.append(
                {
                    "fold": fold_name,
                    "start": fold_start,
                    "end": fold_end,
                    "metrics": fold_row["metrics"],
                }
            )
        fold_passed, fold_summary = _fold_gate(candidate_folds)
        candidate["chronological_fold_gate"] = fold_passed
        candidate["chronological_fold_summary"] = fold_summary
        fold_results[candidate["id"]] = candidate_folds
        fold_summaries[candidate["id"]] = fold_summary
        if fold_passed:
            robust_eligible.append(candidate)

    winner = robust_eligible[0] if robust_eligible else control

    attribution: dict[str, Any] | None = None
    if robust_eligible:
        detail_rows = _evaluate_batch(
            [winner],
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=min(args.max_workers, 1),
            cache_path=output_dir / "attribution_cache.json",
            source_fingerprint=source_fingerprint,
            code_fingerprint=f"{code_fingerprint}:attribution-v1",
            evaluation_fn=evaluate_candidate_attribution,
        )
        detail = detail_rows[0]
        if detail.get("error"):
            _write_json(output_dir / "attribution_errors.json", detail_rows)
            raise RuntimeError("Structural winner attribution failed")
        trades = detail.get("trade_attribution", [])
        attribution = {
            "candidate_id": winner["id"],
            "score_attribution": _score_attribution(trades),
            "entry_geometry": _entry_geometry(trades),
            "funnel_counters": detail.get("funnel_counters", {}),
        }
        _write_json(output_dir / "winner_trade_attribution.json", trades)
        _write_json(output_dir / "winner_attribution_summary.json", attribution)

    _write_json(output_dir / "ranking.json", rows)
    _write_json(output_dir / "chronological_folds.json", fold_results)
    _write_json(output_dir / "preferred_config.json", dict(sorted(winner["mutations"].items())))
    next_decision = (
        "use_repaired_structural_baseline_for_entry_signal_discrimination"
        if robust_eligible
        else "structural_entry_transition_rejected_continue_root_cause_audit_before_phased_auto"
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_structural_phase0_complete",
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
            "shared_transition_contract": {
                "signal": "completed bar arms setup",
                "confirmation": "later completed pullback/recovery bar",
                "fill": "following 5-minute bar open",
                "score_components_changed": False,
                "live_replay_core": "strategies.stock.iaric.core.logic",
                "full_day_session_atr_lookahead_repaired": True,
            },
            "materiality_gate": {
                "min_trades": 80,
                "min_total_r_uplift": 5.0,
                "min_avg_r_uplift": 0.05,
                "min_profit_factor": 1.35,
                "min_sharpe": 0.90,
                "max_drawdown": "min(8%, control + 0.5 percentage point)",
            },
            "chronological_fold_gate": {
                "folds": [
                    {"name": name, "start": start, "end": end}
                    for name, start, end in FOLDS
                ],
                "min_positive_folds": 2,
                "min_trades_each_fold": 15,
                "worst_fold_avg_r_floor": -0.05,
                "mean_fold_avg_r_floor": 0.03,
                "candidate_summaries": fold_summaries,
            },
            "preferred_candidate_id": winner["id"],
            "preferred_passed_materiality_gate": bool(robust_eligible),
            "preferred_passed_chronological_fold_gate": bool(robust_eligible),
            "preferred_signature": _signature(winner["mutations"]),
            "winner_attribution_collected": attribution is not None,
            "next_decision": next_decision,
        },
    )

    print("IARIC SHARED RETEST STRUCTURAL PHASE 0", flush=True)
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"material={row['structural_materiality_gate']} "
            f"fold={row.get('chronological_fold_gate', False)}",
            flush=True,
        )
    print(f"Preferred: {winner['id']}", flush=True)
    print(f"Next decision: {next_decision}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
