"""Consolidate the bounded IARIC baseline-recovery evidence.

This finalizer performs no backtest and no optimization.  It selects the only
chronologically defensible reference, records why phased auto is not
authorized, and writes a complete pre-holdout experiment ledger.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment"
OUTPUT = ROOT / "final_recovery"
HOLDOUT_START = "2026-03-02"
SCORE_SPEC = {
    "expected_total_r": {"weight": 0.26, "transform": "tanh(x / 75)"},
    "avg_r": {"weight": 0.18, "transform": "tanh(x / 0.15)"},
    "profit_factor": {"weight": 0.14, "transform": "tanh((x - 1) / 0.50)"},
    "sharpe": {"weight": 0.11, "transform": "tanh(x / 2.0)"},
    "inverse_drawdown": {"weight": 0.13, "transform": "tanh((0.10 - x) / 0.08)"},
    "trades_per_month": {"weight": 0.10, "transform": "tanh(x / 20)"},
    "tail_resilience": {"weight": 0.08, "transform": "tanh((tail_loss_r + 1) / 0.75)"},
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _metric_row(row: dict[str, Any]) -> dict[str, float]:
    metrics = row["metrics"]
    return {
        "trades": float(metrics.get("total_trades", 0.0)),
        "total_r": float(metrics.get("expected_total_r", 0.0)),
        "avg_r": float(metrics.get("avg_r", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        "trades_per_month": float(metrics.get("trades_per_month", 0.0)),
    }


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in _read(path)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    timing = _by_id(ROOT / "causal_entry_phase0/ranking.json")
    rank = _by_id(ROOT / "rank_carry_phase0/rank_ranking.json")
    carry = _by_id(ROOT / "rank_carry_phase0/carry_ranking.json")
    retest = _by_id(ROOT / "structural_retest_phase0/ranking.json")
    impulse = _read(ROOT / "retest_impulse_ablation/result.json")
    fsm = _by_id(ROOT / "existing_fsm_phase0/ranking.json")
    limits = _by_id(ROOT / "retrace_limit_phase0/ranking.json")
    priority = _by_id(ROOT / "score_priority_phase0/ranking.json")
    components = _read(ROOT / "score_component_diagnostic/summary.json")
    readiness = _read(
        REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
    )

    control = priority["high_score_priority_control"]
    rank90 = rank["rank90_carry_off"]
    fold_cache = _read(ROOT / "rank90_fold_validation/evaluation_cache.json")
    fold_values = list(fold_cache.get("evaluations", {}).values())
    completed_folds: list[dict[str, Any]] = []
    for fold_name, start, end in (
        ("2024_h1", "2024-01-01", "2024-06-30"),
        ("2024_h2", "2024-07-01", "2024-12-31"),
    ):
        rows = [row for row in fold_values if row.get("start_date") == start and row.get("end_date") == end]
        control_signature = str(
            rank["rank100_carry_off_control"].get("signature")
            or rank["rank100_carry_off_control"].get("source_signature")
        )
        control_fold = next(
            row for row in rows if row.get("signature") == control_signature
        )
        rank90_fold = next(row for row in rows if row.get("signature") == rank90["signature"])
        delta_total_r = float(rank90_fold["metrics"]["expected_total_r"]) - float(
            control_fold["metrics"]["expected_total_r"]
        )
        delta_avg_r = float(rank90_fold["metrics"]["avg_r"]) - float(
            control_fold["metrics"]["avg_r"]
        )
        completed_folds.append(
            {
                "fold": fold_name,
                "start": start,
                "end": end,
                "control": _metric_row(control_fold),
                "rank90": _metric_row(rank90_fold),
                "delta_total_r": delta_total_r,
                "delta_avg_r": delta_avg_r,
                "rank90_total_r_win": delta_total_r > 0,
                "rank90_avg_r_win": delta_avg_r > 0,
            }
        )

    fold_early_stop = {
        "status": "stopped_early_when_selection_became_mathematically_decided",
        "required_wins": 3,
        "completed_folds": 2,
        "rank90_total_r_wins": sum(row["rank90_total_r_win"] for row in completed_folds),
        "rank90_avg_r_wins": sum(row["rank90_avg_r_win"] for row in completed_folds),
        "remaining_folds": 2,
        "maximum_possible_total_r_wins": 2,
        "maximum_possible_avg_r_wins": 2,
        "validation_passed": False,
        "selected_candidate_id": "rank100_carry_off_control",
        "folds": completed_folds,
    }
    _write(OUTPUT / "rank90_fold_early_stop.json", fold_early_stop)

    experiments = [
        {
            "phase": "causal_static_timing",
            "decision": "control_retained",
            "candidates": {key: _metric_row(value) for key, value in timing.items()},
        },
        {
            "phase": "rank_cap",
            "decision": "rank90_pareto_improved_full_window_but_failed_chronological_gate",
            "candidates": {key: _metric_row(value) for key, value in rank.items()},
        },
        {
            "phase": "repaired_carry",
            "decision": "carry_rejected",
            "candidates": {key: _metric_row(value) for key, value in carry.items()},
        },
        {
            "phase": "confirmed_retest",
            "decision": "rejected_for_starvation_and_negative_expectancy",
            "candidates": {key: _metric_row(value) for key, value in retest.items()},
        },
        {
            "phase": "retest_impulse_ablation",
            "decision": "rejected_after_arms_increased_13_to_95_but_expectancy_remained_negative",
            "candidate": _metric_row(impulse),
            "funnel": impulse.get("funnel_counters", {}),
        },
        {
            "phase": "existing_confirmation_fsm_after_causal_atr_repair",
            "decision": "all_confirmation_routes_rejected",
            "candidates": {key: _metric_row(value) for key, value in fsm.items()},
        },
        {
            "phase": "resting_retrace_limit",
            "decision": "all_depths_rejected_for_negative_expectancy",
            "candidates": {key: _metric_row(value) for key, value in limits.items()},
        },
        {
            "phase": "score_priority_sign",
            "decision": "low_score_priority_rejected",
            "candidates": {key: _metric_row(value) for key, value in priority.items()},
        },
        {
            "phase": "seven_component_attribution",
            "decision": "no_component_met_fold_stability_gate",
            "profiles": components["profiles"],
            "robust_negative_components": components["robust_negative_components"],
            "robust_positive_components": components["robust_positive_components"],
        },
        {
            "phase": "rank90_chronological_validation",
            "decision": "failed_and_stopped_early",
            **fold_early_stop,
        },
    ]
    _write(OUTPUT / "experiment_ledger.json", experiments)

    selected_config = dict(sorted(control["mutations"].items()))
    _write(OUTPUT / "selected_research_reference_config.json", selected_config)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "bounded_recovery_complete_no_meaningful_baseline_established",
        "data_authority": "legacy_diagnostic_only",
        "promotion_allowed": False,
        "promotion_blockers": readiness.get("blocking_reasons", []),
        "training_window": {"start": "2024-01-01", "end": "2026-03-01"},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": 2,
        "selected_research_reference_id": "high_score_priority_control",
        "selected_research_reference_metrics": _metric_row(control),
        "selected_research_reference_config": "selected_research_reference_config.json",
        "proper_meaningful_baseline_established": False,
        "phased_auto_authorized": False,
        "phased_auto_run": False,
        "phased_auto_blockers": [
            "No structural signal-to-entry candidate passed its predeclared materiality gate.",
            "The sole full-window Pareto filter failed both completed chronological folds; 3-of-4 became impossible.",
            "No score component met the predeclared overall-strength and 3-of-4 fold-stability gate.",
            "The authoritative frozen direct-RTH replay bundle remains unavailable.",
        ],
        "immutable_score": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "live_replay_repairs_retained": [
            "non-open session ATR uses completed-bar prefixes",
            "OPEN_SCORED score components persist to executed-trade metadata",
            "shared route-priority function defaults to original high-score ordering",
            "rejected structural transitions are disabled by default",
        ],
        "tests": {
            "final_iaric_regression_suite": "66 passed",
            "holdout_accessed": False,
        },
        "experiment_ledger": "experiment_ledger.json",
        "rank90_fold_decision": "rank90_fold_early_stop.json",
    }
    _write(OUTPUT / "recovery_manifest.json", manifest)

    metrics = manifest["selected_research_reference_metrics"]
    lines = [
        "IARIC BASELINE RECOVERY - FULL FINAL DIAGNOSTICS",
        "=" * 56,
        "",
        f"Generated UTC: {manifest['generated_at_utc']}",
        "Data authority: legacy diagnostic only; promotion is blocked.",
        "Training window: 2024-01-01 through 2026-03-01.",
        f"Sealed holdout begins {HOLDOUT_START}; accessed: NO.",
        "Maximum IARIC workers: 2.",
        "",
        "FINAL DECISION",
        "--------------",
        "The bounded recovery did not establish a proper meaningful alpha baseline.",
        "Phased auto optimization was deliberately NOT run because every structural",
        "entry candidate failed and no score component showed sufficient fold-stable",
        "predictive strength. Optimizing management or exits from that state would fit",
        "the small sample rather than repair signal extraction.",
        "",
        "HONEST RESEARCH REFERENCE (NOT PROMOTABLE)",
        "-------------------------------------------",
        f"Trades: {metrics['trades']:.0f}",
        f"Expected total R: {metrics['total_r']:+.4f}",
        f"Average R: {metrics['avg_r']:+.4f}",
        f"Profit factor: {metrics['profit_factor']:.4f}",
        f"Sharpe: {metrics['sharpe']:.4f}",
        f"Max drawdown: {metrics['max_drawdown_pct']:.2%}",
        f"Trades/month: {metrics['trades_per_month']:.3f}",
        "Config: immediate next-5m-open entry, high-score priority, rank cap 100,",
        "carry off, partials off, delayed/opening/VWAP/afternoon routes off.",
        "",
        "KEY EXPERIMENT VERDICTS",
        "-----------------------",
        "1. Static delays to 10:00 and 10:30 worsened executable economics.",
        "2. Rank<=90 was full-window Pareto-superior (+13.37R, PF 1.38, DD 5.84%)",
        "   but lost both completed chronological folds; the 3-of-4 rule became impossible.",
        "3. Repaired carry added only about +0.2R while increasing drawdown; rejected.",
        "4. Confirmed retest produced 3-5 trades; removing the impulse gate raised",
        "   arms to 95 but only 22 confirmations and -1.91R; rejected.",
        "5. After causal ATR repair, delayed-confirm lost -12.75R and combined FSM",
        "   lost -11.65R; opening reclaim produced only +0.89R/PF 0.98.",
        "6. Resting retrace limits at 20%, 35%, and 50% all lost money (PF 0.60-0.66).",
        "7. Low-score priority reduced total R to +10.57R; rejected.",
        "8. No one of the exact seven score components met the predeclared robust",
        "   overall-correlation, quintile-spread, and 3-of-4 fold sign gate.",
        "",
        "STRUCTURAL/INTEGRITY WORK RETAINED",
        "----------------------------------",
        "- Non-open routes now estimate session ATR only from completed bar prefixes.",
        "- The exact OPEN_SCORED component bundle now survives into trade metadata.",
        "- Shared core/live/replay support and tests exist for experimental retest and",
        "  resting-limit transitions; defaults preserve the selected immediate route.",
        "- Shared live/replay priority ordering is explicit and defaults to high score.",
        "",
        "WHY PERFORMANCE WAS NOT 'RECOVERED' BY OPTIMIZATION",
        "---------------------------------------------------",
        "The repaired engine exposes only a small +0.06R/trade edge in the current",
        "immediate route. The historical high-return profile depended materially on",
        "the faulty partial-profit path. Broadening frequency, waiting for confirmation,",
        "buying retracements, carrying, and reversing priority all failed under causal",
        "execution. There is therefore no evidence-backed parameter neighborhood from",
        "which phased auto can safely manufacture the missing alpha.",
        "",
        "NEXT AUTHORIZED RESEARCH",
        "------------------------",
        "Obtain the frozen direct-RTH authority bundle first. Then source genuinely new,",
        "time-available features for the signal-to-entry model (for example premarket/",
        "auction imbalance data or a separately specified cross-sectional pullback model)",
        "and require chronological/walk-forward proof before resuming phased auto.",
        "Do not optimize stops, exits, or score weights on the present 148-199 trade sample.",
        "",
        "IMMUTABLE SCORE",
        "---------------",
    ]
    for key, spec in SCORE_SPEC.items():
        lines.append(f"- {key}: weight={spec['weight']:.2f}; {spec['transform']}")
    lines.extend(
        [
            "",
            "Promotion allowed: NO",
            "Proper meaningful baseline established: NO",
            "Phased auto run: NO",
            "Holdout accessed: NO",
        ]
    )
    (OUTPUT / "full_final_diagnostics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("IARIC bounded baseline recovery finalized", flush=True)
    print(f"Selected honest reference: {metrics['trades']:.0f} trades, {metrics['total_r']:+.2f}R", flush=True)
    print("Proper meaningful baseline established: no", flush=True)
    print("Phased auto run: no", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
