"""Run the bounded executable IARIC rank-cap then repaired-carry controls.

This follows causal timing Phase 0.  The rank stage tests whether rejecting the
weak daily-rank tail improves both total and average R.  Only the winning rank
then enters the carry stage, where the repaired shared overnight-stop path is
tested at two predeclared profit-lock thresholds.  The sealed holdout is never
accessed.
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
DEFAULT_PHASE0_RANKING = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/causal_entry_phase0/ranking.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/rank_carry_phase0"
)
READINESS_PATH = (
    REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
)
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--phase0-ranking", default=str(DEFAULT_PHASE0_RANKING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
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
    family: str,
) -> dict[str, Any]:
    mutations = deepcopy(base)
    mutations.update(overrides)
    return {
        "id": candidate_id,
        "family": family,
        "sources": ["post_integrity_reference", "causal_entry_phase0"],
        "mutations": mutations,
    }


def _decorate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["immutable_score"], row["immutable_score_components"] = _score(row["metrics"])
    return sorted(
        rows,
        key=lambda row: (
            float(row["immutable_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
            -float(row["metrics"].get("max_drawdown_pct", 1.0)),
        ),
        reverse=True,
    )


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


def _rank_gate(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    delta = _delta(metrics, control)
    return bool(
        float(metrics.get("total_trades", 0.0)) >= 120.0
        and delta["expected_total_r"] >= 2.0
        and delta["avg_r"] >= 0.02
        and float(metrics.get("profit_factor", 0.0)) >= float(control.get("profit_factor", 0.0))
        and float(metrics.get("max_drawdown_pct", 1.0))
        <= max(0.10, float(control.get("max_drawdown_pct", 1.0)) + 0.01)
    )


def _carry_gate(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    delta = _delta(metrics, control)
    return bool(
        float(metrics.get("total_trades", 0.0)) >= 100.0
        and delta["expected_total_r"] >= 3.0
        and delta["avg_r"] >= 0.025
        and float(metrics.get("profit_factor", 0.0)) >= 1.25
        and float(metrics.get("max_drawdown_pct", 1.0))
        <= max(0.10, float(control.get("max_drawdown_pct", 1.0)) + 0.015)
    )


def _reused_control(
    phase0_ranking: list[dict[str, Any]],
    base: dict[str, Any],
) -> dict[str, Any]:
    source = next(row for row in phase0_ranking if row["id"] == "timing_bar0_fill_0935")
    mutations = deepcopy(base)
    mutations["param_overrides.pb_v2_open_scored_after_bar"] = 0
    mutations["param_overrides.pb_v2_open_scored_rank_pct_max"] = 100.0
    mutations["param_overrides.pb_carry_enabled"] = False
    return {
        "id": "rank100_carry_off_control",
        "family": "rank_cap_control",
        "sources": ["causal_entry_phase0:timing_bar0_fill_0935"],
        "mutations": mutations,
        "metrics": dict(source["metrics"]),
        "error": "",
        "reused_executable_control": True,
        "source_signature": source["signature"],
    }


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
    phase0_path = Path(args.phase0_ranking).resolve()
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    base["param_overrides.pb_v2_open_scored_after_bar"] = 0
    base["param_overrides.pb_carry_enabled"] = False
    phase0_ranking = json.loads(phase0_path.read_text(encoding="utf-8"))
    control = _reused_control(phase0_ranking, base)
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _runner_fingerprint()
    cache_path = output_dir / "evaluation_cache.json"

    rank_candidates = [
        _candidate(
            base,
            f"rank{rank}_carry_off",
            {
                "param_overrides.pb_v2_open_scored_rank_pct_max": float(rank),
                "param_overrides.pb_carry_enabled": False,
            },
            "rank_cap",
        )
        for rank in (90, 80)
    ]
    rank_rows = [control] + _evaluate_batch(
        rank_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    rank_errors = [row for row in rank_rows if row.get("error")]
    if rank_errors:
        _write_json(output_dir / "rank_errors.json", rank_errors)
        raise RuntimeError(f"{len(rank_errors)} rank-cap evaluations failed")
    rank_rows = _decorate(rank_rows)
    for row in rank_rows:
        row["delta_vs_rank100"] = _delta(row["metrics"], control["metrics"])
        row["rank_improvement_gate"] = bool(
            row["id"] != control["id"] and _rank_gate(row["metrics"], control["metrics"])
        )
    rank_eligible = [row for row in rank_rows if row["rank_improvement_gate"]]
    rank_winner = rank_eligible[0] if rank_eligible else control
    _write_json(output_dir / "rank_ranking.json", rank_rows)

    carry_control = deepcopy(rank_winner)
    carry_control["id"] = f"{rank_winner['id']}__carry_off_control"
    carry_control["family"] = "carry_control"
    carry_candidates = [
        _candidate(
            rank_winner["mutations"],
            f"{rank_winner['id']}__carry_on_lock_{str(lock).replace('.', '_')}",
            {
                "param_overrides.pb_carry_enabled": True,
                "param_overrides.pb_v2_carry_profit_lock_r": lock,
            },
            "repaired_carry",
        )
        for lock in (0.75, 0.25)
    ]
    carry_rows = [carry_control] + _evaluate_batch(
        carry_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    carry_errors = [row for row in carry_rows if row.get("error")]
    if carry_errors:
        _write_json(output_dir / "carry_errors.json", carry_errors)
        raise RuntimeError(f"{len(carry_errors)} carry evaluations failed")
    carry_rows = _decorate(carry_rows)
    for row in carry_rows:
        row["delta_vs_carry_off"] = _delta(row["metrics"], carry_control["metrics"])
        row["carry_improvement_gate"] = bool(
            row["id"] != carry_control["id"]
            and _carry_gate(row["metrics"], carry_control["metrics"])
        )
    carry_eligible = [row for row in carry_rows if row["carry_improvement_gate"]]
    winner = carry_eligible[0] if carry_eligible else carry_control
    _write_json(output_dir / "carry_ranking.json", carry_rows)
    _write_json(output_dir / "preferred_config.json", dict(sorted(winner["mutations"].items())))

    next_decision = (
        "validate_combined_rank_carry_on_folds_before_structural_rebuild"
        if rank_eligible or carry_eligible
        else "incremental_controls_failed_proceed_to_shared_transition_rebuild"
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_incremental_controls_complete",
            "data_authority": (
                "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle"
            ),
            "promotion_allowed": False,
            "promotion_blockers": readiness.get("blocking_reasons", []),
            "data_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": args.max_workers,
            "immutable_score": SCORE_SPEC,
            "score_component_count": len(SCORE_SPEC),
            "sequence": [
                "rank cap: 100 control, 90 hypothesis, 80 perturbation",
                "repaired carry from rank winner: off, lock 0.75R, lock 0.25R",
            ],
            "reused_control": {
                "source": str(phase0_path.relative_to(REPO_ROOT)),
                "candidate": "timing_bar0_fill_0935",
                "reason": (
                    "Exact executable control from immediately preceding Phase 0; "
                    "strategy code and data fingerprint are unchanged."
                ),
            },
            "rank_winner_id": rank_winner["id"],
            "rank_winner_passed_gate": bool(rank_eligible),
            "preferred_candidate_id": winner["id"],
            "carry_winner_passed_gate": bool(carry_eligible),
            "preferred_signature": _signature(winner["mutations"]),
            "next_decision": next_decision,
        },
    )

    print("IARIC RANK/CARRY PHASE 0", flush=True)
    print("Rank stage:", flush=True)
    for row in rank_rows:
        metrics = row["metrics"]
        print(
            f"  {row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"gate={row['rank_improvement_gate']}",
            flush=True,
        )
    print("Carry stage:", flush=True)
    for row in carry_rows:
        metrics = row["metrics"]
        print(
            f"  {row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"gate={row['carry_improvement_gate']}",
            flush=True,
        )
    print(f"Preferred: {winner['id']}", flush=True)
    print(f"Next decision: {next_decision}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
