"""Run the bounded IARIC corrected-carry Phase 0 experiment.

This is a causal integrity experiment, not a parameter search.  It compares
the archived round-1 configuration with carry disabled against the corrected
overnight-stop semantics at the configured 0.75R lock and one predeclared,
attainable 0.25R sensitivity.  The sealed holdout is excluded by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    SCORE_SPEC,
    _decorate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = REPO_ROOT / "backtests/output/stock/iaric/round_1/optimized_config.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/corrected_carry_phase0"
)
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
MAX_WORKERS = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _candidate(
    base: dict[str, Any],
    candidate_id: str,
    overrides: dict[str, Any],
    hypothesis: str,
) -> dict[str, Any]:
    mutations = dict(base)
    mutations.update(overrides)
    return {
        "id": candidate_id,
        "family": "corrected_overnight_stop",
        "sources": ["backtests/output/stock/iaric/round_1/optimized_config.json"],
        "hypothesis": hypothesis,
        "mutations": mutations,
    }


def _candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _candidate(
            base,
            "carry_disabled_control",
            {"param_overrides.pb_carry_enabled": False},
            "No overnight exposure; isolates the value added by corrected carry.",
        ),
        _candidate(
            base,
            "corrected_carry_lock_075",
            {
                "param_overrides.pb_carry_enabled": True,
                "param_overrides.pb_v2_carry_profit_lock_r": 0.75,
            },
            "Correct semantics at the archived/default 0.75R retrace allowance.",
        ),
        _candidate(
            base,
            "corrected_carry_lock_025",
            {
                "param_overrides.pb_carry_enabled": True,
                "param_overrides.pb_v2_carry_profit_lock_r": 0.25,
            },
            "One predeclared attainable-lock sensitivity; not a threshold sweep.",
        ),
    ]


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    return {
        "id": row["id"],
        "signature": row["signature"],
        "hypothesis": row["hypothesis"],
        "baseline_score": row["baseline_score"],
        "full_period_eligible": row["full_period_eligible"],
        "metrics": {
            key: metrics.get(key)
            for key in (
                "total_trades",
                "expected_total_r",
                "avg_r",
                "profit_factor",
                "sharpe",
                "max_drawdown_pct",
                "trades_per_month",
                "tail_loss_r",
                "net_profit",
            )
        },
    }


def main() -> None:
    args = _parse_args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"End date overlaps sealed holdout beginning {HOLDOUT_START}")
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    baseline_path = Path(args.baseline_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    if base.get("param_overrides.pb_open_scored_fill_timing") != "next_5m_open":
        raise ValueError("Baseline violates completed-5m -> next-5m-open execution parity")
    if float(base.get("param_overrides.pb_v2_partial_profit_trigger_r", 0.0)) != 0.0:
        raise ValueError("Bounded test requires the archived partial-profit path to remain disabled")

    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    runner_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    candidates = _candidates(base)
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "evaluation_cache.json",
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    by_id = {candidate["id"]: candidate for candidate in candidates}
    for row in rows:
        row["hypothesis"] = by_id[row["id"]]["hypothesis"]
    ranking = _decorate([row for row in rows if not row.get("error")])
    if len(ranking) != len(candidates):
        errors = {row["id"]: row.get("error") for row in rows if row.get("error")}
        raise RuntimeError(f"One or more bounded candidates failed: {errors}")

    # An economically identical sensitivity is not evidence for changing the
    # archived/default lock.  Make that parsimony rule explicit rather than
    # accepting alphabetical candidate order as an accidental tie-break.
    tie_preference = {
        "corrected_carry_lock_075": 2,
        "corrected_carry_lock_025": 1,
        "carry_disabled_control": 0,
    }
    ranking.sort(
        key=lambda row: (
            1 if row["full_period_eligible"] else 0,
            round(float(row["baseline_score"]), 12),
            round(float(row["metrics"].get("expected_total_r", 0.0)), 12),
            -round(float(row["metrics"].get("max_drawdown_pct", 1.0)), 12),
            tie_preference[row["id"]],
        ),
        reverse=True,
    )

    _write_json(output_dir / "ranking.json", ranking)
    winner = ranking[0]
    control = next(row for row in ranking if row["id"] == "carry_disabled_control")
    winner_r = float(winner["metrics"].get("expected_total_r", 0.0))
    control_r = float(control["metrics"].get("expected_total_r", 0.0))
    lock_075 = next(row for row in ranking if row["id"] == "corrected_carry_lock_075")
    lock_025 = next(row for row in ranking if row["id"] == "corrected_carry_lock_025")
    lock_sensitivity_identical = all(
        float(lock_075["metrics"].get(key, 0.0))
        == float(lock_025["metrics"].get(key, 0.0))
        for key in (
            "total_trades",
            "expected_total_r",
            "avg_r",
            "profit_factor",
            "sharpe",
            "max_drawdown_pct",
            "actual_carried_count",
        )
    )
    materially_beats_control = (
        winner["id"] != control["id"]
        and winner_r >= control_r + 5.0
        and float(winner["metrics"].get("avg_r", 0.0))
        > float(control["metrics"].get("avg_r", 0.0))
        and float(winner["metrics"].get("max_drawdown_pct", 1.0)) <= 0.12
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "corrected_carry_phase0",
        "scope": "three_predeclared_candidates_no_search",
        "status": "bounded_integrity_result",
        "data_authority": "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle",
        "baseline_config": str(baseline_path),
        "baseline_signature": _signature(base),
        "data_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "runner_fingerprint": runner_fingerprint,
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": args.max_workers,
        "score_spec": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "winner": _compact(winner),
        "control": _compact(control),
        "winner_delta_total_r_vs_carry_disabled": winner_r - control_r,
        "materially_beats_carry_disabled": materially_beats_control,
        "lock_025_and_075_economically_identical": lock_sensitivity_identical,
        "tie_break_policy": "prefer archived/default 0.75R lock when economics are identical",
        "full_diagnostics_run": False,
        "full_diagnostics_reason": "corrected carry did not clear the predeclared materiality threshold",
        "promotion_allowed": False,
        "promotion_blockers": [
            "winner fails full-period baseline gates",
            "legacy replay is diagnostic-only",
            "bounded Phase 0 is not a phased optimization round",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)

    print("\nIARIC CORRECTED-CARRY PHASE 0", flush=True)
    for rank, row in enumerate(ranking, 1):
        metrics = row["metrics"]
        print(
            f"{rank}. {row['id']}: trades={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.4f} "
            f"PF={metrics.get('profit_factor', 0):.3f} "
            f"Sharpe={metrics.get('sharpe', 0):+.3f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"score={row['baseline_score']:+.4f}",
            flush=True,
        )
    print(f"Holdout accessed: no (sealed from {HOLDOUT_START})", flush=True)


if __name__ == "__main__":
    main()
