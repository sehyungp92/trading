"""Test the sign of IARIC OPEN_SCORED capacity priority.

The route score is anti-predictive in every executable timing diagnostic, but
the current adapters allocate scarce slots to the highest score first.  This
bounded test changes only that ordering among already-qualified candidates.
Admission, all seven score components, thresholds, timing, frequency caps,
risk, management, and exits remain fixed.  The holdout stays sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.phase_scoring import (
    V5R1_PHASE_SCORING_WEIGHTS,
    score_v5r1_pullback_phase,
)
from backtests.stock.auto.iaric.worker import evaluate_candidate_diagnostics
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import MAX_WORKERS
from backtests.stock.auto.runners.run_iaric_structural_retest_phase0 import (
    DEFAULT_BASELINE,
    READINESS_PATH,
    SCORE_SPEC,
    _candidate,
    _fixed_base,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/score_priority_phase0"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-01")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


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
    candidates = [
        _candidate(
            base,
            "high_score_priority_control",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_open_scored_priority": "high_score",
            },
        ),
        _candidate(
            base,
            "low_score_priority",
            {
                "param_overrides.pb_open_scored_transition": "next_bar",
                "param_overrides.pb_open_scored_priority": "low_score",
            },
        ),
    ]
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "evaluation_cache.json",
        source_fingerprint=_replay_source_fingerprint(),
        code_fingerprint=_fingerprint(),
        evaluation_fn=evaluate_candidate_diagnostics,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} score-priority evaluations failed")

    control = next(row for row in rows if row["id"] == "high_score_priority_control")
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
        delta = row["delta_vs_control"]
        row["priority_materiality_gate"] = bool(
            row["id"] != control["id"]
            and delta["expected_total_r"] >= 3.0
            and delta["avg_r"] >= 0.03
            and float(row["metrics"].get("profit_factor", 0.0)) >= 1.35
            and float(row["metrics"].get("sharpe", 0.0)) >= 0.90
            and float(row["metrics"].get("max_drawdown_pct", 1.0)) <= 0.08
        )
    rows.sort(
        key=lambda row: (
            1 if row["priority_materiality_gate"] else 0,
            float(row["immutable_score"]),
        ),
        reverse=True,
    )
    eligible = [row for row in rows if row["priority_materiality_gate"]]
    winner = eligible[0] if eligible else control
    _write_json(output_dir / "ranking.json", rows)
    _write_json(output_dir / "preferred_config.json", dict(sorted(winner["mutations"].items())))
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_score_direction_phase0_complete",
            "data_authority": "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle",
            "promotion_allowed": False,
            "promotion_blockers": readiness.get("blocking_reasons", []),
            "baseline_path": str(baseline_path.relative_to(REPO_ROOT)),
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": args.max_workers,
            "immutable_score": SCORE_SPEC,
            "score_component_count": len(SCORE_SPEC),
            "single_changed_dimension": "OPEN_SCORED capacity priority sign",
            "admission_or_score_changed": False,
            "live_replay_priority_core": "strategies.stock.iaric.core.logic.route_priority_value",
            "preferred_candidate_id": winner["id"],
            "preferred_passed_materiality_gate": bool(eligible),
            "preferred_signature": _signature(winner["mutations"]),
            "next_decision": (
                "validate_low_score_priority_on_chronological_folds"
                if eligible
                else "priority_inversion_rejected_run_component_attribution"
            ),
        },
    )
    print("IARIC OPEN-SCORED PRIORITY SIGN PHASE 0", flush=True)
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"gate={row['priority_materiality_gate']}",
            flush=True,
        )
    print(f"Preferred: {winner['id']}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
