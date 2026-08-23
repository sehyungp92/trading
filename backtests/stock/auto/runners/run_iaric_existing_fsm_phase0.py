"""Re-evaluate IARIC's existing confirmation FSM after the causal ATR repair.

Earlier non-open route results used a full-session ATR at every intraday bar and
are invalid for live-parity decisions.  This bounded screen compares the
unchanged immediate control with delayed-confirm, opening-reclaim, and their
combination.  Score, selector, execution, risk, management, and exits remain
fixed; the sealed holdout is never accessed.
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
from backtests.stock.auto.runners.run_iaric_structural_retest_phase0 import (
    DEFAULT_BASELINE,
    READINESS_PATH,
    _candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/existing_fsm_phase0"
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


def _base(raw: dict[str, Any]) -> dict[str, Any]:
    base = deepcopy(raw)
    base.update(
        {
            "param_overrides.pb_v2_open_scored_after_bar": 0,
            "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
            "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
            "param_overrides.pb_open_scored_transition": "next_bar",
            "param_overrides.pb_carry_enabled": False,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
            "param_overrides.pb_v2_vwap_bounce_enabled": False,
            "param_overrides.pb_v2_afternoon_retest_enabled": False,
        }
    )
    return base


def _candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    def routes(*, open_scored: bool, delayed: bool, reclaim: bool) -> dict[str, Any]:
        return {
            "param_overrides.pb_v2_open_scored_enabled": open_scored,
            "param_overrides.pb_open_scored_enabled": open_scored,
            "param_overrides.pb_delayed_confirm_enabled": delayed,
            "param_overrides.pb_opening_reclaim_enabled": reclaim,
        }

    return [
        _candidate(base, "immediate_next_bar_control", routes(open_scored=True, delayed=False, reclaim=False)),
        _candidate(base, "delayed_confirm_only", routes(open_scored=False, delayed=True, reclaim=False)),
        _candidate(base, "opening_reclaim_only", routes(open_scored=False, delayed=False, reclaim=True)),
        _candidate(base, "confirmation_fsm_combined", routes(open_scored=False, delayed=True, reclaim=True)),
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


def _material(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    delta = _delta(metrics, control)
    return bool(
        float(metrics.get("total_trades", 0.0)) >= 80.0
        and delta["expected_total_r"] >= 5.0
        and delta["avg_r"] >= 0.05
        and float(metrics.get("profit_factor", 0.0)) >= 1.35
        and float(metrics.get("sharpe", 0.0)) >= 0.90
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.08
    )


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
    base = _base(json.loads(baseline_path.read_text(encoding="utf-8")))
    candidates = _candidates(base)
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "evaluation_cache.json",
        source_fingerprint=_replay_source_fingerprint(),
        code_fingerprint=_fingerprint(),
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} existing-FSM evaluations failed")

    control = next(row for row in rows if row["id"] == "immediate_next_bar_control")
    for row in rows:
        row["immutable_score"], row["immutable_score_components"] = _score(row["metrics"])
        row["delta_vs_control"] = _delta(row["metrics"], control["metrics"])
        row["structural_materiality_gate"] = bool(
            row["id"] != control["id"] and _material(row["metrics"], control["metrics"])
        )
    rows.sort(
        key=lambda row: (
            1 if row["structural_materiality_gate"] else 0,
            float(row["immutable_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
        ),
        reverse=True,
    )
    eligible = [row for row in rows if row["structural_materiality_gate"]]
    winner = eligible[0] if eligible else control
    _write_json(output_dir / "ranking.json", rows)
    _write_json(output_dir / "preferred_config.json", dict(sorted(winner["mutations"].items())))
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_structural_phase0_complete",
            "data_authority": "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle",
            "promotion_allowed": False,
            "promotion_blockers": readiness.get("blocking_reasons", []),
            "baseline_path": str(baseline_path.relative_to(REPO_ROOT)),
            "baseline_signature": _signature(base),
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": args.max_workers,
            "immutable_score": SCORE_SPEC,
            "score_component_count": len(SCORE_SPEC),
            "causal_atr_repair": "session ATR uses completed prefix at every decision bar",
            "preferred_candidate_id": winner["id"],
            "preferred_passed_materiality_gate": bool(eligible),
            "preferred_signature": _signature(winner["mutations"]),
            "next_decision": (
                "validate_existing_fsm_winner_on_chronological_folds"
                if eligible
                else "existing_fsm_rejected_design_causal_resting_retrace_entry"
            ),
        },
    )
    print("IARIC EXISTING FSM PHASE 0 AFTER CAUSAL ATR REPAIR", flush=True)
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['id']}: n={metrics.get('total_trades', 0):.0f} "
            f"R={metrics.get('expected_total_r', 0):+.2f} "
            f"avgR={metrics.get('avg_r', 0):+.3f} "
            f"PF={metrics.get('profit_factor', 0):.2f} "
            f"Sharpe={metrics.get('sharpe', 0):+.2f} "
            f"DD={metrics.get('max_drawdown_pct', 0):.2%} "
            f"gate={row['structural_materiality_gate']}",
            flush=True,
        )
    print(f"Preferred: {winner['id']}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
