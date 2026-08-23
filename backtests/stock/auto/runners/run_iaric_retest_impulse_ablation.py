"""Ablate only the starving impulse gate in IARIC's causal retest route.

The prior structural screen showed 13 arms from 420 eligible candidates.  This
single-candidate replay removes only the 0.15 daily-ATR minimum impulse; score,
retrace, confirmation, next-bar execution, risk, management, and exits remain
fixed.  It records both economics and the causal funnel without touching the
sealed holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.worker import evaluate_candidate_attribution
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import _score
from backtests.stock.auto.runners.run_iaric_structural_retest_phase0 import (
    DEFAULT_BASELINE,
    READINESS_PATH,
    _candidate,
    _fixed_base,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/retest_impulse_ablation"
)
CONTROL_METRICS = {
    "total_trades": 199.0,
    "expected_total_r": 11.903437561313723,
    "avg_r": 0.059816269152330265,
    "profit_factor": 1.2376668723130022,
    "sharpe": 0.6456252664839413,
    "max_drawdown_pct": 0.0649235789596363,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-01")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _delta(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0)) - float(value)
        for key, value in CONTROL_METRICS.items()
    }


def _material(metrics: dict[str, Any]) -> bool:
    delta = _delta(metrics)
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
    if args.max_workers != 1:
        raise ValueError("This one-candidate ablation requires max-workers=1")
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
    base = _fixed_base(
        json.loads(Path(args.baseline_config).resolve().read_text(encoding="utf-8"))
    )
    candidate = _candidate(
        base,
        "retest_no_min_impulse_35pct_12bar",
        {
            "param_overrides.pb_open_scored_transition": "confirmed_retest",
            "param_overrides.pb_open_scored_retest_retrace_frac": 0.35,
            "param_overrides.pb_open_scored_retest_window_bars": 12,
            "param_overrides.pb_open_scored_retest_min_close_pct": 0.55,
            "param_overrides.pb_open_scored_retest_min_impulse_atr": 0.0,
            "param_overrides.pb_open_scored_retest_max_extension_atr": 0.35,
        },
    )
    rows = _evaluate_batch(
        [candidate],
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=1,
        cache_path=output_dir / "evaluation_cache.json",
        source_fingerprint=_replay_source_fingerprint(),
        code_fingerprint=_fingerprint(),
        evaluation_fn=evaluate_candidate_attribution,
    )
    row = rows[0]
    if row.get("error"):
        _write_json(output_dir / "errors.json", rows)
        raise RuntimeError(row["error"])

    metrics = row["metrics"]
    score, score_components = _score(metrics)
    result = {
        "id": candidate["id"],
        "mutations": candidate["mutations"],
        "metrics": metrics,
        "delta_vs_immediate_control": _delta(metrics),
        "immutable_score": score,
        "immutable_score_components": score_components,
        "structural_materiality_gate": _material(metrics),
        "funnel_counters": row.get("funnel_counters", {}),
    }
    trades = row.pop("trade_attribution", [])
    _write_json(output_dir / "trade_attribution.json", trades)
    _write_json(output_dir / "result.json", result)
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_structural_ablation_complete",
            "data_authority": "legacy_diagnostic_only",
            "promotion_allowed": False,
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": 1,
            "candidate_signature": _signature(candidate["mutations"]),
            "single_changed_dimension": "pb_open_scored_retest_min_impulse_atr: 0.15 -> 0.0",
            "control_metrics": CONTROL_METRICS,
            "materiality_gate_passed": result["structural_materiality_gate"],
        },
    )
    print("IARIC RETEST IMPULSE ABLATION", flush=True)
    print(
        f"n={metrics.get('total_trades', 0):.0f} "
        f"R={metrics.get('expected_total_r', 0):+.2f} "
        f"avgR={metrics.get('avg_r', 0):+.3f} "
        f"PF={metrics.get('profit_factor', 0):.2f} "
        f"Sharpe={metrics.get('sharpe', 0):+.2f} "
        f"DD={metrics.get('max_drawdown_pct', 0):.2%}",
        flush=True,
    )
    print(json.dumps(result["funnel_counters"], indent=2, sort_keys=True), flush=True)
    print(f"Materiality gate: {result['structural_materiality_gate']}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
