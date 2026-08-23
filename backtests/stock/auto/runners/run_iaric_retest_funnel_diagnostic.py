"""Attribute starvation in the rejected IARIC open-scored retest route.

This is deliberately a one-candidate diagnostic.  It replays the least
restrictive predeclared retest window, records the full causal funnel, and
does not inspect the sealed holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backtests.stock.auto.iaric.worker import evaluate_candidate_attribution
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _code_fingerprint,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_retest_phase0 import (
    DEFAULT_BASELINE,
    READINESS_PATH,
    _candidate,
    _fixed_base,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/retest_funnel_diagnostic"
)
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = _args()
    if args.max_workers != 1:
        raise ValueError("This one-candidate diagnostic requires max-workers=1")
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
        "retest_35pct_12bar_funnel",
        {
            "param_overrides.pb_open_scored_transition": "confirmed_retest",
            "param_overrides.pb_open_scored_retest_retrace_frac": 0.35,
            "param_overrides.pb_open_scored_retest_window_bars": 12,
            "param_overrides.pb_open_scored_retest_min_close_pct": 0.55,
            "param_overrides.pb_open_scored_retest_min_impulse_atr": 0.15,
            "param_overrides.pb_open_scored_retest_max_extension_atr": 0.35,
        },
    )
    rows = _evaluate_batch(
        [candidate],
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=1,
        cache_path=output_dir / "attribution_cache.json",
        source_fingerprint=_replay_source_fingerprint(),
        code_fingerprint=_fingerprint(),
        evaluation_fn=evaluate_candidate_attribution,
    )
    row = rows[0]
    if row.get("error"):
        _write_json(output_dir / "errors.json", rows)
        raise RuntimeError(row["error"])

    funnel = row.get("funnel_counters", {})
    trades = row.pop("trade_attribution", [])
    _write_json(output_dir / "trade_attribution.json", trades)
    _write_json(output_dir / "result.json", row)
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_only_complete",
            "data_authority": "legacy_diagnostic_only",
            "promotion_allowed": False,
            "training_window": {"start": args.start_date, "end": args.end_date},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": 1,
            "candidate_signature": _signature(candidate["mutations"]),
            "purpose": "attribute rejected retest-route starvation before relaxation",
            "funnel_counters": funnel,
        },
    )
    print("IARIC RETEST FUNNEL DIAGNOSTIC", flush=True)
    print(json.dumps(funnel, indent=2, sort_keys=True), flush=True)
    print(f"Executed trades: {row['metrics'].get('total_trades', 0):.0f}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
