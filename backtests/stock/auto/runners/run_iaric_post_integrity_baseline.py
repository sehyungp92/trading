"""Evaluate the bounded IARIC baseline frontier after integrity repairs.

This is intentionally not a general search.  It compares only the two score
floors whose boundary was contaminated by the KLAC price-basis mismatch, with
the empirically harmful overnight path genuinely disabled.  The sealed
holdout is never accessed.
"""
from __future__ import annotations

import argparse
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
    _write_json,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import (
    MAX_WORKERS,
    SCORE_SPEC,
    _decorate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment/routes_ranking.json"
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ranking", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-01")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _candidate(base: dict[str, Any], score_floor: float) -> dict[str, Any]:
    mutations = deepcopy(base)
    mutations.update(
        {
            "param_overrides.pb_entry_score_min": score_floor,
            "param_overrides.pb_carry_enabled": False,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
            "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
        }
    )
    return {
        "id": f"post_integrity_open_only_score_{int(score_floor)}_intraday",
        "family": "post_integrity_boundary",
        "sources": ["routes_score40_open_only", "KLAC/CRWD price-basis audit", "carry ablation repair"],
        "mutations": mutations,
    }


def main() -> None:
    args = _args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"End date overlaps sealed holdout beginning {HOLDOUT_START}")
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    ranking = json.loads(Path(args.input_ranking).resolve().read_text(encoding="utf-8"))
    route = next(row for row in ranking if row["id"] == "routes_score40_open_only")
    candidates = [_candidate(route["mutations"], score) for score in (35.0, 40.0)]
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    rows = _evaluate_batch(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=output_dir / "post_integrity_evaluation_cache.json",
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output_dir / "post_integrity_errors.json", errors)
        raise RuntimeError(f"{len(errors)} post-integrity evaluations failed")

    ranked = _decorate(rows)
    winner = ranked[0]
    _write_json(output_dir / "post_integrity_baseline_ranking.json", ranked)
    _write_json(
        output_dir / "post_integrity_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window": {"start": args.start_date, "end": args.end_date},
            "holdout_start": HOLDOUT_START,
            "holdout_accessed": False,
            "max_workers": args.max_workers,
            "source_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "immutable_score": SCORE_SPEC,
            "integrity_repairs": [
                "causal split-basis normalization at the first completed RTH observation",
                "shared live/replay session ATR estimator",
                "shared live/replay V2 carry decision; pb_carry_enabled is effective",
            ],
            "bounded_experiment": {
                "reason": "Only the 35-40 entry-score boundary was contaminated by the false KLAC -43R trade.",
                "candidate_ids": [candidate["id"] for candidate in candidates],
            },
            "selected_candidate_id": winner["id"],
            "selected_full_period_eligible": winner["full_period_eligible"],
            "status": "promotion_candidate" if winner["full_period_eligible"] else "honest_reference_only",
        },
    )
    print(
        json.dumps(
            [
                {
                    "id": row["id"],
                    "trades": row["metrics"].get("total_trades"),
                    "total_r": row["metrics"].get("expected_total_r"),
                    "avg_r": row["metrics"].get("avg_r"),
                    "profit_factor": row["metrics"].get("profit_factor"),
                    "sharpe": row["metrics"].get("sharpe"),
                    "max_drawdown_pct": row["metrics"].get("max_drawdown_pct"),
                    "eligible": row["full_period_eligible"],
                }
                for row in ranked
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
