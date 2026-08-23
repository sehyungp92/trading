"""Validate IARIC's sole Pareto-improving rank cap across time.

Rank<=90 improved total R, expectancy, PF, Sharpe, and drawdown on the full
pre-holdout window, but missed the predeclared +2R materiality hurdle.  This
final bounded baseline-selection check compares it with rank<=100 in four
chronological folds.  No parameters are fitted and the holdout remains sealed.
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
from backtests.stock.auto.runners.run_iaric_structural_baseline import MAX_WORKERS
from backtests.stock.auto.runners.run_iaric_structural_retest_phase0 import READINESS_PATH


REPO_ROOT = Path(__file__).resolve().parents[4]
RANKING_PATH = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/rank_carry_phase0/rank_ranking.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/baseline_establishment/rank90_fold_validation"
)
FOLDS = (
    ("2024_h1", "2024-01-01", "2024-06-30"),
    ("2024_h2", "2024-07-01", "2024-12-31"),
    ("2025_h1", "2025-01-01", "2025-06-30"),
    ("2025_h2_to_2026_03", "2025-07-01", "2026-03-01"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_code_fingerprint().encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _candidate(row: dict[str, Any], fold: str) -> dict[str, Any]:
    return {
        "id": f"{fold}__{row['id']}",
        "family": "rank90_fold_validation",
        "sources": ["rank_carry_phase0"],
        "mutations": dict(row["mutations"]),
    }


def main() -> None:
    args = _args()
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {MAX_WORKERS}")
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    if not readiness.get("frozen_bundle_available") and not args.allow_legacy_data:
        raise RuntimeError(
            "Authoritative frozen replay bundle is unavailable; pass "
            "--allow-legacy-data for diagnostic-only work."
        )
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    control = next(row for row in ranking if row["id"] == "rank100_carry_off_control")
    challenger = next(row for row in ranking if row["id"] == "rank90_carry_off")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "evaluation_cache.json"
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _fingerprint()
    fold_rows: list[dict[str, Any]] = []

    for fold, start, end in FOLDS:
        if end >= HOLDOUT_START:
            raise ValueError(f"Fold {fold} overlaps sealed holdout")
        rows = _evaluate_batch(
            [_candidate(control, fold), _candidate(challenger, fold)],
            start_date=start,
            end_date=end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
        errors = [row for row in rows if row.get("error")]
        if errors:
            _write_json(output_dir / "errors.json", errors)
            raise RuntimeError(f"{len(errors)} rank-fold evaluations failed")
        control_result = next(row for row in rows if row["id"].endswith(control["id"]))
        challenger_result = next(row for row in rows if row["id"].endswith(challenger["id"]))
        delta = {
            key: float(challenger_result["metrics"].get(key, 0.0))
            - float(control_result["metrics"].get(key, 0.0))
            for key in (
                "total_trades",
                "expected_total_r",
                "avg_r",
                "profit_factor",
                "sharpe",
                "max_drawdown_pct",
            )
        }
        fold_rows.append(
            {
                "fold": fold,
                "start": start,
                "end": end,
                "control": control_result,
                "challenger": challenger_result,
                "delta": delta,
                "challenger_wins_total_r": delta["expected_total_r"] > 0,
                "challenger_wins_avg_r": delta["avg_r"] > 0,
            }
        )

    total_r_wins = sum(row["challenger_wins_total_r"] for row in fold_rows)
    avg_r_wins = sum(row["challenger_wins_avg_r"] for row in fold_rows)
    max_dd_worsening = max(row["delta"]["max_drawdown_pct"] for row in fold_rows)
    validation_passed = bool(
        total_r_wins >= 3
        and avg_r_wins >= 3
        and max_dd_worsening <= 0.02
        and float(challenger["metrics"]["expected_total_r"])
        > float(control["metrics"]["expected_total_r"])
    )
    selected = challenger if validation_passed else control
    summary = {
        "validation_passed": validation_passed,
        "total_r_fold_wins": total_r_wins,
        "avg_r_fold_wins": avg_r_wins,
        "max_fold_drawdown_worsening": max_dd_worsening,
        "full_window_control": control,
        "full_window_challenger": challenger,
        "selected_candidate_id": selected["id"],
        "folds": fold_rows,
    }
    _write_json(output_dir / "fold_results.json", fold_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "selected_config.json", dict(sorted(selected["mutations"].items())))
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "diagnostic_baseline_fold_validation_complete",
            "data_authority": "legacy_diagnostic_only",
            "promotion_allowed": False,
            "training_window": {"start": FOLDS[0][1], "end": FOLDS[-1][2]},
            "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
            "max_workers": args.max_workers,
            "candidate_count": 2,
            "fold_count": len(FOLDS),
            "selection_rule": {
                "min_total_r_fold_wins": 3,
                "min_avg_r_fold_wins": 3,
                "max_fold_drawdown_worsening": 0.02,
                "full_window_total_r_must_improve": True,
            },
            "validation_passed": validation_passed,
            "selected_candidate_id": selected["id"],
            "selected_signature": _signature(selected["mutations"]),
        },
    )
    print("IARIC RANK90 CHRONOLOGICAL FOLD VALIDATION", flush=True)
    for row in fold_rows:
        control_metrics = row["control"]["metrics"]
        challenger_metrics = row["challenger"]["metrics"]
        print(
            f"{row['fold']}: control={control_metrics.get('expected_total_r', 0):+.2f}R "
            f"rank90={challenger_metrics.get('expected_total_r', 0):+.2f}R "
            f"delta={row['delta']['expected_total_r']:+.2f}R "
            f"avg_delta={row['delta']['avg_r']:+.3f}",
            flush=True,
        )
    print(f"Validation passed: {validation_passed}", flush=True)
    print(f"Selected: {selected['id']}", flush=True)
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
