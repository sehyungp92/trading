"""Attribute IARIC's exact seven OPEN_SCORED components before rebuilding it.

The diagnostic replays only the unchanged immediate reference, using repaired
score metadata, and measures each component against realized R overall and in
four chronological pre-holdout folds.  It does not optimize weights or inspect
the sealed holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    / "backtests/output/stock/iaric/baseline_establishment/score_component_diagnostic"
)
COMPONENTS = (
    "daily_signal",
    "reclaim",
    "volume",
    "vwap_hold",
    "cpr",
    "speed",
    "quality_adjustment",
)
FOLDS = (
    ("2024_h1", "2024-01-01", "2024-06-30"),
    ("2024_h2", "2024-07-01", "2024-12-31"),
    ("2025_h1", "2025-01-01", "2025-06-30"),
    ("2025_h2_to_2026_03", "2025-07-01", "2026-03-01"),
)


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


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def _spearman(values: np.ndarray, outcomes: np.ndarray) -> float:
    if len(values) < 3 or np.ptp(values) <= 1e-12 or np.ptp(outcomes) <= 1e-12:
        return 0.0
    return float(np.corrcoef(_rank(values), _rank(outcomes))[0, 1])


def _profile(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if key == "route_score":
        values = np.asarray([float(row.get("route_score", 0.0)) for row in rows], dtype=float)
    else:
        values = np.asarray(
            [float((row.get("score_components") or {}).get(key, 0.0)) for row in rows],
            dtype=float,
        )
    outcomes = np.asarray([float(row.get("r", 0.0)) for row in rows], dtype=float)
    order = np.argsort(values, kind="mergesort")
    buckets = np.array_split(order, 5)
    quintiles = [
        {
            "n": int(len(bucket)),
            "component_mean": float(np.mean(values[bucket])) if len(bucket) else 0.0,
            "avg_r": float(np.mean(outcomes[bucket])) if len(bucket) else 0.0,
            "total_r": float(np.sum(outcomes[bucket])) if len(bucket) else 0.0,
        }
        for bucket in buckets
    ]
    fold_correlations: dict[str, float] = {}
    for name, start, end in FOLDS:
        indices = [
            idx
            for idx, row in enumerate(rows)
            if start <= str(row.get("entry_time", ""))[:10] <= end
        ]
        fold_correlations[name] = _spearman(values[indices], outcomes[indices]) if indices else 0.0
    q5_minus_q1 = (
        float(quintiles[-1]["avg_r"] - quintiles[0]["avg_r"])
        if quintiles
        else 0.0
    )
    overall = _spearman(values, outcomes)
    negative_folds = sum(value < 0 for value in fold_correlations.values())
    positive_folds = sum(value > 0 for value in fold_correlations.values())
    if overall <= -0.10 and q5_minus_q1 <= -0.10 and negative_folds >= 3:
        classification = "robust_negative"
    elif overall >= 0.10 and q5_minus_q1 >= 0.10 and positive_folds >= 3:
        classification = "robust_positive"
    else:
        classification = "weak_or_unstable"
    return {
        "n": len(rows),
        "spearman_rho": overall,
        "q5_minus_q1_avg_r": q5_minus_q1,
        "quintiles": quintiles,
        "fold_spearman_rho": fold_correlations,
        "negative_fold_count": negative_folds,
        "positive_fold_count": positive_folds,
        "classification": classification,
    }


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
    baseline_path = Path(args.baseline_config).resolve()
    base = _fixed_base(json.loads(baseline_path.read_text(encoding="utf-8")))
    candidate = _candidate(
        base,
        "immediate_high_score_priority_component_attribution",
        {
            "param_overrides.pb_open_scored_transition": "next_bar",
            "param_overrides.pb_open_scored_priority": "high_score",
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
    trades = row.pop("trade_attribution", [])
    profiles = {
        key: _profile(trades, key)
        for key in ("route_score", *COMPONENTS)
    }
    summary = {
        "candidate_id": candidate["id"],
        "metrics": row["metrics"],
        "trade_count": len(trades),
        "component_count": len(COMPONENTS),
        "components": list(COMPONENTS),
        "profiles": profiles,
        "robust_negative_components": [
            key for key in COMPONENTS if profiles[key]["classification"] == "robust_negative"
        ],
        "robust_positive_components": [
            key for key in COMPONENTS if profiles[key]["classification"] == "robust_positive"
        ],
    }
    _write_json(output_dir / "trade_attribution.json", trades)
    _write_json(output_dir / "summary.json", summary)
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
            "component_count": len(COMPONENTS),
            "folds": [dict(name=name, start=start, end=end) for name, start, end in FOLDS],
            "purpose": "attribute score components before any structural score rebuild",
        },
    )
    print("IARIC SCORE COMPONENT DIAGNOSTIC", flush=True)
    for key, profile in profiles.items():
        print(
            f"{key}: rho={profile['spearman_rho']:+.3f} "
            f"Q5-Q1={profile['q5_minus_q1_avg_r']:+.3f} "
            f"negative_folds={profile['negative_fold_count']}/4 "
            f"class={profile['classification']}",
            flush=True,
        )
    print("Holdout accessed: no", flush=True)


if __name__ == "__main__":
    main()
