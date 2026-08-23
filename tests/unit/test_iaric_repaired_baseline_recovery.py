from __future__ import annotations

import json
from pathlib import Path

from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    FOLDS,
    SCORE_SPEC,
    _fold_summary,
    _merge_evaluation_with_candidate,
    _read_cache,
    _signature,
)
from backtests.stock.auto.runners.run_iaric_structural_baseline import (
    SCORE_SPEC as STRUCTURAL_SCORE_SPEC,
    _full_eligible as structural_full_eligible,
    _score as structural_score,
)


def test_recovery_score_is_capped_at_seven_components() -> None:
    assert len(SCORE_SPEC) == 7


def test_structural_score_is_seven_component_and_economically_monotonic() -> None:
    assert len(STRUCTURAL_SCORE_SPEC) == 7
    weak = {
        "expected_total_r": 5.0, "avg_r": 0.01, "profit_factor": 1.02,
        "sharpe": 0.2, "max_drawdown_pct": 0.09, "trades_per_month": 8.0,
        "tail_loss_r": -1.2,
    }
    strong = {
        "expected_total_r": 80.0, "avg_r": 0.15, "profit_factor": 1.6,
        "sharpe": 2.5, "max_drawdown_pct": 0.05, "trades_per_month": 25.0,
        "tail_loss_r": -0.7,
    }
    assert structural_score(strong)[0] > structural_score(weak)[0]
    assert structural_full_eligible({**strong, "total_trades": 500.0})
    assert not structural_full_eligible({**strong, "total_trades": 500.0, "expected_total_r": 10.0})


def test_period_evaluation_cannot_be_overwritten_by_candidate_metrics() -> None:
    candidate = {
        "id": "candidate",
        "family": "test",
        "sources": ["baseline"],
        "mutations": {"param": 1},
        "metrics": {"net_profit": 999.0},
        "economic_score": 0.99,
    }
    evaluation = {
        "signature": _signature(candidate["mutations"]),
        "metrics": {"net_profit": 12.0},
        "economic_score": 0.12,
        "error": "",
    }

    merged = _merge_evaluation_with_candidate(evaluation, candidate)

    assert merged["metrics"]["net_profit"] == 12.0
    assert merged["economic_score"] == 0.12
    assert merged["id"] == "candidate"


def test_fold_summary_uses_distinct_period_metrics() -> None:
    candidate = {
        "id": "candidate",
        "mutations": {"param": 1},
        "economic_score": 0.4,
    }
    sig = _signature(candidate["mutations"])
    fold_results = {}
    expected_avg_rs = [0.01, 0.02, -0.03, 0.04]
    for (fold_name, _, _), avg_r in zip(FOLDS, expected_avg_rs, strict=True):
        fold_results[fold_name] = [
            {
                "signature": sig,
                "metrics": {
                    "avg_r": avg_r,
                    "profit_factor": 1.1,
                    "max_drawdown_pct": 0.05,
                    "net_profit": 10.0,
                    "total_trades": 20.0,
                    "expected_total_r": avg_r * 20.0,
                    "sharpe": 0.2,
                },
                "error": "",
            }
        ]

    summary = _fold_summary(candidate, fold_results)

    assert [row["avg_r"] for row in summary["folds"]] == expected_avg_rs
    assert summary["positive_fold_count"] == 3
    assert summary["worst_fold_avg_r"] == -0.03


def test_recovery_cache_is_source_fingerprint_namespaced(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"evaluations": {"2024-01-01|2024-06-30|abc": {"metrics": {}}}}),
        encoding="utf-8",
    )

    migrated = _read_cache(path, "source-a", "code-a")

    assert migrated["source_fingerprint"] == "source-a"
    assert "source-a|code-a|2024-01-01|2024-06-30|abc" in migrated["evaluations"]


def test_recovery_cache_invalidates_on_source_change(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"source_fingerprint": "source-a", "evaluations": {"cached": {}}}),
        encoding="utf-8",
    )

    invalidated = _read_cache(path, "source-b", "code-a")

    assert invalidated["evaluations"] == {}
    assert invalidated["invalidated_previous_source_fingerprint"] == "source-a"


def test_recovery_cache_invalidates_on_code_change(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps(
            {
                "source_fingerprint": "source-a",
                "code_fingerprint": "code-a",
                "evaluations": {"cached": {}},
            }
        ),
        encoding="utf-8",
    )

    invalidated = _read_cache(path, "source-a", "code-b")

    assert invalidated["evaluations"] == {}
    assert invalidated["invalidated_previous_code_fingerprint"] == "code-a"
