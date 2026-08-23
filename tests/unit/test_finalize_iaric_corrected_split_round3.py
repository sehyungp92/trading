from __future__ import annotations

from backtests.scripts.finalize_iaric_corrected_split_round3 import (
    _paired_attribution,
    _settings_diff,
    _stats,
)


def _trade(symbol: str, entry: str, r_multiple: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "entry_time": entry,
        "r_multiple": r_multiple,
        "net_pnl": r_multiple * 100.0,
    }


def test_stats_and_paired_attribution_reconcile_candidate_delta() -> None:
    control = [_trade("A", "2026-03-02", -1.0), _trade("B", "2026-03-03", 0.5)]
    winner = [_trade("B", "2026-03-03", 0.6), _trade("C", "2026-03-04", 2.0)]

    attribution = _paired_attribution(control, winner)

    assert _stats(winner)["total_r"] == 2.6
    assert attribution["common"]["trades"] == 1
    assert attribution["control_only"]["total_r"] == -1.0
    assert attribution["winner_only"]["total_r"] == 2.0
    assert abs(attribution["reconciliation_delta_r"] - 3.1) < 1e-12


def test_settings_diff_is_granular_and_includes_added_settings() -> None:
    assert _settings_diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == [
        {"setting": "b", "before": 2, "after": 3},
        {"setting": "c", "before": "<not_set>", "after": 4},
    ]
