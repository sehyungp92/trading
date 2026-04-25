from __future__ import annotations

import pytest

from backtests.shared.parity.diagnostic_baselines import (
    collect_baseline_snapshot,
    load_manifest,
    repo_root,
)


_MANIFEST = load_manifest()
_REPO_ROOT = repo_root()


@pytest.mark.parametrize("entry", _MANIFEST["artifacts"], ids=lambda entry: entry["id"])
def test_canonical_backtest_baselines_are_frozen(entry: dict) -> None:
    snapshot = collect_baseline_snapshot(entry, root=_REPO_ROOT)

    assert snapshot["sha256"] == entry["sha256"]
    for metric_name, expected_value in entry["expected_metrics"].items():
        assert snapshot["metrics"][metric_name] == pytest.approx(expected_value, rel=0.0, abs=1e-9)
