import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.services.deployment import (
    ALCB_ALLOCATION_ENV,
    DEPLOY_MODE_ENV,
    IARIC_ALLOCATION_ENV,
    PAPER_CAPITAL_ENV,
    US_ORB_ALLOCATION_ENV,
    DeploymentConfigError,
    resolve_paper_nav,
    resolve_strategy_capital_allocation,
)

ALGO_TRADER_ENV = "ALGO_TRADER_ENV"


# --- Strategy capital allocation tests (pure, no env dependency) ---


def test_combined_mode_defaults_to_equal_split(monkeypatch):
    monkeypatch.delenv(DEPLOY_MODE_ENV, raising=False)
    monkeypatch.delenv(ALCB_ALLOCATION_ENV, raising=False)
    monkeypatch.delenv(IARIC_ALLOCATION_ENV, raising=False)
    monkeypatch.delenv(US_ORB_ALLOCATION_ENV, raising=False)

    iaric = resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)
    us_orb = resolve_strategy_capital_allocation("US_ORB_v1", raw_nav=100_000.0)
    alcb = resolve_strategy_capital_allocation("ALCB_v1", raw_nav=100_000.0)

    assert iaric.enabled_for_strategy is True
    assert iaric.capital_fraction == pytest.approx(1.0 / 3.0)
    assert iaric.allocated_nav == pytest.approx(100_000.0 / 3.0)
    assert us_orb.capital_fraction == pytest.approx(1.0 / 3.0)
    assert us_orb.allocated_nav == pytest.approx(100_000.0 / 3.0)
    assert alcb.capital_fraction == pytest.approx(1.0 / 3.0)
    assert alcb.allocated_nav == pytest.approx(100_000.0 / 3.0)


def test_combined_mode_supports_uneven_split(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "both")
    monkeypatch.setenv(IARIC_ALLOCATION_ENV, "30")
    monkeypatch.setenv(US_ORB_ALLOCATION_ENV, "50")
    monkeypatch.setenv(ALCB_ALLOCATION_ENV, "20")

    iaric = resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)
    us_orb = resolve_strategy_capital_allocation("US_ORB_v1", raw_nav=100_000.0)
    alcb = resolve_strategy_capital_allocation("ALCB_v1", raw_nav=100_000.0)

    assert iaric.capital_fraction == pytest.approx(0.30)
    assert iaric.allocated_nav == pytest.approx(30_000.0)
    assert us_orb.capital_fraction == pytest.approx(0.50)
    assert us_orb.allocated_nav == pytest.approx(50_000.0)
    assert alcb.capital_fraction == pytest.approx(0.20)
    assert alcb.allocated_nav == pytest.approx(20_000.0)


def test_combined_mode_rejects_invalid_total(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "both")
    monkeypatch.setenv(IARIC_ALLOCATION_ENV, "60")
    monkeypatch.setenv(US_ORB_ALLOCATION_ENV, "30")
    monkeypatch.setenv(ALCB_ALLOCATION_ENV, "20")

    with pytest.raises(DeploymentConfigError, match="must equal 100"):
        resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)


def test_combined_mode_rejects_non_positive_split(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "both")
    monkeypatch.setenv(IARIC_ALLOCATION_ENV, "0")
    monkeypatch.setenv(US_ORB_ALLOCATION_ENV, "50")
    monkeypatch.setenv(ALCB_ALLOCATION_ENV, "50")

    with pytest.raises(DeploymentConfigError, match="must be > 0"):
        resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "swing")

    with pytest.raises(DeploymentConfigError, match="must be one of"):
        resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)


def test_solo_modes_grant_full_capital_to_active_strategy(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "iaric")

    iaric = resolve_strategy_capital_allocation("IARIC_v1", raw_nav=100_000.0)
    us_orb = resolve_strategy_capital_allocation("US_ORB_v1", raw_nav=100_000.0)
    alcb = resolve_strategy_capital_allocation("ALCB_v1", raw_nav=100_000.0)

    assert iaric.enabled_for_strategy is True
    assert iaric.capital_fraction == 1.0
    assert iaric.allocated_nav == 100_000.0
    assert us_orb.enabled_for_strategy is False
    assert us_orb.capital_fraction == 0.0
    assert us_orb.allocated_nav == 0.0
    assert alcb.enabled_for_strategy is False
    assert alcb.capital_fraction == 0.0
    assert alcb.allocated_nav == 0.0
    with pytest.raises(RuntimeError, match="disabled by"):
        us_orb.assert_enabled()

    monkeypatch.setenv(DEPLOY_MODE_ENV, "us_orb")
    us_orb_only = resolve_strategy_capital_allocation("US_ORB_v1", raw_nav=100_000.0)
    assert us_orb_only.enabled_for_strategy is True
    assert us_orb_only.capital_fraction == 1.0
    assert us_orb_only.allocated_nav == 100_000.0

    monkeypatch.setenv(DEPLOY_MODE_ENV, "alcb")
    alcb_only = resolve_strategy_capital_allocation("ALCB_v1", raw_nav=100_000.0)
    assert alcb_only.enabled_for_strategy is True
    assert alcb_only.capital_fraction == 1.0
    assert alcb_only.allocated_nav == 100_000.0


def test_positive_allocated_nav_validation(monkeypatch):
    monkeypatch.setenv(DEPLOY_MODE_ENV, "both")
    allocation = resolve_strategy_capital_allocation("IARIC_v1", raw_nav=0.0)

    with pytest.raises(RuntimeError, match="positive raw NAV"):
        allocation.assert_positive_allocated_nav()


# --- Paper capital (resolve_paper_nav) tests ---


def test_paper_nav_returns_raw_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(PAPER_CAPITAL_ENV, raising=False)
    state_file = tmp_path / "state.json"

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 30_000.0
    assert not state_file.exists()


def test_paper_nav_creates_baseline_on_first_call(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 10_000.0  # paper_capital + (30K - 30K baseline) = 10K
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["baseline_net_liq"] == 30_000.0
    assert data["paper_capital"] == 10_000.0


def test_paper_nav_tracks_pnl_on_subsequent_calls(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"

    # First call: establish baseline at 30K
    resolve_paper_nav(30_000.0, state_file)

    # Second call: account grew by $500 (from our trades)
    result = resolve_paper_nav(30_500.0, state_file)

    assert result == 10_500.0  # 10K + (30.5K - 30K) = 10.5K


def test_paper_nav_tracks_negative_pnl(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"

    resolve_paper_nav(30_000.0, state_file)
    result = resolve_paper_nav(29_200.0, state_file)

    assert result == 9_200.0  # 10K + (29.2K - 30K) = 9.2K


def test_paper_nav_ignored_in_live_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "live")
    state_file = tmp_path / "state.json"

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 30_000.0
    assert not state_file.exists()


def test_paper_nav_handles_corrupt_state_file(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"
    state_file.write_text("not valid json {{{")

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 10_000.0  # Resets baseline to current
    data = json.loads(state_file.read_text())
    assert data["baseline_net_liq"] == 30_000.0


def test_paper_nav_resets_on_nan_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"baseline_net_liq": float("nan"), "paper_capital": 10_000.0}))

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 10_000.0  # Resets baseline to 30K, offset = 0
    data = json.loads(state_file.read_text())
    assert data["baseline_net_liq"] == 30_000.0


def test_paper_nav_resets_baseline_when_capital_changes(monkeypatch, tmp_path):
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"

    # Initial setup with 10K capital
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    resolve_paper_nav(30_000.0, state_file)

    # Change to 15K capital — should reset baseline
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "15000")
    result = resolve_paper_nav(31_000.0, state_file)

    assert result == 15_000.0  # New baseline at 31K, offset = 0
    data = json.loads(state_file.read_text())
    assert data["baseline_net_liq"] == 31_000.0
    assert data["paper_capital"] == 15_000.0


def test_paper_nav_works_in_dev_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "10000")
    monkeypatch.setenv(ALGO_TRADER_ENV, "dev")
    state_file = tmp_path / "state.json"

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 10_000.0


def test_paper_nav_invalid_value_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv(PAPER_CAPITAL_ENV, "not_a_number")
    monkeypatch.setenv(ALGO_TRADER_ENV, "paper")
    state_file = tmp_path / "state.json"

    result = resolve_paper_nav(30_000.0, state_file)

    assert result == 30_000.0
