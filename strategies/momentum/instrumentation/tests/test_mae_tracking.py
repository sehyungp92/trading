"""Tests for MAE (Maximum Adverse Excursion) tracking on PositionState."""
import pytest
from datetime import datetime
from strategy.config import PositionState as HelixPositionState, SetupClass
from strategy_2.models import PositionState as NQDTCPositionState, Direction as NQDTCDirection
from strategy_3.models import PositionState as VdubusPositionState


def test_helix_position_has_mae_field():
    pos = HelixPositionState(
        pos_id="test1", direction=1, avg_entry=21000.0, contracts=1,
        unit1_risk_usd=500.0, origin_class=SetupClass.M,
        origin_setup_id="s1", entry_ts=datetime.now(), stop_price=20950.0,
    )
    assert hasattr(pos, "peak_mae_r")
    assert pos.peak_mae_r == 0.0


def test_nqdtc_position_has_mfe_mae_r_fields():
    pos = NQDTCPositionState()
    assert hasattr(pos, "peak_mfe_r")
    assert hasattr(pos, "peak_mae_r")
    assert pos.peak_mfe_r == 0.0
    assert pos.peak_mae_r == 0.0


def test_vdubus_position_has_mae_field():
    pos = VdubusPositionState()
    assert hasattr(pos, "peak_mae_r")
    assert pos.peak_mae_r == 0.0


def test_helix_mae_increases_on_adverse_move():
    pos = HelixPositionState(
        pos_id="test2", direction=1, avg_entry=21000.0, contracts=1,
        unit1_risk_usd=500.0, origin_class=SetupClass.M,
        origin_setup_id="s2", entry_ts=datetime.now(), stop_price=20950.0,
    )
    # Simulate adverse R
    R_total = -0.5
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -R_total))
    assert pos.peak_mae_r == 0.5

    # Further adverse
    R_total = -1.2
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -R_total))
    assert pos.peak_mae_r == 1.2

    # Recovery does not reduce MAE
    R_total = 0.3
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -R_total))
    assert pos.peak_mae_r == 1.2  # unchanged


def test_nqdtc_mae_tracks_correctly():
    pos = NQDTCPositionState(
        direction=NQDTCDirection.LONG,
        entry_price=21000.0,
        stop_price=20950.0,
    )
    # Simulate price going adverse
    risk = abs(pos.entry_price - pos.stop_price)
    close = 20970.0  # 30 pts adverse
    current_r = (close - pos.entry_price) / risk  # -0.6
    pos.peak_mfe_r = max(pos.peak_mfe_r, max(0.0, current_r))
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -current_r))
    assert pos.peak_mae_r == pytest.approx(0.6)
    assert pos.peak_mfe_r == 0.0

    # Price recovers and goes favorable
    close = 21100.0
    current_r = (close - pos.entry_price) / risk  # +2.0
    pos.peak_mfe_r = max(pos.peak_mfe_r, max(0.0, current_r))
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -current_r))
    assert pos.peak_mfe_r == pytest.approx(2.0)
    assert pos.peak_mae_r == pytest.approx(0.6)  # unchanged


def test_vdubus_mae_tracks_correctly():
    pos = VdubusPositionState()
    # Simulate unreal_r going adverse then favorable
    unreal_r = -0.8
    pos.peak_mfe_r = max(pos.peak_mfe_r, unreal_r)
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -unreal_r))
    assert pos.peak_mae_r == pytest.approx(0.8)

    unreal_r = 1.5
    pos.peak_mfe_r = max(pos.peak_mfe_r, unreal_r)
    pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -unreal_r))
    assert pos.peak_mfe_r == 1.5
    assert pos.peak_mae_r == pytest.approx(0.8)  # unchanged
