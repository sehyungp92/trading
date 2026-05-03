from __future__ import annotations

from types import SimpleNamespace

import pytest

from backtests.swing.auto.breakout.cli import _sanitize_breakout_baseline_mutations
from backtests.swing.auto.breakout.phase_candidates import get_phase_candidates
from backtests.swing.auto.breakout.scoring import score_phase_metrics
from strategies.swing.breakout import signals, stops
from strategies.swing.breakout.models import CampaignState, Direction, EntryType, Regime4H, SymbolCampaign


def test_entry_c_early_standard_signal_extracts_early_low_disp_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_ENABLE", True)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_MIN_SCORE", 0)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_MIN_QUALITY", 0.45)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_REQUIRE_ALIGNED", False)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_ALLOW_NEUTRAL", True)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_CLV_Q", 0.25)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_MAX_DISP_H", 2.75)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_MAX_BREAKOUT_BARS", 6)
    monkeypatch.setattr(signals, "ENTRY_C_EARLY_MIN_RVOL_H", 0.85)
    monkeypatch.setattr(signals, "ENTRY_C_HOLD_BARS", 2)

    campaign = SymbolCampaign(bars_since_breakout=2)
    ok = signals.entry_c_early_standard_signal(
        direction=Direction.LONG,
        campaign=campaign,
        hourly_close=101.9,
        hourly_low=100.8,
        hourly_high=102.0,
        hourly_closes=[101.1, 101.9],
        avwap_h=100.5,
        atr14_h=0.8,
        atr14_d=1.5,
        rvol_h=1.0,
        atr_expanding=True,
        ema20_h=100.9,
        score_total=1,
        quality_mult=0.60,
        regime_4h=Regime4H.RANGE_CHOP,
        continuation=False,
    )

    assert ok is True


@pytest.mark.parametrize(
    ("continuation", "hourly_close", "atr14_d", "expected_reason"),
    [
        (True, 101.2, 1.0, "continuation"),
        (False, 110.0, 1.0, "displacement"),
    ],
)
def test_assess_c_fresh_signal_returns_structural_reject_reasons(
    continuation: bool,
    hourly_close: float,
    atr14_d: float,
    expected_reason: str,
) -> None:
    ok, reason = signals._assess_c_fresh_signal(
        direction=Direction.LONG,
        campaign=SymbolCampaign(bars_since_breakout=1),
        hourly_close=hourly_close,
        hourly_low=100.1,
        hourly_high=max(hourly_close, 100.1) + 0.5,
        hourly_closes=[99.8, hourly_close],
        hourly_highs=[100.4, max(hourly_close, 100.1) + 0.5],
        hourly_lows=[99.6, 100.1],
        avwap_h=100.0,
        atr14_h=1.0,
        atr14_d=atr14_d,
        score_total=1,
        quality_mult=0.70,
        regime_4h=Regime4H.BULL_TREND,
        continuation=continuation,
        enabled=True,
        min_score=0,
        min_quality=0.45,
        require_aligned=False,
        allow_countertrend=False,
        clv_q=0.25,
        max_disp_h=2.75,
        touch_tol_atr_h=0.50,
        max_breakout_bars=8,
        min_rvol_h=0.0,
        rvol_h=1.0,
    )

    assert ok is False
    assert reason == expected_reason


def test_compute_initial_stop_uses_early_branch_anchor_for_fast_continuation_routes() -> None:
    early_stop = stops.compute_initial_stop(
        direction=Direction.LONG,
        entry_type="C_early_standard",
        box_high=110.0,
        box_low=100.0,
        box_mid=105.0,
        atr14_d=2.0,
        atr_stop_mult=0.5,
        sq_good=False,
        tick_size=0.01,
    )
    continuation_stop = stops.compute_initial_stop(
        direction=Direction.LONG,
        entry_type="C_continuation",
        box_high=110.0,
        box_low=100.0,
        box_mid=105.0,
        atr14_d=2.0,
        atr_stop_mult=0.5,
        sq_good=False,
        tick_size=0.01,
    )
    standard_stop = stops.compute_initial_stop(
        direction=Direction.LONG,
        entry_type="C_standard",
        box_high=110.0,
        box_low=100.0,
        box_mid=105.0,
        atr14_d=2.0,
        atr_stop_mult=0.5,
        sq_good=False,
        tick_size=0.01,
    )

    assert early_stop == 109.0
    assert continuation_stop == 109.0
    assert standard_stop == 104.0


def test_select_entry_type_routes_continuation_before_generic_c(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_ENABLE", True)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_MIN_SCORE", 0)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_MIN_QUALITY", 0.50)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_REQUIRE_ALIGNED", False)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_ALLOW_NEUTRAL", True)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_CLV_Q", 0.20)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_HOLD_BARS", 1)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_PAUSE_ATR_H", 1.25)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_MAX_DISP_H", 3.50)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_MAX_BREAKOUT_BARS", 999)
    monkeypatch.setattr(signals, "ENTRY_C_CONTINUATION_MIN_RVOL_H", 0.70)
    monkeypatch.setattr(signals, "ENTRY_C_STANDARD_ALLOW_CONTINUATION", False)
    monkeypatch.setattr(signals, "entry_b_resume_signal", lambda *args, **kwargs: False)
    monkeypatch.setattr(signals, "entry_a_reclaim_strong_signal", lambda *args, **kwargs: False)
    monkeypatch.setattr(signals, "entry_a_signal", lambda *args, **kwargs: False)
    monkeypatch.setattr(signals, "entry_b_signal", lambda *args, **kwargs: False)

    campaign = SymbolCampaign(
        state=CampaignState.BREAKOUT,
        continuation=True,
        bars_since_breakout=8,
    )
    trace: dict[str, object] = {}
    selected = signals.select_entry_type(
        direction=Direction.LONG,
        campaign=campaign,
        hourly_close=101.9,
        hourly_low=101.0,
        hourly_high=102.0,
        hourly_closes=[101.9],
        hourly_highs=[102.0],
        hourly_lows=[101.0],
        avwap_h=100.0,
        atr14_h=1.0,
        atr14_d=1.0,
        rvol_h=1.0,
        disp_mult=1.0,
        quality_mult=0.60,
        regime_4h=Regime4H.RANGE_CHOP,
        score_total=0,
        selection_trace=trace,
    )

    assert selected == EntryType.C_CONTINUATION
    assert trace["selected_entry_type"] == EntryType.C_CONTINUATION.value


def test_sanitize_breakout_baseline_migrates_legacy_fresh_stop_limit_flag() -> None:
    cleaned = _sanitize_breakout_baseline_mutations(
        {
            "param_overrides.ENTRY_C_FRESH_ENABLE": True,
            "param_overrides.ENTRY_C_FRESH_USE_STOP_LIMIT": True,
        }
    )

    assert "param_overrides.ENTRY_C_FRESH_USE_STOP_LIMIT" not in cleaned
    assert cleaned["param_overrides.ENTRY_C_FRESH_STOP_ENABLE"] is True


def test_round5_candidate_pool_replaces_cleanup_with_structural_paths() -> None:
    candidate_names = {name for name, _ in get_phase_candidates(1)}
    later_phase_names = {name for phase in (2, 3, 4) for name, _ in get_phase_candidates(phase)}

    assert "continuation_route_core" in candidate_names
    assert "momentum_exit_redesign" in candidate_names
    assert "fresh_market_restore_only" not in candidate_names
    assert "late_flow_risk_split" in later_phase_names


def test_round5_scoring_rejects_frequency_lift_without_return_velocity() -> None:
    old_round5_style = SimpleNamespace(
        total_trades=139,
        max_drawdown_pct=0.0544,
        profit_factor=6.56,
        net_profit=5905.0,
        expectancy_dollar=42.48,
        trades_per_month=2.259,
        avg_mfe_r=0.326,
        winner_capture_ratio=0.55,
    )
    redesigned_round5 = SimpleNamespace(
        total_trades=141,
        max_drawdown_pct=0.0543,
        profit_factor=6.50,
        net_profit=6047.52,
        expectancy_dollar=42.89,
        trades_per_month=2.291,
        avg_mfe_r=0.470,
        winner_capture_ratio=0.547,
    )

    replay_floors = {
        "min_trades": 140,
        "min_net_profit": 6000.0,
        "min_expectancy_dollar": 42.70,
        "min_tpm": 2.27,
        "min_edge_velocity": 98.0,
    }
    rejected = score_phase_metrics(1, old_round5_style, 10_000.0, hard_rejects=replay_floors)
    accepted = score_phase_metrics(1, redesigned_round5, 10_000.0, hard_rejects=replay_floors)

    assert rejected.rejected is True
    assert "too_few_trades" in rejected.reject_reason
    assert accepted.rejected is False
