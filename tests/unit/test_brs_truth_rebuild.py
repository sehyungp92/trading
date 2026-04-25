from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtests.swing.config_brs import BRSConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.brs_engine import BRSEngine
from backtests.swing.engine.brs_portfolio_engine import BRSSymbolData, run_brs_synchronized
from backtests.swing.analysis.brs_diagnostics import compute_brs_diagnostics
from backtests.swing.analysis.brs_trade_metrics import summarize_brs_campaigns
from strategies.swing.brs.models import BRSRegime, Direction, EntrySignal, EntryType, ExitReason


def _make_bars(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = "2024-01-02 15:00",
    freq: str = "1h",
) -> NumpyBars:
    closes_arr = np.array(closes, dtype=np.float64)
    opens_arr = np.array(opens if opens is not None else closes, dtype=np.float64)
    highs_arr = np.array(highs if highs is not None else [value + 0.5 for value in closes], dtype=np.float64)
    lows_arr = np.array(lows if lows is not None else [value - 0.5 for value in closes], dtype=np.float64)
    volumes_arr = np.full(len(closes), 1_000.0, dtype=np.float64)
    times_arr = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC").to_numpy()
    return NumpyBars(opens_arr, highs_arr, lows_arr, closes_arr, volumes_arr, times_arr)


def _make_daily_bars() -> NumpyBars:
    return _make_bars(
        [102.0, 101.0, 100.0],
        start="2023-12-31 00:00",
        freq="1d",
    )


def _make_four_hour_bars() -> NumpyBars:
    return _make_bars(
        [101.0, 100.0, 99.0],
        start="2024-01-02 12:00",
        freq="4h",
    )


def _maps(hourly_len: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(hourly_len, 1, dtype=np.int64), np.full(hourly_len, 1, dtype=np.int64)


def _config(*, symbols: list[str] | None = None, max_concurrent: int = 3) -> BRSConfig:
    return BRSConfig(
        symbols=symbols or ["QQQ"],
        data_dir=Path("."),
        initial_equity=10_000.0,
        warmup_daily=1,
        warmup_hourly=1,
        warmup_4h=1,
        disable_s1=True,
        disable_s2=True,
        disable_s3=True,
        disable_l1=True,
        disable_bd=True,
        max_concurrent=max_concurrent,
    )


def _short_signal(price: float) -> EntrySignal:
    return EntrySignal(
        entry_type=EntryType.LH_REJECTION,
        direction=Direction.SHORT,
        signal_price=price,
        signal_high=price + 0.5,
        signal_low=price - 0.5,
        stop_price=price + 1.0,
        risk_per_unit=1.0,
        bear_conviction=72.0,
        quality_score=1.0,
        regime_at_entry=BRSRegime.BEAR_TREND,
        vol_factor=1.0,
    )


def _patch_common(monkeypatch, *, signal_closes: set[float], qty_fn=None, exit_fn=None, scale_out_fn=None):
    from backtests.swing.engine import brs_engine as brs_engine_module

    monkeypatch.setattr(
        brs_engine_module,
        "check_lh_rejection",
        lambda _d, h, *_args, **_kwargs: _short_signal(h.close) if h.close in signal_closes else None,
    )
    monkeypatch.setattr(brs_engine_module, "check_bd_continuation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brs_engine_module, "check_s1", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brs_engine_module, "check_s2", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brs_engine_module, "check_s3", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brs_engine_module, "check_l1", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        brs_engine_module,
        "compute_position_size",
        qty_fn or (lambda *_args, **_kwargs: 10),
    )
    monkeypatch.setattr(
        brs_engine_module,
        "compute_initial_stop",
        lambda direction, entry_price, *_args, **_kwargs: entry_price + 1.0 if direction == Direction.SHORT else entry_price - 1.0,
    )
    monkeypatch.setattr(
        brs_engine_module,
        "update_trailing_stop",
        lambda *_args, current_stop, be_triggered, **_kwargs: (current_stop, be_triggered),
    )
    monkeypatch.setattr(
        brs_engine_module,
        "check_exits",
        exit_fn or (lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        brs_engine_module,
        "check_scale_out",
        scale_out_fn or (lambda *_args, **_kwargs: False),
    )


def test_brs_signal_cannot_fill_same_bar(monkeypatch):
    _patch_common(monkeypatch, signal_closes={99.0})

    hourly = _make_bars([100.0, 99.0, 98.0, 97.0])
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    engine = BRSEngine("QQQ", _config().get_symbol_config("QQQ"), _config())
    engine._prepare_run_context(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)

    engine._step_bar(0)
    engine._step_bar(1)
    assert engine._in_position is False
    assert engine._pending_entry is not None

    engine._step_bar(2)
    assert engine._in_position is True
    assert engine._pending_entry is None
    assert engine._pos_signal_bar_index == 1
    assert engine._pos_fill_bar_index == 2
    assert engine._pos_fill_time != engine._pos_signal_time


def test_brs_stop_fill_uses_gap_through_stop_and_slippage(monkeypatch):
    _patch_common(monkeypatch, signal_closes={99.0})
    cfg = _config()
    cfg.min_hold_bars = 0

    hourly = _make_bars(
        [100.0, 99.0, 96.0, 101.0],
        opens=[100.0, 99.0, 98.0, 101.0],
        highs=[100.5, 99.5, 98.5, 101.5],
        lows=[99.5, 98.5, 95.5, 100.5],
    )
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    engine = BRSEngine("QQQ", cfg.get_symbol_config("QQQ"), cfg)
    engine._prepare_run_context(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)
    for idx in range(len(hourly)):
        engine._step_bar(idx)

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade.exit_reason == ExitReason.STOP.value
    assert trade.exit_price > trade.initial_stop


def test_brs_min_hold_bars_delays_protective_stop_activation(monkeypatch):
    _patch_common(monkeypatch, signal_closes={99.0})

    cfg = _config()
    cfg.min_hold_bars = 3

    hourly = _make_bars(
        [100.0, 99.0, 98.0, 100.0, 101.0],
        opens=[100.0, 99.0, 98.0, 100.0, 101.0],
        highs=[100.5, 99.5, 98.5, 101.0, 102.0],
        lows=[99.5, 98.5, 97.5, 99.0, 100.0],
    )
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    engine = BRSEngine("QQQ", cfg.get_symbol_config("QQQ"), cfg)
    engine._prepare_run_context(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)

    engine._step_bar(0)
    engine._step_bar(1)
    engine._step_bar(2)
    assert engine._in_position is True
    assert engine._protective_stop_live is False

    engine._step_bar(3)
    assert engine._in_position is True
    assert engine._protective_stop_live is False
    assert engine.trades == []

    engine._step_bar(4)
    assert len(engine.trades) == 1
    assert engine.trades[0].exit_reason == ExitReason.STOP.value


def test_brs_discretionary_exit_submits_next_bar_market_order(monkeypatch):
    _patch_common(
        monkeypatch,
        signal_closes={99.0},
        exit_fn=lambda *_args, bars_held, **_kwargs: ExitReason.STALE if bars_held >= 1 else None,
    )

    hourly = _make_bars([100.0, 99.0, 97.0, 96.0])
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    engine = BRSEngine("QQQ", _config().get_symbol_config("QQQ"), _config())
    engine._prepare_run_context(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)

    engine._step_bar(0)
    engine._step_bar(1)
    engine._step_bar(2)
    assert engine._in_position is True
    assert engine._pending_exit_order_id is not None
    assert engine.trades == []

    engine._step_bar(3)
    assert len(engine.trades) == 1
    assert engine.trades[0].exit_reason == ExitReason.STALE.value
    assert engine.trades[0].exit_time == engine._to_datetime(hourly.times[3])


def test_brs_run_realizes_open_position_at_end_of_data(monkeypatch):
    _patch_common(monkeypatch, signal_closes={99.0})

    hourly = _make_bars([100.0, 99.0, 98.0])
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    cfg = _config()
    engine = BRSEngine("QQQ", cfg.get_symbol_config("QQQ"), cfg)
    result = engine.run(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)

    assert engine._in_position is False
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "END_OF_DATA"
    assert result.trades[0].exit_time == engine._to_datetime(hourly.times[-1])
    assert result.equity_curve[-1] == pytest.approx(engine.equity)


def test_brs_campaign_count_and_commission_allocation(monkeypatch):
    _patch_common(
        monkeypatch,
        signal_closes={99.0, 96.0},
        exit_fn=lambda *_args, bars_held, **_kwargs: ExitReason.STALE if bars_held >= 3 else None,
        scale_out_fn=lambda cur_r, _target_r, tranche_b_open: tranche_b_open and cur_r >= 1.0,
    )

    cfg = _config()
    cfg.scale_out_enabled = True
    cfg.scale_out_pct = 0.33
    cfg.pyramid_enabled = True
    cfg.pyramid_scale = 0.5
    cfg.pyramid_min_r = 1.0

    hourly = _make_bars(
        [100.0, 99.0, 96.0, 92.0, 90.0, 89.0],
        opens=[100.0, 99.0, 98.0, 95.0, 91.0, 89.0],
        highs=[100.5, 99.5, 98.5, 95.5, 91.5, 89.5],
        lows=[99.5, 98.5, 95.5, 91.5, 89.5, 88.5],
    )
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    engine = BRSEngine("QQQ", cfg.get_symbol_config("QQQ"), cfg)
    engine._prepare_run_context(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)
    for idx in range(len(hourly)):
        engine._step_bar(idx)

    assert len(engine.trades) == 2
    campaigns = summarize_brs_campaigns(engine.trades)
    assert len(campaigns) == 1
    assert campaigns[0].realized_leg_count == 2
    assert campaigns[0].terminal_exit_reason == ExitReason.STALE.value
    assert engine.trades[0].campaign_id == engine.trades[1].campaign_id
    assert sum(trade.commission for trade in engine.trades) == pytest.approx(engine.total_commission)


def test_brs_synchronized_blocks_same_bar_entries_when_max_concurrent_is_hit(monkeypatch):
    _patch_common(
        monkeypatch,
        signal_closes={99.0},
        qty_fn=lambda _signal, _equity, _sym_cfg, _daily_ctx, cfg, _heat, concurrent_positions: 10 if concurrent_positions < cfg.max_concurrent else 0,
    )

    cfg = _config(symbols=["QQQ", "GLD"], max_concurrent=1)
    hourly = _make_bars(
        [100.0, 99.0, 96.0, 101.0],
        opens=[100.0, 99.0, 98.0, 101.0],
        highs=[100.5, 99.5, 98.5, 101.5],
        lows=[99.5, 98.5, 95.5, 100.5],
    )
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    data = {
        "QQQ": BRSSymbolData(daily=daily, hourly=hourly, four_hour=four_hour, daily_idx_map=daily_idx_map, four_hour_idx_map=four_hour_idx_map),
        "GLD": BRSSymbolData(daily=daily, hourly=hourly, four_hour=four_hour, daily_idx_map=daily_idx_map, four_hour_idx_map=four_hour_idx_map),
    }

    result = run_brs_synchronized(data, cfg)
    assert len(result.symbol_results["QQQ"].trades) == 1
    assert len(result.symbol_results["GLD"].trades) == 0


def test_brs_diagnostics_report_includes_new_strength_weakness_sections(monkeypatch):
    _patch_common(
        monkeypatch,
        signal_closes={99.0},
        exit_fn=lambda *_args, bars_held, **_kwargs: ExitReason.STALE if bars_held >= 1 else None,
    )

    cfg = _config(symbols=["QQQ"])
    hourly = _make_bars([100.0, 99.0, 97.0, 96.0])
    daily = _make_daily_bars()
    four_hour = _make_four_hour_bars()
    daily_idx_map, four_hour_idx_map = _maps(len(hourly))

    data = {
        "QQQ": BRSSymbolData(
            daily=daily,
            hourly=hourly,
            four_hour=four_hour,
            daily_idx_map=daily_idx_map,
            four_hour_idx_map=four_hour_idx_map,
        ),
    }

    result = run_brs_synchronized(data, cfg)
    diagnostics = compute_brs_diagnostics(
        result.symbol_results,
        cfg.initial_equity,
        combined_equity=result.combined_equity,
        combined_timestamps=result.combined_timestamps,
    )

    assert "Terminal Exit Reason Metrics" in diagnostics.report
    assert "Downturn Alignment Audit" in diagnostics.report
    assert "Robustness / Concentration" in diagnostics.report
