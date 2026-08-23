from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from strategies.stock.iaric.bar_policy import (
    Completed5mContractError,
    Completed5mGapError,
    apply_completed_5m_bar,
    completed_rth_5m_bars,
    validate_completed_5m_bar,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.engine import IARICEngine
from strategies.stock.iaric.models import Bar, MarketSnapshot, RegimeSnapshot, WatchlistArtifact


UTC = timezone.utc
ET = ZoneInfo("America/New_York")


def _bar(symbol: str, start: datetime, index: int = 0) -> Bar:
    open_ = 100.0 + index
    return Bar(
        symbol=symbol,
        start_time=start,
        end_time=start + timedelta(minutes=5),
        open=open_,
        high=open_ + 1.0,
        low=open_ - 0.5,
        close=open_ + 0.5,
        volume=1_000.0 + index * 100.0,
    )


def _item(
    symbol: str,
    *,
    daily_rank: float = 0.5,
    entry_rank_pct: float = 50.0,
    entry_rsi: float = 50.0,
):
    return SimpleNamespace(
        symbol=symbol,
        sector="Technology",
        daily_signal_score=70.0,
        trigger_types=["RSI2"],
        trigger_tier="STANDARD",
        trend_tier="STRONG",
        rescue_flow_candidate=False,
        sizing_mult=1.0,
        daily_atr_estimate=2.0,
        cdd_value=2,
        ema10_daily=99.0,
        rsi14_daily=45.0,
        tradable_flag=True,
        daily_rank=daily_rank,
        entry_rank=1,
        entry_rank_pct=entry_rank_pct,
        entry_rsi=entry_rsi,
        expected_5m_volume=1_000.0,
        average_30m_volume=6_000.0,
        tick_size=0.01,
    )


def _engine(*items, settings: StrategySettings | None = None) -> IARICEngine:
    trade_date = date(2026, 5, 20)
    artifact = WatchlistArtifact(
        trade_date=trade_date,
        generated_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        regime=RegimeSnapshot(
            score=0.8,
            tier="A",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
        ),
        items=list(items),
        tradable=list(items),
        overflow=[],
    )
    return IARICEngine(
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        artifact=artifact,
        account_id="TEST",
        nav=100_000.0,
        settings=settings or StrategySettings(completed_5m_batch_grace_s=999.0),
        disable_background_tasks=True,
    )


def test_golden_completed_5m_sequence_matches_backtest_session_transition() -> None:
    market = MarketSnapshot(symbol="AAA")
    state = SimpleNamespace(
        bars_seen_today=0,
        last_5m_bar_time=None,
        session_high=0.0,
        session_low=0.0,
    )
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    bars = [_bar("AAA", start + timedelta(minutes=5 * index), index) for index in range(6)]

    transitions = [apply_completed_5m_bar(market, bar, state=state) for bar in bars]

    expected_vwap = sum(bar.typical_price * bar.volume for bar in bars) / sum(bar.volume for bar in bars)
    assert market.session_vwap == pytest.approx(expected_vwap)
    assert market.session_high == max(bar.high for bar in bars)
    assert market.session_low == min(bar.low for bar in bars)
    assert state.bars_seen_today == 6
    assert state.last_5m_bar_time == bars[-1].end_time
    assert transitions[-1].completed_30m_bar is not None
    assert transitions[-1].completed_30m_bar.start_time == bars[0].start_time
    assert transitions[-1].completed_30m_bar.end_time == bars[-1].end_time
    assert transitions[-1].completed_30m_bar.volume == sum(bar.volume for bar in bars)


def test_one_minute_bar_cannot_reach_action_generating_ingress(monkeypatch) -> None:
    engine = _engine(_item("AAA"))
    evaluated = []
    monkeypatch.setattr(engine, "_process_intraday_bar", lambda *args: evaluated.append(args))
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    one_minute = Bar(
        symbol="AAA",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        open=100.0,
        high=100.5,
        low=99.5,
        close=100.1,
        volume=100.0,
    )

    with pytest.raises(Completed5mContractError, match="exactly one completed 5-minute bar"):
        engine.on_completed_5m_bar("AAA", one_minute, received_at=one_minute.end_time)

    assert evaluated == []
    assert engine.liveness_payload()["completed_5m"]["rejected"] == 1


def test_incomplete_duplicate_and_gap_contracts(monkeypatch) -> None:
    engine = _engine(_item("AAA"))
    monkeypatch.setattr(engine, "_process_intraday_bar", lambda *_args: None)
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    first = _bar("AAA", start)

    with pytest.raises(Completed5mContractError, match="incomplete"):
        engine.on_completed_5m_bar("AAA", first, received_at=first.end_time - timedelta(seconds=1))

    assert engine.on_completed_5m_bar("AAA", first, received_at=first.end_time)
    engine.flush_completed_5m_batch(first.end_time)
    assert not engine.on_completed_5m_bar("AAA", first, received_at=first.end_time)

    gap = _bar("AAA", first.end_time + timedelta(minutes=5), 2)
    with pytest.raises(Completed5mGapError):
        engine.on_completed_5m_bar("AAA", gap, received_at=gap.end_time)


def test_gap_is_rejected_even_before_the_prior_batch_is_flushed(monkeypatch) -> None:
    engine = _engine(_item("AAA"))
    monkeypatch.setattr(engine, "_process_intraday_bar", lambda *_args: None)
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    first = _bar("AAA", start)
    gap = _bar("AAA", start + timedelta(minutes=10), 2)

    engine.on_completed_5m_bar("AAA", first, received_at=first.end_time)
    with pytest.raises(Completed5mGapError):
        engine.on_completed_5m_bar("AAA", gap, received_at=gap.end_time)


@pytest.mark.parametrize("start_time", [time(9, 25), time(16, 0)])
def test_contract_rejects_bars_outside_session_boundaries(start_time: time) -> None:
    start = datetime.combine(date(2026, 5, 20), start_time, tzinfo=ET).astimezone(UTC)
    bar = _bar("AAA", start)

    with pytest.raises(Completed5mContractError, match="RTH session"):
        validate_completed_5m_bar(bar, received_at=bar.end_time, expected_symbol="AAA")


@pytest.mark.parametrize(
    "session_date",
    [date(2026, 1, 15), date(2026, 7, 15)],
)
def test_contract_alignment_is_dst_safe(session_date: date) -> None:
    start_et = datetime.combine(session_date, datetime.min.time(), tzinfo=ET).replace(hour=9, minute=30)
    bar = _bar("AAA", start_et.astimezone(UTC))

    validate_completed_5m_bar(bar, received_at=bar.end_time, expected_symbol="AAA")

    misaligned = _bar("AAA", (start_et + timedelta(minutes=1)).astimezone(UTC))
    with pytest.raises(Completed5mContractError, match="not aligned"):
        validate_completed_5m_bar(misaligned, received_at=misaligned.end_time, expected_symbol="AAA")


def test_replay_rth_filter_makes_bar_index_session_relative() -> None:
    session_date = date(2026, 7, 15)
    premarket = _bar("AAA", datetime(2026, 7, 15, 8, 0, tzinfo=ET).astimezone(UTC))
    session_open = datetime(2026, 7, 15, 9, 30, tzinfo=ET).astimezone(UTC)
    rth = [_bar("AAA", session_open + timedelta(minutes=5 * index), index) for index in range(3)]

    filtered = completed_rth_5m_bars([premarket, *rth])

    assert [bar.start_time for bar in filtered] == [bar.start_time for bar in rth]
    assert filtered[0].start_time.astimezone(ET).date() == session_date


def test_live_engine_refuses_non_parity_same_open_mode() -> None:
    with pytest.raises(ValueError, match="next_5m_open"):
        _engine(
            _item("AAA"),
            settings=StrategySettings(pb_open_scored_fill_timing="same_open"),
        )


def test_timestamp_batch_ranks_entries_before_capacity(monkeypatch) -> None:
    settings = StrategySettings(
        pb_max_positions=1,
        pb_intraday_priority_reserve_slots=0,
        completed_5m_batch_grace_s=999.0,
    )
    engine = _engine(
        _item("LOW", daily_rank=0.9, entry_rank_pct=80.0, entry_rsi=8.0),
        _item("HIGH", daily_rank=0.2, entry_rank_pct=20.0, entry_rsi=30.0),
        settings=settings,
    )
    selected: list[str] = []

    def evaluate(symbol: str, bar: Bar, now: datetime) -> None:
        state = engine._symbols[symbol]
        state.intraday_score = 55.0 if symbol == "LOW" else 85.0
        engine._fire_entry(symbol, bar, now, "OPEN_SCORED_ENTRY")

    monkeypatch.setattr(engine, "_process_intraday_bar", evaluate)
    monkeypatch.setattr(engine, "_dispatch_entry_candidate", lambda candidate: selected.append(candidate.symbol))
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    low_bar = _bar("LOW", start)
    high_bar = _bar("HIGH", start)

    engine.on_completed_5m_bar("LOW", low_bar, received_at=low_bar.end_time)
    engine.on_completed_5m_bar("HIGH", high_bar, received_at=high_bar.end_time)

    assert selected == ["HIGH"]
    assert engine._symbols["LOW"].invalid_reason == "slot_cap_reject"


def test_hydrated_restart_rebuilds_session_without_replaying_actions(monkeypatch) -> None:
    item = _item("AAA")
    original = _engine(item)
    monkeypatch.setattr(original, "_process_intraday_bar", lambda *_args: None)
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    bars = [_bar("AAA", start + timedelta(minutes=5 * index), index) for index in range(3)]
    for bar in bars[:2]:
        original.on_completed_5m_bar("AAA", bar, received_at=bar.end_time)
        original.flush_completed_5m_batch(bar.end_time)
    snapshot = original.snapshot_state()

    restarted = _engine(item)
    restarted.hydrate_state(snapshot)
    evaluated = []
    monkeypatch.setattr(restarted, "_process_intraday_bar", lambda *args: evaluated.append(args))
    for bar in bars[:2]:
        restarted.on_completed_5m_bar("AAA", bar, received_at=bar.end_time)
        restarted.flush_completed_5m_batch(bar.end_time)

    assert evaluated == []
    assert restarted._symbols["AAA"].bars_seen_today == 2
    assert len(restarted._markets["AAA"].bars_5m) == 2

    restarted.on_completed_5m_bar("AAA", bars[2], received_at=bars[2].end_time)
    restarted.flush_completed_5m_batch(bars[2].end_time)

    assert len(evaluated) == 1
    assert restarted._symbols["AAA"].bars_seen_today == 3


def test_reconnect_history_replay_is_idempotent(monkeypatch) -> None:
    engine = _engine(_item("AAA"))
    evaluated = []
    monkeypatch.setattr(engine, "_process_intraday_bar", lambda *args: evaluated.append(args))
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    bar = _bar("AAA", start)

    engine.on_completed_5m_bar("AAA", bar, received_at=bar.end_time)
    engine.flush_completed_5m_batch(bar.end_time)
    assert not engine.on_completed_5m_bar("AAA", bar, received_at=bar.end_time + timedelta(minutes=1))

    assert len(evaluated) == 1
    assert engine._symbols["AAA"].bars_seen_today == 1
    assert len(engine._markets["AAA"].bars_5m) == 1
    assert engine.liveness_payload()["completed_5m"]["duplicates"] == 1


def test_batch_telemetry_reports_missing_bar_rate(monkeypatch) -> None:
    engine = _engine(_item("AAA"), _item("BBB"))
    monkeypatch.setattr(engine, "_process_intraday_bar", lambda *_args: None)
    start = datetime(2026, 5, 20, 9, 30, tzinfo=ET).astimezone(UTC)
    bar = _bar("AAA", start)

    engine.on_completed_5m_bar("AAA", bar, received_at=bar.end_time + timedelta(seconds=2))
    engine.flush_completed_5m_batch(bar.end_time)

    telemetry = engine.liveness_payload()["completed_5m"]
    assert telemetry["arrival_latency_s"]["AAA"] == 2.0
    assert telemetry["expected"] == 2
    assert telemetry["missing"] == 1
    assert telemetry["missing_rate"] == 0.5
    assert telemetry["last_missing_symbols"] == ["BBB"]
