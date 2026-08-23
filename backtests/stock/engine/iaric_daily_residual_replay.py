"""Deterministic shared-core replay for IARIC daily residual reversion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from bisect import bisect_left
from datetime import date, datetime, time, timezone, timedelta
from math import isfinite
import os
from pathlib import Path
import pickle
from statistics import fmean
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.core.actions import (
    SubmitEntry,
    SubmitMarketExit,
    SubmitPartialExit,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core.daily_residual import (
    DAILY_RESIDUAL_SLEEVE,
    DailyResidualExecutionPosition,
    DailyResidualFill,
    apply_daily_residual_fill,
    build_daily_residual_execution_state,
    plan_daily_residual_forced_exit,
    plan_daily_residual_session_orders,
)
from strategies.stock.iaric.daily_residual_selection import (
    PreparedDailyResidualSelection,
    SECTOR_REFERENCE,
    _residual_contracts,
    _returns,
    build_daily_residual_artifact,
    prepare_daily_residual_selection,
)
from strategies.stock.iaric.core.residual import FrozenResidualModel
from strategies.stock.iaric.models import (
    HeldPositionResearch,
    MarketResearch,
    RegimeSnapshot,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
)
from strategies.stock.iaric.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


ET = ZoneInfo("America/New_York")
PREPARED_SELECTION_CACHE_CONTRACT = "exact_daily_residual_prepared_selection_v1"


@dataclass(slots=True)
class DailyResidualReplayBundle:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    sectors: dict[str, str]
    primary_exchanges: dict[str, str]
    residuals: dict[str, dict[object, float]]
    residual_models: dict[str, dict[object, FrozenResidualModel]]
    factor_model: str
    source_fingerprint: str
    bars_by_symbol: dict[str, list[ResearchDailyBar]]
    bar_dates_by_symbol: dict[str, list[date]]
    stamp_by_date: dict[date, Any]
    prior_date_by_date: dict[date, date | None]
    frozen_history_cache: dict[tuple[object, ...], dict[object, float]]
    snapshot_cache: dict[date, ResearchSnapshot]
    prepared_selection_cache: dict[
        tuple[date, int], PreparedDailyResidualSelection
    ]
    prepared_selection_cache_dir: Path | None
    prepared_selection_disk_hits: int
    prepared_selection_disk_misses: int
    stock_returns: dict[str, dict[object, float]]
    reference_returns: dict[str, dict[object, float]]


@dataclass(slots=True)
class DailyResidualReplayTrade:
    symbol: str
    sector: str
    entry_date: date
    entry_time: datetime
    entry_price: float
    qty_entry: int
    initial_risk_dollars: float
    factor_model: str
    formation_sessions: int
    score: float
    residual_lane_id: str = ""
    residual_model_contract_version: str = ""
    failed_continuation_r: float = 0.0
    sector_return_5d: float = 0.0
    signal_close_price: float = 0.0
    exit_date: date | None = None
    exit_time: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    held_sessions: int = 0

    @property
    def overnight_return(self) -> float:
        if self.signal_close_price <= 0.0:
            return 0.0
        return self.entry_price / self.signal_close_price - 1.0

    @property
    def open_to_exit_return(self) -> float:
        if self.entry_price <= 0.0 or self.exit_price <= 0.0:
            return 0.0
        return self.exit_price / self.entry_price - 1.0

    @property
    def signal_close_to_exit_return(self) -> float:
        if self.signal_close_price <= 0.0 or self.exit_price <= 0.0:
            return 0.0
        return self.exit_price / self.signal_close_price - 1.0


@dataclass(slots=True)
class DailyResidualReplayResult:
    initial_equity: float
    final_equity: float
    trades: list[DailyResidualReplayTrade]
    equity_curve: list[dict[str, Any]]
    decision_events: list[dict[str, Any]]
    source_fingerprint: str
    factor_model: str
    entry_clock: str = "next_session_open"
    shared_core_contract: str = "iaric_daily_residual_execution_v2"

    def metrics(self) -> dict[str, Any]:
        returns = [trade.r_multiple for trade in self.trades]
        wins = [value for value in returns if value > 0.0]
        losses = [value for value in returns if value < 0.0]
        peak = self.initial_equity
        max_drawdown = 0.0
        for row in self.equity_curve:
            equity = float(row["mtm_equity"])
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / max(peak, 1e-9))
        return {
            "trades": len(self.trades),
            "total_r": sum(returns),
            "average_r": fmean(returns) if returns else 0.0,
            "profit_factor": (
                sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0.0)
            ),
            "win_rate": len(wins) / len(returns) if returns else 0.0,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "return_pct": self.final_equity / self.initial_equity - 1.0,
            "max_drawdown_pct": max_drawdown,
        }


def _bars_from_panel(
    symbol: str,
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    before: date | None = None,
    tail: int | None = None,
) -> list[ResearchDailyBar]:
    rows: list[ResearchDailyBar] = []
    for stamp in close.index:
        session = pd.Timestamp(stamp).date()
        if before is not None and session >= before:
            continue
        values = (
            open_.at[stamp, symbol],
            high.at[stamp, symbol],
            low.at[stamp, symbol],
            close.at[stamp, symbol],
            volume.at[stamp, symbol],
        )
        if not all(pd.notna(value) and isfinite(float(value)) for value in values):
            continue
        rows.append(
            ResearchDailyBar(
                trade_date=session,
                open=float(values[0]),
                high=float(values[1]),
                low=float(values[2]),
                close=float(values[3]),
                volume=float(values[4]),
            )
        )
    return rows[-tail:] if tail else rows


def _symbol(
    symbol: str,
    sector: str,
    primary_exchange: str,
    bars: list[ResearchDailyBar],
) -> ResearchSymbol:
    prior = bars[-20:]
    adv = (
        fmean(bar.close * bar.volume * 100.0 for bar in prior) if prior else 0.0
    )
    return ResearchSymbol(
        symbol=symbol,
        exchange="SMART",
        primary_exchange=primary_exchange,
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector=sector,
        price=float(bars[-1].close) if bars else 0.0,
        adv20_usd=adv,
        median_spread_pct=0.0,
        earnings_within_sessions=None,
        blacklist_flag=False,
        halted_flag=False,
        severe_news_flag=False,
        daily_bars=bars,
    )


def _neutral_regime() -> RegimeSnapshot:
    return RegimeSnapshot(
        score=0.5,
        tier="B",
        risk_multiplier=1.0,
        price_ok=True,
        breadth_ok=True,
        vol_ok=True,
        credit_ok=True,
    )


def build_daily_residual_replay_bundle(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    sectors: Mapping[str, str],
    *,
    factor_model: str,
    source_fingerprint: str,
) -> DailyResidualReplayBundle:
    """Precompute causal residuals once; no optimization-phase refits."""

    stocks = set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    if set(sectors) != stocks:
        raise ValueError("replay bundle must use exactly the frozen 98 stocks")
    references = {"SPY", *SECTOR_REFERENCE.values()}
    if not references <= set(close):
        raise ValueError("replay bundle is missing market/sector references")
    metadata = {
        symbol: (sector, primary)
        for symbol, sector, primary in SP500_CONSTITUENTS
    }
    symbols = {
        symbol: _symbol(
            symbol,
            sectors[symbol],
            metadata.get(symbol, (sectors[symbol], "SMART"))[1],
            _bars_from_panel(
                symbol,
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            ),
        )
        for symbol in sorted(stocks)
    }
    reference_bars = {
        symbol: _bars_from_panel(
            symbol,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        for symbol in sorted(references)
    }
    final_date = max(pd.Timestamp(value).date() for value in close.index) + timedelta(days=1)
    snapshot = ResearchSnapshot(
        trade_date=final_date,
        market=MarketResearch(True, 50.0, 50.0, 0.0),
        sectors={},
        symbols=symbols,
        reference_daily_bars=reference_bars,
    )
    residuals, residual_models, _peer_memberships = _residual_contracts(
        snapshot, factor_model
    )
    all_bars = {
        **{symbol: item.daily_bars for symbol, item in symbols.items()},
        **reference_bars,
    }
    ordered_dates = sorted({pd.Timestamp(stamp).date() for stamp in close.index})
    return DailyResidualReplayBundle(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        sectors=dict(sectors),
        primary_exchanges={
            symbol: metadata.get(symbol, (sectors[symbol], "SMART"))[1]
            for symbol in stocks
        },
        residuals=residuals,
        residual_models=residual_models,
        factor_model=factor_model,
        source_fingerprint=source_fingerprint,
        bars_by_symbol=all_bars,
        bar_dates_by_symbol={
            symbol: [bar.trade_date for bar in bars]
            for symbol, bars in all_bars.items()
        },
        stamp_by_date={pd.Timestamp(stamp).date(): stamp for stamp in close.index},
        prior_date_by_date={
            session: (ordered_dates[index - 1] if index > 0 else None)
            for index, session in enumerate(ordered_dates)
        },
        frozen_history_cache={},
        snapshot_cache={},
        prepared_selection_cache={},
        prepared_selection_cache_dir=None,
        prepared_selection_disk_hits=0,
        prepared_selection_disk_misses=0,
        stock_returns={
            symbol: _returns(symbols[symbol].daily_bars)
            for symbol in sorted(stocks)
        },
        reference_returns={
            symbol: _returns(reference_bars[symbol])
            for symbol in sorted(references)
        },
    )


def _held_research(state) -> list[HeldPositionResearch]:
    held: list[HeldPositionResearch] = []
    for symbol_state in state.symbols.values():
        position: DailyResidualExecutionPosition | None = symbol_state.position
        if position is None or position.qty_open <= 0:
            continue
        held.append(
            HeldPositionResearch(
                symbol=symbol_state.symbol,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                size=position.qty_open,
                stop=max(
                    position.entry_price - position.catastrophic_stop_distance,
                    symbol_state.tick_size,
                ),
                initial_r=position.initial_risk_per_share,
                setup_tag=DAILY_RESIDUAL_SLEEVE,
                sleeve_id=DAILY_RESIDUAL_SLEEVE,
                issuer=position.issuer,
                sector=position.sector,
                residual_factor_model=position.residual_factor_model,
                residual_formation_sessions=position.residual_formation_sessions,
                residual_volatility=position.residual_volatility,
                residual_lane_id=position.residual_lane_id,
                residual_model_contract_version=(
                    position.residual_model_contract_version
                ),
                residual_model_intercept=position.residual_model_intercept,
                residual_factor_names=position.residual_factor_names,
                residual_factor_betas=position.residual_factor_betas,
                residual_peer_symbols=position.residual_peer_symbols,
                residual_model_estimation_session=(
                    position.residual_model_estimation_session
                ),
                residual_initial_dislocation_r=(
                    position.management.initial_dislocation_r
                ),
                residual_cumulative_normalization_r=(
                    position.management.cumulative_normalization_r
                ),
                residual_peak_normalization_r=(
                    position.management.peak_normalization_r
                ),
                residual_held_sessions=position.management.held_sessions,
                residual_partial_taken=position.management.partial_taken,
                residual_last_processed_session=position.last_processed_session,
                residual_qty_entry=position.qty_entry,
                residual_entry_commission=position.entry_commission,
                residual_exit_commission=position.exit_commission,
                residual_realized_pnl_usd=position.realized_pnl_usd,
                residual_entry_score=position.entry_score,
                residual_trade_id=position.trade_id,
                residual_protective_stop_client_order_id=(
                    symbol_state.protective_stop_client_order_id
                ),
                residual_protective_stop_price=symbol_state.protective_stop_price,
                residual_protective_stop_qty=symbol_state.protective_stop_qty,
            )
        )
    return held


def _snapshot_for_session(
    bundle: DailyResidualReplayBundle,
    session: date,
    held_positions: list[HeldPositionResearch],
) -> ResearchSnapshot:
    cached = bundle.snapshot_cache.get(session)
    if cached is not None:
        return replace(cached, held_positions=held_positions)
    symbols = {}
    for symbol in sorted(bundle.sectors):
        position = bisect_left(bundle.bar_dates_by_symbol[symbol], session)
        bars = bundle.bars_by_symbol[symbol][max(0, position - 260) : position]
        symbols[symbol] = _symbol(
            symbol,
            bundle.sectors[symbol],
            bundle.primary_exchanges[symbol],
            bars,
        )
    references = {"SPY", *SECTOR_REFERENCE.values()}
    reference_bars = {
        symbol: bundle.bars_by_symbol[symbol][
            max(0, bisect_left(bundle.bar_dates_by_symbol[symbol], session) - 260) :
            bisect_left(bundle.bar_dates_by_symbol[symbol], session)
        ]
        for symbol in references
    }
    snapshot = ResearchSnapshot(
        trade_date=session,
        market=MarketResearch(True, 50.0, 50.0, 0.0),
        sectors={},
        symbols=symbols,
        held_positions=held_positions,
        reference_daily_bars=reference_bars,
    )
    bundle.snapshot_cache[session] = snapshot
    return replace(snapshot, held_positions=held_positions)


def _at(
    bundle: DailyResidualReplayBundle,
    frame: pd.DataFrame,
    session: date,
    symbol: str,
) -> float | None:
    stamp = bundle.stamp_by_date.get(session)
    if stamp is None:
        return None
    value = frame.at[stamp, symbol]
    return float(value) if pd.notna(value) and isfinite(float(value)) else None


def _prior_at(
    bundle: DailyResidualReplayBundle,
    frame: pd.DataFrame,
    session: date,
    symbol: str,
) -> float | None:
    prior = bundle.prior_date_by_date.get(session)
    if prior is None:
        return None
    return _at(bundle, frame, prior, symbol)


def _open_time(session: date) -> datetime:
    return datetime.combine(session, time(9, 30), tzinfo=ET).astimezone(timezone.utc)


def _prepared_selection_disk_path(
    bundle: DailyResidualReplayBundle,
    session: date,
    formation_sessions: int,
) -> Path | None:
    root = bundle.prepared_selection_cache_dir
    if root is None:
        return None
    fingerprint = bundle.source_fingerprint[:24]
    return (
        root
        / PREPARED_SELECTION_CACHE_CONTRACT
        / bundle.factor_model
        / str(int(formation_sessions))
        / f"{session.isoformat()}__{fingerprint}.pickle"
    )


def _load_prepared_selection(
    path: Path | None,
    bundle: DailyResidualReplayBundle,
    session: date,
    formation_sessions: int,
) -> PreparedDailyResidualSelection | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if not isinstance(payload, dict):
        return None
    prepared = payload.get("prepared")
    if (
        payload.get("contract") != PREPARED_SELECTION_CACHE_CONTRACT
        or payload.get("source_fingerprint") != bundle.source_fingerprint
        or not isinstance(prepared, PreparedDailyResidualSelection)
        or prepared.trade_date != session
        or prepared.factor_model != bundle.factor_model
        or prepared.formation_sessions != int(formation_sessions)
    ):
        return None
    return prepared


def _store_prepared_selection(
    path: Path | None,
    bundle: DailyResidualReplayBundle,
    prepared: PreparedDailyResidualSelection,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.{os.getpid()}.tmp")
    payload = {
        "contract": PREPARED_SELECTION_CACHE_CONTRACT,
        "source_fingerprint": bundle.source_fingerprint,
        "prepared": prepared,
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_daily_residual_replay(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    start: date,
    end: date,
    initial_equity: float = 100_000.0,
    round_trip_cost_bps: float = 20.0,
) -> DailyResidualReplayResult:
    """Replay selection, neutral actions, fills and MTM on shared capital."""

    if settings.strategy_mode != DAILY_RESIDUAL_SLEEVE:
        settings = replace(settings, strategy_mode=DAILY_RESIDUAL_SLEEVE)
    if settings.daily_residual_factor_model != bundle.factor_model:
        raise ValueError("settings factor model does not match replay bundle")
    per_side_cost = float(round_trip_cost_bps) / 2.0 / 10_000.0
    # Half is adverse price movement, half explicit commission.
    slippage = per_side_cost / 2.0
    commission_rate = per_side_cost / 2.0
    cash = float(initial_equity)
    previous_state = None
    open_trades: dict[str, DailyResidualReplayTrade] = {}
    completed: list[DailyResidualReplayTrade] = []
    equity_curve: list[dict[str, Any]] = []
    decision_events: list[dict[str, Any]] = []
    sessions = sorted(
        {
            pd.Timestamp(stamp).date()
            for stamp in bundle.close.index
            if start <= pd.Timestamp(stamp).date() <= end
        }
    )
    for session in sessions:
        held = _held_research(previous_state) if previous_state is not None else []
        snapshot = _snapshot_for_session(bundle, session, held)
        preparation_key = (
            session,
            int(settings.daily_residual_formation_sessions),
        )
        prepared_selection = bundle.prepared_selection_cache.get(preparation_key)
        if prepared_selection is None:
            disk_path = _prepared_selection_disk_path(
                bundle,
                session,
                int(settings.daily_residual_formation_sessions),
            )
            prepared_selection = _load_prepared_selection(
                disk_path,
                bundle,
                session,
                int(settings.daily_residual_formation_sessions),
            )
            if prepared_selection is None:
                bundle.prepared_selection_disk_misses += 1
                prepared_selection = prepare_daily_residual_selection(
                    snapshot,
                    factor_model=bundle.factor_model,
                    formation_sessions=int(
                        settings.daily_residual_formation_sessions
                    ),
                    precomputed_residuals=bundle.residuals,
                    precomputed_models=bundle.residual_models,
                    frozen_history_cache=bundle.frozen_history_cache,
                    precomputed_stock_returns=bundle.stock_returns,
                    precomputed_reference_returns=bundle.reference_returns,
                )
                _store_prepared_selection(disk_path, bundle, prepared_selection)
            else:
                bundle.prepared_selection_disk_hits += 1
            bundle.prepared_selection_cache[preparation_key] = prepared_selection
        artifact = build_daily_residual_artifact(
            snapshot,
            settings,
            _neutral_regime(),
            precomputed_residuals=bundle.residuals,
            precomputed_models=bundle.residual_models,
            frozen_history_cache=bundle.frozen_history_cache,
            precomputed_stock_returns=bundle.stock_returns,
            precomputed_reference_returns=bundle.reference_returns,
            prepared_selection=prepared_selection,
        )
        prior_mtm = cash
        if previous_state is not None:
            for symbol_state in previous_state.symbols.values():
                position = symbol_state.position
                if position is None or position.qty_open <= 0:
                    continue
                prior_close = _prior_at(
                    bundle, bundle.close, session, symbol_state.symbol
                )
                if prior_close is None:
                    prior_close = position.entry_price
                prior_mtm += position.qty_open * prior_close
        state = build_daily_residual_execution_state(
            artifact,
            nav=prior_mtm,
            catastrophic_stop_atr=settings.daily_residual_catastrophic_stop_atr,
            catastrophic_stop_residual_r=(
                settings.daily_residual_catastrophic_stop_residual_r
            ),
        )
        state, actions, events = plan_daily_residual_session_orders(
            state,
            ts=_open_time(session) - timedelta(minutes=5),
            allow_entries=True,
        )
        decision_events.extend(asdict(event) for event in events)
        order_actions = [
            action
            for action in actions
            if isinstance(action, (SubmitEntry, SubmitPartialExit, SubmitMarketExit))
        ]
        # Exits execute before new entries at the same open, releasing capital.
        order_actions.sort(key=lambda action: isinstance(action, SubmitEntry))
        for action in order_actions:
            raw_open = _at(bundle, bundle.open, session, action.symbol)
            if raw_open is None:
                continue
            is_entry = isinstance(action, SubmitEntry)
            fill_price = raw_open * (1.0 + slippage if is_entry else 1.0 - slippage)
            commission = fill_price * action.qty * commission_rate
            role = (
                "ENTRY"
                if is_entry
                else ("PARTIAL_EXIT" if isinstance(action, SubmitPartialExit) else "EXIT")
            )
            pre_position = state.symbols[action.symbol].position
            pre_qty = pre_position.qty_open if pre_position else 0
            state, _followups, fill_events = apply_daily_residual_fill(
                state,
                DailyResidualFill(
                    client_order_id=action.client_order_id,
                    symbol=action.symbol,
                    role=role,
                    qty=action.qty,
                    price=fill_price,
                    ts=_open_time(session),
                    commission=commission,
                ),
            )
            decision_events.extend(asdict(event) for event in fill_events)
            if is_entry:
                cash -= fill_price * action.qty + commission
                item = artifact.by_symbol[action.symbol]
                position = state.symbols[action.symbol].position
                if position is None:
                    raise RuntimeError("shared core failed to create replay position")
                open_trades[action.symbol] = DailyResidualReplayTrade(
                    symbol=action.symbol,
                    sector=item.sector,
                    entry_date=session,
                    entry_time=_open_time(session),
                    entry_price=fill_price,
                    qty_entry=action.qty,
                    initial_risk_dollars=(
                        action.qty * position.initial_risk_per_share
                    ),
                    factor_model=item.residual_factor_model,
                    formation_sessions=item.residual_formation_sessions,
                    score=item.daily_signal_score,
                    residual_lane_id=item.residual_lane_id,
                    residual_model_contract_version=(
                        item.residual_model_contract_version
                    ),
                    failed_continuation_r=(
                        item.residual_failed_continuation_r
                    ),
                    sector_return_5d=item.residual_sector_return_5d,
                    signal_close_price=float(item.previous_close),
                    commission=commission,
                )
            else:
                cash += fill_price * action.qty - commission
                trade = open_trades[action.symbol]
                gross = (fill_price - trade.entry_price) * action.qty
                trade.gross_pnl += gross
                trade.commission += commission
                position = state.symbols[action.symbol].position
                if position is not None and position.qty_open == 0:
                    trade.exit_date = session
                    trade.exit_time = _open_time(session)
                    trade.exit_price = fill_price
                    trade.exit_reason = state.symbols[action.symbol].pending_management_reason
                    trade.net_pnl = trade.gross_pnl - trade.commission
                    trade.r_multiple = trade.net_pnl / max(
                        trade.initial_risk_dollars, 1e-9
                    )
                    trade.held_sessions = position.management.held_sessions
                    completed.append(trade)
                    del open_trades[action.symbol]

        # Catastrophic stop-first sequencing uses the day's open/low after all
        # pre-open decisions.  Gap-through fills at the adverse opening price.
        for symbol in sorted(list(open_trades)):
            symbol_state = state.symbols.get(symbol)
            position = symbol_state.position if symbol_state else None
            if position is None or position.qty_open <= 0:
                continue
            day_low = _at(bundle, bundle.low, session, symbol)
            day_open = _at(bundle, bundle.open, session, symbol)
            stop_price = symbol_state.protective_stop_price
            if day_low is None or day_open is None or stop_price <= 0.0 or day_low > stop_price:
                continue
            raw_stop_fill = day_open if day_open < stop_price else stop_price
            fill_price = raw_stop_fill * (1.0 - slippage)
            qty = position.qty_open
            commission = fill_price * qty * commission_rate
            state, _followups, fill_events = apply_daily_residual_fill(
                state,
                DailyResidualFill(
                    client_order_id=symbol_state.protective_stop_client_order_id,
                    symbol=symbol,
                    role="STOP",
                    qty=qty,
                    price=fill_price,
                    ts=_open_time(session),
                    commission=commission,
                ),
            )
            decision_events.extend(asdict(event) for event in fill_events)
            cash += fill_price * qty - commission
            trade = open_trades.pop(symbol)
            trade.gross_pnl += (fill_price - trade.entry_price) * qty
            trade.commission += commission
            trade.exit_date = session
            trade.exit_time = _open_time(session)
            trade.exit_price = fill_price
            trade.exit_reason = "catastrophic_stop"
            trade.net_pnl = trade.gross_pnl - trade.commission
            trade.r_multiple = trade.net_pnl / max(trade.initial_risk_dollars, 1e-9)
            trade.held_sessions = position.management.held_sessions
            completed.append(trade)

        mtm = cash
        for symbol_state in state.symbols.values():
            position = symbol_state.position
            if position is None or position.qty_open <= 0:
                continue
            close_price = _at(bundle, bundle.close, session, symbol_state.symbol)
            if close_price is None:
                close_price = position.entry_price
            mtm += position.qty_open * close_price
        equity_curve.append(
            {
                "date": session.isoformat(),
                "cash": cash,
                "mtm_equity": mtm,
                "open_positions": len(open_trades),
            }
        )
        previous_state = state

    # Final marked liquidation is explicit and costed; it is never credited at
    # a future bar beyond the requested fold.  Even this operational boundary
    # event must pass through the same shared-core action/fill reducer.
    if sessions and previous_state is not None:
        final_session = sessions[-1]
        close_ts = datetime.combine(final_session, time(16, 0), tzinfo=ET).astimezone(timezone.utc)
        for symbol in sorted(list(open_trades)):
            symbol_state = previous_state.symbols[symbol]
            position = symbol_state.position
            if position is None or position.qty_open <= 0:
                continue
            raw_close = _at(bundle, bundle.close, final_session, symbol)
            if raw_close is None:
                continue
            fill_price = raw_close * (1.0 - slippage)
            qty = position.qty_open
            commission = fill_price * qty * commission_rate
            previous_state, _action, forced_event = plan_daily_residual_forced_exit(
                previous_state,
                symbol=symbol,
                ts=close_ts,
                reason="fold_end_marked_liquidation",
            )
            decision_events.append(asdict(forced_event))
            previous_state, _followups, fill_events = apply_daily_residual_fill(
                previous_state,
                DailyResidualFill(
                    client_order_id=_action.client_order_id,
                    symbol=symbol,
                    role="EXIT",
                    qty=qty,
                    price=fill_price,
                    ts=close_ts,
                    commission=commission,
                ),
            )
            decision_events.extend(asdict(event) for event in fill_events)
            cash += fill_price * qty - commission
            trade = open_trades.pop(symbol)
            trade.gross_pnl += (fill_price - trade.entry_price) * qty
            trade.commission += commission
            trade.exit_date = final_session
            trade.exit_time = close_ts
            trade.exit_price = fill_price
            trade.exit_reason = "fold_end_marked_liquidation"
            trade.net_pnl = trade.gross_pnl - trade.commission
            trade.r_multiple = trade.net_pnl / max(trade.initial_risk_dollars, 1e-9)
            trade.held_sessions = position.management.held_sessions
            completed.append(trade)
        equity_curve[-1]["cash"] = cash
        equity_curve[-1]["mtm_equity"] = cash
        equity_curve[-1]["open_positions"] = 0
    return DailyResidualReplayResult(
        initial_equity=float(initial_equity),
        final_equity=float(cash),
        trades=completed,
        equity_curve=equity_curve,
        decision_events=decision_events,
        source_fingerprint=bundle.source_fingerprint,
        factor_model=bundle.factor_model,
    )
