"""File-backed storage for IARIC research, artifacts, and live state."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import StrategySettings
from .models import (
    HeldPositionDirective,
    HeldPositionResearch,
    IntradayStateSnapshot,
    MarketResearch,
    PBSymbolState,
    PendingOrderState,
    PositionState,
    RegimeSnapshot,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
    SectorResearch,
    SymbolIntradayState,
    WatchlistArtifact,
    WatchlistItem,
)


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def research_snapshot_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.research_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def artifact_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.artifact_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def state_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.state_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def load_research_snapshot(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> ResearchSnapshot:
    payload = _read_json(research_snapshot_path(trade_date, root=root, settings=settings))
    sectors = {
        name: SectorResearch(
            name=name,
            flow_trend_20d=float(data["flow_trend_20d"]),
            breadth_20d=float(data["breadth_20d"]),
            participation=float(data["participation"]),
        )
        for name, data in payload["sectors"].items()
    }
    symbols: dict[str, ResearchSymbol] = {}
    for symbol, data in payload["symbols"].items():
        daily_bars = [
            ResearchDailyBar(
                trade_date=date.fromisoformat(bar["trade_date"]),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                event_tag=str(bar.get("event_tag", "")),
            )
            for bar in data.get("daily_bars", [])
        ]
        symbols[symbol] = ResearchSymbol(
            symbol=symbol,
            exchange=str(data["exchange"]),
            primary_exchange=str(data["primary_exchange"]),
            currency=str(data["currency"]),
            tick_size=float(data["tick_size"]),
            point_value=float(data.get("point_value", 1.0)),
            sector=str(data.get("sector", "")),
            price=float(data["price"]),
            adv20_usd=float(data["adv20_usd"]),
            median_spread_pct=float(data.get("median_spread_pct", 0.0)),
            earnings_within_sessions=(
                int(data["earnings_within_sessions"])
                if data.get("earnings_within_sessions") is not None
                else None
            ),
            blacklist_flag=bool(data.get("blacklist_flag", False)),
            halted_flag=bool(data.get("halted_flag", False)),
            severe_news_flag=bool(data.get("severe_news_flag", False)),
            etf_flag=bool(data.get("etf_flag", False)),
            adr_flag=bool(data.get("adr_flag", False)),
            preferred_flag=bool(data.get("preferred_flag", False)),
            otc_flag=bool(data.get("otc_flag", False)),
            hard_to_borrow_flag=bool(data.get("hard_to_borrow_flag", False)),
            flow_proxy_history=[float(value) for value in data.get("flow_proxy_history", [])],
            daily_bars=daily_bars,
            sector_return_20d=float(data.get("sector_return_20d", 0.0)),
            sector_return_60d=float(data.get("sector_return_60d", 0.0)),
            intraday_atr_seed=float(data.get("intraday_atr_seed", 0.0)),
            average_30m_volume=float(data.get("average_30m_volume", 0.0)),
            expected_5m_volume=float(data.get("expected_5m_volume", 0.0)),
            expected_5m_profile=tuple(
                float(value) for value in data.get("expected_5m_profile", ())
            ),
            information_state_available=bool(
                data.get("information_state_available", False)
            ),
        )
    held_positions = [
        HeldPositionResearch(
            symbol=str(item["symbol"]),
            entry_time=datetime.fromisoformat(item["entry_time"]),
            entry_price=float(item["entry_price"]),
            size=int(item["size"]),
            stop=float(item["stop"]),
            initial_r=float(item["initial_r"]),
            setup_tag=str(item.get("setup_tag", "")),
            carry_eligible_flag=bool(item.get("carry_eligible_flag", False)),
            sleeve_id=str(item.get("sleeve_id", "")),
            issuer=str(item.get("issuer", "")),
            sector=str(item.get("sector", "")),
            residual_factor_model=str(item.get("residual_factor_model", "")),
            residual_formation_sessions=int(item.get("residual_formation_sessions", 0)),
            residual_volatility=float(item.get("residual_volatility", 0.0)),
            residual_initial_dislocation_r=float(
                item.get("residual_initial_dislocation_r", 0.0)
            ),
            residual_cumulative_normalization_r=float(
                item.get("residual_cumulative_normalization_r", 0.0)
            ),
            residual_peak_normalization_r=float(
                item.get(
                    "residual_peak_normalization_r",
                    max(
                        0.0,
                        float(item.get("residual_cumulative_normalization_r", 0.0)),
                    ),
                )
            ),
            residual_held_sessions=int(item.get("residual_held_sessions", 0)),
            residual_partial_taken=bool(item.get("residual_partial_taken", False)),
            residual_last_processed_session=(
                date.fromisoformat(str(item["residual_last_processed_session"])[:10])
                if item.get("residual_last_processed_session")
                else None
            ),
            residual_qty_entry=int(item.get("residual_qty_entry", item["size"])),
            residual_entry_commission=float(
                item.get("residual_entry_commission", 0.0)
            ),
            residual_exit_commission=float(
                item.get("residual_exit_commission", 0.0)
            ),
            residual_realized_pnl_usd=float(
                item.get("residual_realized_pnl_usd", 0.0)
            ),
            residual_trade_id=str(item.get("residual_trade_id", "")),
            residual_protective_stop_client_order_id=str(
                item.get("residual_protective_stop_client_order_id", "")
            ),
            residual_protective_stop_price=float(
                item.get("residual_protective_stop_price", item.get("stop", 0.0))
            ),
            residual_protective_stop_qty=int(
                item.get("residual_protective_stop_qty", item.get("size", 0))
            ),
            residual_lane_id=str(item.get("residual_lane_id", "")),
            residual_model_contract_version=str(
                item.get("residual_model_contract_version", "")
            ),
            residual_model_intercept=float(
                item.get("residual_model_intercept", 0.0)
            ),
            residual_factor_names=tuple(
                str(value) for value in item.get("residual_factor_names", ())
            ),
            residual_factor_betas=tuple(
                float(value) for value in item.get("residual_factor_betas", ())
            ),
            residual_peer_symbols=tuple(
                str(value) for value in item.get("residual_peer_symbols", ())
            ),
            residual_model_estimation_session=(
                date.fromisoformat(str(item["residual_model_estimation_session"])[:10])
                if item.get("residual_model_estimation_session")
                else None
            ),
        )
        for item in payload.get("held_positions", [])
    ]
    market = payload["market"]
    reference_daily_bars = {
        str(symbol): [
            ResearchDailyBar(
                trade_date=date.fromisoformat(str(bar["trade_date"])[:10]),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar.get("volume", 0.0)),
                event_tag=str(bar.get("event_tag", "")),
            )
            for bar in bars
        ]
        for symbol, bars in payload.get("reference_daily_bars", {}).items()
    }
    return ResearchSnapshot(
        trade_date=date.fromisoformat(payload["trade_date"]),
        market=MarketResearch(
            price_ok=bool(market["price_ok"]),
            breadth_pct_above_20dma=float(market["breadth_pct_above_20dma"]),
            vix_percentile_1y=float(market["vix_percentile_1y"]),
            hy_spread_5d_bps_change=float(market["hy_spread_5d_bps_change"]),
            market_wide_institutional_selling=bool(market.get("market_wide_institutional_selling", False)),
        ),
        sectors=sectors,
        symbols=symbols,
        held_positions=held_positions,
        benchmark_dates=[
            date.fromisoformat(str(value)[:10])
            for value in payload.get("benchmark_dates", [])
        ],
        benchmark_closes=[
            float(value) for value in payload.get("benchmark_closes", [])
        ],
        reference_daily_bars=reference_daily_bars,
    )


def persist_watchlist_artifact(artifact: WatchlistArtifact, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    payload = {
        "trade_date": artifact.trade_date.isoformat(),
        "generated_at": artifact.generated_at.isoformat(),
        "regime": _serialize(asdict(artifact.regime)),
        "items": [_serialize(asdict(item)) for item in artifact.items],
        "tradable": [_serialize(asdict(item)) for item in artifact.tradable],
        "overflow": [_serialize(asdict(item)) for item in artifact.overflow],
        "market_wide_institutional_selling": artifact.market_wide_institutional_selling,
        "held_positions": [_serialize(asdict(item)) for item in artifact.held_positions],
        "strategy_mode": artifact.strategy_mode,
        "selection_contract_version": artifact.selection_contract_version,
        "strategy_parameters": _serialize(artifact.strategy_parameters),
    }
    path = artifact_path(artifact.trade_date, root=root, settings=settings)
    _write_json(path, payload)
    return path


def _watchlist_item_from_dict(data: dict[str, Any]) -> WatchlistItem:
    return WatchlistItem(
        symbol=str(data["symbol"]),
        exchange=str(data["exchange"]),
        primary_exchange=str(data["primary_exchange"]),
        currency=str(data["currency"]),
        tick_size=float(data["tick_size"]),
        point_value=float(data["point_value"]),
        sector=str(data["sector"]),
        regime_score=float(data["regime_score"]),
        regime_tier=str(data["regime_tier"]),
        regime_risk_multiplier=float(data["regime_risk_multiplier"]),
        sector_score=float(data["sector_score"]),
        sector_rank_weight=float(data["sector_rank_weight"]),
        sponsorship_score=float(data["sponsorship_score"]),
        sponsorship_state=str(data["sponsorship_state"]),
        persistence=float(data["persistence"]),
        intensity_z=float(data["intensity_z"]),
        accel_z=float(data["accel_z"]),
        rs_percentile=float(data["rs_percentile"]),
        leader_pass=bool(data["leader_pass"]),
        trend_pass=bool(data["trend_pass"]),
        trend_strength=float(data["trend_strength"]),
        earnings_risk_flag=bool(data["earnings_risk_flag"]),
        blacklist_flag=bool(data["blacklist_flag"]),
        anchor_date=date.fromisoformat(data["anchor_date"]),
        anchor_type=str(data["anchor_type"]),
        acceptance_pass=bool(data["acceptance_pass"]),
        avwap_ref=float(data["avwap_ref"]),
        avwap_band_lower=float(data["avwap_band_lower"]),
        avwap_band_upper=float(data["avwap_band_upper"]),
        daily_atr_estimate=float(data["daily_atr_estimate"]),
        intraday_atr_seed=float(data["intraday_atr_seed"]),
        daily_rank=float(data["daily_rank"]),
        tradable_flag=bool(data["tradable_flag"]),
        conviction_bucket=str(data["conviction_bucket"]),
        conviction_multiplier=float(data["conviction_multiplier"]),
        recommended_risk_r=float(data["recommended_risk_r"]),
        average_30m_volume=float(data.get("average_30m_volume", 0.0)),
        expected_5m_volume=float(data.get("expected_5m_volume", 0.0)),
        expected_5m_profile=tuple(float(value) for value in data.get("expected_5m_profile", ())),
        information_state_available=bool(data.get("information_state_available", False)),
        entry_gap_pct=float(data.get("entry_gap_pct", 0.0)),
        flow_proxy_gate_pass=bool(data.get("flow_proxy_gate_pass", True)),
        overflow_rank=int(data["overflow_rank"]) if data.get("overflow_rank") is not None else None,
        # Pullback V2 fields
        daily_signal_score=float(data.get("daily_signal_score", 0.0)),
        trigger_types=list(data.get("trigger_types", [])),
        trigger_tier=str(data.get("trigger_tier", "STANDARD")),
        trend_tier=str(data.get("trend_tier", "STRONG")),
        rescue_flow_candidate=bool(data.get("rescue_flow_candidate", False)),
        sizing_mult=float(data.get("sizing_mult", 1.0)),
        cdd_value=int(data.get("cdd_value", 0)),
        ema10_daily=float(data.get("ema10_daily", 0.0)),
        rsi14_daily=float(data.get("rsi14_daily", 0.0)),
        entry_rank=int(data.get("entry_rank", 0)),
        entry_rank_pct=float(data.get("entry_rank_pct", 100.0)),
        entry_rsi=float(data.get("entry_rsi", 50.0)),
        previous_close=float(data.get("previous_close", 0.0)),
        aperture_candidate=bool(data.get("aperture_candidate", False)),
        aperture_context_score=float(data.get("aperture_context_score", 0.0)),
        previous_high=float(data.get("previous_high", 0.0)),
        previous_low=float(data.get("previous_low", 0.0)),
        five_day_return=float(data.get("five_day_return", 0.0)),
        sma20_slope_atr=float(data.get("sma20_slope_atr", 0.0)),
        sleeve_id=str(data.get("sleeve_id", "")),
        residual_factor_model=str(data.get("residual_factor_model", "")),
        residual_formation_sessions=int(data.get("residual_formation_sessions", 0)),
        residual_z=float(data.get("residual_z", 0.0)),
        residual_volatility=float(data.get("residual_volatility", 0.0)),
        residual_initial_dislocation_r=float(
            data.get("residual_initial_dislocation_r", 0.0)
        ),
        residual_anchor_price=float(data.get("residual_anchor_price", 0.0)),
        residual_remaining_room_r=float(data.get("residual_remaining_room_r", 0.0)),
        residual_score_components={
            str(name): float(value)
            for name, value in data.get("residual_score_components", {}).items()
        },
        residual_admission_score=float(data.get("residual_admission_score", 0.0)),
        residual_ranking_score=float(data.get("residual_ranking_score", 0.0)),
        residual_failed_continuation_r=float(
            data.get("residual_failed_continuation_r", 0.0)
        ),
        residual_sector_return_5d=float(
            data.get("residual_sector_return_5d", 0.0)
        ),
        residual_lane_id=str(data.get("residual_lane_id", "")),
        residual_model_contract_version=str(
            data.get("residual_model_contract_version", "")
        ),
        residual_model_intercept=float(data.get("residual_model_intercept", 0.0)),
        residual_factor_names=tuple(
            str(value) for value in data.get("residual_factor_names", ())
        ),
        residual_factor_betas=tuple(
            float(value) for value in data.get("residual_factor_betas", ())
        ),
        residual_peer_symbols=tuple(
            str(value) for value in data.get("residual_peer_symbols", ())
        ),
        residual_model_estimation_session=(
            date.fromisoformat(str(data["residual_model_estimation_session"])[:10])
            if data.get("residual_model_estimation_session")
            else None
        ),
        entry_clock=str(data.get("entry_clock", "")),
    )


def load_watchlist_artifact(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> WatchlistArtifact:
    payload = _read_json(artifact_path(trade_date, root=root, settings=settings))
    regime_data = payload["regime"]
    return WatchlistArtifact(
        trade_date=date.fromisoformat(payload["trade_date"]),
        generated_at=datetime.fromisoformat(payload["generated_at"]),
        regime=RegimeSnapshot(
            score=float(regime_data["score"]),
            tier=str(regime_data["tier"]),
            risk_multiplier=float(regime_data["risk_multiplier"]),
            price_ok=bool(regime_data["price_ok"]),
            breadth_ok=bool(regime_data["breadth_ok"]),
            vol_ok=bool(regime_data["vol_ok"]),
            credit_ok=bool(regime_data["credit_ok"]),
        ),
        items=[_watchlist_item_from_dict(data) for data in payload.get("items", [])],
        tradable=[_watchlist_item_from_dict(data) for data in payload.get("tradable", [])],
        overflow=[_watchlist_item_from_dict(data) for data in payload.get("overflow", [])],
        market_wide_institutional_selling=bool(payload.get("market_wide_institutional_selling", False)),
        held_positions=[
            HeldPositionDirective(
                symbol=str(data["symbol"]),
                entry_time=datetime.fromisoformat(data["entry_time"]),
                entry_price=float(data["entry_price"]),
                size=int(data["size"]),
                stop=float(data["stop"]),
                initial_r=float(data["initial_r"]),
                setup_tag=str(data.get("setup_tag", "")),
                time_stop_deadline=(
                    datetime.fromisoformat(data["time_stop_deadline"])
                    if data.get("time_stop_deadline")
                    else None
                ),
                carry_eligible_flag=bool(data.get("carry_eligible_flag", False)),
                flow_reversal_flag=bool(data.get("flow_reversal_flag", False)),
                issuer=str(data.get("issuer", "")),
                sector=str(data.get("sector", "")),
                exchange=str(data.get("exchange", "SMART")),
                primary_exchange=str(data.get("primary_exchange", "")),
                currency=str(data.get("currency", "USD")),
                tick_size=float(data.get("tick_size", 0.01)),
                point_value=float(data.get("point_value", 1.0)),
                sleeve_id=str(data.get("sleeve_id", "")),
                residual_factor_model=str(data.get("residual_factor_model", "")),
                residual_formation_sessions=int(
                    data.get("residual_formation_sessions", 0)
                ),
                residual_volatility=float(data.get("residual_volatility", 0.0)),
                residual_initial_dislocation_r=float(
                    data.get("residual_initial_dislocation_r", 0.0)
                ),
                residual_cumulative_normalization_r=float(
                    data.get("residual_cumulative_normalization_r", 0.0)
                ),
                residual_peak_normalization_r=float(
                    data.get(
                        "residual_peak_normalization_r",
                        max(
                            0.0,
                            float(
                                data.get(
                                    "residual_cumulative_normalization_r", 0.0
                                )
                            ),
                        ),
                    )
                ),
                residual_held_sessions=int(data.get("residual_held_sessions", 0)),
                residual_partial_taken=bool(data.get("residual_partial_taken", False)),
                residual_last_processed_session=(
                    date.fromisoformat(str(data["residual_last_processed_session"])[:10])
                    if data.get("residual_last_processed_session")
                    else None
                ),
                residual_pending_action=str(data.get("residual_pending_action", "hold")),
                residual_pending_reason=str(data.get("residual_pending_reason", "")),
                residual_pending_exit_fraction=float(
                    data.get("residual_pending_exit_fraction", 0.0)
                ),
                residual_qty_entry=int(data.get("residual_qty_entry", data["size"])),
                residual_entry_commission=float(
                    data.get("residual_entry_commission", 0.0)
                ),
                residual_exit_commission=float(
                    data.get("residual_exit_commission", 0.0)
                ),
                residual_realized_pnl_usd=float(
                    data.get("residual_realized_pnl_usd", 0.0)
                ),
                residual_trade_id=str(data.get("residual_trade_id", "")),
                residual_protective_stop_client_order_id=str(
                    data.get("residual_protective_stop_client_order_id", "")
                ),
                residual_protective_stop_price=float(
                    data.get("residual_protective_stop_price", data.get("stop", 0.0))
                ),
                residual_protective_stop_qty=int(
                    data.get("residual_protective_stop_qty", data.get("size", 0))
                ),
                residual_lane_id=str(data.get("residual_lane_id", "")),
                residual_model_contract_version=str(
                    data.get("residual_model_contract_version", "")
                ),
                residual_model_intercept=float(
                    data.get("residual_model_intercept", 0.0)
                ),
                residual_factor_names=tuple(
                    str(value) for value in data.get("residual_factor_names", ())
                ),
                residual_factor_betas=tuple(
                    float(value) for value in data.get("residual_factor_betas", ())
                ),
                residual_peer_symbols=tuple(
                    str(value) for value in data.get("residual_peer_symbols", ())
                ),
                residual_model_estimation_session=(
                    date.fromisoformat(
                        str(data["residual_model_estimation_session"])[:10]
                    )
                    if data.get("residual_model_estimation_session")
                    else None
                ),
            )
            for data in payload.get("held_positions", [])
        ],
        strategy_mode=str(payload.get("strategy_mode", "legacy_pullback")),
        selection_contract_version=str(payload.get("selection_contract_version", "")),
        strategy_parameters=dict(payload.get("strategy_parameters", {})),
    )


def persist_intraday_state(snapshot: IntradayStateSnapshot, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    path = state_path(snapshot.trade_date, root=root, settings=settings)
    payload = {
        "trade_date": snapshot.trade_date.isoformat(),
        "saved_at": snapshot.saved_at.isoformat(),
        "symbols": [_serialize(asdict(symbol)) for symbol in snapshot.symbols],
        "last_decision_code": snapshot.last_decision_code,
        "meta": _serialize(snapshot.meta),
    }
    _write_json(path, payload)
    return path


def coerce_intraday_state_snapshot(
    payload: IntradayStateSnapshot | dict[str, Any],
) -> IntradayStateSnapshot:
    if isinstance(payload, IntradayStateSnapshot):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            "Intraday state payload must be an IntradayStateSnapshot or dict"
        )

    def _as_datetime(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    def _as_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    meta = dict(payload.get("meta", {}))
    if meta.get("strategy_mode") == "daily_residual_reversion":
        from .core.daily_residual import hydrate_daily_residual_symbol_state

        return IntradayStateSnapshot(
            trade_date=_as_date(payload["trade_date"]),
            saved_at=_as_datetime(payload["saved_at"]),
            symbols=[
                hydrate_daily_residual_symbol_state(data)
                for data in payload.get("symbols", [])
            ],
            last_decision_code=str(payload.get("last_decision_code", "")),
            meta=meta,
        )

    def _pending(data: dict[str, Any] | None) -> PendingOrderState | None:
        if not data:
            return None
        return PendingOrderState(
            oms_order_id=str(data["oms_order_id"]),
            submitted_at=_as_datetime(data["submitted_at"]),
            role=str(data["role"]),
            requested_qty=int(data["requested_qty"]),
            limit_price=float(data["limit_price"]) if data.get("limit_price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            cancel_requested=bool(data.get("cancel_requested", False)),
        )

    def _position(data: dict[str, Any] | None) -> PositionState | None:
        if not data:
            return None
        return PositionState(
            entry_price=float(data["entry_price"]),
            qty_entry=int(data["qty_entry"]),
            qty_open=int(data["qty_open"]),
            final_stop=float(data["final_stop"]),
            current_stop=float(data["current_stop"]),
            entry_time=_as_datetime(data["entry_time"]),
            initial_risk_per_share=float(data["initial_risk_per_share"]),
            max_favorable_price=float(data["max_favorable_price"]),
            max_adverse_price=float(data.get("max_adverse_price", data["entry_price"])),
            partial_taken=bool(data.get("partial_taken", False)),
            stop_order_id=str(data.get("stop_order_id", "")),
            trade_id=str(data.get("trade_id", "")),
            realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
            entry_commission=float(data.get("entry_commission", 0.0)),
            exit_commission=float(data.get("exit_commission", 0.0)),
            opportunity_event_id=str(data.get("opportunity_event_id", "")),
            reversion_anchor=float(data.get("reversion_anchor", 0.0)),
            structural_stop_anchor=float(data.get("structural_stop_anchor", 0.0)),
            initial_remaining_room_atr=float(data.get("initial_remaining_room_atr", 0.0)),
            prospective_reward_risk=float(data.get("prospective_reward_risk", 0.0)),
            setup_tag=str(data.get("setup_tag", "UNCLASSIFIED")),
            time_stop_deadline=_as_datetime(data.get("time_stop_deadline")),
            pending_partial_stop=float(data.get("pending_partial_stop", 0.0)),
            pending_partial_stop_buffer=float(data.get("pending_partial_stop_buffer", 0.0)),
        )

    def _pb_symbol(data: dict[str, Any]) -> PBSymbolState:
        return PBSymbolState(
            symbol=str(data["symbol"]),
            stage=str(data.get("stage", "WATCHING")),
            route_family=str(data.get("route_family", "")),
            setup_low=float(data.get("setup_low", 0.0)),
            reclaim_level=float(data.get("reclaim_level", 0.0)),
            stop_level=float(data.get("stop_level", 0.0)),
            acceptance_count=int(data.get("acceptance_count", 0)),
            required_acceptance=int(data.get("required_acceptance", 1)),
            intraday_score=float(data.get("intraday_score", 0.0)),
            score_components=dict(data.get("score_components", {})),
            bars_seen_today=int(data.get("bars_seen_today", 0)),
            session_low=float(data.get("session_low", 0.0)),
            session_high=float(data.get("session_high", 0.0)),
            in_position=bool(data.get("in_position", False)),
            position=_position(data.get("position")),
            entry_order=_pending(data.get("entry_order")),
            exit_order=_pending(data.get("exit_order")),
            pending_hard_exit=bool(data.get("pending_hard_exit", False)),
            daily_signal_score=float(data.get("daily_signal_score", 0.0)),
            trigger_types=list(data.get("trigger_types", [])),
            trigger_tier=str(data.get("trigger_tier", "STANDARD")),
            trend_tier=str(data.get("trend_tier", "STRONG")),
            rescue_flow_candidate=bool(data.get("rescue_flow_candidate", False)),
            sizing_mult=float(data.get("sizing_mult", 1.0)),
            daily_atr=float(data.get("daily_atr", 0.0)),
            entry_atr=float(data.get("entry_atr", 0.0)),
            last_1m_bar_time=_as_datetime(data.get("last_1m_bar_time")),
            last_5m_bar_time=_as_datetime(data.get("last_5m_bar_time")),
            active_order_id=str(data["active_order_id"]) if data.get("active_order_id") else None,
            last_transition_reason=str(data.get("last_transition_reason", "")),
            mfe_stage=int(data.get("mfe_stage", 0)),
            breakeven_activated=bool(data.get("breakeven_activated", False)),
            trail_active=bool(data.get("trail_active", False)),
            hold_bars=int(data.get("hold_bars", 0)),
            risk_per_share=float(data.get("risk_per_share", 0.0)),
            v2_partial_taken=bool(data.get("v2_partial_taken", False)),
            carry_decision_path=str(data.get("carry_decision_path", "")),
            consecutive_bars_below_vwap=int(data.get("consecutive_bars_below_vwap", 0)),
            ema10_daily=float(data.get("ema10_daily", 0.0)),
            rsi14_daily=float(data.get("rsi14_daily", 0.0)),
            opportunity_family=str(data.get("opportunity_family", "")),
            opportunity_signal_bar_idx=int(data.get("opportunity_signal_bar_idx", -1)),
            opportunity_signal_close=float(data.get("opportunity_signal_close", 0.0)),
            opportunity_event_id=str(data.get("opportunity_event_id", "")),
            opportunity_reversion_anchor=float(data.get("opportunity_reversion_anchor", 0.0)),
            opportunity_stop_anchor=float(data.get("opportunity_stop_anchor", 0.0)),
            opportunity_remaining_room_atr=float(data.get("opportunity_remaining_room_atr", 0.0)),
            opportunity_prospective_reward_risk=float(
                data.get("opportunity_prospective_reward_risk", 0.0)
            ),
            opportunity_consumed_families=list(data.get("opportunity_consumed_families", [])),
            opportunity_audit_bar_idx=int(data.get("opportunity_audit_bar_idx", -1)),
            opportunity_audit_events=list(data.get("opportunity_audit_events", [])),
            stopped_out_today=bool(data.get("stopped_out_today", False)),
            flush_bar_idx=int(data.get("flush_bar_idx", 0)),
            ready_bar_idx=int(data.get("ready_bar_idx", -1)),
            target_entry_price=float(data.get("target_entry_price", 0.0)),
            improvement_expires=int(data.get("improvement_expires", 0)),
            invalid_reason=str(data.get("invalid_reason", "")),
            invalid_reset_bar=int(data.get("invalid_reset_bar", 0)),
            ready_cpr=float(data.get("ready_cpr", 0.0)),
            ready_volume_ratio=float(data.get("ready_volume_ratio", 0.0)),
            ready_timestamp=_as_datetime(data.get("ready_timestamp")),
            accepted_bar_idx=int(data.get("accepted_bar_idx", -1)),
            accepted_timestamp=_as_datetime(data.get("accepted_timestamp")),
            accepted_entry_price=float(data.get("accepted_entry_price", 0.0)),
            accepted_entry_trigger=str(data.get("accepted_entry_trigger", "")),
            accepted_route_family=str(data.get("accepted_route_family", "")),
            accepted_score=float(data.get("accepted_score", 0.0)),
            accepted_session_atr=float(data.get("accepted_session_atr", 0.0)),
            accepted_score_components=dict(data.get("accepted_score_components", {})),
            accepted_lane_id=str(data.get("accepted_lane_id", "")),
            accepted_event_id=str(data.get("accepted_event_id", "")),
            accepted_reversion_anchor=float(data.get("accepted_reversion_anchor", 0.0)),
            accepted_stop_anchor=float(data.get("accepted_stop_anchor", 0.0)),
            accepted_remaining_room_atr=float(data.get("accepted_remaining_room_atr", 0.0)),
            accepted_prospective_reward_risk=float(
                data.get("accepted_prospective_reward_risk", 0.0)
            ),
            entry_rank=int(data.get("entry_rank", 0)),
            entry_rank_pct=float(data.get("entry_rank_pct", 100.0)),
            entry_rsi=float(data.get("entry_rsi", 50.0)),
        )

    symbols = []
    for data in payload.get("symbols", []):
        if "stage" in data:
            symbols.append(_pb_symbol(data))
        else:
            symbols.append(
                SymbolIntradayState(
                    symbol=str(data["symbol"]),
                    tier=str(data.get("tier", "COLD")),
                    fsm_state=str(data.get("fsm_state", "IDLE")),
                    in_position=bool(data.get("in_position", False)),
                    position_qty=int(data.get("position_qty", 0)),
                    avg_price=float(data["avg_price"]) if data.get("avg_price") is not None else None,
                    setup_type=str(data["setup_type"]) if data.get("setup_type") else None,
                    setup_low=float(data["setup_low"]) if data.get("setup_low") is not None else None,
                    reclaim_level=float(data["reclaim_level"]) if data.get("reclaim_level") is not None else None,
                    stop_level=float(data["stop_level"]) if data.get("stop_level") is not None else None,
                    setup_time=datetime.fromisoformat(data["setup_time"]) if data.get("setup_time") else None,
                    invalidated_at=datetime.fromisoformat(data["invalidated_at"]) if data.get("invalidated_at") else None,
                    acceptance_count=int(data.get("acceptance_count", 0)),
                    required_acceptance_count=int(data.get("required_acceptance_count", 0)),
                    location_grade=str(data["location_grade"]) if data.get("location_grade") else None,
                    session_vwap=float(data["session_vwap"]) if data.get("session_vwap") is not None else None,
                    avwap_live=float(data["avwap_live"]) if data.get("avwap_live") is not None else None,
                    sponsorship_signal=str(data.get("sponsorship_signal", "NEUTRAL")),
                    micropressure_signal=str(data.get("micropressure_signal", "NEUTRAL")),
                    micropressure_mode=str(data.get("micropressure_mode", "PROXY")),
                    flowproxy_signal=str(data.get("flowproxy_signal", "UNAVAILABLE")),
                    confidence=str(data["confidence"]) if data.get("confidence") else None,
                    last_1m_bar_time=_as_datetime(data.get("last_1m_bar_time")),
                    last_5m_bar_time=_as_datetime(data.get("last_5m_bar_time")),
                    active_order_id=str(data["active_order_id"]) if data.get("active_order_id") else None,
                    time_stop_deadline=_as_datetime(data.get("time_stop_deadline")),
                    setup_tag=str(data["setup_tag"]) if data.get("setup_tag") else None,
                    expected_volume_pct=float(data.get("expected_volume_pct", 0.0)),
                    average_30m_volume=float(data.get("average_30m_volume", 0.0)),
                    last_transition_reason=str(data.get("last_transition_reason", "")),
                    entry_order=_pending(data.get("entry_order")),
                    position=_position(data.get("position")),
                    exit_order=_pending(data.get("exit_order")),
                    pending_hard_exit=bool(data.get("pending_hard_exit", False)),
                )
            )

    return IntradayStateSnapshot(
        trade_date=_as_date(payload["trade_date"]),
        saved_at=_as_datetime(payload["saved_at"]),
        symbols=symbols,
        last_decision_code=str(payload.get("last_decision_code", "")),
        meta=dict(payload.get("meta", {})),
    )


def load_intraday_state(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> IntradayStateSnapshot:
    payload = _read_json(state_path(trade_date, root=root, settings=settings))
    return coerce_intraday_state_snapshot(payload)
