"""Live/replay-identical nightly selection for daily residual reversion.

The adapter consumes only completed daily price/volume bars from a
``ResearchSnapshot``.  It deliberately has no news, quote or order-imbalance
input.  Production research generation and historical replay both call this
module, so residual construction, score inputs and capacity arbitration cannot
drift between the two paths.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from math import exp, isfinite, log, log10, sqrt
from statistics import fmean, median
from typing import Mapping, Sequence

import numpy as np

from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS
from strategies.stock.volume_units import dollar_volume

from .config import StrategySettings
from .core.daily_residual import (
    DAILY_RESIDUAL_SLEEVE,
    DailyResidualFeatures,
    DailyResidualOpportunity,
    DailyResidualReplacementIncumbent,
    ResidualManagementPolicy,
    ResidualManagementState,
    advance_residual_management,
    choose_capacity_neutral_replacements,
    rank_daily_residual_opportunities,
)
from .core.lanes import issuer_key
from .core.residual import (
    FrozenResidualModel,
    causal_rolling_factor_contracts,
)
from .models import (
    RegimeSnapshot,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
    HeldPositionDirective,
    WatchlistArtifact,
    WatchlistItem,
)


SECTOR_REFERENCE: dict[str, str] = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
SUPPORTED_FACTOR_MODELS = {
    "market_only",
    "market_sector",
    "market_sector_peer",
    "peer_demeaned",
}


@dataclass(frozen=True, slots=True)
class PreparedDailyResidualSelection:
    """Settings-independent nightly inputs reusable by exact portfolio replays."""

    trade_date: date
    factor_model: str
    formation_sessions: int
    market_trend_z_20d: float
    opportunities: tuple[DailyResidualOpportunity, ...]
    frozen_model_by_symbol: Mapping[str, FrozenResidualModel]
    percentile_by_symbol: Mapping[str, float]
    residual_volatility_by_symbol: Mapping[str, float]
    failed_continuation_by_symbol: Mapping[str, float]
    sector_return_by_symbol: Mapping[str, float]


def _returns(bars: Sequence[ResearchDailyBar]) -> dict[object, float]:
    """Completed-session log returns; additive across formation horizons."""

    result: dict[object, float] = {}
    previous = None
    for bar in sorted(bars, key=lambda value: value.trade_date):
        close = float(bar.close)
        if previous is not None and previous > 0.0 and close > 0.0:
            result[bar.trade_date] = log(close / previous)
        previous = close
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return float("nan")
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    return value


def _causal_peer_contract(
    returns_by_symbol: Mapping[str, Mapping[object, float]],
    sector_by_symbol: Mapping[str, str],
    dates: Sequence[object],
    *,
    lookback: int = 120,
    min_observations: int = 60,
    peer_count: int = 5,
    rebalance_sessions: int = 21,
) -> tuple[
    dict[str, dict[object, float]],
    dict[str, dict[object, tuple[str, ...]]],
]:
    """Causal correlated-peer median, rebalanced from prior sessions only.

    Sector peers are preferred.  A singleton sector falls back to prior-return
    correlations across the frozen execution universe so it is not silently
    excluded from a peer-capable residual model.
    """

    output = {symbol: {} for symbol in returns_by_symbol}
    membership_history: dict[str, dict[object, tuple[str, ...]]] = {
        symbol: {} for symbol in returns_by_symbol
    }
    by_sector: dict[str, list[str]] = defaultdict(list)
    for symbol, sector in sector_by_symbol.items():
        by_sector[str(sector)].append(symbol)
    memberships: dict[str, tuple[str, ...]] = {}
    for index, session in enumerate(dates):
        if index >= min_observations and (
            not memberships or (index - min_observations) % rebalance_sessions == 0
        ):
            history_dates = dates[max(0, index - lookback) : index]
            refreshed: dict[str, tuple[str, ...]] = {}
            for sector_symbols in by_sector.values():
                for symbol in sector_symbols:
                    ranked: list[tuple[float, str]] = []
                    sector_candidates = [
                        peer for peer in sector_symbols if peer != symbol
                    ]
                    candidates = sector_candidates or [
                        peer for peer in returns_by_symbol if peer != symbol
                    ]
                    for peer in candidates:
                        if peer == symbol:
                            continue
                        pairs = [
                            (
                                returns_by_symbol[symbol].get(day),
                                returns_by_symbol[peer].get(day),
                            )
                            for day in history_dates
                        ]
                        valid = [
                            (float(left), float(right))
                            for left, right in pairs
                            if left is not None
                            and right is not None
                            and isfinite(float(left))
                            and isfinite(float(right))
                        ]
                        if len(valid) < min_observations:
                            continue
                        correlation = _correlation(
                            [row[0] for row in valid], [row[1] for row in valid]
                        )
                        if isfinite(correlation):
                            ranked.append((correlation, peer))
                    ranked.sort(key=lambda row: (-row[0], row[1]))
                    selected = tuple(peer for _correlation_value, peer in ranked[:peer_count])
                    if selected:
                        refreshed[symbol] = selected
            memberships = refreshed
        for symbol, peers in memberships.items():
            values = [
                float(returns_by_symbol[peer][session])
                for peer in peers
                if session in returns_by_symbol[peer]
                and isfinite(float(returns_by_symbol[peer][session]))
            ]
            if len(values) >= min(2, len(peers)):
                output[symbol][session] = float(median(values))
                membership_history[symbol][session] = tuple(peers)
    return output, membership_history


def _causal_peer_returns(
    returns_by_symbol: Mapping[str, Mapping[object, float]],
    sector_by_symbol: Mapping[str, str],
    dates: Sequence[object],
    *,
    lookback: int = 120,
    min_observations: int = 60,
    peer_count: int = 5,
    rebalance_sessions: int = 21,
) -> dict[str, dict[object, float]]:
    """Compatibility wrapper returning only the causal peer-return series."""

    returns, _memberships = _causal_peer_contract(
        returns_by_symbol,
        sector_by_symbol,
        dates,
        lookback=lookback,
        min_observations=min_observations,
        peer_count=peer_count,
        rebalance_sessions=rebalance_sessions,
    )
    return returns


def _residual_histories(
    snapshot: ResearchSnapshot,
    factor_model: str,
) -> dict[str, dict[object, float]]:
    residuals, _models, _memberships = _residual_contracts(snapshot, factor_model)
    return residuals


def _residual_contracts(
    snapshot: ResearchSnapshot,
    factor_model: str,
) -> tuple[
    dict[str, dict[object, float]],
    dict[str, dict[object, FrozenResidualModel]],
    dict[str, dict[object, tuple[str, ...]]],
]:
    """Build causal residual histories plus the exact point-in-time models."""

    if factor_model not in SUPPORTED_FACTOR_MODELS:
        raise ValueError(f"unsupported daily residual factor model: {factor_model}")
    symbols = {
        symbol: item
        for symbol, item in snapshot.symbols.items()
        if symbol in BACKTESTED_INTRADAY_STOCK_SYMBOLS
    }
    if set(symbols) != set(BACKTESTED_INTRADAY_STOCK_SYMBOLS):
        missing = sorted(set(BACKTESTED_INTRADAY_STOCK_SYMBOLS) - set(symbols))
        raise ValueError(f"daily residual snapshot is missing frozen-universe names: {missing}")
    returns_by_symbol = {
        symbol: _returns(item.daily_bars) for symbol, item in symbols.items()
    }
    dates = sorted({day for values in returns_by_symbol.values() for day in values})
    sectors = {symbol: item.sector for symbol, item in symbols.items()}
    if factor_model in {"market_sector_peer", "peer_demeaned"}:
        peer_returns, peer_memberships = _causal_peer_contract(
            returns_by_symbol, sectors, dates
        )
    else:
        peer_returns = {symbol: {} for symbol in returns_by_symbol}
        peer_memberships = {symbol: {} for symbol in returns_by_symbol}
    reference_returns = {
        symbol: _returns(bars)
        for symbol, bars in snapshot.reference_daily_bars.items()
    }
    market = reference_returns.get("SPY", {})
    output: dict[str, dict[object, float]] = {}
    models: dict[str, dict[object, FrozenResidualModel]] = {}
    for symbol, item in symbols.items():
        stock = returns_by_symbol[symbol]
        if factor_model == "peer_demeaned":
            output[symbol] = {
                day: float(value) - float(peer_returns[symbol][day])
                for day, value in stock.items()
                if day in peer_returns[symbol]
            }
            models[symbol] = {
                day: FrozenResidualModel(
                    factor_names=("peer",),
                    intercept=0.0,
                    factor_betas=(1.0,),
                    peer_symbols=peer_memberships[symbol][day],
                    estimation_session=day,
                )
                for day in output[symbol]
                if day in peer_memberships[symbol]
            }
            continue
        factor_rows: dict[object, dict[str, float]] = {}
        sector_reference = reference_returns.get(
            SECTOR_REFERENCE.get(item.sector, ""), {}
        )
        for day in stock:
            row: dict[str, float] = {}
            if day in market:
                row["market"] = market[day]
            if factor_model in {"market_sector", "market_sector_peer"} and day in sector_reference:
                row["sector"] = sector_reference[day]
            if factor_model == "market_sector_peer" and day in peer_returns[symbol]:
                row["peer"] = peer_returns[symbol][day]
            factor_rows[day] = row
        factor_names = {
            "market_only": ("market",),
            "market_sector": ("market", "sector"),
            "market_sector_peer": ("market", "sector", "peer"),
        }[factor_model]
        residual_history, model_history = causal_rolling_factor_contracts(
            stock,
            factor_rows,
            factor_names=factor_names,
            window=120,
            min_observations=60,
            ridge=1e-5,
        )
        output[symbol] = residual_history
        models[symbol] = {
            day: FrozenResidualModel(
                factor_names=model.factor_names,
                intercept=model.intercept,
                factor_betas=model.factor_betas,
                peer_symbols=peer_memberships[symbol].get(day, ()),
                estimation_session=day,
            )
            for day, model in model_history.items()
        }
    return output, models, peer_memberships


def _prior_atr_fraction(bars: Sequence[ResearchDailyBar]) -> float:
    ordered = sorted(bars, key=lambda value: value.trade_date)
    if len(ordered) < 17:
        return 0.0
    true_ranges: list[float] = []
    for previous, current in zip(ordered[-22:-1], ordered[-21:-1]):
        true_ranges.append(
            max(
                float(current.high) - float(current.low),
                abs(float(current.high) - float(previous.close)),
                abs(float(current.low) - float(previous.close)),
            )
        )
    previous_close = float(ordered[-2].close)
    return fmean(true_ranges[-20:]) / previous_close if previous_close > 0.0 else 0.0


def _fixed_peer_returns(
    returns_by_symbol: Mapping[str, Mapping[object, float]],
    peers: Sequence[str],
) -> dict[object, float]:
    sessions = sorted(
        {session for peer in peers for session in returns_by_symbol.get(peer, {})}
    )
    result: dict[object, float] = {}
    for session in sessions:
        values = [
            float(returns_by_symbol[peer][session])
            for peer in peers
            if session in returns_by_symbol.get(peer, {})
            and isfinite(float(returns_by_symbol[peer][session]))
        ]
        if len(values) >= min(2, len(peers)):
            result[session] = float(median(values))
    return result


def _frozen_model_residual_history(
    *,
    symbol: str,
    sector: str,
    model: FrozenResidualModel,
    stock_returns: Mapping[str, Mapping[object, float]],
    reference_returns: Mapping[str, Mapping[object, float]],
    cache: dict[tuple[object, ...], dict[object, float]] | None = None,
) -> dict[object, float]:
    """Re-express formation and management history on one immutable model."""

    market = reference_returns.get("SPY", {})
    sector_factor = reference_returns.get(SECTOR_REFERENCE.get(sector, ""), {})
    cache_key = (
        symbol,
        sector,
        model.contract_version,
        model.factor_names,
        float(model.intercept),
        tuple(float(value) for value in model.factor_betas),
        model.peer_symbols,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result = cache.setdefault(cache_key, {}) if cache is not None else {}
    for session, stock_value in stock_returns.get(symbol, {}).items():
        if session in result:
            continue
        factors: dict[str, float] = {}
        if "market" in model.factor_names and session in market:
            factors["market"] = float(market[session])
        if "sector" in model.factor_names and session in sector_factor:
            factors["sector"] = float(sector_factor[session])
        if "peer" in model.factor_names:
            peer_values = [
                float(stock_returns[peer][session])
                for peer in model.peer_symbols
                if peer in stock_returns
                and session in stock_returns[peer]
                and isfinite(float(stock_returns[peer][session]))
            ]
            if len(peer_values) >= min(2, len(model.peer_symbols)):
                factors["peer"] = float(median(peer_values))
        residual = model.residual_return(float(stock_value), factors)
        if isfinite(residual):
            result[session] = float(residual)
    return result


def _component_inputs(
    symbol: ResearchSymbol,
    residuals: Mapping[object, float],
    *,
    formation_sessions: int,
    residual_z: float,
    percentile: float,
    dispersion_ratio: float,
) -> tuple[DailyResidualFeatures, float, float, float]:
    bars = sorted(symbol.daily_bars, key=lambda value: value.trade_date)
    dates = [bar.trade_date for bar in bars if bar.trade_date in residuals]
    values = [float(residuals[day]) for day in dates]
    residual_history = values[-61:-1]
    residual_volatility = (
        float(np.std(residual_history, ddof=1))
        if len(residual_history) >= 40
        else 0.0
    )
    latest = bars[-1]
    if formation_sessions == 1:
        # A one-session loser cannot simultaneously have a positive daily
        # residual. Its honest OHLCV-resolution failure signal is recovery
        # from the completed session low, scaled by residual volatility.
        low = max(float(latest.low), 1e-12)
        failed_continuation_r = (
            log(max(float(latest.close), low) / low) / residual_volatility
            if residual_volatility > 0.0
            else 0.0
        )
    else:
        # Multi-session normalization requires the latest completed residual
        # return itself to be positive. Merely being less negative than an
        # earlier session is still continuation and must fail the hard gate.
        failed_continuation_r = (
            values[-1] / residual_volatility
            if residual_volatility > 0.0
            else 0.0
        )
    formation = values[-formation_sessions:]
    shock = max(-sum(formation), 0.0)
    prior_trend = max(-sum(values[-(formation_sessions + 5) : -formation_sessions]), 0.0)
    freshness = shock / (shock + prior_trend + 1e-12)
    width = max(float(latest.high) - float(latest.low), 1e-12)
    close_location = min(max((float(latest.close) - float(latest.low)) / width, 0.0), 1.0)
    lower_wick = min(
        max((min(float(latest.open), float(latest.close)) - float(latest.low)) / width, 0.0),
        1.0,
    )
    rejection = 0.65 * close_location + 0.35 * lower_wick
    prior_volumes = [float(bar.volume) for bar in bars[-21:-1] if float(bar.volume) >= 0.0]
    median_volume = median(prior_volumes) if prior_volumes else 0.0
    relative_volume = float(latest.volume) / median_volume if median_volume > 0.0 else 0.0
    earlier = bars[-24:-21]
    earlier_relatives = [
        float(bar.volume) / median_volume for bar in earlier if median_volume > 0.0
    ]
    prior_relative = fmean(earlier_relatives) if earlier_relatives else 0.0
    volume_transition = min(max(relative_volume / 2.0, 0.0), 1.0) * (
        1.0 - min(max(prior_relative / 3.0, 0.0), 1.0) * 0.50
    )
    # Reversion is strongest when the dislocation has enough participation to
    # be real but not the extreme volume characteristic of fresh information.
    # A broad triangular transform is deliberately economic, smooth and fixed;
    # it is not an outcome-fitted score band.
    volume_exhaustion_quality = max(
        0.0,
        1.0 - abs(volume_transition - 0.55) / 0.35,
    )
    atr_fraction = _prior_atr_fraction(bars)
    room = min(max(shock / max(atr_fraction * 3.0, 1e-12), 0.0), 1.0)
    prior_adv_rows = bars[-21:-1]
    adv = (
        fmean(
            float(dollar_volume(float(bar.close), float(bar.volume)))
            for bar in prior_adv_rows
        )
        if prior_adv_rows
        else 0.0
    )
    adv_quality = min(
        max(
            (log10(max(adv, 50_000_000.0)) - log10(50_000_000.0))
            / (log10(1_000_000_000.0) - log10(50_000_000.0)),
            0.0,
        ),
        1.0,
    )
    volatility_quality = 1.0 - min(max(residual_volatility / 0.06, 0.0), 1.0)
    dispersion_quality = 1.0 - min(abs(dispersion_ratio - 1.0) / 1.5, 1.0)
    regime_quality = 0.45 * adv_quality + 0.30 * volatility_quality + 0.25 * dispersion_quality
    extremeness = 0.50 * (1.0 - percentile) + 0.50 * min(max(-residual_z / 3.0, 0.0), 1.0)
    return (
        DailyResidualFeatures(
            residual_extremeness=extremeness,
            shock_freshness=freshness,
            price_rejection_recovery=rejection,
            volume_transition=volume_transition,
            volume_exhaustion_quality=volume_exhaustion_quality,
            residual_normalization_room=room,
            regime_execution_quality=regime_quality,
            failed_continuation=min(max(failed_continuation_r, 0.0), 1.0),
        ),
        residual_volatility,
        adv,
        failed_continuation_r,
    )


def prepare_daily_residual_selection(
    snapshot: ResearchSnapshot,
    *,
    factor_model: str,
    formation_sessions: int,
    precomputed_residuals: Mapping[str, Mapping[object, float]] | None = None,
    precomputed_models: Mapping[
        str, Mapping[object, FrozenResidualModel]
    ] | None = None,
    frozen_history_cache: dict[
        tuple[object, ...], dict[object, float]
    ] | None = None,
    precomputed_stock_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
    precomputed_reference_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
) -> PreparedDailyResidualSelection:
    """Compute the expensive nightly feature frame without policy thresholds."""

    formation = int(formation_sessions)
    if formation not in {1, 3, 5}:
        raise ValueError("daily residual formation must be one, three or five sessions")
    factor_model = str(factor_model)
    if precomputed_residuals is None or precomputed_models is None:
        raw_residuals, raw_models, _peer_memberships = _residual_contracts(
            snapshot, factor_model
        )
    else:
        raw_residuals = {
            symbol: {
                session: float(value)
                for session, value in history.items()
                if session < snapshot.trade_date
            }
            for symbol, history in precomputed_residuals.items()
        }
        raw_models = {
            symbol: {
                session: model
                for session, model in history.items()
                if session < snapshot.trade_date
            }
            for symbol, history in precomputed_models.items()
        }
    stock_returns = (
        precomputed_stock_returns
        if precomputed_stock_returns is not None
        else {
            symbol: _returns(item.daily_bars)
            for symbol, item in snapshot.symbols.items()
            if symbol in BACKTESTED_INTRADAY_STOCK_SYMBOLS
        }
    )
    reference_returns = (
        precomputed_reference_returns
        if precomputed_reference_returns is not None
        else {
            symbol: _returns(bars)
            for symbol, bars in snapshot.reference_daily_bars.items()
        }
    )
    market_history = sorted(
        float(value)
        for session, value in reference_returns.get("SPY", {}).items()
        if session < snapshot.trade_date and isfinite(float(value))
    )
    market_window = market_history[-20:]
    market_volatility = (
        float(np.std(market_window, ddof=1))
        if len(market_window) >= 20
        else 0.0
    )
    market_trend_z_20d = (
        sum(market_window) / (market_volatility * sqrt(20.0))
        if market_volatility > 0.0
        else float("-inf")
    )
    residuals: dict[str, dict[object, float]] = {}
    frozen_model_by_symbol: dict[str, FrozenResidualModel] = {}
    for symbol in BACKTESTED_INTRADAY_STOCK_SYMBOLS:
        model_history = raw_models.get(symbol, {})
        available_sessions = [
            session for session in model_history if session < snapshot.trade_date
        ]
        if not available_sessions:
            continue
        signal_session = max(available_sessions)
        model = model_history[signal_session]
        frozen = _frozen_model_residual_history(
            symbol=symbol,
            sector=snapshot.symbols[symbol].sector,
            model=model,
            stock_returns=stock_returns,
            reference_returns=reference_returns,
            cache=frozen_history_cache,
        )
        frozen = {
            session: value
            for session, value in frozen.items()
            if session < snapshot.trade_date
        }
        if frozen:
            residuals[symbol] = frozen
            frozen_model_by_symbol[symbol] = model
    latest_z: dict[str, float] = {}
    prior_dispersions: dict[object, list[float]] = defaultdict(list)
    for symbol, history in residuals.items():
        ordered = [
            (day, float(value))
            for day, value in sorted(history.items())
            if isfinite(float(value))
        ]
        for day, value in ordered:
            prior_dispersions[day].append(value)
        if len(ordered) < max(formation, 41):
            continue
        values = [row[1] for row in ordered]
        sample = values[-61:-1]
        volatility = float(np.std(sample, ddof=1)) if len(sample) >= 40 else 0.0
        if volatility <= 0.0:
            continue
        latest_z[symbol] = sum(values[-formation:]) / (
            volatility * sqrt(float(formation))
        )
    if not latest_z:
        raise ValueError("daily residual snapshot has no factor-ready observations")
    dispersion_series = [
        float(np.std(values, ddof=1))
        for _day, values in sorted(prior_dispersions.items())
        if len(values) >= 10
    ]
    current_dispersion = dispersion_series[-1] if dispersion_series else 0.0
    reference_dispersion = (
        median(dispersion_series[-121:-1])
        if len(dispersion_series) >= 61
        else current_dispersion
    )
    dispersion_ratio = (
        current_dispersion / reference_dispersion
        if reference_dispersion > 0.0
        else 1.0
    )
    ordered_z = sorted(latest_z.items(), key=lambda row: (row[1], row[0]))
    percentile = {
        symbol: (index + 1) / len(ordered_z)
        for index, (symbol, _value) in enumerate(ordered_z)
    }
    opportunities: list[DailyResidualOpportunity] = []
    residual_vol_by_symbol: dict[str, float] = {}
    failed_continuation_by_symbol: dict[str, float] = {}
    sector_return_by_symbol: dict[str, float] = {}
    for symbol, residual_z in latest_z.items():
        research_symbol = snapshot.symbols[symbol]
        sector_reference = SECTOR_REFERENCE.get(research_symbol.sector, "")
        sector_history = sorted(
            (session, float(value))
            for session, value in reference_returns.get(sector_reference, {}).items()
            if session < snapshot.trade_date and isfinite(float(value))
        )
        sector_return_5d = (
            sum(value for _session, value in sector_history[-5:])
            if len(sector_history) >= 5
            else float("-inf")
        )
        features, residual_volatility, adv, failed_continuation_r = (
            _component_inputs(
                research_symbol,
                residuals[symbol],
                formation_sessions=formation,
                residual_z=residual_z,
                percentile=percentile[symbol],
                dispersion_ratio=dispersion_ratio,
            )
        )
        residual_vol_by_symbol[symbol] = residual_volatility
        failed_continuation_by_symbol[symbol] = failed_continuation_r
        sector_return_by_symbol[symbol] = sector_return_5d
        opportunities.append(
            DailyResidualOpportunity(
                symbol=symbol,
                issuer=issuer_key(symbol),
                sector=research_symbol.sector,
                side="long",
                residual_z=residual_z,
                remaining_room_r=features.residual_normalization_room,
                features=features,
                failed_continuation_r=failed_continuation_r,
                sector_return_5d=sector_return_5d,
                cost_feasible=adv >= 50_000_000.0,
                data_ready=(
                    len(research_symbol.daily_bars) >= 62
                    and research_symbol.daily_bars[-1].trade_date
                    < snapshot.trade_date
                ),
            )
        )
    return PreparedDailyResidualSelection(
        trade_date=snapshot.trade_date,
        factor_model=factor_model,
        formation_sessions=formation,
        market_trend_z_20d=float(market_trend_z_20d),
        opportunities=tuple(opportunities),
        frozen_model_by_symbol=frozen_model_by_symbol,
        percentile_by_symbol=percentile,
        residual_volatility_by_symbol=residual_vol_by_symbol,
        failed_continuation_by_symbol=failed_continuation_by_symbol,
        sector_return_by_symbol=sector_return_by_symbol,
    )


def build_daily_residual_artifact(
    snapshot: ResearchSnapshot,
    settings: StrategySettings,
    regime: RegimeSnapshot,
    *,
    precomputed_residuals: Mapping[str, Mapping[object, float]] | None = None,
    precomputed_models: Mapping[
        str, Mapping[object, FrozenResidualModel]
    ] | None = None,
    frozen_history_cache: dict[
        tuple[object, ...], dict[object, float]
    ] | None = None,
    precomputed_stock_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
    precomputed_reference_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
    prepared_selection: PreparedDailyResidualSelection | None = None,
) -> WatchlistArtifact:
    """Build the frozen-98 next-open residual artifact."""

    formation = int(settings.daily_residual_formation_sessions)
    factor_model = str(settings.daily_residual_factor_model)
    prepared = prepared_selection or prepare_daily_residual_selection(
        snapshot,
        factor_model=factor_model,
        formation_sessions=formation,
        precomputed_residuals=precomputed_residuals,
        precomputed_models=precomputed_models,
        frozen_history_cache=frozen_history_cache,
        precomputed_stock_returns=precomputed_stock_returns,
        precomputed_reference_returns=precomputed_reference_returns,
    )
    expected_contract = (snapshot.trade_date, factor_model, formation)
    actual_contract = (
        prepared.trade_date,
        prepared.factor_model,
        prepared.formation_sessions,
    )
    if actual_contract != expected_contract:
        raise ValueError(
            "prepared daily residual selection contract does not match snapshot/settings"
        )
    market_trend_z_20d = prepared.market_trend_z_20d
    market_regime_feasible = market_trend_z_20d >= float(
        settings.daily_residual_minimum_market_trend_z_20d
    )
    opportunities = [
        replace(
            opportunity,
            regime_feasible=(
                market_regime_feasible
                and opportunity.sector_return_5d
                >= float(settings.daily_residual_minimum_sector_return_5d)
            ),
        )
        for opportunity in prepared.opportunities
    ]
    frozen_model_by_symbol = prepared.frozen_model_by_symbol
    percentile = prepared.percentile_by_symbol
    residual_vol_by_symbol = prepared.residual_volatility_by_symbol
    failed_continuation_by_symbol = prepared.failed_continuation_by_symbol
    sector_return_by_symbol = prepared.sector_return_by_symbol
    held_directives = _build_residual_held_directives(
        snapshot=snapshot,
        residuals={},
        settings=settings,
        precomputed_stock_returns=precomputed_stock_returns,
        precomputed_reference_returns=precomputed_reference_returns,
        frozen_history_cache=frozen_history_cache,
    )
    # A causal full-exit directive is known before the next open and replay/live
    # both execute exits before entries.  Such a position must not reserve a
    # portfolio or sector slot for the same open.  Keep its issuer blocked to
    # avoid sell/rebuy churn, while allowing the released slot to be filled by
    # the next-ranked independent opportunity.
    capacity_held = [
        position
        for position in held_directives
        if position.residual_pending_action != "full_exit"
    ]
    releasing_issuers = [
        issuer_key(position.symbol)
        for position in held_directives
        if position.residual_pending_action == "full_exit"
    ]
    active_issuers = [issuer_key(position.symbol) for position in capacity_held]
    active_sectors = [
        snapshot.symbols[position.symbol].sector
        for position in capacity_held
        if position.symbol in snapshot.symbols
    ]
    components = tuple(settings.daily_residual_score_components)
    score_weights = {name: 1.0 for name in components}
    ranking_components = tuple(
        settings.daily_residual_ranking_score_components or components
    )
    ranking_score_weights = {name: 1.0 for name in ranking_components}
    ranked = rank_daily_residual_opportunities(
        opportunities,
        max_positions=int(settings.daily_residual_max_positions),
        max_positions_per_sector=int(
            settings.daily_residual_max_positions_per_sector
        ),
        sector_overflow_slots=int(
            settings.daily_residual_sector_overflow_slots
        ),
        sector_overflow_minimum_score=float(
            settings.daily_residual_sector_overflow_minimum_score
        ),
        sector_overflow_minimum_z=float(
            settings.daily_residual_sector_overflow_minimum_z
        ),
        active_issuers=active_issuers,
        active_sectors=active_sectors,
        blocked_issuers=releasing_issuers,
        minimum_residual_z=float(settings.daily_residual_minimum_z),
        minimum_score=float(settings.daily_residual_minimum_score),
        minimum_failed_continuation_r=float(
            settings.daily_residual_minimum_failed_continuation_r
        ),
        score_weights=score_weights,
        ranking_score_weights=ranking_score_weights,
    )
    replacement_decisions = ()
    if settings.daily_residual_replacement_mode != "disabled":
        incumbents: list[DailyResidualReplacementIncumbent] = []
        for held in capacity_held:
            symbol = snapshot.symbols.get(held.symbol)
            if symbol is None or not symbol.daily_bars or held.initial_r <= 0.0:
                continue
            completed_close = float(symbol.daily_bars[-1].close)
            dislocation = max(
                abs(float(held.residual_initial_dislocation_r)), 1e-12
            )
            incumbents.append(
                DailyResidualReplacementIncumbent(
                    symbol=held.symbol,
                    issuer=issuer_key(held.issuer or held.symbol),
                    sector=held.sector or symbol.sector,
                    entry_score=float(held.residual_entry_score),
                    held_sessions=int(held.residual_held_sessions),
                    normalization_fraction=(
                        float(held.residual_cumulative_normalization_r)
                        / dislocation
                    ),
                    unrealized_r=(
                        completed_close - float(held.entry_price)
                    )
                    / float(held.initial_r),
                )
            )
        replacement_decisions = choose_capacity_neutral_replacements(
            opportunities,
            incumbents,
            ranked,
            mode=settings.daily_residual_replacement_mode,
            loss_only=settings.daily_residual_replacement_loss_only,
            minimum_held_sessions=(
                settings.daily_residual_replacement_minimum_held_sessions
            ),
            maximum_normalization_fraction=(
                settings.daily_residual_replacement_maximum_normalization_fraction
            ),
            minimum_score_margin=(
                settings.daily_residual_replacement_minimum_score_margin
            ),
            maximum_replacements=(
                settings.daily_residual_replacement_max_per_session
            ),
            max_positions=int(settings.daily_residual_max_positions),
            max_positions_per_sector=int(
                settings.daily_residual_max_positions_per_sector
            ),
            minimum_residual_z=float(settings.daily_residual_minimum_z),
            minimum_score=float(settings.daily_residual_minimum_score),
            minimum_failed_continuation_r=float(
                settings.daily_residual_minimum_failed_continuation_r
            ),
            score_weights=score_weights,
            ranking_score_weights=ranking_score_weights,
            blocked_issuers=releasing_issuers,
        )
        if replacement_decisions:
            by_symbol = {row.symbol: row for row in held_directives}
            for replacement in replacement_decisions:
                outgoing = by_symbol[replacement.incumbent_symbol]
                outgoing.residual_pending_action = "full_exit"
                outgoing.residual_pending_reason = (
                    "capacity_neutral_alpha_replacement"
                    f"|candidate={replacement.candidate_symbol}"
                    f"|blocker={replacement.blocker_kind}"
                )
                outgoing.residual_pending_exit_fraction = 1.0
            capacity_held = [
                position
                for position in held_directives
                if position.residual_pending_action != "full_exit"
            ]
            releasing_issuers = [
                issuer_key(position.symbol)
                for position in held_directives
                if position.residual_pending_action == "full_exit"
            ]
            ranked = rank_daily_residual_opportunities(
                opportunities,
                max_positions=int(settings.daily_residual_max_positions),
                max_positions_per_sector=int(
                    settings.daily_residual_max_positions_per_sector
                ),
                sector_overflow_slots=int(
                    settings.daily_residual_sector_overflow_slots
                ),
                sector_overflow_minimum_score=float(
                    settings.daily_residual_sector_overflow_minimum_score
                ),
                sector_overflow_minimum_z=float(
                    settings.daily_residual_sector_overflow_minimum_z
                ),
                active_issuers=[
                    issuer_key(position.symbol) for position in capacity_held
                ],
                active_sectors=[
                    snapshot.symbols[position.symbol].sector
                    for position in capacity_held
                    if position.symbol in snapshot.symbols
                ],
                blocked_issuers=releasing_issuers,
                minimum_residual_z=float(settings.daily_residual_minimum_z),
                minimum_score=float(settings.daily_residual_minimum_score),
                minimum_failed_continuation_r=float(
                    settings.daily_residual_minimum_failed_continuation_r
                ),
                score_weights=score_weights,
                ranking_score_weights=ranking_score_weights,
                required_symbols=[
                    row.candidate_symbol for row in replacement_decisions
                ],
            )
            selected_symbols = {
                row.opportunity.symbol for row in ranked
            }
            missing = [
                row.candidate_symbol
                for row in replacement_decisions
                if row.candidate_symbol not in selected_symbols
            ]
            if missing:
                raise RuntimeError(
                    "capacity-neutral replacement failed to admit its causal "
                    f"candidate: {missing}"
                )
    items: list[WatchlistItem] = []
    for selected in ranked:
        opportunity = selected.opportunity
        symbol = snapshot.symbols[opportunity.symbol]
        frozen_model = frozen_model_by_symbol[opportunity.symbol]
        latest = symbol.daily_bars[-1]
        # Management is expressed in units of one-session residual volatility,
        # so undo the formation-horizon sqrt scaling used by the z score.
        dislocation = abs(float(opportunity.residual_z)) * sqrt(float(formation))
        score_components = opportunity.features.as_mapping()
        items.append(
            WatchlistItem(
                symbol=symbol.symbol,
                exchange=symbol.exchange,
                primary_exchange=symbol.primary_exchange,
                currency=symbol.currency,
                tick_size=symbol.tick_size,
                point_value=symbol.point_value,
                sector=symbol.sector,
                regime_score=regime.score,
                regime_tier=regime.tier,
                regime_risk_multiplier=regime.risk_multiplier,
                sector_score=0.0,
                sector_rank_weight=1.0,
                sponsorship_score=0.0,
                sponsorship_state="RESIDUAL",
                persistence=score_components["shock_freshness"],
                intensity_z=-float(opportunity.residual_z),
                accel_z=0.0,
                rs_percentile=(1.0 - percentile[opportunity.symbol]) * 100.0,
                leader_pass=True,
                trend_pass=True,
                trend_strength=0.0,
                earnings_risk_flag=False,
                blacklist_flag=symbol.blacklist_flag,
                anchor_date=latest.trade_date,
                anchor_type="FROZEN_FACTOR_RESIDUAL",
                acceptance_pass=True,
                avwap_ref=float(latest.close),
                avwap_band_lower=float(latest.close),
                avwap_band_upper=float(latest.close),
                daily_atr_estimate=symbol.daily_atr_estimate,
                intraday_atr_seed=symbol.intraday_atr_seed,
                daily_rank=float(selected.rank),
                tradable_flag=True,
                conviction_bucket="RESIDUAL_RANKED",
                conviction_multiplier=1.0,
                recommended_risk_r=float(settings.daily_residual_risk_fraction)
                * (
                    float(
                        settings.daily_residual_sector_overflow_risk_multiplier
                    )
                    if selected.sector_overflow
                    else 1.0
                ),
                daily_signal_score=selected.decision.score,
                entry_rank=selected.rank,
                entry_rank_pct=selected.rank / max(len(ranked), 1) * 100.0,
                previous_close=float(latest.close),
                sleeve_id=DAILY_RESIDUAL_SLEEVE,
                residual_factor_model=factor_model,
                residual_formation_sessions=formation,
                residual_z=float(opportunity.residual_z),
                residual_volatility=residual_vol_by_symbol[opportunity.symbol],
                residual_initial_dislocation_r=dislocation,
                residual_anchor_price=float(latest.close)
                * exp(dislocation * residual_vol_by_symbol[opportunity.symbol]),
                residual_remaining_room_r=float(opportunity.remaining_room_r),
                residual_score_components=score_components,
                residual_admission_score=float(
                    selected.decision.admission_score
                    if selected.decision.admission_score is not None
                    else selected.decision.score
                ),
                residual_ranking_score=float(selected.decision.score),
                residual_failed_continuation_r=failed_continuation_by_symbol[
                    opportunity.symbol
                ],
                residual_sector_return_5d=sector_return_by_symbol[
                    opportunity.symbol
                ],
                residual_lane_id=str(settings.daily_residual_lane_id),
                residual_model_contract_version=frozen_model.contract_version,
                residual_model_intercept=frozen_model.intercept,
                residual_factor_names=frozen_model.factor_names,
                residual_factor_betas=frozen_model.factor_betas,
                residual_peer_symbols=frozen_model.peer_symbols,
                residual_model_estimation_session=frozen_model.estimation_session,
                entry_clock="next_session_open",
            )
        )
    return WatchlistArtifact(
        trade_date=snapshot.trade_date,
        generated_at=datetime.now(timezone.utc),
        regime=regime,
        items=items,
        tradable=list(items),
        overflow=[],
        market_wide_institutional_selling=False,
        held_positions=held_directives,
        strategy_mode="daily_residual_reversion",
        selection_contract_version="daily_residual_shared_selector_v5",
        strategy_parameters={
            "factor_model": factor_model,
            "formation_sessions": formation,
            "minimum_z": float(settings.daily_residual_minimum_z),
            "minimum_score": float(settings.daily_residual_minimum_score),
            "minimum_failed_continuation_r": float(
                settings.daily_residual_minimum_failed_continuation_r
            ),
            "minimum_sector_return_5d": float(
                settings.daily_residual_minimum_sector_return_5d
            ),
            "minimum_market_trend_z_20d": float(
                settings.daily_residual_minimum_market_trend_z_20d
            ),
            "market_trend_z_20d": float(market_trend_z_20d),
            "lane_id": str(settings.daily_residual_lane_id),
            "score_components": list(components),
            "ranking_score_components": list(ranking_components),
            "max_positions": int(settings.daily_residual_max_positions),
            "max_positions_per_sector": int(
                settings.daily_residual_max_positions_per_sector
            ),
            "sector_overflow_slots": int(
                settings.daily_residual_sector_overflow_slots
            ),
            "sector_overflow_minimum_score": float(
                settings.daily_residual_sector_overflow_minimum_score
            ),
            "sector_overflow_minimum_z": float(
                settings.daily_residual_sector_overflow_minimum_z
            ),
            "sector_overflow_risk_multiplier": float(
                settings.daily_residual_sector_overflow_risk_multiplier
            ),
            "risk_fraction": float(settings.daily_residual_risk_fraction),
            "maximum_notional_fraction": float(
                settings.daily_residual_maximum_notional_fraction
            ),
            "catastrophic_stop_atr": float(
                settings.daily_residual_catastrophic_stop_atr
            ),
            "catastrophic_stop_residual_r": float(
                settings.daily_residual_catastrophic_stop_residual_r
            ),
            "partial_normalization_fraction": float(
                settings.daily_residual_partial_normalization_fraction
            ),
            "full_normalization_fraction": float(
                settings.daily_residual_full_normalization_fraction
            ),
            "structural_failure_extension_fraction": float(
                settings.daily_residual_structural_failure_extension_fraction
            ),
            "profit_retention_activation_fraction": float(
                settings.daily_residual_profit_retention_activation_fraction
            ),
            "profit_retention_giveback_fraction": float(
                settings.daily_residual_profit_retention_giveback_fraction
            ),
            "replacement_mode": settings.daily_residual_replacement_mode,
            "replacement_loss_only": bool(
                settings.daily_residual_replacement_loss_only
            ),
            "replacement_minimum_held_sessions": int(
                settings.daily_residual_replacement_minimum_held_sessions
            ),
            "replacement_maximum_normalization_fraction": float(
                settings.daily_residual_replacement_maximum_normalization_fraction
            ),
            "replacement_minimum_score_margin": float(
                settings.daily_residual_replacement_minimum_score_margin
            ),
            "replacement_max_per_session": int(
                settings.daily_residual_replacement_max_per_session
            ),
            "capacity_neutral_replacements": [
                {
                    "incumbent_symbol": row.incumbent_symbol,
                    "candidate_symbol": row.candidate_symbol,
                    "blocker_kind": row.blocker_kind,
                    "incumbent_entry_score": row.incumbent_entry_score,
                    "candidate_score": row.candidate_score,
                    "score_margin": row.score_margin,
                    "incumbent_held_sessions": row.incumbent_held_sessions,
                    "incumbent_normalization_fraction": (
                        row.incumbent_normalization_fraction
                    ),
                    "incumbent_unrealized_r": row.incumbent_unrealized_r,
                }
                for row in replacement_decisions
            ],
            "maximum_holding_sessions": int(
                settings.daily_residual_maximum_holding_sessions
            ),
            "partial_exit_fraction": float(
                settings.daily_residual_partial_exit_fraction
            ),
            "entry_clock": "next_session_open",
            "universe_contract": "frozen_98_intraday_symbols_only",
        },
    )


def _build_residual_held_directives(
    *,
    snapshot: ResearchSnapshot,
    residuals: Mapping[str, Mapping[object, float]],
    settings: StrategySettings,
    precomputed_stock_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
    precomputed_reference_returns: Mapping[
        str, Mapping[object, float]
    ] | None = None,
    frozen_history_cache: dict[
        tuple[object, ...], dict[object, float]
    ] | None = None,
) -> list[HeldPositionDirective]:
    """Advance held residual positions using each newly completed session.

    The state is frozen at entry and persisted by the execution adapter.  A
    missed nightly run can consume multiple completed sessions deterministically
    without using any bar from the session about to open.
    """

    residual_held = [
        held
        for held in snapshot.held_positions
        if held.sleeve_id == DAILY_RESIDUAL_SLEEVE
    ]
    if not residual_held:
        return []
    policy = ResidualManagementPolicy(
        partial_normalization_fraction=float(
            settings.daily_residual_partial_normalization_fraction
        ),
        full_normalization_fraction=float(
            settings.daily_residual_full_normalization_fraction
        ),
        structural_failure_extension_fraction=float(
            settings.daily_residual_structural_failure_extension_fraction
        ),
        profit_retention_activation_fraction=float(
            settings.daily_residual_profit_retention_activation_fraction
        ),
        profit_retention_giveback_fraction=float(
            settings.daily_residual_profit_retention_giveback_fraction
        ),
        maximum_holding_sessions=int(settings.daily_residual_maximum_holding_sessions),
        partial_exit_fraction=float(settings.daily_residual_partial_exit_fraction),
    )
    stock_returns = (
        precomputed_stock_returns
        if precomputed_stock_returns is not None
        else {
            symbol: _returns(item.daily_bars)
            for symbol, item in snapshot.symbols.items()
            if symbol in BACKTESTED_INTRADAY_STOCK_SYMBOLS
        }
    )
    reference_returns = (
        precomputed_reference_returns
        if precomputed_reference_returns is not None
        else {
            symbol: _returns(bars)
            for symbol, bars in snapshot.reference_daily_bars.items()
        }
    )
    directives: list[HeldPositionDirective] = []
    for held in residual_held:
        if held.residual_initial_dislocation_r <= 0.0:
            raise ValueError(
                f"held residual position {held.symbol} lacks frozen dislocation state"
            )
        if held.residual_model_contract_version != "frozen_residual_model_v2":
            raise ValueError(
                f"held residual position {held.symbol} lacks a frozen v2 model"
            )
        model = FrozenResidualModel(
            factor_names=tuple(held.residual_factor_names),
            intercept=float(held.residual_model_intercept),
            factor_betas=tuple(held.residual_factor_betas),
            peer_symbols=tuple(held.residual_peer_symbols),
            estimation_session=held.residual_model_estimation_session,
            contract_version=held.residual_model_contract_version,
        )
        history = _frozen_model_residual_history(
            symbol=held.symbol,
            sector=held.sector or snapshot.symbols[held.symbol].sector,
            model=model,
            stock_returns=stock_returns,
            reference_returns=reference_returns,
            cache=frozen_history_cache,
        )
        state = ResidualManagementState(
            initial_dislocation_r=float(held.residual_initial_dislocation_r),
            cumulative_normalization_r=float(
                held.residual_cumulative_normalization_r
            ),
            peak_normalization_r=float(
                getattr(
                    held,
                    "residual_peak_normalization_r",
                    max(0.0, held.residual_cumulative_normalization_r),
                )
            ),
            held_sessions=int(held.residual_held_sessions),
            partial_taken=bool(held.residual_partial_taken),
        )
        last_processed = held.residual_last_processed_session
        pending_action = "hold"
        pending_reason = "await_residual_normalization"
        pending_fraction = 0.0
        for session, value in sorted(history.items()):
            if not isfinite(float(value)):
                continue
            if session >= snapshot.trade_date:
                continue
            if last_processed is not None and session <= last_processed:
                continue
            if session < held.entry_time.date():
                continue
            volatility = float(held.residual_volatility)
            if volatility <= 0.0:
                raise ValueError(
                    f"held residual position {held.symbol} lacks entry residual volatility"
                )
            decision = advance_residual_management(
                state,
                incremental_normalization_r=float(value) / volatility,
                policy=policy,
            )
            state = decision.next_state
            last_processed = session
            pending_action = decision.action
            pending_reason = decision.reason
            pending_fraction = decision.exit_fraction
            if decision.action == "full_exit":
                break
        directives.append(
            HeldPositionDirective(
                symbol=held.symbol,
                entry_time=held.entry_time,
                entry_price=held.entry_price,
                size=held.size,
                stop=held.stop,
                initial_r=held.initial_r,
                setup_tag=held.setup_tag,
                time_stop_deadline=None,
                carry_eligible_flag=False,
                flow_reversal_flag=False,
                issuer=issuer_key(held.issuer or held.symbol),
                sector=held.sector or snapshot.symbols[held.symbol].sector,
                exchange=snapshot.symbols[held.symbol].exchange,
                primary_exchange=snapshot.symbols[held.symbol].primary_exchange,
                currency=snapshot.symbols[held.symbol].currency,
                tick_size=snapshot.symbols[held.symbol].tick_size,
                point_value=snapshot.symbols[held.symbol].point_value,
                sleeve_id=DAILY_RESIDUAL_SLEEVE,
                residual_factor_model=held.residual_factor_model,
                residual_formation_sessions=held.residual_formation_sessions,
                residual_volatility=held.residual_volatility,
                residual_initial_dislocation_r=state.initial_dislocation_r,
                residual_cumulative_normalization_r=state.cumulative_normalization_r,
                residual_peak_normalization_r=state.peak_normalization_r,
                residual_held_sessions=state.held_sessions,
                residual_partial_taken=state.partial_taken,
                residual_last_processed_session=last_processed,
                residual_pending_action=pending_action,
                residual_pending_reason=pending_reason,
                residual_pending_exit_fraction=pending_fraction,
                residual_qty_entry=(held.residual_qty_entry or held.size),
                residual_entry_commission=held.residual_entry_commission,
                residual_exit_commission=held.residual_exit_commission,
                residual_realized_pnl_usd=held.residual_realized_pnl_usd,
                residual_entry_score=held.residual_entry_score,
                residual_trade_id=held.residual_trade_id,
                residual_protective_stop_client_order_id=(
                    held.residual_protective_stop_client_order_id
                ),
                residual_protective_stop_price=held.residual_protective_stop_price,
                residual_protective_stop_qty=held.residual_protective_stop_qty,
                residual_lane_id=held.residual_lane_id,
                residual_model_contract_version=held.residual_model_contract_version,
                residual_model_intercept=held.residual_model_intercept,
                residual_factor_names=held.residual_factor_names,
                residual_factor_betas=held.residual_factor_betas,
                residual_peer_symbols=held.residual_peer_symbols,
                residual_model_estimation_session=(
                    held.residual_model_estimation_session
                ),
            )
        )
    return directives
