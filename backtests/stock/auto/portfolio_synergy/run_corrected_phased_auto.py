from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)
from backtests.stock.auto.portfolio_synergy.core.logic import (
    CURRENT_ALCB_ID,
    CURRENT_IARIC_ID,
    run_portfolio_replay,
)
from backtests.stock.auto.portfolio_synergy.core.market import CausalPriceBook
from backtests.stock.auto.portfolio_synergy.evaluator import (
    _load_stock_price_bars,
    load_trade_records,
)
from backtests.stock.auto.portfolio_synergy.run_current_rebaseline import (
    CURRENT_ALCB_CONFIG,
    CURRENT_ALCB_TRADES,
    CURRENT_IARIC_CONFIG,
    CURRENT_IARIC_TRADES,
    DATA_DIR,
    REPO_ROOT,
    _load_daily_panel_unsealed,
    _parity_receipt,
    _residual_records,
    _run_alcb,
)
from backtests.stock.models import TradeRecord
from strategies.stock.iaric.config import StrategySettings


START = date(2024, 3, 25)
IS_END = date(2026, 3, 1)
OOS_START = date(2026, 3, 2)
OOS_END = date(2026, 5, 1)
INITIAL_EQUITY = 25_000.0
REFERENCE_RISK_PCT = 0.0025
NATIVE_IARIC_RISK = 0.002375
NATIVE_ALCB_RISK = 0.00702
EMBARGO_DAYS = 14
MAX_INTERACTION_PBO = 0.25
MIN_INTERACTION_ROBUST_SCORE_GAIN = 0.005

FOLDS = (
    ("f1", date(2024, 3, 25), date(2024, 6, 30)),
    ("f2", date(2024, 7, 1), date(2024, 10, 31)),
    ("f3", date(2024, 11, 1), date(2025, 2, 28)),
    ("f4", date(2025, 3, 1), date(2025, 6, 30)),
    ("f5", date(2025, 7, 1), date(2025, 10, 31)),
    ("f6", date(2025, 11, 1), IS_END),
)

ALCB_FEATURES = (
    "intercept",
    "momentum_score",
    "log_rvol",
    "selection_score",
    "orb_quality",
    "signal_cpr",
    "signal_range_r",
    "gap_pct",
    "daily_adx",
    "daily_atr_pct",
    "breakout_distance_r",
    "relative_strength",
    "signal_minutes",
    "reentry_sequence",
    "route_pdh",
    "route_or",
    "route_combined",
    "regime_a",
    "regime_b",
    "sector_technology",
    "sector_financials",
    "sector_healthcare",
    "sector_consumer",
    "sector_industrials",
    "sector_other",
)

IARIC_FEATURES = (
    "intercept",
    "residual_score",
    "failed_continuation",
    "sector_return_5d",
    "formation_sessions",
    "sector_technology",
    "sector_financials",
    "sector_healthcare",
    "sector_consumer",
    "sector_industrials",
    "sector_other",
)


@dataclass(frozen=True)
class AlphaPrediction:
    expected_r: float
    uncertainty: float


@dataclass(frozen=True)
class StrategyStreams:
    alcb: tuple[TradeRecord, ...]
    iaric: tuple[TradeRecord, ...]


def _status(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, default=str), flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stable_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filter(
    trades: Iterable[TradeRecord], start: date, end: date
) -> tuple[TradeRecord, ...]:
    # Purge positions whose outcomes cross a fold/split boundary.  This prevents
    # a pre-boundary IARIC entry from importing post-boundary PnL into selection.
    return tuple(
        trade
        for trade in trades
        if start <= trade.entry_time.date() <= end and trade.exit_time.date() <= end
    )


def _stream_sha(trades: Iterable[TradeRecord]) -> str:
    rows = [
        (
            trade.strategy,
            trade.symbol,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            round(float(trade.r_multiple), 12),
            round(float(trade.entry_price), 8),
            round(float(trade.exit_price), 8),
        )
        for trade in trades
    ]
    return _stable_sha(rows)


def _meta_float(trade: TradeRecord, key: str, default: float = 0.0) -> float:
    try:
        return float((trade.metadata or {}).get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _sector_features(sector: str) -> dict[str, float]:
    normalized = sector.strip().lower()
    result = {
        "sector_technology": 0.0,
        "sector_financials": 0.0,
        "sector_healthcare": 0.0,
        "sector_consumer": 0.0,
        "sector_industrials": 0.0,
        "sector_other": 0.0,
    }
    if "tech" in normalized or "communication" in normalized:
        result["sector_technology"] = 1.0
    elif "financial" in normalized:
        result["sector_financials"] = 1.0
    elif "health" in normalized:
        result["sector_healthcare"] = 1.0
    elif "consumer" in normalized:
        result["sector_consumer"] = 1.0
    elif "industrial" in normalized or "material" in normalized or "energy" in normalized:
        result["sector_industrials"] = 1.0
    else:
        result["sector_other"] = 1.0
    return result


def _feature_vector(strategy: str, trade: TradeRecord) -> np.ndarray:
    sector = _sector_features(trade.sector or "")
    if strategy == CURRENT_ALCB_ID:
        route = (trade.entry_type or "").upper()
        regime = (trade.regime_tier or "").upper()
        values = {
            "intercept": 1.0,
            "momentum_score": _meta_float(trade, "momentum_score", 5.0),
            "log_rvol": math.log1p(max(_meta_float(trade, "entry_signal_rvol", _meta_float(trade, "rvol_at_entry", 1.0)), 0.0)),
            "selection_score": _meta_float(trade, "selection_score", 0.0),
            "orb_quality": _meta_float(trade, "orb_quality_score", 50.0) / 100.0,
            "signal_cpr": _meta_float(trade, "signal_cpr", 0.5),
            "signal_range_r": _meta_float(trade, "signal_range_r", 0.0),
            "gap_pct": _meta_float(trade, "gap_pct", 0.0),
            "daily_adx": _meta_float(trade, "daily_adx", 0.0) / 100.0,
            "daily_atr_pct": _meta_float(trade, "daily_atr_pct", 0.0),
            "breakout_distance_r": _meta_float(trade, "breakout_distance_r", 0.0),
            "relative_strength": _meta_float(trade, "relative_strength_percentile", 0.5),
            "signal_minutes": _meta_float(trade, "signal_minutes_et", 600.0) / 1_000.0,
            "reentry_sequence": float(trade.reentry_sequence or 0),
            "route_pdh": float(route == "PDH_BREAKOUT"),
            "route_or": float(route == "OR_BREAKOUT"),
            "route_combined": float(route == "COMBINED_BREAKOUT"),
            "regime_a": float(regime == "A"),
            "regime_b": float(regime == "B"),
            **sector,
        }
        return np.asarray([values[name] for name in ALCB_FEATURES], dtype=float)

    values = {
        "intercept": 1.0,
        "residual_score": _meta_float(trade, "residual_score", 0.0) / 100.0,
        "failed_continuation": _meta_float(trade, "failed_continuation_r", 0.0) / 2.0,
        "sector_return_5d": _meta_float(trade, "sector_return_5d", 0.0),
        "formation_sessions": _meta_float(trade, "formation_sessions", 1.0) / 5.0,
        **sector,
    }
    return np.asarray([values[name] for name in IARIC_FEATURES], dtype=float)


def _fit_predict_ridge(
    training: list[TradeRecord],
    targets: list[TradeRecord],
    *,
    ridge: float | None,
) -> list[AlphaPrediction]:
    if not targets:
        return []
    if len(training) < 40:
        return [AlphaPrediction(0.0, 1.0) for _ in targets]

    y_raw = np.asarray([trade.r_multiple for trade in training], dtype=float)
    y = np.clip(y_raw, -2.0, 3.0)
    prior = float(np.mean(y)) * len(y) / (len(y) + 75.0)
    residual_scale = max(float(np.std(y, ddof=1)), 0.10)
    if ridge is None:
        uncertainty = residual_scale / math.sqrt(max(len(y), 1))
        return [
            AlphaPrediction(float(np.clip(prior, -0.50, 0.75)), uncertainty)
            for _ in targets
        ]

    strategy = CURRENT_ALCB_ID if training[0].entry_type != "DAILY_RESIDUAL_REVERSION" else CURRENT_IARIC_ID
    x = np.vstack([_feature_vector(strategy, trade) for trade in training])
    x_target = np.vstack([_feature_vector(strategy, trade) for trade in targets])
    mean = np.mean(x[:, 1:], axis=0)
    std = np.std(x[:, 1:], axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    x[:, 1:] = (x[:, 1:] - mean) / std
    x_target[:, 1:] = (x_target[:, 1:] - mean) / std
    penalty = np.eye(x.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    gram = x.T @ x + penalty
    inverse = np.linalg.pinv(gram)
    beta = inverse @ x.T @ y
    fitted = x @ beta
    sigma = max(float(np.std(y - fitted, ddof=1)), 0.10)
    raw_predictions = x_target @ beta
    predictions: list[AlphaPrediction] = []
    for vector, estimate in zip(x_target, raw_predictions):
        leverage = max(float(vector @ inverse @ vector), 0.0)
        uncertainty = sigma * math.sqrt(min(1.0 + leverage, 4.0))
        shrunk = 0.75 * float(estimate) + 0.25 * prior
        predictions.append(
            AlphaPrediction(
                expected_r=float(np.clip(shrunk, -0.50, 0.75)),
                uncertainty=float(np.clip(uncertainty, 0.10, 2.0)),
            )
        )
    return predictions


def _month_start(value: datetime) -> date:
    return date(value.year, value.month, 1)


def _expanding_predictions(
    trades: tuple[TradeRecord, ...],
    *,
    ridge: float | None,
    train_end: date,
) -> list[AlphaPrediction]:
    predictions = [AlphaPrediction(0.0, 1.0) for _ in trades]
    indexed = sorted(enumerate(trades), key=lambda row: row[1].entry_time)
    by_month: dict[date, list[tuple[int, TradeRecord]]] = defaultdict(list)
    for index, trade in indexed:
        by_month[_month_start(trade.entry_time)].append((index, trade))

    for month, rows in sorted(by_month.items()):
        cutoff = month - timedelta(days=EMBARGO_DAYS)
        training = [
            trade
            for trade in trades
            if trade.exit_time.date() < cutoff
            and trade.entry_time.date() <= train_end
            and trade.exit_time.date() <= train_end
        ]
        month_predictions = _fit_predict_ridge(
            training,
            [trade for _, trade in rows],
            ridge=ridge,
        )
        for (index, _trade), prediction in zip(rows, month_predictions):
            predictions[index] = prediction
    return predictions


def _attach_predictions(
    trades: tuple[TradeRecord, ...], predictions: list[AlphaPrediction]
) -> tuple[TradeRecord, ...]:
    attached = []
    for trade, prediction in zip(trades, predictions):
        metadata = dict(trade.metadata or {})
        metadata.update(
            {
                "portfolio_expected_r": prediction.expected_r,
                "portfolio_expected_r_uncertainty": prediction.uncertainty,
            }
        )
        attached.append(replace(trade, metadata=metadata))
    return tuple(attached)


def _alpha_model_diagnostics(
    streams: StrategyStreams,
    variants: dict[str, float | None],
) -> tuple[str, dict[str, Any], dict[str, StrategyStreams]]:
    diagnostics: dict[str, Any] = {}
    predicted_streams: dict[str, StrategyStreams] = {}
    for name, ridge in variants.items():
        alcb_predictions = _expanding_predictions(
            streams.alcb,
            ridge=ridge,
            train_end=IS_END,
        )
        iaric_predictions = _expanding_predictions(
            streams.iaric,
            ridge=ridge,
            train_end=IS_END,
        )
        attached = StrategyStreams(
            _attach_predictions(streams.alcb, alcb_predictions),
            _attach_predictions(streams.iaric, iaric_predictions),
        )
        predicted_streams[name] = attached
        fold_rows = []
        positive_lifts = 0
        lifts: list[float] = []
        for fold_name, start, end in FOLDS:
            for strategy, trades in (
                (CURRENT_ALCB_ID, attached.alcb),
                (CURRENT_IARIC_ID, attached.iaric),
            ):
                rows = [
                    trade
                    for trade in trades
                    if start <= trade.entry_time.date() <= end
                ]
                if len(rows) < 20:
                    continue
                forecasts = np.asarray(
                    [_meta_float(trade, "portfolio_expected_r") for trade in rows],
                    dtype=float,
                )
                if float(np.std(forecasts)) < 1e-10:
                    lift = 0.0
                    top = bottom = float(np.mean([trade.r_multiple for trade in rows]))
                    lifts.append(lift)
                    fold_rows.append(
                        {
                            "fold": fold_name,
                            "strategy": strategy,
                            "count": len(rows),
                            "top_bottom_lift_R": lift,
                            "top_quartile_R": top,
                            "bottom_quartile_R": bottom,
                        }
                    )
                    continue
                ordered = sorted(
                    rows,
                    key=lambda trade: _meta_float(trade, "portfolio_expected_r"),
                )
                size = max(len(ordered) // 4, 1)
                bottom = float(np.mean([trade.r_multiple for trade in ordered[:size]]))
                top = float(np.mean([trade.r_multiple for trade in ordered[-size:]]))
                lift = top - bottom
                lifts.append(lift)
                positive_lifts += int(lift > 0.0)
                fold_rows.append(
                    {
                        "fold": fold_name,
                        "strategy": strategy,
                        "count": len(rows),
                        "top_bottom_lift_R": lift,
                        "top_quartile_R": top,
                        "bottom_quartile_R": bottom,
                    }
                )
        median_lift = float(np.median(lifts)) if lifts else 0.0
        lower_lift = float(np.quantile(lifts, 0.25)) if lifts else 0.0
        stable = positive_lifts >= max(7, math.ceil(0.60 * len(lifts))) and median_lift > 0.0
        diagnostics[name] = {
            "ridge": ridge,
            "folds": fold_rows,
            "positive_lifts": positive_lifts,
            "comparisons": len(lifts),
            "median_top_bottom_lift_R": median_lift,
            "lower_quartile_lift_R": lower_lift,
            "stable": stable,
            "selection_score": median_lift + 0.35 * lower_lift,
        }
    eligible = [name for name, row in diagnostics.items() if row["stable"]]
    selected = max(
        eligible or ["constant"],
        key=lambda name: float(diagnostics[name]["selection_score"]),
    )
    return selected, diagnostics, predicted_streams


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = deepcopy(value)


def _patched(config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for path, value in patch.items():
        _set_path(result, path, value)
    return result


def baseline_config() -> dict[str, Any]:
    return {
        "initial_equity": INITIAL_EQUITY,
        "risk_stance": "aggressive_bounded_realism_round6",
        "account_rules": {
            "account_type": "reg_t_margin",
            "enforce_shared_buying_power": True,
            "allow_fractional_shares": False,
            "oversize_action": "resize",
            "max_gross_notional_pct": 1.75,
            "max_net_notional_pct": 1.65,
            "max_overnight_gross_notional_pct": 1.15,
            "max_symbol_notional_pct": 0.20,
            "max_position_notional_pct": 0.20,
            "initial_margin_long_pct": 0.50,
            "initial_margin_short_pct": 0.60,
            "maintenance_margin_long_pct": 0.25,
            "maintenance_margin_short_pct": 0.30,
            "minimum_margin_buffer_pct": 0.10,
            "annual_margin_interest_rate": 0.08,
            "annual_cash_interest_rate": 0.00,
        },
        "portfolio_rules": {
            "reference_risk_pct": REFERENCE_RISK_PCT,
            "heat_cap_R": 99.0,
            "max_total_active_positions": 99,
            "max_symbol_heat_R": 99.0,
            "max_long_heat_R": 99.0,
            "portfolio_daily_stop_R": 0.0,
            "portfolio_weekly_stop_R": 0.0,
            "max_single_strategy_trade_share": 1.0,
            "max_single_strategy_risk_share": 1.0,
            "drawdown_tiers": ((1.0, 1.0),),
        },
        "strategy_allocations": {
            CURRENT_IARIC_ID: {
                "unit_risk_pct": NATIVE_IARIC_RISK,
                "max_position_notional_pct": 0.18,
                "max_heat_R": 99.0,
                "max_concurrent": 99,
                "daily_stop_R": 0.0,
                "priority": 0,
                "role": "fixed Round-3 daily residual signal stream",
            },
            CURRENT_ALCB_ID: {
                "unit_risk_pct": NATIVE_ALCB_RISK,
                "max_position_notional_pct": 0.20,
                "max_heat_R": 99.0,
                "max_concurrent": 99,
                "daily_stop_R": 0.0,
                "priority": 0,
                "role": "fixed Round-3 intraday breakout signal stream",
            },
        },
        "dynamic_allocation": {
            "enabled": False,
            "lookback_trades": 60,
            "min_mult": 1.0,
            "max_mult": 1.0,
            "positive_expectancy_boost": 0.0,
            "negative_expectancy_cut": 0.0,
        },
        "cross_strategy_rules": {
            "apply_duplicate_native_limits": False,
            "candidate_rank_mode": "expected_net_r",
            "alpha_admission_enabled": False,
            "alpha_uncertainty_penalty": 0.0,
            "minimum_expected_r": -99.0,
            "capacity_action": "block",
            "minimum_capacity_size_mult": 0.35,
            "same_symbol_policy": "none",
            "same_symbol_size_mult": 1.0,
            "same_sector_heat_cap_R": 99.0,
            "intraday_reserved_slots": 0,
            "intraday_reserved_heat_R": 0.0,
        },
        "strategy_filters": {CURRENT_IARIC_ID: {}, CURRENT_ALCB_ID: {}},
    }


def _no_overlay(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["portfolio_rules"].update(
        {
            "heat_cap_R": 99.0,
            "max_total_active_positions": 99,
            "max_symbol_heat_R": 99.0,
            "max_long_heat_R": 99.0,
            "portfolio_daily_stop_R": 0.0,
            "portfolio_weekly_stop_R": 0.0,
            "drawdown_tiers": ((1.0, 1.0),),
        }
    )
    result["cross_strategy_rules"].update(
        {
            "apply_duplicate_native_limits": False,
            "alpha_admission_enabled": False,
            "capacity_action": "block",
            "same_symbol_policy": "none",
            "same_sector_heat_cap_R": 99.0,
            "intraday_reserved_slots": 0,
            "intraday_reserved_heat_R": 0.0,
        }
    )
    return result


def _max_drawdown(curve: np.ndarray) -> float:
    if len(curve) < 2:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    drawdowns = np.divide(peaks - curve, peaks, out=np.zeros_like(curve), where=peaks > 0)
    return float(np.max(drawdowns))


def _months(start: date, end: date) -> float:
    return max((end - start).days / 30.4375, 1.0)


def _daily_mtm_metrics(
    result,
    *,
    close: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, float]:
    positions = result.state.accepted_positions
    timeline = close.loc[start.isoformat() : end.isoformat()].index
    if len(timeline) == 0:
        return {
            "annual_log_growth": 0.0,
            "annualized_volatility": 0.0,
            "expected_shortfall_95": 0.0,
            "max_drawdown_pct_mtm_daily": 0.0,
            "certainty_equivalent_growth": 0.0,
            "peak_open_risk_pct": 0.0,
        }

    values: list[float] = []
    peak_open_risk_pct = 0.0
    for timestamp in timeline:
        day = timestamp.date()
        realized = INITIAL_EQUITY + sum(
            position.pnl for position in positions if position.exit_time.date() <= day
        )
        open_positions = [
            position
            for position in positions
            if position.entry_time.date() <= day < position.exit_time.date()
        ]
        unrealized = 0.0
        for position in open_positions:
            if position.symbol not in close.columns or timestamp not in close.index:
                continue
            raw_close = float(close.at[timestamp, position.symbol])
            if not np.isfinite(raw_close) or raw_close <= 0.0:
                continue
            ratio = raw_close / max(position.entry_price, 1e-9)
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                scale_power = round(math.log10(ratio))
                raw_close /= 10.0**scale_power
            unrealized += (
                (raw_close - position.entry_price)
                * float(position.direction)
                * position.quantity
            )
        equity = realized + unrealized
        values.append(equity)
        open_risk = sum(position.risk_dollars for position in open_positions)
        peak_open_risk_pct = max(
            peak_open_risk_pct,
            open_risk / max(equity, 1.0),
        )

    curve = np.asarray([INITIAL_EQUITY, *values], dtype=float)
    safe = np.maximum(curve, 1.0)
    daily_returns = np.diff(safe) / safe[:-1]
    log_returns = np.log(safe[1:] / safe[:-1])
    annual_log_growth = float(np.mean(log_returns) * 252.0) if len(log_returns) else 0.0
    annualized_volatility = (
        float(np.std(daily_returns, ddof=1) * math.sqrt(252.0))
        if len(daily_returns) >= 2
        else 0.0
    )
    tail_count = max(int(math.ceil(0.05 * len(daily_returns))), 1)
    expected_shortfall = (
        max(0.0, -float(np.mean(np.sort(daily_returns)[:tail_count])))
        if len(daily_returns)
        else 0.0
    )
    certainty_equivalent = annual_log_growth - 0.625 * annualized_volatility**2
    return {
        "annual_log_growth": annual_log_growth,
        "annualized_volatility": annualized_volatility,
        "expected_shortfall_95": expected_shortfall,
        "max_drawdown_pct_mtm_daily": _max_drawdown(curve),
        "certainty_equivalent_growth": certainty_equivalent,
        "peak_open_risk_pct": peak_open_risk_pct,
        "daily_points": float(len(values)),
    }


def _evaluate_window(
    streams: StrategyStreams,
    config: dict[str, Any],
    *,
    close: pd.DataFrame,
    price_book: CausalPriceBook,
    start: date,
    end: date,
) -> tuple[dict[str, float], Any]:
    alcb = _filter(streams.alcb, start, end)
    iaric = _filter(streams.iaric, start, end)
    result = run_portfolio_replay(
        alcb,
        iaric,
        config,
        mark_price_provider=price_book,
    )
    metrics = dict(result.metrics)
    metrics.update(_daily_mtm_metrics(result, close=close, start=start, end=end))
    blocked_total_r = sum(candidate.r_multiple for candidate in result.state.blocked_candidates)
    period_months = _months(start, end)
    # Position metadata carries the entry-time calibrated forecast.
    alpha_trades = sum(
        float(position.metadata.get("portfolio_expected_r", 0.0) or 0.0) > 0.0
        for position in result.state.accepted_positions
    )
    metrics.update(
        {
            "blocked_total_r": float(blocked_total_r),
            "blocked_value_r_per_month": float(-blocked_total_r / period_months),
            "alpha_positive_trades_per_month": float(alpha_trades / period_months),
        }
    )
    return metrics, result


def _score_components(
    metrics: dict[str, float],
    no_overlay_metrics: dict[str, float],
) -> dict[str, float]:
    growth = float(metrics.get("annual_log_growth", 0.0))
    r_month = float(metrics.get("total_r_per_month", 0.0))
    frequency = float(metrics.get("active_trades_per_month", 0.0))
    blocker_value = float(metrics.get("blocked_value_r_per_month", 0.0))
    synergy = float(metrics.get("certainty_equivalent_growth", 0.0)) - float(
        no_overlay_metrics.get("certainty_equivalent_growth", 0.0)
    )
    drawdown = float(metrics.get("max_drawdown_pct_mtm_daily", 0.0))
    expected_shortfall = float(metrics.get("expected_shortfall_95", 0.0))
    drawdown_penalty = 0.30 * max(0.0, (drawdown - 0.10) / 0.05) ** 2
    es_penalty = 0.15 * max(0.0, (expected_shortfall - 0.025) / 0.015) ** 2
    coverage = float(metrics.get("mark_coverage_ratio", 0.0))
    gross_leverage = float(metrics.get("gross_leverage_peak", 0.0))
    overnight_leverage = float(
        metrics.get("overnight_gross_leverage_peak", 0.0)
    )
    margin_breaches = float(metrics.get("margin_breach_count", 0.0))
    realism_penalty = (
        0.50 * margin_breaches
        + 0.25 * max(0.0, (0.98 - coverage) / 0.02) ** 2
        + 0.25 * max(0.0, (gross_leverage - 1.75) / 0.10) ** 2
        + 0.25 * max(0.0, (overnight_leverage - 1.15) / 0.10) ** 2
    )
    components = {
        "growth": 0.45 * math.tanh(growth / 0.50),
        "alpha_throughput": 0.25 * math.tanh(r_month / 15.0),
        "frequency": 0.10 * math.tanh(frequency / 100.0),
        "blocker_value": 0.10 * math.tanh(blocker_value / 2.0),
        "matched_risk_synergy": 0.10 * math.tanh(synergy / 0.10),
        "drawdown_penalty": -drawdown_penalty,
        "expected_shortfall_penalty": -es_penalty,
        "realism_penalty": -realism_penalty,
        "synergy_ce_delta": synergy,
    }
    components["score"] = sum(
        value for key, value in components.items() if key not in {"synergy_ce_delta"}
    )
    return components


def _passes_realism_gates(
    metrics: dict[str, float],
    config: dict[str, Any],
) -> bool:
    account = config.get("account_rules", {})
    gross_cap = float(account.get("max_gross_notional_pct", 0.0) or 0.0)
    net_cap = float(account.get("max_net_notional_pct", 0.0) or 0.0)
    overnight_cap = float(
        account.get("max_overnight_gross_notional_pct", 0.0) or 0.0
    )
    return (
        float(metrics.get("margin_breach_count", 1.0)) == 0.0
        and float(metrics.get("minimum_margin_buffer_pct", -1.0)) >= 0.0
        and float(metrics.get("mark_coverage_ratio", 0.0)) >= 0.98
        and (
            gross_cap <= 0.0
            or float(metrics.get("gross_leverage_peak", math.inf))
            <= gross_cap * 1.05
        )
        and (
            net_cap <= 0.0
            or float(metrics.get("net_leverage_peak_abs", math.inf))
            <= net_cap * 1.05
        )
        and (
            overnight_cap <= 0.0
            or float(metrics.get("overnight_gross_leverage_peak", math.inf))
            <= overnight_cap * 1.05
        )
    )


class CandidateEvaluator:
    def __init__(
        self,
        streams: StrategyStreams,
        close: pd.DataFrame,
        price_book: CausalPriceBook,
    ) -> None:
        self.streams = streams
        self.close = close
        self.price_book = price_book
        self._window_cache: dict[str, dict[str, float]] = {}

    def _no_overlay_metrics(
        self,
        config: dict[str, Any],
        start: date,
        end: date,
    ) -> dict[str, float]:
        no_overlay = _no_overlay(config)
        cache_key = _stable_sha(
            {
                "start": start,
                "end": end,
                "iaric_risk": no_overlay["strategy_allocations"][CURRENT_IARIC_ID][
                    "unit_risk_pct"
                ],
                "alcb_risk": no_overlay["strategy_allocations"][CURRENT_ALCB_ID][
                    "unit_risk_pct"
                ],
            }
        )
        if cache_key not in self._window_cache:
            metrics, _result = _evaluate_window(
                self.streams,
                no_overlay,
                close=self.close,
                price_book=self.price_book,
                start=start,
                end=end,
            )
            self._window_cache[cache_key] = metrics
        return self._window_cache[cache_key]

    def evaluate(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        aggregate, _result = _evaluate_window(
            self.streams,
            config,
            close=self.close,
            price_book=self.price_book,
            start=START,
            end=IS_END,
        )
        aggregate_control = self._no_overlay_metrics(config, START, IS_END)
        aggregate_components = _score_components(aggregate, aggregate_control)
        folds: dict[str, Any] = {}
        fold_scores: list[float] = []
        fold_synergy: list[float] = []
        positive_folds = 0
        for fold_name, fold_start, fold_end in FOLDS:
            metrics, _fold_result = _evaluate_window(
                self.streams,
                config,
                close=self.close,
                price_book=self.price_book,
                start=fold_start,
                end=fold_end,
            )
            control = self._no_overlay_metrics(config, fold_start, fold_end)
            components = _score_components(metrics, control)
            folds[fold_name] = {
                "metrics": metrics,
                "no_overlay_metrics": control,
                "score_components": components,
            }
            fold_scores.append(float(components["score"]))
            fold_synergy.append(float(components["synergy_ce_delta"]))
            positive_folds += int(float(metrics.get("net_pnl", 0.0)) > 0.0)

        median_score = float(np.median(fold_scores))
        lower_score = float(np.quantile(fold_scores, 0.25))
        worst_score = min(fold_scores)
        instability = float(np.quantile(fold_scores, 0.75) - np.quantile(fold_scores, 0.25))
        robust_score = (
            0.55 * median_score
            + 0.30 * lower_score
            + 0.15 * worst_score
            - 0.10 * instability
        )
        strategy_positive = (
            float(aggregate.get(f"pnl_{CURRENT_ALCB_ID}", 0.0)) > 0.0
            and float(aggregate.get(f"pnl_{CURRENT_IARIC_ID}", 0.0)) > 0.0
        )
        worst_fold_dd = max(
            float(row["metrics"].get("max_drawdown_pct_mtm_daily", 1.0))
            for row in folds.values()
        )
        median_pf = float(
            np.median(
                [float(row["metrics"].get("profit_factor", 0.0)) for row in folds.values()]
            )
        )
        negative_synergy_folds = sum(value < -0.05 for value in fold_synergy)
        eligible = (
            float(aggregate.get("max_drawdown_pct_mtm_daily", 1.0)) <= 0.15
            and worst_fold_dd <= 0.18
            and positive_folds >= 5
            and median_pf >= 1.15
            and float(aggregate.get("entry_accept_rate", 0.0)) >= 0.90
            and strategy_positive
            and float(np.median(fold_synergy)) >= -0.03
            and negative_synergy_folds <= 2
            and _passes_realism_gates(aggregate, config)
            and all(
                _passes_realism_gates(row["metrics"], config)
                for row in folds.values()
            )
        )
        return {
            "name": name,
            "config": config,
            "aggregate": aggregate,
            "aggregate_no_overlay": aggregate_control,
            "aggregate_score_components": aggregate_components,
            "folds": folds,
            "fold_scores": fold_scores,
            "positive_folds": positive_folds,
            "median_profit_factor": median_pf,
            "worst_fold_mtm_drawdown": worst_fold_dd,
            "median_synergy_ce_delta": float(np.median(fold_synergy)),
            "negative_synergy_folds": negative_synergy_folds,
            "robust_score": robust_score,
            "eligible": eligible,
        }

    def evaluate_oos(self, config: dict[str, Any]) -> dict[str, Any]:
        metrics, result = _evaluate_window(
            self.streams,
            config,
            close=self.close,
            price_book=self.price_book,
            start=OOS_START,
            end=OOS_END,
        )
        control = self._no_overlay_metrics(config, OOS_START, OOS_END)
        return {
            "window": [OOS_START, OOS_END],
            "metrics": metrics,
            "no_overlay_metrics": control,
            "score_components": _score_components(metrics, control),
            "blocked_by_reason": _blocked_by_reason(result),
        }


def _blocked_by_reason(result) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_r": 0.0, "positive_count": 0.0}
    )
    for candidate in result.state.blocked_candidates:
        row = rows[candidate.reason]
        row["count"] += 1.0
        row["total_r"] += float(candidate.r_multiple)
        row["positive_count"] += float(candidate.r_multiple > 0.0)
    return dict(rows)


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    return max(eligible or rows, key=lambda row: float(row["robust_score"]))


def _risk_candidates(seed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for global_multiplier in (0.85, 1.0, 1.15, 1.30):
        for iaric_tilt in (0.75, 1.0, 1.25):
            config = _patched(
                seed,
                {
                    f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct": (
                        NATIVE_ALCB_RISK * global_multiplier
                    ),
                    f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct": (
                        NATIVE_IARIC_RISK * global_multiplier * iaric_tilt
                    ),
                },
            )
            rows.append(
                (
                    f"risk_g{global_multiplier:.2f}_itilt{iaric_tilt:.2f}",
                    config,
                )
            )
    return rows


def _capacity_candidates(seed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for heat_pct in (0.045, 0.060, 0.075):
        heat_r = heat_pct / REFERENCE_RISK_PCT
        for action in ("block", "resize"):
            for max_positions in (19, 99):
                for net_cap in (1.50, 1.65, 1.75):
                    config = _patched(
                        seed,
                        {
                            "portfolio_rules.heat_cap_R": heat_r,
                            "portfolio_rules.max_long_heat_R": heat_r,
                            "portfolio_rules.max_symbol_heat_R": 8.0,
                            "portfolio_rules.max_total_active_positions": max_positions,
                            "cross_strategy_rules.capacity_action": action,
                            "cross_strategy_rules.minimum_capacity_size_mult": 0.35,
                            "account_rules.max_net_notional_pct": net_cap,
                        },
                    )
                    rows.append(
                        (
                            f"capacity_h{heat_pct:.3f}_{action}_n{max_positions}_net{net_cap:.2f}",
                            config,
                        )
                    )
    return rows


def _admission_candidates(seed: dict[str, Any], *, alpha_stable: bool) -> list[tuple[str, dict[str, Any]]]:
    rows = [("admission_control", deepcopy(seed))]
    if not alpha_stable:
        return rows
    for floor in (-0.10, -0.02, 0.02, 0.05):
        for uncertainty_penalty in (0.0, 0.25, 0.50):
            rows.append(
                (
                    f"admission_floor{floor:+.2f}_u{uncertainty_penalty:.2f}",
                    _patched(
                        seed,
                        {
                            "cross_strategy_rules.alpha_admission_enabled": True,
                            "cross_strategy_rules.minimum_expected_r": floor,
                            "cross_strategy_rules.alpha_uncertainty_penalty": uncertainty_penalty,
                            "cross_strategy_rules.candidate_rank_mode": "expected_net_r",
                        },
                    ),
                )
            )
    return rows


def _governor_candidates(seed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    stop_pairs = ((0.0, 0.0), (8.0, 24.0), (10.0, 28.0), (12.0, 32.0))
    for reserve_r in (0.0, 2.0, 4.0, 6.0):
        for daily_stop, weekly_stop in stop_pairs:
            rows.append(
                (
                    f"governor_reserve{reserve_r:.1f}_d{daily_stop:.1f}_w{weekly_stop:.1f}",
                    _patched(
                        seed,
                        {
                            "cross_strategy_rules.intraday_reserved_heat_R": reserve_r,
                            "portfolio_rules.portfolio_daily_stop_R": daily_stop,
                            "portfolio_rules.portfolio_weekly_stop_R": weekly_stop,
                        },
                    ),
                )
            )
    return rows


def _interaction_candidates(seed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    base_i = float(seed["strategy_allocations"][CURRENT_IARIC_ID]["unit_risk_pct"])
    base_a = float(seed["strategy_allocations"][CURRENT_ALCB_ID]["unit_risk_pct"])
    base_heat = float(seed["portfolio_rules"]["heat_cap_R"])
    base_floor = float(seed["cross_strategy_rules"].get("minimum_expected_r", -99.0))
    base_reserve = float(seed["cross_strategy_rules"].get("intraday_reserved_heat_R", 0.0))
    patterns = (
        (-1, -1, -1, 1, 1),
        (-1, -1, 1, -1, 1),
        (-1, 1, -1, -1, -1),
        (-1, 1, 1, 1, -1),
        (1, -1, -1, 1, -1),
        (1, -1, 1, -1, -1),
        (1, 1, -1, -1, 1),
        (1, 1, 1, 1, 1),
    )
    rows = [("interaction_control", deepcopy(seed))]
    for index, (risk_sign, tilt_sign, heat_sign, floor_sign, reserve_sign) in enumerate(
        patterns,
        start=1,
    ):
        global_mult = 1.0 + 0.10 * risk_sign
        iaric_tilt = 1.0 + 0.15 * tilt_sign
        patch = {
            f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct": base_a * global_mult,
            f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct": (
                base_i * global_mult * iaric_tilt
            ),
            "portfolio_rules.heat_cap_R": base_heat * (1.0 + 0.10 * heat_sign),
            "portfolio_rules.max_long_heat_R": base_heat * (1.0 + 0.10 * heat_sign),
            "cross_strategy_rules.intraday_reserved_heat_R": max(
                0.0, base_reserve + 2.0 * reserve_sign
            ),
        }
        if bool(seed["cross_strategy_rules"].get("alpha_admission_enabled", False)):
            patch["cross_strategy_rules.minimum_expected_r"] = base_floor + 0.02 * floor_sign
        rows.append((f"interaction_ff{index:02d}", _patched(seed, patch)))
    # Add isolated local perturbations so main effects are identifiable.
    for label, path, multiplier in (
        ("risk_down", f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct", 0.90),
        ("risk_up", f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct", 1.10),
        ("iaric_down", f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct", 0.85),
        ("iaric_up", f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct", 1.15),
        ("heat_down", "portfolio_rules.heat_cap_R", 0.90),
        ("heat_up", "portfolio_rules.heat_cap_R", 1.10),
        ("net_cap_down", "account_rules.max_net_notional_pct", 0.94),
        ("net_cap_up", "account_rules.max_net_notional_pct", 1.06),
        (
            "overnight_cap_down",
            "account_rules.max_overnight_gross_notional_pct",
            0.92,
        ),
        (
            "overnight_cap_up",
            "account_rules.max_overnight_gross_notional_pct",
            1.08,
        ),
    ):
        if CURRENT_ALCB_ID in path:
            value = base_a
        elif CURRENT_IARIC_ID in path:
            value = base_i
        elif path == "portfolio_rules.heat_cap_R":
            value = base_heat
        else:
            cursor: Any = seed
            for part in path.split("."):
                cursor = cursor[part]
            value = float(cursor)
        patch = {path: value * multiplier}
        if path == "portfolio_rules.heat_cap_R":
            patch["portfolio_rules.max_long_heat_R"] = value * multiplier
        rows.append((f"interaction_{label}", _patched(seed, patch)))
    return rows


def _generate_streams() -> tuple[StrategyStreams, pd.DataFrame, dict[str, Any]]:
    _status("generate_alcb_round3_stream")
    alcb = _run_alcb(CURRENT_ALCB_CONFIG, end=OOS_END)
    stored_alcb = load_trade_records(CURRENT_ALCB_TRADES)
    alcb_parity = _parity_receipt(
        stored_alcb,
        _filter(alcb, START, IS_END),
        "alcb_round3_config",
    )

    _status("generate_iaric_round3_stream")
    settings_payload = json.loads(CURRENT_IARIC_CONFIG.read_text(encoding="utf-8"))[
        "settings"
    ]
    settings = StrategySettings(**settings_payload)
    close, open_, high, low, volume, sectors, paths = _load_daily_panel_unsealed(
        DATA_DIR,
        OOS_END,
    )
    fingerprint, _rows = discovery._selection_data_fingerprint(
        close,
        open_,
        high,
        low,
        volume,
        paths,
    )
    residual_bundle = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model=settings.daily_residual_factor_model,
        source_fingerprint=fingerprint,
    )
    iaric_result = run_daily_residual_replay(
        residual_bundle,
        settings,
        start=START,
        end=OOS_END,
        initial_equity=100_000.0,
        round_trip_cost_bps=20.0,
    )
    iaric = _residual_records(iaric_result)
    stored_iaric = load_trade_records(CURRENT_IARIC_TRADES)
    iaric_parity = _parity_receipt(
        stored_iaric,
        _filter(iaric, START, OOS_END),
        "iaric_round3_config",
    )
    iaric_parity.update(
        {
            "source_fingerprint": fingerprint,
            "shared_core_contract": iaric_result.shared_core_contract,
        }
    )
    receipt = {
        "source": "regenerated_from_latest_round3_configs",
        "data_authority": "legacy_cache_diagnostic_only",
        "alcb": {
            "config": str(CURRENT_ALCB_CONFIG.relative_to(REPO_ROOT)),
            "config_sha256": _file_sha(CURRENT_ALCB_CONFIG),
            "trades": len(alcb),
            "is_trades": len(_filter(alcb, START, IS_END)),
            "oos_trades": len(_filter(alcb, OOS_START, OOS_END)),
            "stream_sha256": _stream_sha(alcb),
            "parity": alcb_parity,
        },
        "iaric": {
            "config": str(CURRENT_IARIC_CONFIG.relative_to(REPO_ROOT)),
            "config_sha256": _file_sha(CURRENT_IARIC_CONFIG),
            "trades": len(iaric),
            "is_trades": len(_filter(iaric, START, IS_END)),
            "oos_trades": len(_filter(iaric, OOS_START, OOS_END)),
            "stream_sha256": _stream_sha(iaric),
            "parity": iaric_parity,
        },
    }
    return StrategyStreams(tuple(alcb), tuple(iaric)), close, receipt


def _baseline_matrix(
    streams: StrategyStreams,
    close: pd.DataFrame,
    config: dict[str, Any],
    price_book: CausalPriceBook,
) -> dict[str, Any]:
    combined, _ = _evaluate_window(
        streams,
        config,
        close=close,
        price_book=price_book,
        start=START,
        end=IS_END,
    )
    alcb_only, _ = _evaluate_window(
        StrategyStreams(streams.alcb, ()),
        config,
        close=close,
        price_book=price_book,
        start=START,
        end=IS_END,
    )
    iaric_only, _ = _evaluate_window(
        StrategyStreams((), streams.iaric),
        config,
        close=close,
        price_book=price_book,
        start=START,
        end=IS_END,
    )
    return {
        "contract": "same_capital_same_cost_same_risk_no_overlay",
        "combined_union": combined,
        "alcb_standalone": alcb_only,
        "iaric_standalone": iaric_only,
        "daily_and_weekly_return_correlation": _stream_correlations(streams),
    }


def _stream_correlations(streams: StrategyStreams) -> dict[str, float]:
    def series(trades: tuple[TradeRecord, ...]) -> pd.Series:
        if not trades:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(
            [(trade.exit_time.date(), trade.r_multiple) for trade in trades],
            columns=["date", "r"],
        )
        return frame.groupby("date")["r"].sum()

    joined = pd.concat(
        [series(streams.alcb).rename("alcb"), series(streams.iaric).rename("iaric")],
        axis=1,
    ).fillna(0.0)
    daily = float(joined.corr().iloc[0, 1]) if len(joined) >= 2 else 0.0
    weekly_frame = joined.set_axis(pd.to_datetime(joined.index)).resample("W-FRI").sum()
    weekly = (
        float(weekly_frame.corr().iloc[0, 1]) if len(weekly_frame) >= 2 else 0.0
    )
    return {"daily_R_correlation": daily, "weekly_R_correlation": weekly}


def _stress_streams(streams: StrategyStreams, bps: float) -> StrategyStreams:
    def stress(trades: tuple[TradeRecord, ...]) -> tuple[TradeRecord, ...]:
        rows = []
        for trade in trades:
            risk = max(float(trade.risk_per_share * trade.quantity), 1e-9)
            notional = abs(float(trade.entry_price * trade.quantity))
            delta_r = notional * bps / 10_000.0 / risk
            rows.append(replace(trade, r_multiple=float(trade.r_multiple) - delta_r))
        return tuple(rows)

    return StrategyStreams(stress(streams.alcb), stress(streams.iaric))


def _pbo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"splits": 0, "probability_backtest_overfit": 1.0}
    from itertools import combinations

    logits = []
    selections: dict[str, int] = defaultdict(int)
    fold_indices = range(len(FOLDS))
    for training_tuple in combinations(fold_indices, len(FOLDS) // 2):
        if 0 not in training_tuple:
            continue
        training = set(training_tuple)
        testing = [index for index in fold_indices if index not in training]
        winner = max(
            rows,
            key=lambda row: float(
                np.mean([row["fold_scores"][index] for index in training])
            ),
        )
        selections[winner["name"]] += 1
        test_scores = sorted(
            float(np.mean([row["fold_scores"][index] for index in testing]))
            for row in rows
        )
        winner_test = float(
            np.mean([winner["fold_scores"][index] for index in testing])
        )
        rank = sum(score <= winner_test for score in test_scores) / len(test_scores)
        rank = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))
    return {
        "splits": len(logits),
        "probability_backtest_overfit": float(np.mean(np.asarray(logits) < 0.0))
        if logits
        else 1.0,
        "median_test_rank_logit": float(np.median(logits)) if logits else float("-inf"),
        "training_selection_counts": dict(selections),
    }


def _robustness(
    config: dict[str, Any],
    streams: StrategyStreams,
    close: pd.DataFrame,
    interaction_rows: list[dict[str, Any]],
    price_book: CausalPriceBook,
) -> dict[str, Any]:
    control, control_result = _evaluate_window(
        streams,
        config,
        close=close,
        price_book=price_book,
        start=START,
        end=IS_END,
    )
    perturbations = []
    paths = (
        f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct",
        f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct",
        "portfolio_rules.heat_cap_R",
        "cross_strategy_rules.intraday_reserved_heat_R",
        "account_rules.max_net_notional_pct",
        "account_rules.max_overnight_gross_notional_pct",
    )
    for path in paths:
        cursor: Any = config
        for part in path.split("."):
            cursor = cursor[part]
        value = float(cursor)
        if value <= 0.0:
            continue
        for multiplier in (0.90, 1.10):
            patch = {path: value * multiplier}
            if path == "portfolio_rules.heat_cap_R":
                patch["portfolio_rules.max_long_heat_R"] = value * multiplier
            metrics, _ = _evaluate_window(
                streams,
                _patched(config, patch),
                close=close,
                price_book=price_book,
                start=START,
                end=IS_END,
            )
            perturbations.append(
                {"path": path, "multiplier": multiplier, "metrics": metrics}
            )

    costs = []
    for bps in (5.0, 10.0, 20.0):
        metrics, _ = _evaluate_window(
            _stress_streams(streams, bps),
            config,
            close=close,
            price_book=price_book,
            start=START,
            end=IS_END,
        )
        costs.append({"extra_round_trip_bps": bps, "metrics": metrics})

    weekly: dict[str, float] = defaultdict(float)
    for outcome in control_result.trade_outcomes:
        iso = outcome.exit_time.isocalendar()
        weekly[f"{iso.year:04d}-W{iso.week:02d}"] += outcome.net_pnl
    weekly_values = np.asarray(list(weekly.values()), dtype=float)
    rng = np.random.default_rng(20260823)
    samples = (
        np.asarray(
            [
                rng.choice(weekly_values, size=len(weekly_values), replace=True).sum()
                for _ in range(2_000)
            ]
        )
        if len(weekly_values)
        else np.asarray([0.0])
    )
    return {
        "control": control,
        "local_perturbations": perturbations,
        "incremental_cost_stress": costs,
        "weekly_block_bootstrap": {
            "samples": len(samples),
            "probability_total_pnl_positive": float(np.mean(samples > 0.0)),
            "ci_95_total_pnl": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        },
        "cscv_pbo": _pbo(interaction_rows),
    }


def _phase(
    output: Path,
    filename: str,
    evaluator: CandidateEvaluator,
    candidates: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _status("evaluate_phase", phase=filename, candidates=len(candidates))
    rows = [evaluator.evaluate(name, config) for name, config in candidates]
    selected = _select(rows)
    _write_json(
        output / filename,
        {"selected": selected["name"], "results": rows},
    )
    return selected, rows


def _interaction_phase(
    output: Path,
    evaluator: CandidateEvaluator,
    candidates: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    filename = "phase_5_interactions.json"
    _status("evaluate_phase", phase=filename, candidates=len(candidates))
    rows = [evaluator.evaluate(name, config) for name, config in candidates]
    unconstrained = _select(rows)
    incumbent = next(row for row in rows if row["name"] == "interaction_control")
    stability = _pbo(rows)
    robust_score_gain = float(unconstrained["robust_score"]) - float(
        incumbent["robust_score"]
    )
    challenger_stable = (
        bool(unconstrained["eligible"])
        and float(stability["probability_backtest_overfit"])
        <= MAX_INTERACTION_PBO
        and robust_score_gain >= MIN_INTERACTION_ROBUST_SCORE_GAIN
    )
    selected = unconstrained if challenger_stable else incumbent
    selection_audit = {
        "unconstrained_winner": unconstrained["name"],
        "incumbent": incumbent["name"],
        "selected": selected["name"],
        "challenger_stable": challenger_stable,
        "robust_score_gain": robust_score_gain,
        "minimum_robust_score_gain": MIN_INTERACTION_ROBUST_SCORE_GAIN,
        "maximum_probability_backtest_overfit": MAX_INTERACTION_PBO,
        "cscv_pbo": stability,
        "fallback_reason": (
            "none"
            if challenger_stable
            else "retain_incumbent_due_to_interaction_instability"
        ),
    }
    _write_json(
        output / filename,
        {
            "selected": selected["name"],
            "selection_audit": selection_audit,
            "results": rows,
        },
    )
    return selected, rows, selection_audit


def _render_report(
    selected: dict[str, Any],
    oos: dict[str, Any],
    receipt: dict[str, Any],
    alpha_model: str,
    robustness: dict[str, Any],
) -> str:
    is_metrics = selected["aggregate"]
    oos_metrics = oos["metrics"]
    lines = [
        "# Stock portfolio shared-account phased auto — Round 6",
        "",
        "Status: **research_only_shared_account_round_complete**",
        "",
        "## Contract",
        "",
        f"- IS: {START} through {IS_END}",
        f"- OOS: {OOS_START} through {OOS_END}",
        f"- Alpha model: `{alpha_model}`",
        f"- Selected candidate: `{selected['name']}`",
        "- OOS was evaluated once after the configuration hash was frozen.",
        "",
        "## IS",
        "",
        f"- Return / PF / daily-MTM DD: {is_metrics['net_return_pct']:.2%} / {is_metrics['profit_factor']:.2f} / {is_metrics['max_drawdown_pct_mtm_daily']:.2%}",
        f"- Trades / acceptance: {int(is_metrics['total_trades'])} / {is_metrics['entry_accept_rate']:.1%}",
        f"- Matched-risk CE synergy: {selected['aggregate_score_components']['synergy_ce_delta']:.4f}",
        f"- Peak gross / overnight leverage: {is_metrics['gross_leverage_peak']:.2f}x / {is_metrics['overnight_gross_leverage_peak']:.2f}x",
        f"- Mark coverage / margin breaches: {is_metrics['mark_coverage_ratio']:.1%} / {int(is_metrics['margin_breach_count'])}",
        "",
        "## OOS",
        "",
        f"- Return / PF / daily-MTM DD: {oos_metrics['net_return_pct']:.2%} / {oos_metrics['profit_factor']:.2f} / {oos_metrics['max_drawdown_pct_mtm_daily']:.2%}",
        f"- Trades / acceptance: {int(oos_metrics['total_trades'])} / {oos_metrics['entry_accept_rate']:.1%}",
        f"- Matched-risk CE synergy: {oos['score_components']['synergy_ce_delta']:.4f}",
        f"- Peak gross / overnight leverage: {oos_metrics['gross_leverage_peak']:.2f}x / {oos_metrics['overnight_gross_leverage_peak']:.2f}x",
        f"- Mark coverage / margin breaches: {oos_metrics['mark_coverage_ratio']:.1%} / {int(oos_metrics['margin_breach_count'])}",
        "",
        "## Research restrictions",
        "",
        f"- ALCB parity: {receipt['alcb']['parity']['passed']}",
        f"- IARIC parity: {receipt['iaric']['parity']['passed']}",
        "- The retained cache is diagnostic-only and IARIC Round 3 was strategy-level OOS-informed.",
        f"- CSCV PBO estimate: {robustness['cscv_pbo']['probability_backtest_overfit']:.1%}",
        "- Production activation is not approved.",
    ]
    return "\n".join(lines) + "\n"


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "family": "stock",
        "strategy": "portfolio_synergy",
        "round": 6,
        "contract": "shared_account_round3_stream_synergy_phased_auto_v3_stability_guard",
        "is_window": [START, IS_END],
        "oos_window": [OOS_START, OOS_END],
        "oos_policy": "single_evaluation_after_selection_freeze",
        "boundary_policy": "purged complete trades; no outcome may cross a fold or IS/OOS boundary",
        "initial_equity": INITIAL_EQUITY,
        "reference_risk_pct_fixed": REFERENCE_RISK_PCT,
        "immutable_score": {
            "growth": 0.45,
            "alpha_throughput": 0.25,
            "frequency": 0.10,
            "blocker_value": 0.10,
            "matched_risk_synergy": 0.10,
            "fold_aggregation": "0.55*median + 0.30*q25 + 0.15*worst - 0.10*IQR",
            "fixed_scales": {
                "annual_log_growth": 0.50,
                "R_per_month": 15.0,
                "trades_per_month": 100.0,
                "blocked_value_R_per_month": 2.0,
                "synergy_CE": 0.10,
            },
            "risk_target": "10-12% daily-MTM drawdown; hard IS gate 15%",
            "interaction_selection": {
                "maximum_cscv_pbo": MAX_INTERACTION_PBO,
                "minimum_robust_score_gain": MIN_INTERACTION_ROBUST_SCORE_GAIN,
                "fallback": "retain simpler phase-4 incumbent",
            },
            "realism_penalties": {
                "minimum_mark_coverage": 0.98,
                "zero_margin_breaches": True,
                "gross_leverage_cap": 1.75,
                "overnight_gross_leverage_cap": 1.15,
            },
        },
        "realism_contract": {
            "shared_cash_nlv_and_margin_ledger": True,
            "sizing_uses_causal_mark_to_market_nlv": True,
            "completed_bar_marks_only": True,
            "integer_share_floor_without_forced_minimum": True,
            "debit_cash_financing_included": True,
            "reg_t_initial_margin_long_pct": 0.50,
            "gross_intraday_cap": 1.75,
            "net_cap": 1.65,
            "gross_overnight_cap": 1.15,
            "raw_signal_level_cosimulation": False,
            "point_in_time_universe": False,
        },
        "data_authority": {
            "mode": "legacy_cache_diagnostic_only",
            "production_eligible": False,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "run_spec.json", run_spec)

    raw_streams, close, receipt = _generate_streams()
    _write_json(output / "stream_receipt.json", receipt)
    symbols = {
        trade.symbol
        for trade in (*raw_streams.alcb, *raw_streams.iaric)
    }
    _status("load_causal_marks", symbols=len(symbols))
    intraday_bars = _load_stock_price_bars(DATA_DIR, symbols)
    price_book = CausalPriceBook(intraday_bars, daily_close=close)
    is_raw = StrategyStreams(
        _filter(raw_streams.alcb, START, IS_END),
        _filter(raw_streams.iaric, START, IS_END),
    )
    variants = {"constant": None, "ridge_10": 10.0, "ridge_50": 50.0, "ridge_200": 200.0}
    _status("alpha_calibration", variants=len(variants))
    alpha_model, alpha_diagnostics, predicted_variants = _alpha_model_diagnostics(
        is_raw,
        variants,
    )
    _write_json(
        output / "alpha_calibration.json",
        {"selected": alpha_model, "variants": alpha_diagnostics},
    )

    # Refit the chosen model on IS-only information for the declared OOS stream.
    selected_ridge = variants[alpha_model]
    alcb_predictions = _expanding_predictions(
        raw_streams.alcb,
        ridge=selected_ridge,
        train_end=IS_END,
    )
    iaric_predictions = _expanding_predictions(
        raw_streams.iaric,
        ridge=selected_ridge,
        train_end=IS_END,
    )
    all_predicted = StrategyStreams(
        _attach_predictions(raw_streams.alcb, alcb_predictions),
        _attach_predictions(raw_streams.iaric, iaric_predictions),
    )
    is_predicted = predicted_variants[alpha_model]
    evaluator = CandidateEvaluator(is_predicted, close, price_book)

    base = baseline_config()
    _write_json(
        output / "phase_0_matched_baselines.json",
        _baseline_matrix(is_predicted, close, base, price_book),
    )
    risk_winner, risk_rows = _phase(
        output,
        "phase_1_risk_allocation.json",
        evaluator,
        _risk_candidates(base),
    )
    capacity_winner, capacity_rows = _phase(
        output,
        "phase_2_capacity.json",
        evaluator,
        _capacity_candidates(risk_winner["config"]),
    )
    alpha_stable = bool(alpha_diagnostics[alpha_model]["stable"])
    admission_winner, admission_rows = _phase(
        output,
        "phase_3_alpha_admission.json",
        evaluator,
        _admission_candidates(capacity_winner["config"], alpha_stable=alpha_stable),
    )
    governor_winner, governor_rows = _phase(
        output,
        "phase_4_reserve_governors.json",
        evaluator,
        _governor_candidates(admission_winner["config"]),
    )
    final_winner, interaction_rows, interaction_selection = _interaction_phase(
        output,
        evaluator,
        _interaction_candidates(governor_winner["config"]),
    )

    final_config = final_winner["config"]
    _write_json(output / "optimized_config.json", final_config)
    config_sha = _file_sha(output / "optimized_config.json")
    freeze_receipt = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "selected_candidate": final_winner["name"],
        "interaction_selection": interaction_selection,
        "selection_used_oos": False,
        "oos_window": [OOS_START, OOS_END],
    }
    _write_json(output / "freeze_receipt.json", freeze_receipt)

    _status("evaluate_oos_once", config_sha256=config_sha)
    oos_evaluator = CandidateEvaluator(all_predicted, close, price_book)
    oos = oos_evaluator.evaluate_oos(final_config)
    oos["config_sha256_rechecked"] = _file_sha(output / "optimized_config.json")
    oos["config_unchanged_after_freeze"] = oos["config_sha256_rechecked"] == config_sha
    _write_json(output / "oos_validation.json", oos)

    _status("final_robustness")
    robustness = _robustness(
        final_config,
        is_predicted,
        close,
        interaction_rows,
        price_book,
    )
    _write_json(output / "final_robustness.json", robustness)

    oos_metrics = oos["metrics"]
    oos_gate = (
        float(oos_metrics.get("net_pnl", 0.0)) > 0.0
        and float(oos_metrics.get("profit_factor", 0.0)) >= 1.20
        and float(oos_metrics.get("max_drawdown_pct_mtm_daily", 1.0)) <= 0.15
        and float(oos_metrics.get("entry_accept_rate", 0.0)) >= 0.90
        and float(oos["score_components"].get("synergy_ce_delta", -1.0)) >= -0.03
        and _passes_realism_gates(oos_metrics, final_config)
    )
    cost_10 = next(
        row
        for row in robustness["incremental_cost_stress"]
        if row["extra_round_trip_bps"] == 10.0
    )
    decision = {
        "status": "research_only_shared_account_round_complete",
        "config_sha256": config_sha,
        "gates": {
            "is_selection_eligible": bool(final_winner["eligible"]),
            "oos": oos_gate,
            "local_perturbations_positive": all(
                float(row["metrics"].get("net_pnl", 0.0)) > 0.0
                for row in robustness["local_perturbations"]
            ),
            "incremental_10bps_positive": float(cost_10["metrics"].get("net_pnl", 0.0)) > 0.0,
            "alcb_parity": bool(receipt["alcb"]["parity"]["passed"]),
            "iaric_parity": bool(receipt["iaric"]["parity"]["passed"]),
            "frozen_data_authority": False,
            "strategy_level_oos_clean": False,
            "shared_account_realism": _passes_realism_gates(
                oos_metrics,
                final_config,
            ),
            "raw_signal_level_cosimulation": False,
            "point_in_time_universe": False,
            "interaction_selection_stable": bool(
                interaction_selection["challenger_stable"]
                or final_winner["name"] == "interaction_control"
            ),
            "cscv_pbo_below_limit": float(
                interaction_selection["cscv_pbo"][
                    "probability_backtest_overfit"
                ]
            )
            <= MAX_INTERACTION_PBO,
        },
        "production_activation_approved": False,
        "research_restrictions": [
            "no frozen authoritative stock-data bundle",
            "IARIC Round 3 configuration was informed by the declared portfolio OOS period",
            "live synchronization and shadow parity remain required",
            "the two strategy inputs remain completed trade streams rather than raw rejected and accepted signals",
            "the stock universe is not yet point-in-time survivorship controlled",
        ],
    }
    _write_json(output / "promotion_decision.json", decision)
    report = _render_report(final_winner, oos, receipt, alpha_model, robustness)
    (output / "round_final_diagnostics.md").write_text(report, encoding="utf-8")
    (output / "round_final_diagnostics.txt").write_text(report, encoding="utf-8")

    artifact_manifest = {
        "round": 6,
        "active": False,
        "status": decision["status"],
        "config_sha256": config_sha,
        "artifacts": {
            path.name: _file_sha(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        },
        "candidate_counts": {
            "risk": len(risk_rows),
            "capacity": len(capacity_rows),
            "admission": len(admission_rows),
            "governors": len(governor_rows),
            "interactions": len(interaction_rows),
        },
    }
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    _status(
        "complete",
        status=decision["status"],
        selected=final_winner["name"],
        oos=oos_metrics,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "backtests"
        / "output"
        / "stock"
        / "portfolio_synergy"
        / "round_6_realism_stable",
    )
    args = parser.parse_args()
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
