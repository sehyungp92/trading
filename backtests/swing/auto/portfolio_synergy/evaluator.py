from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.swing.analysis.metrics import compute_metrics
from backtests.swing.auto.brs.config_mutator import mutate_brs_config
from backtests.swing.auto.config_mutator import mutate_unified_config
from backtests.swing.config_brs import BRSConfig
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified
from backtests.swing.data.replay_cache import load_unified_portfolio_replay_bundle

from .phase_candidates import INITIAL_EQUITY, SEED_PORTFOLIO_CONFIG, STRATEGY_ORDER


ACTIVE_STRATEGIES = ("ATRSS", "AKC_HELIX", "BRS_R9", "SWING_BREAKOUT_V3")


def load_evaluation_data(data_dir: Path, initial_equity: float):
    return load_evaluation_bundle(data_dir, initial_equity).data, None


def load_evaluation_bundle(data_dir: Path, initial_equity: float):
    unified_cfg = UnifiedBacktestConfig(initial_equity=initial_equity, data_dir=Path(data_dir))
    return load_unified_portfolio_replay_bundle(unified_cfg)


def evaluate_portfolio(
    mutations: dict[str, Any] | None,
    *,
    data_dir: Path,
    initial_equity: float = INITIAL_EQUITY,
    unified_data=None,
    brs_data=None,
) -> dict[str, Any]:
    mutations = dict(mutations or {})
    unified_data = unified_data if unified_data is not None else load_unified_data(
        UnifiedBacktestConfig(initial_equity=initial_equity, data_dir=Path(data_dir))
    )

    effective = build_effective_portfolio_config(mutations, initial_equity=initial_equity)
    unified_cfg = build_replay_config(effective, Path(data_dir), initial_equity)

    unified_result = run_unified(unified_data, unified_cfg)

    all_trades = _collect_unified_trades(unified_result)
    equity = _headline_equity_curve(unified_result)
    realized_equity = _realized_equity_curve(unified_result)
    timestamps = unified_result.combined_timestamps

    metrics = compute_metrics(
        np.array([_trade_pnl(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_risk(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_hold_hours(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_commission(trade) for trade in all_trades], dtype=np.float64),
        equity,
        timestamps,
        initial_equity,
        trade_symbols=[_trade_symbol(trade) for trade in all_trades],
    )
    realized_metrics = compute_metrics(
        np.array([_trade_pnl(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_risk(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_hold_hours(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_commission(trade) for trade in all_trades], dtype=np.float64),
        realized_equity,
        timestamps,
        initial_equity,
        trade_symbols=[_trade_symbol(trade) for trade in all_trades],
    )

    strategy_counts = _strategy_trade_counts(unified_result)
    strategy_pnls = _strategy_pnls(unified_result)
    total_active_trades = sum(strategy_counts.values())
    active_strategies = sum(1 for strategy in ACTIVE_STRATEGIES if strategy_counts.get(strategy, 0) > 0)
    overlay_enabled = bool(effective["strategy_allocations"]["OVERLAY"].get("enabled", True))
    if overlay_enabled:
        active_strategies += 1

    max_strategy_trade_share = (
        max(strategy_counts.values()) / total_active_trades
        if total_active_trades > 0 and strategy_counts else 0.0
    )
    entry_signals_fired = sum(
        getattr(item, "entry_signals_fired", 0)
        for item in unified_result.strategy_results.values()
    )
    entries_accepted = sum(
        getattr(item, "entries_accepted_by_portfolio", 0)
        for item in unified_result.strategy_results.values()
    )
    entries_blocked = sum(
        getattr(item, "entries_blocked_by_heat", 0)
        for item in unified_result.strategy_results.values()
    )
    capture_denominator = entry_signals_fired if entry_signals_fired else total_active_trades + entries_blocked
    heat_block_rate = entries_blocked / capture_denominator if capture_denominator else 0.0

    final_equity = float(equity[-1]) if len(equity) else initial_equity
    final_equity_realized = float(realized_equity[-1]) if len(realized_equity) else initial_equity
    net_return_pct = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
    net_return_pct_realized = (
        (final_equity_realized - initial_equity) / initial_equity if initial_equity > 0 else 0.0
    )
    total_r = _total_r(all_trades)
    months = _months(timestamps)
    overlay_pnl = float(getattr(unified_result, "overlay_pnl", 0.0))
    overlay_commission = float(getattr(unified_result, "overlay_commission", 0.0))
    overlay_return_pct = overlay_pnl / initial_equity if initial_equity > 0 else 0.0
    overlay_cap = float(effective["strategy_allocations"]["OVERLAY"].get("max_equity_pct", 0.0) or 0.0)

    return {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "net_pnl": final_equity - initial_equity,
        "net_return_pct": net_return_pct,
        "final_equity_realized": final_equity_realized,
        "net_pnl_realized": final_equity_realized - initial_equity,
        "net_return_pct_realized": net_return_pct_realized,
        "total_trades": float(total_active_trades),
        "entry_signals_fired": float(entry_signals_fired),
        "entries_accepted_by_portfolio": float(entries_accepted),
        "entries_blocked_by_portfolio": float(entries_blocked),
        "entry_accept_rate": float(entries_accepted / entry_signals_fired) if entry_signals_fired else 0.0,
        "active_trades_per_month": total_active_trades / months if months > 0 else 0.0,
        "total_r": total_r,
        "total_r_per_month": total_r / months if months > 0 else 0.0,
        "profit_factor": float(metrics.profit_factor),
        "win_rate": float(metrics.win_rate),
        "expectancy_r": float(metrics.expectancy),
        "sharpe": float(metrics.sharpe),
        "sortino": float(metrics.sortino),
        "calmar": float(metrics.calmar),
        "max_drawdown_pct": float(metrics.max_drawdown_pct),
        "max_drawdown_dollar": float(metrics.max_drawdown_dollar),
        "risk_basis": _risk_basis(unified_result),
        "calmar_mtm": float(metrics.calmar),
        "max_drawdown_pct_mtm": float(metrics.max_drawdown_pct),
        "max_drawdown_dollar_mtm": float(metrics.max_drawdown_dollar),
        "calmar_realized": float(realized_metrics.calmar),
        "max_drawdown_pct_realized": float(realized_metrics.max_drawdown_pct),
        "max_drawdown_dollar_realized": float(realized_metrics.max_drawdown_dollar),
        "active_strategy_count": float(active_strategies),
        "overlay_enabled": 1.0 if overlay_enabled else 0.0,
        "overlay_pnl": overlay_pnl,
        "overlay_pnl_net": overlay_pnl,
        "overlay_commission": overlay_commission,
        "overlay_return_pct": overlay_return_pct,
        "overlay_max_equity_pct": overlay_cap,
        "overlay_score_drag_pct": max(0.0, -overlay_return_pct),
        "idle_cash_deployment_share": overlay_cap if overlay_enabled else 0.0,
        "max_strategy_trade_share": float(max_strategy_trade_share),
        "positive_alpha_block_rate": float(heat_block_rate),
        "trade_capture_ratio": float(1.0 - heat_block_rate),
        "positive_years": float(_positive_years(equity, timestamps)),
        "strategy_quality_regression_pct": 0.0,
        **{f"trades_{key}": float(value) for key, value in strategy_counts.items()},
        **{f"pnl_{key}": float(value) for key, value in strategy_pnls.items()},
        **{
            f"fired_{key}": float(getattr(value, "entry_signals_fired", 0))
            for key, value in unified_result.strategy_results.items()
        },
        **{
            f"accepted_{key}": float(getattr(value, "entries_accepted_by_portfolio", 0))
            for key, value in unified_result.strategy_results.items()
        },
        **{
            f"blocked_{key}": float(getattr(value, "entries_blocked_by_heat", 0))
            for key, value in unified_result.strategy_results.items()
        },
    }


def _headline_equity_curve(unified_result) -> np.ndarray:
    mtm = np.asarray(getattr(unified_result, "combined_equity_mtm", np.array([])), dtype=np.float64)
    if len(mtm):
        return mtm
    return np.asarray(getattr(unified_result, "combined_equity", np.array([])), dtype=np.float64)


def _realized_equity_curve(unified_result) -> np.ndarray:
    realized = np.asarray(getattr(unified_result, "combined_equity_realized", np.array([])), dtype=np.float64)
    if len(realized):
        return realized
    return np.asarray(getattr(unified_result, "combined_equity", np.array([])), dtype=np.float64)


def _risk_basis(unified_result) -> str:
    mtm = getattr(unified_result, "combined_equity_mtm", np.array([]))
    return "child_mtm_plus_overlay_mtm" if len(mtm) else "realized_delta_plus_overlay_mtm"


def build_effective_portfolio_config(
    mutations: dict[str, Any] | None,
    *,
    initial_equity: float = INITIAL_EQUITY,
) -> dict[str, Any]:
    config = deepcopy(SEED_PORTFOLIO_CONFIG)
    config["initial_equity"] = float(initial_equity)
    for key, value in (mutations or {}).items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            _deep_merge(config[key], value)
        elif key in config:
            config[key] = deepcopy(value)
        elif "." in key:
            _set_path(config, key.split("."), value)
        else:
            config[key] = deepcopy(value)
    return config


def build_replay_config(
    effective: dict[str, Any],
    data_dir: Path,
    initial_equity: float,
) -> UnifiedBacktestConfig:
    latest_unified, latest_brs = _load_latest_strategy_mutations(_repo_root())
    unified_mutations = dict(latest_unified)
    brs_mutations = dict(latest_brs)

    allocations = effective["strategy_allocations"]
    rules = effective["portfolio_rules"]
    cross = effective.get("cross_strategy_rules", {})

    unified_mutations.update(
        {
            "heat_cap_R": float(rules.get("heat_cap_R", 3.0)),
            "portfolio_daily_stop_R": float(rules.get("portfolio_daily_stop_R", 4.0)),
            "dynamic_risk_enabled": bool(rules.get("dynamic_risk_enabled", False)),
            "drawdown_risk_tiers": tuple(tuple(item) for item in rules.get("drawdown_tiers", ())),
            "symbol_risk_multipliers": dict(effective.get("symbol_risk_multipliers", {})),
            "atrss.unit_risk_pct": float(allocations["ATRSS"].get("unit_risk_pct", 0.01)),
            "atrss.max_heat_R": float(allocations["ATRSS"].get("max_heat_R", 1.0)),
            "atrss.daily_stop_R": float(allocations["ATRSS"].get("daily_stop_R", 2.0)),
            "atrss.max_working_orders": int(allocations["ATRSS"].get("max_working_orders", 4)),
            "helix.unit_risk_pct": float(allocations["AKC_HELIX"].get("unit_risk_pct", 0.005)),
            "helix.max_heat_R": float(allocations["AKC_HELIX"].get("max_heat_R", 1.0)),
            "helix.daily_stop_R": float(allocations["AKC_HELIX"].get("daily_stop_R", 2.5)),
            "helix.max_working_orders": int(allocations["AKC_HELIX"].get("max_working_orders", 4)),
            "breakout.unit_risk_pct": float(allocations["SWING_BREAKOUT_V3"].get("unit_risk_pct", 0.005)),
            "breakout.max_heat_R": float(allocations["SWING_BREAKOUT_V3"].get("max_heat_R", 0.75)),
            "breakout.daily_stop_R": float(allocations["SWING_BREAKOUT_V3"].get("daily_stop_R", 1.5)),
            "breakout.max_working_orders": int(allocations["SWING_BREAKOUT_V3"].get("max_working_orders", 2)),
            "brs.unit_risk_pct": float(allocations["BRS_R9"].get("unit_risk_pct", 0.006)),
            "brs.max_heat_R": float(allocations["BRS_R9"].get("max_heat_R", 1.25)),
            "brs.daily_stop_R": float(allocations["BRS_R9"].get("daily_stop_R", 2.0)),
            "brs.max_working_orders": int(allocations["BRS_R9"].get("max_working_orders", 3)),
            "overlay_enabled": bool(allocations["OVERLAY"].get("enabled", True)),
            "overlay_max_pct": float(allocations["OVERLAY"].get("max_equity_pct", 0.70)),
            "overlay_min_alloc_pct": float(allocations["OVERLAY"].get("min_alloc_pct", 0.25)),
            "overlay_max_alloc_pct": float(allocations["OVERLAY"].get("max_alloc_pct", 1.00)),
            "overlay_weights": dict(allocations["OVERLAY"].get("weights", {"QQQ": 0.5, "GLD": 0.5})),
            "enable_atrss_helix_tighten": bool(
                cross.get("atrss_helix_alignment", {}).get("tighten_helix_to_be_on_atrss_entry", True)
            ),
            "enable_atrss_helix_size_boost": bool(cross.get("atrss_helix_alignment", {}).get("enabled", True)),
        }
    )

    filters = effective.get("strategy_filters", {})
    if filters.get("AKC_HELIX", {}).get("stale_exit_tighten"):
        unified_mutations.update(
            {
                "helix_param.STALE_1H_BARS": 20,
                "helix_param.STALE_4H_BARS": 10,
                "helix_param.EARLY_STALE_BARS": 14,
            }
        )
    if filters.get("SWING_BREAKOUT_V3", {}).get("entry_window_extension_hours", 0):
        unified_mutations.update(
            {
                "breakout_param.ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE": True,
                "breakout_param.ENTRY_OUTSIDE_WINDOW_CARRY_TTL_HOURS": int(
                    filters["SWING_BREAKOUT_V3"].get("entry_window_extension_hours", 1)
                ),
            }
        )
    if filters.get("SWING_BREAKOUT_V3", {}).get("enable_continuation_probe"):
        unified_mutations.update(
            {
                "breakout_param.ENTRY_C_CONTINUATION_ENABLE": True,
                "breakout_param.C_CONTINUATION_RISK_MULT": float(
                    allocations["SWING_BREAKOUT_V3"].get("continuation_size_mult", 0.50)
                ),
            }
        )
    if filters.get("ATRSS", {}).get("breakout_reactivation_probe"):
        unified_mutations.update(
            {
                "atrss_flags.breakout_entries": True,
                "atrss_param.breakout_direct_entry": True,
            }
        )
    if filters.get("ATRSS", {}).get("rank_mode"):
        unified_mutations["atrss_param.rank_mode"] = str(filters["ATRSS"]["rank_mode"])

    brs_alloc = allocations["BRS_R9"]
    brs_risk = float(brs_alloc.get("unit_risk_pct", 0.006))
    brs_mutations.update(
        {
            "heat_cap_r": float(brs_alloc.get("max_heat_R", 1.25)),
            "max_concurrent": int(brs_alloc.get("max_working_orders", 3)),
            "symbol_configs.QQQ.base_risk_pct": brs_risk * 0.75,
            "symbol_configs.GLD.base_risk_pct": brs_risk,
        }
    )
    if filters.get("BRS_R9", {}).get("block_lh_rejection_bear_strong"):
        brs_mutations["disable_lh"] = True

    unified_cfg = mutate_unified_config(
        UnifiedBacktestConfig(initial_equity=initial_equity, data_dir=data_dir, fixed_qty=None),
        unified_mutations,
    )
    brs_cfg = mutate_brs_config(
        BRSConfig(initial_equity=initial_equity, data_dir=data_dir),
        brs_mutations,
    )
    unified_cfg = replace(
        unified_cfg,
        atrss=replace(unified_cfg.atrss, strategy_id="ATRSS", priority=int(allocations["ATRSS"].get("priority", 0))),
        helix=replace(unified_cfg.helix, strategy_id="AKC_HELIX", priority=int(allocations["AKC_HELIX"].get("priority", 1))),
        brs=replace(unified_cfg.brs, strategy_id="BRS_R9", priority=int(allocations["BRS_R9"].get("priority", 2))),
        breakout=replace(
            unified_cfg.breakout,
            strategy_id="SWING_BREAKOUT_V3",
            priority=int(allocations["SWING_BREAKOUT_V3"].get("priority", 3)),
        ),
        brs_runtime_config=brs_cfg,
    )
    return unified_cfg


def _load_latest_strategy_mutations(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load_latest_strategy_mutations_cached(str(repo_root))


@lru_cache(maxsize=1)
def _load_latest_strategy_mutations_cached(repo_root_str: str) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = Path(repo_root_str)
    unified: dict[str, Any] = {}
    brs: dict[str, Any] = {}

    atrss = _read_json(repo_root / "backtests/swing/auto/atrss/output/phase_state.json")
    for key, value in (atrss.get("cumulative_mutations") or {}).items():
        if key.startswith("flags."):
            unified[f"atrss_flags.{key.split('.', 1)[1]}"] = value
        elif key.startswith("param_overrides."):
            unified[f"atrss_param.{key.split('.', 1)[1]}"] = value

    helix = _read_json(repo_root / "backtests/swing/auto/output/helix_exit_alpha_optimized_baseline_summary.json")
    for key, value in (helix.get("mutations") or {}).items():
        if key.startswith("flags."):
            unified[f"helix_flags.{key.split('.', 1)[1]}"] = value
        elif key.startswith("param_overrides."):
            unified[f"helix_param.{key.split('.', 1)[1]}"] = value

    breakout = _read_json(repo_root / "backtests/swing/auto/output/breakout_optimized_seed_summary.json")
    for key, value in (breakout.get("mutations") or {}).items():
        if key.startswith("flags."):
            unified[f"breakout_flags.{key.split('.', 1)[1]}"] = value
        elif key.startswith("param_overrides."):
            unified[f"breakout_param.{key.split('.', 1)[1]}"] = value

    brs_state = _read_json(repo_root / "backtests/swing/auto/brs/output/phase_state.json")
    brs.update(brs_state.get("cumulative_mutations") or {})
    return unified, brs


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_unified_trades(result) -> list[Any]:
    trades: list[Any] = []
    for attr in ("atrss_trades", "helix_trades", "brs_trades", "breakout_trades"):
        trades.extend(getattr(result, attr, []) or [])
    return trades


def _strategy_trade_counts(unified_result) -> dict[str, int]:
    counts = {
        "ATRSS": len(getattr(unified_result, "atrss_trades", []) or []),
        "AKC_HELIX": len(getattr(unified_result, "helix_trades", []) or []),
        "SWING_BREAKOUT_V3": len(getattr(unified_result, "breakout_trades", []) or []),
        "BRS_R9": len(getattr(unified_result, "brs_trades", []) or []),
    }
    return {strategy: counts.get(strategy, 0) for strategy in ACTIVE_STRATEGIES}


def _strategy_pnls(unified_result) -> dict[str, float]:
    pnls = {
        "ATRSS": sum(_trade_pnl(trade) - _trade_commission(trade) for trade in getattr(unified_result, "atrss_trades", []) or []),
        "AKC_HELIX": sum(_trade_pnl(trade) - _trade_commission(trade) for trade in getattr(unified_result, "helix_trades", []) or []),
        "SWING_BREAKOUT_V3": sum(_trade_pnl(trade) - _trade_commission(trade) for trade in getattr(unified_result, "breakout_trades", []) or []),
        "BRS_R9": sum(_trade_pnl(trade) - _trade_commission(trade) for trade in getattr(unified_result, "brs_trades", []) or []),
        "OVERLAY": float(getattr(unified_result, "overlay_pnl", 0.0)),
    }
    return pnls


def _combine_equity_curves(
    unified_equity: np.ndarray,
    unified_timestamps: np.ndarray,
    brs_equity: np.ndarray,
    brs_timestamps: np.ndarray,
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    pieces = []
    for equity, timestamps in ((unified_equity, unified_timestamps), (brs_equity, brs_timestamps)):
        if len(equity) == 0 or len(timestamps) == 0:
            continue
        n = min(len(equity), len(timestamps))
        index = pd.DatetimeIndex(timestamps[:n])
        series = pd.Series(np.asarray(equity[:n], dtype=np.float64) - initial_equity, index=index)
        pieces.append(series[~series.index.duplicated(keep="last")])

    if not pieces:
        return np.array([initial_equity], dtype=np.float64), np.array([], dtype="datetime64[ns]")

    union = pieces[0].index
    for series in pieces[1:]:
        union = union.union(series.index)
    union = union.sort_values()
    combined_delta = pd.Series(0.0, index=union)
    for series in pieces:
        combined_delta = combined_delta.add(series.reindex(union).ffill().fillna(0.0), fill_value=0.0)
    return (initial_equity + combined_delta.to_numpy(dtype=np.float64)), union.to_numpy()


def _positive_years(equity: np.ndarray, timestamps: np.ndarray) -> int:
    if len(equity) == 0 or len(timestamps) == 0:
        return 0
    n = min(len(equity), len(timestamps))
    frame = pd.DataFrame({"equity": np.asarray(equity[:n], dtype=np.float64)}, index=pd.DatetimeIndex(timestamps[:n]))
    yearly = frame["equity"].resample("YE").agg(["first", "last"]).dropna()
    if yearly.empty:
        return 0
    return int(((yearly["last"] - yearly["first"]) > 0).sum())


def _months(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 1.0
    start = pd.Timestamp(timestamps[0])
    end = pd.Timestamp(timestamps[-1])
    span_days = max((end - start).total_seconds() / 86400.0, 1.0)
    return span_days / 30.4375


def _total_r(trades: list[Any]) -> float:
    total = 0.0
    for trade in trades:
        risk = _trade_risk(trade)
        if risk > 0:
            total += (_trade_pnl(trade) - _trade_commission(trade)) / risk
    return float(total)


def _trade_pnl(trade: Any) -> float:
    if hasattr(trade, "pnl_dollars"):
        return float(getattr(trade, "pnl_dollars") or 0.0)
    if hasattr(trade, "pnl"):
        return float(getattr(trade, "pnl") or 0.0)
    return 0.0


def _trade_commission(trade: Any) -> float:
    return float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_risk(trade: Any) -> float:
    for attr in ("initial_risk_dollars", "risk_dollars", "initial_risk"):
        value = getattr(trade, attr, None)
        if value:
            return abs(float(value))
    if hasattr(trade, "initial_stop") and hasattr(trade, "entry_price"):
        qty = abs(float(getattr(trade, "qty", 1) or 1))
        risk = abs(float(getattr(trade, "entry_price") or 0.0) - float(getattr(trade, "initial_stop") or 0.0)) * qty
        if risk > 0:
            return risk
    return 1.0


def _trade_hold_hours(trade: Any) -> float:
    if hasattr(trade, "bars_held"):
        return float(getattr(trade, "bars_held") or 0.0)
    entry = getattr(trade, "entry_time", None)
    exit_ = getattr(trade, "exit_time", None)
    if entry is not None and exit_ is not None:
        return float((exit_ - entry).total_seconds() / 3600.0)
    return 0.0


def _trade_symbol(trade: Any) -> str:
    return str(getattr(trade, "symbol", "") or "")


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _set_path(target: dict[str, Any], parts: list[str], value: Any) -> None:
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = deepcopy(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
