"""Re-baseline and optimize the current two-sleeve stock portfolio.

Selection is restricted to data through 2026-05-01.  The post-2026-05-01
chronological block is generated and evaluated only after the final portfolio
configuration has been serialized and hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.auto.portfolio_synergy.core.logic import (
    CURRENT_ALCB_ID,
    CURRENT_IARIC_ID,
    run_portfolio_replay,
)
from backtests.stock.auto.portfolio_synergy.evaluator import (
    StrategyTradeBundle,
    _headline_mtm_metrics,
    load_trade_records,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.data.replay_cache import load_research_replay_bundle
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.iaric_daily_residual_replay import (
    DailyResidualReplayResult,
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)
from backtests.stock.models import Direction, TradeRecord
from strategies.stock.iaric.config import StrategySettings


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
OUTPUT_DIR = REPO_ROOT / "backtests/output/stock/portfolio_synergy/round_4"
START = date(2024, 3, 25)
IS_END = date(2026, 3, 1)
RESEARCH_END = date(2026, 5, 1)
LOCKBOX_START = date(2026, 5, 2)
LOCKBOX_END = date(2026, 7, 10)
INITIAL_EQUITY = 25_000.0

CURRENT_IARIC_CONFIG = REPO_ROOT / "backtests/output/stock/iaric/round_3/optimized_config.json"
CURRENT_IARIC_TRADES = REPO_ROOT / "backtests/output/stock/iaric/round_3/final_trades.json"
CURRENT_ALCB_CONFIG = REPO_ROOT / "backtests/output/stock/alcb/round_3/optimized_config.json"
CURRENT_ALCB_TRADES = REPO_ROOT / "backtests/output/stock/alcb/round_3/final_trades.json"
OLD_IARIC_TRADES = (
    REPO_ROOT
    / "backtests/output/stock/iaric/archive/20260816_022651_pre_recovery_reset/round_2/final_trades.json"
)
OLD_ALCB_CONFIG = REPO_ROOT / "backtests/output/stock/alcb/round_2/optimized_config.json"

FOLDS = (
    ("f1", date(2024, 3, 25), date(2024, 7, 31)),
    ("f2", date(2024, 8, 1), date(2024, 11, 30)),
    ("f3", date(2024, 12, 1), date(2025, 3, 31)),
    ("f4", date(2025, 4, 1), date(2025, 7, 31)),
    ("f5", date(2025, 8, 1), date(2025, 11, 30)),
    ("f6", date(2025, 12, 1), IS_END),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _status(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, default=str), flush=True)


def _filter(
    trades: Iterable[TradeRecord], start: date, end: date
) -> tuple[TradeRecord, ...]:
    return tuple(trade for trade in trades if start <= trade.entry_time.date() <= end)


def _bundle(
    alcb: Iterable[TradeRecord], iaric: Iterable[TradeRecord], start: date, end: date
) -> StrategyTradeBundle:
    return StrategyTradeBundle(_filter(alcb, start, end), _filter(iaric, start, end))


def neutral_config(*, equal_risk: bool = False) -> dict[str, Any]:
    iaric_risk = 0.005 if equal_risk else 0.002375
    alcb_risk = 0.005 if equal_risk else 0.00702
    return {
        "initial_equity": INITIAL_EQUITY,
        "risk_stance": "current_contract_rebaseline",
        "portfolio_rules": {
            "reference_risk_pct": 0.005,
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
                "unit_risk_pct": iaric_risk,
                "max_heat_R": 99.0,
                "max_concurrent": 99,
                "daily_stop_R": 0.0,
                "priority": 0,
                "role": "multi-session daily residual reversion",
            },
            CURRENT_ALCB_ID: {
                "unit_risk_pct": alcb_risk,
                "max_heat_R": 99.0,
                "max_concurrent": 99,
                "daily_stop_R": 0.0,
                "priority": 0,
                "role": "intraday breakout complement",
            },
        },
        "dynamic_allocation": {
            "enabled": False,
            "lookback_trades": 60,
            "min_mult": 0.75,
            "max_mult": 1.15,
            "positive_expectancy_boost": 0.10,
            "negative_expectancy_cut": 0.20,
        },
        "cross_strategy_rules": {
            "candidate_rank_mode": "strategy_priority",
            "same_symbol_policy": "none",
            "same_symbol_size_mult": 1.0,
            "same_sector_heat_cap_R": 99.0,
            "intraday_reserved_slots": 0,
            "intraday_reserved_heat_R": 0.0,
        },
        "strategy_filters": {CURRENT_IARIC_ID: {}, CURRENT_ALCB_ID: {}},
    }


def research_seed_config() -> dict[str, Any]:
    config = neutral_config()
    config["portfolio_rules"].update(
        {
            "heat_cap_R": 10.0,
            "max_total_active_positions": 19,
            "max_symbol_heat_R": 3.0,
            "max_long_heat_R": 10.0,
            "portfolio_daily_stop_R": 3.5,
            "portfolio_weekly_stop_R": 8.0,
            "drawdown_tiers": (
                (0.04, 1.0),
                (0.08, 0.75),
                (0.12, 0.40),
                (0.16, 0.0),
            ),
        }
    )
    config["strategy_allocations"][CURRENT_IARIC_ID].update(
        {"max_heat_R": 6.0, "max_concurrent": 12, "daily_stop_R": 2.75}
    )
    config["strategy_allocations"][CURRENT_ALCB_ID].update(
        {"max_heat_R": 8.0, "max_concurrent": 7, "daily_stop_R": 2.35}
    )
    config["cross_strategy_rules"].update(
        {"same_symbol_policy": "half_size", "same_symbol_size_mult": 0.5}
    )
    return config


def incumbent_config() -> dict[str, Any]:
    config = research_seed_config()
    config["portfolio_rules"].update(
        {
            "reference_risk_pct": 0.0075168,
            "heat_cap_R": 6.5,
            "max_total_active_positions": 12,
            "max_symbol_heat_R": 2.2,
            "max_long_heat_R": 6.25,
        }
    )
    config["strategy_allocations"][CURRENT_IARIC_ID].update(
        {"unit_risk_pct": 0.00864, "max_heat_R": 5.4, "max_concurrent": 9}
    )
    config["strategy_allocations"][CURRENT_ALCB_ID].update(
        {"unit_risk_pct": 0.00702, "max_heat_R": 4.0, "max_concurrent": 6}
    )
    config["dynamic_allocation"]["enabled"] = True
    config["cross_strategy_rules"].update(
        {"candidate_rank_mode": "diagnostic_alpha_score", "same_sector_heat_cap_R": 3.8}
    )
    return config


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = deepcopy(value)


def _patched(config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for path, value in patch.items():
        _set_path(result, path, value)
    return result


def _metrics(
    bundle: StrategyTradeBundle,
    config: dict[str, Any],
    *,
    mtm: bool = False,
) -> tuple[dict[str, Any], Any]:
    result = run_portfolio_replay(bundle.alcb_trades, bundle.iaric_trades, config)
    metrics = dict(result.metrics)
    if mtm:
        metrics.update(
            _headline_mtm_metrics(
                result.state.accepted_positions,
                metrics,
                initial_equity=float(config["initial_equity"]),
                data_dir=DATA_DIR,
            )
        )
    return metrics, result


def _utility(metrics: dict[str, Any]) -> float:
    ret = float(metrics.get("net_return_pct", 0.0) or 0.0)
    dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    pf = min(max(float(metrics.get("profit_factor", 0.0) or 0.0), 0.01), 4.0)
    capture = float(metrics.get("trade_capture_ratio", 0.0) or 0.0)
    concentration = float(metrics.get("max_strategy_trade_share", 1.0) or 1.0)
    return ret - 1.35 * dd + 0.018 * math.log(pf) + 0.012 * capture - 0.02 * max(
        concentration - 0.85, 0.0
    )


def _candidate_evaluation(
    name: str,
    config: dict[str, Any],
    is_bundle: StrategyTradeBundle,
    all_alcb: tuple[TradeRecord, ...],
    all_iaric: tuple[TradeRecord, ...],
) -> dict[str, Any]:
    aggregate, _ = _metrics(is_bundle, config)
    folds: dict[str, dict[str, Any]] = {}
    fold_utilities: list[float] = []
    for fold_name, start, end in FOLDS:
        fold_metrics, _ = _metrics(_bundle(all_alcb, all_iaric, start, end), config)
        folds[fold_name] = fold_metrics
        fold_utilities.append(_utility(fold_metrics))
    positive_folds = sum(float(row.get("net_pnl", 0.0)) > 0.0 for row in folds.values())
    robust_score = (
        _utility(aggregate)
        + 0.50 * float(np.median(fold_utilities))
        + 0.25 * min(fold_utilities)
        + 0.005 * positive_folds
    )
    eligible = (
        int(aggregate.get("active_strategy_count", 0)) == 2
        and float(aggregate.get("profit_factor", 0.0)) >= 1.15
        and float(aggregate.get("max_drawdown_pct", 1.0)) <= 0.20
        and positive_folds >= 4
        and float(aggregate.get("max_strategy_trade_share", 1.0)) <= 0.95
    )
    return {
        "name": name,
        "config": config,
        "aggregate": aggregate,
        "folds": folds,
        "positive_folds": positive_folds,
        "robust_score": robust_score,
        "eligible": eligible,
    }


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    return max(eligible or rows, key=lambda row: float(row["robust_score"]))


def _run_alcb(config_path: Path, *, end: date) -> tuple[TradeRecord, ...]:
    mutations = json.loads(config_path.read_text(encoding="utf-8"))
    # No frozen authority bundle currently exists in this workspace.  Legacy replay is
    # therefore explicit and the promotion gate below remains closed.
    replay = load_research_replay_bundle(DATA_DIR, require_bundle=False).data
    config = mutate_alcb_config(
        ALCBBacktestConfig(
            start_date=START.isoformat(),
            end_date=end.isoformat(),
            initial_equity=10_000.0,
            tier=2,
            data_dir=DATA_DIR,
        ),
        hydrate_time_mutations(mutations),
    )
    return tuple(ALCBIntradayEngine(config, replay).run().trades)


def _residual_records(result: DailyResidualReplayResult) -> tuple[TradeRecord, ...]:
    records = []
    for trade in result.trades:
        qty = float(trade.qty_entry)
        records.append(
            TradeRecord(
                strategy="iaric_daily_residual",
                symbol=trade.symbol,
                direction=Direction.LONG,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time or trade.entry_time,
                entry_price=float(trade.entry_price),
                exit_price=float(trade.exit_price),
                quantity=qty,
                pnl=float(trade.gross_pnl),
                r_multiple=float(trade.r_multiple),
                risk_per_share=float(trade.initial_risk_dollars) / max(qty, 1.0),
                commission=float(trade.commission),
                slippage=0.0,
                entry_type="DAILY_RESIDUAL_REVERSION",
                exit_reason=trade.exit_reason,
                sector=trade.sector,
                hold_bars=int(trade.held_sessions),
                metadata={
                    "residual_score": float(trade.score),
                    "failed_continuation_r": float(trade.failed_continuation_r),
                    "sector_return_5d": float(trade.sector_return_5d),
                    "factor_model": trade.factor_model,
                    "formation_sessions": int(trade.formation_sessions),
                    "residual_lane_id": trade.residual_lane_id,
                },
                signal_time=trade.entry_time,
                fill_time=trade.entry_time,
            )
        )
    return tuple(records)


def _run_current_iaric_full() -> tuple[tuple[TradeRecord, ...], dict[str, Any]]:
    settings_payload = json.loads(CURRENT_IARIC_CONFIG.read_text(encoding="utf-8"))["settings"]
    settings = StrategySettings(**settings_payload)
    close, open_, high, low, volume, sectors, paths = _load_daily_panel_unsealed(
        DATA_DIR, LOCKBOX_END
    )
    fingerprint, _rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
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
    parity_result = run_daily_residual_replay(
        residual_bundle,
        settings,
        start=START,
        end=RESEARCH_END,
        initial_equity=100_000.0,
        round_trip_cost_bps=20.0,
    )
    full_result = run_daily_residual_replay(
        residual_bundle,
        settings,
        start=START,
        end=LOCKBOX_END,
        initial_equity=100_000.0,
        round_trip_cost_bps=20.0,
    )
    stored = load_trade_records(CURRENT_IARIC_TRADES)
    parity = _parity_receipt(stored, _residual_records(parity_result), "iaric_current")
    parity.update(
        {
            "source_fingerprint": fingerprint,
            "shared_core_contract": full_result.shared_core_contract,
            "full_run_metrics": full_result.metrics(),
        }
    )
    return _residual_records(full_result), parity


def _load_daily_panel_unsealed(
    data_dir: Path, end: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], list[Path]]:
    from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS
    from strategies.stock.iaric.daily_residual_selection import SECTOR_REFERENCE
    from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS

    metadata = {symbol: sector for symbol, sector, _exchange in SP500_CONSTITUENTS}
    stocks = sorted(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    references = sorted({"SPY", *SECTOR_REFERENCE.values()})
    available = {path.stem[:-3]: path for path in data_dir.glob("*_1d.parquet")}
    requested = [*stocks, *references]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise RuntimeError("missing daily residual files: " + ", ".join(missing))
    series: dict[str, dict[str, pd.Series]] = {
        field: {} for field in ("open", "high", "low", "close", "volume")
    }
    for symbol in requested:
        frame = pd.read_parquet(
            available[symbol],
            columns=["open", "high", "low", "close", "volume"],
            filters=[("time", "<=", pd.Timestamp(f"{end.isoformat()} 23:59:59", tz="UTC"))],
        )
        index = pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)
        frame = frame.set_axis(index).sort_index().loc["2023-06-01" : end.isoformat()]
        for field in series:
            series[field][symbol] = frame[field].astype(float)
    close = pd.DataFrame(series["close"]).sort_index()
    return (
        close,
        pd.DataFrame(series["open"]).reindex(close.index),
        pd.DataFrame(series["high"]).reindex(close.index),
        pd.DataFrame(series["low"]).reindex(close.index),
        pd.DataFrame(series["volume"]).reindex(close.index),
        {symbol: metadata[symbol] for symbol in stocks},
        [available[symbol] for symbol in requested],
    )


def _parity_receipt(
    expected: Iterable[TradeRecord], actual: Iterable[TradeRecord], name: str
) -> dict[str, Any]:
    expected = tuple(expected)
    actual = tuple(actual)
    expected_r = sum(trade.r_multiple for trade in expected)
    actual_r = sum(trade.r_multiple for trade in actual)
    return {
        "name": name,
        "expected_trades": len(expected),
        "actual_trades": len(actual),
        "expected_total_r": expected_r,
        "actual_total_r": actual_r,
        "trade_count_match": len(expected) == len(actual),
        "total_r_abs_delta": abs(expected_r - actual_r),
        "passed": len(expected) == len(actual) and abs(expected_r - actual_r) <= 1e-6,
    }


def _risk_grid(seed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for ref in (0.004, 0.005, 0.006):
        for iaric in (0.00190, 0.002375, 0.00285, 0.003325):
            for alcb in (0.00560, 0.00702, 0.00840):
                patch = {
                    "portfolio_rules.reference_risk_pct": ref,
                    f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct": iaric,
                    f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct": alcb,
                }
                rows.append((f"risk_ref{ref}_i{iaric}_a{alcb}", _patched(seed, patch)))
    return rows


def _structural_groups() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "capacity",
            [
                {"portfolio_rules.max_total_active_positions": value}
                for value in (12, 16, 19, 22)
            ]
            + [{"portfolio_rules.heat_cap_R": value} for value in (6.0, 8.0, 10.0, 12.0)],
        ),
        (
            "sleeve_caps",
            [
                {
                    f"strategy_allocations.{CURRENT_IARIC_ID}.max_heat_R": i_heat,
                    f"strategy_allocations.{CURRENT_ALCB_ID}.max_heat_R": a_heat,
                }
                for i_heat, a_heat in ((3.0, 5.0), (5.0, 7.0), (6.0, 8.0), (8.0, 10.0))
            ],
        ),
        (
            "intraday_reserve",
            [
                {
                    "cross_strategy_rules.intraday_reserved_slots": slots,
                    "cross_strategy_rules.intraday_reserved_heat_R": heat,
                }
                for slots, heat in ((0, 0.0), (2, 0.0), (4, 0.0), (0, 1.5), (2, 1.5), (4, 2.5))
            ],
        ),
        (
            "collision_sector",
            [
                {
                    "cross_strategy_rules.same_symbol_policy": policy,
                    "cross_strategy_rules.same_sector_heat_cap_R": sector,
                }
                for policy, sector in (
                    ("none", 99.0),
                    ("half_size", 99.0),
                    ("block", 99.0),
                    ("half_size", 3.0),
                    ("half_size", 5.0),
                )
            ],
        ),
        (
            "loss_governors",
            [
                {
                    "portfolio_rules.portfolio_daily_stop_R": daily,
                    "portfolio_rules.portfolio_weekly_stop_R": weekly,
                }
                for daily, weekly in ((0.0, 0.0), (2.5, 6.0), (3.5, 8.0), (4.5, 10.0))
            ],
        ),
    ]


def _targeted_candidates(config: dict[str, Any], top_blocker: str) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        (
            "rank_alpha_per_heat",
            _patched(config, {"cross_strategy_rules.candidate_rank_mode": "expected_alpha_per_heat"}),
        ),
        (
            "rank_diagnostic_quality",
            _patched(config, {"cross_strategy_rules.candidate_rank_mode": "diagnostic_alpha_score"}),
        ),
        (
            "dynamic_40_defensive",
            _patched(
                config,
                {
                    "dynamic_allocation.enabled": True,
                    "dynamic_allocation.lookback_trades": 40,
                    "dynamic_allocation.min_mult": 0.75,
                    "dynamic_allocation.max_mult": 1.12,
                    "dynamic_allocation.positive_expectancy_boost": 0.08,
                    "dynamic_allocation.negative_expectancy_cut": 0.22,
                },
            ),
        ),
        (
            "dynamic_80_slow",
            _patched(
                config,
                {
                    "dynamic_allocation.enabled": True,
                    "dynamic_allocation.lookback_trades": 80,
                    "dynamic_allocation.min_mult": 0.80,
                    "dynamic_allocation.max_mult": 1.10,
                },
            ),
        ),
        (
            "alcb_pdh_quality_tilt",
            _patched(
                config,
                {
                    f"strategy_filters.{CURRENT_ALCB_ID}.pdh_size_mult": 1.10,
                    f"strategy_filters.{CURRENT_ALCB_ID}.financials_size_mult": 0.75,
                    f"strategy_filters.{CURRENT_ALCB_ID}.score5_no_surge_mult": 0.75,
                },
            ),
        ),
    ]
    if top_blocker in {"portfolio_heat_cap", "long_heat_cap"}:
        rows.append(
            (
                "target_top_heat_blocker",
                _patched(
                    config,
                    {
                        "portfolio_rules.heat_cap_R": float(config["portfolio_rules"]["heat_cap_R"]) * 1.15,
                        "portfolio_rules.max_long_heat_R": float(config["portfolio_rules"]["max_long_heat_R"]) * 1.15,
                    },
                ),
            )
        )
    elif top_blocker in {"max_total_active_positions", "intraday_reserved_slots"}:
        rows.append(
            (
                "target_top_slot_blocker",
                _patched(
                    config,
                    {
                        "portfolio_rules.max_total_active_positions": int(
                            config["portfolio_rules"]["max_total_active_positions"]
                        )
                        + 2
                    },
                ),
            )
        )
    rows.append(("target_control", deepcopy(config)))
    return rows


def _stress_trades(
    trades: Iterable[TradeRecord], extra_round_trip_bps: float
) -> tuple[TradeRecord, ...]:
    stressed = []
    for trade in trades:
        source_risk = float(trade.risk_per_share * trade.quantity)
        notional = abs(float(trade.entry_price * trade.quantity))
        delta_r = notional * extra_round_trip_bps / 10_000.0 / max(source_risk, 1e-9)
        stressed.append(replace(trade, r_multiple=float(trade.r_multiple) - delta_r))
    return tuple(stressed)


def _robustness(
    final_config: dict[str, Any],
    research_bundle: StrategyTradeBundle,
) -> dict[str, Any]:
    control_metrics, control_result = _metrics(research_bundle, final_config, mtm=True)
    ablation_patches = {
        "ablate_position_caps": {
            "portfolio_rules.max_total_active_positions": 99,
            f"strategy_allocations.{CURRENT_IARIC_ID}.max_concurrent": 99,
            f"strategy_allocations.{CURRENT_ALCB_ID}.max_concurrent": 99,
        },
        "ablate_heat_caps": {
            "portfolio_rules.heat_cap_R": 99.0,
            "portfolio_rules.max_long_heat_R": 99.0,
            f"strategy_allocations.{CURRENT_IARIC_ID}.max_heat_R": 99.0,
            f"strategy_allocations.{CURRENT_ALCB_ID}.max_heat_R": 99.0,
        },
        "ablate_loss_stops": {
            "portfolio_rules.portfolio_daily_stop_R": 0.0,
            "portfolio_rules.portfolio_weekly_stop_R": 0.0,
            f"strategy_allocations.{CURRENT_IARIC_ID}.daily_stop_R": 0.0,
            f"strategy_allocations.{CURRENT_ALCB_ID}.daily_stop_R": 0.0,
        },
        "ablate_drawdown_tiers": {"portfolio_rules.drawdown_tiers": ((1.0, 1.0),)},
        "ablate_collision_sector": {
            "cross_strategy_rules.same_symbol_policy": "none",
            "cross_strategy_rules.same_sector_heat_cap_R": 99.0,
        },
        "ablate_intraday_reserve": {
            "cross_strategy_rules.intraday_reserved_slots": 0,
            "cross_strategy_rules.intraday_reserved_heat_R": 0.0,
        },
        "ablate_dynamic": {"dynamic_allocation.enabled": False},
    }
    ablations = []
    for name, patch in ablation_patches.items():
        metrics, _ = _metrics(research_bundle, _patched(final_config, patch))
        ablations.append({"name": name, "patch": patch, "metrics": metrics, "utility_delta": _utility(metrics) - _utility(control_metrics)})

    perturbations = []
    numeric_paths = [
        "portfolio_rules.reference_risk_pct",
        "portfolio_rules.heat_cap_R",
        "portfolio_rules.max_long_heat_R",
        "portfolio_rules.portfolio_daily_stop_R",
        "portfolio_rules.portfolio_weekly_stop_R",
        f"strategy_allocations.{CURRENT_IARIC_ID}.unit_risk_pct",
        f"strategy_allocations.{CURRENT_IARIC_ID}.max_heat_R",
        f"strategy_allocations.{CURRENT_ALCB_ID}.unit_risk_pct",
        f"strategy_allocations.{CURRENT_ALCB_ID}.max_heat_R",
        "cross_strategy_rules.intraday_reserved_heat_R",
    ]
    for path in numeric_paths:
        value = _get_path(final_config, path)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            continue
        for multiplier in (0.90, 1.10):
            patch = {path: float(value) * multiplier}
            metrics, _ = _metrics(research_bundle, _patched(final_config, patch))
            perturbations.append({"name": f"{path}__x{multiplier}", "patch": patch, "metrics": metrics, "utility_delta": _utility(metrics) - _utility(control_metrics)})

    costs = []
    for bps in (5.0, 10.0, 20.0):
        bundle = StrategyTradeBundle(
            _stress_trades(research_bundle.alcb_trades, bps),
            _stress_trades(research_bundle.iaric_trades, bps),
        )
        metrics, _ = _metrics(bundle, final_config)
        costs.append({"extra_round_trip_bps": bps, "metrics": metrics})

    months = sorted({trade.entry_time.strftime("%Y-%m") for trade in (*research_bundle.alcb_trades, *research_bundle.iaric_trades)})
    leave_month = []
    for month in months:
        bundle = StrategyTradeBundle(
            tuple(t for t in research_bundle.alcb_trades if t.entry_time.strftime("%Y-%m") != month),
            tuple(t for t in research_bundle.iaric_trades if t.entry_time.strftime("%Y-%m") != month),
        )
        metrics, _ = _metrics(bundle, final_config)
        leave_month.append({"excluded": month, "metrics": metrics})

    sectors = sorted({trade.sector for trade in (*research_bundle.alcb_trades, *research_bundle.iaric_trades) if trade.sector})
    leave_sector = []
    for sector in sectors:
        bundle = StrategyTradeBundle(
            tuple(t for t in research_bundle.alcb_trades if t.sector != sector),
            tuple(t for t in research_bundle.iaric_trades if t.sector != sector),
        )
        metrics, _ = _metrics(bundle, final_config)
        leave_sector.append({"excluded": sector, "metrics": metrics})

    weekly: dict[str, float] = defaultdict(float)
    for outcome in control_result.trade_outcomes:
        iso = outcome.exit_time.isocalendar()
        weekly[f"{iso.year:04d}-W{iso.week:02d}"] += outcome.net_pnl
    values = np.array(list(weekly.values()), dtype=float)
    rng = np.random.default_rng(20260823)
    boot = np.array([rng.choice(values, size=len(values), replace=True).sum() for _ in range(2000)]) if len(values) else np.array([0.0])
    bootstrap = {
        "samples": int(len(boot)),
        "probability_total_pnl_positive": float(np.mean(boot > 0.0)),
        "ci_95_total_pnl": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }
    return {
        "control": control_metrics,
        "ablations": ablations,
        "perturbations": perturbations,
        "cost_stress": costs,
        "leave_one_month_out": leave_month,
        "leave_one_sector_out": leave_sector,
        "weekly_block_bootstrap": bootstrap,
    }


def _get_path(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


def _render_report(
    selected: dict[str, Any],
    attribution: list[dict[str, Any]],
    robustness: dict[str, Any],
    lockbox: dict[str, Any],
    promotion: dict[str, Any],
) -> str:
    metrics = selected["aggregate"]
    lock = lockbox["metrics"]
    lines = [
        "# Stock portfolio current-contract re-baseline — Round 4",
        "",
        f"Status: **{promotion['status']}**",
        "",
        "## Selected research configuration",
        "",
        f"- Candidate: `{selected['name']}`",
        f"- IS return / PF / realized DD: {metrics['net_return_pct']:.2%} / {metrics['profit_factor']:.2f} / {metrics['max_drawdown_pct']:.2%}",
        f"- Positive chronological folds: {selected['positive_folds']}/6",
        f"- Configuration SHA-256: `{promotion['config_sha256']}`",
        "",
        "## Untouched post-selection lockbox",
        "",
        f"- Window: {LOCKBOX_START} through {LOCKBOX_END}",
        f"- Trades: {int(lock['total_trades'])}",
        f"- Return / PF / MTM DD: {lock['net_return_pct']:.2%} / {lock['profit_factor']:.2f} / {lock['max_drawdown_pct']:.2%}",
        f"- IARIC / ALCB trades: {int(lock.get('trades_'+CURRENT_IARIC_ID, 0))} / {int(lock.get('trades_'+CURRENT_ALCB_ID, 0))}",
        "",
        "## Interpretation",
        "",
        "The old portfolio optimum was retired as a research baseline because it referenced the pullback IARIC contract. Round 4 uses the residual IARIC stream, current ALCB stream, enforced loss stops, and explicit intraday capacity reservation.",
        "",
        f"Weekly block-bootstrap P(PnL > 0): {robustness['weekly_block_bootstrap']['probability_total_pnl_positive']:.1%}.",
        f"Old/new attribution cells evaluated: {len(attribution)}.",
        "",
        "Production activation remains separate from this research promotion and requires live configuration synchronization plus shadow parity.",
    ]
    return "\n".join(lines) + "\n"


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "family": "stock",
        "strategy": "portfolio_synergy",
        "round": 4,
        "contract": "current_residual_iaric_alcb_r3_portfolio_v1",
        "selection_window": [START, IS_END],
        "consumed_research_validation": [date(2026, 3, 2), RESEARCH_END],
        "untouched_lockbox": [LOCKBOX_START, LOCKBOX_END],
        "initial_equity": INITIAL_EQUITY,
        "data_authority": {
            "mode": "legacy_cache_diagnostic_only",
            "frozen_bundle_present": False,
            "production_eligible": False,
        },
        "source_artifacts": [
            str(CURRENT_IARIC_CONFIG.relative_to(REPO_ROOT)),
            str(CURRENT_IARIC_TRADES.relative_to(REPO_ROOT)),
            str(CURRENT_ALCB_CONFIG.relative_to(REPO_ROOT)),
            str(CURRENT_ALCB_TRADES.relative_to(REPO_ROOT)),
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "run_spec.json", run_spec)

    _status("load_current_and_predecessor_trades")
    new_iaric = load_trade_records(CURRENT_IARIC_TRADES)
    old_iaric = load_trade_records(OLD_IARIC_TRADES)
    new_alcb_research = _run_alcb(CURRENT_ALCB_CONFIG, end=RESEARCH_END)
    old_alcb_is = _run_alcb(OLD_ALCB_CONFIG, end=IS_END)
    stored_alcb = load_trade_records(CURRENT_ALCB_TRADES)
    alcb_parity = _parity_receipt(stored_alcb, _filter(new_alcb_research, START, IS_END), "alcb_current")
    _write_json(output / "preselection_parity.json", {"alcb": alcb_parity})

    is_new = _bundle(new_alcb_research, new_iaric, START, IS_END)
    research_new = _bundle(new_alcb_research, new_iaric, START, RESEARCH_END)
    validation_new = _bundle(new_alcb_research, new_iaric, date(2026, 3, 2), RESEARCH_END)

    _status("old_new_factorial_attribution")
    factorial_streams = {
        "old_iaric__old_alcb": _bundle(old_alcb_is, old_iaric, START, IS_END),
        "new_iaric__old_alcb": _bundle(old_alcb_is, new_iaric, START, IS_END),
        "old_iaric__new_alcb": _bundle(new_alcb_research, old_iaric, START, IS_END),
        "new_iaric__new_alcb": is_new,
    }
    attribution = []
    for rules_name, config in (("neutral_equal_risk", neutral_config(equal_risk=True)), ("incumbent_rules", incumbent_config())):
        for stream_name, bundle in factorial_streams.items():
            metrics, _ = _metrics(bundle, config)
            attribution.append({"rules": rules_name, "streams": stream_name, "metrics": metrics})
    _write_json(output / "old_new_factorial_attribution.json", attribution)

    _status("risk_grid", candidates=36)
    seed = research_seed_config()
    risk_rows = [
        _candidate_evaluation(name, config, is_new, new_alcb_research, new_iaric)
        for name, config in _risk_grid(seed)
    ]
    risk_winner = _select(risk_rows)
    _write_json(output / "risk_grid.json", {"selected": risk_winner["name"], "results": risk_rows})

    current = risk_winner
    stage_results = []
    for group_name, patches in _structural_groups():
        _status("structural_group", group=group_name, candidates=len(patches) + 1)
        candidates = [(f"{group_name}_control", deepcopy(current["config"]))]
        candidates.extend((f"{group_name}_{index:02d}", _patched(current["config"], patch)) for index, patch in enumerate(patches, start=1))
        rows = [
            _candidate_evaluation(name, config, is_new, new_alcb_research, new_iaric)
            for name, config in candidates
        ]
        winner = _select(rows)
        if winner["robust_score"] + 1e-9 >= current["robust_score"]:
            current = winner
        stage_results.append({"group": group_name, "selected": current["name"], "results": rows})
    _write_json(output / "structural_screen.json", stage_results)

    _, current_result = _metrics(is_new, current["config"])
    positive_block_value: dict[str, float] = defaultdict(float)
    for blocked in current_result.state.blocked_candidates:
        if blocked.r_multiple > 0:
            positive_block_value[blocked.reason] += blocked.r_multiple
    top_blocker = max(positive_block_value, key=positive_block_value.get) if positive_block_value else "none"
    _status("targeted_round", top_positive_blocker=top_blocker)
    targeted_rows = [
        _candidate_evaluation(name, config, is_new, new_alcb_research, new_iaric)
        for name, config in _targeted_candidates(current["config"], top_blocker)
    ]
    finalists = sorted(
        [*targeted_rows, current], key=lambda row: float(row["robust_score"]), reverse=True
    )[:8]
    for row in finalists:
        validation_metrics, _ = _metrics(validation_new, row["config"])
        row["research_validation"] = validation_metrics
        row["joint_score"] = float(row["robust_score"]) + 0.25 * _utility(validation_metrics)
        row["validation_gate"] = (
            float(validation_metrics.get("net_pnl", 0.0)) > 0.0
            and float(validation_metrics.get("profit_factor", 0.0)) >= 1.0
            and int(validation_metrics.get("active_strategy_count", 0)) == 2
        )
    selected_pool = [row for row in finalists if row["eligible"] and row["validation_gate"]]
    selected = max(selected_pool or finalists, key=lambda row: float(row["joint_score"]))
    _write_json(output / "targeted_round.json", {"top_positive_blocker": top_blocker, "selected": selected["name"], "results": targeted_rows, "finalists": finalists})

    final_config = selected["config"]
    _write_json(output / "optimized_config.json", final_config)
    config_sha = _sha(output / "optimized_config.json")
    _status("final_robustness", selected=selected["name"], config_sha256=config_sha)
    robustness = _robustness(final_config, research_new)
    _write_json(output / "final_robustness.json", robustness)

    freeze_receipt = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "selection_complete_before_lockbox_generation": True,
        "selected_candidate": selected["name"],
        "lockbox_window": [LOCKBOX_START, LOCKBOX_END],
    }
    _write_json(output / "freeze_receipt.json", freeze_receipt)

    _status("generate_untouched_lockbox_streams")
    new_alcb_full = _run_alcb(CURRENT_ALCB_CONFIG, end=LOCKBOX_END)
    new_iaric_full, iaric_parity = _run_current_iaric_full()
    parity = {"alcb": alcb_parity, "iaric": iaric_parity}
    _write_json(output / "deployment_parity.json", parity)

    lockbox_bundle = _bundle(new_alcb_full, new_iaric_full, LOCKBOX_START, LOCKBOX_END)
    lockbox_metrics, lockbox_result = _metrics(lockbox_bundle, final_config, mtm=True)
    lockbox = {
        "window": [LOCKBOX_START, LOCKBOX_END],
        "metrics": lockbox_metrics,
        "strategy_trade_counts": {
            CURRENT_IARIC_ID: len(lockbox_bundle.iaric_trades),
            CURRENT_ALCB_ID: len(lockbox_bundle.alcb_trades),
        },
        "config_sha256_rechecked": _sha(output / "optimized_config.json"),
        "config_unchanged_after_freeze": _sha(output / "optimized_config.json") == config_sha,
    }
    _write_json(output / "lockbox_validation.json", lockbox)

    perturbation_pass = all(
        float(row["metrics"].get("net_pnl", 0.0)) > 0.0
        for row in robustness["perturbations"]
    )
    cost_pass = all(
        float(row["metrics"].get("net_pnl", 0.0)) > 0.0
        for row in robustness["cost_stress"]
    )
    lockbox_gate = (
        float(lockbox_metrics.get("net_pnl", 0.0)) > 0.0
        and float(lockbox_metrics.get("profit_factor", 0.0)) >= 1.20
        and float(lockbox_metrics.get("max_drawdown_pct", 1.0)) <= 0.10
        and int(lockbox_metrics.get("active_strategy_count", 0)) == 2
    )
    parity_pass = all(row["passed"] for row in parity.values())
    data_authority_pass = bool(run_spec["data_authority"]["production_eligible"])
    research_promoted = (
        data_authority_pass
        and parity_pass
        and lockbox_gate
        and perturbation_pass
        and cost_pass
    )
    promotion = {
        "status": (
            "research_promoted_live_sync_and_shadow_required"
            if research_promoted
            else "research_only_not_promoted"
        ),
        "config_sha256": config_sha,
        "gates": {
            "frozen_data_authority": data_authority_pass,
            "deployment_parity": parity_pass,
            "untouched_lockbox": lockbox_gate,
            "local_perturbations_positive": perturbation_pass,
            "incremental_cost_stress_positive": cost_pass,
        },
        "production_activation_approved": False,
        "remaining_requirements": [
            "build and verify a frozen authoritative stock-data bundle",
            "synchronize live IARIC/ALCB and portfolio configuration",
            "pass live-shadow signal/order/fill parity",
        ],
    }
    _write_json(output / "promotion_decision.json", promotion)

    report = _render_report(selected, attribution, robustness, lockbox, promotion)
    (output / "round_final_diagnostics.md").write_text(report, encoding="utf-8")
    (output / "round_final_diagnostics.txt").write_text(report, encoding="utf-8")
    manifest = {
        "round": 4,
        "active": research_promoted,
        "status": promotion["status"],
        "contract": run_spec["contract"],
        "config_sha256": config_sha,
        "artifacts": {
            path.name: _sha(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    _write_json(output / "artifact_manifest.json", manifest)
    _status("complete", status=promotion["status"], lockbox=lockbox_metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
