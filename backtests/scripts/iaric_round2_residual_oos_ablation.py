"""Round-2 IARIC residual OOS attribution, ablation, and perturbation research.

This runner intentionally consumes the 2026-03-02..2026-05-01 sealed extension.
It is a diagnostic/research tool, not a promotion tool.  Results selected with
that extension require a later untouched chronological validation window.

The experiment is staged and resumable:

* ``lineage`` exact-replays every accepted cumulative prefix plus granular
  one-at-a-time removals spanning the Round-1 and Round-2 residual lineages.
* ``perturb`` runs broad one-lever neighbourhoods on the full post-selection
  window, then evaluates every promising/Pareto candidate in sample.
* ``targeted`` builds bounded combinations from the strongest distinct
  one-lever families and validates the surviving candidates across folds and
  cost stresses.
* ``all`` runs every stage and writes the final report.

The default output directory is isolated from the frozen round artifacts.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners.run_iaric_residual_phased_auto import (
    _load_round2_baseline,
)
from backtests.stock.engine.iaric_daily_residual_replay import (
    DailyResidualReplayBundle,
    DailyResidualReplayResult,
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUND_ROOT = REPO_ROOT / "backtests/output/stock/iaric/round_2"
CURRENT_CANDIDATE = (
    ROUND_ROOT
    / "phased_auto_alpha_v5_selective_sector_overflow/frozen_selection_candidate.json"
)
ROUND1_CONFIG = REPO_ROOT / "backtests/output/stock/iaric/round_1/optimized_config.json"
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
DEFAULT_OUTPUT = ROUND_ROOT / "oos_ablation_perturbation_corrected_split_20260823"

IS_START = date(2024, 3, 25)
IS_END = date(2026, 3, 1)
OOS_START = date(2026, 3, 2)
OOS_END = date(2026, 5, 1)
EARLY_OOS_END = date(2026, 3, 20)
LATEST_OOS_START = date(2026, 3, 21)

WINDOWS = {
    "is": (IS_START, IS_END),
    "oos": (OOS_START, OOS_END),
}
SUBWINDOWS = {
    "early_oos": (OOS_START, EARLY_OOS_END),
    "latest_oos": (LATEST_OOS_START, OOS_END),
}
DAILY_FIELDS = tuple(
    name
    for name in StrategySettings.__dataclass_fields__
    if name == "strategy_mode" or name.startswith("daily_residual_")
)
ACCELERATION_CONTRACT = "factor_sharded_process_and_persistent_prepared_features_v2"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.length = ctypes.sizeof(self)


def _available_memory_bytes() -> int | None:
    if os.name != "nt":
        return None
    status = _MemoryStatusEx()
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_physical)


def _effective_worker_count(requested: int) -> tuple[int, dict[str, Any]]:
    """Bound process parallelism by measured RAM, not logical CPU count."""

    available = _available_memory_bytes()
    hard_cap = 3
    if available is None:
        memory_cap = 2
    else:
        reserve = 1.0 * 1024**3
        measured_bundle_budget = 1.75 * 1024**3
        memory_cap = max(1, int(max(available - reserve, 0.0) // measured_bundle_budget))
    effective = max(1, min(int(requested), hard_cap, memory_cap))
    return effective, {
        "requested_workers": int(requested),
        "effective_workers": effective,
        "hard_cap": hard_cap,
        "available_memory_gb": (
            float(available) / 1024**3 if available is not None else None
        ),
        "reserved_memory_gb": 1.0,
        "estimated_bundle_memory_gb": 1.75,
        "reason": "processes are GIL-independent; RAM bounds exact factor bundles",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _settings_payload(settings: StrategySettings) -> dict[str, Any]:
    return {
        name: _jsonable(getattr(settings, name))
        for name in DAILY_FIELDS
    }


def _sha(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _months(start: date, end: date) -> float:
    return max((end - start).days + 1, 1) / 30.4375


def _trade_payload(trade: Any) -> dict[str, Any]:
    return _jsonable(asdict(trade))


def _period_metrics(
    trades: Iterable[Any],
    *,
    start: date,
    end: date,
    initial_equity: float = 100_000.0,
) -> dict[str, Any]:
    rows = [
        trade
        for trade in trades
        if start <= trade.entry_date <= end
    ]
    values = [float(trade.r_multiple) for trade in rows]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    pnls = [float(trade.net_pnl) for trade in rows]
    equity = float(initial_equity)
    peak = equity
    close_to_close_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        close_to_close_dd = max(close_to_close_dd, (peak - equity) / max(peak, 1e-9))
    sorted_values = sorted(values)
    tail_count = max(1, math.ceil(0.05 * len(values))) if values else 0
    top_losses = sorted_values[:tail_count]
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "trades": len(rows),
        "trades_per_month": len(rows) / _months(start, end),
        "total_r": sum(values),
        "r_per_month": sum(values) / _months(start, end),
        "average_r": fmean(values) if values else 0.0,
        "median_r": median(values) if values else 0.0,
        "win_rate": len(wins) / len(values) if values else 0.0,
        "profit_factor": (
            sum(wins) / abs(sum(losses))
            if losses
            else (float("inf") if wins else 0.0)
        ),
        "net_pnl": sum(pnls),
        "close_to_close_trade_drawdown_pct": close_to_close_dd,
        "expected_shortfall_r_5pct": fmean(top_losses) if top_losses else 0.0,
        "worst_r": min(values, default=0.0),
        "best_r": max(values, default=0.0),
        "positive_sectors": len(
            {
                trade.sector
                for trade in rows
                if float(trade.r_multiple) > 0.0
            }
        ),
    }


def _cohort_rows(trades: list[Any], key_fn) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        grouped[str(key_fn(trade))].append(trade)
    rows = []
    for key, group in grouped.items():
        values = [float(trade.r_multiple) for trade in group]
        rows.append(
            {
                "cohort": key,
                "trades": len(group),
                "total_r": sum(values),
                "average_r": fmean(values),
                "win_rate": sum(value > 0.0 for value in values) / len(values),
            }
        )
    return sorted(rows, key=lambda row: (-row["trades"], row["cohort"]))


def _baseline_diagnostics(
    oos_result: DailyResidualReplayResult,
    is_result: DailyResidualReplayResult,
) -> dict[str, Any]:
    trades = oos_result.trades
    oos = [trade for trade in trades if OOS_START <= trade.entry_date <= OOS_END]
    losses = sorted(oos, key=lambda trade: float(trade.r_multiple))
    total_loss_r = abs(sum(min(float(trade.r_multiple), 0.0) for trade in oos))
    top_loss_rows = [_trade_payload(trade) for trade in losses[:15]]
    removal = {}
    for count in (1, 2, 3, 5, 10):
        removed = losses[:count]
        removal[str(count)] = {
            "removed_r": sum(float(trade.r_multiple) for trade in removed),
            "counterfactual_total_r": (
                sum(float(trade.r_multiple) for trade in oos)
                - sum(float(trade.r_multiple) for trade in removed)
            ),
            "loss_share": (
                abs(sum(float(trade.r_multiple) for trade in removed)) / total_loss_r
                if total_loss_r > 0.0
                else 0.0
            ),
        }
    scores = [float(trade.score) for trade in oos]
    if scores:
        q1, q2, q3 = np.quantile(np.asarray(scores), [0.25, 0.50, 0.75])
        score_bucket = lambda trade: (
            "Q1"
            if float(trade.score) <= q1
            else "Q2"
            if float(trade.score) <= q2
            else "Q3"
            if float(trade.score) <= q3
            else "Q4"
        )
    else:
        score_bucket = lambda _trade: "empty"
    return {
        "window_metrics": {
            "is": _period_metrics(is_result.trades, start=IS_START, end=IS_END),
            "oos": _period_metrics(oos_result.trades, start=OOS_START, end=OOS_END),
            **{
                name: _period_metrics(oos_result.trades, start=start, end=end)
                for name, (start, end) in SUBWINDOWS.items()
            },
        },
        "loss_concentration": {
            "total_loss_r": total_loss_r,
            "worst_trade_removal_counterfactual": removal,
            "largest_losses": top_loss_rows,
        },
        "cohorts": {
            "month": _cohort_rows(oos, lambda trade: trade.entry_date.strftime("%Y-%m")),
            "sector": _cohort_rows(oos, lambda trade: trade.sector),
            "symbol": _cohort_rows(oos, lambda trade: trade.symbol),
            "exit_reason": _cohort_rows(oos, lambda trade: trade.exit_reason),
            "held_sessions": _cohort_rows(oos, lambda trade: trade.held_sessions),
            "weekday": _cohort_rows(oos, lambda trade: trade.entry_date.strftime("%A")),
            "score_quartile": _cohort_rows(oos, score_bucket),
            "failed_continuation_bin": _cohort_rows(
                oos,
                lambda trade: (
                    "lt0p25"
                    if trade.failed_continuation_r < 0.25
                    else "0p25_0p50"
                    if trade.failed_continuation_r < 0.50
                    else "0p50_1p00"
                    if trade.failed_continuation_r < 1.00
                    else "gte1p00"
                ),
            ),
            "sector_return_5d_bin": _cohort_rows(
                oos,
                lambda trade: (
                    "lt_m5pct"
                    if trade.sector_return_5d < -0.05
                    else "m5pct_m2pct"
                    if trade.sector_return_5d < -0.02
                    else "m2pct_0"
                    if trade.sector_return_5d < 0.0
                    else "gte0"
                ),
            ),
        },
        "decision_event_counts": dict(
            Counter(str(row.get("code", row.get("event_type", "unknown"))) for row in oos_result.decision_events)
        ),
    }


def _load_full_panel(data_dir: Path, end: date = OOS_END):
    """Load the explicitly consumed extension without the sealed-loader guard."""

    metadata = {symbol: sector for symbol, sector, _exchange in SP500_CONSTITUENTS}
    stocks = set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    references = {"SPY", *discovery.SECTOR_ETFS.values()}
    available = {path.stem[:-3]: path for path in data_dir.glob("*_1d.parquet")}
    missing = sorted((stocks | references) - set(available))
    if missing:
        raise RuntimeError("missing full-panel daily files: " + ", ".join(missing))
    symbols = [*sorted(stocks), *sorted(references)]
    frames: dict[str, dict[str, pd.Series]] = {
        field: {} for field in ("open", "high", "low", "close", "volume")
    }
    paths = [available[symbol] for symbol in symbols]
    for symbol in symbols:
        frame = pd.read_parquet(
            available[symbol],
            columns=["open", "high", "low", "close", "volume"],
            filters=[
                ("time", ">=", pd.Timestamp(discovery.WARMUP_START, tz="UTC")),
                ("time", "<=", pd.Timestamp(end.isoformat() + " 23:59:59", tz="UTC")),
            ],
        )
        index = pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)
        frame = frame.set_axis(index).sort_index().loc[discovery.WARMUP_START : end.isoformat()]
        for field in frames:
            frames[field][symbol] = frame[field].astype(float)
    close = pd.DataFrame(frames["close"]).sort_index()
    panels = {
        field: pd.DataFrame(values).reindex(close.index)
        for field, values in frames.items()
    }
    fingerprint, rows = discovery._selection_data_fingerprint(
        panels["close"],
        panels["open"],
        panels["high"],
        panels["low"],
        panels["volume"],
        paths,
    )
    return (
        panels["close"],
        panels["open"],
        panels["high"],
        panels["low"],
        panels["volume"],
        {symbol: metadata[symbol] for symbol in stocks},
        fingerprint,
        rows,
    )


def _candidate(
    name: str,
    group: str,
    base: StrategySettings,
    patch: Mapping[str, Any],
    *,
    family: str,
    note: str = "",
) -> dict[str, Any]:
    settings = replace(base, **dict(patch))
    return {
        "name": name,
        "group": group,
        "family": family,
        "patch": _jsonable(dict(patch)),
        "note": note,
        "settings": _settings_payload(settings),
        "settings_sha256": _sha(_settings_payload(settings)),
    }


def _round1_settings() -> StrategySettings:
    payload = _read_json(ROUND1_CONFIG)
    return StrategySettings(**payload["settings"])


def _lineage_candidates(current: StrategySettings) -> list[dict[str, Any]]:
    round1 = _round1_settings()
    feature = replace(
        round1,
        daily_residual_lane_id="round2_fresh_residual_quality_rejection_1d",
        daily_residual_minimum_score=25.0,
        daily_residual_score_components=(
            "volume_transition",
            "price_rejection_recovery",
        ),
    )
    cap12 = replace(feature, daily_residual_max_positions=12)
    risk25 = replace(cap12, daily_residual_risk_fraction=0.0025)
    prefixes = [
        ("prefix_round1", round1, "Frozen continuous-reconciled Round 1."),
        ("prefix_r2_feature_quality", feature, "Round-2 admission/ranking mutation."),
        ("prefix_r2_capacity12", cap12, "Accepted twelve-position neighbour."),
        ("prefix_r2_risk025", risk25, "Accepted 0.25% risk frontier."),
        ("current", current, "Final Round-2 champion after risk/notional synergy."),
    ]
    rows = [
        _candidate(
            name,
            "cumulative_prefix",
            settings,
            {},
            family="lineage_prefix",
            note=note,
        )
        for name, settings, note in prefixes
    ]
    ablations: list[tuple[str, str, dict[str, Any]]] = [
        ("ablate_r2_price_rejection", "r2_price_rejection", {"daily_residual_score_components": ("volume_transition",)}),
        ("ablate_r2_score_floor", "r2_score_floor", {"daily_residual_minimum_score": 0.0}),
        ("ablate_r2_feature_and_floor", "r2_feature_quality", {"daily_residual_score_components": ("volume_transition",), "daily_residual_minimum_score": 0.0}),
        ("ablate_r2_lane_label", "r2_lane", {"daily_residual_lane_id": round1.daily_residual_lane_id}),
        ("ablate_r2_capacity12", "r2_capacity", {"daily_residual_max_positions": 10}),
        ("ablate_r2_risk_to_phase10", "r2_phase12_risk", {"daily_residual_risk_fraction": 0.0025}),
        ("ablate_r2_notional12", "r2_phase12_notional", {"daily_residual_maximum_notional_fraction": 0.10}),
        ("ablate_r2_phase12_joint", "r2_phase12_joint", {"daily_residual_risk_fraction": 0.0025, "daily_residual_maximum_notional_fraction": 0.10}),
        ("ablate_r2_all_risk_reduction", "r2_risk", {"daily_residual_risk_fraction": 0.0035}),
        ("ablate_r1_factor_model", "r1_factor_model", {"daily_residual_factor_model": "peer_demeaned"}),
        ("ablate_r1_formation", "r1_formation", {"daily_residual_formation_sessions": 5}),
        ("ablate_r1_volume_component", "r1_score_component", {"daily_residual_score_components": ("failed_continuation", "price_rejection_recovery")}),
        ("ablate_r1_failed_continuation_gate", "r1_failed_continuation", {"daily_residual_minimum_failed_continuation_r": 0.20}),
        ("ablate_r1_market_gate", "r1_market_gate", {"daily_residual_minimum_market_trend_z_20d": -8.0}),
        ("ablate_r1_stop_extension", "r1_stop", {"daily_residual_catastrophic_stop_residual_r": 4.0}),
        ("ablate_r1_partial_normalization_disable", "r1_partial_normalization", {"daily_residual_partial_normalization_fraction": 0.50}),
        ("ablate_r1_full_normalization_disable", "r1_full_normalization", {"daily_residual_full_normalization_fraction": 1.0}),
        ("ablate_r1_structural_failure_disable", "r1_structural_failure", {"daily_residual_structural_failure_extension_fraction": 0.50}),
        ("ablate_r1_partial_exit_disable", "r1_partial_exit", {"daily_residual_partial_exit_fraction": 0.50}),
        (
            "ablate_r1_management_bundle",
            "r1_management_bundle",
            {
                "daily_residual_catastrophic_stop_residual_r": 4.0,
                "daily_residual_partial_normalization_fraction": 0.50,
                "daily_residual_full_normalization_fraction": 1.0,
                "daily_residual_structural_failure_extension_fraction": 0.50,
                "daily_residual_partial_exit_fraction": 0.50,
            },
        ),
    ]
    rows.extend(
        _candidate(name, "atomic_ablation", current, patch, family=family)
        for name, family, patch in ablations
    )
    return rows


def _perturbation_candidates(current: StrategySettings) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, dict[str, Any]]] = []

    def sweep(family: str, field: str, values: Iterable[Any]) -> None:
        for value in values:
            if getattr(current, field) == value:
                continue
            label = str(value).replace("-", "m").replace(".", "p")
            specs.append((f"{family}_{label}", family, {field: value}))

    sweep("minimum_z", "daily_residual_minimum_z", (0.90, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50))
    sweep("score_floor", "daily_residual_minimum_score", (0.0, 10.0, 15.0, 20.0, 30.0, 35.0, 40.0, 45.0, 50.0))
    sweep("max_positions", "daily_residual_max_positions", (6, 8, 10, 14, 16))
    sweep("sector_cap", "daily_residual_max_positions_per_sector", (1, 3, 4))
    sweep("risk", "daily_residual_risk_fraction", (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040))
    sweep("notional", "daily_residual_maximum_notional_fraction", (0.08, 0.10, 0.15, 0.20))
    sweep("holding", "daily_residual_maximum_holding_sessions", (3, 5, 7, 8, 9))
    sweep("residual_stop", "daily_residual_catastrophic_stop_residual_r", (0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0))
    sweep("atr_stop", "daily_residual_catastrophic_stop_atr", (1.5, 2.0, 3.0, 3.5, 4.0))
    sweep("market_gate", "daily_residual_minimum_market_trend_z_20d", (-8.0, -2.0, -1.5, -0.75, -0.5, -0.25, 0.0))
    sweep("sector_gate", "daily_residual_minimum_sector_return_5d", (-0.10, -0.07, -0.05, -0.03, -0.01, 0.0))
    sweep("failed_continuation", "daily_residual_minimum_failed_continuation_r", (0.10, 0.20, 0.30, 0.50, 0.75, 1.0))
    sweep("formation", "daily_residual_formation_sessions", (3, 5))
    sweep("factor_model", "daily_residual_factor_model", ("market_only", "market_sector", "peer_demeaned"))

    component_sets = {
        "component_volume_only": ("volume_transition",),
        "component_price_only": ("price_rejection_recovery",),
        "component_failed_only": ("failed_continuation",),
        "component_extreme_only": ("residual_extremeness",),
        "component_fresh_only": ("shock_freshness",),
        "component_exhaust_only": ("volume_exhaustion_quality",),
        "component_regime_only": ("regime_execution_quality",),
        "component_volume_failed": ("volume_transition", "failed_continuation"),
        "component_volume_extreme": ("volume_transition", "residual_extremeness"),
        "component_volume_fresh": ("volume_transition", "shock_freshness"),
        "component_volume_exhaust": ("volume_transition", "volume_exhaustion_quality"),
        "component_volume_regime": ("volume_transition", "regime_execution_quality"),
        "component_price_failed": ("price_rejection_recovery", "failed_continuation"),
        "component_current_failed": ("volume_transition", "price_rejection_recovery", "failed_continuation"),
        "component_current_extreme": ("volume_transition", "price_rejection_recovery", "residual_extremeness"),
        "component_current_fresh": ("volume_transition", "price_rejection_recovery", "shock_freshness"),
        "component_current_exhaust": ("volume_transition", "price_rejection_recovery", "volume_exhaustion_quality"),
        "component_current_regime": ("volume_transition", "price_rejection_recovery", "regime_execution_quality"),
    }
    for name, components in component_sets.items():
        specs.append((name, "components", {"daily_residual_score_components": components}))
    ranking_sets = {
        "ranking_volume": ("volume_transition",),
        "ranking_price": ("price_rejection_recovery",),
        "ranking_failed": ("failed_continuation",),
        "ranking_extreme": ("residual_extremeness",),
    }
    for name, components in ranking_sets.items():
        specs.append((name, "ranking", {"daily_residual_ranking_score_components": components}))

    management = [
        ("normalization_partial50", "normalization", {"daily_residual_partial_normalization_fraction": 0.50, "daily_residual_partial_exit_fraction": 0.50}),
        ("normalization_partial75", "normalization", {"daily_residual_partial_normalization_fraction": 0.75, "daily_residual_partial_exit_fraction": 0.50}),
        ("normalization_full75", "normalization", {"daily_residual_full_normalization_fraction": 0.75}),
        ("normalization_full100", "normalization", {"daily_residual_full_normalization_fraction": 1.00}),
        ("normalization_full125", "normalization", {"daily_residual_full_normalization_fraction": 1.25}),
        ("structural_failure50", "structural_failure", {"daily_residual_structural_failure_extension_fraction": 0.50}),
        ("structural_failure75", "structural_failure", {"daily_residual_structural_failure_extension_fraction": 0.75}),
        ("profit_retention_075_035", "profit_retention", {"daily_residual_profit_retention_activation_fraction": 0.75, "daily_residual_profit_retention_giveback_fraction": 0.35}),
        ("profit_retention_100_035", "profit_retention", {"daily_residual_profit_retention_activation_fraction": 1.00, "daily_residual_profit_retention_giveback_fraction": 0.35}),
        ("profit_retention_125_050", "profit_retention", {"daily_residual_profit_retention_activation_fraction": 1.25, "daily_residual_profit_retention_giveback_fraction": 0.50}),
    ]
    specs.extend(management)
    capacity = [
        ("overflow1_score50_z1", "sector_overflow", {"daily_residual_sector_overflow_slots": 1}),
        ("overflow1_score60_z110", "sector_overflow", {"daily_residual_sector_overflow_slots": 1, "daily_residual_sector_overflow_minimum_score": 60.0, "daily_residual_sector_overflow_minimum_z": 1.10}),
        ("overflow1_half_risk", "sector_overflow", {"daily_residual_sector_overflow_slots": 1, "daily_residual_sector_overflow_risk_multiplier": 0.50}),
        ("replacement_sector_loss", "replacement", {"daily_residual_replacement_mode": "sector_stale", "daily_residual_replacement_loss_only": True}),
        ("replacement_diversifying_loss", "replacement", {"daily_residual_replacement_mode": "portfolio_diversifying", "daily_residual_replacement_loss_only": True}),
        ("replacement_combined_loss", "replacement", {"daily_residual_replacement_mode": "combined", "daily_residual_replacement_loss_only": True}),
    ]
    specs.extend(capacity)
    return [
        _candidate(name, "single_perturbation", current, patch, family=family)
        for name, family, patch in specs
    ]


def _settings_from_payload(payload: Mapping[str, Any]) -> StrategySettings:
    values = dict(payload)
    for name in (
        "daily_residual_score_components",
        "daily_residual_ranking_score_components",
    ):
        if name in values:
            values[name] = tuple(values[name])
    return StrategySettings(**values)


_REPLAY_WORKER_BUNDLE: DailyResidualReplayBundle | None = None


def _initialize_replay_worker(
    data_dir: str,
    factor_model: str,
    prepared_cache_dir: str,
) -> None:
    """Load exactly one factor bundle per process and retain its feature caches."""

    global _REPLAY_WORKER_BUNDLE
    panels = _load_full_panel(Path(data_dir))
    close, open_, high, low, volume, sectors, fingerprint, _rows = panels
    _REPLAY_WORKER_BUNDLE = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model=factor_model,
        source_fingerprint=fingerprint,
    )
    _REPLAY_WORKER_BUNDLE.prepared_selection_cache_dir = Path(
        prepared_cache_dir
    )
    del panels
    gc.collect()
    print(
        f"worker-ready pid={os.getpid()} factor={factor_model}",
        flush=True,
    )


def _evaluate_replay_worker(
    candidate: Mapping[str, Any],
    window: str,
    cost_bps: float,
    cache_path_text: str,
) -> dict[str, Any]:
    """Run one exact replay inside a persistent factor-sharded process."""

    bundle = _REPLAY_WORKER_BUNDLE
    if bundle is None:
        raise RuntimeError("replay worker was not initialized")
    cache_path = Path(cache_path_text)
    if cache_path.is_file():
        return {"cache_path": cache_path_text, "worker_pid": os.getpid()}
    settings = _settings_from_payload(candidate["settings"])
    if settings.daily_residual_factor_model != bundle.factor_model:
        raise ValueError("candidate was dispatched to the wrong factor shard")
    start, end = WINDOWS[window]
    prepared_before = len(bundle.prepared_selection_cache)
    snapshots_before = len(bundle.snapshot_cache)
    disk_hits_before = bundle.prepared_selection_disk_hits
    disk_misses_before = bundle.prepared_selection_disk_misses
    started = time.perf_counter()
    result = run_daily_residual_replay(
        bundle,
        settings,
        start=start,
        end=end,
        round_trip_cost_bps=cost_bps,
    )
    payload = {
        "name": candidate["name"],
        "settings_sha256": candidate["settings_sha256"],
        "window": window,
        "cost_bps": cost_bps,
        "metrics": result.metrics(),
        "period_metrics": {
            window: _period_metrics(result.trades, start=start, end=end),
            **(
                {
                    name: _period_metrics(result.trades, start=sub_start, end=sub_end)
                    for name, (sub_start, sub_end) in SUBWINDOWS.items()
                }
                if window == "oos"
                else {}
            ),
        },
        "trades": [_trade_payload(trade) for trade in result.trades],
        "decision_events": _jsonable(result.decision_events),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_hit": False,
        "acceleration": {
            "contract": ACCELERATION_CONTRACT,
            "worker_pid": os.getpid(),
            "factor_shard": bundle.factor_model,
            "prepared_feature_cache_entries_before": prepared_before,
            "prepared_feature_cache_entries_after": len(
                bundle.prepared_selection_cache
            ),
            "snapshot_cache_entries_before": snapshots_before,
            "snapshot_cache_entries_after": len(bundle.snapshot_cache),
            "prepared_disk_hits": (
                bundle.prepared_selection_disk_hits - disk_hits_before
            ),
            "prepared_disk_misses": (
                bundle.prepared_selection_disk_misses - disk_misses_before
            ),
        },
    }
    _write_json(cache_path, payload)
    metrics = payload["period_metrics"][window]
    print(
        f"worker-done pid={os.getpid()} {window} {candidate['name']}: "
        f"T={metrics['trades']} R={metrics['total_r']:.2f} "
        f"elapsed={payload['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return {
        "cache_path": cache_path_text,
        "worker_pid": os.getpid(),
        "elapsed_seconds": payload["elapsed_seconds"],
    }


class Experiment:
    def __init__(
        self,
        *,
        output: Path,
        data_dir: Path,
        workers: int,
        candidates: list[dict[str, Any]],
        prepared_cache_dir: Path | None = None,
    ) -> None:
        self.output = output
        self.data_dir = data_dir
        self.workers = workers
        self.candidates = {row["name"]: row for row in candidates}
        self.prepared_cache_dir = (
            prepared_cache_dir
            if prepared_cache_dir is not None
            else self.output / "prepared_feature_cache"
        )
        factor_counts = Counter(
            str(row["settings"]["daily_residual_factor_model"])
            for row in candidates
        )
        self.primary_factor = factor_counts.most_common(1)[0][0]
        self._primary_executor: ProcessPoolExecutor | None = None
        self._spawn_context = multiprocessing.get_context("spawn")

    def load_data(self) -> dict[str, Any]:
        receipt_path = self.output / "data_consumption_receipt.json"
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            if receipt.get("data_end") == OOS_END.isoformat():
                return {**receipt, "receipt_cache_hit": True}
        started = time.perf_counter()
        panels = _load_full_panel(self.data_dir)
        close, _open, _high, _low, _volume, sectors, fingerprint, rows = panels
        receipt = {
            "contract": "explicitly_consumed_round2_oos_extension_v1",
            "selection_end": IS_END.isoformat(),
            "post_selection_start": OOS_START.isoformat(),
            "sealed_extension_start": OOS_START.isoformat(),
            "data_end": OOS_END.isoformat(),
            "latest_generic_oos_start": LATEST_OOS_START.isoformat(),
            "sealed_extension_consumed_by_this_research": True,
            "promotion_eligible_without_new_untouched_data": False,
            "data_fingerprint": fingerprint,
            "rows": rows,
            "panel_sessions": len(close),
            "stock_symbols": len(sectors),
            "load_seconds": time.perf_counter() - started,
        }
        _write_json(receipt_path, receipt)
        del panels, close, _open, _high, _low, _volume
        gc.collect()
        return receipt

    def _new_executor(
        self,
        factor_model: str,
        task_count: int,
    ) -> ProcessPoolExecutor:
        worker_count = (
            self.workers
            if factor_model == self.primary_factor
            else max(1, min(self.workers, task_count))
        )
        print(
            f"starting factor shard: {factor_model} workers={worker_count}",
            flush=True,
        )
        return ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=self._spawn_context,
            initializer=_initialize_replay_worker,
            initargs=(
                str(self.data_dir),
                factor_model,
                str(self.prepared_cache_dir),
            ),
        )

    def _shutdown_primary(self) -> None:
        if self._primary_executor is None:
            return
        self._primary_executor.shutdown(wait=True, cancel_futures=False)
        self._primary_executor = None

    def close(self) -> None:
        self._shutdown_primary()

    def _run_factor_group(
        self,
        factor_model: str,
        candidates: list[Mapping[str, Any]],
        *,
        window: str,
        cost_bps: float,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        persistent = factor_model == self.primary_factor
        if persistent:
            if self._primary_executor is None:
                self._primary_executor = self._new_executor(
                    factor_model,
                    len(candidates),
                )
            executor = self._primary_executor
        else:
            # Never retain multiple large factor bundles.  Releasing the idle
            # primary shard makes alternate-factor ablations RAM-safe.
            self._shutdown_primary()
            executor = self._new_executor(factor_model, len(candidates))
        try:
            futures = {
                executor.submit(
                    _evaluate_replay_worker,
                    candidate,
                    window,
                    cost_bps,
                    str(self._cache_path(candidate, window, cost_bps)),
                ): candidate
                for candidate in candidates
            }
            rows = []
            for future in as_completed(futures):
                marker = future.result()
                row = _read_json(Path(marker["cache_path"]))
                rows.append(row)
            return rows
        finally:
            if not persistent:
                executor.shutdown(wait=True, cancel_futures=False)

    def _cache_path(self, candidate: Mapping[str, Any], window: str, cost_bps: float) -> Path:
        signature = str(candidate["settings_sha256"])[:16]
        return self.output / "cache" / f"{signature}__{window}__{cost_bps:.0f}bps.json"

    def evaluate_one(
        self,
        candidate: Mapping[str, Any],
        *,
        window: str,
        cost_bps: float = 20.0,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(candidate, window, cost_bps)
        if cache_path.is_file():
            cached = _read_json(cache_path)
            return {**cached, "name": candidate["name"], "cache_hit": True}
        factor_model = str(
            candidate["settings"]["daily_residual_factor_model"]
        )
        rows = self._run_factor_group(
            factor_model,
            [candidate],
            window=window,
            cost_bps=cost_bps,
        )
        return {**rows[0], "name": candidate["name"], "cache_hit": False}

    def evaluate_many(
        self,
        names: Iterable[str],
        *,
        window: str,
        cost_bps: float = 20.0,
    ) -> list[dict[str, Any]]:
        names = list(dict.fromkeys(names))
        results: list[dict[str, Any]] = []
        aliases_by_signature: dict[str, list[str]] = defaultdict(list)
        representative_by_signature: dict[str, str] = {}
        for name in names:
            signature = str(self.candidates[name]["settings_sha256"])
            aliases_by_signature[signature].append(name)
            representative_by_signature.setdefault(signature, name)
        rows_by_representative: dict[str, dict[str, Any]] = {}
        missing_by_factor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for name in representative_by_signature.values():
            candidate = self.candidates[name]
            cache_path = self._cache_path(candidate, window, cost_bps)
            if cache_path.is_file():
                rows_by_representative[name] = {
                    **_read_json(cache_path),
                    "name": name,
                    "cache_hit": True,
                }
                continue
            factor = str(candidate["settings"]["daily_residual_factor_model"])
            missing_by_factor[factor].append(candidate)
        factor_order = sorted(
            missing_by_factor,
            key=lambda factor: (factor == self.primary_factor, factor),
        )
        for factor in factor_order:
            rows = self._run_factor_group(
                factor,
                missing_by_factor[factor],
                window=window,
                cost_bps=cost_bps,
            )
            rows_by_representative.update(
                (str(row["name"]), row) for row in rows
            )
        completed = 0
        for signature, name in representative_by_signature.items():
            row = rows_by_representative[name]
            aliases = aliases_by_signature[signature]
            for alias in aliases:
                results.append(
                    {**row, "name": alias, "shared_replay_alias_of": name}
                )
                completed += 1
                metrics = row["period_metrics"][window]
                print(
                    f"[{completed}/{len(names)}] {window} {alias}: "
                    f"T={metrics['trades']} R={metrics['total_r']:.2f} "
                    f"avg={metrics['average_r']:.3f}",
                    flush=True,
                )
        return sorted(results, key=lambda row: row["name"])


def _flatten_results(
    candidates: Mapping[str, Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        candidate = candidates[row["name"]]
        base = {
            "name": row["name"],
            "group": candidate["group"],
            "family": candidate["family"],
            "patch": candidate["patch"],
            "window": row["window"],
            "cost_bps": row["cost_bps"],
        }
        for window, metrics in row["period_metrics"].items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    base[f"{window}_{key}"] = value
        exact = row["metrics"]
        for key in ("return_pct", "max_drawdown_pct", "final_equity"):
            base[f"exact_{key}"] = exact.get(key)
        output.append(base)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _promising_oos_names(
    rows: list[dict[str, Any]],
    *,
    baseline: Mapping[str, Any],
    limit: int = 36,
) -> list[str]:
    base = baseline["period_metrics"]
    scored = []
    for row in rows:
        whole = row["period_metrics"]["oos"]
        latest = row["period_metrics"]["latest_oos"]
        early = row["period_metrics"]["early_oos"]
        if whole["trades"] < 0.75 * base["oos"]["trades"]:
            continue
        score = (
            whole["r_per_month"]
            + 0.20 * early["r_per_month"]
            + 0.20 * latest["r_per_month"]
            + 0.02 * whole["trades_per_month"]
        )
        scored.append((score, row["name"]))
    top = [name for _score, name in sorted(scored, reverse=True)[:limit]]
    # Preserve the best member of every family so low-value accepted mutations
    # and trade-frequency frontiers cannot disappear behind aggregate ranking.
    by_family: dict[str, tuple[float, str]] = {}
    for score, name in scored:
        family = str(baseline["candidate_map"][name]["family"])
        if family not in by_family or score > by_family[family][0]:
            by_family[family] = (score, name)
    return list(dict.fromkeys([*top, *(name for _score, name in by_family.values())]))


def _eligibility(
    oos: Mapping[str, Any],
    is_row: Mapping[str, Any],
    baseline_oos: Mapping[str, Any],
    baseline_is: Mapping[str, Any],
) -> dict[str, Any]:
    om = oos["period_metrics"]["oos"]
    im = is_row["period_metrics"]["is"]
    bo = baseline_oos["period_metrics"]["oos"]
    bi = baseline_is["period_metrics"]["is"]
    gates = {
        "oos_total_r_improves_5pct": om["total_r"] >= bo["total_r"] * 1.05,
        "oos_average_r_improves": om["average_r"] > bo["average_r"],
        "oos_frequency_retains_90pct": om["trades_per_month"] >= bo["trades_per_month"] * 0.90,
        "early_oos_not_worse_by_1r": (
            oos["period_metrics"]["early_oos"]["total_r"]
            >= baseline_oos["period_metrics"]["early_oos"]["total_r"] - 1.0
        ),
        "latest_oos_positive": oos["period_metrics"]["latest_oos"]["total_r"] > 0.0,
        "is_total_r_retains_90pct": im["total_r"] >= bi["total_r"] * 0.90,
        "is_average_r_retains_90pct": im["average_r"] >= bi["average_r"] * 0.90,
        "is_frequency_retains_90pct": im["trades_per_month"] >= bi["trades_per_month"] * 0.90,
        "is_drawdown_not_materially_worse": float(is_row["metrics"]["max_drawdown_pct"]) <= float(baseline_is["metrics"]["max_drawdown_pct"]) * 1.10 + 0.005,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _run_lineage(exp: Experiment, names: list[str]) -> dict[str, Any]:
    oos = exp.evaluate_many(names, window="oos")
    is_rows = exp.evaluate_many(names, window="is")
    by_oos = {row["name"]: row for row in oos}
    by_is = {row["name"]: row for row in is_rows}
    current_oos = by_oos["current"]
    current_is = by_is["current"]
    comparison = []
    for name in names:
        eligibility = _eligibility(by_oos[name], by_is[name], current_oos, current_is)
        comparison.append(
            {
                "name": name,
                "group": exp.candidates[name]["group"],
                "family": exp.candidates[name]["family"],
                "patch": exp.candidates[name]["patch"],
                "oos": by_oos[name]["period_metrics"],
                "is": by_is[name]["period_metrics"]["is"],
                "exact_oos": by_oos[name]["metrics"],
                "exact_is": by_is[name]["metrics"],
                "eligibility": eligibility,
            }
        )
    result = {
        "contract": "all_cumulative_prefixes_and_atomic_lineage_ablation_v1",
        "candidate_count": len(names),
        "comparison": comparison,
        "literal_prefix_names": [name for name in names if name.startswith("prefix_") or name == "current"],
        "atomic_lineage_families": sorted({exp.candidates[name]["family"] for name in names}),
    }
    _write_json(exp.output / "phase_1_lineage_ablation.json", result)
    flat = _flatten_results(exp.candidates, [*oos, *is_rows])
    _write_csv(exp.output / "phase_1_lineage_ablation.csv", flat)
    baseline_oos_cache = _read_json(exp._cache_path(exp.candidates["current"], "oos", 20.0))
    baseline_is_cache = _read_json(exp._cache_path(exp.candidates["current"], "is", 20.0))
    baseline_oos_result = _result_from_cache(baseline_oos_cache)
    baseline_is_result = _result_from_cache(baseline_is_cache)
    _write_json(
        exp.output / "baseline_diagnostics.json",
        _baseline_diagnostics(baseline_oos_result, baseline_is_result),
    )
    return {"oos": oos, "is": is_rows, "comparison": comparison}


def _result_from_cache(payload: Mapping[str, Any]) -> DailyResidualReplayResult:
    """Rehydrate only the fields required by diagnostics."""
    from backtests.stock.engine.iaric_daily_residual_replay import DailyResidualReplayTrade

    trades = []
    for row in payload["trades"]:
        values = dict(row)
        for field in ("entry_date", "exit_date"):
            if values.get(field):
                values[field] = date.fromisoformat(values[field])
        for field in ("entry_time", "exit_time"):
            if values.get(field):
                values[field] = datetime.fromisoformat(values[field])
        trades.append(DailyResidualReplayTrade(**values))
    return DailyResidualReplayResult(
        initial_equity=100_000.0,
        final_equity=float(payload["metrics"]["final_equity"]),
        trades=trades,
        equity_curve=[],
        decision_events=list(payload.get("decision_events", [])),
        source_fingerprint="cached",
        factor_model="cached",
    )


def _run_perturbations(exp: Experiment, names: list[str]) -> dict[str, Any]:
    oos = exp.evaluate_many(names, window="oos")
    current_oos = exp.evaluate_one(exp.candidates["current"], window="oos")
    selector_context = {
        "period_metrics": current_oos["period_metrics"],
        "candidate_map": exp.candidates,
    }
    promising = _promising_oos_names(oos, baseline=selector_context)
    is_rows = exp.evaluate_many(promising, window="is")
    by_is = {row["name"]: row for row in is_rows}
    current_is = exp.evaluate_one(exp.candidates["current"], window="is")
    comparison = []
    for row in oos:
        item = {
            "name": row["name"],
            "family": exp.candidates[row["name"]]["family"],
            "patch": exp.candidates[row["name"]]["patch"],
            "oos": row["period_metrics"],
            "is_evaluated": row["name"] in by_is,
        }
        if row["name"] in by_is:
            item["is"] = by_is[row["name"]]["period_metrics"]["is"]
            item["eligibility"] = _eligibility(row, by_is[row["name"]], current_oos, current_is)
        comparison.append(item)
    payload = {
        "contract": "broad_one_lever_oos_screen_then_is_retention_v1",
        "oos_candidate_count": len(oos),
        "is_candidate_count": len(is_rows),
        "promising_names": promising,
        "comparison": comparison,
    }
    _write_json(exp.output / "phase_2_perturbation.json", payload)
    _write_csv(
        exp.output / "phase_2_perturbation.csv",
        _flatten_results(exp.candidates, [*oos, *is_rows]),
    )
    return {"oos": oos, "is": is_rows, "comparison": comparison}


def _distinct_best_patches(
    exp: Experiment,
    phase2: Mapping[str, Any],
    *,
    count: int = 7,
) -> list[tuple[str, dict[str, Any]]]:
    eligible = [
        row
        for row in phase2["comparison"]
        if row.get("eligibility", {}).get("passed")
    ]
    if not eligible:
        eligible = [row for row in phase2["comparison"] if row.get("is_evaluated")]
    by_family: dict[str, tuple[float, str, dict[str, Any]]] = {}
    for row in eligible:
        oos = row["oos"]["oos"]
        latest = row["oos"]["latest_oos"]
        is_metrics = row.get("is", {})
        score = (
            float(oos["r_per_month"])
            + 0.25 * float(latest["r_per_month"])
            + 0.10 * float(is_metrics.get("r_per_month", 0.0))
            + 0.01 * float(oos["trades_per_month"])
        )
        family = str(row["family"])
        candidate = (score, row["name"], dict(row["patch"]))
        if family not in by_family or score > by_family[family][0]:
            by_family[family] = candidate
    return [
        (name, patch)
        for _score, name, patch in sorted(by_family.values(), reverse=True)[:count]
    ]


def _targeted_candidates(
    current: StrategySettings,
    exp: Experiment,
    phase2: Mapping[str, Any],
) -> list[dict[str, Any]]:
    levers = _distinct_best_patches(exp, phase2)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for left_index, (left_name, left_patch) in enumerate(levers):
        for right_name, right_patch in levers[left_index + 1 :]:
            if set(left_patch) & set(right_patch):
                continue
            patch = {**left_patch, **right_patch}
            signature = _sha(patch)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(
                _candidate(
                    f"target_{left_name}__{right_name}",
                    "targeted_pair",
                    current,
                    patch,
                    family="targeted_pair",
                    note="Pair of distinct OOS-positive/IS-retentive one-lever families.",
                )
            )
    prescribed = {
        "target_z110_volume_only": {
            "daily_residual_minimum_z": 1.10,
            "daily_residual_score_components": ("volume_transition",),
        },
        "target_z110_score20": {
            "daily_residual_minimum_z": 1.10,
            "daily_residual_minimum_score": 20.0,
        },
        "target_z110_cap10": {
            "daily_residual_minimum_z": 1.10,
            "daily_residual_max_positions": 10,
        },
        "target_volume_only_score20": {
            "daily_residual_score_components": ("volume_transition",),
            "daily_residual_minimum_score": 20.0,
        },
        "target_volume_only_hold7": {
            "daily_residual_score_components": ("volume_transition",),
            "daily_residual_maximum_holding_sessions": 7,
        },
        "target_z110_market_m075": {
            "daily_residual_minimum_z": 1.10,
            "daily_residual_minimum_market_trend_z_20d": -0.75,
        },
        "target_z110_stop5": {
            "daily_residual_minimum_z": 1.10,
            "daily_residual_catastrophic_stop_residual_r": 5.0,
        },
        "target_score20_cap14": {
            "daily_residual_minimum_score": 20.0,
            "daily_residual_max_positions": 14,
        },
    }
    for name, patch in prescribed.items():
        signature = _sha(patch)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(
            _candidate(
                name,
                "targeted_hypothesis",
                current,
                patch,
                family="targeted_hypothesis",
                note="Mechanism-targeted response to score, tail, capacity, or regime diagnostics.",
            )
        )
    return rows[:30]


def _bootstrap_daily_delta(
    baseline: list[Mapping[str, Any]],
    candidate: list[Mapping[str, Any]],
    *,
    seed: int = 20260823,
    samples: int = 4000,
    block: int = 5,
) -> dict[str, Any]:
    def daily(rows):
        result: dict[date, float] = defaultdict(float)
        for row in rows:
            entry = date.fromisoformat(str(row["entry_date"]))
            result[entry] += float(row["r_multiple"])
        return result

    left, right = daily(baseline), daily(candidate)
    dates = sorted(set(left) | set(right))
    delta = np.asarray([right.get(day, 0.0) - left.get(day, 0.0) for day in dates])
    if not len(delta):
        return {"observations": 0, "mean_delta_r": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    block_count = math.ceil(len(delta) / block)
    starts = np.arange(len(delta))
    for index in range(samples):
        sample = []
        for start in rng.choice(starts, size=block_count, replace=True):
            sample.extend(delta[(start + offset) % len(delta)] for offset in range(block))
        means[index] = float(np.mean(sample[: len(delta)]))
    return {
        "contract": "paired_entry_date_circular_block_bootstrap_v1",
        "observations": len(delta),
        "block_sessions": block,
        "samples": samples,
        "mean_delta_r_per_session": float(delta.mean()),
        "ci_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "probability_positive": float(np.mean(means > 0.0)),
    }


def _verify_acceleration(
    reference_exp: Experiment,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove old-cache parity and measure prepared-feature reuse in one worker."""

    by_name = {row["name"]: row for row in candidates}
    current = by_name["current"]
    reuse_probe = by_name["ablate_r2_risk_to_phase10"]
    restart_probe = by_name["ablate_r2_notional12"]
    reference = reference_exp.evaluate_one(current, window="oos")
    validation_output = reference_exp.output / "acceleration_validation_v2"
    validation_output.mkdir(parents=True, exist_ok=True)
    receipt = _read_json(reference_exp.output / "data_consumption_receipt.json")
    _write_json(validation_output / "data_consumption_receipt.json", receipt)
    accelerated = Experiment(
        output=validation_output,
        data_dir=reference_exp.data_dir,
        workers=1,
        candidates=[current, reuse_probe, restart_probe],
    )
    try:
        cold = accelerated.evaluate_one(current, window="oos")
        warm = accelerated.evaluate_one(reuse_probe, window="oos")
    finally:
        accelerated.close()
    restarted = Experiment(
        output=validation_output,
        data_dir=reference_exp.data_dir,
        workers=1,
        candidates=[restart_probe],
    )
    try:
        disk_warm = restarted.evaluate_one(restart_probe, window="oos")
    finally:
        restarted.close()
    parity_fields = ("metrics", "period_metrics", "trades", "decision_events")
    differences = [
        field for field in parity_fields if reference[field] != cold[field]
    ]
    payload = {
        "contract": "exact_trade_and_event_acceleration_parity_v1",
        "passed": not differences,
        "compared_fields": list(parity_fields),
        "differences": differences,
        "reference_cache": str(
            reference_exp._cache_path(current, "oos", 20.0).resolve()
        ),
        "cold_preparation_seconds": cold["elapsed_seconds"],
        "warm_reuse_probe_seconds": warm["elapsed_seconds"],
        "restarted_disk_reuse_probe_seconds": disk_warm["elapsed_seconds"],
        "observed_warm_speedup": (
            float(cold["elapsed_seconds"]) / max(float(warm["elapsed_seconds"]), 1e-9)
        ),
        "cold_acceleration": cold.get("acceleration", {}),
        "warm_acceleration": warm.get("acceleration", {}),
        "restarted_disk_acceleration": disk_warm.get("acceleration", {}),
    }
    _write_json(reference_exp.output / "acceleration_verification.json", payload)
    if differences:
        raise RuntimeError(
            "accelerated replay failed exact parity: " + ", ".join(differences)
        )
    return payload


def _run_targeted(
    exp: Experiment,
    current: StrategySettings,
    phase2: Mapping[str, Any],
) -> dict[str, Any]:
    targeted = _targeted_candidates(current, exp, phase2)
    for row in targeted:
        exp.candidates[row["name"]] = row
    _write_json(exp.output / "phase_3_targeted_candidate_catalog.json", targeted)
    names = [row["name"] for row in targeted]
    oos = exp.evaluate_many(names, window="oos")
    current_oos = exp.evaluate_one(exp.candidates["current"], window="oos")
    selector_context = {
        "period_metrics": current_oos["period_metrics"],
        "candidate_map": exp.candidates,
    }
    promising = _promising_oos_names(oos, baseline=selector_context, limit=16)
    is_rows = exp.evaluate_many(promising, window="is")
    current_is = exp.evaluate_one(exp.candidates["current"], window="is")
    by_is = {row["name"]: row for row in is_rows}
    comparison = []
    for row in oos:
        item = {
            "name": row["name"],
            "patch": exp.candidates[row["name"]]["patch"],
            "oos": row["period_metrics"],
            "is_evaluated": row["name"] in by_is,
        }
        if row["name"] in by_is:
            item["is"] = by_is[row["name"]]["period_metrics"]["is"]
            item["eligibility"] = _eligibility(row, by_is[row["name"]], current_oos, current_is)
        comparison.append(item)
    eligible = [row for row in comparison if row.get("eligibility", {}).get("passed")]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["oos"]["oos"]["r_per_month"],
            row["oos"]["oos"]["trades_per_month"],
            row["is"]["r_per_month"],
        ),
        reverse=True,
    )
    finalists = ranked[:5]
    robustness = []
    baseline_oos_trades = current_oos["trades"]
    for index, row in enumerate(finalists):
        name = row["name"]
        cost30 = exp.evaluate_one(exp.candidates[name], window="oos", cost_bps=30.0)
        cost40 = exp.evaluate_one(exp.candidates[name], window="oos", cost_bps=40.0)
        candidate_oos = next(item for item in oos if item["name"] == name)
        robustness.append(
            {
                "name": name,
                "patch": exp.candidates[name]["patch"],
                "base": row,
                "oos_cost30": cost30["period_metrics"],
                "oos_cost40": cost40["period_metrics"],
                "paired_oos_bootstrap": _bootstrap_daily_delta(
                    baseline_oos_trades,
                    candidate_oos["trades"],
                    seed=20260823 + index,
                ),
            }
        )
    selected = None
    if robustness:
        qualified = [
            row
            for row in robustness
            if row["oos_cost30"]["oos"]["total_r"] > 0.0
            and row["oos_cost40"]["oos"]["total_r"] > 0.0
            and row["base"]["is"]["total_r"] >= current_is["period_metrics"]["is"]["total_r"] * 0.90
        ]
        if qualified:
            selected = max(
                qualified,
                key=lambda row: (
                    row["base"]["oos"]["oos"]["r_per_month"],
                    row["base"]["oos"]["oos"]["trades_per_month"],
                ),
            )
    payload = {
        "contract": "diagnosis_targeted_combinations_with_is_and_cost_confirmation_v1",
        "candidate_count": len(targeted),
        "is_evaluated_count": len(is_rows),
        "comparison": comparison,
        "finalist_robustness": robustness,
        "selected_research_candidate": selected,
        "promotion_eligible": False,
        "promotion_blocker": "2026-03-02..2026-05-01 outcomes were used for diagnosis and ranking",
    }
    _write_json(exp.output / "phase_3_targeted.json", payload)
    _write_csv(
        exp.output / "phase_3_targeted.csv",
        _flatten_results(exp.candidates, [*oos, *is_rows]),
    )
    if selected is not None:
        name = selected["name"]
        _write_json(
            exp.output / "recommended_research_config.json",
            {
                "configuration_role": "post_oos_diagnostic_research_candidate_not_promotion_eligible",
                "candidate_name": name,
                "settings": exp.candidates[name]["settings"],
                "settings_sha256": exp.candidates[name]["settings_sha256"],
                "patch_vs_round2": exp.candidates[name]["patch"],
                "validation_required": "new untouched chronological data after 2026-05-01",
            },
        )
    return payload


def _format_report(
    output: Path,
    lineage: Mapping[str, Any],
    phase2: Mapping[str, Any] | None,
    phase3: Mapping[str, Any] | None,
) -> str:
    baseline = _read_json(output / "baseline_diagnostics.json")
    windows = baseline["window_metrics"]
    concentration = baseline["loss_concentration"]
    lines = [
        "# IARIC Round 2 residual OOS ablation and perturbation",
        "",
        "## Evidence contract",
        "",
        "In-sample evaluation ends 2026-03-01. This diagnostic explicitly consumes the "
        "2026-03-02 through 2026-05-01 OOS period. Any selected change is research-only until "
        "validated on new untouched chronological data after 2026-05-01.",
        "",
        "## Baseline discrepancy",
        "",
        "| Window | Trades | Trades/month | Total R | Average R | Win rate | Profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("is", "early_oos", "latest_oos", "oos"):
        row = windows[name]
        lines.append(
            f"| {name} | {row['trades']} | {row['trades_per_month']:.1f} | "
            f"{row['total_r']:.2f} | {row['average_r']:.3f} | "
            f"{row['win_rate']:.1%} | {row['profit_factor']:.2f} |"
        )
    worst = concentration["worst_trade_removal_counterfactual"]
    lines.extend(
        [
            "",
            "## Edge-case concentration",
            "",
            f"The worst trade represents {worst['1']['loss_share']:.1%} of gross losing R; "
            f"the worst five represent {worst['5']['loss_share']:.1%}. Removing the worst "
            f"five mechanically changes OOS total R to {worst['5']['counterfactual_total_r']:.2f}R. "
            "This counterfactual is attribution only, not an implementable rule.",
            "",
            "## Lineage ablation",
            "",
        ]
    )
    eligible_lineage = [row for row in lineage["comparison"] if row["eligibility"]["passed"]]
    if eligible_lineage:
        best = max(eligible_lineage, key=lambda row: row["oos"]["oos"]["r_per_month"])
        lines.append(
            f"{len(eligible_lineage)} lineage candidates cleared the joint OOS uplift/IS-retention "
            f"screen. Best: `{best['name']}` ({best['oos']['oos']['total_r']:.2f} OOS R, "
            f"{best['oos']['oos']['trades']} trades)."
        )
    else:
        lines.append("No lineage removal cleared every joint OOS uplift and IS-retention gate.")
    if phase2 is not None:
        eligible = [row for row in phase2["comparison"] if row.get("eligibility", {}).get("passed")]
        lines.extend(
            [
                "",
                "## Broad perturbation",
                "",
                f"Evaluated {phase2['oos_candidate_count']} one-lever variants OOS and "
                f"{phase2['is_candidate_count']} promising/Pareto variants in sample. "
                f"{len(eligible)} cleared every joint screen.",
            ]
        )
    if phase3 is not None:
        selected = phase3.get("selected_research_candidate")
        lines.extend(["", "## Targeted follow-up", ""])
        if selected:
            base = selected["base"]
            lines.append(
                f"Research candidate `{selected['name']}` is the strongest cost-positive joint "
                f"candidate: OOS {base['oos']['oos']['total_r']:.2f}R across "
                f"{base['oos']['oos']['trades']} trades; IS {base['is']['total_r']:.2f}R across "
                f"{base['is']['trades']} trades. It is not promotion-eligible because the OOS "
                "extension informed selection."
            )
        else:
            lines.append(
                "No targeted combination survived OOS uplift, IS retention, and 30/40 bps cost "
                "confirmation. The current configuration remains the honest research control."
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "Machine-readable candidate-level results, trades, lineage coverage, cost stresses, "
            "and the explicit holdout-consumption receipt are stored beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Requested process workers (1-4); RAM guard may reduce this to 1-3.",
    )
    parser.add_argument(
        "--phase",
        choices=("lineage", "perturb", "targeted", "all"),
        default="all",
    )
    parser.add_argument(
        "--acknowledge-consumed-holdout",
        action="store_true",
        help="Required: confirms this diagnostic consumes data through 2026-05-01.",
    )
    parser.add_argument(
        "--verify-acceleration-only",
        action="store_true",
        help="Run exact old-cache parity plus cold/warm acceleration benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.acknowledge_consumed_holdout:
        raise SystemExit("--acknowledge-consumed-holdout is required")
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    effective_workers, worker_policy = _effective_worker_count(args.workers)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _candidate_row, baseline_lineage = _load_round2_baseline(CURRENT_CANDIDATE)
    current = StrategySettings(**baseline_lineage["settings"])
    lineage_candidates = _lineage_candidates(current)
    perturbation_candidates = _perturbation_candidates(current)
    all_candidates = [*lineage_candidates, *perturbation_candidates]
    # Exact duplicate settings are retained in the catalog to prove metadata/no-op
    # mutations, while their deterministic replay cache is shared by signature.
    _write_json(output / "candidate_catalog.json", all_candidates)
    _write_json(
        output / "run_spec.json",
        {
            "contract": "iaric_round2_residual_corrected_is_oos_split_v2",
            "phase": args.phase,
            "backend": "factor_sharded_process_pool",
            "acceleration_contract": ACCELERATION_CONTRACT,
            "workers": worker_policy,
            "current_candidate": str(CURRENT_CANDIDATE.resolve()),
            "round1_config": str(ROUND1_CONFIG.resolve()),
            "windows": {
                **{name: [start, end] for name, (start, end) in WINDOWS.items()},
                **{name: [start, end] for name, (start, end) in SUBWINDOWS.items()},
            },
            "round_trip_cost_bps": 20.0,
            "sealed_extension_consumed": True,
            "promotion_eligible": False,
            "started_at_utc": datetime.now(timezone.utc),
        },
    )
    exp = Experiment(
        output=output,
        data_dir=args.data_dir.resolve(),
        workers=effective_workers,
        candidates=all_candidates,
    )
    exp.load_data()
    if args.verify_acceleration_only:
        verification = _verify_acceleration(exp, all_candidates)
        exp.close()
        print(
            "acceleration verified: "
            f"{verification['observed_warm_speedup']:.2f}x warm speedup",
            flush=True,
        )
        return 0
    lineage_names = [row["name"] for row in lineage_candidates]
    lineage_path = output / "phase_1_lineage_ablation.json"
    if args.phase in {"lineage", "all"} or not lineage_path.is_file():
        lineage = _run_lineage(exp, lineage_names)
        lineage_payload = _read_json(lineage_path)
    else:
        lineage_payload = _read_json(lineage_path)
        lineage = None
    phase2_payload = None
    phase2_path = output / "phase_2_perturbation.json"
    if args.phase in {"perturb", "all", "targeted"}:
        perturb_names = [row["name"] for row in perturbation_candidates]
        phase2 = _run_perturbations(exp, perturb_names)
        phase2_payload = _read_json(phase2_path)
    elif phase2_path.is_file():
        phase2_payload = _read_json(phase2_path)
    phase3_payload = None
    if args.phase in {"targeted", "all"}:
        if phase2_payload is None:
            raise RuntimeError("targeted phase requires completed perturbation results")
        phase3_payload = _run_targeted(exp, current, phase2_payload)
    elif (output / "phase_3_targeted.json").is_file():
        phase3_payload = _read_json(output / "phase_3_targeted.json")
    exp.close()
    report = _format_report(output, lineage_payload, phase2_payload, phase3_payload)
    (output / "report.md").write_text(report, encoding="utf-8")
    _write_json(
        output / "completion.json",
        {
            "status": "complete" if args.phase == "all" else f"complete_{args.phase}",
            "report": str((output / "report.md").resolve()),
            "sealed_extension_consumed": True,
            "promotion_eligible": False,
            "completed_at_utc": datetime.now(timezone.utc),
        },
    )
    print(f"completed: {output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
