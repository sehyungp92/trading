from __future__ import annotations

import io
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_key: tuple[str, tuple[str, ...]] | None = None


def init_worker(data_dir_str: str, equity: float, symbols_csv: str, fixed_qty: int | None = None) -> None:
    global _worker_data, _worker_config, _worker_equity, _worker_key

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing.config_breakout import BreakoutBacktestConfig
    from backtests.swing.engine.breakout_portfolio_engine import load_breakout_data

    data_dir = Path(data_dir_str)
    symbols = tuple(symbol.strip().upper() for symbol in symbols_csv.split(",") if symbol.strip())
    worker_key = (str(data_dir.resolve()), symbols)

    if _worker_data is None or _worker_key != worker_key:
        _worker_data = load_breakout_data(list(symbols), data_dir)
        _worker_key = worker_key
    _worker_equity = equity
    _worker_config = BreakoutBacktestConfig(
        symbols=list(symbols),
        initial_equity=equity,
        data_dir=data_dir,
        fixed_qty=fixed_qty,
        track_signals=False,
        track_shadows=False,
    )


def score_candidate(args: tuple[str, dict, dict, int, dict | None, dict | None]) -> ScoredCandidate:
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from backtests.swing.auto.config_mutator import mutate_breakout_config
        from backtests.swing.auto.scoring import extract_metrics
        from backtests.swing.engine.breakout_portfolio_engine import run_breakout_synchronized

        from .scoring import score_phase_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_breakout_config(_worker_config, all_muts)
        result = run_breakout_synchronized(_worker_data, config)
        all_trades = _collect_trades(result)
        metrics = extract_metrics(
            all_trades,
            result.combined_equity,
            _timestamps_to_numeric(result.combined_timestamps),
            _worker_equity,
        )
        metrics.avg_mfe_r = _avg_mfe_r(all_trades)
        metrics.winner_capture_ratio = _winner_capture_ratio(all_trades)
        score = score_phase_metrics(
            phase,
            metrics,
            _worker_equity,
            equity_curve=result.combined_equity,
            weight_overrides=scoring_weights,
            hard_rejects=hard_rejects,
        )
        metrics_dict = asdict(metrics)
        metrics_dict["edge_velocity"] = float(metrics.expectancy_dollar * metrics.trades_per_month)
        return ScoredCandidate(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason if score.rejected else "",
            metrics=metrics_dict,
        )
    except Exception as exc:
        logger.error("Breakout worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")


def _collect_trades(result) -> list:
    trades: list = []
    for symbol, symbol_result in result.symbol_results.items():
        for trade in symbol_result.trades:
            if not getattr(trade, "symbol", ""):
                trade.symbol = symbol
            trades.append(trade)
    trades.sort(key=_trade_sort_key)
    return trades


def _timestamps_to_numeric(timestamps) -> np.ndarray:
    if timestamps is None or len(timestamps) == 0:
        return np.array([])
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        return np.array([dt.timestamp() for dt in timestamps], dtype=float)
    if hasattr(first, "item"):
        first = first.item()
        if hasattr(first, "timestamp"):
            return np.array([ts.item().timestamp() for ts in timestamps], dtype=float)
    return np.asarray(timestamps)


def _trade_sort_key(trade) -> str:
    dt = getattr(trade, "entry_time", None) or getattr(trade, "exit_time", None)
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def _avg_mfe_r(trades: list) -> float:
    if not trades:
        return 0.0
    values = [float(getattr(trade, "mfe_r", 0.0) or 0.0) for trade in trades]
    return float(np.mean(values)) if values else 0.0


def _winner_capture_ratio(trades: list) -> float:
    captures = []
    for trade in trades:
        r_multiple = float(getattr(trade, "r_multiple", 0.0) or 0.0)
        mfe_r = float(getattr(trade, "mfe_r", 0.0) or 0.0)
        if r_multiple > 0 and mfe_r > 0:
            captures.append(r_multiple / mfe_r)
    return float(np.mean(captures)) if captures else 0.0
