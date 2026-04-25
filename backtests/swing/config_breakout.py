"""Breakout v3.3-ETF backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.swing.config import SlippageConfig


def _default_breakout_symbols() -> list[str]:
    try:
        from strategies.swing.breakout.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


@dataclass
class BreakoutAblationFlags:
    """Toggle each Breakout-specific filter/condition for ablation testing.

    Baseline: all False (nothing disabled).  Set one to True to disable it.
    """

    disable_score_threshold: bool = False
    disable_chop_mode: bool = False
    disable_dirty: bool = False
    disable_breakout_quality_reject: bool = False
    disable_entry_b_rvol_gating: bool = False
    disable_add_rvol_acceptance: bool = False
    disable_micro_guard: bool = False
    disable_friction_gate: bool = False
    disable_corr_heat_penalty: bool = False
    disable_pending: bool = False
    disable_hard_block: bool = False
    disable_displacement: bool = False
    disable_continuation: bool = False
    disable_c_continuation_entry: bool = False
    disable_regime_chop_block: bool = False
    disable_reentry_gate: bool = False
    disable_stale_exit: bool = False


@dataclass
class BreakoutBacktestConfig:
    """Top-level Breakout backtest configuration."""

    symbols: list[str] = field(default_factory=_default_breakout_symbols)
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    flags: BreakoutAblationFlags = field(default_factory=BreakoutAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_shadows: bool = True
    track_signals: bool = True
    warmup_daily: int = 60
    warmup_hourly: int = 55
    warmup_4h: int = 50
    fixed_qty: int | None = None
    breakout_day_entry: bool = False
