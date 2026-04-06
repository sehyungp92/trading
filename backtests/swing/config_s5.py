"""Backtest configuration for Strategy 5 (Keltner Momentum Breakout)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backtest.config import SlippageConfig


@dataclass
class S5BacktestConfig:
    """Configuration for a single S5 backtest run."""

    symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD", "IBIT"])
    initial_equity: float = 10_000.0
    slippage: SlippageConfig = field(default_factory=lambda: SlippageConfig(
        commission_per_contract=0.0035,
        spread_bps=2.0,
    ))
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    warmup_daily: int = 30

    # Keltner Channel
    kelt_ema_period: int = 20
    kelt_atr_period: int = 14
    kelt_atr_mult: float = 2.0

    # RSI
    rsi_period: int = 14
    rsi_entry_long: float = 50.0
    rsi_entry_short: float = 50.0

    # ROC
    roc_period: int = 10

    # Volume
    vol_sma_period: int = 20
    volume_filter: bool = True

    # Entry mode: breakout, pullback, momentum, dual
    entry_mode: str = "breakout"

    # Exit mode: trail_only, reversal, midline
    exit_mode: str = "trail_only"

    # Stop / risk
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    trail_atr_mult: float = 1.5
    risk_pct: float = 0.01

    # Shorts
    shorts_enabled: bool = True
