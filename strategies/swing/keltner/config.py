"""Keltner Channel strategies — configuration.

Keltner Channels + RSI + ROC + Volume for multi-factor entries.
Active strategies: S5_PB (pullback), S5_DUAL (dual entry).
Base SYMBOL_CONFIGS retained for backtesting.
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

S5_PB_STRATEGY_ID = "S5_PB"
S5_DUAL_STRATEGY_ID = "S5_DUAL"


@dataclass(frozen=True)
class SymbolConfig:
    """Per-symbol configuration."""

    symbol: str
    tick_size: float = 0.01
    multiplier: float = 1.0
    exchange: str = "SMART"
    contract_expiry: str = ""
    is_etf: bool = True
    base_risk_pct: float = 0.01  # 1% of equity per trade

    # Keltner Channel
    kelt_ema_period: int = 20
    kelt_atr_period: int = 14
    kelt_atr_mult: float = 2.0

    # RSI
    rsi_period: int = 14
    rsi_entry_long: float = 50.0   # RSI must be above this for long
    rsi_entry_short: float = 50.0  # RSI must be below this for short

    # Rate of Change
    roc_period: int = 10

    # Volume filter
    vol_sma_period: int = 20
    volume_filter: bool = True

    # Entry mode: "breakout", "pullback", "momentum", "dual"
    entry_mode: str = "breakout"

    # Exit mode: "trail_only" (max hold), "reversal", "midline"
    exit_mode: str = "trail_only"

    # Stop / risk
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    trail_atr_mult: float = 1.5

    # Shorts
    shorts_enabled: bool = True
    # Experiment A/B tracking
    experiment_id: str = ""
    experiment_variant: str = ""


SYMBOL_CONFIGS: dict[str, SymbolConfig] = {
    "QQQ": SymbolConfig(symbol="QQQ"),
    "GLD": SymbolConfig(symbol="GLD"),
    "IBIT": SymbolConfig(symbol="IBIT"),
}

# ---------------------------------------------------------------------------
# Live variant configs (match backtest/config_unified.py optimized_v2)
# ---------------------------------------------------------------------------

S5_PB_CONFIGS: dict[str, SymbolConfig] = {
    "IBIT": SymbolConfig(
        symbol="IBIT", entry_mode="pullback", kelt_ema_period=10,
        roc_period=5, atr_stop_mult=1.5, base_risk_pct=0.008,
    ),
}

S5_DUAL_CONFIGS: dict[str, SymbolConfig] = {
    "GLD": SymbolConfig(
        symbol="GLD", entry_mode="dual", kelt_ema_period=15,
        shorts_enabled=False, rsi_entry_long=45.0, base_risk_pct=0.008,
        trail_atr_mult=2.0, atr_stop_mult=1.5,  # greedy v4: trail wider, stop tighter
    ),
    "IBIT": SymbolConfig(
        symbol="IBIT", entry_mode="dual", kelt_ema_period=15,
        shorts_enabled=False, rsi_entry_long=45.0, base_risk_pct=0.008,
        trail_atr_mult=2.0, atr_stop_mult=1.5,  # greedy v4: trail wider, stop tighter
    ),
}


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_instruments(configs: dict[str, SymbolConfig] | None = None) -> dict[str, Instrument]:
    """Create Instrument objects for every symbol and register them."""
    if configs is None:
        configs = SYMBOL_CONFIGS
    instruments: dict[str, Instrument] = {}
    for sym, cfg in configs.items():
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue=cfg.exchange,
            tick_size=cfg.tick_size,
            tick_value=cfg.tick_size * cfg.multiplier,
            multiplier=cfg.multiplier,
            contract_expiry=cfg.contract_expiry,
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
