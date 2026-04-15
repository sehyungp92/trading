"""BRS R9 live configuration -- adapted from backtest config_brs.py.

Backtest-only fields removed (initial_equity, start_date, end_date, data_dir,
slippage, warmup_*, param_overrides).  Live-specific fields added (STRATEGY_ID,
COMMISSION_PER_SHARE, build_instruments, OMS risk constants).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from libs.oms.models.instrument import Instrument


# ---------------------------------------------------------------------------
# Strategy identity & OMS risk constants
# ---------------------------------------------------------------------------

STRATEGY_ID = "BRS_R9"
COMMISSION_PER_SHARE = 0.0035  # match backtest (IBKR tiered US equities)
BASE_RISK_PCT = 0.003          # GLD rate (higher of the two; per-symbol handled internally)
DAILY_STOP_R = 2.0
HEAT_CAP_R = 1.80
PORTFOLIO_DAILY_STOP_R = 1.5


class TF(Enum):
    """Timeframe enum for bar series."""
    D1 = "1 day"
    H4 = "4 hours"
    H1 = "1 hour"


# ---------------------------------------------------------------------------
# Per-symbol config (spec Table 15)
# ---------------------------------------------------------------------------

@dataclass
class BRSSymbolConfig:
    """Per-symbol thresholds and risk parameters."""
    adx_on: int = 16
    adx_off: int = 14
    daily_mult: float = 2.3
    hourly_mult: float = 2.9
    chand_mult: float = 3.2
    stop_buffer_atr: float = 0.4
    stop_floor_atr: float = 1.0
    base_risk_pct: float = 0.005
    limit_pct: float = 0.0010
    allow_long: bool = False
    cooldown_bars: int = 8


BRS_SYMBOL_DEFAULTS: dict[str, BRSSymbolConfig] = {
    "QQQ": BRSSymbolConfig(
        adx_on=16, adx_off=14,
        daily_mult=2.3, hourly_mult=2.9, chand_mult=2.5,
        stop_buffer_atr=0.3, stop_floor_atr=0.6,
        base_risk_pct=0.002, limit_pct=0.0010,
        allow_long=False, cooldown_bars=8,
    ),
    "GLD": BRSSymbolConfig(
        adx_on=14, adx_off=12,
        daily_mult=2.0, hourly_mult=2.6, chand_mult=3.6,
        stop_buffer_atr=0.18, stop_floor_atr=0.6,
        base_risk_pct=0.003, limit_pct=0.0010,
        allow_long=True, cooldown_bars=8,
    ),
}


def _default_symbol_configs() -> dict[str, BRSSymbolConfig]:
    return {k: BRSSymbolConfig(**v.__dict__) for k, v in BRS_SYMBOL_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# Top-level BRS config (live -- backtest fields removed)
# ---------------------------------------------------------------------------

@dataclass
class BRSConfig:
    """Top-level BRS live configuration."""

    symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    symbol_configs: dict[str, BRSSymbolConfig] = field(default_factory=_default_symbol_configs)

    # Commission (replaces SlippageConfig)
    commission_per_share: float = COMMISSION_PER_SHARE

    # Global regime params -- R9 optimal
    adx_strong: int = 25
    ema_fast_period: int = 15
    ema_slow_period: int = 40
    regime_bear_min_conditions: int = 4

    # Fast-crash bias confirmation (Path D)
    fast_crash_enabled: bool = True
    fast_crash_return_thresh: float = -0.02
    fast_crash_atr_ratio: float = 2.0

    # Crash override
    crash_override_enabled: bool = True
    crash_override_return: float = -0.03
    crash_override_atr_ratio: float = 1.8

    # Peak drawdown regime trigger
    peak_drop_enabled: bool = True
    peak_drop_lookback: int = 16
    peak_drop_pct: float = -0.02

    # Return-only bias confirmation Path F
    return_only_enabled: bool = False
    return_only_thresh: float = -0.015

    # Cumulative return bias confirmation Path G
    cum_return_enabled: bool = False
    cum_return_lookback: int = 5
    cum_return_thresh: float = -0.03

    # 4H regime as bias accelerator
    bias_4h_accel_enabled: bool = True
    bias_4h_accel_reduction: int = 1

    # BEAR_FORMING entry with vol expansion gate
    forming_entry_enabled: bool = False
    forming_vol_gate: float = 1.5

    # RANGE_CHOP+SHORT entry override (R8)
    chop_short_entry_enabled: bool = True

    # Churn bridge
    churn_bridge_bars: int = 0

    # GLD cross-symbol override
    gld_qqq_override_enabled: bool = False
    gld_qqq_override_bear_trend: bool = False

    # Persistence override
    persistence_override_bars: int = 7

    # Pyramiding -- R9 optimal
    pyramid_enabled: bool = True
    pyramid_min_r: float = 1.0
    pyramid_scale: float = 0.5
    max_pyramid_adds: int = 1

    # Regime-adaptive sizing multipliers -- R9 optimal
    size_mult_bear_trend: float = 1.0
    size_mult_bear_strong: float = 1.0
    size_mult_range_chop: float = 1.0
    size_mult_bear_forming: float = 1.0

    # Quality multipliers -- R9 optimal
    chop_quality_mult: float = 0.84
    persist_quality_mult_lh: float = 0.50
    persist_quality_mult_bd: float = 0.96

    # BD regime restrictions
    bd_allow_bear_strong: bool = True

    # Quality score gate for L1
    min_quality_score: float = 0.0

    # Global entry params
    s1_ema_pull_strong: int = 34
    s1_ema_pull_normal: int = 50
    s2_box_containment: float = 0.75
    s2_disp_quantile: float = 0.65
    s3_donchian_period: int = 26

    # LH Rejection params
    lh_swing_lookback: int = 5
    lh_arm_bars: int = 26
    lh_max_stop_atr: float = 2.5
    v_reversal_max_rally: float = 1.5
    tod_filter_enabled: bool = False
    tod_filter_start: int = 840
    tod_filter_end: int = 930

    # BD Continuation params -- R9 optimal
    bd_volume_mult: float = 1.2
    bd_close_quality: float = 0.45
    bd_arm_bars: int = 24
    bd_max_stop_atr: float = 4.2
    bd_donchian_period: int = 10
    bd_allow_bear_trend: bool = True

    # BEAR_TREND volume filter for LHR
    bt_volume_mult: float = 2.0

    # Global exit params -- R9 optimal
    catastrophic_cap_r: float = 2.5
    be_trigger_r: float = 0.75
    trail_trigger_r: float = 0.75
    stale_bars_short: int = 50
    stale_bars_long: int = 30
    stale_early_bars: int = 35
    time_decay_hours: int = 360
    profit_floor_scale: float = 4.1472

    # Minimum hold time
    min_hold_bars: int = 5

    # Regime-adaptive cooldown
    cooldown_bear_strong: int = 2
    cooldown_bear_trend: int = 4

    # Wider BEAR_STRONG initial stops
    stop_floor_bear_strong_mult: float = 1.5

    # Regime-aware trailing
    trail_regime_scaling: bool = True

    # Scale-out / partial exit -- R9 optimal
    scale_out_enabled: bool = True
    scale_out_pct: float = 0.33
    scale_out_target_r: float = 3.6
    scale_out_trail_bonus: float = 0.3

    # Portfolio -- R9 optimal
    heat_cap_r: float = 1.80
    crisis_heat_cap_r: float = 1.25
    max_concurrent: int = 3

    # Ablation flags
    disable_s1: bool = False
    disable_s2: bool = True
    disable_s3: bool = True
    disable_l1: bool = True
    disable_long_safety: bool = False
    disable_lh: bool = False
    disable_bd: bool = False

    # Overrides (live: just extreme_vol_pct)
    extreme_vol_pct: float = 90.0

    def get_symbol_config(self, symbol: str) -> BRSSymbolConfig:
        """Get per-symbol config, falling back to QQQ defaults."""
        return self.symbol_configs.get(symbol, BRS_SYMBOL_DEFAULTS.get("QQQ", BRSSymbolConfig()))

    def resolve_param(self, name: str, symbol: str | None = None) -> float:
        """Resolve a parameter by field name."""
        if hasattr(self, name):
            return float(getattr(self, name))
        return 0.0


# ---------------------------------------------------------------------------
# Instrument builder
# ---------------------------------------------------------------------------

def build_instruments() -> list[Instrument]:
    """Register QQQ and GLD as OMS instruments."""
    from libs.oms.models.instrument_registry import InstrumentRegistry

    instruments = []
    for sym in ("QQQ", "GLD"):
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue="SMART",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            currency="USD",
            point_value=1.0,
            contract_expiry="",
        )
        InstrumentRegistry.register(inst)
        instruments.append(inst)
    return instruments
