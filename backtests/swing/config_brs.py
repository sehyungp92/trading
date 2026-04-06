"""BRS (Bear Regime Swing) backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtest.config import SlippageConfig


# ---------------------------------------------------------------------------
# Per-symbol config (spec Table 15, §12.1, §16.1)
# ---------------------------------------------------------------------------

@dataclass
class BRSSymbolConfig:
    """Per-symbol thresholds and risk parameters."""
    adx_on: int = 16
    adx_off: int = 14
    daily_mult: float = 2.3
    hourly_mult: float = 2.9
    chand_mult: float = 3.2
    stop_buffer_atr: float = 0.4   # ATR_h buffer added to structure stop
    stop_floor_atr: float = 1.0    # minimum stop distance in ATR_h units
    base_risk_pct: float = 0.005
    limit_pct: float = 0.0010
    allow_long: bool = False
    cooldown_bars: int = 8  # simplified fixed cooldown (RTH hours)


# Pre-built defaults from spec
BRS_SYMBOL_DEFAULTS: dict[str, BRSSymbolConfig] = {
    "QQQ": BRSSymbolConfig(
        adx_on=16, adx_off=14,
        daily_mult=2.3, hourly_mult=2.9, chand_mult=2.5,
        stop_buffer_atr=0.3, stop_floor_atr=0.6,  # R9 optimal
        base_risk_pct=0.002, limit_pct=0.0010,  # R9 leverage-capped
        allow_long=False, cooldown_bars=8,
    ),
    "GLD": BRSSymbolConfig(
        adx_on=14, adx_off=12,  # R9 optimal
        daily_mult=2.0, hourly_mult=2.6, chand_mult=3.6,
        stop_buffer_atr=0.18, stop_floor_atr=0.6,  # R9 optimal
        base_risk_pct=0.003, limit_pct=0.0010,  # R9 leverage-capped
        allow_long=True, cooldown_bars=8,
    ),
    "IBIT": BRSSymbolConfig(
        adx_on=16, adx_off=14,
        daily_mult=2.7, hourly_mult=3.2, chand_mult=3.5,
        base_risk_pct=0.003, limit_pct=0.0020,
        allow_long=False, cooldown_bars=8,
    ),
}


def _default_symbol_configs() -> dict[str, BRSSymbolConfig]:
    return {k: BRSSymbolConfig(**v.__dict__) for k, v in BRS_SYMBOL_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# Top-level BRS config
# ---------------------------------------------------------------------------

@dataclass
class BRSConfig:
    """Top-level BRS backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    symbol_configs: dict[str, BRSSymbolConfig] = field(default_factory=_default_symbol_configs)
    initial_equity: float = 10_000.0
    start_date: datetime | None = None
    end_date: datetime | None = None
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    # Global regime params (spec §2) — Round 2 optimal
    adx_strong: int = 25  # R9 optimal
    ema_fast_period: int = 15
    ema_slow_period: int = 40
    regime_bear_min_conditions: int = 4  # 4=current behavior; 2-3 activates BEAR_FORMING

    # Fast-crash bias confirmation (Path D) — enabled by default (Change #9)
    fast_crash_enabled: bool = True
    fast_crash_return_thresh: float = -0.02   # -2% daily drop
    fast_crash_atr_ratio: float = 2.0         # ATR14 > 2x ATR50 (volatility expansion)

    # Crash override: bypass regime_on on extreme daily drops
    crash_override_enabled: bool = True
    crash_override_return: float = -0.03   # -3% daily drop
    crash_override_atr_ratio: float = 1.8  # ATR14 > 1.8x ATR50

    # Peak drawdown regime trigger (Part A — Round 6)
    peak_drop_enabled: bool = True        # R9 optimal
    peak_drop_lookback: int = 16         # R9 optimal
    peak_drop_pct: float = -0.02         # R9 optimal

    # Return-only bias confirmation Path F (Part B — Round 6)
    return_only_enabled: bool = False
    return_only_thresh: float = -0.015   # -1.5% daily return

    # Cumulative return bias confirmation Path G (Round 7)
    cum_return_enabled: bool = False       # off by default, optimizer enables
    cum_return_lookback: int = 5           # rolling window in daily bars
    cum_return_thresh: float = -0.03       # -3% cumulative triggers

    # 4H regime as bias accelerator (Part C — Round 6)
    bias_4h_accel_enabled: bool = True   # R9 optimal
    bias_4h_accel_reduction: int = 1     # reduce hold_count requirement by N

    # BEAR_FORMING entry with vol expansion gate
    forming_entry_enabled: bool = False  # off by default, optimizer can enable
    forming_vol_gate: float = 1.5        # ATR14/ATR50 threshold

    # RANGE_CHOP+SHORT entry override (R8) — allow LH/BD when peak_drop fires
    chop_short_entry_enabled: bool = True  # R9 optimal

    # Churn bridge: carry forward hold_count across short FLAT interruptions (R8)
    churn_bridge_bars: int = 0  # 0=disabled; e.g., 3=bridge up to 3 bars of FLAT

    # GLD cross-symbol override: treat GLD regime_on=True when QQQ is BEAR_STRONG (R8)
    gld_qqq_override_enabled: bool = False

    # GLD override: also trigger on BEAR_TREND (not just BEAR_STRONG) (R8)
    gld_qqq_override_bear_trend: bool = False

    # Persistence override: relax signal structure after N SHORT-bias bars without trade (R8)
    persistence_override_bars: int = 7  # R9 optimal

    # Pyramiding: add to winning positions on new signals (R9)
    pyramid_enabled: bool = True   # R9 optimal
    pyramid_min_r: float = 1.0      # min current R-multiple to allow add-on
    pyramid_scale: float = 0.5      # add-on size as fraction of original qty
    max_pyramid_adds: int = 1        # max add-ons per trade (1 or 2)

    # Regime-adaptive sizing multipliers (R9)
    size_mult_bear_trend: float = 1.0    # R9 analysis: 1.3x hurts Sortino without proportional benefit
    size_mult_bear_strong: float = 1.0   # default same, optimizer can reduce
    size_mult_range_chop: float = 1.0    # default same, optimizer can reduce
    size_mult_bear_forming: float = 1.0  # default same

    # Configurable quality multipliers for override entries (R9)
    chop_quality_mult: float = 0.84         # R9 optimal
    persist_quality_mult_lh: float = 0.50   # unarmed LH persistence entries
    persist_quality_mult_bd: float = 0.96   # R9 optimal

    # BD regime restrictions
    bd_allow_bear_strong: bool = True  # ablatable — BD/BEAR_STRONG has 25% WR

    # Quality score gate for L1
    min_quality_score: float = 0.0  # 0=no filter

    # Global entry params
    s1_ema_pull_strong: int = 34    # EMA period for pullback in BEAR_STRONG
    s1_ema_pull_normal: int = 50    # EMA period for pullback in BEAR_TREND
    s2_box_containment: float = 0.75
    s2_disp_quantile: float = 0.65
    s3_donchian_period: int = 26

    # LH Rejection params
    lh_swing_lookback: int = 5
    lh_arm_bars: int = 26  # R9 optimal
    lh_max_stop_atr: float = 2.5
    v_reversal_max_rally: float = 1.5
    tod_filter_enabled: bool = False  # Round 2 optimal: disabled
    tod_filter_start: int = 840   # 14:00 ET in minutes
    tod_filter_end: int = 930     # 15:30 ET in minutes

    # BD Continuation params — relaxed defaults (Change #3)
    bd_volume_mult: float = 1.2
    bd_close_quality: float = 0.45
    bd_arm_bars: int = 24
    bd_max_stop_atr: float = 4.2   # R9 optimal
    bd_donchian_period: int = 10   # R9 optimal
    bd_allow_bear_trend: bool = True  # BD arms in BEAR_TREND too

    # BEAR_TREND volume filter for LHR
    bt_volume_mult: float = 2.0   # R9 optimal

    # Global exit params (spec §12) — Round 2 optimal
    catastrophic_cap_r: float = 2.5
    be_trigger_r: float = 0.75  # R9 optimal
    trail_trigger_r: float = 0.75
    stale_bars_short: int = 50
    stale_bars_long: int = 30
    stale_early_bars: int = 35     # was hardcoded 15 in exits.py
    time_decay_hours: int = 360
    profit_floor_scale: float = 4.1472  # R9 optimal

    # Minimum hold time — stop immunity window (Change #4)
    min_hold_bars: int = 5  # R9 optimal

    # Regime-adaptive cooldown (Change #5)
    cooldown_bear_strong: int = 2
    cooldown_bear_trend: int = 4

    # Wider BEAR_STRONG initial stops (Change #7)
    stop_floor_bear_strong_mult: float = 1.5  # R9 optimal

    # Regime-aware trailing (Change #8)
    trail_regime_scaling: bool = True

    # Scale-out / partial exit (Change #6)
    scale_out_enabled: bool = True    # R9 optimal
    scale_out_pct: float = 0.33       # R9 optimal
    scale_out_target_r: float = 3.6   # R9 optimal
    scale_out_trail_bonus: float = 0.3  # widen chand_mult after B exits

    # Portfolio (spec §16)
    heat_cap_r: float = 1.80
    crisis_heat_cap_r: float = 1.25
    max_concurrent: int = 3

    # Warmup
    warmup_daily: int = 60
    warmup_hourly: int = 55
    warmup_4h: int = 50

    # Ablation flags — set True to disable
    disable_s1: bool = False  # R9 optimal — S1 re-enabled
    disable_s2: bool = True   # Legacy — disabled
    disable_s3: bool = True   # Legacy — disabled
    disable_l1: bool = True   # Legacy — disabled
    disable_long_safety: bool = False

    # New signal flags
    disable_lh: bool = False  # LH Rejection — enabled
    disable_bd: bool = False  # BD Continuation — enabled

    # Param overrides (for greedy optimizer mutations)
    param_overrides: dict[str, float] = field(
        default_factory=lambda: {"extreme_vol_pct": 90}  # R9 optimal
    )

    def get_symbol_config(self, symbol: str) -> BRSSymbolConfig:
        """Get per-symbol config, falling back to QQQ defaults."""
        return self.symbol_configs.get(symbol, BRS_SYMBOL_DEFAULTS.get("QQQ", BRSSymbolConfig()))

    def resolve_param(self, name: str, symbol: str | None = None) -> float:
        """Resolve a parameter with overrides: symbol-specific > global > default.

        Checks param_overrides for '{name}_{symbol}' then '{name}'.
        """
        if symbol:
            sym_key = f"{name}_{symbol}"
            if sym_key in self.param_overrides:
                return float(self.param_overrides[sym_key])
        if name in self.param_overrides:
            return float(self.param_overrides[name])
        if hasattr(self, name):
            return float(getattr(self, name))
        return 0.0
