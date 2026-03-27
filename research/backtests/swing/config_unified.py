"""Unified multi-strategy backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from backtest.config import AblationFlags, BacktestConfig, SlippageConfig
from backtest.config_helix import HelixAblationFlags, HelixBacktestConfig
from backtest.config_breakout import BreakoutAblationFlags, BreakoutBacktestConfig


@dataclass
class StrategySlot:
    """Per-strategy portfolio-level parameters (mirrors main_multi.py)."""

    strategy_id: str
    priority: int            # 0 = highest
    unit_risk_pct: float     # fraction of NAV per 1R
    max_heat_R: float        # per-strategy heat ceiling in R
    daily_stop_R: float      # per-strategy daily stop in R (positive value)
    max_working_orders: int = 4


# Default strategy slots — P1 heat-unlock optimized (Mar 14).
# Raised heat_cap 2.0→3.0 unlocked 682 blocked entries, +54% total PnL,
# Sharpe 1.24→1.52, ratio 72:28→58:42 overlay:active.
# ATRSS(0): highest expectancy, always gets first fill.
# S5_PB(1): 80% WR on IBIT, rare signals.
# S5_DUAL(2): 70.7% WR on GLD+IBIT, fills IBIT coverage gap.
# Breakout(3): expanded to QQQ/GLD/IBIT.
# Helix(4): biggest beneficiary of heat unlock (87→323 trades, $1.5k→$6.5k).
ATRSS_SLOT = StrategySlot(
    strategy_id="ATRSS", priority=0,
    unit_risk_pct=0.018, max_heat_R=1.50, daily_stop_R=2.0,
    max_working_orders=4,
)
S5_PB_SLOT = StrategySlot(
    strategy_id="S5_PB", priority=1,
    unit_risk_pct=0.012, max_heat_R=2.00, daily_stop_R=2.0,  # greedy v4: was 1.50
    max_working_orders=2,
)
S5_DUAL_SLOT = StrategySlot(
    strategy_id="S5_DUAL", priority=2,
    unit_risk_pct=0.012, max_heat_R=2.00, daily_stop_R=2.0,  # greedy v4: was 1.50
    max_working_orders=2,
)
BREAKOUT_SLOT = StrategySlot(
    strategy_id="SWING_BREAKOUT_V3", priority=3,
    unit_risk_pct=0.008, max_heat_R=1.50, daily_stop_R=2.0,  # greedy v4: was 1.00
    max_working_orders=2,
)
HELIX_SLOT = StrategySlot(
    strategy_id="AKC_HELIX", priority=4,
    unit_risk_pct=0.008, max_heat_R=1.20, daily_stop_R=2.5,
    max_working_orders=4,
)


@dataclass
class UnifiedBacktestConfig:
    """Top-level config for unified 5-strategy portfolio backtest."""

    initial_equity: float = 10_000.0
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    start_date: str | None = None  # "YYYY-MM-DD" — trim data before this date
    end_date: str | None = None    # "YYYY-MM-DD" — trim data after this date
    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    # Symbol lists per strategy (defaults match production)
    atrss_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    helix_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD", "IBIT"])
    breakout_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD", "IBIT"])

    # Portfolio-level risk rules
    heat_cap_R: float = 3.0
    portfolio_daily_stop_R: float = 4.0
    atrss: StrategySlot = field(default_factory=lambda: ATRSS_SLOT)
    helix: StrategySlot = field(default_factory=lambda: HELIX_SLOT)
    breakout: StrategySlot = field(default_factory=lambda: BREAKOUT_SLOT)
    s5_pb: StrategySlot = field(default_factory=lambda: S5_PB_SLOT)
    s5_dual: StrategySlot = field(default_factory=lambda: S5_DUAL_SLOT)
    s5_pb_symbols: list[str] = field(default_factory=lambda: ["IBIT"])
    s5_dual_symbols: list[str] = field(default_factory=lambda: ["GLD", "IBIT"])

    # Cross-strategy coordination
    enable_atrss_helix_tighten: bool = True
    enable_atrss_helix_size_boost: bool = True

    # Warmup periods
    warmup_daily: int = 60
    warmup_hourly: int = 55
    warmup_4h: int = 50

    # Position sizing — fixed_qty=10 for ETFs (matches individual runners)
    fixed_qty: int | None = None

    # Idle-capital overlay (EMA crossover on daily closes)
    overlay_enabled: bool = True
    overlay_mode: str = "ema"            # "ema" (legacy) or "multi" (EMA+RSI+MACD)
    overlay_ema_fast: int = 13           # fast EMA period
    overlay_ema_slow: int = 48           # slow EMA period
    overlay_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    overlay_ema_overrides: dict[str, tuple[int, int]] = field(default_factory=lambda: {"QQQ": (10, 21), "GLD": (13, 21)})
    # Maps symbol -> (fast_period, slow_period). Symbols not in dict use overlay_ema_fast/slow defaults.
    overlay_max_pct: float = 0.85        # max fraction of equity for overlay
    overlay_weights: dict[str, float] | None = None
    # Per-symbol relative weights for overlay capital allocation.
    # None = equal-weight (current behavior). Renormalized across bullish symbols only.

    # Multi-overlay RSI parameters
    overlay_rsi_period: int = 14
    overlay_rsi_overbought: float = 70.0
    overlay_rsi_bull_min: float = 40.0
    overlay_rsi_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-symbol RSI overrides: {"QQQ": {"period": 10, "overbought": 75}} etc.

    # Multi-overlay MACD parameters
    overlay_macd_fast: int = 12
    overlay_macd_slow: int = 26
    overlay_macd_signal: int = 9
    overlay_macd_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-symbol MACD overrides: {"IBIT": {"fast": 8, "slow": 21, "signal": 7}} etc.

    # Multi-overlay scoring and sizing
    overlay_entry_score_min: float = 0.6   # minimum composite score to enter
    overlay_exit_score_max: float = 0.3    # exit below this (asymmetric hysteresis)
    overlay_adaptive_sizing: bool = True
    overlay_min_alloc_pct: float = 0.30    # min allocation pct when adaptive
    overlay_max_alloc_pct: float = 1.00    # max allocation pct when adaptive

    # Multi-overlay score component weights (EMA, RSI, MACD) — must sum to 1.0
    overlay_score_weights: tuple[float, float, float] = (0.40, 0.30, 0.30)
    # EMA spread normalization: spread/norm capped at 1.0. Lower = more sensitive
    overlay_ema_spread_norm: float = 0.01
    # MACD histogram score mapping: (pos_rising, pos_falling, neg_falling, neg_rising)
    overlay_macd_scores: tuple[float, float, float, float] = (1.0, 0.6, 0.0, 0.3)

    # Simulate live OMS R normalization bug: use per-strategy URD for
    # portfolio R sums instead of consistent portfolio base. This makes the
    # backtest match live's more conservative behavior for impact measurement.
    simulate_live_r_normalization: bool = False

    # Per-strategy per-symbol risk multipliers.
    # Maps "STRATEGY_ID:SYMBOL" -> multiplier that scales base_risk_pct.
    # e.g. {"ATRSS:QQQ": 0.8, "ATRSS:GLD": 1.2}. Missing entries default to 1.0.
    symbol_risk_multipliers: dict[str, float] = field(default_factory=dict)

    # Per-strategy ablation flags (baseline = all enabled)
    atrss_flags: AblationFlags = field(default_factory=AblationFlags)
    helix_flags: HelixAblationFlags = field(default_factory=HelixAblationFlags)
    breakout_flags: BreakoutAblationFlags = field(default_factory=BreakoutAblationFlags)

    # Per-strategy engine param overrides (applied by build_*_config methods)
    # These allow greedy optimizers to route strategy-specific params through
    # the unified config without modifying per-strategy config classes.
    atrss_param_overrides: dict = field(default_factory=dict)
    helix_param_overrides: dict = field(default_factory=dict)
    breakout_param_overrides: dict = field(default_factory=dict)
    s5_pb_param_overrides: dict = field(default_factory=dict)
    s5_dual_param_overrides: dict = field(default_factory=dict)

    def build_atrss_config(self) -> BacktestConfig:
        slippage = self.slippage
        if self.fixed_qty is not None:
            slippage = SlippageConfig(commission_per_contract=1.00)
        cfg = BacktestConfig(
            symbols=self.atrss_symbols,
            initial_equity=self.initial_equity,
            slippage=slippage,
            flags=self.atrss_flags,
            data_dir=self.data_dir,
            track_shadows=False,
            warmup_daily=self.warmup_daily,
            warmup_hourly=self.warmup_hourly,
            fixed_qty=self.fixed_qty,
        )
        if self.atrss_param_overrides:
            merged = {**cfg.param_overrides, **self.atrss_param_overrides}
            cfg = replace(cfg, param_overrides=merged)
        return cfg

    def build_helix_config(self) -> HelixBacktestConfig:
        slippage = self.slippage
        if self.fixed_qty is not None:
            slippage = SlippageConfig(commission_per_contract=1.00)
        cfg = HelixBacktestConfig(
            symbols=self.helix_symbols,
            initial_equity=self.initial_equity,
            slippage=slippage,
            flags=self.helix_flags,
            data_dir=self.data_dir,
            track_shadows=False,
            warmup_daily=self.warmup_daily,
            warmup_hourly=self.warmup_hourly,
            warmup_4h=self.warmup_4h,
            fixed_qty=self.fixed_qty,
        )
        if self.helix_param_overrides:
            merged = {**cfg.param_overrides, **self.helix_param_overrides}
            cfg = replace(cfg, param_overrides=merged)
        return cfg

    def build_breakout_config(self) -> BreakoutBacktestConfig:
        slippage = self.slippage
        if self.fixed_qty is not None:
            slippage = SlippageConfig(commission_per_contract=1.00)
        cfg = BreakoutBacktestConfig(
            symbols=self.breakout_symbols,
            initial_equity=self.initial_equity,
            slippage=slippage,
            flags=self.breakout_flags,
            data_dir=self.data_dir,
            track_shadows=False,
            warmup_daily=self.warmup_daily,
            warmup_hourly=self.warmup_hourly,
            warmup_4h=self.warmup_4h,
            fixed_qty=self.fixed_qty,
        )
        if self.breakout_param_overrides:
            merged = {**getattr(cfg, 'param_overrides', {}), **self.breakout_param_overrides}
            cfg = replace(cfg, param_overrides=merged)
        return cfg

    def build_s5_pb_config(self) -> "S5BacktestConfig":
        from backtest.config_s5 import S5BacktestConfig
        cfg = S5BacktestConfig(
            symbols=self.s5_pb_symbols,
            initial_equity=self.initial_equity,
            slippage=self.slippage,
            data_dir=self.data_dir,
            entry_mode="pullback", kelt_ema_period=10,
            roc_period=5, atr_stop_mult=1.5,
            risk_pct=self.s5_pb.unit_risk_pct,
        )
        if self.s5_pb_param_overrides:
            cfg = replace(cfg, **self.s5_pb_param_overrides)
        return cfg

    def build_s5_dual_config(self) -> "S5BacktestConfig":
        from backtest.config_s5 import S5BacktestConfig
        cfg = S5BacktestConfig(
            symbols=self.s5_dual_symbols,
            initial_equity=self.initial_equity,
            slippage=self.slippage,
            data_dir=self.data_dir,
            entry_mode="dual", kelt_ema_period=15,
            shorts_enabled=False, rsi_entry_long=45.0,
            risk_pct=self.s5_dual.unit_risk_pct,
        )
        if self.s5_dual_param_overrides:
            cfg = replace(cfg, **self.s5_dual_param_overrides)
        return cfg


# ---------------------------------------------------------------------------
# Preset factory functions — each returns a fully configured UnifiedBacktestConfig
# with risk-based sizing (fixed_qty=None).
# ---------------------------------------------------------------------------

def _slot(sid: str, priority: int, urp: float, mh: float, ds: float,
          mwo: int = 4) -> StrategySlot:
    """Shorthand for building a StrategySlot."""
    return StrategySlot(
        strategy_id=sid, priority=priority,
        unit_risk_pct=urp, max_heat_R=mh, daily_stop_R=ds,
        max_working_orders=mwo,
    )


def make_baseline(equity: float) -> UnifiedBacktestConfig:
    """Pre-optimized production parameters: ATRSS(0) > Breakout(1) > Helix(2)."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.005, 0.85, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.65, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_a1_atrss_tilt(equity: float) -> UnifiedBacktestConfig:
    """A1: Increase ATRSS allocation (best expectancy), reduce others."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.015, 1.2, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.003, 0.5, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.003, 0.4, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_a2_equal_alloc(equity: float) -> UnifiedBacktestConfig:
    """A2: Equal allocation — all strategies get same risk budget."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.0075, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.0075, 1.0, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.0075, 1.0, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_b1_tight_heat(equity: float) -> UnifiedBacktestConfig:
    """B1: Tight heat — reduce portfolio heat cap to force selective entry."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 0.6, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.005, 0.5, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.3, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_b2_expanded_heat(equity: float) -> UnifiedBacktestConfig:
    """B2: Expanded heat — allow more concurrent positions."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.2, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.005, 1.0, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.8, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_c1_old_priority(equity: float) -> UnifiedBacktestConfig:
    """C1: Old priority ordering (Helix > Breakout) for comparison."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 2, 0.005, 0.65, 2.0, mwo=2),
        overlay_ema_overrides={},
    )


def make_d1_no_coordination(equity: float) -> UnifiedBacktestConfig:
    """D1: Disable tighten + boost to measure coordination net impact."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.005, 0.85, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.65, 2.0, mwo=2),
        enable_atrss_helix_tighten=False,
        enable_atrss_helix_size_boost=False,
        overlay_ema_overrides={},
    )


def make_e1_tighter_daily_stops(equity: float) -> UnifiedBacktestConfig:
    """E1: Tighter daily stops — conservative, better for small accounts."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=2.5,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 1.5),
        helix=_slot("AKC_HELIX", 2, 0.005, 0.85, 2.0),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.65, 1.5, mwo=2),
        overlay_ema_overrides={},
    )


def make_optimized_v1(equity: float) -> UnifiedBacktestConfig:
    """Optimized v1: per-asset EMAs (QQQ 10/21, GLD 13/21) + ATRSS risk boost to 1.2%.

    Pinned to pre-P1 defaults for backward compatibility.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.012, 1.0, 2.0),
        s5_pb=_slot("S5_PB", 1, 0.008, 1.5, 2.0, mwo=2),
        s5_dual=_slot("S5_DUAL", 2, 0.008, 1.5, 2.0, mwo=2),
        breakout=_slot("SWING_BREAKOUT_V3", 3, 0.005, 0.65, 2.0, mwo=2),
        helix=_slot("AKC_HELIX", 4, 0.005, 0.85, 2.5),
        breakout_symbols=["QQQ", "GLD"],
    )


def make_multi_overlay(equity: float) -> UnifiedBacktestConfig:
    """Multi-overlay: EMA+RSI+MACD scoring with adaptive sizing on QQQ/GLD (optimized).

    Uses the optimized_v1 active strategy parameters with the enhanced
    multi-indicator overlay for idle capital deployment. Parameter sweep
    (Feb 19) found 50/50 QQQ/GLD allocation (no IBIT) achieves best Sharpe (1.22).
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        overlay_enabled=True,
        overlay_mode="multi",
        overlay_symbols=["QQQ", "GLD"],
        overlay_ema_overrides={"QQQ": (10, 21), "GLD": (13, 21)},
        overlay_weights={"QQQ": 0.50, "GLD": 0.50},
        overlay_rsi_period=14,
        overlay_rsi_overbought=70.0,
        overlay_rsi_bull_min=40.0,
        overlay_macd_fast=12,
        overlay_macd_slow=26,
        overlay_macd_signal=9,
        overlay_entry_score_min=0.6,
        overlay_exit_score_max=0.3,
        overlay_adaptive_sizing=True,
        overlay_min_alloc_pct=0.30,
        overlay_max_alloc_pct=1.00,
        overlay_score_weights=(0.40, 0.30, 0.30),
        overlay_macd_scores=(1.0, 0.6, 0.0, 0.3),
    )


def make_f1_breakout_expansion(equity: float) -> UnifiedBacktestConfig:
    """F1: Add IBIT to Breakout symbols."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 2, 0.005, 0.85, 2.5),
        breakout=_slot("SWING_BREAKOUT_V3", 1, 0.005, 0.65, 2.0, mwo=2),
        breakout_symbols=["QQQ", "GLD", "IBIT"],
        overlay_ema_overrides={},
    )


def make_live_parity(equity: float) -> UnifiedBacktestConfig:
    """Live parity: greedy v4 optimized defaults (Mar 25).

    Uses default StrategySlot values:
      ATRSS(0)  URD 1.8%  max_heat 1.50R  daily_stop 2.0R
      S5_PB(1)  URD 1.2%  max_heat 2.00R  daily_stop 2.0R
      S5_DUAL(2) URD 1.2% max_heat 2.00R  daily_stop 2.0R
      Breakout(3) URD 0.8% max_heat 1.50R daily_stop 2.0R  (+IBIT)
      Helix(4)  URD 0.8%  max_heat 1.20R  daily_stop 2.5R
      heat_cap_R=3.0, portfolio_daily_stop_R=4.0
    Greedy v4 mutations: stall_exit disabled, S5_DUAL trail/stop,
    breakout score_threshold=0, regime_chop_block=False.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        atrss_flags=AblationFlags(stall_exit=False),  # greedy v4
        s5_dual_param_overrides={                      # greedy v4
            "trail_atr_mult": 2.0,
            "atr_stop_mult": 1.5,
        },
    )


def make_live_r_simulation(equity: float) -> UnifiedBacktestConfig:
    """Simulate live OMS R normalization bug for impact measurement.

    Uses current live_parity defaults with per-strategy URD for portfolio
    R sums instead of consistent portfolio base.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        simulate_live_r_normalization=True,
    )


def make_p1_heat_unlock(equity: float) -> UnifiedBacktestConfig:
    """P1: Heat unlock — now the default configuration (Mar 14).

    Winner of P1-P4 optimization sweep: +54% PnL, Sharpe 1.24→1.52,
    ratio 72:28→58:42, 952→270 blocked entries. Kept as a named preset
    for backward compatibility — identical to live_parity/defaults.
    """
    return UnifiedBacktestConfig(initial_equity=equity)


def make_p2_aggressive_active(equity: float) -> UnifiedBacktestConfig:
    """P2: Maximum active push — highest risk on proven strategies.

    Pushes active allocation to the limit with aggressive sizing on
    high-expectancy strategies (ATRSS, S5) and moderate Helix increase.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=3.5,
        portfolio_daily_stop_R=4.5,
        atrss=_slot("ATRSS", 0, 0.020, 1.8, 2.0),
        s5_pb=_slot("S5_PB", 1, 0.015, 2.5, 2.0, mwo=2),
        s5_dual=_slot("S5_DUAL", 2, 0.015, 2.5, 2.0, mwo=2),
        breakout=_slot("SWING_BREAKOUT_V3", 3, 0.010, 1.2, 2.0, mwo=2),
        helix=_slot("AKC_HELIX", 4, 0.006, 1.0, 2.5),
        breakout_symbols=["QQQ", "GLD", "IBIT"],
        symbol_risk_multipliers={
            "ATRSS:QQQ": 1.3,
            "ATRSS:GLD": 1.3,
        },
    )


def make_p3_balanced_split(equity: float) -> UnifiedBacktestConfig:
    """P3: Approach 50:50 from both sides — moderate active boost + overlay reduction.

    Reduces overlay_max_pct to constrain overlay capital while moderately
    increasing active risk. May reduce total PnL — included as a data point.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.5,
        portfolio_daily_stop_R=3.5,
        atrss=_slot("ATRSS", 0, 0.015, 1.2, 2.0),
        s5_pb=_slot("S5_PB", 1, 0.010, 1.5, 2.0, mwo=2),
        s5_dual=_slot("S5_DUAL", 2, 0.010, 1.5, 2.0, mwo=2),
        breakout=_slot("SWING_BREAKOUT_V3", 3, 0.007, 0.65, 2.0, mwo=2),
        helix=_slot("AKC_HELIX", 4, 0.007, 1.0, 2.5),
        overlay_max_pct=0.65,
    )


def make_p4_multiplier_boost(equity: float) -> UnifiedBacktestConfig:
    """P4: Symbol risk multipliers + heat unlock — selective high-edge boosts.

    Uses symbol_risk_multipliers to selectively scale risk on proven
    strategy:symbol pairs instead of blanket increases.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        atrss=_slot("ATRSS", 0, 0.015, 1.5, 2.0),
        breakout_symbols=["QQQ", "GLD", "IBIT"],
        symbol_risk_multipliers={
            "ATRSS:QQQ": 1.5,
            "ATRSS:GLD": 1.3,
            "AKC_HELIX:IBIT": 1.4,
            "AKC_HELIX:GLD": 1.2,
        },
    )


PRESETS: dict[str, Callable[[float], UnifiedBacktestConfig]] = {
    "live_parity": make_live_parity,
    "live_r_simulation": make_live_r_simulation,
    "baseline": make_baseline,
    "a1_atrss_tilt": make_a1_atrss_tilt,
    "a2_equal_alloc": make_a2_equal_alloc,
    "b1_tight_heat": make_b1_tight_heat,
    "b2_expanded_heat": make_b2_expanded_heat,
    "c1_old_priority": make_c1_old_priority,
    "d1_no_coordination": make_d1_no_coordination,
    "e1_tighter_daily_stops": make_e1_tighter_daily_stops,
    "f1_breakout_expansion": make_f1_breakout_expansion,
    "optimized_v1": make_optimized_v1,
    "multi_overlay": make_multi_overlay,
    "p1_heat_unlock": make_p1_heat_unlock,
    "p2_aggressive_active": make_p2_aggressive_active,
    "p3_balanced_split": make_p3_balanced_split,
    "p4_multiplier_boost": make_p4_multiplier_boost,
}
