"""Helix v4.0 Apex Trail backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.momentum.config import SlippageConfig


@dataclass
class Helix4AblationFlags:
    """Toggle each gate/mechanism for Helix v4.0 ablation testing.

    Baseline: all ``True`` (enabled).
    Set one to ``False`` to measure its contribution.
    """

    # Gate toggles
    use_extension_gate: bool = True
    use_spike_filter: bool = True
    use_spread_gate: bool = True
    use_dead_zone_midday: bool = True       # RTH_DEAD block
    use_reopen_dead_zone: bool = True       # REOPEN_DEAD block
    use_extreme_vol_mode: bool = True       # Extreme vol tightens sizing
    use_min_stop_gate: bool = True
    use_news_guard: bool = True
    use_teleport_penalty: bool = True       # Teleport locks out adds until +2R

    # Class toggles
    use_class_F: bool = False              # Class F fast re-entry (disabled: -$6.3k drag)
    use_class_T: bool = True               # Class T 4H trend continuation
    enable_ETH_Europe: bool = False        # ETH_EUROPE longs (disabled: -$6.9k drag)
    use_eth_europe_regime: bool = False    # Regime-conditional ETH_EUROPE (off by default)
    use_rtd_dead_enhanced: bool = True     # RTH_DEAD enhanced sizing (matches live)

    # Position management toggles
    use_momentum_stall: bool = False       # bar 8 stall exit (disabled: -$9.9k drag)
    use_trailing: bool = True
    use_stale_exit: bool = True
    use_early_stale: bool = True
    use_mfe_ratchet: bool = True           # NEW: MFE ratchet floor
    use_time_decay: bool = True            # Single BE force at bar 12

    # Risk management
    use_drawdown_throttle: bool = True     # DD-based sizing reduction + daily loss cap
    block_monday: bool = True              # Validated: Monday -$6.7k at 27% WR
    block_wednesday: bool = True           # Block all entries on Wednesdays
    use_hour_sizing: bool = True           # Hour-of-day sizing multiplier
    use_chop_sizing: bool = True           # Chop zone sizing penalty (replaces hard gate)
    short_min_score: int = 1              # Minimum alignment score required for short entries
    vol_50_80_sizing_mult: float = 0.60    # Size penalty in vol 50-80% regime


@dataclass
class Helix4BacktestConfig:
    """Top-level Helix v4.0 Apex Trail backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["NQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(
        default_factory=lambda: SlippageConfig(
            commission_per_contract=0.62,  # MNQ per side
            illiquid_hours=(0, 1, 2, 3, 4, 5, 6, 7, 23),
        ),
    )
    flags: Helix4AblationFlags = field(default_factory=Helix4AblationFlags)
    param_overrides: dict[str, float] = field(default_factory=lambda: {"HIGH_VOL_M_THRESHOLD": 97})
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_signals: bool = True
    track_shadows: bool = True
    news_calendar_path: Path | None = None

    # Warmup periods (in bars of each timeframe)
    warmup_daily: int = 60
    warmup_1h: int = 55
    warmup_4h: int = 55

    # MNQ instrument defaults
    tick_size: float = 0.25
    point_value: float = 2.0
    fixed_qty: int | None = 10

    # Synthetic spread model (points by session)
    spread_model_rth: float = 0.50
    spread_model_eth_quality: float = 1.00
    spread_model_eth_europe: float = 1.25
    spread_model_overnight: float = 1.75

    catastrophic_r: float = -1.5
