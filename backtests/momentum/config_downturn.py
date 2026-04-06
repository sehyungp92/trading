"""Downturn Dominator backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtest.config import SlippageConfig


@dataclass
class DownturnAblationFlags:
    """Toggle each gate/mechanism for ablation testing.

    Baseline: all ``True`` (enabled) except deferred features.
    """

    # Regime & Volatility
    use_composite_regime: bool = True
    use_volatility_states: bool = True
    use_shock_block: bool = True
    use_strong_bear_bonus: bool = True

    # Engine enables
    reversal_engine: bool = True
    breakdown_engine: bool = True
    fade_engine: bool = True

    # Reversal gates
    reversal_divergence_gate: bool = True
    reversal_trend_weakness_gate: bool = True  # 2-of-3 gate
    reversal_extension_gate: bool = True
    reversal_corridor_cap: bool = True

    # Breakdown gates
    breakdown_containment_gate: bool = True
    breakdown_displacement_gate: bool = True
    breakdown_chop_filter: bool = True
    breakdown_spike_reject: bool = True

    # Fade gates
    fade_vwap_rejection: bool = True
    fade_momentum_confirm: bool = True
    fade_bear_regime_required: bool = True
    fade_cap_gate: bool = True

    # Entry common
    use_entry_windows: bool = True
    use_dead_zones: bool = True
    use_news_blackout: bool = True

    # Exit / Position management
    tiered_exits: bool = True
    chandelier_trailing: bool = True
    stale_exit: bool = True
    climax_exit: bool = True
    vwap_failure_exit: bool = True  # Fade pre-+1R

    # Risk
    daily_circuit_breaker: bool = True
    directional_entry_caps: bool = True
    friction_gate: bool = True

    # Regime filters
    block_counter_regime: bool = False    # Block all entries during COUNTER regime
    correction_sizing_bonus: bool = False  # Boost sizing during correction windows
    non_correction_penalty: bool = False   # Reduce sizing outside correction windows

    # Structural gap fixes
    correction_regime_override: bool = False  # Override NEUTRAL/RANGE → EMERGING_BEAR during correction windows
    short_sma_trend: bool = False             # Use short SMA (e.g. 50) as alternative bear signal
    allow_reversal_strong_bear: bool = False  # Allow reversal signals during strong bear
    stale_to_tp: bool = False                 # Tag profitable stale exits as tp0 instead of stale

    # Exit enhancements
    profit_floor_trail: bool = False  # Lock fraction of profit after threshold R
    multi_tier_profit_floor: bool = False  # 5-tier MFE-based profit floor ratchet
    regime_adaptive_chandelier: bool = False  # Regime multiplier on chandelier trailing

    # BRS-inspired fast-crash override (paths E/F/G)
    fast_crash_override: bool = False  # Override regime to EMERGING_BEAR on real-time price drops
    conviction_scoring: bool = False   # Bear conviction quality gate (0-100) for overrides

    # BRS-inspired bear structure override (paths B/C + BEAR_FORMING)
    bear_structure_override: bool = False  # ADX hysteresis + gradual bear detection

    # Correction-specific gates
    allow_reversal_in_correction: bool = False  # Exempt reversal from block_counter_regime in corrections

    # Scale-out
    scale_out_enabled: bool = False  # Partial profit lock at target R

    # R6 — Adaptive exits
    adaptive_profit_floor: bool = False  # MFE-tiered lock_pct (higher capture from big winners)

    # R6 — Coverage expansion
    drawdown_regime_override: bool = False  # Real-time rolling-high drawdown → EMERGING_BEAR override
    progressive_sma: bool = False           # Use SMA(available_bars) when < sma200_period bars

    # R6 — Signal expansion
    momentum_signal: bool = False  # ROC-based fade alternative (no VWAP rejection needed)

    # R6 — Hold period
    min_hold_period: bool = False  # Skip exits for first N bars after entry (except catastrophic)

    # Deferred (optimize later)
    earn_the_hold: bool = False
    adds_enabled: bool = False
    breakdown_reentry: bool = False
    fade_reentry: bool = False


@dataclass
class DownturnBacktestConfig:
    """Top-level Downturn Dominator backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["NQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 10_000.0
    slippage: SlippageConfig = field(
        default_factory=lambda: SlippageConfig(
            commission_per_contract=0.62,
            slip_ticks_normal=1,
            slip_ticks_illiquid=2,
        ),
    )
    flags: DownturnAblationFlags = field(default_factory=DownturnAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("research/backtests/momentum/data/raw"))
    track_signals: bool = True
    warmup_days: int = 60

    # Instrument (MNQ — uses NQ price data, trades Micro E-mini)
    tick_size: float = 0.50
    point_value: float = 2.0
    max_contracts: int = 0  # 0 = unlimited; >0 = hard cap on position qty
    max_notional_leverage: float = 20.0  # cap qty to equity * leverage / notional (20x = realistic MNQ)
