"""Static configuration for the IARIC v1 strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from libs.oms.models.instrument import Instrument

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
STRATEGY_ID = "IARIC_v1"
PROXY_SYMBOLS = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class StrategySettings:
    premarket_start: time = time(7, 0)
    premarket_end: time = time(8, 30)
    market_open: time = time(9, 30)
    open_block_end: time = time(9, 35)
    entry_end: time = time(15, 0)
    close_block_start: time = time(15, 45)
    forced_flatten: time = time(15, 58)

    hot_max: int = 40
    active_monitoring_target: int = 20
    warm_poll_interval_s: int = 30
    cold_poll_interval_s: int = 90
    monitoring_refresh_minutes: int = 60

    min_price: float = 10.0
    min_adv_usd: float = 20_000_000.0
    max_median_spread_pct: float = 0.0050
    avwap_band_pct: float = 0.0050
    avwap_acceptance_band_pct: float = 0.0100
    anchor_lookback_sessions: int = 40

    tier_a_min: float = 0.65
    tier_b_min: float = 0.40
    regime_full_mult_score: float = 0.825
    breadth_threshold_pct: float = 55.0
    vix_percentile_threshold: float = 80.0
    hy_spread_5d_bps_threshold: float = 15.0

    panic_flush_drop_pct: float = 0.03
    drift_exhaustion_drop_pct: float = 0.02
    warm_drop_pct: float = 0.007
    hot_drop_pct: float = 0.015
    panic_flush_minutes: int = 15
    drift_exhaustion_minutes: int = 60
    setup_stale_minutes: int = 45
    invalidation_cooldown_minutes: int = 15

    acceptance_base_closes: int = 2
    partial_r_multiple: float = 1.5
    partial_exit_fraction: float = 0.50
    partial_r_min: float = 1.2
    partial_r_max: float = 2.0
    partial_frac_min: float = 0.33
    partial_frac_max: float = 0.60
    time_stop_minutes: int = 45
    avwap_breakdown_pct: float = 0.007
    avwap_breakdown_volume_mult: float = 1.5

    base_risk_fraction: float = 0.0075
    intraday_leverage: float = 2.0             # Max leverage (2.0 = Reg T, 4.0 = PDT intraday)
    daily_stop_r: float = 2.0
    heat_cap_r: float = 1.25
    portfolio_daily_stop_r: float = 3.0
    sector_risk_cap_pct: float = 0.35
    max_positions_tier_a: int = 8
    max_positions_tier_b: int = 4
    max_positions_per_sector: int = 2
    minimum_remaining_size_pct: float = 0.30

    confidence_green_mult: float = 1.0
    confidence_yellow_mult: float = 0.65
    stale_penalty: float = 0.85

    # Backtest-sweepable parameters (defaults match hardcoded originals)
    stop_risk_cap_pct: float = 0.02          # max risk offset as fraction of price
    max_carry_days: int = 7                   # max overnight carry duration
    min_carry_r: float = 0.5                  # minimum R-multiple to qualify for carry
    flow_reversal_lookback: int = 1           # consecutive negative flow days to trigger exit
    min_conviction_multiplier: float = 0.25   # minimum conviction to enter (0 = any positive)

    # T2 intraday engine tuning (legacy FSM engine)
    t2_avwap_band_mult: float = 1.0        # multiplier on avwap_band_pct for T2 FSM band check

    # T2 v2: Entry triggers
    t2_vwap_pullback_atr_mult: float = 0.5   # VWAP pullback distance (ATR units)
    t2_vwap_reclaim_vol_pct: float = 0.80    # min reclaim bar volume vs expected
    t2_orb_cpr_min: float = 0.60             # min CPR on ORB breakout bar
    t2_afternoon_vol_mult: float = 1.20      # PM volume acceleration threshold
    t2_fallback_entry_bar: int = 6           # bar index for fallback entry (10:00 = bar 6)
    t2_max_entries_per_symbol: int = 1       # max entries per symbol per day (excl. PM re-entry)
    t2_pm_reentry: bool = True               # allow afternoon re-entry for stopped-out symbols

    # T2 v2: Adaptive stop
    t2_initial_atr_mult: float = 1.5         # Phase 1 stop width (ATR units)
    t2_breakeven_r: float = 0.5              # MFE to move stop to breakeven (Phase 2)
    t2_profit_trail_atr: float = 1.0         # Phase 3 trailing distance (ATR)
    t2_tight_trail_atr: float = 0.5          # Phase 4 tight trailing (ATR)
    t2_tight_trail_r: float = 2.0            # MFE to engage Phase 4

    # T2 v2: Staleness & partial
    t2_staleness_hours: float = 5.0          # hours before staleness timeout (daily-ATR scale)
    t2_staleness_min_r: float = 0.1          # min MFE R to avoid staleness exit
    t2_partial_r: float = 1.5               # R-mult for partial take
    t2_partial_frac: float = 0.33            # fraction to exit on partial

    # T2 v2: EOD carry scoring
    t2_carry_threshold: float = 65.0         # min carry score for full carry
    t2_carry_partial_threshold: float = 45.0 # min carry score for partial carry

    # T2 v2: Day-of-week sizing
    t2_tuesday_mult: float = 0.50           # sizing multiplier for Tuesdays
    t2_friday_mult: float = 1.25            # sizing multiplier for Fridays

    # T2 v2: Structural improvements (Phase 5)
    t2_open_entry: bool = False              # enter at bar 1 (near open), like T1
    t2_open_entry_stop_atr: float = 1.5      # stop for open entries (daily ATR mult)
    t2_open_entry_size_mult: float = 1.0     # sizing multiplier for open entries
    t2_default_carry_profitable: bool = False # carry all profitable positions by default
    t2_default_carry_min_r: float = 0.0      # min unrealized R to default-carry
    t2_default_carry_stop_atr: float = 1.0   # protective overnight stop (ATR mult)
    t2_entry_strength_gate: float = 0.0      # min entry strength score (0 = no gate)
    t2_entry_strength_sizing: bool = False   # multiply sizing by entry_strength
    t2_orb_window_bars: int = 1              # ORB trigger window (1 = bar 6 only)
    t2_orb_require_bullish: bool = True      # require close > open on ORB bar
    t2_vwap_pullback_multibar: bool = False  # multi-bar VWAP pullback detection
    t2_staleness_action: str = "CLOSE"       # CLOSE | TIGHTEN | SKIP
    t2_staleness_tighten_atr: float = 0.5    # tighten distance when action=TIGHTEN
    t2_avwap_breakdown_action: str = "CLOSE" # CLOSE | TIGHTEN
    t2_regime_b_sizing_mult: float = 1.0     # sizing multiplier for Tier B entries
    t2_breakeven_use_closes: bool = False     # use highest_close instead of mfe_r for BE
    t2_fallback_require_above_vwap: bool = False  # VWAP gate for fallback entries
    t2_fallback_momentum_bars: int = 0       # consecutive green bars required (0 = none)

    # Structural alpha amplification (defaults = current behavior)
    carry_only_entry: bool = True            # only enter if carry conditions possible
    strong_only_entry: bool = False          # only enter STRONG sponsorship
    entry_order_by_conviction: bool = False  # sort tradable by conviction desc before entering
    use_close_stop: bool = True              # exit at close if underwater (replaces hard stop)
    intraday_flow_check: bool = False        # check flow reversal on same-day positions
    regime_b_carry_mult: float = 0.6         # >0 enables Regime B carry at this size mult
    carry_top_quartile: bool = False         # require close in top 25% of daily range to carry

    # T1 entry flow gate: require positive flow proxy at entry (default OFF = current behavior)
    t1_entry_flow_gate: bool = True          # gate entries on positive flow proxy
    t1_entry_flow_lookback: int = 1          # number of recent flow values to check

    # T1 intraday partial takes (default OFF = current behavior)
    t1_partial_takes: bool = False           # enable intraday partial profit-taking
    t1_partial_r_trigger: float = 0.5        # R-multiple at which to take partial
    t1_partial_fraction: float = 0.25        # fraction of position to exit

    # T1 gap-down entry filter (default 0 = no filter = current behavior)
    t1_gap_down_skip_pct: float = 0.0        # skip entry if gap-down exceeds this (0 = disabled)

    # Carry close-pct minimum (default 0.75 = current hardcoded value)
    carry_close_pct_min: float = 0.75        # min close-in-range pct for carry eligibility

    # Carry trailing stop (default 0 = disabled = current behavior)
    carry_trail_activate_days: int = 0       # days before trailing activates (0 = disabled)
    carry_trail_atr_mult: float = 1.5        # trail distance in ATR (0 = breakeven)

    # T1 day-of-week sizing (default 1.0 = no adjustment = current behavior)
    dow_tuesday_mult: float = 1.0            # sizing multiplier for Tuesdays
    dow_friday_mult: float = 1.0             # sizing multiplier for Fridays

    # Hybrid daily-alpha mode: disable intraday exits, use daily-level decisions only
    t2_daily_alpha_mode: bool = False       # skip TIME_STOP, AVWAP_BD, PARTIAL intraday
    t2_bar_minutes: int = 5                 # bar granularity: 5 (default) or 30

    # ── V3 Enhancement Flags (all OFF = T1 FSM behavior + EOD flatten) ──

    # Overnight carry (combines FSM entries with carry alpha)
    v3_carry_enabled: bool = False               # Enable overnight carry
    v3_carry_score_fallback: bool = False         # Use carry score when binary check fails
    v3_carry_score_threshold: float = 65.0        # Min score for fallback carry

    # Adaptive trailing stop (above structural stop only, no breakeven trap)
    v3_adaptive_trail: bool = False               # Enable trailing
    v3_trail_activation_r: float = 1.0            # R-multiple to start trailing
    v3_trail_atr_mult: float = 1.5               # Trail distance in session ATR

    # Entry price improvement window (after FSM fires READY_TO_ENTER)
    v3_entry_improvement: bool = False            # Enable improvement window
    v3_improvement_window_bars: int = 3           # Bars to wait for better price
    v3_improvement_discount_pct: float = 0.003    # Min improvement to take

    # PM re-entry for stopped-out FSM setups
    v3_pm_reentry: bool = False
    v3_pm_reentry_after_bar: int = 48             # 13:30 ET

    # Staleness tighten (tighten stop, don't exit)
    v3_staleness_tighten: bool = False
    v3_staleness_hours: float = 5.0
    v3_staleness_tighten_atr: float = 0.5

    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "cache"
    )
    blacklist_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "blacklist.txt"
    )
    diagnostics_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "diagnostics"
    )
    research_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "research"
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "artifacts"
    )
    state_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "state"
    )

    @property
    def timing_sizing(self) -> tuple[tuple[time, time, float], ...]:
        return (
            (time(9, 35), time(10, 30), 1.00),
            (time(10, 30), time(12, 0), 0.85),
            (time(12, 0), time(13, 30), 0.70),
            (time(13, 30), time(14, 30), 0.90),
            (time(14, 30), time(15, 0), 0.75),
        )


def build_proxy_instruments() -> list[Instrument]:
    primary = {"SPY": "ARCA", "QQQ": "NASDAQ", "IWM": "ARCA"}
    return [
        Instrument(
            symbol=symbol,
            root=symbol,
            venue="SMART",
            primary_exchange=primary[symbol],
            sec_type="STK",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            point_value=1.0,
            currency="USD",
        )
        for symbol in PROXY_SYMBOLS
    ]
