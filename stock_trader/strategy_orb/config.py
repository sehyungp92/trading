"""Static configuration for the U.S. ORB strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from shared.oms.models.instrument import Instrument

ET = ZoneInfo("America/New_York")
STRATEGY_ID = "US_ORB_v1"
PROXY_SYMBOLS = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class ScannerSettings:
    scan_codes: tuple[str, ...] = ("TOP_PERC_GAIN", "HOT_BY_VOLUME")
    instrument: str = "STK"
    location_code: str = "STK.US.MAJOR"
    rows_per_scan: int = 40
    above_price: float = 5.0
    above_volume: int = 250_000
    stock_type_filter: str = "CORP"


@dataclass(frozen=True)
class StrategySettings:
    opening_noise_end: time = time(9, 34, 59)
    scanner_start: time = time(9, 34, 45)
    or_start: time = time(9, 35)
    or_end: time = time(9, 50)
    entry_start: time = time(9, 51)
    entry_end: time = time(11, 15)
    forced_flatten: time = time(15, 45)

    min_price: float = 5.0
    core_adv_threshold: float = 20_000_000.0
    secondary_adv_threshold: float = 10_000_000.0
    surge_threshold: float = 3.0
    rvol_threshold: float = 2.2
    spread_limit_pct: float = 0.0035
    tighter_gap_spread_pct: float = 0.0030
    leader_breadth_min: int = 9

    min_or_pct: float = 0.009
    max_or_pct: float = 0.0425
    acceptance_timeout_s: int = 300
    entry_ttl_s: int = 25
    max_rearms: int = 1

    base_risk_pct: float = 0.0035
    halt_new_entries_pnl_pct: float = -0.02
    flatten_all_pnl_pct: float = -0.03
    max_positions: int = 4
    max_positions_per_sector: int = 1
    max_last5m_value_pct: float = 0.04
    max_notional_nav_pct: float = 0.125
    max_displayed_ask_pct: float = 0.10
    secondary_size_penalty: float = 0.75
    caution_size_penalty: float = 0.70

    partial_r_multiple: float = 1.0
    partial_exit_fraction: float = 0.33
    scratch_after_minutes: int = 8
    scratch_min_r: float = 0.40

    flow_alpha: float = 0.35
    strong_inflow_mult: float = 1.10
    mixed_flow_mult: float = 1.00
    outflow_mult: float = 0.82

    scanner_cap: int = 20
    heat_cap_r: float = 4.0
    caution_quality_min: float = 80.0
    minimum_quality_score: float = 50.0
    top_quality_score: float = 85.0
    caution_score_floor: float = 70.0
    top_quality_multiplier: float = 1.25

    blocked_spread_pct: float = 0.0060
    danger_spread_pct: float = 0.0045
    caution_spread_pct: float = 0.0035
    blocked_range_atr: float = 2.0
    caution_range_atr: float = 1.4
    entry_range_atr: float = 1.6
    blocked_range_spread_pct: float = 0.0035
    quote_gap_abnormal_pct: float = 0.0040
    price_extension_vwap_danger_atr: float = 2.2
    price_extension_vwap_entry_atr: float = 2.0
    price_extension_or_danger_atr: float = 1.5
    caution_trail_floor: float = 0.76
    spread_trail_floor_pct: float = 0.0040
    midpoint_velocity_threshold_pct_per_s: float = 0.0010

    past_limit_window_s: int = 300
    direct_flag_window_s: int = 180
    recent_halt_window_s: int = 600
    recent_past_limit_block_count: int = 2
    halt_cooldown_s: int = 600
    ask_past_high_cooldown_s: int = 300
    danger_cooldown_s: int = 180

    warmup_start: time = time(9, 15)

    diagnostics_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_orb" / "diagnostics"
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_orb" / "cache"
    )
    scanner: ScannerSettings = field(default_factory=ScannerSettings)

    @property
    def daily_stop_r(self) -> float:
        if self.base_risk_pct <= 0:
            return 0.0
        return abs(self.halt_new_entries_pnl_pct) / self.base_risk_pct

    @property
    def portfolio_daily_stop_r(self) -> float:
        return self.daily_stop_r


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
