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

    base_risk_fraction: float = 0.0025
    daily_stop_r: float = 2.0
    heat_cap_r: float = 1.25
    portfolio_daily_stop_r: float = 3.0
    sector_risk_cap_pct: float = 0.35
    max_positions_tier_a: int = 8
    max_positions_tier_b: int = 4
    max_positions_per_sector: int = 3
    minimum_remaining_size_pct: float = 0.30

    confidence_green_mult: float = 1.0
    confidence_yellow_mult: float = 0.65
    stale_penalty: float = 0.85

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
