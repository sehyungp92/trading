"""Static configuration for the ALCB v1 strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from shared.oms.models.instrument import Instrument

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

STRATEGY_ID = "ALCB_v1"
STRATEGY_TYPE = "strategy_alcb"
PROXY_SYMBOLS = ("SPY", "QQQ", "IWM")

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Semiconductors": "SMH",
}


@dataclass(frozen=True)
class ScannerSettings:
    scan_codes: tuple[str, ...] = ("TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME")
    instrument: str = "STK"
    location_code: str = "STK.US.MAJOR"
    rows_per_scan: int = 75
    above_price: float = 15.0
    above_volume: int = 250_000
    stock_type_filter: str = "CORP"


@dataclass(frozen=True)
class StrategySettings:
    premarket_start: time = time(7, 0)
    post_close_scan: time = time(16, 10)
    market_open: time = time(9, 30)
    first_30m_close: time = time(10, 0)
    entry_end: time = time(15, 30)
    close_block_start: time = time(15, 50)
    forced_flatten: time = time(15, 58)

    hot_max: int = 40
    active_monitoring_target: int = 35
    warm_poll_interval_s: int = 30
    cold_poll_interval_s: int = 120

    min_price: float = 15.0
    min_adv_usd: float = 20_000_000.0
    max_median_spread_pct: float = 0.0050
    max_friction_to_risk: float = 0.10
    earnings_block_days: int = 3
    allow_adrs: bool = False
    allow_biotech: bool = False

    atr_ratio_low: float = 0.75
    atr_ratio_high: float = 1.20
    box_length_low: int = 8
    box_length_mid: int = 12
    box_length_high: int = 18
    hysteresis_days: int = 3
    min_containment: float = 0.80
    max_squeeze_metric: float = 1.10
    squeeze_lookback: int = 60
    sq_good_quantile: float = 0.30
    sq_loose_quantile: float = 0.65

    base_q_disp: float = 0.70
    atr_expansion_q_disp_adj: float = -0.05
    breakout_reject_range_atr: float = 2.0
    breakout_reject_min_body_ratio: float = 0.25
    breakout_reject_max_wick_ratio: float = 0.55
    breakout_reject_min_rvol_d: float = 2.0
    # Opt 4: Breakout Volume Confirmation
    breakout_min_rvol_d: float = 1.0
    breakout_low_vol_disp_premium: float = 0.15

    # Opt 5: Enhanced Accumulation/Distribution Scoring
    accum_vol_trend_bonus: float = 0.15

    # Opt 6: Time-of-Day Entry Quality Adjustment
    early_entry_end: time = time(11, 30)
    late_entry_start: time = time(14, 0)
    early_entry_rvol_bonus_min: float = 1.5

    normal_intraday_score_min: int = 2
    degraded_intraday_score_min: int = 3
    intraday_rvol_min: float = 1.2
    # Opt 2: Graduated RVOL + Displacement Bonus
    intraday_rvol_strong: float = 2.0
    evidence_disp_bonus_threshold: float = 1.30

    base_risk_fraction: float = 0.0050
    volatile_base_risk_fraction: float = 0.0035
    daily_stop_r: float = 2.0
    heat_cap_r: float = 6.0
    portfolio_daily_stop_r: float = 5.0
    max_portfolio_heat_fraction: float = 0.03
    final_risk_min_mult: float = 0.20
    final_risk_max_mult: float = 1.00
    atr_stop_mult_std: float = 1.0
    atr_stop_mult_volatile: float = 1.2
    max_participation_30m: float = 0.01
    thin_participation_30m: float = 0.005

    max_positions: int = 5
    max_positions_per_sector: int = 2
    max_adds: int = 2
    max_campaign_risk_mult: float = 1.5
    corr_threshold: float = 0.70
    corr_lookback: int = 60

    # Opt 3: Scaled quality_mult + Quality-Scaled TP2
    evidence_score_top_tier: int = 6
    quality_mult_top_score: float = 1.10
    high_conviction_quality_mult_min: float = 1.0

    tp1_aligned_r: float = 1.25
    tp2_aligned_r: float = 2.5
    tp2_aligned_r_high_conviction: float = 3.0
    tp1_neutral_r: float = 1.0
    tp2_neutral_r: float = 2.0
    tp2_neutral_r_high_conviction: float = 2.5
    tp1_fraction: float = 0.30
    tp2_fraction: float = 0.30
    runner_fraction: float = 0.40
    breakeven_buffer_atr: float = 0.10

    # Opt 1: Adaptive Runner Trailing Stop
    runner_structure_atr_base: float = 1.0
    runner_structure_atr_strong: float = 1.5
    runner_adx_strong_threshold: float = 30.0
    runner_atr_expansion_threshold: float = 1.15
    runner_profit_ratchet_r_base: float = 2.0
    runner_profit_ratchet_r_strong: float = 3.0

    stale_warn_days: int = 8
    stale_exit_days: int = 10
    stale_exit_runner_r_threshold: float = 0.0
    continuation_box_mult: float = 1.5
    continuation_r_mult: float = 2.0

    dirty_break_fail_days: int = 3
    dirty_reset_days: int = 5

    selection_long_count: int = 20
    selection_short_count: int = 20
    universe_cap: int = 650

    diagnostics_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "diagnostics"
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "cache"
    )
    research_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "research"
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "artifacts"
    )
    state_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "state"
    )
    blacklist_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "blacklist.txt"
    )
    scanner: ScannerSettings = field(default_factory=ScannerSettings)


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
