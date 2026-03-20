"""Multi-Asset Swing Breakout v3.3-ETF — constants and per-symbol configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from shared.ibkr_core.config.schemas import ContractTemplate, ExchangeRoute
from shared.oms.models.instrument import Instrument
from shared.oms.models.instrument_registry import InstrumentRegistry

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "SWING_BREAKOUT_V3"

# ---------------------------------------------------------------------------
# Tuning flags — set any to False to revert that category to baseline
# ---------------------------------------------------------------------------
TUNE_COMPRESSION = True       # Cat 1: Relax SQUEEZE_CEIL, CONTAINMENT_MIN
TUNE_DISPLACEMENT = True      # Cat 1: Lower Q_DISP, raise ATR_EXPAND_ADJ
TUNE_SCORE = True             # Cat 1: Lower SCORE_THRESHOLD_NORMAL
TUNE_ENTRY_UNLOCK = True      # Cat 2: Relax Entry A/B gates, allow neutral regime
TUNE_TP_TARGETS = False       # Cat 3: Reverted — baseline TPs now achievable with tighter stops (A5)
TUNE_REENTRY = True           # Cat 4: Relax re-entry cooldown, DIRTY gates
TUNE_CONTINUATION = False     # Cat 5: Reverted — blocks Entry A/B by entering continuation too early
TUNE_PORTFOLIO = True         # Cat 6: Widen portfolio heat, pending, hard block
TUNE_REGIME_MULT = False      # Cat 6: Reverted — marginal sizing-only, risks larger caution losses
TUNE_STALE = False            # Cat 7: Reverted — hurts 3/4 symbols (only helps GLD)

# ---------------------------------------------------------------------------
# ETF Symbols
# ---------------------------------------------------------------------------
SYMBOLS = ["QQQ", "GLD"]

# ---------------------------------------------------------------------------
# Per-symbol configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    # Instrument classification
    is_etf: bool = True
    sec_type: str = "STK"
    tick_size: float = 0.01
    multiplier: float = 1.0
    exchange: str = "SMART"
    trading_class: str = ""
    contract_expiry: str = ""
    # Entry windows (ET) - RTH only for ETFs
    entry_window_start_et: str = "09:30"
    entry_window_end_et: str = "16:00"
    # Spread gate (spec §7.1)
    max_spread_dollars: float = 0.02
    max_spread_bps: float = 5.0
    # Min stop floor
    min_stop_floor_dollars: float = 0.10
    # Slippage guard (spec §15)
    slip_max_dollars: float = 0.05
    slip_max_bps: float = 5.0
    # Position limits
    max_shares: int = 1000
    # ATR stop multiplier (spec §16.1)
    atr_stop_mult: float = 1.0
    # Base risk percentage (spec §14.1)
    base_risk_pct: float = 0.005
    # DIRTY trigger window M_BREAK (spec §10.1)
    m_break: int = 2
    # Fee bps estimate for friction (spec §15.1)
    fee_bps_est: float = 0.0003  # 3 bps
    # TP target scale factor (spec §20.2) — higher for volatile symbols
    tp_scale: float = 1.0
    # Allowed trade directions — filter negative-EV sides per symbol
    allowed_directions: tuple[str, ...] = ("LONG", "SHORT")
    # Experiment A/B tracking
    experiment_id: str = ""
    experiment_variant: str = ""


# ---------------------------------------------------------------------------
# ETF configs (spec §1.1)
# ---------------------------------------------------------------------------

_ETF_CONFIGS: dict[str, SymbolConfig] = {
    "QQQ": SymbolConfig(
        symbol="QQQ",
        tick_size=0.01,
        multiplier=1.0,
        exchange="SMART",
        max_spread_dollars=0.02,
        max_spread_bps=2.0,
        min_stop_floor_dollars=0.10,
        max_shares=500,
        atr_stop_mult=0.5,   # A1: halved from 1.0 to tighten R-unit
        base_risk_pct=0.0075,  # was 0.60% — proven winner, allocate more
        m_break=2,
        fee_bps_est=0.0002,  # 2 bps
        tp_scale=1.0,
    ),
    "USO": SymbolConfig(
        symbol="USO",
        tick_size=0.01,
        multiplier=1.0,
        exchange="SMART",
        max_spread_dollars=0.05,
        max_spread_bps=5.0,
        min_stop_floor_dollars=0.10,
        max_shares=1000,
        atr_stop_mult=0.65,  # A1: halved from 1.3 to tighten R-unit
        base_risk_pct=0.0030,
        m_break=3,
        fee_bps_est=0.0006,  # 6 bps
        allowed_directions=("LONG",),
    ),
    "GLD": SymbolConfig(
        symbol="GLD",
        tick_size=0.01,
        multiplier=1.0,
        exchange="SMART",
        max_spread_dollars=0.02,
        max_spread_bps=2.0,
        min_stop_floor_dollars=0.10,
        max_shares=500,
        atr_stop_mult=0.5,   # A1: halved from 1.0 to tighten R-unit
        base_risk_pct=0.0065,
        m_break=2,
        fee_bps_est=0.0002,
        allowed_directions=("LONG",),
    ),
    "IBIT": SymbolConfig(
        symbol="IBIT",
        tick_size=0.01,
        multiplier=1.0,
        exchange="SMART",
        max_spread_dollars=0.10,
        max_spread_bps=10.0,
        min_stop_floor_dollars=0.10,
        max_shares=1000,
        atr_stop_mult=0.7,   # A1: halved from 1.3 to tighten R-unit
        base_risk_pct=0.0035,
        m_break=3,
        fee_bps_est=0.0008,  # 8 bps
        tp_scale=1.0,
    ),
}

SYMBOL_CONFIGS = {s: _ETF_CONFIGS[s] for s in SYMBOLS if s in _ETF_CONFIGS}

# ---------------------------------------------------------------------------
# ATR periods (spec §3)
# ---------------------------------------------------------------------------
ATR_DAILY_PERIOD: int = 14
ATR_DAILY_LONG_PERIOD: int = 50
ATR_HOURLY_PERIOD: int = 14

# ---------------------------------------------------------------------------
# EMA periods (spec §7)
# ---------------------------------------------------------------------------
EMA_4H_PERIOD: int = 50
EMA_DAILY_PERIOD: int = 50
EMA_1H_PERIOD: int = 20

# ---------------------------------------------------------------------------
# ADX period (spec §7.1)
# ---------------------------------------------------------------------------
ADX_PERIOD: int = 14

# ---------------------------------------------------------------------------
# Compression detection (spec §5)
# ---------------------------------------------------------------------------
SQUEEZE_CEIL: float = 1.25 if TUNE_COMPRESSION else 1.10            # baseline: 1.10
SQUEEZE_CEIL_ADAPTIVE: bool = True
SQUEEZE_CEIL_PCTL: float = 0.75     # 75th percentile of rolling squeeze history
SQUEEZE_CEIL_FALLBACK: float = 5.00  # cap when history < 20 bars
LOOKBACK_SQ: int = 60  # days for squeeze quantiles
CONTAINMENT_MIN: float = 0.70 if TUNE_COMPRESSION else 0.80         # baseline: 0.80

# Adaptive L buckets (spec §5.1)
ATR_RATIO_LOW: float = 0.70
ATR_RATIO_HIGH: float = 1.25
L_LOW: int = 8
L_MID: int = 12
L_HIGH: int = 18
HYSTERESIS_BARS: int = 3

# ---------------------------------------------------------------------------
# Displacement (spec §8.2)
# ---------------------------------------------------------------------------
Q_DISP: float = 0.60 if TUNE_DISPLACEMENT else 0.70                 # baseline: 0.70
Q_DISP_ATR_EXPAND_ADJ: float = 0.10 if TUNE_DISPLACEMENT else 0.05  # baseline: 0.05
DISP_STRONG_MULT: float = 1.15  # low-volume day exception
DISP_LOOKBACK_MAX: int = 750

# ---------------------------------------------------------------------------
# RVOL (spec §3)
# ---------------------------------------------------------------------------
LOOKBACK_RVOL_D: int = 20
LOOKBACK_SLOT_WEEKS: int = 12

# ---------------------------------------------------------------------------
# Entry parameters (spec §12)
# ---------------------------------------------------------------------------
ENTRYB_RVOLH_MIN: float = 0.6 if TUNE_ENTRY_UNLOCK else 0.8              # baseline: 0.8
ENTRYB_DISPMULT_OVERRIDE: float = 0.80 if TUNE_ENTRY_UNLOCK else 0.90    # baseline: 0.90
ENTRY_A_TOUCH_TOL_ATR_H: float = 0.50                                    # near-touch tolerance for Entry A
RECLAIM_BUFFER_ATR_H_MULT: float = 0.05                                  # was 0.08 — relaxed to enable Entry A
RECLAIM_BUFFER_ATR_D_MULT: float = 0.01                                   # was 0.02 — relaxed to enable Entry A
SWEEP_DEPTH_ATR_D_MULT: float = 0.06                                     # was 0.15 — achievable on volatile hourly bars
ENTRY_A_TTL_RTH_HOURS: int = 8 if TUNE_ENTRY_UNLOCK else 6               # baseline: 6
ENTRY_C_TTL_RTH_HOURS: int = 4
C_ENTRY_MAX_DISP_H: float = 3.0                                         # reject C entries > 3.0 ATR_D from AVWAP
# Entry B neutral regime permission (Cat 2)
ENTRYB_REQUIRE_ALIGNED: bool = not TUNE_ENTRY_UNLOCK                     # baseline: True
ENTRYB_NEUTRAL_QUALITY_MIN: float = 0.60 if TUNE_ENTRY_UNLOCK else 0.70  # baseline: 0.70

# ---------------------------------------------------------------------------
# Adds (spec §13)
# ---------------------------------------------------------------------------
MAX_ADDS_PER_CAMPAIGN: int = 2
CAMPAIGN_RISK_BUDGET_MULT: float = 1.5
ADD_RISK_MULT: float = 0.5
ADD_RISK_LOWVOLH_MULT: float = 0.7
ADD_CLV_STRONG_Q: float = 0.30
ADD_STOP_ATR_MULT: float = 0.5

# ---------------------------------------------------------------------------
# Sizing (spec §14)
# ---------------------------------------------------------------------------
RISK_REGIME_MIN: float = 0.75
RISK_REGIME_MAX: float = 1.05
QUALITY_MULT_MIN: float = 0.25
QUALITY_MULT_MAX: float = 1.0
EXPIRY_MULT_FLOOR: float = 0.30
EXPIRY_DECAY_STEP: float = 0.12

# Regime multipliers (spec §14.3)
REGIME_MULT_ALIGNED: float = 1.00
REGIME_MULT_NEUTRAL: float = 0.90 if TUNE_REGIME_MULT else 0.85         # baseline: 0.85
REGIME_MULT_CAUTION: float = 0.50 if TUNE_REGIME_MULT else 0.40         # baseline: 0.40

# Squeeze multipliers (spec §14.3)
SQUEEZE_MULT_GOOD: float = 1.05
SQUEEZE_MULT_NEUTRAL: float = 1.00
SQUEEZE_MULT_LOOSE: float = 0.85

# Correlation multiplier (spec §18)
CORR_MULT_ALIGNED: float = 0.85
CORR_MULT_OTHER: float = 0.70
CORR_HEAT_PENALTY: float = 1.25

# ---------------------------------------------------------------------------
# Friction (spec §15)
# ---------------------------------------------------------------------------
FRICTION_CAP: float = 0.15  # was 0.10 — allow friction up to 15% of risk$

# ---------------------------------------------------------------------------
# Expiry (spec §11.2)
# ---------------------------------------------------------------------------
EXPIRY_BARS_MIN: int = 3
EXPIRY_BARS_MAX: int = 10
HARD_EXPIRY_BARS_ADD: int = 5

# ---------------------------------------------------------------------------
# Continuation mode (spec §11.3)
# ---------------------------------------------------------------------------
MM_BOX_HEIGHT_MULT: float = 1.5
R_PROXY_CONT_THRESHOLD: float = 1.5 if TUNE_CONTINUATION else 2.0       # baseline: 2.0
R_PROXY_TIME_THRESHOLD: float = 1.0 if TUNE_CONTINUATION else 1.5       # baseline: 1.5
CONT_BARS_MIN: int = 3 if TUNE_CONTINUATION else 5                      # baseline: 5

# ---------------------------------------------------------------------------
# DIRTY handling (spec §10)
# ---------------------------------------------------------------------------
DIRTY_OPPOSITE_MULT: float = 1.10
DIRTY_BOX_SHIFT_ATR_MULT: float = 0.25 if TUNE_REENTRY else 0.35        # baseline: 0.35
DIRTY_DURATION_L_FRAC: float = 0.35 if TUNE_REENTRY else 0.5            # baseline: 0.5

# ---------------------------------------------------------------------------
# Chop mode (spec §7.2)
# ---------------------------------------------------------------------------
CHOP_ATR_PCTL_HIGH: float = 0.75
CHOP_CROSS_LOOKBACK: int = 10
CHOP_CROSS_THRESHOLD: int = 4
CHOP_DEGRADED_SIZE_MULT: float = 0.75
CHOP_DEGRADED_STALE_ADJ: int = -2

# ---------------------------------------------------------------------------
# Score thresholds (spec §9)
# ---------------------------------------------------------------------------
SCORE_THRESHOLD_NORMAL: int = 1 if TUNE_SCORE else 2                    # baseline: 2
SCORE_THRESHOLD_DEGRADED: int = 2
SCORE_THRESHOLD_RANGE: int = 2
REGIME_CHOP_BLOCK: bool = True
REGIME_CHOP_SCORE_OVERRIDE: int = 3   # allow very high-conviction through

# Hard block slope threshold (spec §7.3)
HARD_BLOCK_SLOPE_MULT: float = 0.12 if TUNE_PORTFOLIO else 0.08         # baseline: 0.08

# ---------------------------------------------------------------------------
# Breakout invalidation (spec §10)
# ---------------------------------------------------------------------------
INSIDE_CLOSE_INVALIDATION_COUNT: int = 2

# ---------------------------------------------------------------------------
# Gap handling (spec §17)
# ---------------------------------------------------------------------------
GAP_GUARD_ATR_MULT: float = 1.5

# ---------------------------------------------------------------------------
# Pending mechanism (spec §19)
# ---------------------------------------------------------------------------
PENDING_MAX_RTH_HOURS: int = 18 if TUNE_PORTFOLIO else 12               # baseline: 12

# ---------------------------------------------------------------------------
# Exit tiers (spec §20)
# ---------------------------------------------------------------------------
# Aligned — calibrated to actual MFE (mean ~0.30R)
TP1_R_ALIGNED: float = 0.20                                               # was 1.5 — hit by majority of trades
TP2_R_ALIGNED: float = 0.50                                               # was 3.0 — reached by best trades
# Neutral
TP1_R_NEUTRAL: float = 0.15                                               # was 0.75
TP2_R_NEUTRAL: float = 0.35                                               # was 2.0
# Caution
TP1_R_CAUTION: float = 0.10                                               # was 0.50
TP2_R_CAUTION: float = 0.25                                               # was 1.6

BE_BUFFER_ATR_MULT: float = 0.1

# Stale exit (spec §20.4)
STALE_WARN_DAYS: int = 6
STALE_EXIT_DAYS_MIN: int = 10
STALE_EXIT_DAYS_MAX: int = 15
STALE_R_THRESH: float = 0.0
STALE_TIGHTEN_MULT: float = 0.8

# Trailing (spec §20.3)
TRAIL_4H_ATR_MULT: float = 0.40
EMA_4H_FLOOR_ATR_MULT: float = 0.5

# ---------------------------------------------------------------------------
# Re-entry (spec §21)
# ---------------------------------------------------------------------------
MAX_REENTRY_PER_BOX_VERSION: int = 2 if TUNE_REENTRY else 1             # baseline: 1
REENTRY_COOLDOWN_DAYS: int = 1 if TUNE_REENTRY else 3                   # baseline: 3
REENTRY_MIN_REALIZED_R: float = -1.5 if TUNE_REENTRY else -0.75         # baseline: -0.75

# ---------------------------------------------------------------------------
# Portfolio risk controls (spec §22)
# ---------------------------------------------------------------------------
MAX_POSITIONS: int = 4
MAX_SAME_DIRECTION: int = 2
MAX_PORTFOLIO_HEAT: float = 0.030 if TUNE_PORTFOLIO else 0.025          # baseline: 2.5%
WEEKLY_THROTTLE_R: float = -5.0
WEEKLY_THROTTLE_MULT: float = 0.5
MONTHLY_HALT_R: float = -8.0
MONTHLY_HALT_DAYS: int = 20

# ---------------------------------------------------------------------------
# Correlation (spec §18)
# ---------------------------------------------------------------------------
CORR_THRESHOLD: float = 0.70
CORR_LOOKBACK_BARS: int = 60

# ---------------------------------------------------------------------------
# News windows (spec §4)
# ---------------------------------------------------------------------------
NEWS_WINDOWS: dict[str, tuple[int, int]] = {
    "CPI": (-60, 30),
    "NFP": (-60, 30),
    "FOMC": (-60, 60),
    "FED_SPEECH": (-30, 30),
    "CL_INVENTORY": (-20, 20),
    "ECB": (-60, 60),
    "BOJ": (-60, 60),
    "CRYPTO_EVENT": (-30, 30),
}

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_instruments() -> dict[str, Instrument]:
    """Create Instrument objects for every symbol and register them."""
    instruments: dict[str, Instrument] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
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


def build_contract_templates() -> dict[str, ContractTemplate]:
    """Build ContractTemplate dict matching config/contracts.yaml."""
    templates: dict[str, ContractTemplate] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        templates[sym] = ContractTemplate(
            symbol=sym,
            sec_type=cfg.sec_type,
            exchange=cfg.exchange,
            currency="USD",
            multiplier=cfg.multiplier,
            tick_size=cfg.tick_size,
            tick_value=cfg.tick_size * cfg.multiplier,
            trading_class=cfg.trading_class or None,
            primary_exchange=cfg.exchange if cfg.is_etf else None,
        )
    return templates


def build_exchange_routes() -> dict[str, ExchangeRoute]:
    """Build ExchangeRoute dict matching config/routing.yaml."""
    routes: dict[str, ExchangeRoute] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        routes[sym] = ExchangeRoute(
            root_symbol=sym,
            exchange=cfg.exchange,
            trading_class=cfg.trading_class or None,
            primary_exchange=cfg.exchange if cfg.is_etf else None,
        )
    return routes
