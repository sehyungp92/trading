"""Downturn Dominator v1 -- R7c production configuration.

Exports consumed by the coordinator and engine:
  - STRATEGY_ID, BASE_RISK_PCT, DAILY_STOP_R, HEAT_CAP_R, PORTFOLIO_DAILY_STOP_R
  - MAX_LEVERAGE_MULT (per-strategy leverage ceiling for engine)
  - build_instruments() -> dict[str, Instrument]
  - R7C_FLAGS: DownturnAblationFlags   (backtest dataclass, R7c values)
  - R7C_PARAM_OVERRIDES: dict[str, float]

Authoritative source: R7C_MUTATIONS in
  research/backtests/momentum/auto/downturn/output/generate_r7b_diagnostics.py
"""
from __future__ import annotations

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry
from backtests.momentum.config_downturn import DownturnAblationFlags

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "DownturnDominator_v1"

# ---------------------------------------------------------------------------
# Contract specifications (MNQ primary, NQ for reference)
# ---------------------------------------------------------------------------
NQ_SPECS = {
    "NQ":  {"tick": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick": 0.25, "tick_value": 0.50, "point_value":  2.00},
}
DEFAULT_SYMBOL = "MNQ"

# ---------------------------------------------------------------------------
# Risk -- coordinator-level exports
# ---------------------------------------------------------------------------
BASE_RISK_PCT = 0.024          # R7c: 2.4% per trade
DAILY_STOP_R = 2.5             # halt after -2.5R daily realized
HEAT_CAP_R = 3.5               # max simultaneous heat
PORTFOLIO_DAILY_STOP_R = 1.5   # portfolio-level daily stop
MAX_LEVERAGE_MULT = 20.0       # R7c research-validated leverage ceiling for MNQ

# ---------------------------------------------------------------------------
# R7c ablation flags (16 changes from defaults)
# ---------------------------------------------------------------------------
R7C_FLAGS = DownturnAblationFlags(
    # --- Disable (default True -> False) ---
    breakdown_engine=False,         # 0 trades; never fires
    tiered_exits=False,             # 0% TP hit rate
    use_volatility_states=False,    # interferes with regime overrides
    # --- Enable (default False -> True) ---
    profit_floor_trail=True,
    adaptive_profit_floor=True,
    regime_adaptive_chandelier=True,
    min_hold_period=True,
    correction_regime_override=True,
    fast_crash_override=True,
    conviction_scoring=True,
    bear_structure_override=True,
    drawdown_regime_override=True,
    block_counter_regime=True,
    short_sma_trend=True,
    correction_sizing_bonus=True,
    momentum_signal=True,
)

# ---------------------------------------------------------------------------
# R7c parameter overrides (25 overrides)
# ---------------------------------------------------------------------------
R7C_PARAM_OVERRIDES: dict[str, float] = {
    # Profit floor
    "profit_floor_r_threshold": 1.35,
    "profit_floor_lock_pct": 0.60,
    # Adaptive profit floor
    "adaptive_lock_t1": 1.5,
    "adaptive_lock_bonus_1": 0.15,
    # Chandelier
    "chandelier_lookback": 8,
    "chandelier_mult_floor": 2.2,
    "chandelier_mult_ceiling": 4.5,
    # Breakeven
    "be_trigger_r": 0.5,
    "be_stop_buffer_mult": 0.10,
    # Min hold
    "min_hold_bars": 24,
    # Fast crash
    "crash_daily_threshold": -0.018,
    # Conviction
    "conviction_threshold": 35,
    # VWAP caps
    "vwap_cap_core": 0.55,
    "vwap_cap_extended": 0.72,
    # Divergence
    "divergence_mag_threshold": 0.1,
    # Indicators
    "ema_fast_period": 10,
    "sma200_period": 250,
    "short_sma_period": 40,
    "adx_trending_threshold": 30,
    "adx_range_threshold": 10,
    # Sizing
    "base_risk_pct": 0.024,
    "regime_mult_emerging": 1.8,
    "correction_sizing_mult": 1.2,
    # Circuit breaker
    "circuit_breaker_threshold": -2400,
    # Momentum
    "momentum_cooldown_bars": 36,
}


# ---------------------------------------------------------------------------
# Instrument builder (coordinator pattern)
# ---------------------------------------------------------------------------
def build_instruments() -> dict[str, Instrument]:
    """Register NQ / MNQ instruments with the OMS."""
    instruments: dict[str, Instrument] = {}
    for sym, spec in NQ_SPECS.items():
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue="CME",
            tick_size=spec["tick"],
            tick_value=spec["tick_value"],
            multiplier=spec["point_value"],
            contract_expiry="",
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
