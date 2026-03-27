"""Portfolio configuration for multi-strategy NQ system.

Defines two account tiers ($10K and $100K) with strategy allocations,
risk limits, and cross-strategy rules (Helix veto, NQDTC slope filter).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AccountTier(Enum):
    SMALL = "10K"
    STANDARD = "100K"


@dataclass(frozen=True)
class StrategyAllocation:
    """Per-strategy risk allocation within a portfolio."""
    strategy_id: str
    enabled: bool
    base_risk_pct: float          # e.g. 0.005 = 0.50%
    daily_stop_R: float
    max_concurrent: int = 1
    priority: int = 0             # lower = higher priority
    continuation_half_size: bool = False   # NQDTC slope filter
    continuation_size_mult: float = 0.50  # multiplier for continuation breakouts (when half_size=True)
    reversal_only: bool = False           # block continuation breakouts entirely
    helix_veto_enabled: bool = False
    helix_veto_window_minutes: int = 60


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio-level configuration for combined backtest."""
    tier: AccountTier
    initial_equity: float
    heat_cap_R: float
    portfolio_daily_stop_R: float
    max_total_positions: int
    directional_cap_R: float
    strategies: tuple[StrategyAllocation, ...]
    # MNQ instrument
    point_value: float = 2.0
    tick_size: float = 0.25
    tick_value: float = 0.50
    commission_per_side: float = 0.62
    # Drawdown tiers: (dd_pct_threshold, size_multiplier)
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.08, 1.00),
        (0.12, 0.50),
        (0.15, 0.25),
        (1.00, 0.00),
    )
    # Cross-strategy rules (v2 — from empirical combo analysis)
    helix_nqdtc_cooldown_minutes: int = 0         # bidirectional proximity cooldown (0=disabled)
    nqdtc_direction_filter_enabled: bool = False   # Vdubus sizing based on NQDTC direction
    nqdtc_agree_size_mult: float = 1.0            # Vdubus mult when NQDTC same direction
    nqdtc_oppose_size_mult: float = 1.0           # Vdubus mult when NQDTC opposing (0=block)
    cooldown_session_only: bool = False              # only enforce proximity during 09:45-11:30 ET overlap

    def get_strategy(self, strategy_id: str) -> StrategyAllocation | None:
        for s in self.strategies:
            if s.strategy_id == strategy_id:
                return s
        return None

    def priority_order(self) -> list[StrategyAllocation]:
        """Return strategies sorted by priority (lowest number first)."""
        return sorted(self.strategies, key=lambda s: s.priority)


def make_10k_config() -> PortfolioConfig:
    """$10K account: Helix + Vdubus + NQDTC reversal-only."""
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=2.5,
        portfolio_daily_stop_R=2.0,
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,      # 1.50% — same 1 MNQ, lower risk_R denominator
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=0,               # highest priority
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,      # 0.80% — min viable for 1 MNQ
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=2,
                reversal_only=True,       # only reversal breakouts (slope opposes)
                helix_veto_enabled=True,
                helix_veto_window_minutes=60,
            ),
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,      # 0.80%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=1,
            ),
        ),
    )


def make_100k_config() -> PortfolioConfig:
    """$100K account: All 3 strategies, NQDTC reversal-only."""
    return PortfolioConfig(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=3.0,
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.003,      # 0.30%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=0,               # highest priority
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.005,      # 0.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=1,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.0035,     # 0.35%
                daily_stop_R=3.0,         # raised from 2.5 — captures +16.3R winner
                max_concurrent=1,
                priority=2,               # lowest priority
                reversal_only=True,
                helix_veto_enabled=True,
                helix_veto_window_minutes=60,
            ),
        ),
    )


def make_10k_optimized_config() -> PortfolioConfig:
    """$10K optimized: empirical synergy rules from combo analysis.

    Backtest results: +55.2R / $+5,055 vs baseline +41.4R / $+3,590 (+33% R).

    Key changes vs baseline:
    - NQDTC: continuation_half_size instead of reversal_only — captures 27
      trades (vs 8 baseline), half-sizing continuations rather than blocking
      them entirely.  This provides directional signal on many more days.
    - Priority reorder: Vdubus > NQDTC > Helix (Vdubus most consistent,
      NQDTC sets directional signal, Helix enters latest)
    - Replace Helix veto with 240-min bidirectional proximity cooldown
      (proximity of any kind degrades both, not just opposing)
    - NQDTC direction filter: boost Vdubus 1.25x when NQDTC agrees,
      block Vdubus entirely when NQDTC opposes (100% vs 43% WR in overlap)
    - Raise portfolio daily stop 2.0R -> 2.5R (reduce cascade shutdowns)
    - Raise NQDTC daily stop 2.0R -> 2.5R (capture big trending-day winners)
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=2.5,
        portfolio_daily_stop_R=2.5,       # raised: reduce cascade shutdowns
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,      # 0.80%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=0,               # highest — most consistent performer
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,      # 0.80%
                daily_stop_R=2.5,         # raised: capture big trending winners
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,  # half-size continuations, don't block
                helix_veto_enabled=False,     # replaced by proximity cooldown
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,      # 1.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=2,               # lowest — enters later in day
            ),
        ),
        helix_nqdtc_cooldown_minutes=240,       # 4hr bidirectional cooldown
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.25,             # boost Vdubus when NQDTC agrees
        nqdtc_oppose_size_mult=0.0,             # block Vdubus when NQDTC opposes
    )


def make_100k_optimized_config() -> PortfolioConfig:
    """$100K optimized: empirical synergy rules from combo analysis.

    Backtest results: +64.2R / $+18,490 vs baseline +52.6R / $+14,311 (+22% R).
    Captures 91% of isolated R (vs 74% baseline).

    Same structural changes as $10K optimized but with $100K risk sizing.
    - NQDTC: continuation_half_size (27 trades vs 9 baseline)
    - Raise portfolio daily stop 3.0R -> 3.5R
    - Raise NQDTC daily stop to 3.0R
    """
    return PortfolioConfig(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=3.5,       # raised: reduce cascade shutdowns
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.003,      # 0.30%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=0,               # highest
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.0035,     # 0.35%
                daily_stop_R=3.0,         # captures +16.3R winner
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,  # half-size continuations, don't block
                helix_veto_enabled=False,     # replaced by proximity cooldown
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.005,      # 0.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=240,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.25,
        nqdtc_oppose_size_mult=0.0,
    )


def make_10k_v3_config() -> PortfolioConfig:
    """$10K v3: sweep-validated refinements over optimized config.

    Expected: +60.9R ($+5,519) vs optimized +55.2R ($+5,055) = +10%.

    Changes vs optimized (v2):
    - Heat cap: 2.5 -> 3.5R (headroom for combined improvements)
    - Vdubus daily_stop: 2.0 -> 2.5R (biggest single lever, +3.6R)
    - Continuation mult: 0.50 -> 0.70 (captures more upside, +0.9R)
    - Agree mult: 1.25 -> 1.50 (stronger directional synergy, +1.2R)
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=3.5,                          # raised from 2.5
        portfolio_daily_stop_R=2.5,
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,                 # raised from 2.0
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,      # raised from 0.50
                helix_veto_enabled=False,
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,              # 1.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=240,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,               # raised from 1.25
        nqdtc_oppose_size_mult=0.0,
    )


def make_100k_v3_config() -> PortfolioConfig:
    """$100K v3: sweep-validated refinements over optimized config.

    Expected: ~+67-68R ($+19,600) vs optimized +64.2R ($+18,490) = +5%.

    Changes vs optimized (v2):
    - Vdubus daily_stop: 2.0 -> 2.5R (+3.6R)
    - Continuation mult: 0.50 -> 0.70 (+2.0R)
    - Agree mult: 1.25 -> 1.50 (+1.1R)
    - Directional cap: 2.5 -> 3.0R (+0.8R)
    - Proximity cooldown: 240 -> 120 min (+2.1R)
    """
    return PortfolioConfig(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=3.5,
        max_total_positions=3,
        directional_cap_R=3.0,                    # raised from 2.5
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.003,              # 0.30%
                daily_stop_R=2.5,                 # raised from 2.0
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.0035,             # 0.35%
                daily_stop_R=3.0,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,      # raised from 0.50
                helix_veto_enabled=False,
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.005,              # 0.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,         # reduced from 240
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,               # raised from 1.25
        nqdtc_oppose_size_mult=0.0,
    )


def make_100k_v3_max_config() -> PortfolioConfig:
    """$100K v3 MAX: aggressive variant maximizing dollar PnL.

    Expected: +70.8R ($+25,907) — captures 100% of isolated R.

    Changes vs v3:
    - NQDTC risk: 0.35% -> 0.50% (biggest PnL lever, +$6,476)
    - Helix max_concurrent: 1 -> 2 (+1.2R)
    """
    return PortfolioConfig(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=3.5,
        max_total_positions=3,
        directional_cap_R=3.0,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.003,              # 0.30%
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.005,              # 0.50% — raised from 0.35%
                daily_stop_R=3.0,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,
                helix_veto_enabled=False,
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.005,              # 0.50%
                daily_stop_R=2.0,
                max_concurrent=2,                 # raised from 1
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,
        nqdtc_oppose_size_mult=0.0,
    )


def make_10k_v4_config() -> PortfolioConfig:
    """$10K v4: NQDTC v2.1 dual-session (ETH+RTH) portfolio.

    NQDTC changes vs v3:
    - RTH entries enabled (09:45-12:00 ET, NORMAL mode only)
    - ETH window extended (04:30-09:15 ET, was 05:00-09:00)
    - BLOCK_RTH_DEGRADED = True (no chop entries during RTH)
    - Trades/month: 4.9 (up from 3.6)
    - Standalone: 54 trades, 64.8% WR, +1.255R, $42,887, Sharpe 2.89

    Portfolio changes vs v3:
    - Proximity cooldown: 240 -> 120 min. NQDTC RTH (09:45-12:00) now
      overlaps Helix RTH_PRIME1 (09:35-11:30). At 240 min, a Helix entry
      at 09:35 blocks NQDTC until 13:35 — past the 12:00 RTH cutoff,
      effectively nullifying all RTH entries when Helix trades early RTH.
      120 min allows NQDTC RTH entries ~2h after Helix while still
      preventing immediate overlap degradation.

    Session overlap map (ET):
        04:30-09:15  NQDTC ETH only
        09:35-09:45  Helix RTH_PRIME1 only
        09:45-11:30  ALL THREE (Helix RTH_PRIME1 + NQDTC RTH + Vdubus OPEN/CORE)
        11:30-12:00  NQDTC RTH + Vdubus CORE + Helix RTH_DEAD
        12:00-15:50  Helix + Vdubus only (NQDTC RTH closed)
        15:50-17:00  Helix only (ETH_QUALITY_PM)
        19:00-22:30  Vdubus EVENING + Helix ETH_OVERNIGHT

    Risk budget at $10K MNQ:
        Vdubus:  0.80% = $80/trade = ~16 MNQ @ 2.5pt stop
        NQDTC:   0.80% = $80/trade = ~16 MNQ @ 2.5pt stop
        Helix:   1.50% = $150/trade = ~30 MNQ @ 2.5pt stop
        Max simultaneous heat: 3.5R ≈ $280 risk (2.8% of equity)
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=3.5,                          # unchanged from v3
        portfolio_daily_stop_R=2.5,              # unchanged from v3
        max_total_positions=3,
        directional_cap_R=2.5,                   # unchanged from v3
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,                 # unchanged from v3
                max_concurrent=1,
                priority=0,                       # highest — most stable performer
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,                 # unchanged from v3
                max_concurrent=1,
                priority=1,                       # provides direction signal for Vdubus
                continuation_half_size=True,
                continuation_size_mult=0.70,      # unchanged from v3
                helix_veto_enabled=False,         # replaced by proximity cooldown
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,              # 1.50%
                daily_stop_R=2.0,                 # unchanged from v3
                max_concurrent=1,
                priority=2,                       # lowest — enters across all sessions
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,         # reduced from 240: RTH overlap
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,               # unchanged from v3
        nqdtc_oppose_size_mult=0.0,               # block Vdubus when NQDTC opposes
    )


def make_10k_v5_config() -> PortfolioConfig:
    """$10K v5: sweep-optimized from v4 baseline.

    Sweep results vs v4:
      BASELINE:          404 trades, 169.6R, 67.8% R-cap, Sharpe 1.52, $21,027
      cooldown_session:  409 trades, 188.9R, 75.5% R-cap, Sharpe 1.68, $26,285 (+19.3R, +0.155 Sharpe)
      dir_cap_3.5:       414 trades, 176.7R, 70.6% R-cap, Sharpe 1.54, $21,693 (+7.1R, +0.019 Sharpe)
      helix_concurrent2: 405 trades, 171.6R, 68.6% R-cap, Sharpe 1.55, $21,794 (+2.0R, +0.023 Sharpe)
      COMBINED:          425 trades, 204.4R, 81.7% R-cap, Sharpe 1.73, $29,794 (+34.8R, +0.210 Sharpe)

    Changes vs v4:
    - cooldown_session_only=True: only enforce Helix<->NQDTC proximity during
      09:45-11:30 ET overlap. Outside overlap, strategies can trade independently.
      Biggest single lever (+19.3R, +$5,258).
    - directional_cap_R: 2.5 -> 3.5 (match heat_cap). Allows more same-direction
      capacity when diversified strategies agree. +7.1R, +$666.
    - Helix max_concurrent: 1 -> 2. Allows overlapping Helix trades (rare but
      profitable). +2.0R, +$767.
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=3.5,                          # unchanged from v4
        portfolio_daily_stop_R=2.5,              # unchanged from v4
        max_total_positions=3,
        directional_cap_R=3.5,                   # raised from 2.5 (match heat_cap)
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,              # 0.80%
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,
                helix_veto_enabled=False,
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,              # 1.50%
                daily_stop_R=2.0,
                max_concurrent=2,                 # raised from 1
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,         # unchanged — only applies during overlap
        cooldown_session_only=True,               # only enforce during 09:45-11:30 ET
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,
        nqdtc_oppose_size_mult=0.0,
    )


def make_10k_v6_config() -> PortfolioConfig:
    """$10K v6: tighter portfolio daily stop from v5.

    Daily stop sweep results vs v5:
      v5_baseline:    422 trades, 197.6R, Sharpe 1.75, Sortino 1.83, CAGR 82.6%, MaxDD 13.5%, $29,865
      port_daily_2.0: 417 trades, 204.2R, Sharpe 1.83, Sortino 2.05, CAGR 86.0%, MaxDD 13.5%, $31,591
      port_daily_1.5: 413 trades, 209.9R, Sharpe 1.93, Sortino 2.26, CAGR 93.2%, MaxDD 13.5%, $35,392
      port_daily_1.0: 364 trades, 158.4R, Sharpe 1.72, Sortino 1.79, CAGR 64.3%, MaxDD 11.8%, $21,237

    Changes vs v5:
    - portfolio_daily_stop_R: 2.5 -> 1.5. Blocks 9 more trades that are overwhelmingly
      losers — second/third entries on hostile days. Clips NQDTC tail-loss days
      (R=-5.63, -3.86 events). Pure improvement: +0.18 Sharpe, +0.43 Sortino,
      +$5,527 PnL with zero MaxDD increase.
    - 1.0R is too aggressive (drops to 364 trades, Sharpe 1.72).
    - nqdtc_stop_2.0 is subsumed by port_daily_1.5 (identical results when combined).
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=3.5,
        portfolio_daily_stop_R=1.5,              # tightened from 2.5
        max_total_positions=3,
        directional_cap_R=3.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.008,
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,
                helix_veto_enabled=False,
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.015,
                daily_stop_R=2.0,
                max_concurrent=2,
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,
        cooldown_session_only=True,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.50,
        nqdtc_oppose_size_mult=0.0,
    )


def make_10k_v7_config() -> PortfolioConfig:
    """$10K account v7: greedy-optimized from v6 baseline.

    12 mutations from hierarchical optimization pipeline:
    - Net profit: $79,919 vs $35,392 (+126%)
    - Max DD: 8.78% vs 13.5% (-35%)
    - PF: 2.60 vs 2.11 (+23%)
    - Composite score: 0.896 vs 0.822 (+9%)

    Strategy-level changes (applied in live config.py files):
    - Helix: use_momentum_stall=True, use_drawdown_throttle=False, VOL_50_80_SIZING_MULT=0.85
    - NQDTC: max_loss_cap_R=-3.0, MAX_STOP_WIDTH_PTS=200.0
    - Vdubus: PLUS_1R_PARTIAL_ENABLED=False, CHOP_THRESHOLD=40

    Portfolio-level changes (below):
    - Vdubus risk: 0.008 -> 0.01
    - Helix risk: 0.015 -> 0.02
    - nqdtc_direction_filter: disabled (no effect in live — already absent)
    - helix_veto: enabled with 120min window (backtest-only until live implementation)
    """
    return PortfolioConfig(
        tier=AccountTier.SMALL,
        initial_equity=10_000.0,
        heat_cap_R=3.5,
        portfolio_daily_stop_R=1.5,
        max_total_positions=3,
        directional_cap_R=3.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.01,               # v7: was 0.008
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=0,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.008,
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=1,
                continuation_half_size=True,
                continuation_size_mult=0.70,
                helix_veto_enabled=True,           # v7: enabled
                helix_veto_window_minutes=120,     # v7: 120min window
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.02,               # v7: was 0.015
                daily_stop_R=2.0,
                max_concurrent=2,
                priority=2,
            ),
        ),
        helix_nqdtc_cooldown_minutes=120,
        cooldown_session_only=True,
        nqdtc_direction_filter_enabled=False,      # v7: disabled
        nqdtc_agree_size_mult=1.50,
        nqdtc_oppose_size_mult=0.0,
    )


def make_100k_halfsized_config() -> PortfolioConfig:
    """$100K account: All 3 strategies, NQDTC continuation half-sized (original)."""
    return PortfolioConfig(
        tier=AccountTier.STANDARD,
        initial_equity=100_000.0,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=3.0,
        max_total_positions=3,
        directional_cap_R=2.5,
        strategies=(
            StrategyAllocation(
                strategy_id="Vdubus",
                enabled=True,
                base_risk_pct=0.003,      # 0.30%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=0,               # highest priority
            ),
            StrategyAllocation(
                strategy_id="Helix",
                enabled=True,
                base_risk_pct=0.005,      # 0.50%
                daily_stop_R=2.0,
                max_concurrent=1,
                priority=1,
            ),
            StrategyAllocation(
                strategy_id="NQDTC",
                enabled=True,
                base_risk_pct=0.0035,     # 0.35%
                daily_stop_R=2.5,
                max_concurrent=1,
                priority=2,               # lowest priority
                continuation_half_size=True,
                helix_veto_enabled=True,
                helix_veto_window_minutes=60,
            ),
        ),
    )
