"""Risk engine: single-tier heat, chop sizing penalty, hour sizing.

v4.0 changes:
- Removed Class A and R sizing logic
- Single-tier heat caps (no alignment-extended or extreme branches)
- Chop zone: sizing penalty instead of hard block
- Hour-of-day sizing multiplier
"""
from __future__ import annotations

from typing import Optional

from .config import (
    Setup, PositionState, SetupClass, SessionBlock,
    BASE_RISK_PCT, NQ_POINT_VALUE,
    EXTREME_VOL_RISK_MULT, STRONG_TREND_BONUS,
    SIZE_MULT_M, SIZE_MULT_M_STRONG, SIZE_MULT_M_CHOP,
    CLASS_F_SIZE_MULT, CLASS_T_SIZE_MULT,
    CHOP_ZONE_VOL_LOW, CHOP_ZONE_VOL_HIGH,
    REENTRY_SIZE_PENALTY,
    HEAT_CAP_R, HEAT_CAP_DIR_R,
    HOUR_SIZE_MULT, DOW_SIZE_MULT,
    RTH_DEAD_ENHANCED_SIZE, RTH_DEAD_ENHANCED_MIN_TREND,
    RTH_DEAD_ENHANCED_MAX_VOL,
    ETH_EUROPE_REGIME_SIZE_MULT,
)
from .indicators import VolEngine


class RiskEngine:
    """Computes unit risk, sizes contracts, tracks heat."""

    def __init__(self, equity: float, vol_engine: VolEngine, point_value: float = NQ_POINT_VALUE):
        self.equity = equity
        self.vol = vol_engine
        self.pv = point_value
        self.open_risk_r: float = 0.0
        self.pending_risk_r: float = 0.0
        self.dir_risk_r: dict[int, float] = {1: 0.0, -1: 0.0}

    def update_equity(self, equity: float) -> None:
        self.equity = equity

    def heat_cap_r(self) -> float:
        return HEAT_CAP_R

    def heat_cap_dir_r(self) -> float:
        return HEAT_CAP_DIR_R

    def compute_risk_r(self, entry: float, stop: float, contracts: int, unit1_risk: float) -> float:
        risk_usd = abs(entry - stop) * self.pv * contracts
        return risk_usd / max(unit1_risk, 1e-9)

    def compute_unit1_risk_usd(self, setup: Setup) -> float:
        base = BASE_RISK_PCT * self.equity * self.vol.vol_factor
        if self.vol.extreme_vol:
            base *= EXTREME_VOL_RISK_MULT
        if (setup.strong_trend
                and setup.cls == SetupClass.M
                and not setup.is_extended):
            base *= STRONG_TREND_BONUS
        return base

    def setup_size_mult(self, setup: Setup) -> float:
        if setup.cls == SetupClass.M:
            mult = SIZE_MULT_M.get(setup.alignment_score, 0.0)
            if mult == 0.0:
                return 0.0
            if setup.strong_trend and setup.alignment_score == 2:
                mult *= SIZE_MULT_M_STRONG
        elif setup.cls == SetupClass.F:
            mult = CLASS_F_SIZE_MULT
        elif setup.cls == SetupClass.T:
            mult = CLASS_T_SIZE_MULT
        else:
            return 0.0

        # Chop zone sizing penalty (replaces hard gate block)
        if setup.cls == SetupClass.M and CHOP_ZONE_VOL_LOW < self.vol.vol_pct < CHOP_ZONE_VOL_HIGH:
            mult *= SIZE_MULT_M_CHOP

        if setup.is_reentry:
            mult *= REENTRY_SIZE_PENALTY
        return mult

    def hour_size_mult(self, hour: int) -> float:
        """Return hour-of-day sizing multiplier."""
        return HOUR_SIZE_MULT.get(hour, 1.0)

    def dow_size_mult(self, weekday: int) -> float:
        """Return day-of-week sizing multiplier (live trading only)."""
        return DOW_SIZE_MULT.get(weekday, 1.0)

    def rtd_dead_enhanced_mult(
        self, block: SessionBlock, score: int, ts_daily: float,
    ) -> float:
        """Return enhanced RTH_DEAD sizing if conditions met, else default."""
        if block != SessionBlock.RTH_DEAD:
            return 1.0
        if (score >= 2
                and ts_daily > RTH_DEAD_ENHANCED_MIN_TREND
                and self.vol.vol_pct < RTH_DEAD_ENHANCED_MAX_VOL
                and not self.vol.extreme_vol):
            return RTH_DEAD_ENHANCED_SIZE / 0.70  # ratio vs base session_size_mult
        return 1.0

    def eth_europe_regime_mult(self, block: SessionBlock) -> float:
        """Return ETH_EUROPE regime sizing (overrides session_size_mult)."""
        if block != SessionBlock.ETH_EUROPE:
            return 1.0
        return ETH_EUROPE_REGIME_SIZE_MULT / 0.50  # ratio vs base session_size_mult

    def size_contracts(
        self, setup: Setup, session_mult: float, fill_price: Optional[float] = None,
        hour_mult: float = 1.0, dow_mult: float = 1.0,
    ) -> int:
        ref_price = fill_price if fill_price is not None else setup.entry_stop
        risk_per_contract = abs(ref_price - setup.stop0) * self.pv
        if risk_per_contract <= 0:
            return 0

        unit_risk = self.compute_unit1_risk_usd(setup)
        mult = self.setup_size_mult(setup)
        if mult <= 0:
            return 0

        contracts = int((unit_risk * mult * session_mult * hour_mult * dow_mult) / risk_per_contract)
        return max(0, contracts)

    # ── Heat tracking ────────────────────────────────────────────

    def add_pending_risk(self, direction: int, risk_r: float) -> None:
        self.pending_risk_r += risk_r
        self.dir_risk_r[direction] = self.dir_risk_r.get(direction, 0.0) + risk_r

    def promote_to_open(self, direction: int, risk_r: float) -> None:
        self.pending_risk_r = max(0, self.pending_risk_r - risk_r)
        self.open_risk_r += risk_r

    def release_pending_risk(self, direction: int, risk_r: float) -> None:
        self.pending_risk_r = max(0, self.pending_risk_r - risk_r)
        self.dir_risk_r[direction] = max(0, self.dir_risk_r.get(direction, 0.0) - risk_r)

    def release_open_risk(self, direction: int, risk_r: float) -> None:
        self.open_risk_r = max(0, self.open_risk_r - risk_r)
        self.dir_risk_r[direction] = max(0, self.dir_risk_r.get(direction, 0.0) - risk_r)

    def adjust_open_risk(self, direction: int, delta_r: float) -> None:
        self.open_risk_r = max(0, self.open_risk_r + delta_r)
        self.dir_risk_r[direction] = max(0, self.dir_risk_r.get(direction, 0.0) + delta_r)

    def heat_allows(self, setup: Setup, session_mult: float) -> bool:
        unit_risk = self.compute_unit1_risk_usd(setup)
        risk_per_contract = abs(setup.entry_stop - setup.stop0) * self.pv
        if risk_per_contract <= 0:
            return False
        mult = self.setup_size_mult(setup)
        if mult <= 0:
            return False
        contracts = int((unit_risk * mult * session_mult) / risk_per_contract)
        est_risk_r = self.compute_risk_r(setup.entry_stop, setup.stop0, max(1, contracts), unit_risk)
        total = self.open_risk_r + self.pending_risk_r + est_risk_r
        dir_total = self.dir_risk_r.get(setup.direction, 0.0) + est_risk_r
        return total <= self.heat_cap_r() and dir_total <= self.heat_cap_dir_r()
