"""Cross-strategy portfolio risk rules for live trading.

Implements the rules from PortfolioConfig v6:
  1. Proximity cooldown (Helix <-> NQDTC, session-only during 09:45-11:30 ET)
  2. NQDTC direction filter (affects Vdubus sizing)
  3. Directional cap (max same-direction risk)
  4. Drawdown tiers (size reduction as DD increases)

These rules query the shared `strategy_signals` and `positions` tables
to coordinate across independently-running strategy containers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Optional, Callable, Awaitable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioRulesConfig:
    """Live portfolio rules matching PortfolioConfig v6."""

    # Proximity cooldown (Helix <-> NQDTC, bidirectional)
    helix_nqdtc_cooldown_minutes: int = 120
    cooldown_session_only: bool = True           # only enforce during 09:45-11:30 ET overlap
    helix_strategy_id: str = "AKC_Helix_v40"
    nqdtc_strategy_id: str = "NQDTC_v2.1"

    # NQDTC direction filter (affects Vdubus)
    nqdtc_direction_filter_enabled: bool = True
    nqdtc_agree_size_mult: float = 1.50
    nqdtc_oppose_size_mult: float = 0.0  # 0 = block
    vdubus_strategy_id: str = "VdubusNQ_v4"

    # Directional cap
    directional_cap_R: float = 3.5               # raised from 2.5 to match heat_cap

    # NQDTC chop throttle (affects Helix sizing)
    nqdtc_chop_throttle_enabled: bool = False
    nqdtc_chop_throttle_score: int = 2        # apply throttle when chop_score >= this
    nqdtc_chop_throttle_mult: float = 0.70    # 30% size reduction during chop

    # Drawdown tiers: (dd_pct_threshold, size_multiplier)
    # Applied in order: first tier where current_dd < threshold wins
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.08, 1.00),  # < 8%: full size
        (0.12, 0.50),  # 8-12%: half size
        (0.15, 0.25),  # 12-15%: quarter size
        (1.00, 0.00),  # > 15%: halt
    )
    initial_equity: float = 10_000.0


# ── Result ────────────────────────────────────────────────────────────

@dataclass
class PortfolioRuleResult:
    """Result of portfolio rule checks."""
    approved: bool = True
    denial_reason: Optional[str] = None
    size_multiplier: float = 1.0  # Applied to position size


# ── Checker ───────────────────────────────────────────────────────────

class PortfolioRuleChecker:
    """Checks cross-strategy portfolio rules using shared DB state.

    Each strategy container creates one of these at startup and calls
    check_entry() before submitting orders.
    """

    def __init__(
        self,
        config: PortfolioRulesConfig,
        get_strategy_signal: Callable[[str], Awaitable[Optional[dict]]],
        get_directional_risk_R: Callable[[str], Awaitable[float]],
        get_current_equity: Callable[[], float],
        on_rule_event: Optional[Callable[[dict], None]] = None,
    ):
        self._cfg = config
        self._get_signal = get_strategy_signal
        self._get_dir_risk = get_directional_risk_R
        self._get_equity = get_current_equity
        self._on_rule_event = on_rule_event

    def _emit(self, event: dict) -> None:
        if self._on_rule_event:
            try:
                self._on_rule_event(event)
            except Exception:
                pass

    def _current_dd_pct(self) -> float:
        equity = self._get_equity()
        initial = self._cfg.initial_equity
        if initial <= 0 or equity >= initial:
            return 0.0
        return (initial - equity) / initial

    async def check_entry(
        self,
        strategy_id: str,
        direction: str,  # "LONG" or "SHORT"
        new_risk_R: float = 1.0,
    ) -> PortfolioRuleResult:
        """Run all portfolio rules. Returns result with approval and size multiplier."""
        result = PortfolioRuleResult()

        # 1. Proximity cooldown
        denial = await self._check_proximity_cooldown(strategy_id)
        if denial:
            self._emit({"rule": "proximity_cooldown", "strategy_id": strategy_id,
                         "approved": False, "denial_reason": denial})
            return PortfolioRuleResult(approved=False, denial_reason=denial)

        # 2. NQDTC direction filter (Vdubus only)
        size_mult = await self._check_direction_filter(strategy_id, direction)
        if size_mult == 0.0:
            reason = f"nqdtc_direction_filter: NQDTC opposes {direction}"
            self._emit({"rule": "nqdtc_direction_filter", "strategy_id": strategy_id,
                         "direction": direction, "approved": False, "denial_reason": reason})
            return PortfolioRuleResult(approved=False, denial_reason=reason)
        if size_mult != 1.0:
            self._emit({"rule": "nqdtc_direction_filter", "strategy_id": strategy_id,
                         "direction": direction, "approved": True, "size_multiplier": size_mult})
        result.size_multiplier *= size_mult

        # 3. Directional cap
        denial = await self._check_directional_cap(direction, new_risk_R)
        if denial:
            self._emit({"rule": "directional_cap", "strategy_id": strategy_id,
                         "direction": direction, "approved": False, "denial_reason": denial})
            return PortfolioRuleResult(approved=False, denial_reason=denial)

        # 4. Drawdown tiers
        dd_mult = self._check_drawdown_tier()
        if dd_mult == 0.0:
            reason = "drawdown_halt: equity drawdown exceeds maximum tier"
            self._emit({"rule": "drawdown_tier", "strategy_id": strategy_id,
                         "approved": False, "denial_reason": reason,
                         "drawdown_pct": self._current_dd_pct(), "size_multiplier": 0.0})
            return PortfolioRuleResult(approved=False, denial_reason=reason)
        if dd_mult < 1.0:
            self._emit({"rule": "drawdown_tier", "strategy_id": strategy_id,
                         "approved": True, "size_multiplier": dd_mult,
                         "drawdown_pct": self._current_dd_pct()})
        result.size_multiplier *= dd_mult

        # 5. NQDTC chop throttle (affects Helix only)
        chop_mult = await self._check_chop_throttle(strategy_id)
        if chop_mult < 1.0:
            self._emit({"rule": "nqdtc_chop_throttle", "strategy_id": strategy_id,
                         "approved": True, "size_multiplier": chop_mult})
        result.size_multiplier *= chop_mult

        return result

    async def _check_proximity_cooldown(self, strategy_id: str) -> Optional[str]:
        """Helix-NQDTC bidirectional proximity cooldown."""
        cooldown_min = self._cfg.helix_nqdtc_cooldown_minutes
        if cooldown_min <= 0:
            return None

        helix_id = self._cfg.helix_strategy_id
        nqdtc_id = self._cfg.nqdtc_strategy_id

        # Only applies to Helix and NQDTC
        if strategy_id == helix_id:
            other_id = nqdtc_id
        elif strategy_id == nqdtc_id:
            other_id = helix_id
        else:
            return None

        # Session-only mode: skip cooldown outside 09:45-11:30 ET overlap
        if self._cfg.cooldown_session_only:
            now_et = datetime.now(ZoneInfo("America/New_York"))
            hour_min = now_et.hour * 60 + now_et.minute
            if not (9 * 60 + 45 <= hour_min <= 11 * 60 + 30):
                return None

        other_signal = await self._get_signal(other_id)
        if other_signal is None or other_signal["last_entry_ts"] is None:
            return None

        now = datetime.now(timezone.utc)
        elapsed = now - other_signal["last_entry_ts"]
        cooldown = timedelta(minutes=cooldown_min)

        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (
                f"proximity_cooldown: {other_id} entered {int(elapsed.total_seconds() / 60)}m ago, "
                f"{int(remaining.total_seconds() / 60)}m remaining"
            )
        return None

    async def _check_direction_filter(self, strategy_id: str, direction: str) -> float:
        """NQDTC direction filter — affects Vdubus sizing."""
        if not self._cfg.nqdtc_direction_filter_enabled:
            return 1.0

        if strategy_id != self._cfg.vdubus_strategy_id:
            return 1.0

        nqdtc_signal = await self._get_signal(self._cfg.nqdtc_strategy_id)
        if nqdtc_signal is None or nqdtc_signal["last_direction"] is None:
            return 1.0  # No NQDTC trade today — no filter

        # Only apply if NQDTC traded today
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if nqdtc_signal["signal_date"] != today:
            return 1.0

        if nqdtc_signal["last_direction"] == direction:
            return self._cfg.nqdtc_agree_size_mult
        else:
            return self._cfg.nqdtc_oppose_size_mult

    async def _check_directional_cap(self, direction: str, new_risk_R: float) -> Optional[str]:
        """Max same-direction risk."""
        cap = self._cfg.directional_cap_R
        if cap <= 0:
            return None

        current_dir_risk = await self._get_dir_risk(direction)
        total = current_dir_risk + new_risk_R

        if total > cap:
            return (
                f"directional_cap: {direction} risk {current_dir_risk:.2f}R + "
                f"new {new_risk_R:.2f}R = {total:.2f}R > cap {cap}R"
            )
        return None

    async def _check_chop_throttle(self, strategy_id: str) -> float:
        """NQDTC chop score throttle — reduces Helix sizing during choppy markets."""
        if not self._cfg.nqdtc_chop_throttle_enabled:
            return 1.0

        # Only applies to Helix
        if strategy_id != self._cfg.helix_strategy_id:
            return 1.0

        nqdtc_signal = await self._get_signal(self._cfg.nqdtc_strategy_id)
        if nqdtc_signal is None:
            return 1.0

        chop_score = nqdtc_signal.get("chop_score", 0)
        if chop_score >= self._cfg.nqdtc_chop_throttle_score:
            logger.info(
                "Chop throttle: NQDTC chop_score=%d >= %d → %.0f%% sizing for %s",
                chop_score, self._cfg.nqdtc_chop_throttle_score,
                self._cfg.nqdtc_chop_throttle_mult * 100, strategy_id,
            )
            return self._cfg.nqdtc_chop_throttle_mult
        return 1.0

    def _check_drawdown_tier(self) -> float:
        """Drawdown-based size multiplier."""
        equity = self._get_equity()
        initial = self._cfg.initial_equity
        if initial <= 0 or equity >= initial:
            return 1.0

        dd_pct = (initial - equity) / initial
        for threshold, mult in self._cfg.dd_tiers:
            if dd_pct < threshold:
                if mult < 1.0:
                    logger.info(
                        "Drawdown tier active: %.1f%% DD → %.0f%% size",
                        dd_pct * 100, mult * 100,
                    )
                return mult

        return 0.0  # Beyond all tiers
