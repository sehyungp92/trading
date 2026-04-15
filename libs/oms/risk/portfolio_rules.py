"""Cross-strategy portfolio risk rules for live trading.

Implements the rules from PortfolioConfig v7:
  1. Proximity cooldown (Helix <-> NQDTC, session-only during 09:45-11:30 ET)
  2. NQDTC direction filter (affects Vdubus sizing)
  3. Directional cap (max same-direction risk; asymmetric long/short supported)
  3a. Family contract cap (MNQ-equivalent ceiling across momentum family)
  3b. Symbol collision guard (stock family: block/reduce when sibling holds same ticker)
  4. Drawdown tiers (size reduction as DD increases)
  5. NQDTC chop throttle (affects Helix sizing)

These rules query the shared `strategy_signals` and `positions` tables
to coordinate across independently-running strategy containers.
Used by momentum family (rules 1-5) and stock family (rules 3, 3b, 4).
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
    """Live portfolio rules matching PortfolioConfig v7."""

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

    # Directional cap (symmetric fallback)
    directional_cap_R: float = 3.5               # raised from 2.5 to match heat_cap
    # Asymmetric directional caps (0.0 = use symmetric directional_cap_R)
    directional_cap_long_R: float = 0.0
    directional_cap_short_R: float = 0.0

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

    # Family-scoped rules (stock family)
    family_strategy_ids: tuple[str, ...] = ()  # if set, scope directional cap to these IDs
    symbol_collision_action: str = "none"       # "none", "block", "half_size"

    # Per-pair symbol collision overrides: (holder_id, requester_id, action)
    # When holder has position on symbol, apply action to requester instead of default.
    # Checked before the generic symbol_collision_action fallback.
    symbol_collision_pairs: tuple[tuple[str, str, str], ...] = ()

    # Dynamic family contract cap (0 = disabled)
    max_family_contracts_mnq_eq: int = 0

    # Strategy priority for directional cap (lower number = higher priority)
    strategy_priorities: tuple[tuple[str, int], ...] = ()
    # When remaining directional cap <= this value, only strategies with
    # priority <= priority_reserve_threshold may enter
    priority_headroom_R: float = 0.0       # 0 = disabled (backward compatible)
    priority_reserve_threshold: int = 0

    # Regime-driven sizing scalar (applied as multiplier in check_entry)
    regime_unit_risk_mult: float = 1.0
    # Strategies blocked from new entries by regime (checked first in check_entry)
    disabled_strategies: frozenset[str] = frozenset()

    _VALID_COLLISION_ACTIONS = frozenset({"none", "block", "half_size"})
    _VALID_PAIR_ACTIONS = frozenset({"block", "half_size"})

    def __post_init__(self):
        if self.symbol_collision_action not in self._VALID_COLLISION_ACTIONS:
            raise ValueError(
                f"Invalid symbol_collision_action {self.symbol_collision_action!r}, "
                f"must be one of {sorted(self._VALID_COLLISION_ACTIONS)}"
            )
        for holder, requester, action in self.symbol_collision_pairs:
            if action not in self._VALID_PAIR_ACTIONS:
                raise ValueError(
                    f"Invalid pair action {action!r} for ({holder}, {requester}), "
                    f"must be one of {sorted(self._VALID_PAIR_ACTIONS)}"
                )


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
        get_directional_risk_R_for_strategies: Optional[
            Callable[[str, list[str]], Awaitable[float]]
        ] = None,
        get_sibling_positions_for_symbol: Optional[
            Callable[[list[str], str], Awaitable[bool]]
        ] = None,
        get_family_aggregate_mnq_eq: Optional[
            Callable[[list[str]], Awaitable[int]]
        ] = None,
    ):
        self._cfg = config
        self._get_signal = get_strategy_signal
        self._get_equity = get_current_equity
        self._on_rule_event = on_rule_event
        self._get_sibling = get_sibling_positions_for_symbol
        self._get_family_mnq_eq = get_family_aggregate_mnq_eq

        # Family-scoped directional risk: wrap callback if strategy IDs provided
        family_ids = config.family_strategy_ids
        if family_ids and get_directional_risk_R_for_strategies is not None:
            ids_list = list(family_ids)
            self._get_dir_risk = lambda d: get_directional_risk_R_for_strategies(d, ids_list)
            logger.info("Directional cap scoped to strategies: %s", family_ids)
        else:
            self._get_dir_risk = get_directional_risk_R

    def update_config(self, new_cfg: PortfolioRulesConfig) -> None:
        """Atomically replace config for regime updates. GIL-safe for single attr assign."""
        self._cfg = new_cfg

    def _emit(self, event: dict) -> None:
        if self._on_rule_event:
            try:
                # Normalize to TA pipeline schema: rule_name, result, details
                normalized = {
                    "rule_name": event.get("rule", "unknown"),
                    "result": "pass" if event.get("approved", True) else "block",
                    "details": {},
                }
                details = normalized["details"]
                if "denial_reason" in event:
                    details["reason"] = event["denial_reason"]
                if "symbol" in event:
                    details["blocked_symbol"] = event["symbol"]
                for k in ("strategy_id", "direction", "size_multiplier", "drawdown_pct"):
                    if k in event:
                        details[k] = event[k]
                self._on_rule_event(normalized)
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
        symbol: Optional[str] = None,
        new_qty: int = 0,
    ) -> PortfolioRuleResult:
        """Run all portfolio rules. Returns result with approval and size multiplier."""
        result = PortfolioRuleResult()

        # 0. Regime strategy disable
        if strategy_id in self._cfg.disabled_strategies:
            self._emit({"rule": "regime_disabled", "strategy_id": strategy_id, "approved": False})
            return PortfolioRuleResult(
                approved=False,
                denial_reason=f"regime_disabled: {strategy_id} blocked in current regime",
            )

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
        denial = await self._check_directional_cap(strategy_id, direction, new_risk_R)
        if denial:
            self._emit({"rule": "directional_cap", "strategy_id": strategy_id,
                         "direction": direction, "approved": False, "denial_reason": denial})
            return PortfolioRuleResult(approved=False, denial_reason=denial)

        # 3a. Family contract cap (MNQ-equivalent ceiling)
        denial = await self._check_family_contract_cap(
            strategy_id, symbol or "", new_qty,
        )
        if denial:
            self._emit({"rule": "family_contract_cap", "strategy_id": strategy_id,
                         "approved": False, "denial_reason": denial})
            return PortfolioRuleResult(approved=False, denial_reason=denial)

        # 3b. Symbol collision (stock family -- block/reduce when sibling holds same ticker)
        collision_result = await self._check_symbol_collision(strategy_id, symbol)
        if collision_result is not None:
            if collision_result == 0.0:
                reason = f"symbol_collision: sibling strategy holds {symbol}"
                self._emit({"rule": "symbol_collision", "strategy_id": strategy_id,
                             "symbol": symbol, "approved": False, "denial_reason": reason})
                return PortfolioRuleResult(approved=False, denial_reason=reason)
            self._emit({"rule": "symbol_collision", "strategy_id": strategy_id,
                         "symbol": symbol, "approved": True, "size_multiplier": collision_result})
            result.size_multiplier *= collision_result

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

        # 4b. Regime unit-risk multiplier
        regime_mult = self._cfg.regime_unit_risk_mult
        if regime_mult != 1.0:
            self._emit({"rule": "regime_unit_risk", "strategy_id": strategy_id,
                        "approved": True, "size_multiplier": regime_mult})
            result.size_multiplier *= regime_mult

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

    async def _check_directional_cap(
        self, strategy_id: str, direction: str, new_risk_R: float,
    ) -> Optional[str]:
        """Max same-direction risk, with optional priority-based reservation."""
        # Resolve per-direction cap (asymmetric overrides symmetric fallback)
        cap = self._cfg.directional_cap_R
        if direction == "LONG" and self._cfg.directional_cap_long_R > 0:
            cap = self._cfg.directional_cap_long_R
        elif direction == "SHORT" and self._cfg.directional_cap_short_R > 0:
            cap = self._cfg.directional_cap_short_R
        if cap <= 0:
            return None

        current_dir_risk = await self._get_dir_risk(direction)
        total = current_dir_risk + new_risk_R

        # Hard cap: no strategy can exceed the absolute cap
        if total > cap:
            return (
                f"directional_cap: {direction} risk {current_dir_risk:.2f}R + "
                f"new {new_risk_R:.2f}R = {total:.2f}R > cap {cap}R"
            )

        # Soft reservation: when headroom is tight, reserve for higher-priority strategies
        headroom_R = self._cfg.priority_headroom_R
        if headroom_R <= 0 or not self._cfg.strategy_priorities:
            return None

        remaining = cap - current_dir_risk
        priority_map = dict(self._cfg.strategy_priorities)
        my_priority = priority_map.get(strategy_id, 99)

        if remaining <= headroom_R and my_priority > self._cfg.priority_reserve_threshold:
            return (
                f"directional_cap_reserved: {direction} remaining "
                f"{remaining:.2f}R <= headroom {headroom_R:.1f}R, "
                f"strategy {strategy_id} priority {my_priority} "
                f"> threshold {self._cfg.priority_reserve_threshold}"
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

    async def _check_symbol_collision(
        self, strategy_id: str, symbol: Optional[str],
    ) -> Optional[float]:
        """Check if a sibling strategy already holds the same symbol.

        Returns None if check not applicable, 0.0 to block, or a multiplier to reduce size.
        Per-pair overrides in symbol_collision_pairs are checked first; if none match,
        falls through to the generic symbol_collision_action.
        """
        if not symbol:
            return None
        family_ids = self._cfg.family_strategy_ids
        if not family_ids or self._get_sibling is None:
            return None

        # --- Per-pair overrides (checked first) ---
        for holder_id, requester_id, pair_action in self._cfg.symbol_collision_pairs:
            if requester_id != strategy_id:
                continue
            holder_has = await self._get_sibling([holder_id], symbol)
            if not holder_has:
                continue
            logger.info(
                "Symbol collision pair: %s holds %s → %s for %s",
                holder_id, symbol, pair_action, strategy_id,
            )
            # Actions validated in __post_init__
            if pair_action == "block":
                return 0.0
            return 0.5  # half_size

        # --- Generic fallback ---
        action = self._cfg.symbol_collision_action
        if action == "none":
            return None

        sibling_ids = [sid for sid in family_ids if sid != strategy_id]
        if not sibling_ids:
            return None

        has_collision = await self._get_sibling(sibling_ids, symbol)
        if not has_collision:
            return None

        if action == "block":
            return 0.0
        if action == "half_size":
            logger.info(
                "Symbol collision: sibling holds %s → half size for %s",
                symbol, strategy_id,
            )
            return 0.5
        return None

    async def _check_family_contract_cap(
        self, strategy_id: str, symbol: str, new_qty: int,
    ) -> Optional[str]:
        """Limit total MNQ-equivalent contracts across the family."""
        cap = self._cfg.max_family_contracts_mnq_eq
        if cap <= 0 or self._get_family_mnq_eq is None:
            return None
        family_ids = list(self._cfg.family_strategy_ids)
        if not family_ids:
            return None
        current = await self._get_family_mnq_eq(family_ids)
        new_mnq_eq = abs(new_qty) * (10 if symbol == "NQ" else 1)
        total = current + new_mnq_eq
        if total > cap:
            return (
                f"family_contract_cap: {current} + {new_mnq_eq} = "
                f"{total} MNQ-eq > cap {cap}"
            )
        return None

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
