"""Multi-Asset Swing Breakout v3.3-ETF — entry gates and eligibility checks.

Pure gate functions returning bool or (bool, str).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from .config import (
    CORR_HEAT_PENALTY,
    CORR_THRESHOLD,
    FRICTION_CAP,
    GAP_GUARD_ATR_MULT,
    HARD_BLOCK_SLOPE_MULT,
    MAX_PORTFOLIO_HEAT,
    MAX_POSITIONS,
    MAX_SAME_DIRECTION,
    MONTHLY_HALT_DAYS,
    MONTHLY_HALT_R,
    NEWS_WINDOWS,
    PENDING_MAX_RTH_HOURS,
    REENTRY_COOLDOWN_DAYS,
    REENTRY_MIN_REALIZED_R,
    MAX_REENTRY_PER_BOX_VERSION,
    WEEKLY_THROTTLE_R,
    SymbolConfig,
)
from .models import (
    CampaignState,
    CircuitBreakerState,
    Direction,
    PendingEntry,
    PositionState,
    Regime4H,
    SymbolCampaign,
)


# ---------------------------------------------------------------------------
# RTH check
# ---------------------------------------------------------------------------

_RTH_START = time(9, 30)
_RTH_END = time(16, 0)


def is_rth_time(dt: datetime) -> bool:
    """Check if datetime falls within RTH (09:30–16:00 ET)."""
    t = dt.time()
    return _RTH_START <= t <= _RTH_END


def next_rth_open(dt: datetime) -> datetime:
    """Compute next RTH open (09:30 ET) from a given datetime."""
    candidate = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    if candidate <= dt:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


# ---------------------------------------------------------------------------
# Entry window
# ---------------------------------------------------------------------------

def is_entry_window_open(now_et: datetime, cfg: SymbolConfig) -> bool:
    """Check if current time is within RTH entry window."""
    start = _parse_time(cfg.entry_window_start_et)
    end = _parse_time(cfg.entry_window_end_et)
    current = now_et.time()
    return start <= current <= end


def _parse_time(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


# ---------------------------------------------------------------------------
# Hard block — double counter (spec §7.3)
# ---------------------------------------------------------------------------

def hard_block(
    direction: Direction,
    regime_4h: Regime4H,
    daily_slope: float,
    atr14_d: float,
) -> bool:
    """Hard block direction if BOTH 4H trend opposes AND daily slope opposes strongly."""
    # 4H opposes
    regime_opposes = (
        (direction == Direction.LONG and regime_4h == Regime4H.BEAR_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BULL_TREND)
    )
    if not regime_opposes:
        return False

    # Daily slope opposes strongly (magnitude > threshold*ATR14_D and sign opposes)
    strong_oppose_th = HARD_BLOCK_SLOPE_MULT * atr14_d
    if direction == Direction.LONG:
        return daily_slope < -strong_oppose_th
    else:
        return daily_slope > strong_oppose_th


# ---------------------------------------------------------------------------
# Friction gate (spec §15)
# ---------------------------------------------------------------------------

def friction_gate(
    symbol: str,
    fee_bps_est: float,
    entry_price: float,
    shares: int,
    final_risk_dollars: float,
) -> bool:
    """Block if friction_$ > 10% of risk$."""
    notional = entry_price * shares
    friction = fee_bps_est * notional
    if final_risk_dollars <= 0:
        return False
    return friction <= FRICTION_CAP * final_risk_dollars


def compute_friction_dollars(
    fee_bps_est: float,
    entry_price: float,
    shares: int,
) -> float:
    """Compute estimated round-trip friction in dollars."""
    return fee_bps_est * entry_price * shares


# ---------------------------------------------------------------------------
# Gap guard (spec §17.3)
# ---------------------------------------------------------------------------

def gap_guard_ok(
    expected_open: float,
    prior_close: float,
    atr14_d: float,
) -> bool:
    """Skip MOO if gap > 1.5*ATR14_D."""
    if atr14_d <= 0:
        return True
    return abs(expected_open - prior_close) <= GAP_GUARD_ATR_MULT * atr14_d


# ---------------------------------------------------------------------------
# Position / heat limits (spec §22)
# ---------------------------------------------------------------------------

def position_limit_check(
    positions: dict[str, PositionState],
    direction: Direction,
) -> tuple[bool, str]:
    """Check MaxPositions and MaxSameDirection."""
    active = [p for p in positions.values() if p.qty > 0]
    if len(active) >= MAX_POSITIONS:
        return False, "max_positions"
    same_dir = sum(1 for p in active if p.direction == direction)
    if same_dir >= MAX_SAME_DIRECTION:
        return False, "max_same_direction"
    return True, ""


def heat_check(
    positions: dict[str, PositionState],
    candidate_risk_dollars: float,
    equity: float,
    candidate_corr_penalty: float = 1.0,
) -> tuple[bool, str]:
    """Check portfolio heat (spec §22).

    candidate_corr_penalty: 1.25 if correlated same-direction position exists.
    """
    if equity <= 0:
        return False, "zero_equity"
    total_risk = sum(p.total_risk_dollars for p in positions.values() if p.qty > 0)
    effective_risk = candidate_risk_dollars * candidate_corr_penalty
    heat = (total_risk + effective_risk) / equity
    if heat > MAX_PORTFOLIO_HEAT:
        return False, "portfolio_heat"
    return True, ""


# ---------------------------------------------------------------------------
# Correlation-aware heat penalty (spec §18.2)
# ---------------------------------------------------------------------------

def compute_corr_heat_penalty(
    direction: Direction,
    positions: dict[str, PositionState],
    correlation_map: dict[tuple[str, str], float],
    symbol: str,
) -> float:
    """If correlated (>0.70) same-direction position exists, return 1.25 penalty."""
    for sym, pos in positions.items():
        if pos.qty <= 0 or pos.direction != direction:
            continue
        pair = tuple(sorted([symbol, sym]))
        corr = correlation_map.get(pair, 0.0)
        if corr > CORR_THRESHOLD:
            return CORR_HEAT_PENALTY
    return 1.0


# ---------------------------------------------------------------------------
# Circuit breakers (spec §22)
# ---------------------------------------------------------------------------

def circuit_breaker_check(cb: CircuitBreakerState) -> tuple[bool, str, float]:
    """Check weekly throttle and monthly halt.

    Returns (ok, reason, size_mult). size_mult = 0.5 if throttled, 0.0 if halted.
    """
    if cb.halted:
        return False, "monthly_halt", 0.0
    if cb.weekly_realized_r <= WEEKLY_THROTTLE_R:
        return True, "weekly_throttle", 0.5
    return True, "", 1.0


# ---------------------------------------------------------------------------
# Re-entry check (spec §21)
# ---------------------------------------------------------------------------

def reentry_allowed(
    direction: Direction,
    campaign: SymbolCampaign,
    realized_r: float,
    days_since_exit: int,
) -> bool:
    """Check re-entry constraints: max 1 per direction per box_version,
    cooldown 3 days, realized_R >= -0.75."""
    dir_key = "LONG" if direction == Direction.LONG else "SHORT"
    count = campaign.reentry_count.get(dir_key, {}).get(campaign.box_version, 0)
    if count >= MAX_REENTRY_PER_BOX_VERSION:
        return False
    if days_since_exit < REENTRY_COOLDOWN_DAYS:
        return False
    if realized_r < REENTRY_MIN_REALIZED_R:
        return False
    return True


# ---------------------------------------------------------------------------
# Pending mechanism (spec §19)
# ---------------------------------------------------------------------------

def check_transient_blocks(
    symbol: str,
    direction: Direction,
    positions: dict[str, PositionState],
    candidate_risk_dollars: float,
    equity: float,
    correlation_map: dict[tuple[str, str], float] | None = None,
) -> Optional[str]:
    """Check if entry is blocked only by transient conditions (MaxPos/Heat/Ops).

    Returns block reason or None if no block.
    """
    pos_ok, pos_reason = position_limit_check(positions, direction)
    if not pos_ok:
        return pos_reason

    corr_map = correlation_map or {}
    penalty = compute_corr_heat_penalty(direction, positions, corr_map, symbol)
    heat_ok, heat_reason = heat_check(positions, candidate_risk_dollars, equity, penalty)
    if not heat_ok:
        return heat_reason

    return None


def pending_expired(pending: PendingEntry, now: datetime) -> bool:
    """Check if pending entry has exceeded RTH time limit."""
    return pending.rth_hours_elapsed >= PENDING_MAX_RTH_HOURS


def pending_block_cleared(
    block_reason: str,
    positions: dict[str, PositionState],
    direction: Direction,
    candidate_risk_dollars: float,
    equity: float,
) -> bool:
    """Re-check if the transient block has cleared."""
    if block_reason in ("max_positions", "max_same_direction"):
        ok, _ = position_limit_check(positions, direction)
        return ok
    if block_reason == "portfolio_heat":
        ok, _ = heat_check(positions, candidate_risk_dollars, equity)
        return ok
    return False


# ---------------------------------------------------------------------------
# News guard
# ---------------------------------------------------------------------------

def is_news_blocked(
    now_et: datetime,
    symbol: str,
    calendar: list[tuple[str, datetime]],
) -> bool:
    """Check if any news event blocks entry."""
    for event_type, event_dt in calendar:
        windows = NEWS_WINDOWS.get(event_type)
        if windows is None:
            continue
        if event_type == "CL_INVENTORY" and symbol not in ("USO",):
            continue
        if event_type == "CRYPTO_EVENT" and symbol not in ("IBIT",):
            continue
        before_min, after_min = windows
        window_start = event_dt + timedelta(minutes=before_min)
        window_end = event_dt + timedelta(minutes=after_min)
        if window_start <= now_et <= window_end:
            return True
    return False


# ---------------------------------------------------------------------------
# Full eligibility (spec §24 execution precedence)
# ---------------------------------------------------------------------------

def full_eligibility_check(
    symbol: str,
    direction: Direction,
    campaign: SymbolCampaign,
    cfg: SymbolConfig,
    now_et: datetime,
    cb: CircuitBreakerState,
    positions: dict[str, PositionState],
    candidate_risk_dollars: float,
    equity: float,
    calendar: list[tuple[str, datetime]],
    correlation_map: dict[tuple[str, str], float] | None = None,
) -> tuple[bool, str, float]:
    """Run all gates per spec §24 execution precedence.

    Returns (ok, reason, circuit_breaker_mult).
    """
    # 1. Monthly halt / weekly throttle
    cb_ok, cb_reason, cb_mult = circuit_breaker_check(cb)
    if not cb_ok:
        return False, cb_reason, 0.0

    # 2. Position/heat limits
    pos_ok, pos_reason = position_limit_check(positions, direction)
    if not pos_ok:
        return False, pos_reason, cb_mult

    corr_map = correlation_map or {}
    penalty = compute_corr_heat_penalty(direction, positions, corr_map, symbol)
    heat_ok, heat_reason = heat_check(
        positions, candidate_risk_dollars, equity, penalty
    )
    if not heat_ok:
        return False, heat_reason, cb_mult

    # 3. Entry window
    if not is_entry_window_open(now_et, cfg):
        return False, "outside_entry_window", cb_mult

    # 4. News guard
    if is_news_blocked(now_et, symbol, calendar):
        return False, "news_blocked", cb_mult

    # 5. Campaign state
    if campaign.state in (CampaignState.INACTIVE, CampaignState.INVALIDATED,
                          CampaignState.EXPIRED):
        return False, "campaign_inactive", cb_mult

    return True, "", cb_mult
