from __future__ import annotations

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.nq_contract import compute_contracts, round_to_tick, spec_for

from .config import BASE_RISK_PCT, IvbModule, STOP_CAP_IVB_FRACTION, TICK_SIZE, TradeDirection


def continuation_stop(
    *,
    direction: TradeDirection,
    ivb: IVBLevels,
    absorption_extreme: float | None = None,
    reload_zone_edge: float | None = None,
    tick_size: float = TICK_SIZE,
) -> float:
    if direction is TradeDirection.LONG:
        candidates = [ivb.val or ivb.low, ivb.low]
        if absorption_extreme is not None:
            candidates.append(absorption_extreme)
        if reload_zone_edge is not None:
            candidates.append(reload_zone_edge)
        return round_to_tick(min(candidates) - tick_size, tick_size, "down")
    candidates = [ivb.vah or ivb.high, ivb.high]
    if absorption_extreme is not None:
        candidates.append(absorption_extreme)
    if reload_zone_edge is not None:
        candidates.append(reload_zone_edge)
    return round_to_tick(max(candidates) + tick_size, tick_size, "up")


def reclaim_stop(
    *,
    direction: TradeDirection,
    ivb: IVBLevels,
    failed_break_extreme: float,
    tick_size: float = TICK_SIZE,
) -> float:
    if direction is TradeDirection.LONG:
        return round_to_tick(min(failed_break_extreme, ivb.low) - tick_size, tick_size, "down")
    return round_to_tick(max(failed_break_extreme, ivb.high) + tick_size, tick_size, "up")


def stop_within_cap(entry: float, stop: float, ivb: IVBLevels) -> bool:
    return abs(entry - stop) <= STOP_CAP_IVB_FRACTION * ivb.range_pts


def reward_to_risk(entry: float, stop: float, target: float, direction: TradeDirection) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = target - entry if direction is TradeDirection.LONG else entry - target
    return max(0.0, reward / risk)


def risk_pct_for_module(module: IvbModule, size_multiplier: float) -> float:
    module_mult = 1.0
    if module is IvbModule.A2_RECLAIM:
        module_mult = 0.5
    elif module is IvbModule.A3_MEAN_REVERSION:
        module_mult = 0.25
    return BASE_RISK_PCT * module_mult * max(0.0, min(1.0, size_multiplier))


def compute_position_size(
    *,
    equity: float,
    module: IvbModule,
    size_multiplier: float,
    entry: float,
    stop: float,
    symbol: str = "NQ",
) -> int:
    spec = spec_for(symbol)
    return compute_contracts(
        equity,
        risk_pct_for_module(module, size_multiplier),
        abs(entry - stop),
        spec.point_value,
    )

