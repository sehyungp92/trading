from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


RoundDirection = Literal["nearest", "up", "down"]


@dataclass(frozen=True, slots=True)
class FuturesSpec:
    symbol: str
    tick: float
    tick_value: float
    point_value: float


FUTURES_SPECS: dict[str, FuturesSpec] = {
    "NQ": FuturesSpec("NQ", tick=0.25, tick_value=5.00, point_value=20.00),
    "MNQ": FuturesSpec("MNQ", tick=0.25, tick_value=0.50, point_value=2.00),
    "ES": FuturesSpec("ES", tick=0.25, tick_value=12.50, point_value=50.00),
}


def spec_for(symbol: str) -> FuturesSpec:
    try:
        return FUTURES_SPECS[symbol.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported futures symbol: {symbol!r}") from exc


def round_to_tick(
    price: float,
    tick_size: float = 0.25,
    direction: RoundDirection = "nearest",
) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    units = price / tick_size
    if direction == "up":
        rounded = math.ceil(units - 1e-12) * tick_size
    elif direction == "down":
        rounded = math.floor(units + 1e-12) * tick_size
    elif direction == "nearest":
        rounded = round(units) * tick_size
    else:
        raise ValueError(f"Unsupported round direction: {direction!r}")
    decimals = max(0, int(abs(math.log10(tick_size))) + 3) if tick_size < 1 else 6
    return round(rounded, decimals)


def compute_contracts(
    equity: float,
    risk_pct: float,
    stop_distance: float,
    point_value: float,
    *,
    min_contracts: int = 0,
    max_contracts: int | None = None,
) -> int:
    """Return conservative whole-contract sizing from dollar risk."""
    if equity <= 0 or risk_pct <= 0 or stop_distance <= 0 or point_value <= 0:
        return 0
    risk_dollars = equity * risk_pct
    per_contract = stop_distance * point_value
    qty = int(math.floor(risk_dollars / per_contract))
    qty = max(min_contracts, qty)
    if max_contracts is not None:
        qty = min(qty, max_contracts)
    return max(0, qty)


def choose_nq_or_mnq(
    equity: float,
    risk_pct: float,
    stop_distance: float,
    *,
    prefer_micro: bool = False,
) -> tuple[str, int]:
    """Size NQ first unless the account requires MNQ granularity."""
    nq = spec_for("NQ")
    mnq = spec_for("MNQ")
    if not prefer_micro:
        nq_qty = compute_contracts(equity, risk_pct, stop_distance, nq.point_value)
        if nq_qty >= 1:
            return "NQ", nq_qty
    mnq_qty = compute_contracts(equity, risk_pct, stop_distance, mnq.point_value)
    return ("MNQ", mnq_qty) if mnq_qty >= 1 else ("", 0)

