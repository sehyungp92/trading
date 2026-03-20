"""Execution engine: order placement via OMS (v4.0).

v4.0 changes: imports from strategy config, STRATEGY_ID updated.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from shared.oms.models.intent import Intent, IntentType
from shared.oms.models.order import (
    OMSOrder, OrderSide, OrderType, OrderRole,
    EntryPolicy, RiskContext,
)

from .config import (
    STRATEGY_ID, Setup, SetupState, PositionState, SetupClass, SessionBlock,
    NQ_POINT_VALUE, NQ_TICK,
    TTL_1H_S, TTL_CATCHUP_S,
    CATCHUP_OVERSHOOT_ATR_FRAC, CATCHUP_OFFSET_PTS,
    TELEPORT_RTH_TICKS, TELEPORT_RTH_ATR_FRAC,
    TELEPORT_ETH_TICKS, TELEPORT_ETH_ATR_FRAC,
    CATASTROPHIC_FILL_MULT,
)
from .session import get_session_block, is_rth

logger = logging.getLogger(__name__)


def _round_price(price: float, tick: float = NQ_TICK) -> float:
    return round(price / tick) * tick


def _teleport_ticks(now_et: datetime) -> int:
    block = get_session_block(now_et)
    return TELEPORT_RTH_TICKS if is_rth(block) else TELEPORT_ETH_TICKS


class ExecutionEngine:
    def __init__(self, oms_service, instrument, point_value: float = NQ_POINT_VALUE, tick_size: float = NQ_TICK):
        self.oms = oms_service
        self.instrument = instrument
        self.pv = point_value
        self.tick = tick_size
        self.pending_setups: list[Setup] = []

    async def arm_setup(self, setup: Setup, now_et: datetime, contracts: int) -> bool:
        if contracts < 1:
            logger.warning("Skip arm %s: 0 contracts", setup.setup_id)
            setup.state = SetupState.CANCELED
            return False

        setup.contracts = contracts
        setup.oca_group = f"OCA_{setup.setup_id}"

        side = OrderSide.BUY if setup.direction == 1 else OrderSide.SELL
        client_id = f"{STRATEGY_ID}_{setup.setup_id}_entry"

        order = OMSOrder(
            client_order_id=client_id,
            strategy_id=STRATEGY_ID,
            instrument=self.instrument,
            side=side,
            qty=contracts,
            order_type=OrderType.STOP,
            stop_price=_round_price(setup.entry_stop, self.tick),
            tif="GTC",
            role=OrderRole.ENTRY,
            oca_group=setup.oca_group,
            entry_policy=EntryPolicy(
                ttl_seconds=setup.ttl_seconds(),
                teleport_ticks=_teleport_ticks(now_et),
            ),
            risk_context=RiskContext(
                stop_for_risk=_round_price(setup.stop0, self.tick),
                planned_entry_price=_round_price(setup.entry_stop, self.tick),
                risk_budget_tag=f"Class_{setup.cls.value}",
                risk_dollars=abs(setup.entry_stop - setup.stop0) * self.pv * contracts,
            ),
        )

        receipt = await self.oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        ))

        if receipt.result.value == "ACCEPTED":
            setup.entry_oms_id = receipt.oms_order_id
            setup.placed_ts = now_et
            setup.ttl_expiry_ts = now_et + timedelta(seconds=setup.ttl_seconds())
            setup.state = SetupState.PENDING
            self.pending_setups.append(setup)
            logger.info("Armed %s %s dir=%d entry=%.2f stop=%.2f qty=%d",
                        setup.cls.value, setup.setup_id, setup.direction,
                        setup.entry_stop, setup.stop0, contracts)
            return True
        else:
            logger.warning("Arm denied %s: %s", setup.setup_id, receipt.denial_reason)
            setup.state = SetupState.CANCELED
            return False

    async def place_catch_up(
        self, setup: Setup, last_price: float, atr1h: float, now_et: datetime,
    ) -> bool:
        overshoot_cap = CATCHUP_OVERSHOOT_ATR_FRAC * atr1h
        if setup.direction == 1:
            overshoot = last_price - setup.entry_stop
            if overshoot < 0 or overshoot > overshoot_cap:
                return False
            limit = last_price + CATCHUP_OFFSET_PTS
            side = OrderSide.BUY
        else:
            overshoot = setup.entry_stop - last_price
            if overshoot < 0 or overshoot > overshoot_cap:
                return False
            limit = last_price - CATCHUP_OFFSET_PTS
            side = OrderSide.SELL

        client_id = f"{STRATEGY_ID}_{setup.setup_id}_catchup"
        order = OMSOrder(
            client_order_id=client_id,
            strategy_id=STRATEGY_ID,
            instrument=self.instrument,
            side=side,
            qty=setup.contracts,
            order_type=OrderType.LIMIT,
            limit_price=_round_price(limit, self.tick),
            tif="GTC",
            role=OrderRole.ENTRY,
            oca_group=setup.oca_group,
            entry_policy=EntryPolicy(ttl_seconds=TTL_CATCHUP_S),
            risk_context=RiskContext(
                stop_for_risk=_round_price(setup.stop0, self.tick),
                planned_entry_price=_round_price(limit, self.tick),
                risk_budget_tag=f"Class_{setup.cls.value}_catchup",
                risk_dollars=abs(limit - setup.stop0) * self.pv * setup.contracts,
            ),
        )

        receipt = await self.oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        ))
        if receipt.result.value == "ACCEPTED":
            setup.catchup_oms_id = receipt.oms_order_id
            logger.info("Catch-up placed %s at %.2f", setup.setup_id, limit)
            return True
        return False

    async def place_stop(self, pos: PositionState, stop_price: float) -> Optional[str]:
        side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
        client_id = f"{STRATEGY_ID}_{pos.pos_id}_stop_{uuid.uuid4().hex[:6]}"

        if pos.stop_oms_id:
            await self._cancel_order(pos.stop_oms_id)

        order = OMSOrder(
            client_order_id=client_id,
            strategy_id=STRATEGY_ID,
            instrument=self.instrument,
            side=side,
            qty=pos.contracts,
            order_type=OrderType.STOP,
            stop_price=_round_price(stop_price, self.tick),
            tif="GTC",
            role=OrderRole.STOP,
        )

        receipt = await self.oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        ))
        if receipt.result.value == "ACCEPTED":
            pos.stop_oms_id = receipt.oms_order_id
            pos.stop_price = stop_price
            return receipt.oms_order_id
        logger.error("Failed to place stop for %s: %s", pos.pos_id, receipt.denial_reason)
        return None

    async def tighten_stop(self, pos: PositionState, new_stop: float, reason: str) -> None:
        if pos.direction == 1:
            if new_stop <= pos.stop_price:
                return
        else:
            if new_stop >= pos.stop_price:
                return
        logger.info("Tighten stop %s: %.2f -> %.2f (%s)", pos.pos_id, pos.stop_price, new_stop, reason)
        await self.place_stop(pos, new_stop)

    async def partial_exit(self, pos: PositionState, qty: int, reason: str) -> Optional[str]:
        side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
        client_id = f"{STRATEGY_ID}_{pos.pos_id}_partial_{uuid.uuid4().hex[:6]}"

        order = OMSOrder(
            client_order_id=client_id,
            strategy_id=STRATEGY_ID,
            instrument=self.instrument,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.EXIT,
        )

        receipt = await self.oms.submit_intent(Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=STRATEGY_ID,
            order=order,
        ))
        if receipt.result.value == "ACCEPTED":
            pos.exit_oms_ids.append(receipt.oms_order_id)
            logger.info("Partial exit %s: %d contracts (%s)", pos.pos_id, qty, reason)
            return receipt.oms_order_id
        logger.warning("Partial exit failed %s: %s", pos.pos_id, receipt.denial_reason)
        return None

    async def flatten(self, pos: PositionState, reason: str) -> None:
        logger.info("Flatten %s: %d contracts (%s)", pos.pos_id, pos.contracts, reason)
        receipt = await self.oms.submit_intent(Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=STRATEGY_ID,
            instrument_symbol=self.instrument.symbol,
        ))
        if receipt.result.value == "ACCEPTED":
            pos.contracts = 0

    async def cancel_setup(self, setup: Setup, reason: str) -> None:
        for oid in (setup.entry_oms_id, setup.catchup_oms_id):
            if oid:
                await self._cancel_order(oid)
        setup.state = SetupState.CANCELED
        if setup in self.pending_setups:
            self.pending_setups.remove(setup)
        logger.info("Canceled setup %s: %s", setup.setup_id, reason)

    async def cancel_all_pending_entries(self, reason: str) -> None:
        for s in list(self.pending_setups):
            await self.cancel_setup(s, reason)

    async def _cancel_order(self, oms_order_id: str) -> None:
        try:
            await self.oms.submit_intent(Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            ))
        except Exception as e:
            logger.warning("Cancel failed for %s: %s", oms_order_id, e)

    def expire_setups(self, now_et: datetime) -> list[Setup]:
        expired = []
        for s in list(self.pending_setups):
            if s.ttl_expiry_ts and now_et >= s.ttl_expiry_ts:
                expired.append(s)
        return expired

    def check_teleport(
        self, fill_price: float, trigger_price: float, now_et: datetime, atr1h: float,
    ) -> tuple[bool, bool]:
        slip_pts = abs(fill_price - trigger_price)
        block = get_session_block(now_et)
        if is_rth(block):
            thresh = max(TELEPORT_RTH_TICKS * self.tick, TELEPORT_RTH_ATR_FRAC * atr1h)
        else:
            thresh = max(TELEPORT_ETH_TICKS * self.tick, TELEPORT_ETH_ATR_FRAC * atr1h)
        is_cat = slip_pts > CATASTROPHIC_FILL_MULT * thresh
        is_teleport = slip_pts > thresh
        return is_teleport, is_cat
