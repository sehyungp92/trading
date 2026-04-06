"""BRS OMS execution interface -- order submission via Intent system.

Translates BRS position actions into OMS Intents (LIMIT entries, STOP exits,
market flatten).  Follows the same pattern as Helix ExecutionEngine.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .config import STRATEGY_ID
from .models import Direction
from .positions import BRSPositionState

from libs.oms.models.intent import Intent, IntentType, IntentResult
from libs.oms.models.order import OMSOrder, OrderSide, OrderType, OrderRole, RiskContext

logger = logging.getLogger(__name__)


class BRSExecutionEngine:
    """Submits BRS orders via the OMS intent system."""

    def __init__(self, oms_service: Any, instruments: dict[str, Any]) -> None:
        self.oms = oms_service
        self.instruments = instruments  # {symbol: Instrument}

    # ── helpers ────────────────────────────────────────────────────────

    def _round_price(self, price: float, tick: float = 0.01) -> float:
        return round(round(price / tick) * tick, 2)

    def _side_for(self, direction: Direction, is_exit: bool = False) -> OrderSide:
        if is_exit:
            return OrderSide.BUY if direction == Direction.SHORT else OrderSide.SELL
        return OrderSide.SELL if direction == Direction.SHORT else OrderSide.BUY

    # ── entry ─────────────────────────────────────────────────────────

    async def submit_entry(
        self,
        symbol: str,
        pos_id: str,
        direction: Direction,
        qty: int,
        limit_price: float,
        stop_for_risk: float,
        risk_dollars: float,
    ) -> Optional[str]:
        """Submit LIMIT entry order. Returns oms_order_id or None."""
        instrument = self.instruments.get(symbol)
        if instrument is None:
            logger.error("No instrument for %s", symbol)
            return None

        order = OMSOrder(
            client_order_id=f"{STRATEGY_ID}_{pos_id}_entry",
            strategy_id=STRATEGY_ID,
            instrument=instrument,
            side=self._side_for(direction),
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=self._round_price(limit_price),
            tif="IOC",
            role=OrderRole.ENTRY,
            risk_context=RiskContext(
                stop_for_risk=self._round_price(stop_for_risk),
                planned_entry_price=self._round_price(limit_price),
                risk_budget_tag=f"BRS_{symbol}",
                risk_dollars=risk_dollars,
            ),
        )

        try:
            receipt = await self.oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=order,
            ))
            if receipt.result == IntentResult.ACCEPTED:
                logger.info("Entry submitted: %s %s %d @ %.2f", symbol, direction.name, qty, limit_price)
                return receipt.oms_order_id
            else:
                logger.warning("Entry denied: %s -- %s", symbol, receipt.denial_reason)
                return None
        except Exception:
            logger.exception("Entry submission failed: %s", symbol)
            return None

    # ── stop placement / modification ─────────────────────────────────

    async def place_stop(
        self,
        symbol: str,
        pos: BRSPositionState,
        stop_price: float,
    ) -> Optional[str]:
        """Place initial STOP exit order. Returns oms_order_id or None."""
        instrument = self.instruments.get(symbol)
        if instrument is None:
            return None

        order = OMSOrder(
            client_order_id=f"{STRATEGY_ID}_{pos.pos_id}_stop",
            strategy_id=STRATEGY_ID,
            instrument=instrument,
            side=self._side_for(pos.direction, is_exit=True),
            qty=pos.qty,
            order_type=OrderType.STOP,
            stop_price=self._round_price(stop_price),
            tif="GTC",
            role=OrderRole.EXIT,
        )

        try:
            receipt = await self.oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=order,
            ))
            if receipt.result == IntentResult.ACCEPTED:
                return receipt.oms_order_id
            logger.warning("Stop denied: %s -- %s", symbol, receipt.denial_reason)
            return None
        except Exception:
            logger.exception("Stop placement failed: %s", symbol)
            return None

    async def modify_stop(
        self,
        pos: BRSPositionState,
        new_stop: float,
    ) -> None:
        """Modify existing stop order to new price (ratchet only)."""
        if pos.stop_oms_id is None:
            return

        try:
            await self.oms.submit_intent(Intent(
                intent_type=IntentType.REPLACE_ORDER,
                strategy_id=STRATEGY_ID,
                target_oms_order_id=pos.stop_oms_id,
                new_stop_price=self._round_price(new_stop),
            ))
        except Exception:
            logger.exception("Stop modify failed: %s", pos.symbol)

    # ── scale-out ─────────────────────────────────────────────────────

    async def submit_scale_out(
        self,
        symbol: str,
        pos: BRSPositionState,
        qty: int,
        limit_price: float,
    ) -> Optional[str]:
        """Submit LIMIT partial exit (scale-out tranche B)."""
        instrument = self.instruments.get(symbol)
        if instrument is None:
            return None

        order = OMSOrder(
            client_order_id=f"{STRATEGY_ID}_{pos.pos_id}_scaleout",
            strategy_id=STRATEGY_ID,
            instrument=instrument,
            side=self._side_for(pos.direction, is_exit=True),
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=self._round_price(limit_price),
            tif="IOC",
            role=OrderRole.EXIT,
        )

        try:
            receipt = await self.oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=order,
            ))
            if receipt.result == IntentResult.ACCEPTED:
                logger.info("Scale-out submitted: %s %d @ %.2f", symbol, qty, limit_price)
                return receipt.oms_order_id
            return None
        except Exception:
            logger.exception("Scale-out failed: %s", symbol)
            return None

    # ── flatten ───────────────────────────────────────────────────────

    async def flatten(self, symbol: str, reason: str = "") -> None:
        """Market exit -- flatten all BRS positions in symbol."""
        try:
            await self.oms.submit_intent(Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=STRATEGY_ID,
                instrument_symbol=symbol,
            ))
            logger.info("Flatten submitted: %s (%s)", symbol, reason)
        except Exception:
            logger.exception("Flatten failed: %s", symbol)

    # ── cancel ────────────────────────────────────────────────────────

    async def cancel_order(self, oms_order_id: str) -> None:
        """Cancel a pending order by OMS ID."""
        try:
            await self.oms.submit_intent(Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            ))
        except Exception:
            logger.exception("Cancel failed: %s", oms_order_id)
