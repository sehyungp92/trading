"""Trade recording service for completed trade telemetry.

Call on trade entry and exit to record trade details and MAE/MFE.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..oms.persistence.postgres import PgStore
from ..oms.persistence.schema import TradeRow, TradeMarksRow


class TradeRecorder:
    """Records completed trades with telemetry."""

    def __init__(self, store: PgStore):
        self._store = store

    async def record_entry(
        self,
        strategy_id: str,
        instrument: str,
        direction: str,
        quantity: int,
        entry_price: Decimal,
        entry_ts: datetime,
        setup_tag: str = None,
        entry_type: str = None,
        meta: dict = None,
        account_id: str = "default",
    ) -> str:
        """Record trade entry. Returns trade_id for later exit recording."""
        trade_id = uuid.uuid4().hex[:16]

        row = TradeRow(
            trade_id=trade_id,
            strategy_id=strategy_id,
            account_id=account_id,
            instrument_symbol=instrument,
            direction=direction,
            quantity=quantity,
            entry_ts=entry_ts,
            entry_price=entry_price,
            setup_tag=setup_tag,
            entry_type=entry_type,
            meta_json=json.dumps(meta) if meta else "{}",
        )
        await self._store.save_trade(row)
        return trade_id

    async def record_exit(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_ts: datetime,
        exit_reason: str,
        realized_r: Decimal,
        realized_usd: Decimal = None,
        notes: str = None,
        # MAE/MFE data
        mae_r: Decimal = None,
        mfe_r: Decimal = None,
        duration_seconds: int = None,
        duration_bars: int = None,
        max_adverse_price: Decimal = None,
        max_favorable_price: Decimal = None,
    ) -> None:
        """Record trade exit with optional MAE/MFE metrics."""
        # Update trade record
        await self._store._pool.execute(
            """
            UPDATE trades SET
                exit_ts = $2,
                exit_price = $3,
                exit_reason = $4,
                realized_r = $5,
                realized_usd = $6,
                notes = $7
            WHERE trade_id = $1
            """,
            trade_id,
            exit_ts,
            exit_price,
            exit_reason,
            realized_r,
            realized_usd,
            notes,
        )

        # Record trade marks if provided
        if mae_r is not None or mfe_r is not None:
            marks = TradeMarksRow(
                trade_id=trade_id,
                duration_seconds=duration_seconds,
                duration_bars=duration_bars,
                mae_r=mae_r,
                mfe_r=mfe_r,
                max_adverse_price=max_adverse_price,
                max_favorable_price=max_favorable_price,
            )
            await self._store.save_trade_marks(marks)
