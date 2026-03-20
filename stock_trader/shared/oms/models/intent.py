"""Intent models for strategy -> OMS communication."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .order import OMSOrder


class IntentType(Enum):
    NEW_ORDER = "NEW_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    REPLACE_ORDER = "REPLACE_ORDER"
    FLATTEN = "FLATTEN"


class IntentResult(Enum):
    ACCEPTED = "ACCEPTED"
    DENIED = "DENIED"


@dataclass
class Intent:
    intent_type: IntentType
    strategy_id: str
    # For NEW_ORDER: full OMSOrder
    order: Optional["OMSOrder"] = None
    # For CANCEL/REPLACE: target oms_order_id
    target_oms_order_id: Optional[str] = None
    # For REPLACE: new parameters
    new_qty: Optional[int] = None
    new_limit_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    # For FLATTEN: optional instrument filter
    instrument_symbol: Optional[str] = None


@dataclass
class IntentReceipt:
    result: IntentResult
    intent_id: str
    oms_order_id: Optional[str] = None
    denial_reason: Optional[str] = None
