"""Order Management System - Intent API, risk gateway, execution routing."""

__all__ = [
    "OMSService",
    "IntentHandler",
    "Intent",
    "IntentType",
    "IntentReceipt",
    "IntentResult",
    "OMSOrder",
    "OrderSide",
    "OrderType",
    "OrderRole",
    "OrderStatus",
    "Fill",
    "Position",
    "Instrument",
    "EventBus",
    "RiskGateway",
]


def __getattr__(name: str):
    if name == "OMSService":
        from .services.oms_service import OMSService
        return OMSService
    elif name == "IntentHandler":
        from .intent.handler import IntentHandler
        return IntentHandler
    elif name in ("Intent", "IntentType", "IntentReceipt", "IntentResult"):
        from . import models
        return getattr(models, name)
    elif name in ("OMSOrder", "OrderSide", "OrderType", "OrderRole", "OrderStatus"):
        from . import models
        return getattr(models, name)
    elif name == "Fill":
        from .models.fill import Fill
        return Fill
    elif name == "Position":
        from .models.position import Position
        return Position
    elif name == "Instrument":
        from .models.instrument import Instrument
        return Instrument
    elif name == "EventBus":
        from .events.bus import EventBus
        return EventBus
    elif name == "RiskGateway":
        from .risk.gateway import RiskGateway
        return RiskGateway
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
