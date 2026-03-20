"""Client connection and pacing components."""
from .error_map import classify_error
from .request_ids import RequestIdAllocator
from .throttler import PacingChannel, Throttler

__all__ = [
    "ConnectionManager",
    "classify_error",
    "FarmMonitor",
    "FarmStatus",
    "HeartbeatMonitor",
    "RequestIdAllocator",
    "IBSession",
    "PacingChannel",
    "Throttler",
]


def __getattr__(name: str):
    if name == "ConnectionManager":
        from .connection import ConnectionManager
        return ConnectionManager
    elif name == "FarmMonitor":
        from .farm_monitor import FarmMonitor
        return FarmMonitor
    elif name == "FarmStatus":
        from .farm_monitor import FarmStatus
        return FarmStatus
    elif name == "HeartbeatMonitor":
        from .heartbeat import HeartbeatMonitor
        return HeartbeatMonitor
    elif name == "IBSession":
        from .session import IBSession
        return IBSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
