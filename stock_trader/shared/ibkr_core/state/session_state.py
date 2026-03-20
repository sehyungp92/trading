"""IB session runtime state tracking."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SessionState:
    """Tracks IB session runtime state."""

    connected: bool = False
    next_valid_id: int = 0
    accounts: list[str] = field(default_factory=list)
    last_heartbeat: Optional[datetime] = None
    last_connect_time: Optional[datetime] = None
    last_disconnect_time: Optional[datetime] = None
