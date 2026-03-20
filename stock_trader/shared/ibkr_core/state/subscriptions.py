"""Market data subscription management with reference counting."""
from dataclasses import dataclass, field


@dataclass
class SubscriptionManager:
    """Tracks active market data subscriptions with reference counting."""

    _subs: dict[int, int] = field(default_factory=dict)  # req_id -> ref_count
    _con_to_req: dict[int, int] = field(default_factory=dict)  # con_id -> req_id

    def subscribe(self, con_id: int, req_id: int) -> bool:
        """Returns True if this is a new subscription (ref_count was 0)."""
        if con_id in self._con_to_req:
            existing_req = self._con_to_req[con_id]
            self._subs[existing_req] = self._subs.get(existing_req, 0) + 1
            return False
        self._con_to_req[con_id] = req_id
        self._subs[req_id] = 1
        return True

    def unsubscribe(self, con_id: int) -> bool:
        """Returns True if ref_count hits 0 (should cancel with IB)."""
        req_id = self._con_to_req.get(con_id)
        if req_id is None:
            return False
        self._subs[req_id] -= 1
        if self._subs[req_id] <= 0:
            del self._subs[req_id]
            del self._con_to_req[con_id]
            return True
        return False

    def get_req_id(self, con_id: int) -> int | None:
        return self._con_to_req.get(con_id)

    def is_subscribed(self, con_id: int) -> bool:
        return con_id in self._con_to_req
