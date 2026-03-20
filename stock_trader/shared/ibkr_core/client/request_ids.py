"""Request and order ID allocation."""
import threading


class RequestIdAllocator:
    """Thread-safe allocator for IB request and order IDs."""

    def __init__(self, initial_order_id: int = 0):
        self._next_order_id = initial_order_id
        self._next_req_id = 1
        self._lock = threading.Lock()

    def set_next_valid_id(self, order_id: int) -> None:
        """Called when IB sends nextValidId."""
        with self._lock:
            self._next_order_id = max(self._next_order_id, order_id)

    def next_order_id(self) -> int:
        with self._lock:
            oid = self._next_order_id
            self._next_order_id += 1
            return oid

    def next_request_id(self) -> int:
        with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1
            return rid
