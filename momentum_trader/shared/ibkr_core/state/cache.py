"""In-memory caches for contracts and order ID mappings."""
from dataclasses import dataclass, field
from ..models.types import IBContractSpec


@dataclass
class IBCache:
    """In-memory caches for contracts and order ID mappings."""

    # conId -> IBContractSpec
    contracts: dict[int, IBContractSpec] = field(default_factory=dict)

    # oms_order_id -> (broker_order_id, perm_id)
    order_map: dict[str, tuple[int, int]] = field(default_factory=dict)

    # broker_order_id -> oms_order_id (reverse lookup)
    reverse_order_map: dict[int, str] = field(default_factory=dict)

    # exec_id set for fill deduplication
    seen_exec_ids: set[str] = field(default_factory=set)

    def register_order(self, oms_order_id: str, broker_order_id: int, perm_id: int) -> None:
        self.order_map[oms_order_id] = (broker_order_id, perm_id)
        self.reverse_order_map[broker_order_id] = oms_order_id

    def lookup_oms_id(self, broker_order_id: int) -> str | None:
        return self.reverse_order_map.get(broker_order_id)

    def lookup_broker_id(self, oms_order_id: str) -> tuple[int, int] | None:
        return self.order_map.get(oms_order_id)

    def is_fill_seen(self, exec_id: str) -> bool:
        return exec_id in self.seen_exec_ids

    def mark_fill_seen(self, exec_id: str) -> None:
        self.seen_exec_ids.add(exec_id)

    def clear(self) -> None:
        self.contracts.clear()
        self.order_map.clear()
        self.reverse_order_map.clear()
        self.seen_exec_ids.clear()

    async def rebuild_from_broker(self, snapshot_fetcher, oms_order_id_resolver=None) -> None:
        """H6 fix: Rebuild order mappings from broker open orders and recent executions.

        Called on startup to restore order ID mappings lost after a restart.

        Args:
            snapshot_fetcher: SnapshotFetcher instance to query broker state
            oms_order_id_resolver: Optional callable(order_ref: str) -> Optional[str]
                that resolves an order reference to an OMS order ID from the DB
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Rebuild from open orders
            open_orders = await snapshot_fetcher.fetch_open_orders()
            for order_event in open_orders:
                broker_id = order_event.broker_order_id
                perm_id = order_event.perm_id
                # If we already know this order, skip
                if self.lookup_oms_id(broker_id) is not None:
                    continue
                # Try to resolve via OMS DB
                if oms_order_id_resolver:
                    oms_id = await oms_order_id_resolver(broker_id)
                    if oms_id:
                        self.register_order(oms_id, broker_id, perm_id)
                        logger.info(f"Cache rebuilt: {oms_id} -> broker={broker_id}")

            # Rebuild from recent executions to deduplicate fills
            executions = await snapshot_fetcher.fetch_executions()
            for exec_report in executions:
                self.mark_fill_seen(exec_report.exec_id)
                broker_id = exec_report.broker_order_id
                if self.lookup_oms_id(broker_id) is None and oms_order_id_resolver:
                    oms_id = await oms_order_id_resolver(broker_id)
                    if oms_id:
                        self.register_order(oms_id, broker_id, exec_report.perm_id)

            logger.info(
                f"IBCache rebuilt: {len(self.order_map)} order mappings, "
                f"{len(self.seen_exec_ids)} seen exec IDs"
            )
        except Exception as e:
            logger.error(f"Failed to rebuild IBCache from broker: {e}")
