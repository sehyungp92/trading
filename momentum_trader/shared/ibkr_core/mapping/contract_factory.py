"""Contract resolution and caching."""
import time
import logging
from ib_async import IB, Future, Contract
from ..config.schemas import ContractTemplate, ExchangeRoute
from ..models.types import IBContractSpec

logger = logging.getLogger(__name__)


class ContractResolutionError(Exception):
    pass


class ContractFactory:
    """Builds and caches IB Contract objects from config templates."""

    def __init__(
        self,
        ib: IB,
        templates: dict[str, ContractTemplate],
        routes: dict[str, ExchangeRoute],
    ):
        self._ib = ib
        self._templates = templates
        self._routes = routes
        self._cache: dict[tuple[str, str, str], tuple[Contract, IBContractSpec, float]] = {}
        self._cache_ttl_s = 86400  # 24 hours

    async def resolve(self, symbol: str, expiry: str) -> tuple[Contract, IBContractSpec]:
        """Resolve a canonical symbol + expiry to a qualified IB Contract.

        Args:
            symbol: root symbol key from contracts.yaml (e.g. "MNQ")
            expiry: YYYYMM or YYYYMMDD

        Returns:
            (qualified Contract, IBContractSpec with conId and metadata)

        Raises:
            ContractResolutionError if not found or ambiguous.
        """
        if symbol not in self._templates:
            raise ContractResolutionError(f"Unknown symbol: {symbol}")

        tmpl = self._templates[symbol]
        cache_key = (symbol, expiry, tmpl.exchange)
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[2]) < self._cache_ttl_s:
            return cached[0], cached[1]

        if not expiry:
            logger.warning("Blank expiry for %s — IB may resolve to front-month", symbol)

        contract = Future(
            symbol=tmpl.symbol,
            exchange=tmpl.exchange,
            currency=tmpl.currency,
            lastTradeDateOrContractMonth=expiry,
        )
        if tmpl.trading_class:
            contract.tradingClass = tmpl.trading_class

        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise ContractResolutionError(f"Failed to qualify {symbol} {expiry}")

        q = qualified[0]
        if not q.conId:
            raise ContractResolutionError(f"Qualified contract has no conId: {symbol} {expiry}")
        spec = IBContractSpec(
            con_id=q.conId,
            symbol=q.symbol,
            sec_type=q.secType,
            exchange=q.exchange,
            currency=q.currency,
            multiplier=float(q.multiplier) if q.multiplier else tmpl.multiplier,
            tick_size=tmpl.tick_size,
            trading_class=q.tradingClass or "",
            last_trade_date=q.lastTradeDateOrContractMonth,
        )
        self._cache[cache_key] = (q, spec, time.monotonic())
        logger.debug(f"Resolved {symbol} {expiry} -> conId={spec.con_id}")
        return q, spec

    def invalidate(self, symbol: str, expiry: str) -> None:
        """Force cache eviction (e.g. on rollover)."""
        if symbol in self._templates:
            cache_key = (symbol, expiry, self._templates[symbol].exchange)
            self._cache.pop(cache_key, None)

    def clear_cache(self) -> None:
        """Clear all cached contracts."""
        self._cache.clear()
