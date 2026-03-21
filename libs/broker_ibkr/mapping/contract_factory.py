"""Contract resolution and caching."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ib_async import Contract, Future, IB, Stock

from ..config.schemas import ContractTemplate, ExchangeRoute
from ..models.types import IBContractSpec

if TYPE_CHECKING:
    from libs.oms.models.instrument import Instrument

logger = logging.getLogger(__name__)


class ContractResolutionError(Exception):
    pass


class ContractFactory:
    """Builds and caches IB Contract objects from config templates.

    Supports two resolution paths:
    - Template-based: resolve(symbol, expiry) using contracts.yaml templates
    - Instrument-based: resolve(symbol, expiry, instrument=inst) using rich Instrument metadata
    """

    def __init__(
        self,
        ib: IB,
        templates: dict[str, ContractTemplate],
        routes: dict[str, ExchangeRoute],
    ):
        self._ib = ib
        self._templates = templates
        self._routes = routes
        self._cache: dict[tuple[str, str, str, str, str], tuple[Contract, IBContractSpec, float]] = {}
        self._cache_ttl_s = 86400  # 24 hours

    async def resolve(
        self,
        symbol: str,
        expiry: str = "",
        instrument: Instrument | None = None,
    ) -> tuple[Contract, IBContractSpec]:
        """Resolve an instrument or config symbol to a qualified IB Contract.

        Args:
            symbol: root symbol key from contracts.yaml (e.g. "MNQ", "QQQ")
            expiry: YYYYMM or YYYYMMDD, optional for stocks
            instrument: Rich instrument metadata for dynamic symbols

        Returns:
            (qualified Contract, IBContractSpec with conId and metadata)

        Raises:
            ContractResolutionError if not found or ambiguous.
        """
        template = self._templates.get(symbol)
        route = self._routes.get(symbol)
        instrument = instrument or self._instrument_from_template(symbol, template, route, expiry)
        if instrument is None:
            raise ContractResolutionError(f"Unknown symbol: {symbol}")

        cache_key = (
            instrument.sec_type,
            instrument.root or instrument.symbol,
            instrument.contract_expiry or "",
            instrument.exchange,
            instrument.primary_exchange or "",
        )
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[2]) < self._cache_ttl_s:
            return cached[0], cached[1]

        contract = self._build_contract(instrument)

        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            expiry_msg = f" {instrument.contract_expiry}" if instrument.contract_expiry else ""
            raise ContractResolutionError(f"Failed to qualify {instrument.symbol}{expiry_msg}")

        q = qualified[0]
        spec = IBContractSpec(
            con_id=q.conId,
            symbol=q.symbol,
            sec_type=q.secType,
            exchange=q.exchange,
            currency=q.currency,
            multiplier=float(q.multiplier) if q.multiplier else instrument.multiplier,
            tick_size=instrument.tick_size,
            trading_class=q.tradingClass or instrument.trading_class,
            primary_exchange=getattr(q, "primaryExchange", "") or instrument.primary_exchange,
            last_trade_date=getattr(q, "lastTradeDateOrContractMonth", "") or instrument.contract_expiry,
        )
        self._cache[cache_key] = (q, spec, time.monotonic())
        logger.debug(
            "Resolved %s %s %s -> conId=%s",
            instrument.sec_type,
            instrument.symbol,
            instrument.contract_expiry,
            spec.con_id,
        )
        return q, spec

    def _instrument_from_template(
        self,
        symbol: str,
        template: ContractTemplate | None,
        route: ExchangeRoute | None,
        expiry: str,
    ) -> Instrument | None:
        """Build an Instrument from config templates (fallback when no Instrument provided)."""
        if template is None:
            return None
        from libs.oms.models.instrument import Instrument
        return Instrument(
            symbol=template.symbol,
            root=symbol,
            venue=(route.exchange if route else template.exchange),
            tick_size=template.tick_size,
            tick_value=template.tick_value,
            multiplier=template.multiplier,
            currency=template.currency,
            contract_expiry=expiry if template.sec_type == "FUT" else "",
            sec_type=template.sec_type,
            primary_exchange=(route.primary_exchange if route and route.primary_exchange else template.primary_exchange or ""),
            trading_class=(route.trading_class if route and route.trading_class else template.trading_class or ""),
        )

    @staticmethod
    def _build_contract(instrument: Instrument) -> Contract:
        if instrument.sec_type == "STK":
            contract = Stock(
                symbol=instrument.symbol,
                exchange=instrument.exchange,
                currency=instrument.currency,
                primaryExchange=instrument.primary_exchange or "",
            )
        elif instrument.sec_type == "FUT":
            contract = Future(
                symbol=instrument.root or instrument.symbol,
                exchange=instrument.exchange,
                currency=instrument.currency,
                lastTradeDateOrContractMonth=instrument.contract_expiry,
            )
        else:
            contract = Contract(
                symbol=instrument.symbol,
                secType=instrument.sec_type,
                exchange=instrument.exchange,
                currency=instrument.currency,
            )
            if instrument.contract_expiry:
                contract.lastTradeDateOrContractMonth = instrument.contract_expiry
            if instrument.primary_exchange:
                contract.primaryExchange = instrument.primary_exchange

        if instrument.trading_class:
            contract.tradingClass = instrument.trading_class
        return contract

    def invalidate(
        self,
        symbol: str,
        expiry: str = "",
        sec_type: str = "FUT",
        exchange: str = "",
        primary_exchange: str = "",
    ) -> None:
        """Force cache eviction (e.g. on rollover)."""
        template = self._templates.get(symbol)
        exchange_name = exchange or (template.exchange if template else "")
        cache_key = (sec_type, symbol, expiry, exchange_name, primary_exchange)
        self._cache.pop(cache_key, None)

    def clear_cache(self) -> None:
        """Clear all cached contracts."""
        self._cache.clear()
