from __future__ import annotations

import pytest

from libs.broker_ibkr.config.schemas import ContractTemplate, ExchangeRoute
from libs.broker_ibkr.mapping.contract_factory import ContractFactory
from libs.oms.models.instrument import Instrument


class _FakeIB:
    async def qualifyContractsAsync(self, contract):
        contract.conId = 123456
        contract.symbol = "IB1T"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "GBP"
        contract.primaryExchange = "LSEETF"
        contract.tradingClass = "IB1T"
        return [contract]


@pytest.mark.asyncio
async def test_contract_factory_resolves_logical_ibit_to_broker_ib1t():
    factory = ContractFactory(
        ib=_FakeIB(),
        templates={
            "IBIT": ContractTemplate(
                symbol="IB1T",
                sec_type="STK",
                exchange="SMART",
                currency="GBP",
                multiplier=1.0,
                tick_size=0.01,
                tick_value=0.01,
                primary_exchange="LSEETF",
            )
        },
        routes={
            "IBIT": ExchangeRoute(
                root_symbol="IBIT",
                exchange="SMART",
                primary_exchange="LSEETF",
            )
        },
    )
    logical_instrument = Instrument(
        symbol="IBIT",
        root="IBIT",
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )

    contract, spec = await factory.resolve("IBIT", instrument=logical_instrument)

    assert contract.symbol == "IB1T"
    assert contract.currency == "GBP"
    assert getattr(contract, "primaryExchange", "") == "LSEETF"
    assert spec.symbol == "IB1T"
    assert spec.currency == "GBP"
    assert spec.primary_exchange == "LSEETF"
    assert factory.logical_symbol_for_contract(contract) == "IBIT"
