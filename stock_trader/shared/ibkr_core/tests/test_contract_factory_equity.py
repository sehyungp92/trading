import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ib_async import Contract

from shared.ibkr_core.config.schemas import ContractTemplate, ExchangeRoute
from shared.ibkr_core.mapping.contract_factory import ContractFactory
from shared.oms.models.instrument import Instrument


class _FakeIB:
    async def qualifyContractsAsync(self, contract):
        contract.conId = 101
        if getattr(contract, "secType", "") == "STK":
            contract.primaryExchange = getattr(contract, "primaryExchange", "") or "NASDAQ"
        else:
            contract.tradingClass = getattr(contract, "tradingClass", "") or "MNQ"
            contract.multiplier = getattr(contract, "multiplier", "") or "2"
        return [contract]


def test_contract_factory_resolves_dynamic_stock_instrument():
    factory = ContractFactory(ib=_FakeIB(), templates={}, routes={})
    instrument = Instrument(
        symbol="AAPL",
        root="AAPL",
        venue="SMART",
        primary_exchange="NASDAQ",
        sec_type="STK",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
    )

    contract, spec = __import__("asyncio").run(
        factory.resolve(symbol=instrument.root, instrument=instrument)
    )

    assert contract.secType == "STK"
    assert contract.symbol == "AAPL"
    assert spec.symbol == "AAPL"
    assert spec.primary_exchange == "NASDAQ"
    assert spec.last_trade_date == ""


def test_contract_factory_resolves_template_future():
    factory = ContractFactory(
        ib=_FakeIB(),
        templates={
            "MNQ": ContractTemplate(
                symbol="MNQ",
                sec_type="FUT",
                exchange="CME",
                currency="USD",
                multiplier=2.0,
                tick_size=0.25,
                tick_value=0.5,
                trading_class="MNQ",
            )
        },
        routes={"MNQ": ExchangeRoute(root_symbol="MNQ", exchange="CME", trading_class="MNQ")},
    )

    contract, spec = __import__("asyncio").run(factory.resolve("MNQ", "202606"))

    assert contract.secType == "FUT"
    assert contract.lastTradeDateOrContractMonth == "202606"
    assert spec.trading_class == "MNQ"
    assert spec.last_trade_date == "202606"
