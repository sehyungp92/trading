"""Futures contract generation and roll scheduling for IBKR downloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


QUARTER_MONTHS: tuple[tuple[int, str], ...] = (
    (3, "H"),
    (6, "M"),
    (9, "U"),
    (12, "Z"),
)


@dataclass(frozen=True)
class FutureRootSpec:
    symbol: str
    exchange: str = "CME"
    trading_class: str | None = None
    currency: str = "USD"
    tick_size: float = 0.25
    point_value: float = 1.0

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


FUTURE_ROOTS: dict[str, FutureRootSpec] = {
    "NQ": FutureRootSpec("NQ", exchange="CME", trading_class="NQ", tick_size=0.25, point_value=20.0),
    "MNQ": FutureRootSpec("MNQ", exchange="CME", trading_class="MNQ", tick_size=0.25, point_value=2.0),
    "ES": FutureRootSpec("ES", exchange="CME", trading_class="ES", tick_size=0.25, point_value=50.0),
}


@dataclass(frozen=True)
class FuturesContractSpec:
    symbol: str
    yyyymm: str
    expiry: date
    roll_date: date
    code: str
    local_symbol: str
    exchange: str = "CME"
    trading_class: str | None = None
    currency: str = "USD"
    tick_size: float = 0.25

    @property
    def ib_trading_class(self) -> str:
        return self.trading_class or self.symbol


def root_spec(symbol: str) -> FutureRootSpec:
    normalized = symbol.upper()
    return FUTURE_ROOTS.get(normalized, FutureRootSpec(normalized, trading_class=normalized))


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday, weeks=2)


def make_contract_spec(
    symbol: str,
    year: int,
    month: int,
    *,
    roll_days_before_expiry: int = 4,
) -> FuturesContractSpec:
    root = root_spec(symbol)
    code = dict(QUARTER_MONTHS)[month]
    expiry = third_friday(year, month)
    roll_date = expiry - timedelta(days=roll_days_before_expiry)
    return FuturesContractSpec(
        symbol=root.symbol,
        yyyymm=f"{year}{month:02d}",
        expiry=expiry,
        roll_date=roll_date,
        code=code,
        local_symbol=f"{root.symbol}{code}{year % 10}",
        exchange=root.exchange,
        trading_class=root.ib_trading_class,
        currency=root.currency,
        tick_size=root.tick_size,
    )


def generate_quarterly_contracts(
    symbol: str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    years: int = 2,
    as_of: date | datetime | None = None,
    include_buffer_contracts: bool = True,
) -> list[FuturesContractSpec]:
    """Return quarterly futures contracts whose active windows overlap a date span.

    The default span is the latest ``years`` back from ``as_of``. Roll date uses
    the reference scripts' conservative Monday-of-expiry-week convention.
    """
    if as_of is None:
        as_of_date = datetime.now(timezone.utc).date()
    elif isinstance(as_of, datetime):
        as_of_date = as_of.date()
    else:
        as_of_date = as_of

    if end is None:
        end_date = as_of_date
    elif isinstance(end, datetime):
        end_date = end.date()
    else:
        end_date = end

    if start is None:
        start_date = end_date - timedelta(days=365 * years)
    elif isinstance(start, datetime):
        start_date = start.date()
    else:
        start_date = start

    all_contracts: list[FuturesContractSpec] = []
    for year in range(start_date.year - 1, end_date.year + 2):
        for month, _code in QUARTER_MONTHS:
            all_contracts.append(make_contract_spec(symbol, year, month))
    all_contracts.sort(key=lambda contract: contract.expiry)

    relevant: list[FuturesContractSpec] = []
    for idx, contract in enumerate(all_contracts):
        previous_roll = all_contracts[idx - 1].roll_date if idx > 0 else date(1900, 1, 1)
        active_start = previous_roll
        active_end = contract.roll_date
        if active_end >= start_date and active_start <= end_date:
            relevant.append(contract)

    if include_buffer_contracts and relevant:
        first_idx = all_contracts.index(relevant[0])
        last_idx = all_contracts.index(relevant[-1])
        buffer_start = max(0, first_idx - 1)
        buffer_end = min(len(all_contracts), last_idx + 2)
        relevant = all_contracts[buffer_start:buffer_end]

    return relevant


def roll_schedule(
    contracts: list[FuturesContractSpec],
) -> list[tuple[date, str, str]]:
    ordered = sorted(contracts, key=lambda contract: contract.expiry)
    return [
        (old.roll_date, old.yyyymm, new.yyyymm)
        for old, new in zip(ordered, ordered[1:])
    ]


def active_contract(
    contracts: list[FuturesContractSpec],
    *,
    as_of: date | datetime | None = None,
) -> FuturesContractSpec | None:
    if not contracts:
        return None
    if as_of is None:
        as_of_date = datetime.now(timezone.utc).date()
    elif isinstance(as_of, datetime):
        as_of_date = as_of.date()
    else:
        as_of_date = as_of
    ordered = sorted(contracts, key=lambda contract: contract.roll_date)
    for contract in ordered:
        if as_of_date < contract.roll_date:
            return contract
    return ordered[-1]

