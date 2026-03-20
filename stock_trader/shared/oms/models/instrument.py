"""Instrument definition."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str  # e.g. "MNQ" or "AAPL"
    root: str  # e.g. "MNQ" or "AAPL"
    venue: str  # e.g. "CME", "SMART"
    tick_size: float
    tick_value: float  # tick_size * multiplier
    multiplier: float
    currency: str = "USD"
    point_value: float = 0.0  # computed: multiplier
    contract_expiry: str = ""  # YYYYMM or YYYYMMDD for futures
    sec_type: str = "FUT"
    primary_exchange: str = ""
    trading_class: str = ""

    def __post_init__(self):
        if self.point_value == 0.0:
            object.__setattr__(self, "point_value", self.multiplier)
        if not self.root:
            object.__setattr__(self, "root", self.symbol)
        if self.sec_type == "STK":
            object.__setattr__(self, "contract_expiry", "")

    @property
    def exchange(self) -> str:
        return self.venue
