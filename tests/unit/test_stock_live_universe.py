from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import (
    BACKTESTED_INTRADAY_STOCK_SYMBOLS,
    DOW_JONES_SYMBOLS,
    LIVE_STOCK_UNIVERSE,
    LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS,
    LIVE_STOCK_UNIVERSE_SYMBOLS,
    NASDAQ_100_SYMBOLS,
)


def test_live_stock_universe_is_backtested_plus_nasdaq100_and_dow() -> None:
    symbols = list(LIVE_STOCK_UNIVERSE_SYMBOLS)
    backtested = set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    index_symbols = set(NASDAQ_100_SYMBOLS) | set(DOW_JONES_SYMBOLS)

    assert symbols[: len(BACKTESTED_INTRADAY_STOCK_SYMBOLS)] == list(
        BACKTESTED_INTRADAY_STOCK_SYMBOLS
    )
    assert set(NASDAQ_100_SYMBOLS).issubset(symbols)
    assert set(DOW_JONES_SYMBOLS).issubset(symbols)
    assert set(symbols) - backtested <= index_symbols
    assert LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS == tuple(
        symbol for symbol in symbols if symbol not in backtested
    )


def test_live_stock_universe_is_not_the_broad_sp500_list() -> None:
    symbols = [symbol for symbol, _, _ in LIVE_STOCK_UNIVERSE]

    assert len(symbols) == len(set(symbols)) == 173
    assert len(symbols) < len(SP500_CONSTITUENTS)
    assert len(LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS) == 75
    assert all(sector and primary_exchange for _, sector, primary_exchange in LIVE_STOCK_UNIVERSE)


def test_live_stock_universe_prefers_current_index_exchanges() -> None:
    metadata = {
        symbol: (sector, primary_exchange)
        for symbol, sector, primary_exchange in LIVE_STOCK_UNIVERSE
    }

    assert all(metadata[symbol][1] == "NASDAQ" for symbol in NASDAQ_100_SYMBOLS)
    assert metadata["WMT"][1] == "NASDAQ"
    assert metadata["MMM"][1] == "NYSE"
