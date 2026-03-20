# Stock Trader

Automated U.S. equity trading stack running three live strategies through Interactive Brokers. Each strategy runs as its own process/container, shares the same OMS and instrumentation stack, and sizes from an allocation-aware slice of account NAV.

## Strategies

- `IARIC`: all-day intraday accumulation-reversal strategy. IARIC looks to detect institutional accumulation near AVWAP-band discount via tick/volume sponsorship signals and bid/ask micropressure, then requires consecutive 5m closes above the reclaim level before entering — fading noise while riding confirmed institutional flow.
- `US_ORB`: opening-range breakout strategy for the early session. US_ORB captures the post-open price discovery window (9:35-9:50 AM) in high-volume-surge names, entering breakouts only when sustained volume acceptance and broad market breadth confirm directional conviction — exploiting the opening range as a reliable intraday support/resistance anchor.
- `ALCB`: AVWAP Leader Compression Breakout campaign strategy with nightly selection, 30m execution, multi-day management, long/short support, and OMS-managed stops/partials/adds. ALCB targets liquid leaders with strong higher-timeframe sponsorship that compress into tight daily structure, then enters on ATR-normalised displacement breakouts confirmed by intraday AVWAP reclaim — capturing the expansion phase that follows institutional accumulation in trending names.

## Structure

```text
stock_trader/
|- strategy_iaric/
|- strategy_orb/
|- strategy_alcb/
|- shared/
|- instrumentation/
|- config/
|- infra/
|- docs/
`- tests/
```

## Quick Start

```bash
cp .env.example .env
docker compose -f infra/docker-compose.yml up -d
```

## Capital Allocation

`STOCK_TRADER_DEPLOY_MODE=both` now means all three live strategies run together. If no allocation overrides are supplied, capital defaults to an equal-third split:

- `STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT`
- `STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT`
- `STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT`

When any of those are set in combined mode, all three must sum to `100`.

Solo launches are also supported:

- `STOCK_TRADER_DEPLOY_MODE=iaric`
- `STOCK_TRADER_DEPLOY_MODE=us_orb`
- `STOCK_TRADER_DEPLOY_MODE=alcb`

In solo mode, the selected strategy uses `100%` of account capital.

## Configuration

Key variables live in [.env.example](/Users/sehyu/Documents/Other/Projects/stock_trader/.env.example):

- `IB_CLIENT_ID_IARIC`
- `IB_CLIENT_ID_US_ORB`
- `IB_CLIENT_ID_ALCB`
- `STOCK_TRADER_PAPER_CAPITAL`
- `STOCK_TRADER_DEPLOY_MODE`
- `STOCK_TRADER_CAPITAL_ALLOCATION_*`
- `DB_*`
- `INSTRUMENTATION_*`

## Testing

```bash
pytest
```

See [infra/DEPLOY.md](/Users/sehyu/Documents/Other/Projects/stock_trader/infra/DEPLOY.md) for VPS deployment details.
