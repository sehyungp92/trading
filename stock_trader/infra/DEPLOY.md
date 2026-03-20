# Deploying `stock_trader` On The Swing-Trader VPS

This repo joins the existing shared VPS stack and now runs three strategy containers:

- `IARIC_v1` via `python -m strategy_iaric`
- `US_ORB_v1` via `python -m strategy_orb`
- `ALCB_v1` via `python -m strategy_alcb`

## Target topology

```text
Ubuntu VPS (all services reachable on 127.0.0.1 via network_mode: host)
|- IB Gateway on host :4002
|- PostgreSQL on host :5432
|- Relay on host :8001
|- Docker
|  |- stock_trader_strategy_iaric
|  |- stock_trader_strategy_orb
|  `- stock_trader_strategy_alcb
```

## 1. Prepare the repo

```bash
cd /opt/trading
git clone <YOUR_REPO_URL> stock_trader
cd stock_trader
cp .env.example .env
chmod 600 .env
```

Fill in:

- `ALGO_TRADER_ENV`
- `DB_*`
- `IB_HOST`, `IB_PORT`, `IB_ACCOUNT_ID`
- `IB_CLIENT_ID_IARIC`
- `IB_CLIENT_ID_US_ORB`
- `IB_CLIENT_ID_ALCB`
- `STOCK_TRADER_DEPLOY_MODE`
- `STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT`
- `STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT`
- `STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT`
- `INSTRUMENTATION_HMAC_SECRET`

## 2. Capital allocation rules

`STOCK_TRADER_DEPLOY_MODE=both` means all three strategies run together.

If no allocation overrides are set, the runtime defaults to an equal-third split:

- `33.333333 / 33.333333 / 33.333334`

If you override combined-mode allocations, the three variables must sum to `100`.

Solo modes:

- `iaric`
- `us_orb`
- `alcb`

Solo mode gives the selected strategy `100%` of account capital.

## 3. Build and launch

```bash
docker compose -f infra/docker-compose.yml build
docker compose -f infra/docker-compose.yml up -d
```

## 4. Start a single strategy

```bash
STOCK_TRADER_DEPLOY_MODE=iaric docker compose -f infra/docker-compose.yml up -d strategy_iaric
STOCK_TRADER_DEPLOY_MODE=us_orb docker compose -f infra/docker-compose.yml up -d strategy_orb
STOCK_TRADER_DEPLOY_MODE=alcb docker compose -f infra/docker-compose.yml up -d strategy_alcb
```

## 5. Verify

```bash
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml logs -f strategy_iaric
docker compose -f infra/docker-compose.yml logs -f strategy_orb
docker compose -f infra/docker-compose.yml logs -f strategy_alcb
```

Expected signals:

- Heartbeats appear in the shared dashboard.
- Strategy data persists under `/app/data/strategy_<name>`.
- Instrumentation data persists under `/app/instrumentation/data`.
- Signed events reach the shared relay.

## Operations

| Action | Command |
| --- | --- |
| Restart `IARIC_v1` | `docker compose -f infra/docker-compose.yml restart strategy_iaric` |
| Restart `US_ORB_v1` | `docker compose -f infra/docker-compose.yml restart strategy_orb` |
| Restart `ALCB_v1` | `docker compose -f infra/docker-compose.yml restart strategy_alcb` |
| Stop all | `docker compose -f infra/docker-compose.yml down` |
| Rebuild after updates | `git pull && docker compose -f infra/docker-compose.yml build && docker compose -f infra/docker-compose.yml up -d` |
