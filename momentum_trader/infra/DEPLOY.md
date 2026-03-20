# Deploying momentum_trader

Deploy the 3 momentum strategy containers (Helix, NQDTC, Vdubus) on an existing swing_trader VPS, joining the shared infrastructure (PostgreSQL, dashboard, IB Gateway, relay) already provisioned by swing_trader.

## Architecture

```text
DigitalOcean VPS (shared with swing_trader, stock_trader)
├── IB Gateway (host, port 4002) ← swing_trader managed
│   └── via IBC 3.19.0 + Xvfb (headless), systemd service
├── Trading Relay (host, port 8001) ← swing_trader managed
│   └── FastAPI + SQLite event buffer, HMAC auth
├── Docker
│   ├── trading_postgres (shared PostgreSQL) ← swing_trader managed
│   ├── trading_dashboard (port 3000) ← swing_trader managed
│   ├── trading_helix (this repo, network_mode: host)
│   ├── trading_nqdtc (this repo, network_mode: host)
│   └── trading_vdubus (this repo, network_mode: host)
```

Containers use `network_mode: host` — they share the host's network stack directly and reach all services at `127.0.0.1`. No Docker bridge networking, no `trading_net`, no `extra_hosts` needed.

## Prerequisites

- VPS running the swing_trader stack (Postgres, dashboard, IB Gateway, relay)
- `momentum_trader` registered in the relay's secrets (see Step 2 below)
- Docker and Docker Compose installed

Verify shared infrastructure:

```bash
ss -tlnp | grep 4002                                                          # IB Gateway
docker exec trading_postgres pg_isready -U trading_writer -d trading           # PostgreSQL
curl -s http://127.0.0.1:8001/health                                           # Relay
curl -s http://localhost:3000/api/health                                        # Dashboard
```

---

## Step 1 — Clone the Repository

```bash
mkdir -p /opt/trading
cd /opt/trading
git clone <YOUR_REPO_URL> momentum_trader
cd momentum_trader
```

## Step 2 — Register HMAC Secret with Relay

Generate a shared secret for instrumentation forwarding:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Add `momentum_trader` to the relay's secrets file:

```bash
# Edit /opt/trading-relay/secrets.json
# Add alongside existing entries:
{
    "swing_trader": "<existing>",
    "stock_trader": "<existing>",
    "momentum_trader": "<secret-from-above>"
}
```

Restart the relay:

```bash
sudo systemctl restart trading-relay
```

## Step 3 — Configure Environment

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Key values:

| Variable | Value | Notes |
|----------|-------|-------|
| `ALGO_TRADER_ENV` | `paper` | Start with paper, switch to `live` later |
| `DB_HOST` | `127.0.0.1` | localhost (network_mode: host) |
| `DB_PASSWORD` | (match swing_trader) | Same as swing_trader's `trading_writer` password |
| `IB_HOST` | `127.0.0.1` | localhost (network_mode: host) |
| `IB_PORT` | `4002` | Paper: 4002, Live: 4001 |
| `IB_ACCOUNT_ID` | `DU1234567` | Your IBKR account |
| `TRADING_SYMBOL` | `MNQ` | Micro E-mini Nasdaq-100 |
| `INSTRUMENTATION_HMAC_SECRET` | (from Step 2) | Must match relay's secrets.json |
| `INSTRUMENTATION_RELAY_URL` | `http://127.0.0.1:8001/events` | Relay on localhost |

## Step 4 — Build and Start

```bash
cd /opt/trading/momentum_trader

# Build all strategy images
docker compose -f infra/docker-compose.yml \
  --profile helix --profile nqdtc --profile vdubus build

# Start all strategies
docker compose -f infra/docker-compose.yml \
  --profile helix --profile nqdtc --profile vdubus up -d
```

Start specific strategies only:

```bash
docker compose -f infra/docker-compose.yml --profile helix up -d               # Helix only
docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc up -d  # Helix + NQDTC
```

## Step 5 — Verify

Run the verification script:

```bash
bash infra/scripts/verify_deployment.sh
```

Or verify manually:

```bash
# Check containers are running
docker compose -f infra/docker-compose.yml \
  --profile helix --profile nqdtc --profile vdubus ps

# Check strategy logs (look for: DB connected, IB connected, engine started)
docker logs trading_helix --tail 20
docker logs trading_nqdtc --tail 20
docker logs trading_vdubus --tail 20

# Verify localhost connectivity
python3 -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',5432)); print('DB OK'); s.close()"
python3 -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',4002)); print('IB OK'); s.close()"
curl -sf http://127.0.0.1:8001/health && echo "Relay OK"
```

Check the dashboard — all three momentum strategies should appear within 1 minute.

---

## Common Operations

| Action | Command |
|--------|---------|
| Restart a strategy | `docker compose -f infra/docker-compose.yml --profile helix restart helix` |
| Stop all | `docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc --profile vdubus down` |
| Start all | `docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc --profile vdubus up -d` |
| View logs | `docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc --profile vdubus logs -f --tail=100` |
| Rebuild | `git pull && docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc --profile vdubus build && docker compose -f infra/docker-compose.yml --profile helix --profile nqdtc --profile vdubus up -d` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| IB Gateway connection refused | Verify IB Gateway is running: `sudo systemctl status ibgateway` and `ss -tlnp \| grep 4002` |
| PostgreSQL connection refused | Verify postgres is running: `docker ps \| grep postgres`. Check `DB_HOST=127.0.0.1` in `.env`. |
| IB client ID conflict | Each strategy needs a unique ID. ETF: 1-10, momentum: 11-13, stock: 31-32. Check for collisions with `ss -tnp \| grep 4002`. |
| Container exits immediately | Check exit code: `docker inspect trading_helix --format='{{.State.ExitCode}}'`. 0=clean shutdown, 1=app error, 137=OOM, 139=segfault. Check logs. |
| Relay forwarding fails | Verify HMAC secret matches `secrets.json`. Test: `curl -sf http://127.0.0.1:8001/health`. |
| Container won't connect after IB Gateway restart | IB Gateway resets ~midnight ET. Containers have `restart: unless-stopped` and will reconnect automatically. If stuck, restart: `docker restart trading_helix`. |
| `RuntimeError: HMAC secret` | HMAC secret missing or empty in `.env`. Required in paper/live mode. |

---

## Shared VPS Maintenance (managed by swing_trader)

These cron jobs benefit momentum_trader:

| Job | Schedule | What it does |
|-----|----------|--------------|
| DB backup | 01:00 UTC daily | `pg_dump` → `/opt/trading/backups/`, 30-day retention |
| Data retention | 00:05 UTC daily | Deletes old `order_events`, resets daily counters, `VACUUM ANALYZE` |
| Log rotation | Daily | `/etc/logrotate.d/trading`, 30 days, compressed |
| Relay purge | Daily | Purges acked events from relay SQLite buffer |
| IB Gateway reset | ~00:00 ET | IBC `AutoRestartTime=00:00` handles IBKR's daily disconnect |

---

## Key Files

| File | Purpose |
|------|---------|
| `.env` | All environment variables (secrets, connection strings) |
| `Dockerfile` | Shared image for all 3 strategies |
| `infra/docker-compose.yml` | Strategy container orchestration |
| `infra/scripts/verify_deployment.sh` | Post-deploy smoke test |
| `config/contracts.yaml` | Futures contract specs |
| `config/ibkr_profiles.yaml` | IBKR connection profiles |
| `docs/implementation.md` | Full implementation plan |
