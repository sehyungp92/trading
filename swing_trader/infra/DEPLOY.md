# Deploying swing_trader on a VPS (Paper Trading Mode)

Deploy the full swing_trader stack (the `main_multi.py` portfolio launcher, PostgreSQL, and the Next.js trading dashboard) on an Ubuntu VPS connected to an IBKR paper trading account. IB Gateway runs headlessly via IBC + Xvfb.

## Architecture

```
Ubuntu VPS
├── IB Gateway (systemd service, port 4002)
│   └── via IBC + Xvfb (headless)
│
└── Docker
    ├── postgres (127.0.0.1:5432)
    ├── dashboard (port 3000)
    ├── atrss strategy ────────► IB Gateway:4002
    ├── akc_helix strategy ────► IB Gateway:4002
    └── swing_breakout strategy ► IB Gateway:4002
```

## Prerequisites

- Ubuntu 22.04 or 24.04 VPS (minimum 2 vCPU, 4 GB RAM, 40 GB disk)
- SSH access with sudo privileges
- An IBKR account with paper trading enabled
- Your IBKR paper trading credentials (username + password)

---

## Quick Deploy (Automated)

If the repo is already on the server at `/opt/trading/swing_trader`:

```bash
cd /opt/trading/swing_trader
sudo ./infra/deploy.sh
```

Then follow the "Next steps" printed at the end. For manual step-by-step setup, continue reading.

---

## Step 1 — Initial Server Setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget unzip software-properties-common ufw
sudo timedatectl set-timezone America/New_York

# Firewall
sudo ufw allow OpenSSH
sudo ufw allow 3000/tcp    # Trading dashboard (restrict to your IP later)
sudo ufw allow from 172.16.0.0/12 to any port 4002  # Docker containers → IB Gateway
sudo ufw enable
```

## Step 2 — Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit   # re-login for group change
```

Verify:
```bash
docker --version
docker compose version
```

## Step 3 — Install IB Gateway with IBC (Headless)

### 3a — Java and Xvfb

```bash
sudo apt install -y default-jre xvfb
java -version   # confirm Java 11+
```

### 3b — IB Gateway

```bash
cd /tmp
wget -O ibgateway-stable-standalone-linux-x64.sh \
  "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
chmod +x ibgateway-stable-standalone-linux-x64.sh
sudo sh ibgateway-stable-standalone-linux-x64.sh -q -dir /opt/ibgateway
```

### 3c — IBC

```bash
cd /tmp
wget https://github.com/IbcAlpha/IBC/releases/download/3.19.0/IBCLinux-3.19.0.zip
sudo mkdir -p /opt/ibc
sudo unzip IBCLinux-3.19.0.zip -d /opt/ibc
sudo chmod +x /opt/ibc/*.sh /opt/ibc/*/*.sh
```

### 3d — Configure IBC

```bash
sudo mkdir -p /opt/ibc/config
sudo cp /opt/trading/swing_trader/infra/ibc/config.ini.example /opt/ibc/config/config.ini
sudo nano /opt/ibc/config/config.ini   # set your IBKR username + password
sudo chmod 600 /opt/ibc/config/config.ini
```

### 3e — Install systemd Service

```bash
sudo cp /opt/trading/swing_trader/infra/systemd/ibgateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ibgateway
sudo systemctl start ibgateway
```

### 3f — Verify IB Gateway

Wait ~60 seconds, then:
```bash
ss -tlnp | grep 4002
```

Port 4002 should be in LISTEN state. If not:
```bash
sudo journalctl -u ibgateway --no-pager -n 50
```

### 3g — IB Gateway API Access (TrustedIPs)

IB Gateway defaults to `TrustedIPs=127.0.0.1` and overwrites `jts.ini` on every startup. The portfolio container uses `network_mode: host` to connect from `127.0.0.1`, so no manual configuration is needed.

## Step 4 — Clone the Repository

```bash
sudo mkdir -p /opt/trading
sudo chown $USER:$USER /opt/trading
cd /opt/trading
git clone <YOUR_REPO_URL> swing_trader
cd swing_trader
```

Or via scp from your local machine:
```bash
scp -r /path/to/swing_trader user@your-vps-ip:/opt/trading/swing_trader
```

## Step 5 — Configure Environment

```bash
cd /opt/trading/swing_trader
cp .env.example .env
nano .env
```

Key values to set:

| Variable | Example |
|----------|---------|
| `SWING_TRADER_ENV` | `paper` |
| `IB_ACCOUNT_ID` | `DU1234567` |
| `IB_HOST` | `host.docker.internal` |
| `IB_PORT` | `4002` |
| `POSTGRES_PASSWORD` | (strong password) |
| `POSTGRES_READER_PASSWORD` | (strong password) |
| `POSTGRES_WRITER_PASSWORD` | (strong password) |
| `ATRSS_SYMBOL_SET` | `etf` (optional) |
| `AKCHELIX_SYMBOL_SET` | `etf` (optional) |

Secure the file:
```bash
chmod 600 .env
```

## Step 6 — Start Infrastructure

```bash
cd /opt/trading/swing_trader

# Start database and dashboard
docker compose -f infra/docker-compose.yml up -d postgres dashboard

# Wait for health check
docker compose -f infra/docker-compose.yml ps

# Verify postgres is ready
docker exec trading_postgres pg_isready -U trading_admin -d trading
```

**Update database passwords** (init-db.sql uses defaults):
```bash
docker exec -it trading_postgres psql -U trading_admin -d trading -c \
  "ALTER USER trading_writer WITH PASSWORD 'your_actual_writer_password';"

docker exec -it trading_postgres psql -U trading_admin -d trading -c \
  "ALTER USER trading_reader WITH PASSWORD 'your_actual_reader_password';"
```

## Step 7 — Start Portfolio Launcher

```bash
# Build and start the portfolio launcher
docker compose -f infra/docker-compose.yml --profile portfolio build portfolio
docker compose -f infra/docker-compose.yml --profile portfolio up -d portfolio

# Verify
docker compose -f infra/docker-compose.yml --profile portfolio ps portfolio
```

This is the production deployment path that preserves the portfolio heat cap, strategy priorities, and cross-strategy coordination in `main_multi.py`.

Start specific strategies only for isolated debugging:
```bash
docker compose -f infra/docker-compose.yml --profile atrss up -d atrss                         # ATRSS only
docker compose -f infra/docker-compose.yml --profile atrss --profile swing_breakout up -d atrss swing_breakout
```

Do not run the standalone strategy profiles alongside `portfolio` in production, or you will duplicate trading logic and change portfolio behavior.

## Step 8 — Verify

### Strategy logs
```bash
docker compose -f infra/docker-compose.yml --profile portfolio logs -f portfolio
```

You should see: database bootstrap, IB Gateway connection, all five strategy engines starting inside `main_multi.py`, and heartbeat messages.

### Database
```bash
docker exec -it trading_postgres psql -U trading_admin -d trading -c \
  "SELECT * FROM strategy_state;"
```

### IB Gateway connectivity from container
```bash
docker exec -it trading_portfolio python -c \
  'import socket; s = socket.socket(); s.connect(("127.0.0.1", 4002)); print("Connected!"); s.close()'
```

## Step 9 — Trading Dashboard

1. Open `http://YOUR_VPS_IP:3000`
2. Create admin account
3. Add database: PostgreSQL, host `postgres`, port 5432, database `trading`, user `trading_reader`
4. Confirm the dashboard populates portfolio, strategy, position, and order data
5. Set auto-refresh to 30 seconds

## Step 10 — Cron Job

```bash
sudo mkdir -p /var/log/trading
sudo chown $USER:$USER /var/log/trading
chmod +x /opt/trading/swing_trader/infra/cron/retention.sh

# Add to crontab (daily 00:05 UTC)
(crontab -l 2>/dev/null; echo "5 0 * * * /opt/trading/swing_trader/infra/cron/retention.sh") | crontab -
```

## Step 11 — Secure the VPS

```bash
# Restrict the trading dashboard to your IP
sudo ufw delete allow 3000/tcp
sudo ufw allow from YOUR_IP to any port 3000

# Postgres is already bound to 127.0.0.1 (docker-compose.yml)

# Verify file permissions
chmod 600 /opt/trading/swing_trader/.env
sudo chmod 600 /opt/ibc/config/config.ini
```

---

## Common Operations

| Action | Command |
|--------|---------|
| Restart portfolio launcher | `docker compose -f infra/docker-compose.yml --profile portfolio restart portfolio` |
| Stop portfolio launcher | `docker compose -f infra/docker-compose.yml stop portfolio` |
| Stop everything | `docker compose -f infra/docker-compose.yml down && sudo systemctl stop ibgateway` |
| Start everything | `sudo systemctl start ibgateway && sleep 60 && docker compose -f infra/docker-compose.yml up -d postgres dashboard && docker compose -f infra/docker-compose.yml --profile portfolio up -d portfolio` |
| View portfolio logs | `docker compose -f infra/docker-compose.yml --profile portfolio logs -f --tail=100 portfolio` |
| Rebuild after code changes | `git pull && docker compose -f infra/docker-compose.yml --profile portfolio build portfolio && docker compose -f infra/docker-compose.yml --profile portfolio up -d portfolio` |
| Check IB Gateway | `sudo systemctl status ibgateway` / `sudo journalctl -u ibgateway --no-pager -n 30` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Strategy can't connect to IB Gateway | Check `ss -tlnp \| grep 4002`. Verify `ibgateway` service is running. `extra_hosts` is set in docker-compose.yml. |
| IB Gateway won't start | Check `java -version`. Check `journalctl -u ibgateway`. Verify credentials in `/opt/ibc/config/config.ini`. |
| Database connection refused | Check `docker compose -f infra/docker-compose.yml ps` — postgres must show "healthy". Verify passwords match `.env`. |
| Dashboard shows no data | Verify `postgres` is healthy and the portfolio launcher has started writing heartbeats to `strategy_state`. |
| IB Gateway disconnects overnight | Expected — IBKR resets daily ~midnight ET. `AutoRestartTime=00:00` in IBC config handles reconnection. Strategies have `restart: unless-stopped`. |
| "No security definition found" | Market may be closed. Paper trading data is delayed 15 min and may not be available outside market hours. |

---

## Key Files

| File | Purpose |
|------|---------|
| `.env` | All environment variables |
| `infra/docker-compose.yml` | Service orchestration |
| `infra/init-db.sql` | Database initialization |
| `infra/retention.sql` | Daily data cleanup |
| `infra/cron/retention.sh` | Cron job for retention |
| `infra/ibc/config.ini.example` | IBC configuration template |
| `infra/systemd/ibgateway.service` | systemd unit for IB Gateway |
| `infra/deploy.sh` | Automated deployment script |
| `infra/metabase-setup.md` | Dashboard panel definitions |
| `config/contracts.yaml` | Futures contract specs |
| `config/ibkr_profiles.yaml` | IBKR connection profiles |
