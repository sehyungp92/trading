#!/bin/bash
# swing_trader VPS deployment script
# Tested on Ubuntu 22.04 / 24.04
#
# Usage:
#   chmod +x infra/deploy.sh
#   sudo ./infra/deploy.sh
#
# This script:
#   1. Updates the system and installs dependencies
#   2. Installs Docker and Docker Compose
#   3. Installs Java and Xvfb (for headless IB Gateway)
#   4. Downloads and installs IB Gateway (stable)
#   5. Downloads and installs IBC (IB Controller)
#   6. Installs the systemd service for IB Gateway
#   7. Configures the firewall
#   8. Sets up the cron job for data retention
#
# After running this script, you must:
#   - Edit /opt/ibc/config/config.ini with your IBKR credentials
#   - Edit /opt/trading/swing_trader/.env with your configuration
#   - Start services (see infra/DEPLOY.md for details)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
IBC_VERSION="3.19.0"
PROJECT_DIR="/opt/trading/swing_trader"

# ── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Check root ───────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    err "This script must be run as root (use sudo)"
fi

echo ""
echo "============================================="
echo "  swing_trader VPS Deployment"
echo "============================================="
echo ""

# ── Step 1: System update and essentials ─────────────────────
log "Updating system packages..."
apt update && apt upgrade -y

log "Installing essential packages..."
apt install -y git curl wget unzip software-properties-common ufw

log "Setting timezone to America/New_York..."
timedatectl set-timezone America/New_York

# ── Step 2: Install Docker ───────────────────────────────────
if command -v docker &>/dev/null; then
    log "Docker already installed: $(docker --version)"
else
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    log "Docker installed: $(docker --version)"
fi

# Add the invoking user to docker group if running via sudo
if [ -n "${SUDO_USER:-}" ]; then
    usermod -aG docker "$SUDO_USER"
    log "Added $SUDO_USER to docker group (re-login required)"
fi

# ── Step 3: Install Java and Xvfb ───────────────────────────
log "Installing Java and Xvfb..."
apt install -y default-jre xvfb
log "Java version: $(java -version 2>&1 | head -1)"

# ── Step 4: Install IB Gateway ───────────────────────────────
if [ -d /opt/ibgateway ]; then
    log "IB Gateway already installed at /opt/ibgateway"
else
    log "Downloading IB Gateway (stable)..."
    cd /tmp
    wget -q -O ibgateway-stable-standalone-linux-x64.sh \
        "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"

    log "Installing IB Gateway to /opt/ibgateway..."
    chmod +x ibgateway-stable-standalone-linux-x64.sh
    sh ibgateway-stable-standalone-linux-x64.sh -q -dir /opt/ibgateway
    log "IB Gateway installed"
fi

# ── Step 5: Install IBC ─────────────────────────────────────
if [ -d /opt/ibc ] && [ -f /opt/ibc/gatewaystart.sh ]; then
    log "IBC already installed at /opt/ibc"
else
    log "Downloading IBC v${IBC_VERSION}..."
    cd /tmp
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"

    log "Installing IBC to /opt/ibc..."
    mkdir -p /opt/ibc
    unzip -o "IBCLinux-${IBC_VERSION}.zip" -d /opt/ibc
    chmod +x /opt/ibc/*.sh /opt/ibc/*/*.sh 2>/dev/null || true
    log "IBC installed"
fi

# ── Step 6: Configure IBC ───────────────────────────────────
mkdir -p /opt/ibc/config
if [ -f /opt/ibc/config/config.ini ]; then
    warn "IBC config already exists at /opt/ibc/config/config.ini — skipping"
else
    if [ -f "$PROJECT_DIR/infra/ibc/config.ini.example" ]; then
        cp "$PROJECT_DIR/infra/ibc/config.ini.example" /opt/ibc/config/config.ini
        log "IBC config template copied to /opt/ibc/config/config.ini"
        warn "EDIT /opt/ibc/config/config.ini with your IBKR credentials"
    else
        warn "IBC config template not found — copy manually from infra/ibc/config.ini.example"
    fi
fi
chmod 600 /opt/ibc/config/config.ini 2>/dev/null || true

# ── Step 7: Install systemd service ─────────────────────────
log "Installing ibgateway systemd service..."
if [ -f "$PROJECT_DIR/infra/systemd/ibgateway.service" ]; then
    cp "$PROJECT_DIR/infra/systemd/ibgateway.service" /etc/systemd/system/ibgateway.service
else
    warn "systemd service file not found at $PROJECT_DIR/infra/systemd/ibgateway.service"
    warn "Copy it manually: sudo cp infra/systemd/ibgateway.service /etc/systemd/system/"
fi
systemctl daemon-reload
systemctl enable ibgateway
log "ibgateway service enabled (not started — configure IBC first)"

# ── Step 8: Configure firewall ──────────────────────────────
log "Configuring firewall..."
ufw allow OpenSSH
ufw allow 3000/tcp comment 'Trading dashboard'
ufw allow from 172.16.0.0/12 to any port 4002 comment 'Docker containers to IB Gateway'
ufw --force enable
log "Firewall enabled (SSH + trading dashboard port 3000 + Docker→IB Gateway)"
warn "Restrict port 3000 to your IP later: sudo ufw delete allow 3000/tcp && sudo ufw allow from YOUR_IP to any port 3000"

# ── Step 9: Create directories and set permissions ───────────
mkdir -p /opt/trading
mkdir -p /var/log/trading

if [ -n "${SUDO_USER:-}" ]; then
    chown "$SUDO_USER":"$SUDO_USER" /opt/trading
    chown "$SUDO_USER":"$SUDO_USER" /var/log/trading
fi

# ── Step 10: Set up retention cron job ───────────────────────
if [ -f "$PROJECT_DIR/infra/cron/retention.sh" ]; then
    chmod +x "$PROJECT_DIR/infra/cron/retention.sh"
    # Install cron job for the invoking user (or root)
    CRON_USER="${SUDO_USER:-root}"
    CRON_LINE="5 0 * * * $PROJECT_DIR/infra/cron/retention.sh"
    if crontab -u "$CRON_USER" -l 2>/dev/null | grep -qF "retention.sh"; then
        log "Retention cron job already installed for $CRON_USER"
    else
        (crontab -u "$CRON_USER" -l 2>/dev/null; echo "$CRON_LINE") | crontab -u "$CRON_USER" -
        log "Retention cron job installed (daily at 00:05 UTC) for $CRON_USER"
    fi
else
    warn "Retention script not found — skipping cron job setup"
fi

# ── Step 11: Secure .env if it exists ────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    chmod 600 "$PROJECT_DIR/.env"
    log "Secured $PROJECT_DIR/.env (chmod 600)"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Deployment setup complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Edit IBKR credentials:  sudo nano /opt/ibc/config/config.ini"
echo "  2. Edit environment vars:  nano $PROJECT_DIR/.env"
echo "  3. Start IB Gateway:       sudo systemctl start ibgateway"
echo "  4. Wait 60s, then verify:  ss -tlnp | grep 4002"
echo "  5. Start infrastructure:   cd $PROJECT_DIR && docker compose -f infra/docker-compose.yml up -d postgres dashboard"
echo "  6. Start portfolio:        docker compose -f infra/docker-compose.yml --profile portfolio build portfolio && docker compose -f infra/docker-compose.yml --profile portfolio up -d portfolio"
echo ""
echo "See $PROJECT_DIR/infra/DEPLOY.md for full documentation."
echo ""
