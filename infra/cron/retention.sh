#!/bin/bash
# Daily data retention job
# Install: crontab -e → 5 0 * * * /opt/trading/swing_trader/infra/cron/retention.sh
#
# Requires: POSTGRES_PASSWORD set in environment or loaded from .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="/var/log/trading"

# Load .env if POSTGRES_PASSWORD is not already set
if [ -z "${POSTGRES_PASSWORD:-}" ] && [ -f "$PROJECT_DIR/.env" ]; then
    export "$(grep '^POSTGRES_PASSWORD=' "$PROJECT_DIR/.env" | head -1)"
fi

mkdir -p "$LOG_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Running retention job" >> "$LOG_DIR/retention.log"

PGPASSWORD="${POSTGRES_PASSWORD}" psql \
    -h localhost \
    -U trading_admin \
    -d trading \
    -f "$PROJECT_DIR/infra/retention.sql" \
    >> "$LOG_DIR/retention.log" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') - Retention job complete" >> "$LOG_DIR/retention.log"
