#!/usr/bin/env bash
# Daily pre-market runtime restart.
# Ensures fresh artifacts are generated for the new trading day.
# Install: crontab -e -> 30 10 * * 1-5 /opt/trading/monorepo/infra/cron/premarket_restart.sh
# (10:30 UTC = 06:30 ET during EDT, 05:30 ET during EST)
set -uo pipefail

LOG="/var/log/trading/premarket_restart.log"
mkdir -p "$(dirname "$LOG")"

{
  echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') ---"
  cd /opt/trading/monorepo
  if docker compose restart runtime; then
    echo "restart OK"
  else
    echo "restart FAILED (exit=$?)"
  fi
} >> "$LOG" 2>&1
