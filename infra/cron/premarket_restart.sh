#!/usr/bin/env bash
# Daily pre-market restart of IB Gateway + runtime.
# Ensures fresh HMDS connections and artifact generation for the new trading day.
# Install: crontab -e -> 30 10 * * 1-5 /opt/trading/monorepo/infra/cron/premarket_restart.sh
# (10:30 UTC = 06:30 ET during EDT, 05:30 ET during EST)
set -uo pipefail

LOG="/var/log/trading/premarket_restart.log"
mkdir -p "$(dirname "$LOG")"

{
  echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') ---"
  cd /opt/trading/monorepo

  # Restart IB Gateway first to get fresh HMDS connections
  if docker compose restart ib-gateway; then
    echo "ib-gateway restart OK"
  else
    echo "ib-gateway restart FAILED (exit=$?)"
  fi

  # Wait for Gateway health check (TCP port ready)
  echo "Waiting 20s for Gateway to reconnect to data farms..."
  sleep 20

  # Generate stock artifacts (IARIC + ALCB) as a one-shot container
  # before restarting runtime, so startup is not blocked for 40-50 min.
  # Timeout prevents infinite hang if IB is unresponsive (normal run ~50 min).
  echo "Generating stock artifacts..."
  if timeout 3600 docker compose run --rm runtime python -m strategies.stock.artifact_generator; then
    echo "Artifact generation OK"
  else
    echo "Artifact generation FAILED (exit=$?) -- stock family will be skipped until artifacts are regenerated"
  fi

  # Restart runtime (loads pre-generated artifacts from shared volume)
  if docker compose restart runtime; then
    echo "runtime restart OK"
  else
    echo "runtime restart FAILED (exit=$?)"
  fi
} >> "$LOG" 2>&1
