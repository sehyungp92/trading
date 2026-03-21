#!/bin/bash
set -euo pipefail
cd /opt/trading-relay
source venv/bin/activate
export RELAY_SECRETS_FILE="/opt/trading-relay/secrets.json"
export RELAY_DB_PATH="/opt/trading-relay/data/relay.db"

mkdir -p /opt/trading-relay/data

exec uvicorn run_relay:app \
  --host 127.0.0.1 \
  --port 8001 \
  --workers 1 \
  --log-level info
