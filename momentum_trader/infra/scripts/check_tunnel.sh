#!/bin/bash
# WireGuard tunnel health monitor for VPS 2 → VPS 1
# Install: crontab -e → */5 * * * * /opt/trading/momentum_trader/infra/scripts/check_tunnel.sh
set -euo pipefail

VPS1_TUNNEL_IP="10.0.0.1"
LOG="/var/log/trading/tunnel_health.log"
MAX_LOG_SIZE=1048576  # 1MB

# Rotate log if too large
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null)" -gt "$MAX_LOG_SIZE" ]; then
    mv "$LOG" "${LOG}.1"
fi

if ! ping -c 1 -W 3 "$VPS1_TUNNEL_IP" > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') WARN: Tunnel DOWN — restarting wg0" >> "$LOG"
    sudo systemctl restart wg-quick@wg0
    sleep 3

    if ping -c 1 -W 3 "$VPS1_TUNNEL_IP" > /dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') INFO: Tunnel restored after restart" >> "$LOG"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Tunnel still DOWN after restart" >> "$LOG"
    fi
fi
