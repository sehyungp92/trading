#!/bin/bash
# Post-deployment verification for momentum_trader (VPS 2)
# Checks: tunnel, DB, IB Gateway, relay, containers, heartbeats
set -euo pipefail

VPS1="10.0.0.1"
PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [OK]  $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== momentum_trader Deployment Verification ==="
echo ""

echo "1. WireGuard Tunnel"
check "Ping VPS 1 ($VPS1)" ping -c 1 -W 3 "$VPS1"
check "WireGuard interface active" sudo wg show wg0

echo ""
echo "2. VPS 1 Services"
check "PostgreSQL reachable ($VPS1:5432)" bash -c "echo > /dev/tcp/$VPS1/5432"
check "IB Gateway reachable ($VPS1:4002)" bash -c "echo > /dev/tcp/$VPS1/4002"
check "Relay health ($VPS1:8001)" curl -sf "http://$VPS1:8001/health"

echo ""
echo "3. Strategy Containers"
for name in trading_helix trading_nqdtc trading_vdubus; do
    if docker inspect "$name" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
        echo "  [OK]  $name running"
        PASS=$((PASS + 1))
    elif docker inspect "$name" > /dev/null 2>&1; then
        exit_code=$(docker inspect "$name" --format='{{.State.ExitCode}}')
        echo "  [FAIL] $name stopped (exit code: $exit_code)"
        FAIL=$((FAIL + 1))
    else
        echo "  [SKIP] $name not deployed"
    fi
done

echo ""
echo "4. Container Logs (last errors)"
for name in trading_helix trading_nqdtc trading_vdubus; do
    if docker inspect "$name" > /dev/null 2>&1; then
        errors=$(docker logs "$name" --tail 50 2>&1 | grep -ic "error\|exception\|traceback" || true)
        if [ "$errors" -gt 0 ]; then
            echo "  [WARN] $name: $errors error lines in last 50 log lines"
        else
            echo "  [OK]  $name: no errors in recent logs"
            PASS=$((PASS + 1))
        fi
    fi
done

echo ""
echo "5. Environment"
if [ -f /opt/trading/momentum_trader/.env ]; then
    env_mode=$(grep -oP 'ALGO_TRADER_ENV=\K.*' /opt/trading/momentum_trader/.env 2>/dev/null || echo "unknown")
    echo "  [INFO] ALGO_TRADER_ENV=$env_mode"
    ib_port=$(grep -oP 'IB_PORT=\K.*' /opt/trading/momentum_trader/.env 2>/dev/null || echo "unknown")
    echo "  [INFO] IB_PORT=$ib_port (paper=4002, live=4001)"
else
    echo "  [WARN] .env not found at /opt/trading/momentum_trader/.env"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "Review failures above. See docs/implementation.md §11 for troubleshooting."
    exit 1
fi
