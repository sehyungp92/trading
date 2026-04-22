"""Health check coroutines for the watchdog.

Each check returns list[CheckResult]. Checks never raise -- they catch
exceptions internally and return an error CheckResult instead.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp
import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    key: str            # cooldown key, e.g. "heartbeat:Helix_v40"
    check_name: str     # human-readable check name
    detail: str         # description of the problem or OK state
    is_problem: bool


# ---------------------------------------------------------------------------
# 1. Heartbeat staleness (market-hours gated)
# ---------------------------------------------------------------------------

async def check_heartbeats(
    pool: asyncpg.Pool,
    config: dict,
    active_families: set[str],
    strategy_family_map: dict[str, str],
) -> list[CheckResult]:
    """Check v_strategy_health for stale heartbeats."""
    threshold = config.get("checks", {}).get("heartbeat", {}).get("stale_threshold_sec", 180)
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch("SELECT * FROM v_strategy_health")
        for row in rows:
            sid = row["strategy_id"]
            family = strategy_family_map.get(sid)
            # Skip strategies whose family is not active right now
            if family and family not in active_families:
                continue
            age = row["heartbeat_age_sec"]
            status = row["health_status"]
            is_stale = age is not None and float(age) > threshold
            if is_stale:
                results.append(CheckResult(
                    key=f"heartbeat:{sid}",
                    check_name="Heartbeat",
                    detail=f"{sid} -- stale for {int(age)}s (threshold {threshold}s), status={status}",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"heartbeat:{sid}",
                    check_name="Heartbeat",
                    detail=f"{sid} OK ({int(age or 0)}s)",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="heartbeat:__db_error__",
            check_name="Heartbeat",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 2. Adapter health (not market-hours gated)
# ---------------------------------------------------------------------------

async def check_adapters(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check v_adapter_health for disconnected or stale adapters."""
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch("SELECT * FROM v_adapter_health")
        for row in rows:
            aid = row["adapter_id"]
            status = row["health_status"]
            is_problem = status != "OK"
            detail = f"{aid} -- {status}"
            if is_problem and row.get("last_error_message"):
                detail += f" ({row['last_error_message']})"
            results.append(CheckResult(
                key=f"adapter:{aid}",
                check_name="Adapter",
                detail=detail,
                is_problem=is_problem,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="adapter:__db_error__",
            check_name="Adapter",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 3. Active halts
# ---------------------------------------------------------------------------

async def check_halts(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check v_active_halts for any current halts."""
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch(
            "SELECT halt_level, entity, halt_reason, last_update_at FROM v_active_halts"
        )
        for row in rows:
            level = row["halt_level"]
            entity = row["entity"]
            reason = row["halt_reason"] or "unknown"
            results.append(CheckResult(
                key=f"halt:{level}:{entity}",
                check_name="Halt",
                detail=f"{level} halt on {entity} -- {reason}",
                is_problem=True,
            ))
        if not rows:
            results.append(CheckResult(
                key="halt:none",
                check_name="Halt",
                detail="No active halts",
                is_problem=False,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="halt:__db_error__",
            check_name="Halt",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 4. IB Gateway TCP probe
# ---------------------------------------------------------------------------

async def check_ib_gateway(config: dict) -> list[CheckResult]:
    """TCP connect to IB Gateway port."""
    gw_cfg = config.get("checks", {}).get("ib_gateway", {})
    host = gw_cfg.get("host", "127.0.0.1")
    port = gw_cfg.get("port", 4002)
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
        return [CheckResult(
            key="ib_gateway:connectivity",
            check_name="IB Gateway",
            detail=f"Connected to {host}:{port}",
            is_problem=False,
        )]
    except Exception as exc:
        return [CheckResult(
            key="ib_gateway:connectivity",
            check_name="IB Gateway",
            detail=f"Cannot reach {host}:{port} -- {exc}",
            is_problem=True,
        )]


# ---------------------------------------------------------------------------
# 5. Daily P&L drawdown
# ---------------------------------------------------------------------------

async def check_daily_pnl(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check per-strategy and portfolio-level drawdown thresholds."""
    pnl_cfg = config.get("checks", {}).get("daily_pnl", {})
    strat_threshold = pnl_cfg.get("strategy_drawdown_r", -3.0)
    port_threshold = pnl_cfg.get("portfolio_drawdown_r", -5.0)
    results: list[CheckResult] = []
    try:
        # Per-strategy
        rows = await pool.fetch(
            "SELECT strategy_id, family_id, daily_realized_r "
            "FROM risk_daily_strategy WHERE trade_date = CURRENT_DATE"
        )
        for row in rows:
            sid = row["strategy_id"]
            pnl_r = float(row["daily_realized_r"])
            if pnl_r <= strat_threshold:
                results.append(CheckResult(
                    key=f"pnl:{sid}",
                    check_name="Daily P&L",
                    detail=f"{sid} at {pnl_r:+.1f}R (threshold {strat_threshold}R)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"pnl:{sid}",
                    check_name="Daily P&L",
                    detail=f"{sid} at {pnl_r:+.1f}R",
                    is_problem=False,
                ))

        # Portfolio-level
        port_rows = await pool.fetch(
            "SELECT family_id, daily_realized_r "
            "FROM risk_daily_portfolio WHERE trade_date = CURRENT_DATE"
        )
        for row in port_rows:
            fid = row["family_id"]
            pnl_r = float(row["daily_realized_r"])
            if pnl_r <= port_threshold:
                results.append(CheckResult(
                    key=f"pnl:portfolio:{fid}",
                    check_name="Daily P&L",
                    detail=f"Portfolio {fid} at {pnl_r:+.1f}R (threshold {port_threshold}R)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"pnl:portfolio:{fid}",
                    check_name="Daily P&L",
                    detail=f"Portfolio {fid} at {pnl_r:+.1f}R",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="pnl:__db_error__",
            check_name="Daily P&L",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 6. Relay health
# ---------------------------------------------------------------------------

async def check_relay(
    session: aiohttp.ClientSession, config: dict
) -> list[CheckResult]:
    """HTTP GET to relay /health endpoint."""
    relay_cfg = config.get("checks", {}).get("relay", {})
    url = relay_cfg.get("url", "http://127.0.0.1:8000/health")
    backlog_threshold = relay_cfg.get("backlog_threshold", 500)
    stale_threshold = relay_cfg.get("oldest_pending_age_sec", 600)
    results: list[CheckResult] = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                results.append(CheckResult(
                    key="relay:reachable",
                    check_name="Relay",
                    detail=f"HTTP {resp.status} from {url}",
                    is_problem=True,
                ))
                return results
            results.append(CheckResult(
                key="relay:reachable",
                check_name="Relay",
                detail="Relay reachable",
                is_problem=False,
            ))
            body = await resp.json()
            pending = body.get("pending_events", 0)
            if pending > backlog_threshold:
                results.append(CheckResult(
                    key="relay:backlog",
                    check_name="Relay",
                    detail=f"Backlog: {pending} pending (threshold {backlog_threshold})",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key="relay:backlog",
                    check_name="Relay",
                    detail=f"Backlog OK ({pending} pending)",
                    is_problem=False,
                ))
            oldest_age = body.get("oldest_pending_age_seconds", 0)
            if oldest_age > stale_threshold:
                results.append(CheckResult(
                    key="relay:stale_pending",
                    check_name="Relay",
                    detail=f"Oldest pending event: {int(oldest_age)}s (threshold {stale_threshold}s)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key="relay:stale_pending",
                    check_name="Relay",
                    detail=f"Oldest pending: {int(oldest_age)}s",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="relay:reachable",
            check_name="Relay",
            detail=f"Cannot reach relay: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 7. Recent errors in strategy_state
# ---------------------------------------------------------------------------

async def check_errors(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check strategy_state for recent errors."""
    max_age = config.get("checks", {}).get("errors", {}).get("max_age_minutes", 30)
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch(
            "SELECT strategy_id, last_error, last_error_ts "
            "FROM strategy_state "
            "WHERE last_error IS NOT NULL "
            "  AND last_error_ts > now() - make_interval(mins := $1)",
            max_age,
        )
        for row in rows:
            sid = row["strategy_id"]
            err = (row["last_error"] or "")[:120]
            results.append(CheckResult(
                key=f"error:{sid}",
                check_name="Error",
                detail=f"{sid} -- {err}",
                is_problem=True,
            ))
        if not rows:
            results.append(CheckResult(
                key="error:none",
                check_name="Error",
                detail="No recent errors",
                is_problem=False,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="error:__db_error__",
            check_name="Error",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results
