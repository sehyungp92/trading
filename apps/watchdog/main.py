"""Main async loop for the watchdog service."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
import asyncpg

from libs.oms.persistence.db_config import DBConfig

from .alerts import TelegramAlerter, format_alert, format_startup_summary
from .checks import (
    check_adapters,
    check_daily_pnl,
    check_errors,
    check_halts,
    check_heartbeats,
    check_ib_gateway,
    check_relay,
)
from .config import build_strategy_family_map, is_family_active, load_watchdog_config
from .cooldown import CooldownTracker

logger = logging.getLogger("watchdog")
_ET = ZoneInfo("America/New_York")


async def _create_pool(db_cfg: DBConfig, max_retries: int = 10) -> asyncpg.Pool:
    """Create asyncpg pool with retry loop for slow postgres startup."""
    for attempt in range(1, max_retries + 1):
        try:
            pool = await asyncpg.create_pool(
                dsn=db_cfg.to_dsn(), min_size=1, max_size=3
            )
            logger.info("DB pool created on attempt %d", attempt)
            return pool
        except (OSError, asyncpg.PostgresError) as exc:
            if attempt == max_retries:
                raise
            logger.warning("DB connect attempt %d/%d failed: %s", attempt, max_retries, exc)
            await asyncio.sleep(5)
    raise RuntimeError("unreachable")


async def _run_cycle(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    config: dict,
    cooldown: CooldownTracker,
    alerter: TelegramAlerter,
    strategy_family_map: dict[str, str],
) -> None:
    """Execute one monitoring cycle."""
    schedules = config.get("schedules", {})
    checks_cfg = config.get("checks", {})
    now_et = datetime.now(_ET)

    # Determine which families are active right now
    active_families: set[str] = set()
    for family in schedules:
        if is_family_active(family, now_et, schedules):
            active_families.add(family)

    # Build list of enabled check coroutines
    tasks: list[asyncio.Task] = []
    if checks_cfg.get("heartbeat", {}).get("enabled"):
        tasks.append(asyncio.create_task(
            check_heartbeats(pool, config, active_families, strategy_family_map)
        ))
    if checks_cfg.get("adapter", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_adapters(pool, config)))
    if checks_cfg.get("halts", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_halts(pool, config)))
    if checks_cfg.get("ib_gateway", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_ib_gateway(config)))
    if checks_cfg.get("daily_pnl", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_daily_pnl(pool, config)))
    if checks_cfg.get("relay", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_relay(session, config)))
    if checks_cfg.get("errors", {}).get("enabled"):
        tasks.append(asyncio.create_task(check_errors(pool, config)))

    if not tasks:
        logger.warning("No checks enabled")
        return

    # Run all checks concurrently
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    alerts_sent = 0
    recoveries_sent = 0
    problems = 0
    recovery_enabled = config.get("cooldown", {}).get("recovery_enabled", True)

    for result in results_nested:
        if isinstance(result, Exception):
            logger.error("Check raised exception: %s", result)
            continue
        for cr in result:
            if cr.is_problem:
                problems += 1
                if cooldown.should_alert(cr.key):
                    msg = format_alert(cr.check_name, cr.detail, is_recovery=False)
                    await alerter.send(msg)
                    alerts_sent += 1
            else:
                if recovery_enabled and cooldown.clear(cr.key):
                    msg = format_alert(cr.check_name, cr.detail, is_recovery=True)
                    await alerter.send(msg)
                    recoveries_sent += 1

    logger.debug(
        "Cycle done: %d problems, %d alerts sent, %d recoveries, active_families=%s",
        problems, alerts_sent, recoveries_sent, active_families,
    )


async def _main() -> None:
    """Async entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_watchdog_config()
    strategy_family_map = build_strategy_family_map()
    poll_interval = config.get("poll_interval_seconds", 120)

    tg_cfg = config.get("telegram", {})
    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        sys.exit(1)

    cooldown_sec = config.get("cooldown", {}).get("alert_cooldown_sec", 900)
    cooldown = CooldownTracker(cooldown_sec)

    db_cfg = DBConfig.from_env()
    if db_cfg is None:
        logger.error("No database configuration found -- set DB_HOST or DATABASE_URL")
        sys.exit(1)

    # Shutdown event
    stop = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _signal_handler())

    pool = await _create_pool(db_cfg)
    session = aiohttp.ClientSession()
    alerter = TelegramAlerter(bot_token, chat_id, session)

    # Startup notification
    families = set(strategy_family_map.values())
    startup_msg = format_startup_summary(len(strategy_family_map), len(families), poll_interval)
    await alerter.send(startup_msg)
    logger.info("Watchdog started -- %d strategies, %d families", len(strategy_family_map), len(families))

    try:
        while not stop.is_set():
            try:
                await _run_cycle(pool, session, config, cooldown, alerter, strategy_family_map)
            except Exception as exc:
                logger.exception("Cycle failed: %s", exc)
                try:
                    await alerter.send(f"[!] <b>Watchdog internal error</b>\n<code>{str(exc)[:200]}</code>")
                except Exception:
                    pass

            # Sleep in small increments so we can respond to shutdown quickly
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)
                break  # stop was set
            except asyncio.TimeoutError:
                pass  # normal -- poll interval elapsed
    finally:
        logger.info("Shutting down...")
        try:
            await alerter.send("<b>Watchdog shutting down</b>")
        except Exception:
            pass
        await session.close()
        await pool.close()
        logger.info("Watchdog stopped")


def run() -> None:
    """Sync entry point."""
    asyncio.run(_main())
