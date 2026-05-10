-- =============================================================================
-- Daily data retention — run via infra/cron/retention.sh
-- =============================================================================
-- Removes stale operational data to keep the database lean.
-- Trades and orders are kept indefinitely for analysis; only high-volume
-- event/audit tables are pruned.

-- 1. Order events older than 60 days (highest-volume table)
DELETE FROM order_events
WHERE event_ts < now() - INTERVAL '60 days';

-- 2. Fills older than 90 days
DELETE FROM fills
WHERE fill_ts < now() - INTERVAL '90 days';

-- 3. Strategy state older than 30 days
DELETE FROM strategy_state
WHERE last_heartbeat_ts < now() - INTERVAL '30 days';

-- 4. Strategy heartbeat history older than 60 days (snapshot table feeding
--    v_daily_strategy_activity). Bounded growth: ~10 strategies × ~12
--    snapshots/hour × ~16 trading hours/day × 252 days ≈ 484k rows/year.
DELETE FROM strategy_heartbeat_history
WHERE captured_at < now() - INTERVAL '60 days';

-- 5. Reset daily disconnect counters (adapter_state.disconnect_count_24h)
UPDATE adapter_state SET disconnect_count_24h = 0;

-- 6. Vacuum analyze high-churn tables
VACUUM ANALYZE order_events;
VACUUM ANALYZE fills;
VACUUM ANALYZE trades;
VACUUM ANALYZE orders;
VACUUM ANALYZE strategy_state;
VACUUM ANALYZE strategy_heartbeat_history;
