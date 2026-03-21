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
WHERE updated_at < now() - INTERVAL '30 days';

-- 4. Reset daily disconnect counters (adapter_state.disconnect_count_24h)
UPDATE adapter_state SET disconnect_count_24h = 0;

-- 5. Vacuum analyze high-churn tables
VACUUM ANALYZE order_events;
VACUUM ANALYZE fills;
VACUUM ANALYZE trades;
VACUUM ANALYZE orders;
VACUUM ANALYZE strategy_state;
