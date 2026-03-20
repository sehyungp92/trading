-- Run daily via cron or pg_cron extension
-- Deletes old order_events (60 days) and optionally archives old orders

-- Delete old order events
DELETE FROM order_events
WHERE event_ts < now() - INTERVAL '60 days';

-- Optional: Archive completed orders older than 180 days
-- (Uncomment if you want automatic archival)
-- DELETE FROM orders
-- WHERE status IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED')
--   AND last_update_at < now() - INTERVAL '180 days';

-- Reset daily disconnect counters at midnight
UPDATE adapter_state SET disconnect_count_24h = 0;

-- Vacuum analyze for performance
VACUUM ANALYZE order_events;
VACUUM ANALYZE fills;
VACUUM ANALYZE trades;
