-- Migration 007: strategy_heartbeat_history snapshot table.
--
-- Captures a periodic (~5 min) snapshot of strategy_state so the
-- daily-quiet-day classifier can compute bars-processed deltas and
-- denial totals across a session. The watchdog's snapshot writer
-- (apps/watchdog/snapshot.py) populates this table.
--
-- Retention (60 days) is enforced by infra/retention.sql.

BEGIN;

CREATE TABLE IF NOT EXISTS strategy_heartbeat_history (
    captured_at         TIMESTAMPTZ NOT NULL,
    strategy_id         TEXT        NOT NULL,
    mode                TEXT,
    last_decision_code  TEXT,
    bars_processed      BIGINT,
    last_seen_bar_ts    TIMESTAMPTZ,
    consecutive_denials INT,
    denials_today       INT,
    PRIMARY KEY (captured_at, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_heartbeat_history_captured
    ON strategy_heartbeat_history(captured_at);
CREATE INDEX IF NOT EXISTS idx_heartbeat_history_strategy
    ON strategy_heartbeat_history(strategy_id, captured_at);

-- Flatten last_decision_details JSONB for the dashboard (keys defined by
-- libs/services/decision_codes.py).
CREATE OR REPLACE VIEW v_strategy_diagnostics AS
SELECT
    sh.strategy_id,
    sh.mode,
    sh.health_status,
    sh.last_heartbeat_ts,
    sh.heartbeat_age_sec,
    sh.last_decision_code,
    sh.last_seen_bar_ts,
    sh.bar_age_sec,
    NULLIF(sh.last_decision_details->'liveness'->>'bars_processed', '')::bigint
        AS bars_processed,
    sh.last_decision_details->'liveness'->'symbol_freshness'
        AS symbol_freshness,
    NULLIF(sh.last_decision_details->'oms_health'->>'submitted', '')::int
        AS intents_submitted,
    NULLIF(sh.last_decision_details->'oms_health'->>'denied', '')::int
        AS intents_denied,
    NULLIF(sh.last_decision_details->'oms_health'->>'consecutive_denials', '')::int
        AS consecutive_denials,
    sh.last_decision_details->'ib_farm_status'
        AS ib_farm_status,
    sh.last_error,
    sh.last_error_ts
FROM v_strategy_health sh;


CREATE OR REPLACE VIEW v_daily_strategy_activity AS
WITH hb_agg AS (
    SELECT
        (captured_at AT TIME ZONE 'UTC')::date AS day,
        strategy_id,
        MAX(bars_processed) - MIN(bars_processed) AS bars,
        MAX(denials_today)                       AS denials,
        MAX(last_seen_bar_ts)                    AS last_bar_ts,
        (array_agg(last_decision_code ORDER BY captured_at DESC))[1] AS last_decision_code,
        (array_agg(mode ORDER BY captured_at DESC))[1]               AS last_mode
    FROM strategy_heartbeat_history
    GROUP BY (captured_at AT TIME ZONE 'UTC')::date, strategy_id
),
trade_agg AS (
    SELECT
        (entry_ts AT TIME ZONE 'UTC')::date AS day,
        strategy_id,
        count(*) AS trades
    FROM trades
    GROUP BY (entry_ts AT TIME ZONE 'UTC')::date, strategy_id
)
SELECT
    h.day,
    h.strategy_id,
    rds.family_id,
    h.bars,
    h.denials,
    h.last_bar_ts,
    h.last_decision_code,
    h.last_mode,
    COALESCE(t.trades, 0)            AS trades,
    rds.daily_realized_r,
    rds.daily_realized_usd
FROM hb_agg h
LEFT JOIN trade_agg t USING (day, strategy_id)
LEFT JOIN risk_daily_strategy rds
    ON rds.trade_date = h.day AND rds.strategy_id = h.strategy_id;

COMMIT;
