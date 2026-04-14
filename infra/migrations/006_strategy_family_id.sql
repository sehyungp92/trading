-- Migration 006: Add family_id to risk_daily_strategy
-- Enables direct strategy→family mapping for dashboard queries
-- without joining through risk_daily_portfolio.
--
-- Safe to run on existing data: existing rows get family_id='unknown'.
-- After migration, the OMS factory writes the correct family_id on every upsert.

BEGIN;

ALTER TABLE risk_daily_strategy
    ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_risk_daily_strategy_family
    ON risk_daily_strategy(family_id);

COMMIT;
