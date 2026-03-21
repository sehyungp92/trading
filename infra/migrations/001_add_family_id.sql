-- Migration 001: Add family_id to risk_daily_portfolio
-- Enables per-family portfolio risk tracking for single-VPS multi-family deployment.
--
-- Before: PK = (trade_date)        → one row per day, families collide
-- After:  PK = (trade_date, family_id) → one row per day per family
--
-- Safe to run on existing data: existing rows get family_id='unknown'.
-- After migration, each family writes with its own family_id ('swing', 'momentum', 'stock').

BEGIN;

-- 1. Add column with default (backfills existing rows)
ALTER TABLE risk_daily_portfolio
    ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT 'unknown';

-- 2. Drop old PK and create composite PK
ALTER TABLE risk_daily_portfolio
    DROP CONSTRAINT IF EXISTS risk_daily_portfolio_pkey;

ALTER TABLE risk_daily_portfolio
    ADD PRIMARY KEY (trade_date, family_id);

COMMIT;
