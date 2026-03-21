-- Migration 002: Add overlay_positions table
-- Tracks swing overlay ETF positions for dashboard visibility.
-- Safe to run multiple times (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS overlay_positions (
    symbol TEXT NOT NULL,
    shares INT NOT NULL DEFAULT 0,
    notional NUMERIC NOT NULL DEFAULT 0,
    pct_of_nav NUMERIC NOT NULL DEFAULT 0,
    rebalance_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol)
);
