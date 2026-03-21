-- Migration 003: Create paper_equity table
-- Tracks simulated account balances for paper-trading and NAV-based position sizing.
-- Used by PaperEquityManager in libs/persistence/paper_equity.py.
-- Safe to run multiple times (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS paper_equity (
    account_scope TEXT PRIMARY KEY,
    equity DOUBLE PRECISION NOT NULL,
    initial_equity DOUBLE PRECISION NOT NULL,
    total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_commission DOUBLE PRECISION NOT NULL DEFAULT 0,
    trade_count INTEGER NOT NULL DEFAULT 0,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE ON paper_equity TO trading_writer;
GRANT SELECT ON paper_equity TO trading_reader;
