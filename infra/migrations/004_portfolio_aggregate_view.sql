-- Migration 004: Create aggregate portfolio view
-- Provides a unified view across all family_id partitions in risk_daily_portfolio.
-- Used by the dashboard to show combined portfolio metrics.
-- Safe to run multiple times (DROP + CREATE).

DROP VIEW IF EXISTS v_portfolio_daily_summary;
CREATE VIEW v_portfolio_daily_summary AS
SELECT trade_date,
       SUM(daily_realized_r) AS daily_realized_r,
       SUM(daily_realized_usd) AS daily_realized_usd,
       SUM(portfolio_open_risk_r) AS portfolio_open_risk_r,
       BOOL_OR(halted) AS halted,
       MAX(halt_reason) AS halt_reason,
       MAX(last_update_at) AS last_update_at
FROM risk_daily_portfolio
GROUP BY trade_date;

GRANT SELECT ON v_portfolio_daily_summary TO trading_reader;
GRANT SELECT ON v_portfolio_daily_summary TO trading_writer;
