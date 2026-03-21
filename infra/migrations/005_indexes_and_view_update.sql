-- =============================================================================
-- Migration 005: Performance indexes + portfolio view update
-- =============================================================================

-- 1. Indexes on frequently queried columns
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(net_qty) WHERE net_qty != 0;
CREATE INDEX IF NOT EXISTS idx_orders_broker ON orders(broker_order_id) WHERE broker_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_strategy_status ON orders(strategy_id, status);

-- 2. Update portfolio view to aggregate across families (adds halt_reason)
CREATE OR REPLACE VIEW v_portfolio_daily_summary AS
SELECT trade_date,
       SUM(daily_realized_r) AS daily_realized_r,
       SUM(daily_realized_usd) AS daily_realized_usd,
       SUM(portfolio_open_risk_r) AS portfolio_open_risk_r,
       BOOL_OR(halted) AS halted,
       string_agg(DISTINCT halt_reason, '; ')
         FILTER (WHERE halt_reason IS NOT NULL AND halt_reason != '') AS halt_reason,
       MAX(last_update_at) AS last_update_at
FROM risk_daily_portfolio
GROUP BY trade_date;
