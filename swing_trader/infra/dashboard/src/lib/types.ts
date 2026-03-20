// ── System & Strategy constants ──────────────────────────────────────────────

export type SystemId = 'swing_trader' | 'momentum_trader' | 'stock_trader';

export interface SystemConfig {
  label: string;
  shortLabel: string;
  color: string;       // Tailwind border/accent class
  dotColor: string;    // Tailwind bg class for small dots
  textColor: string;   // Tailwind text class
  bgColor: string;     // Tailwind bg class for badges/headers
}

export const SYSTEM_CONFIG: Record<SystemId, SystemConfig> = {
  swing_trader: {
    label: 'Swing Trader',
    shortLabel: 'SWING',
    color: 'border-blue-500',
    dotColor: 'bg-blue-500',
    textColor: 'text-blue-400',
    bgColor: 'bg-blue-950/30',
  },
  momentum_trader: {
    label: 'Momentum Trader',
    shortLabel: 'MOMO',
    color: 'border-purple-500',
    dotColor: 'bg-purple-500',
    textColor: 'text-purple-400',
    bgColor: 'bg-purple-950/30',
  },
  stock_trader: {
    label: 'Stock Trader',
    shortLabel: 'STOCK',
    color: 'border-emerald-500',
    dotColor: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    bgColor: 'bg-emerald-950/30',
  },
};

export const SYSTEM_ORDER: SystemId[] = ['swing_trader', 'momentum_trader', 'stock_trader'];

export interface StrategyConfig {
  system: SystemId;
  maxHeatR: number;
  riskPct: number;
  priority: number;
}

export const STRATEGY_CONFIG: Record<string, StrategyConfig> = {
  ATRSS:               { system: 'swing_trader',    maxHeatR: 1.00, riskPct: 1.2, priority: 0 },
  S5_PB:               { system: 'swing_trader',    maxHeatR: 1.50, riskPct: 0.8, priority: 1 },
  S5_DUAL:             { system: 'swing_trader',    maxHeatR: 1.50, riskPct: 0.8, priority: 2 },
  SWING_BREAKOUT_V3:   { system: 'swing_trader',    maxHeatR: 0.65, riskPct: 0.5, priority: 3 },
  AKC_HELIX:           { system: 'swing_trader',    maxHeatR: 0.85, riskPct: 0.5, priority: 4 },
  'AKC_Helix_v40':     { system: 'momentum_trader', maxHeatR: 3.00, riskPct: 0.8, priority: 0 },
  'NQDTC_v2.1':        { system: 'momentum_trader', maxHeatR: 1.00, riskPct: 0.8, priority: 1 },
  VdubusNQ_v4:         { system: 'momentum_trader', maxHeatR: 3.50, riskPct: 0.8, priority: 2 },
  IARIC_v1:            { system: 'stock_trader',    maxHeatR: 1.50, riskPct: 0.5, priority: 0 },
  US_ORB_v1:           { system: 'stock_trader',    maxHeatR: 1.00, riskPct: 0.35, priority: 1 },
};

/** Get the system a strategy belongs to (defaults to swing_trader for unknown IDs) */
export function getSystem(strategyId: string): SystemId {
  return STRATEGY_CONFIG[strategyId]?.system ?? 'swing_trader';
}

/** Get the SystemConfig for a strategy */
export function getSystemConfig(strategyId: string): SystemConfig {
  return SYSTEM_CONFIG[getSystem(strategyId)];
}

/** Group strategies by system, sorted by SYSTEM_ORDER then priority */
export function groupBySystem<T extends { strategy_id: string }>(items: T[]): Map<SystemId, T[]> {
  const grouped = new Map<SystemId, T[]>();
  for (const sys of SYSTEM_ORDER) grouped.set(sys, []);
  for (const item of items) {
    const sys = getSystem(item.strategy_id);
    grouped.get(sys)!.push(item);
  }
  // Sort each group by priority
  SYSTEM_ORDER.forEach(sys => {
    const list = grouped.get(sys);
    if (list) {
      list.sort((a, b) =>
        (STRATEGY_CONFIG[a.strategy_id]?.priority ?? 99) -
        (STRATEGY_CONFIG[b.strategy_id]?.priority ?? 99)
      );
    }
  });
  return grouped;
}

export const PORTFOLIO_HEAT_CAP = 2.0;
export const PORTFOLIO_DAILY_STOP = 3.0;

// ── API Response Types ──────────────────────────────────────────────────────

export interface PortfolioData {
  daily_realized_r: number;
  daily_realized_usd: number;
  portfolio_open_risk_r: number;
  unrealized_pnl: number;
  halted: boolean;
  halt_reason: string | null;
  heat_r: number; // sum of strategy heat_r from positions
}

export interface StrategyData {
  strategy_id: string;
  mode: string;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  heat_r: number;
  daily_pnl_r: number;
  last_error: string | null;
  last_error_ts: string | null;
  // from risk_daily_strategy
  daily_realized_r: number;
  daily_realized_usd: number;
  open_risk_r: number;
  filled_entries: number;
  halted: boolean;
  halt_reason: string | null;
}

export interface PositionRow {
  account_id: string;
  instrument_symbol: string;
  strategy_id: string;
  net_qty: number;
  avg_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  open_risk_dollars: number;
  open_risk_r: number;
  last_update_at: string;
  stale_minutes: number;
}

export interface TradeRow {
  trade_id: string;
  strategy_id: string;
  instrument_symbol: string;
  direction: string;
  quantity: number;
  entry_ts: string;
  entry_price: number;
  exit_ts: string | null;
  exit_price: number | null;
  realized_r: number | null;
  exit_reason: string | null;
  entry_type: string | null;
  mae_r: number | null;
  mfe_r: number | null;
  duration_minutes: number | null;
}

export interface OrderRow {
  oms_order_id: string;
  strategy_id: string;
  instrument_symbol: string;
  role: string;
  side: string;
  qty: number;
  filled_qty: number;
  stop_price: number | null;
  limit_price: number | null;
  status: string;
  broker_order_id: string | null;
  created_at: string;
  age_minutes: number;
}

export interface HealthData {
  strategies: StrategyHealthRow[];
  adapters: AdapterHealthRow[];
  halts: HaltRow[];
}

export interface StrategyHealthRow {
  strategy_id: string;
  mode: string;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  heat_r: number;
  daily_pnl_r: number;
  last_error: string | null;
  last_error_ts: string | null;
}

export interface AdapterHealthRow {
  adapter_id: string;
  broker: string;
  connected: boolean;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  disconnect_count_24h: number;
  last_error_code: string | null;
  last_error_message: string | null;
}

export interface HaltRow {
  halt_level: string;
  entity: string;
  halt_reason: string | null;
  last_update_at: string;
}

export interface EquityCurvePoint {
  trade_date: string;
  daily_realized_r: number;
  cumulative_r: number;
}

export interface DailyPnlPoint {
  trade_date: string;
  daily_realized_r: number;
}

export interface EnvData {
  mode: 'paper' | 'live' | 'dev';
  account_id: string;
  ib_port: number;
}

// ── Batch API response types ────────────────────────────────────────────────

export interface SystemPnlSummary {
  system: SystemId;
  daily_realized_r: number;
  daily_realized_usd: number;
  heat_r: number;
  filled_entries: number;
  strategy_count: number;
  healthy_count: number;
}

export interface LiveBatchResponse {
  portfolio: PortfolioData;
  strategies: StrategyData[];
  positions: PositionRow[];
  trades: TradeRow[];
  orders: OrderRow[];
  health: HealthData;
  systemPnl: SystemPnlSummary[];
  serverTime: string;
}

export interface ChartBatchResponse {
  equityCurve: EquityCurvePoint[];
  dailyPnl: DailyPnlPoint[];
  serverTime: string;
}
