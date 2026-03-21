'use client';
import { StrategyData, STRATEGY_CONFIG } from '@/lib/types';
import { fmtR, fmtAge, toPercent } from '@/lib/formatters';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  strategy: StrategyData;
}

function statusVariant(status: string): 'success' | 'danger' | 'warning' | 'default' {
  if (status === 'OK') return 'success';
  if (status === 'STALE') return 'warning';
  if (status === 'HALTED') return 'danger';
  if (status === 'STAND_DOWN') return 'default';
  return 'default';
}

export function StrategyCard({ strategy }: Props) {
  const cfg = STRATEGY_CONFIG[strategy.strategy_id];
  const maxHeat = cfg?.maxHeatR ?? 1.0;
  const heatPct = toPercent(strategy.heat_r, maxHeat);
  const dailyStop = 2.0 - Math.abs(strategy.daily_pnl_r);

  const heatColor =
    heatPct > 95 ? 'bg-red-500' : heatPct > 80 ? 'bg-amber-500' : 'bg-green-500';

  return (
    <div className="rounded-lg border border-gray-800 bg-[#111318] p-4 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-mono font-semibold text-sm text-gray-100 leading-tight">
            {strategy.strategy_id}
          </p>
          <p className="text-xs text-gray-500 font-mono mt-0.5">
            P{cfg?.priority ?? '?'} · {cfg?.riskPct ?? '?'}% risk
          </p>
        </div>
        <Badge variant={statusVariant(strategy.health_status)}>
          {strategy.health_status}
        </Badge>
      </div>

      {/* P&L */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <p className="text-xs text-gray-500 font-mono">Daily P&amp;L</p>
          <p className={cn('text-lg font-bold font-mono',
            strategy.daily_realized_r >= 0 ? 'text-green-400' : 'text-red-400'
          )}>
            {fmtR(strategy.daily_realized_r)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 font-mono">Entries</p>
          <p className="text-lg font-bold font-mono text-gray-200">{strategy.filled_entries}</p>
        </div>
      </div>

      {/* Heat bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-gray-500">Heat</span>
          <span className={heatPct > 80 ? 'text-amber-400' : 'text-gray-400'}>
            {strategy.heat_r.toFixed(2)}R / {maxHeat}R
          </span>
        </div>
        <Progress value={heatPct} indicatorClassName={heatColor} className="h-1.5" />
      </div>

      {/* Daily stop remaining */}
      <div className="flex justify-between items-center text-xs font-mono">
        <span className="text-gray-500">Stop remaining</span>
        <span className={cn(
          dailyStop < 0.5 ? 'text-amber-400' : 'text-gray-400',
          dailyStop < 0 ? 'text-red-400' : ''
        )}>
          {dailyStop.toFixed(2)}R
        </span>
      </div>

      {/* Heartbeat */}
      <div className="text-xs text-gray-600 font-mono border-t border-gray-800 pt-2">
        Heartbeat: {fmtAge(strategy.heartbeat_age_sec)} ago
        {strategy.last_error && (
          <p className="text-amber-500 truncate mt-0.5" title={strategy.last_error}>
            ⚠ {strategy.last_error}
          </p>
        )}
      </div>
    </div>
  );
}
