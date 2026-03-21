import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { HealthData, StrategyHealthRow, AdapterHealthRow, HaltRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const [strategies, adapters, halts] = await Promise.all([
      // v_strategy_health: computes heartbeat_age_sec and health_status
      query<StrategyHealthRow>(
        `SELECT * FROM v_strategy_health ORDER BY strategy_id`
      ),
      // v_adapter_health: computes heartbeat_age_sec and health_status
      query<AdapterHealthRow>(
        `SELECT * FROM v_adapter_health ORDER BY adapter_id`
      ),
      // v_active_halts: UNION of strategy + portfolio halts for today
      query<HaltRow>(
        `SELECT halt_level, entity, halt_reason, last_update_at
         FROM v_active_halts
         ORDER BY halt_level, entity`
      ),
    ]);

    const data: HealthData = { strategies, adapters, halts };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/health]', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
