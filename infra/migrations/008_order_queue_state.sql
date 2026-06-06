ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS queued_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS queue_priority INT,
    ADD COLUMN IF NOT EXISTS queue_reason TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS queue_attempt INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS queue_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS queue_claimed_by TEXT,
    ADD COLUMN IF NOT EXISTS queue_claimed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS queue_claim_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS dequeued_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS queue_denial_reason TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_orders_queued_ready
    ON orders(queue_priority, queued_at)
    WHERE status = 'QUEUED';

CREATE INDEX IF NOT EXISTS idx_orders_queue_expiry
    ON orders(queue_expires_at)
    WHERE status = 'QUEUED';

CREATE INDEX IF NOT EXISTS idx_orders_queue_claim_expiry
    ON orders(queue_claim_expires_at)
    WHERE status = 'QUEUED' AND queue_claimed_by IS NOT NULL;
