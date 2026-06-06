CREATE TABLE IF NOT EXISTS reconciliation_authority_leases (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    client_id INT NOT NULL,
    family_id TEXT NOT NULL,
    recon_kind TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_snapshot_id TEXT,
    PRIMARY KEY (broker, account_id, client_id, family_id, recon_kind)
);

CREATE INDEX IF NOT EXISTS idx_recon_authority_expiry
    ON reconciliation_authority_leases(expires_at);
