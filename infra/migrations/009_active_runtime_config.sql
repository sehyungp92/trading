CREATE TABLE IF NOT EXISTS active_runtime_config (
    account_id TEXT NOT NULL DEFAULT '',
    config_scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    runtime_env TEXT NOT NULL,
    config_version TEXT NOT NULL,
    deployment_id TEXT,
    source_hash TEXT,
    payload JSONB NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (account_id, config_scope, scope_id, runtime_env)
);

ALTER TABLE active_runtime_config
    ADD COLUMN IF NOT EXISTS account_id TEXT NOT NULL DEFAULT '';

UPDATE active_runtime_config
SET account_id = COALESCE(
    NULLIF(payload->>'account_id', ''),
    CASE WHEN config_scope = 'account' THEN scope_id ELSE '' END
)
WHERE account_id = '';

ALTER TABLE active_runtime_config
    DROP CONSTRAINT IF EXISTS active_runtime_config_pkey;

ALTER TABLE active_runtime_config
    ADD CONSTRAINT active_runtime_config_pkey
    PRIMARY KEY (account_id, config_scope, scope_id, runtime_env);

CREATE INDEX IF NOT EXISTS idx_active_runtime_config_applied
    ON active_runtime_config(account_id, runtime_env, applied_at DESC);

CREATE INDEX IF NOT EXISTS idx_active_runtime_config_scope
    ON active_runtime_config(account_id, config_scope, runtime_env);

CREATE OR REPLACE VIEW v_active_runtime_config AS
SELECT *
FROM active_runtime_config
WHERE expires_at IS NULL OR expires_at > now();
