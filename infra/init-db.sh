#!/bin/bash
set -e

# =============================================================================
# Unified Trading Database Initialization
# Runs once on first postgres container start via docker-entrypoint-initdb.d
#
# Env vars (from docker-compose / .env):
#   POSTGRES_READER_PASSWORD  — password for dashboard read-only role
#   POSTGRES_WRITER_PASSWORD  — password for OMS writer role (optional)
# =============================================================================

READER_PW="${POSTGRES_READER_PASSWORD:-changeme}"
WRITER_PW="${POSTGRES_WRITER_PASSWORD:-changeme}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
-- Create read-only user for the trading dashboard
CREATE USER trading_reader WITH PASSWORD '${READER_PW}';
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO trading_reader;

-- Create writer role for OMS services
CREATE USER trading_writer WITH PASSWORD '${WRITER_PW}';
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO trading_writer;
GRANT USAGE, CREATE ON SCHEMA public TO trading_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO trading_writer;

-- Read-only permissions for dashboard (trading_reader)
GRANT USAGE ON SCHEMA public TO trading_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO trading_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE trading_writer IN SCHEMA public
    GRANT SELECT ON TABLES TO trading_reader;
EOSQL

# Run numbered migration files from the mounted migrations directory
MIGRATIONS_DIR="/docker-entrypoint-initdb.d/migrations"
if [ -d "$MIGRATIONS_DIR" ]; then
    for f in "$MIGRATIONS_DIR"/*.sql; do
        [ -f "$f" ] || continue
        echo "Running migration: $(basename "$f")"
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
    done
fi
