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

# INFRA-1: pass passwords as psql variables and use a single-quoted heredoc.
# The previous unquoted heredoc let bash interpolate ${READER_PW}/${WRITER_PW}
# straight into SQL, which broke role creation for any password containing
# a single-quote, dollar sign, backtick, or backslash. With -v + :'name' psql
# performs proper SQL literal escaping.
psql \
    -v ON_ERROR_STOP=1 \
    -v reader_pw="${READER_PW}" \
    -v writer_pw="${WRITER_PW}" \
    -v db_name="${POSTGRES_DB}" \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" <<'EOSQL'
-- Pin timezone to ET so CURRENT_DATE matches OMS trade-date semantics
ALTER DATABASE :"db_name" SET timezone = 'America/New_York';

-- Create read-only user for the trading dashboard
CREATE USER trading_reader WITH PASSWORD :'reader_pw';
GRANT CONNECT ON DATABASE :"db_name" TO trading_reader;

-- Create writer role for OMS services
CREATE USER trading_writer WITH PASSWORD :'writer_pw';
GRANT CONNECT ON DATABASE :"db_name" TO trading_writer;
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
