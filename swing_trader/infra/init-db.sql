-- Create read-only user for the trading dashboard
CREATE USER trading_reader WITH PASSWORD 'reader_password_change_me';
GRANT CONNECT ON DATABASE trading TO trading_reader;

-- Switch to trading database
\c trading

-- Create writer role for OMS services
CREATE USER trading_writer WITH PASSWORD 'writer_password_change_me';
GRANT CONNECT ON DATABASE trading TO trading_writer;
GRANT USAGE, CREATE ON SCHEMA public TO trading_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trading_writer;

-- Grant read-only permissions for the dashboard (trading_reader)
-- Covers tables/views created by both the superuser and trading_writer.
-- Note: OMS schema (tables + views) is created at runtime by the OMS process
-- running as trading_writer — ALTER DEFAULT PRIVILEGES below ensures those
-- objects are automatically SELECT-granted to trading_reader on creation.
GRANT USAGE ON SCHEMA public TO trading_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO trading_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE trading_writer IN SCHEMA public GRANT SELECT ON TABLES TO trading_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE trading_writer IN SCHEMA public GRANT SELECT ON SEQUENCES TO trading_reader;
