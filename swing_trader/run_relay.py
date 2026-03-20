"""Relay entry point — loads secrets and creates the app."""
import json
import os
from pathlib import Path

from relay.app import create_relay_app

secrets_file = os.environ.get("RELAY_SECRETS_FILE", "/opt/trading-relay/secrets.json")
secrets = {}
if Path(secrets_file).exists():
    secrets = json.loads(Path(secrets_file).read_text())

app = create_relay_app(
    db_path=os.environ.get("RELAY_DB_PATH", "/opt/trading-relay/data/relay.db"),
    shared_secrets=secrets,
    api_key=os.environ.get("RELAY_API_KEY", ""),
)
