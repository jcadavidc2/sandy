"""Run a full backfill — commits per game so progress is visible immediately.

Usage on EC2:
    MLB_DB_HOST=localhost MLB_DB_PORT=5432 MLB_DB_NAME=sandy \
    MLB_DB_USER=sandy MLB_DB_PASSWORD=sandydev \
    nohup .venv/bin/python run_backfill.py > backfill.log 2>&1 &
"""
from sandy.config import load_config
from sandy.db import bootstrap_schema, create_engine, get_connection
from sandy.ingest.client import MlbStatsClient
from sandy.ingest.service import backfill_seasons
from sandy.logging import configure_logging

configure_logging("INFO")

cfg = load_config()
engine = create_engine(cfg)

# Ensure schema exists
with get_connection(engine) as conn:
    bootstrap_schema(conn)

client = MlbStatsClient(cfg.ingest)

# Pass engine (not conn) so each game commits independently
stats = backfill_seasons(engine, client, seasons=[2022, 2023, 2024])

print(
    f"\nBackfill complete:\n"
    f"  Games processed : {stats.games_processed}\n"
    f"  Games skipped   : {stats.games_skipped}\n"
    f"  Games failed    : {stats.games_failed}\n"
    f"  Elapsed         : {stats.elapsed_seconds}s"
)
