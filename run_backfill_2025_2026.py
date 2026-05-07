"""Backfill 2025 (complete) + 2026 (current season to date)."""
from sandy.config import load_config
from sandy.db import create_engine
from sandy.ingest.client import MlbStatsClient
from sandy.ingest.service import backfill_seasons
from sandy.logging import configure_logging

configure_logging("INFO")
cfg = load_config()
engine = create_engine(cfg)
client = MlbStatsClient(cfg.ingest)

print("Backfilling 2025 + 2026...")
stats = backfill_seasons(engine, client, seasons=[2025, 2026])
print(f"Done: {stats.games_processed} processed, {stats.games_skipped} skipped, {stats.games_failed} failed, {stats.elapsed_seconds}s")
