"""Shared pytest fixtures for Sandy tests.

Provides a live Postgres database via testcontainers for integration and
property-based tests that need real DB state.
"""
from __future__ import annotations

import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import text

from sandy.config import Config, DatabaseConfig, IngestConfig, LoggingConfig, ModelConfig, TrainingConfig
from sandy.db import bootstrap_schema, create_engine, get_connection
from pathlib import Path


@pytest.fixture(scope="session")
def pg_container():
    """Start a throwaway Postgres 16 container for the test session."""
    with PostgresContainer("postgres:16") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_config(pg_container):
    """Return a Config wired to the test container."""
    return Config(
        database=DatabaseConfig(
            host=pg_container.get_container_host_ip(),
            port=int(pg_container.get_exposed_port(5432)),
            name=pg_container.dbname,
            user=pg_container.username,
            password=pg_container.password,
        ),
        model=ModelConfig(path=Path("./models/latest.pkl")),
        ingest=IngestConfig(),
        training=TrainingConfig(),
        logging=LoggingConfig(),
    )


@pytest.fixture(scope="session")
def db_engine(db_config):
    """Return a SQLAlchemy engine connected to the test container."""
    engine = create_engine(db_config)
    with engine.begin() as conn:
        bootstrap_schema(conn)
    return engine


@pytest.fixture()
def clean_db(db_engine):
    """Truncate all raw and derived tables before each test that needs a clean slate."""
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE raw.plays, raw.pitcher_game_stats, raw.ingest_failures CASCADE"))
        conn.execute(text("TRUNCATE derived.inning_labels, derived.inning_features CASCADE"))
        conn.execute(text("DELETE FROM raw.games"))
        conn.execute(text("DELETE FROM raw.players"))
        conn.execute(text("DELETE FROM raw.teams"))
    yield db_engine
