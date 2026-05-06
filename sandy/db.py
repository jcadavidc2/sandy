"""Database layer for Sandy.

Tasks 3.1, 3.2, 3.3:
- SQLAlchemy engine factory from Config.database
- Session context manager
- bootstrap_schema(): idempotent DDL for raw.* and derived.* tables

All DDL uses CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS so
bootstrap_schema() is safe to call on every startup (requirement 3.5).

The raw schema holds canonical MLB Stats API data; the derived schema holds
computed labels and features. A future context schema (Phase 5) will be
additive and is not created here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine as _sa_create_engine, text
from sqlalchemy.engine import Connection, Engine

from sandy.config import Config


def build_dsn(cfg: Config) -> str:
    """Build a psycopg (v3) DSN from DatabaseConfig."""
    db = cfg.database
    return (
        f"postgresql+psycopg://{db.user}:{db.password}"
        f"@{db.host}:{db.port}/{db.name}"
    )


def create_engine(cfg: Config, **kwargs) -> Engine:
    """Return a SQLAlchemy Engine configured from *cfg*.

    pool_pre_ping=True ensures stale connections are detected and recycled,
    which matters on a t3.small where the DB container may restart.
    """
    return _sa_create_engine(
        build_dsn(cfg),
        pool_pre_ping=True,
        **kwargs,
    )


@contextmanager
def get_connection(engine: Engine) -> Generator[Connection, None, None]:
    """Yield a SQLAlchemy Connection, committing on success or rolling back."""
    with engine.begin() as conn:
        yield conn


# ---------------------------------------------------------------------------
# DDL — raw schema
# ---------------------------------------------------------------------------

_RAW_DDL = """
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.teams (
    team_code        CHAR(3)     PRIMARY KEY,
    team_id          INTEGER     NOT NULL UNIQUE,
    name             TEXT        NOT NULL,
    venue_id         INTEGER,
    league           TEXT,
    division         TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.players (
    player_id        INTEGER     PRIMARY KEY,
    full_name        TEXT        NOT NULL,
    primary_position TEXT,
    throws           CHAR(1),
    bats             CHAR(1),
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS players_lower_name_idx
    ON raw.players (LOWER(full_name));

CREATE TABLE IF NOT EXISTS raw.games (
    game_pk          INTEGER     PRIMARY KEY,
    game_date        DATE        NOT NULL,
    season           INTEGER     NOT NULL,
    game_type        CHAR(1)     NOT NULL,
    status           TEXT        NOT NULL,
    home_team_code   CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    away_team_code   CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    venue_id         INTEGER,
    first_pitch_utc  TIMESTAMPTZ,
    home_score       INTEGER,
    away_score       INTEGER,
    home_starter_id  INTEGER     REFERENCES raw.players(player_id),
    away_starter_id  INTEGER     REFERENCES raw.players(player_id),
    raw_payload_hash TEXT        NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS games_date_idx
    ON raw.games (game_date);
CREATE INDEX IF NOT EXISTS games_season_idx
    ON raw.games (season);
CREATE INDEX IF NOT EXISTS games_status_idx
    ON raw.games (status);

CREATE TABLE IF NOT EXISTS raw.plays (
    game_pk           INTEGER     NOT NULL
                                  REFERENCES raw.games(game_pk)
                                  ON DELETE CASCADE,
    at_bat_index      INTEGER     NOT NULL,
    inning            SMALLINT    NOT NULL,
    half_inning       CHAR(6)     NOT NULL
                                  CHECK (half_inning IN ('top','bottom')),
    batting_team_code  CHAR(3)    NOT NULL REFERENCES raw.teams(team_code),
    pitching_team_code CHAR(3)    NOT NULL REFERENCES raw.teams(team_code),
    batter_id         INTEGER     REFERENCES raw.players(player_id),
    pitcher_id        INTEGER     REFERENCES raw.players(player_id),
    batting_order     SMALLINT,
    event_type        TEXT        NOT NULL,
    event_code        TEXT        NOT NULL,
    is_reaches_base   BOOLEAN     NOT NULL,
    pitches_in_pa     SMALLINT    NOT NULL DEFAULT 0,
    start_time_utc    TIMESTAMPTZ,
    end_time_utc      TIMESTAMPTZ,
    raw               JSONB       NOT NULL,
    PRIMARY KEY (game_pk, at_bat_index)
);

CREATE INDEX IF NOT EXISTS plays_inning_idx
    ON raw.plays (game_pk, inning, half_inning, at_bat_index);
CREATE INDEX IF NOT EXISTS plays_pitcher_idx
    ON raw.plays (pitcher_id, game_pk);
CREATE INDEX IF NOT EXISTS plays_batting_team_idx
    ON raw.plays (batting_team_code, game_pk);

CREATE TABLE IF NOT EXISTS raw.pitcher_game_stats (
    game_pk          INTEGER     NOT NULL
                                 REFERENCES raw.games(game_pk)
                                 ON DELETE CASCADE,
    pitcher_id       INTEGER     NOT NULL REFERENCES raw.players(player_id),
    team_code        CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    pitches_thrown   INTEGER     NOT NULL DEFAULT 0,
    outs_recorded    INTEGER     NOT NULL DEFAULT 0,
    runs_allowed     INTEGER     NOT NULL DEFAULT 0,
    walks            INTEGER     NOT NULL DEFAULT 0,
    hits_allowed     INTEGER     NOT NULL DEFAULT 0,
    strikeouts       INTEGER     NOT NULL DEFAULT 0,
    is_starter       BOOLEAN     NOT NULL,
    PRIMARY KEY (game_pk, pitcher_id)
);

CREATE INDEX IF NOT EXISTS pgs_pitcher_idx
    ON raw.pitcher_game_stats (pitcher_id);

CREATE TABLE IF NOT EXISTS raw.ingest_failures (
    id               BIGSERIAL   PRIMARY KEY,
    game_pk          INTEGER,
    endpoint         TEXT,
    error_reason     TEXT        NOT NULL,
    http_status      INTEGER,
    retries          SMALLINT    NOT NULL DEFAULT 0,
    attempted_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_failures_game_idx
    ON raw.ingest_failures (game_pk);
"""

# ---------------------------------------------------------------------------
# DDL — derived schema
# ---------------------------------------------------------------------------

_DERIVED_DDL = """
CREATE SCHEMA IF NOT EXISTS derived;

CREATE TABLE IF NOT EXISTS derived.inning_labels (
    game_pk          INTEGER     NOT NULL
                                 REFERENCES raw.games(game_pk)
                                 ON DELETE CASCADE,
    team_code        CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    inning_number    SMALLINT    NOT NULL
                                 CHECK (inning_number BETWEEN 1 AND 20),
    reached_base     BOOLEAN     NOT NULL,
    labeled_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_code, inning_number)
);

CREATE TABLE IF NOT EXISTS derived.inning_features (
    game_pk                    INTEGER     NOT NULL
                                           REFERENCES raw.games(game_pk)
                                           ON DELETE CASCADE,
    team_code                  CHAR(3)     NOT NULL
                                           REFERENCES raw.teams(team_code),
    inning_number              SMALLINT    NOT NULL
                                           CHECK (inning_number BETWEEN 1 AND 20),
    feature_schema_version     INTEGER     NOT NULL,

    opp_starter_era            REAL,
    opp_starter_whip           REAL,
    opp_starter_k9             REAL,
    opp_starter_pitches_before INTEGER,
    lineup_spot_1              SMALLINT    CHECK (lineup_spot_1 BETWEEN 1 AND 9),
    lineup_spot_2              SMALLINT    CHECK (lineup_spot_2 BETWEEN 1 AND 9),
    lineup_spot_3              SMALLINT    CHECK (lineup_spot_3 BETWEEN 1 AND 9),
    lineup_spot1_season_obp    REAL,
    lineup_spot2_season_obp    REAL,
    lineup_spot3_season_obp    REAL,
    is_home                    BOOLEAN,
    ballpark_id                INTEGER,
    inning_number_feat         SMALLINT,
    trailing15_rpg             REAL,
    trailing15_obp             REAL,
    prev_inning_reached_base   SMALLINT,
    innings_reached_so_far     SMALLINT,
    consecutive_reach_streak   SMALLINT,
    team_season_obp            REAL,
    team_season_rpg            REAL,

    built_at                   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_code, inning_number)
);

CREATE INDEX IF NOT EXISTS inning_features_version_idx
    ON derived.inning_features (feature_schema_version);
"""


def bootstrap_schema(conn: Connection) -> None:
    """Create all raw.* and derived.* tables idempotently.

    Safe to call on every startup — uses IF NOT EXISTS throughout.
    Runs both DDL blocks in the caller's transaction so the whole schema
    either lands or rolls back together.
    """
    conn.execute(text(_RAW_DDL))
    conn.execute(text(_DERIVED_DDL))


__all__ = [
    "bootstrap_schema",
    "build_dsn",
    "create_engine",
    "get_connection",
]
