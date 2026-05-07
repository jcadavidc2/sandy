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

-- Phase 1.5: game-level prediction targets

CREATE TABLE IF NOT EXISTS derived.game_winner_labels (
    game_pk        INTEGER     PRIMARY KEY
                               REFERENCES raw.games(game_pk)
                               ON DELETE CASCADE,
    home_team_wins BOOLEAN     NOT NULL,
    labeled_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS derived.runs_labels (
    game_pk    INTEGER     NOT NULL
                           REFERENCES raw.games(game_pk)
                           ON DELETE CASCADE,
    team_code  CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    runs       INTEGER     NOT NULL,
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_code)
);

CREATE TABLE IF NOT EXISTS derived.game_features (
    game_pk                INTEGER     NOT NULL
                                       REFERENCES raw.games(game_pk)
                                       ON DELETE CASCADE,
    team_code              CHAR(3)     NOT NULL REFERENCES raw.teams(team_code),
    feature_schema_version INTEGER     NOT NULL,
    home_starter_era       REAL,
    home_starter_whip      REAL,
    away_starter_era       REAL,
    away_starter_whip      REAL,
    home_trailing15_rpg    REAL,
    away_trailing15_rpg    REAL,
    home_season_obp        REAL,
    away_season_obp        REAL,
    ballpark_id            INTEGER,
    is_home                BOOLEAN     NOT NULL,
    computed_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_code)
);

CREATE INDEX IF NOT EXISTS game_features_version_idx
    ON derived.game_features (feature_schema_version);

-- Phase 2: prediction logging for self-evaluation

CREATE TABLE IF NOT EXISTS derived.over_under_outcomes (
    id SERIAL PRIMARY KEY,
    game_pk INTEGER NOT NULL,
    game_date DATE NOT NULL,
    home_team_code VARCHAR(3) NOT NULL,
    away_team_code VARCHAR(3) NOT NULL,
    predicted_at_utc TIMESTAMPTZ NOT NULL,
    p_over_5_5 FLOAT NOT NULL,
    p_over_6_5 FLOAT NOT NULL,
    p_over_7_5 FLOAT NOT NULL,
    p_over_8_5 FLOAT NOT NULL,
    p_over_9_5 FLOAT NOT NULL,
    p_over_10_5 FLOAT NOT NULL,
    p_over_11_5 FLOAT NOT NULL,
    feature_vector JSONB NOT NULL,
    home_starter_era FLOAT,
    away_starter_era FLOAT,
    ballpark_id INTEGER,
    home_trailing15_rpg FLOAT,
    away_trailing15_rpg FLOAT,
    pitcher_fallback BOOLEAN DEFAULT false,
    actual_total_runs INTEGER,
    actual_over_5_5 BOOLEAN,
    actual_over_6_5 BOOLEAN,
    actual_over_7_5 BOOLEAN,
    actual_over_8_5 BOOLEAN,
    actual_over_9_5 BOOLEAN,
    actual_over_10_5 BOOLEAN,
    actual_over_11_5 BOOLEAN,
    was_correct_5_5 BOOLEAN,
    was_correct_6_5 BOOLEAN,
    was_correct_7_5 BOOLEAN,
    was_correct_8_5 BOOLEAN,
    was_correct_9_5 BOOLEAN,
    was_correct_10_5 BOOLEAN,
    was_correct_11_5 BOOLEAN,
    outcome_filled_at_utc TIMESTAMPTZ,
    UNIQUE (game_pk, game_date)
);

CREATE TABLE IF NOT EXISTS derived.calibration_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    threshold FLOAT NOT NULL,
    accuracy FLOAT NOT NULL,
    sample_size INTEGER NOT NULL,
    recommended_threshold FLOAT NOT NULL,
    covariate_insights JSONB NOT NULL,
    created_at_utc TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS derived.prediction_log (
    id                    BIGSERIAL       PRIMARY KEY,
    game_pk               INTEGER         NOT NULL,
    target                TEXT            NOT NULL,
    team_code             CHAR(3)         NOT NULL,
    inning_number         SMALLINT,
    probability           REAL            NOT NULL,
    confidence_level      TEXT            NOT NULL
                                          CHECK (confidence_level IN ('HIGH', 'LOW')),
    features_snapshot     JSONB           NOT NULL,
    predicted_at_utc      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    actual_outcome        TEXT,
    outcome_filled_at_utc TIMESTAMPTZ,
    was_correct           BOOLEAN
);

CREATE INDEX IF NOT EXISTS prediction_log_game_idx
    ON derived.prediction_log (game_pk);
CREATE INDEX IF NOT EXISTS prediction_log_target_idx
    ON derived.prediction_log (target, predicted_at_utc);
CREATE INDEX IF NOT EXISTS prediction_log_unresolved_idx
    ON derived.prediction_log (game_pk) WHERE actual_outcome IS NULL;
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
