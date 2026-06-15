-- Sandy Football (World Cup) vertical: Database Migration
-- Creates the `football` schema and its tables for international match
-- ingestion, ratings, predictions, reconciliation and calibration.
-- Idempotent: uses CREATE SCHEMA/TABLE/INDEX IF NOT EXISTS.

CREATE SCHEMA IF NOT EXISTS football;

-- ---------------------------------------------------------------------------
-- Teams (one row per national team, keyed by API-Football team id)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.teams (
    team_id        INTEGER     PRIMARY KEY,        -- API-Football team id
    name           TEXT        NOT NULL,
    fifa_code      VARCHAR(3),                     -- e.g. 'ARG' (nullable)
    country        TEXT,
    confederation  TEXT,                           -- UEFA/CONMEBOL/CONCACAF/CAF/AFC/OFC (filled later)
    fifa_rank      INTEGER,
    logo_url       TEXT,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Matches (one row per fixture, keyed by API-Football fixture id)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.matches (
    fixture_id         INTEGER     PRIMARY KEY,    -- API-Football fixture id
    match_date         DATE        NOT NULL,
    kickoff_utc        TIMESTAMPTZ,
    league_id          INTEGER,                    -- API-Football league id (1 = World Cup)
    season             INTEGER,
    competition        TEXT,                       -- human label, e.g. 'World Cup'
    competition_weight REAL        NOT NULL DEFAULT 1.0,  -- friendly < qualifier < continental < WC
    round              TEXT,                       -- e.g. 'Group Stage - 1'
    home_team_id       INTEGER     NOT NULL REFERENCES football.teams(team_id),
    away_team_id       INTEGER     NOT NULL REFERENCES football.teams(team_id),
    venue_id           INTEGER,
    venue_name         TEXT,
    status             TEXT        NOT NULL,        -- API short status: 'FT','NS','1H','AET','PEN',...
    home_goals         INTEGER,
    away_goals         INTEGER,
    raw_payload_hash   TEXT,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS football_matches_date_idx          ON football.matches (match_date);
CREATE INDEX IF NOT EXISTS football_matches_league_season_idx ON football.matches (league_id, season);
CREATE INDEX IF NOT EXISTS football_matches_status_idx        ON football.matches (status);

-- ---------------------------------------------------------------------------
-- Per-team match statistics (one row per (fixture, team))
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.match_stats (
    fixture_id      INTEGER     NOT NULL REFERENCES football.matches(fixture_id) ON DELETE CASCADE,
    team_id         INTEGER     NOT NULL REFERENCES football.teams(team_id),
    is_home         BOOLEAN     NOT NULL,
    possession      REAL,                           -- percent 0..100
    shots_total     INTEGER,
    shots_on_target INTEGER,
    corners         INTEGER,
    fouls           INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    xg              REAL,                           -- expected goals if available
    raw             JSONB,                          -- source statistics array for re-extraction
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fixture_id, team_id)
);

-- ---------------------------------------------------------------------------
-- Walk-forward team ratings (Dixon-Coles strengths + Elo), as-of a date
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.team_ratings (
    team_id          INTEGER     NOT NULL REFERENCES football.teams(team_id),
    as_of_date       DATE        NOT NULL,
    elo              REAL,
    attack_strength  REAL,
    defense_strength REAL,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (team_id, as_of_date)
);

-- ---------------------------------------------------------------------------
-- Predictions + reconciled outcomes (mirrors derived.over_under_outcomes)
-- One row per fixture; actual_*/was_correct_* filled at reconciliation.
-- No FK on fixture_id so predictions can be written independently.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.match_predictions (
    id                    SERIAL      PRIMARY KEY,
    fixture_id            INTEGER     NOT NULL,
    match_date            DATE        NOT NULL,
    home_team_id          INTEGER     NOT NULL,
    away_team_id          INTEGER     NOT NULL,
    predicted_at_utc      TIMESTAMPTZ NOT NULL,
    lambda_home           REAL        NOT NULL,     -- expected home goals
    lambda_away           REAL        NOT NULL,     -- expected away goals
    p_home_win            REAL        NOT NULL,
    p_draw                REAL        NOT NULL,
    p_away_win            REAL        NOT NULL,
    p_over_1_5            REAL        NOT NULL,
    p_over_2_5            REAL        NOT NULL,
    p_over_3_5            REAL        NOT NULL,
    p_over_4_5            REAL        NOT NULL,
    p_btts                REAL        NOT NULL,
    most_likely_home      INTEGER     NOT NULL,
    most_likely_away      INTEGER     NOT NULL,
    scoreline             JSONB,                    -- top-N scoreline cells with probabilities
    feature_vector        JSONB,
    p_correct             REAL,                     -- meta-model P(our pick correct)
    -- actuals (filled at reconciliation)
    actual_home_goals     INTEGER,
    actual_away_goals     INTEGER,
    actual_total_goals    INTEGER,
    actual_result         CHAR(1),                  -- 'H','D','A'
    actual_btts           BOOLEAN,
    was_correct_result    BOOLEAN,
    was_correct_over_2_5  BOOLEAN,
    was_correct_btts      BOOLEAN,
    was_correct_score     BOOLEAN,
    outcome_filled_at_utc TIMESTAMPTZ,
    UNIQUE (fixture_id)
);
CREATE INDEX IF NOT EXISTS football_predictions_date_idx ON football.match_predictions (match_date);

-- ---------------------------------------------------------------------------
-- Calibration snapshots (mirrors derived.calibration_snapshots)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS football.calibration_snapshots (
    id                    SERIAL      PRIMARY KEY,
    snapshot_date         DATE        NOT NULL,
    market                TEXT        NOT NULL,     -- 'result' | 'over_2_5' | 'btts'
    accuracy              REAL        NOT NULL,
    sample_size           INTEGER     NOT NULL,
    recommended_threshold REAL,
    covariate_insights    JSONB,
    created_at_utc        TIMESTAMPTZ DEFAULT now()
);
