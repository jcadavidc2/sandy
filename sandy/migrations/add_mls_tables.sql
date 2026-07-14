-- Sandy MLS vertical: Database Migration
-- Mirrors the football schema pattern. Data source is ESPN's public API
-- (keyless): scoreboard for fixtures/scores, per-event summary for team stats
-- (corners, shots, possession, ...). Idempotent.

CREATE SCHEMA IF NOT EXISTS mls;

-- One row per club, keyed by ESPN team id.
CREATE TABLE IF NOT EXISTS mls.teams (
    team_id     INTEGER     PRIMARY KEY,
    name        TEXT        NOT NULL,
    abbrev      TEXT,
    logo_url    TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per match, keyed by ESPN event id. Status normalized: NS/LIVE/FT/PPD.
CREATE TABLE IF NOT EXISTS mls.matches (
    event_id      INTEGER     PRIMARY KEY,
    match_date    DATE        NOT NULL,           -- gym-display timezone (America/Los_Angeles) date
    kickoff_utc   TIMESTAMPTZ,
    season        INTEGER,
    status        TEXT        NOT NULL,
    home_team_id  INTEGER     NOT NULL REFERENCES mls.teams(team_id),
    away_team_id  INTEGER     NOT NULL REFERENCES mls.teams(team_id),
    home_goals    INTEGER,
    away_goals    INTEGER,
    home_corners  INTEGER,                        -- denormalized for the corners model
    away_corners  INTEGER,
    stats_filled_at_utc TIMESTAMPTZ,              -- summary fetched?
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mls_matches_date_idx   ON mls.matches (match_date);
CREATE INDEX IF NOT EXISTS mls_matches_status_idx ON mls.matches (status);

-- Per-team match statistics (covariate capture; one row per (event, team)).
CREATE TABLE IF NOT EXISTS mls.match_stats (
    event_id        INTEGER NOT NULL REFERENCES mls.matches(event_id) ON DELETE CASCADE,
    team_id         INTEGER NOT NULL REFERENCES mls.teams(team_id),
    is_home         BOOLEAN NOT NULL,
    corners         INTEGER,
    total_shots     INTEGER,
    shots_on_target INTEGER,
    possession_pct  REAL,
    fouls           INTEGER,
    offsides        INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    saves           INTEGER,
    PRIMARY KEY (event_id, team_id)
);

-- Model snapshots (goals + corners strengths per team).
CREATE TABLE IF NOT EXISTS mls.team_ratings (
    team_id        INTEGER NOT NULL REFERENCES mls.teams(team_id),
    as_of_date     DATE    NOT NULL,
    attack         REAL,
    defense        REAL,
    corner_attack  REAL,
    corner_defense REAL,
    PRIMARY KEY (team_id, as_of_date)
);

-- Predictions + reconciliation + covariates audit trail.
CREATE TABLE IF NOT EXISTS mls.match_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    event_id            INTEGER NOT NULL UNIQUE,
    match_date          DATE    NOT NULL,
    home_team_id        INTEGER NOT NULL,
    away_team_id        INTEGER NOT NULL,
    home_team           TEXT,
    away_team           TEXT,
    lambda_home         REAL,
    lambda_away         REAL,
    p_home_win          REAL,
    p_draw              REAL,
    p_away_win          REAL,
    p_home_or_draw      REAL,   -- the double-chance market: (home wins or ties) vs away wins
    p_over_1_5          REAL,
    p_over_2_5          REAL,
    p_over_3_5          REAL,
    p_over_4_5          REAL,
    corner_lambda_home  REAL,
    corner_lambda_away  REAL,
    p_corners_over_8_5  REAL,
    p_corners_over_9_5  REAL,
    p_corners_over_10_5 REAL,
    p_corners_over_11_5 REAL,
    most_likely_home    INTEGER,
    most_likely_away    INTEGER,
    features            JSONB,  -- covariates at prediction time (rolling form, rest, ...)
    predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_backtest         BOOLEAN NOT NULL DEFAULT FALSE,
    -- reconciliation
    actual_home_goals   INTEGER,
    actual_away_goals   INTEGER,
    actual_total_goals  INTEGER,
    actual_result       TEXT,    -- 'H'/'D'/'A'
    actual_total_corners INTEGER,
    was_correct_double_chance BOOLEAN,
    was_correct_over_2_5      BOOLEAN,
    was_correct_corners_9_5   BOOLEAN,
    outcome_filled_at_utc     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS mls_pred_date_idx ON mls.match_predictions (match_date);
CREATE INDEX IF NOT EXISTS mls_pred_unreconciled_idx ON mls.match_predictions (outcome_filled_at_utc) WHERE outcome_filled_at_utc IS NULL;

-- Calibration snapshots per market ('double_chance' | 'over_2_5' | 'corners_over_9_5').
CREATE TABLE IF NOT EXISTS mls.calibration_snapshots (
    id                    BIGSERIAL PRIMARY KEY,
    snapshot_date         DATE NOT NULL,
    market                TEXT NOT NULL,
    lookback_days         INTEGER,
    sample_size           INTEGER NOT NULL,
    accuracy              REAL,
    brier                 REAL,
    reliability           JSONB,
    recommended_threshold REAL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Broader goals + corners ladders (added 2026-07-05).
ALTER TABLE mls.match_predictions ADD COLUMN IF NOT EXISTS p_over_0_5          REAL;
ALTER TABLE mls.match_predictions ADD COLUMN IF NOT EXISTS p_over_5_5          REAL;
ALTER TABLE mls.match_predictions ADD COLUMN IF NOT EXISTS p_corners_over_7_5  REAL;
ALTER TABLE mls.match_predictions ADD COLUMN IF NOT EXISTS p_corners_over_12_5 REAL;

-- Playoffs covariate (2026-07-14): per-event ESPN stage slug + its model flag.
ALTER TABLE mls.matches           ADD COLUMN IF NOT EXISTS stage      TEXT;
ALTER TABLE mls.match_predictions ADD COLUMN IF NOT EXISTS is_playoff REAL;
