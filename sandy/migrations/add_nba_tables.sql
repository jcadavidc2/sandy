-- Sandy NBA vertical (ESPN data). Totals ("cestas"/points) via a normal model
-- like MLB's runs stack; winner probability from the same margin model. Idempotent.

CREATE SCHEMA IF NOT EXISTS nba;

CREATE TABLE IF NOT EXISTS nba.teams (
    team_id     INTEGER     PRIMARY KEY,
    name        TEXT        NOT NULL,
    abbrev      TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nba.games (
    event_id      INTEGER     PRIMARY KEY,
    match_date    DATE        NOT NULL,
    start_utc     TIMESTAMPTZ,
    season        INTEGER,
    status        TEXT        NOT NULL,          -- NS/LIVE/FT/PPD
    home_team_id  INTEGER     NOT NULL REFERENCES nba.teams(team_id),
    away_team_id  INTEGER     NOT NULL REFERENCES nba.teams(team_id),
    home_points   INTEGER,
    away_points   INTEGER,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS nba_games_date_idx   ON nba.games (match_date);
CREATE INDEX IF NOT EXISTS nba_games_status_idx ON nba.games (status);

CREATE TABLE IF NOT EXISTS nba.game_predictions (
    id                 BIGSERIAL PRIMARY KEY,
    event_id           INTEGER NOT NULL UNIQUE,
    match_date         DATE    NOT NULL,
    home_team_id       INTEGER NOT NULL,
    away_team_id       INTEGER NOT NULL,
    home_team          TEXT,
    away_team          TEXT,
    exp_home_points    REAL,
    exp_away_points    REAL,
    exp_total          REAL,
    sigma_total        REAL,
    p_home_win         REAL,
    p_over_215_5       REAL,
    p_over_220_5       REAL,
    p_over_225_5       REAL,
    p_over_230_5       REAL,
    p_over_235_5       REAL,
    features           JSONB,
    predicted_at_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_backtest        BOOLEAN NOT NULL DEFAULT FALSE,
    actual_home_points INTEGER,
    actual_away_points INTEGER,
    actual_total       INTEGER,
    actual_winner      TEXT,     -- 'H'/'A'
    was_correct_winner    BOOLEAN,
    was_correct_over_225_5 BOOLEAN,
    outcome_filled_at_utc TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS nba_pred_date_idx ON nba.game_predictions (match_date);
CREATE INDEX IF NOT EXISTS nba_pred_unrec_idx ON nba.game_predictions (outcome_filled_at_utc) WHERE outcome_filled_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS nba.calibration_snapshots (
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
