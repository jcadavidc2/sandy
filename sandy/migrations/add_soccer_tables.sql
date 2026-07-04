-- Sandy multi-league soccer vertical (Colombia, México, España, Inglaterra).
-- Same shape as the mls schema but league-tagged everywhere. ESPN data. Idempotent.

CREATE SCHEMA IF NOT EXISTS soccer;

CREATE TABLE IF NOT EXISTS soccer.teams (
    team_id     INTEGER     PRIMARY KEY,      -- ESPN team ids are globally unique
    name        TEXT        NOT NULL,
    abbrev      TEXT,
    league      TEXT,                          -- col/mex/esp/eng (last seen)
    logo_url    TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS soccer.matches (
    event_id      INTEGER     PRIMARY KEY,
    league        TEXT        NOT NULL,
    match_date    DATE        NOT NULL,
    kickoff_utc   TIMESTAMPTZ,
    season        INTEGER,
    status        TEXT        NOT NULL,
    home_team_id  INTEGER     NOT NULL REFERENCES soccer.teams(team_id),
    away_team_id  INTEGER     NOT NULL REFERENCES soccer.teams(team_id),
    home_goals    INTEGER,
    away_goals    INTEGER,
    home_corners  INTEGER,
    away_corners  INTEGER,
    stats_filled_at_utc TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS soccer_matches_lg_date_idx ON soccer.matches (league, match_date);
CREATE INDEX IF NOT EXISTS soccer_matches_status_idx  ON soccer.matches (status);

CREATE TABLE IF NOT EXISTS soccer.match_stats (
    event_id        INTEGER NOT NULL REFERENCES soccer.matches(event_id) ON DELETE CASCADE,
    team_id         INTEGER NOT NULL REFERENCES soccer.teams(team_id),
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

CREATE TABLE IF NOT EXISTS soccer.team_ratings (
    team_id        INTEGER NOT NULL REFERENCES soccer.teams(team_id),
    league         TEXT    NOT NULL,
    as_of_date     DATE    NOT NULL,
    attack         REAL,
    defense        REAL,
    corner_attack  REAL,
    corner_defense REAL,
    PRIMARY KEY (team_id, league, as_of_date)
);

CREATE TABLE IF NOT EXISTS soccer.match_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    event_id            INTEGER NOT NULL UNIQUE,
    league              TEXT    NOT NULL,
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
    p_home_or_draw      REAL,
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
    features            JSONB,
    predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_backtest         BOOLEAN NOT NULL DEFAULT FALSE,
    actual_home_goals   INTEGER,
    actual_away_goals   INTEGER,
    actual_total_goals  INTEGER,
    actual_result       TEXT,
    actual_total_corners INTEGER,
    was_correct_double_chance BOOLEAN,
    was_correct_over_2_5      BOOLEAN,
    was_correct_corners_9_5   BOOLEAN,
    outcome_filled_at_utc     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS soccer_pred_lg_date_idx ON soccer.match_predictions (league, match_date);
CREATE INDEX IF NOT EXISTS soccer_pred_unrec_idx ON soccer.match_predictions (outcome_filled_at_utc) WHERE outcome_filled_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS soccer.calibration_snapshots (
    id                    BIGSERIAL PRIMARY KEY,
    snapshot_date         DATE NOT NULL,
    league                TEXT NOT NULL,
    market                TEXT NOT NULL,
    lookback_days         INTEGER,
    sample_size           INTEGER NOT NULL,
    accuracy              REAL,
    brier                 REAL,
    reliability           JSONB,
    recommended_threshold REAL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
