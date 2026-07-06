-- Sandy NFL vertical (ESPN data). Mirrors NBA's schema: totals (points) via a
-- normal model, winner probability from the same margin model. NFL specifics:
-- games can end TIED (~2/season) → actual_winner is 'H'/'A'/'T' and
-- was_correct_winner stays NULL on ties. Idempotent.

CREATE SCHEMA IF NOT EXISTS nfl;

CREATE TABLE IF NOT EXISTS nfl.teams (
    team_id     INTEGER     PRIMARY KEY,
    name        TEXT        NOT NULL,
    abbrev      TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nfl.games (
    event_id      INTEGER     PRIMARY KEY,
    match_date    DATE        NOT NULL,
    start_utc     TIMESTAMPTZ,
    season        INTEGER,
    status        TEXT        NOT NULL,          -- NS/LIVE/FT/PPD
    home_team_id  INTEGER     NOT NULL REFERENCES nfl.teams(team_id),
    away_team_id  INTEGER     NOT NULL REFERENCES nfl.teams(team_id),
    home_points   INTEGER,
    away_points   INTEGER,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS nfl_games_date_idx   ON nfl.games (match_date);
CREATE INDEX IF NOT EXISTS nfl_games_status_idx ON nfl.games (status);

CREATE TABLE IF NOT EXISTS nfl.game_predictions (
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
    p_over_37_5        REAL,
    p_over_41_5        REAL,
    p_over_44_5        REAL,
    p_over_47_5        REAL,
    p_over_51_5        REAL,
    features           JSONB,
    predicted_at_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_backtest        BOOLEAN NOT NULL DEFAULT FALSE,
    actual_home_points INTEGER,
    actual_away_points INTEGER,
    actual_total       INTEGER,
    actual_winner      TEXT,     -- 'H'/'A'/'T' (NFL ties exist)
    was_correct_winner     BOOLEAN,  -- NULL on ties
    was_correct_over_44_5  BOOLEAN,
    outcome_filled_at_utc  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS nfl_pred_date_idx ON nfl.game_predictions (match_date);
CREATE INDEX IF NOT EXISTS nfl_pred_unrec_idx ON nfl.game_predictions (outcome_filled_at_utc) WHERE outcome_filled_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS nfl.calibration_snapshots (
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
