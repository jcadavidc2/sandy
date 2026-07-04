-- Sandy NHL vertical: Database Migration (official NHL API, keyless). Idempotent.
--
-- Regulation vs final: totals follow betting convention (final score, shootout
-- counts as 1 for the winner). The double-chance market is judged AT REGULATION:
-- OT/SO means regulation ended tied, so "home or tie" wins. reg_* columns store
-- the regulation score (= min score on both sides when the game went past REG).

CREATE SCHEMA IF NOT EXISTS nhl;

CREATE TABLE IF NOT EXISTS nhl.teams (
    team_id     INTEGER     PRIMARY KEY,
    abbrev      TEXT        NOT NULL,
    name        TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nhl.games (
    game_id          BIGINT      PRIMARY KEY,
    match_date       DATE        NOT NULL,          -- America/Los_Angeles date
    start_utc        TIMESTAMPTZ,
    season           INTEGER,                       -- e.g. 20232024
    game_type        INTEGER,                       -- 2 regular, 3 playoffs
    status           TEXT        NOT NULL,           -- FUT/LIVE/FINAL
    home_team_id     INTEGER     NOT NULL REFERENCES nhl.teams(team_id),
    away_team_id     INTEGER     NOT NULL REFERENCES nhl.teams(team_id),
    home_goals       INTEGER,                        -- final (incl OT/SO winner goal)
    away_goals       INTEGER,
    last_period_type TEXT,                           -- REG/OT/SO
    reg_home_goals   INTEGER,
    reg_away_goals   INTEGER,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS nhl_games_date_idx   ON nhl.games (match_date);
CREATE INDEX IF NOT EXISTS nhl_games_status_idx ON nhl.games (status);

CREATE TABLE IF NOT EXISTS nhl.team_ratings (
    team_id    INTEGER NOT NULL REFERENCES nhl.teams(team_id),
    as_of_date DATE    NOT NULL,
    attack     REAL,
    defense    REAL,
    PRIMARY KEY (team_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS nhl.game_predictions (
    id                 BIGSERIAL PRIMARY KEY,
    game_id            BIGINT  NOT NULL UNIQUE,
    match_date         DATE    NOT NULL,
    home_team_id       INTEGER NOT NULL,
    away_team_id       INTEGER NOT NULL,
    home_team          TEXT,
    away_team          TEXT,
    lambda_home        REAL,   -- regulation expected goals
    lambda_away        REAL,
    p_home_win_reg     REAL,
    p_tie_reg          REAL,
    p_away_win_reg     REAL,
    p_home_or_tie      REAL,   -- the double-chance market (regulation)
    p_over_5_5         REAL,   -- final-total convention (reg tie => +1 goal)
    p_over_6_5         REAL,
    most_likely_home   INTEGER,
    most_likely_away   INTEGER,
    features           JSONB,
    predicted_at_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_backtest        BOOLEAN NOT NULL DEFAULT FALSE,
    actual_home_goals  INTEGER,
    actual_away_goals  INTEGER,
    actual_total_goals INTEGER,
    actual_reg_result  TEXT,    -- 'H'/'T'/'A' at regulation
    was_correct_double_chance BOOLEAN,
    was_correct_over_5_5      BOOLEAN,
    was_correct_over_6_5      BOOLEAN,
    outcome_filled_at_utc     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS nhl_pred_date_idx ON nhl.game_predictions (match_date);
CREATE INDEX IF NOT EXISTS nhl_pred_unreconciled_idx ON nhl.game_predictions (outcome_filled_at_utc) WHERE outcome_filled_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS nhl.calibration_snapshots (
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
