-- Over/Under Feedback Loop: Database Migration
-- Creates tables for daily over/under predictions and calibration tracking.
-- Idempotent: uses CREATE TABLE IF NOT EXISTS.

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
