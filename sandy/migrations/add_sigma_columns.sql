-- Migration: Add volatility model columns to over_under_outcomes
-- These columns store per-prediction expected runs and the matchup-specific σ
-- used in the normal approximation for P(total > T).
--
-- Safe to run multiple times (IF NOT EXISTS).

ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS home_expected_runs FLOAT;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS away_expected_runs FLOAT;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS sigma_used FLOAT;
