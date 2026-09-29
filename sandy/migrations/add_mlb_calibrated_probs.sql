-- Calibrated MLB over/under probabilities (2026-09-28, see sandy/over_under/calibration.py).
-- The raw Normal-CDF columns p_over_* stay untouched (backtest safety hash +
-- legacy meta_over_5_5 read them); the calibrated ones are NEW nullable columns
-- filled at predict time and back-filled walk-forward. Idempotent.
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_5_5  DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_6_5  DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_7_5  DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_8_5  DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_9_5  DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_10_5 DOUBLE PRECISION;
ALTER TABLE derived.over_under_outcomes ADD COLUMN IF NOT EXISTS p_cal_over_11_5 DOUBLE PRECISION;

-- The shared betmeta view must expose the new columns. `SELECT *` is frozen at
-- view creation and CREATE OR REPLACE refuses to reorder columns, so drop+create.
DROP VIEW IF EXISTS derived.mlb_predictions_meta;
CREATE VIEW derived.mlb_predictions_meta AS
SELECT *, home_team_code AS home_team, away_team_code AS away_team,
       game_date AS match_date
FROM derived.over_under_outcomes;
