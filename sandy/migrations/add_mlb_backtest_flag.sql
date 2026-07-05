-- MLB walk-forward backtest flag (added 2026-07-05): mark historical replay rows
-- in derived.over_under_outcomes so live daily predictions (is_backtest = FALSE,
-- the column default — live insert paths don't name the column and are unaffected)
-- stay distinguishable from the leakage-free 2023-2026 backfill produced by
-- sandy/over_under/backtest.py. Mirrors the is_backtest column every other
-- vertical's predictions table (mls/nhl/nba/soccer) already has.
ALTER TABLE derived.over_under_outcomes
    ADD COLUMN IF NOT EXISTS is_backtest BOOLEAN NOT NULL DEFAULT FALSE;
