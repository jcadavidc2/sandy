-- MLB per-line meta-model input (added 2026-07-05): expose the reconciled MLB
-- over/under predictions in the shared betmeta shape (match_date / home_team /
-- away_team) so sandy.betmeta SPECS["mlb"] can train a per-line meta for the
-- dashboard. The daily MLB digest pipeline (over_under/*) is untouched.
CREATE OR REPLACE VIEW derived.mlb_predictions_meta AS
SELECT *, home_team_code AS home_team, away_team_code AS away_team,
       game_date AS match_date
FROM derived.over_under_outcomes;
