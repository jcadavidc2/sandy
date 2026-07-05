-- Broader goals ladder for the World Cup (football) vertical (added 2026-07-05).
-- Mirrors the MLS/soccer O0.5/O5.5 extension: two new probability columns on
-- football.match_predictions, filled historically by re-running the walk-forward
-- backtest (sandy football backtest).
ALTER TABLE football.match_predictions ADD COLUMN IF NOT EXISTS p_over_0_5 REAL;
ALTER TABLE football.match_predictions ADD COLUMN IF NOT EXISTS p_over_5_5 REAL;

-- football.predictions_meta was created with an expanded column list, so the
-- new columns do NOT flow through automatically: recreate it with the same
-- definition plus p_over_0_5 / p_over_5_5.
DROP VIEW IF EXISTS football.predictions_meta;
CREATE VIEW football.predictions_meta AS
SELECT mp.id,
    mp.fixture_id,
    mp.match_date,
    mp.home_team_id,
    mp.away_team_id,
    mp.predicted_at_utc,
    mp.lambda_home,
    mp.lambda_away,
    mp.p_home_win,
    mp.p_draw,
    mp.p_away_win,
    mp.p_over_0_5,
    mp.p_over_1_5,
    mp.p_over_2_5,
    mp.p_over_3_5,
    mp.p_over_4_5,
    mp.p_over_5_5,
    mp.p_btts,
    mp.most_likely_home,
    mp.most_likely_away,
    mp.scoreline,
    mp.feature_vector,
    mp.p_correct,
    mp.actual_home_goals,
    mp.actual_away_goals,
    mp.actual_total_goals,
    mp.actual_result,
    mp.actual_btts,
    mp.was_correct_result,
    mp.was_correct_over_2_5,
    mp.was_correct_btts,
    mp.was_correct_score,
    mp.outcome_filled_at_utc,
    mp.p_home_win + mp.p_draw AS p_home_or_draw,
    (mp.feature_vector ->> 'atk_home')::real AS atk_home,
    (mp.feature_vector ->> 'atk_away')::real AS atk_away,
    (mp.feature_vector ->> 'def_home')::real AS def_home,
    (mp.feature_vector ->> 'def_away')::real AS def_away,
    (mp.feature_vector ->> 'home_adv')::real AS home_adv,
    th.name AS home_team,
    ta.name AS away_team
FROM football.match_predictions mp
LEFT JOIN football.teams th ON th.team_id = mp.home_team_id
LEFT JOIN football.teams ta ON ta.team_id = mp.away_team_id;
