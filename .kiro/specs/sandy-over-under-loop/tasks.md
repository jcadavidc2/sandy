# Implementation Plan: Sandy Over/Under Feedback Loop

## Overview

Bottom-up implementation of the daily over/under prediction → reconciliation → calibration → retraining feedback loop. Each task builds on the previous, ending with cron scripts that wire everything together. Property-based tests are placed close to the modules they validate to catch errors early.

## Tasks

- [x] 1. Database migration and schema setup
  - [x] 1.1 Create migration file `sandy/migrations/add_over_under_tables.sql`
    - Create `derived.over_under_outcomes` table with all columns from design (game_pk, game_date, team codes, 7 probability columns, feature_vector JSONB, covariate columns, 7 actual_over columns, 7 was_correct columns, outcome_filled_at_utc)
    - Add UNIQUE constraint on (game_pk, game_date)
    - Create `derived.calibration_snapshots` table (snapshot_date, threshold, accuracy, sample_size, recommended_threshold, covariate_insights JSONB, created_at_utc)
    - Use `CREATE TABLE IF NOT EXISTS` for idempotency
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 2. Domain dataclasses and schemas
  - [x] 2.1 Create `sandy/over_under/__init__.py` and `sandy/over_under/schemas.py`
    - Define `OverUnderPrediction` frozen dataclass (game_pk, game_date, home_team_code, away_team_code, game_time_utc, predicted_at_utc, p_over dict, feature_vector dict, covariate fields, pitcher_fallback)
    - Define `CalibrationSnapshot` frozen dataclass (snapshot_date, accuracy_by_threshold dict, recommended_threshold, sample_size, covariate_insights dict, rolling_4w_accuracy)
    - Define `RetrainingResult` frozen dataclass (success, sample_size, new_mae, previous_mae, skipped_reason)
    - Define `STANDARD_THRESHOLDS = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]` constant
    - _Requirements: 1.1, 1.2, 7.1, 8.1_

- [ ] 3. Predictor module
  - [x] 3.1 Create `sandy/over_under/predictor.py`
    - Implement `compute_over_under_probabilities(total_expected_runs, residual_std=2.8)` using `scipy.stats.norm.cdf` for all 7 thresholds
    - Implement `predict_all_games(config, game_date=None)` that calls `predict_game(target="runs")` for each team, sums expected runs, applies normal approximation
    - Handle pitcher fallback: use team season ERA when no probable pitcher announced, set `pitcher_fallback=True`
    - Implement `persist_predictions(engine, predictions)` with upsert on (game_pk, game_date) using `ON CONFLICT ... DO UPDATE`
    - Serialize feature_vector with `json.dumps` and 6-decimal rounding
    - _Requirements: 1.1, 1.2, 1.4, 1.7, 12.5, 13.1_

  - [ ] 3.2 Write property test for probability validity and monotonicity
    - **Property 1: Over/under probabilities are valid and monotonically decreasing**
    - Generate random total_expected_runs ∈ [0, 30], residual_std ∈ (0.1, 10)
    - Assert all P(over T) ∈ [0, 1] and monotonically decreasing as T increases
    - File: `tests/test_pbt_over_under_probabilities.py`
    - **Validates: Requirements 1.1**

  - [ ] 3.3 Write property test for upsert idempotency
    - **Property 3: Prediction upsert is idempotent**
    - Generate random prediction data, call persist_predictions twice, assert same row count
    - File: `tests/test_pbt_over_under_idempotent.py`
    - **Validates: Requirements 1.7**

  - [ ] 3.4 Write property test for feature vector serialization round-trip
    - **Property 11: Feature vector serialization round-trip**
    - Generate random 10-key float dicts (GAME_FEATURE_NAMES keys), serialize with json.dumps + 6-decimal rounding, deserialize, assert each value differs by at most 1e-5
    - File: `tests/test_pbt_over_under_serialization.py`
    - **Validates: Requirements 12.5, 13.1, 13.2, 13.3**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Reconciler module
  - [x] 5.1 Create `sandy/over_under/reconciler.py`
    - Implement `reconcile_over_under(engine)` that queries Final games with NULL actual_total_runs in `derived.over_under_outcomes`
    - Compute actual_total_runs = home_score + away_score from `raw.games`
    - Compute actual_over_T = actual_total_runs > T for all 7 thresholds
    - Compute was_correct_T = (p_over_T >= 0.5) == actual_over_T for all 7 thresholds
    - Set outcome_filled_at_utc = now()
    - Skip rows where actual_total_runs IS NOT NULL (idempotent)
    - Return count of rows updated
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 5.2 Write property test for reconciliation correctness
    - **Property 4: Reconciliation correctness computation**
    - Generate random (probability ∈ [0,1], actual_total_runs ≥ 0, threshold T) triples
    - Assert was_correct_T == ((p >= 0.5) == (actual_total_runs > T))
    - File: `tests/test_pbt_over_under_reconcile.py`
    - **Validates: Requirements 3.2, 3.3**

  - [ ] 5.3 Write property test for reconciliation idempotency
    - **Property 5: Reconciliation idempotency**
    - Generate random pre-filled rows, run reconciliation, assert already-filled rows unchanged
    - File: `tests/test_pbt_over_under_reconcile.py`
    - **Validates: Requirements 3.4**

- [ ] 6. Calibrator module
  - [x] 6.1 Create `sandy/over_under/calibrator.py`
    - Implement `compute_calibration(engine, lookback_days=7)` that queries reconciled outcomes from past N days
    - Compute accuracy at each threshold T = count(was_correct_T == True) / sample_size
    - Identify recommended_threshold as T with highest accuracy
    - Compute per-covariate miss rates grouped by quartile (ballpark_id, home_starter_era, away_starter_era, home_trailing15_rpg, away_trailing15_rpg)
    - Compute rolling_4w_accuracy for primary threshold (6.5)
    - Return None if fewer than 5 reconciled predictions exist
    - Implement `persist_calibration(engine, snapshot)` to insert into `derived.calibration_snapshots`
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.7, 9.3_

  - [ ] 6.2 Write property test for calibration accuracy and optimal threshold
    - **Property 8: Calibration accuracy and optimal threshold selection**
    - Generate random 7-day outcome sets with at least 5 predictions
    - Assert accuracy at T = count(was_correct_T == True) / sample_size
    - Assert recommended_threshold is T with highest accuracy
    - File: `tests/test_pbt_over_under_calibration.py`
    - **Validates: Requirements 7.1, 7.3**

- [ ] 7. Retrainer module
  - [x] 7.1 Create `sandy/over_under/retrainer.py`
    - Implement `retrain_runs_model(config)` that calls existing `train_runs_model()` from `sandy.train.trainer`
    - Load current artifact's validation MAE for comparison
    - If new_mae > previous_mae * 1.2, do NOT overwrite artifact (guard condition), return RetrainingResult with skipped_reason
    - If guard passes, save new artifact via `save_artifact()`, return RetrainingResult with success=True
    - Log training sample size, new MAE, previous MAE
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 7.2 Write property test for model retraining guard condition
    - **Property 9: Model retraining guard condition**
    - Generate random (previous_mae, new_mae) pairs where both are positive floats
    - Assert artifact overwritten iff new_mae <= previous_mae * 1.2
    - File: `tests/test_pbt_over_under_retrainer.py`
    - **Validates: Requirements 8.5**

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Notifier module
  - [ ] 9.1 Create `sandy/over_under/notifier.py`
    - Implement `format_morning_digest(predictions, calibration)` with trust signal and per-game lines sorted by game_time_utc ascending
    - Implement `format_nightly_report(outcomes, calibration, retraining)` with correct/total header, per-game lines with ✅/❌ and top-3 features, calibration one-liner, retraining result
    - Implement `format_no_games_message()` returning "No games scheduled today — no over/under predictions."
    - Implement `format_no_finals_message()` returning "No final scores yet for today's predictions — will retry tomorrow."
    - Implement `send_telegram(message)` using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars, matching existing curl pattern from daily_refresh.sh
    - _Requirements: 1.3, 1.5, 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 7.4, 7.5, 8.4, 10.5_

  - [ ] 9.2 Write property test for morning message sort order
    - **Property 2: Morning message games are sorted by start time**
    - Generate random lists of OverUnderPrediction objects with distinct game_time_utc values
    - Assert formatted morning message lists games in ascending game_time_utc order
    - File: `tests/test_pbt_over_under_morning_sort.py`
    - **Validates: Requirements 1.3**

  - [ ] 9.3 Write property test for nightly report format
    - **Property 6: Nightly report contains correct count and per-game features**
    - Generate random reconciled outcomes with feature vectors
    - Assert header line has correct count of correct/total
    - Assert each game line includes at least 3 feature values from feature_vector
    - File: `tests/test_pbt_over_under_report.py`
    - **Validates: Requirements 4.1, 4.2, 4.3**

  - [ ] 9.4 Write property test for report summary accuracy
    - **Property 7: Report summary accuracy computation**
    - Generate random outcome lists
    - Assert accuracy = count(was_correct_6_5 == True) / total and correct + incorrect == total
    - File: `tests/test_pbt_over_under_report.py`
    - **Validates: Requirements 5.4**

- [ ] 10. CLI commands
  - [ ] 10.1 Create `sandy/cli/over_under_cmd.py`
    - Implement Click group `over-under` with three subcommands: `predict`, `reconcile`, `calibrate`
    - `predict --date --notify`: run predict_all_games, persist, optionally send morning digest
    - `reconcile --date --notify`: run reconcile_over_under, optionally send nightly report
    - `calibrate --notify --weekly`: run compute_calibration + persist, optionally send calibration message
    - Print results to stdout in all cases
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ] 10.2 Register `over-under` group in `sandy/cli/main.py`
    - Import over_under_cmd and add to cli group
    - _Requirements: 11.1_

- [ ] 11. MCP tools
  - [ ] 11.1 Create `sandy/mcp/over_under_tools.py`
    - Implement `handle_get_daily_over_under_predictions(args)`: query today's (or specified date's) predictions, return sorted descending by p_over_6_5
    - Implement `handle_get_over_under_report(args)`: return reconciled outcomes for yesterday (or specified date) with summary {correct, total, accuracy}
    - Implement `handle_get_over_under_calibration(args)`: return latest calibration snapshot with optional weeks_back history
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.2, 5.3, 5.4, 5.5, 7.6, 9.1, 9.2_

  - [ ] 11.2 Register new tools in `sandy/mcp/tools.py`
    - Add 3 new tool definitions to TOOL_DEFINITIONS list (get_daily_over_under_predictions, get_over_under_report, get_over_under_calibration)
    - Add dispatch cases in handle_tool_call
    - _Requirements: 2.1, 5.1, 7.6_

  - [ ] 11.3 Write property test for MCP predictions sort order
    - **Property 10: MCP predictions sorted descending by p_over_6_5**
    - Generate random prediction row sets
    - Assert each consecutive pair satisfies row[i].p_over_6_5 >= row[i+1].p_over_6_5
    - File: `tests/test_pbt_over_under_mcp.py`
    - **Validates: Requirements 2.4**

- [ ] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Cron scripts
  - [ ] 13.1 Create `sandy/scripts/over_under_morning.sh`
    - Source env vars from `~/.sandy_env` and `.env`
    - Activate venv
    - Run `sandy over-under predict --notify`
    - On failure: send Telegram error alert with last 5 lines of stderr
    - _Requirements: 10.1, 10.5_

  - [ ] 13.2 Create `sandy/scripts/over_under_night.sh`
    - Source env vars, activate venv
    - Run `sandy over-under reconcile --notify`
    - Run `sandy train --target runs` (daily retraining)
    - Run `sandy over-under calibrate --notify`
    - On failure at any step: send Telegram error alert
    - _Requirements: 10.2, 10.5_

  - [ ] 13.3 Create `sandy/scripts/over_under_weekly.sh`
    - Source env vars, activate venv
    - Run `sandy over-under calibrate --notify --weekly` (deeper EDA with rolling 4-week trends)
    - On failure: send Telegram error alert
    - _Requirements: 10.3, 10.5_

  - [ ] 13.4 Update `sandy/scripts/crontab.txt`
    - Add morning job at 7:00 AM UTC daily
    - Add nightly job at 11:00 PM UTC daily
    - Add weekly deeper analysis at 11:30 PM UTC on Sundays
    - _Requirements: 10.4_

- [ ] 14. Integration test
  - [ ] 14.1 Create `tests/test_over_under_integration.py`
    - Test end-to-end flow: predict → persist → reconcile → calibrate against test Postgres
    - Test MCP tool handlers with seeded test data
    - Test CLI commands with `--date` override
    - Test idempotency of predict and reconcile
    - Test calibration skips when < 5 predictions
    - Test retraining guard condition with mock artifact
    - _Requirements: 1.7, 3.4, 7.7, 8.5_

- [ ] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All property-based tests are mandatory (not marked with `*`) per user requirement
- Each task builds on previous tasks — no orphaned code
- The implementation reuses existing `predict_game`, `MlbStatsClient`, `Trainer`, and Telegram patterns
- Python 3.11, uv package manager, same EC2/Postgres infrastructure
- Feature vectors saved as JSONB with 6-decimal rounding for EDA
- Daily retraining (not weekly) with 20% MAE degradation guard
- Morning predictions at 7 AM UTC, nightly reconciliation at 11 PM UTC
- Checkpoints ensure incremental validation throughout implementation
