# Requirements Document

## Introduction

The Sandy Over/Under Feedback Loop is a daily prediction, outcome tracking, and calibration improvement cycle focused on the "total runs over 6.5" use case. Each morning the system generates P(total > threshold) for every scheduled game and sends a Telegram digest. Each night it reconciles actual final scores, reports accuracy, and saves feature vectors alongside outcomes. Weekly it runs a covariate analysis to identify which conditions correlate with misses and recommends a refined threshold. The model is retrained weekly with the accumulated outcome data. Three new MCP tools expose all of this to the user via natural-language queries.

The feature builds on the existing `predict_game` / `runs` model, `derived.prediction_log`, `evaluation.reconciler`, and the Telegram notification pattern already in `daily_refresh.sh`. It adds one new table (`derived.over_under_outcomes`), three new CLI commands, three new MCP tools, two new cron jobs, and a weekly retraining hook.

---

## Glossary

- **Over_Under_Predictor**: The subsystem responsible for computing P(total > threshold) for a game using the existing `runs` model artifact.
- **Outcome_Reconciler**: The existing `evaluation.reconciler` module, extended to also populate `derived.over_under_outcomes`.
- **Calibration_Analyzer**: The new module that computes per-covariate miss rates and threshold accuracy curves from `derived.over_under_outcomes`.
- **Telegram_Notifier**: The existing Telegram notification helper in `daily_refresh.sh`, reused by the new scripts.
- **Feature_Vector**: The game-level feature snapshot (10 values from `GAME_FEATURE_NAMES`) saved alongside each prediction.
- **Threshold**: A runs total line (e.g. 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5) used to define the over/under bet.
- **Primary_Threshold**: The threshold the user is currently testing (default 6.5); stored in config and overridable.
- **Miss**: A prediction where P(total > threshold) ≥ 0.5 but the actual total did not exceed the threshold, or vice versa.
- **Covariate**: A feature dimension (e.g. `ballpark_id`, `home_starter_era`, `home_trailing15_rpg`) used to explain miss patterns.
- **Weekly_Retrainer**: The module that retrains the `runs` model artifact using all available outcome data every Sunday.
- **MCP_Server**: The existing `sandy/mcp/server.py` process that exposes Sandy tools to the LLM agent.

---

## Requirements

### Requirement 1: Morning Over/Under Prediction Batch

**User Story:** As a user, I want to receive a Telegram message every morning with P(total > threshold) for all of today's games, so that I can manually select which games to bet on.

#### Acceptance Criteria

1. WHEN the morning prediction job runs, THE Over_Under_Predictor SHALL compute P(total > T) for T ∈ {5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5} for every game on today's MLB schedule.
2. WHEN the morning prediction job runs, THE Over_Under_Predictor SHALL save one row per game to `derived.over_under_outcomes` with columns: `game_pk`, `game_date`, `home_team_code`, `away_team_code`, `predicted_at_utc`, `p_over_5_5`, `p_over_6_5`, `p_over_7_5`, `p_over_8_5`, `p_over_9_5`, `p_over_10_5`, `p_over_11_5`, `feature_vector` (JSONB), `home_starter_era`, `away_starter_era`, `ballpark_id`, `home_trailing15_rpg`, `away_trailing15_rpg`.
3. WHEN the morning prediction job runs, THE Telegram_Notifier SHALL send one message listing all games with their P(over 6.5) probability and the home/away team codes, sorted by game start time ascending (first game of the day first, last game last).
4. WHEN the morning prediction job runs and a game has no probable pitcher announced, THE Over_Under_Predictor SHALL use the team's season ERA as a fallback and SHALL mark the row with `pitcher_fallback = true`.
5. IF the MLB schedule API returns no games for today, THEN THE Telegram_Notifier SHALL send a message stating "No games scheduled today — no over/under predictions."
6. IF the `runs` model artifact is missing, THEN THE Over_Under_Predictor SHALL log an error and SHALL send a Telegram alert: "Over/under prediction failed: runs model artifact not found."
7. THE Over_Under_Predictor SHALL be idempotent — re-running the morning job for the same date SHALL update existing rows rather than inserting duplicates (upsert on `game_pk` + `game_date`).

---

### Requirement 2: Telegram Query for Today's Predictions

**User Story:** As a user, I want to ask Sandy "What are today's over/under predictions?" via Telegram at any time, so that I can retrieve the morning predictions on demand.

#### Acceptance Criteria

1. WHEN the `get_daily_over_under_predictions` MCP tool is called, THE MCP_Server SHALL return all rows from `derived.over_under_outcomes` for today's date, including `game_pk`, `home_team_code`, `away_team_code`, `p_over_6_5`, and all other threshold probabilities.
2. WHEN the `get_daily_over_under_predictions` MCP tool is called with a `date` parameter, THE MCP_Server SHALL return predictions for that specific date instead of today.
3. IF no predictions exist for the requested date, THEN THE MCP_Server SHALL return a response with `predictions: []` and a `message` field explaining that no predictions were found for that date.
4. THE MCP_Server SHALL return results sorted descending by `p_over_6_5`.

---

### Requirement 3: Nightly Outcome Reconciliation

**User Story:** As a user, I want the system to automatically check final scores each night and record whether each prediction was correct, so that I have a complete accuracy log without manual effort.

#### Acceptance Criteria

1. WHEN the nightly reconciliation job runs, THE Outcome_Reconciler SHALL query `raw.games` for all games with `status = 'Final'` that have a corresponding row in `derived.over_under_outcomes` with `actual_total_runs IS NULL`.
2. WHEN a game reaches Final status, THE Outcome_Reconciler SHALL update the corresponding `derived.over_under_outcomes` row with: `actual_total_runs` (home_score + away_score), `actual_over_6_5` (boolean), `was_correct_6_5` (boolean: predicted P > 0.5 matches actual), `outcome_filled_at_utc`.
3. WHEN the nightly reconciliation job runs, THE Outcome_Reconciler SHALL compute `was_correct_T` for all thresholds T ∈ {5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5} and store each as a boolean column.
4. THE Outcome_Reconciler SHALL be idempotent — rows already having `actual_total_runs IS NOT NULL` SHALL NOT be updated again.

---

### Requirement 4: Nightly Telegram Report

**User Story:** As a user, I want to receive a Telegram message each night summarizing how tonight's over/under predictions performed, so that I can quickly see what went right and wrong.

#### Acceptance Criteria

1. WHEN the nightly report job runs, THE Telegram_Notifier SHALL send a message with the format: "Tonight's over/under (6.5): X/Y correct" followed by one line per game showing predicted probability, actual total, and ✅ or ❌.
2. WHEN the nightly report job runs, THE Telegram_Notifier SHALL include for each incorrect prediction the top-3 feature values from the saved `feature_vector` (home_starter_era, away_starter_era, ballpark_id, home_trailing15_rpg, away_trailing15_rpg) to aid pattern recognition.
3. WHEN the nightly report job runs, THE Telegram_Notifier SHALL include for each correct prediction the same top-3 feature values.
4. IF no games from today's predictions have reached Final status by 11 PM, THEN THE Telegram_Notifier SHALL send: "No final scores yet for today's predictions — will retry tomorrow."
5. IF all games are already reconciled (no new finals since last run), THEN THE Telegram_Notifier SHALL NOT send a duplicate report.

---

### Requirement 5: Nightly Report On-Demand via MCP

**User Story:** As a user, I want to ask Sandy "How did the over/under predictions do last night?" via Telegram, so that I can retrieve the nightly report at any time.

#### Acceptance Criteria

1. WHEN the `get_over_under_report` MCP tool is called without a `date` parameter, THE MCP_Server SHALL return the reconciled outcomes for yesterday's games.
2. WHEN the `get_over_under_report` MCP tool is called with a `date` parameter, THE MCP_Server SHALL return the reconciled outcomes for that specific date.
3. THE MCP_Server SHALL return for each game: `home_team_code`, `away_team_code`, `p_over_6_5`, `actual_total_runs`, `was_correct_6_5`, and the feature snapshot fields (`home_starter_era`, `away_starter_era`, `ballpark_id`, `home_trailing15_rpg`, `away_trailing15_rpg`).
4. THE MCP_Server SHALL include a summary field: `{"correct": N, "total": M, "accuracy": N/M}` for the 6.5 threshold.
5. IF no reconciled outcomes exist for the requested date, THEN THE MCP_Server SHALL return `outcomes: []` and a `message` field.

---

### Requirement 6: Daily Calibration Analysis

**User Story:** As a user, I want a daily calibration analysis that tells me how reliable today's predictions are based on recent performance, so that I know whether to trust the model's calls.

#### Acceptance Criteria

1. WHEN the nightly reconciliation job runs, THE Calibration_Analyzer SHALL compute accuracy at each threshold T ∈ {5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5} using all rows in `derived.over_under_outcomes` from the past 7 days that have `actual_total_runs IS NOT NULL`.
2. WHEN the daily calibration job runs, THE Calibration_Analyzer SHALL compute per-covariate miss rates by grouping misses on `ballpark_id`, `home_starter_era` quartile, `away_starter_era` quartile, `home_trailing15_rpg` quartile, and `away_trailing15_rpg` quartile.
3. WHEN the daily calibration job runs, THE Calibration_Analyzer SHALL identify the optimal threshold as the T with the highest accuracy over the past 7 days and SHALL record it in `derived.calibration_snapshots`.
4. WHEN the morning prediction job runs the next day, THE Telegram_Notifier SHALL include a "trust signal" line based on yesterday's calibration: e.g. "Based on last 7 days: model is 71% accurate at 6.5 — trust HIGH confidence picks (≥84%)."
5. IF fewer than 5 reconciled predictions exist for the past 7 days, THE Calibration_Analyzer SHALL skip the analysis and the morning message SHALL say "Not enough history yet for calibration signal."

### Requirement 8: Daily Model Retraining

**User Story:** As a developer, I want the runs model to be retrained every day using the latest outcome data, so that the model improves continuously as more labeled games accumulate.

#### Acceptance Criteria

1. WHEN the nightly job runs (after reconciliation and calibration), THE Weekly_Retrainer SHALL retrain the `runs` model artifact using all available rows in `derived.game_features` joined with actual run totals from `raw.games`.
2. WHEN the daily retraining job completes, THE Weekly_Retrainer SHALL save the new artifact to `config.model.artifact_path("runs")`, overwriting the previous artifact.
3. WHEN the daily retraining job completes, THE Weekly_Retrainer SHALL log the new artifact's training sample size, validation MAE, and the previous artifact's validation MAE for comparison.
4. WHEN the daily retraining job completes, THE Telegram_Notifier SHALL include in the nightly report: "Runs model updated: N games, MAE = X.XX (prev: Y.YY)."
5. IF the new model's validation MAE is more than 20% worse than the previous artifact's MAE, THEN THE Weekly_Retrainer SHALL NOT overwrite the artifact and SHALL note in the nightly report: "Model update skipped: new MAE X.XX worse than current Y.YY."
6. THE retraining SHALL use the most current team statistics — trailing-15 RPG, season OBP, and pitcher ERA are computed from the latest ingested games, so running `sandy refresh` before retraining ensures the features reflect the team's current form after their most recent game.

---

### Requirement 7: Daily Calibration Analysis + Trust Signal

**User Story:** As a user, I want a daily calibration update that tells me how reliable today's predictions are based on recent performance, and I want this incorporated into the morning message so I know which picks to trust.

#### Acceptance Criteria

1. WHEN the nightly job runs (after reconciliation and retraining), THE Calibration_Analyzer SHALL compute accuracy at each threshold T ∈ {5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5} using all rows in `derived.over_under_outcomes` from the past 7 days with `actual_total_runs IS NOT NULL`.
2. WHEN the daily calibration runs, THE Calibration_Analyzer SHALL compute per-covariate miss rates grouped by `ballpark_id`, `home_starter_era` quartile, `away_starter_era` quartile, `home_trailing15_rpg` quartile, and `away_trailing15_rpg` quartile.
3. WHEN the daily calibration runs, THE Calibration_Analyzer SHALL identify the optimal threshold (highest 7-day accuracy) and record it in `derived.calibration_snapshots`.
4. WHEN the morning prediction message is sent, THE Telegram_Notifier SHALL include a trust signal from the most recent calibration snapshot: "📊 Based on last 7 days: predictions above X% have been Y% accurate. Recommended threshold: X%."
5. WHEN the nightly report is sent, THE Telegram_Notifier SHALL include a one-line calibration update: "Calibration: 6.5 accuracy = X% (7-day). Optimal threshold: Y%."
6. WHEN the `get_over_under_calibration` MCP tool is called, THE MCP_Server SHALL return the most recent calibration snapshot including `accuracy_by_threshold`, `recommended_threshold`, `covariate_insights`, and `sample_size`.
7. IF fewer than 5 reconciled predictions exist for the past 7 days, THE Calibration_Analyzer SHALL skip the analysis and the morning message SHALL note "insufficient data for calibration."

---

### Requirement 8: Daily Model Retraining

**User Story:** As a developer, I want the runs model to be retrained every day using the latest outcome data, so that the model improves continuously as new labeled games accumulate.

#### Acceptance Criteria

1. WHEN the nightly job runs (after reconciliation), THE Weekly_Retrainer SHALL retrain the `runs` model artifact using all available rows in `derived.game_features` joined with actual run totals from `raw.games`.
2. WHEN the daily retraining job completes, THE Weekly_Retrainer SHALL save the new artifact to the path configured in `config.model.artifact_path("runs")`, overwriting the previous artifact.
3. WHEN the daily retraining job completes, THE Weekly_Retrainer SHALL log the new artifact's training sample size, validation MAE, and the previous artifact's validation MAE for comparison.
4. WHEN the daily retraining job completes, THE Telegram_Notifier SHALL send a message: "Runs model retrained: N games, MAE = X.XX (prev: Y.YY)."
5. IF the new model's validation MAE is more than 20% worse than the previous artifact's MAE, THEN THE Weekly_Retrainer SHALL NOT overwrite the artifact and SHALL send a Telegram alert: "Retraining aborted: new model MAE X.XX is worse than current Y.YY."
6. THE Weekly_Retrainer SHALL use the same `Trainer` class and `LightGBM` configuration already used by `sandy train --target runs`, with no new hyperparameters introduced.
7. THE model SHALL always use the most current team covariates — trailing-15 RPG, season OBP, starter ERA — which are updated daily by the morning refresh before predictions run. This ensures predictions always reflect the team's state after their most recent game.

---

### Requirement 9: Accuracy Trend Tracking

**User Story:** As a user, I want to see whether the model's accuracy is improving over time, so that I can evaluate whether the feedback loop is working.

#### Acceptance Criteria

1. THE `derived.calibration_snapshots` table SHALL store one row per weekly run, enabling time-series queries over accuracy at each threshold.
2. WHEN the `get_over_under_calibration` MCP tool is called with `weeks_back` > 1, THE MCP_Server SHALL return accuracy values for each of the requested weeks so the caller can observe the trend.
3. THE Calibration_Analyzer SHALL compute a rolling 4-week accuracy for the primary threshold (6.5) and include it in the `covariate_insights` JSONB field as `rolling_4w_accuracy`.

---

### Requirement 10: Cron Schedule Integration

**User Story:** As an operator, I want the morning prediction, nightly reconciliation/report, and weekly calibration/retraining jobs to run automatically on the EC2 instance, so that the loop requires no manual intervention.

#### Acceptance Criteria

1. THE system SHALL provide a `sandy/scripts/over_under_morning.sh` script that runs the morning prediction batch and sends the Telegram digest, callable standalone or from cron.
2. THE system SHALL provide a `sandy/scripts/over_under_night.sh` script that runs nightly reconciliation and sends the Telegram report, callable standalone or from cron.
3. THE system SHALL provide a `sandy/scripts/over_under_weekly.sh` script that runs calibration analysis and model retraining, callable standalone or from cron.
4. THE system SHALL update `sandy/scripts/crontab.txt` with three new entries: morning job at 6:05 AM UTC daily, nightly job at 11:00 PM UTC daily (reconcile + calibrate + retrain), weekly deeper analysis at 11:30 PM UTC on Sundays (covariate EDA report).
5. WHEN any of the three scripts fails, THE Telegram_Notifier SHALL send an error alert with the script name and the last 5 lines of stderr.

---

### Requirement 11: CLI Commands

**User Story:** As a developer, I want CLI commands for each phase of the loop, so that I can run and test each step manually without waiting for cron.

#### Acceptance Criteria

1. THE system SHALL expose a `sandy over-under predict` CLI command that runs the morning prediction batch for today (or a `--date` override) and prints results to stdout.
2. THE system SHALL expose a `sandy over-under reconcile` CLI command that runs nightly reconciliation for today (or a `--date` override) and prints the number of rows updated.
3. THE system SHALL expose a `sandy over-under calibrate` CLI command that runs the weekly calibration analysis and prints the threshold accuracy table and covariate insights.
4. WHEN `sandy over-under predict` is run with `--notify`, THE Telegram_Notifier SHALL send the morning Telegram digest in addition to printing to stdout.
5. WHEN `sandy over-under reconcile` is run with `--notify`, THE Telegram_Notifier SHALL send the nightly Telegram report in addition to printing to stdout.

---

### Requirement 12: Data Persistence and Schema

**User Story:** As a developer, I want all prediction and outcome data stored in Postgres with a stable schema, so that I can run ad-hoc EDA queries and the data survives restarts.

#### Acceptance Criteria

1. THE system SHALL create a `derived.over_under_outcomes` table with at minimum: `id` (serial PK), `game_pk` (integer), `game_date` (date), `home_team_code` (varchar), `away_team_code` (varchar), `predicted_at_utc` (timestamptz), `p_over_5_5` through `p_over_11_5` (float), `feature_vector` (JSONB), `home_starter_era` (float), `away_starter_era` (float), `ballpark_id` (integer), `home_trailing15_rpg` (float), `away_trailing15_rpg` (float), `pitcher_fallback` (boolean), `actual_total_runs` (integer nullable), `actual_over_6_5` (boolean nullable), `was_correct_5_5` through `was_correct_11_5` (boolean nullable), `outcome_filled_at_utc` (timestamptz nullable).
2. THE system SHALL create a `derived.calibration_snapshots` table with at minimum: `id` (serial PK), `snapshot_date` (date), `threshold` (float), `accuracy` (float), `sample_size` (integer), `recommended_threshold` (float), `covariate_insights` (JSONB), `created_at_utc` (timestamptz).
3. THE `derived.over_under_outcomes` table SHALL have a unique constraint on `(game_pk, game_date)` to enforce idempotency.
4. THE system SHALL provide a migration script at `sandy/migrations/add_over_under_tables.sql` that creates both tables idempotently (`CREATE TABLE IF NOT EXISTS`).
5. THE Feature_Vector stored in `feature_vector` JSONB SHALL contain all 10 keys from `GAME_FEATURE_NAMES` as defined in `sandy/features/schema.py`.

---

### Requirement 13: Parser and Serialization Round-Trip

**User Story:** As a developer, I want the feature vector serialization to be reliable, so that saved JSONB data can be deserialized back to the exact same float values for EDA.

#### Acceptance Criteria

1. THE Over_Under_Predictor SHALL serialize feature vectors to JSONB using `json.dumps` with float values rounded to 6 decimal places.
2. WHEN a feature vector is read back from `derived.over_under_outcomes.feature_vector`, THE system SHALL deserialize it to a `dict[str, float]` with the same 10 keys as `GAME_FEATURE_NAMES`.
3. FOR ALL valid game feature vectors, serializing then deserializing SHALL produce a dict where each value differs from the original by at most 1e-5 (round-trip property).
4. IF a stored `feature_vector` is missing any key from `GAME_FEATURE_NAMES`, THEN THE system SHALL substitute `0.0` for the missing key and SHALL log a warning.
