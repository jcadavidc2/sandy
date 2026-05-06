# Requirements Document

## Introduction

Sandy Phase 2 extends the offline prediction system into a live, conversational assistant. It adds on-demand live game state fetching from the MLB API, exposes Sandy's prediction functions as MCP tools for OpenCLAW (the user's Telegram-connected agent), introduces confidence signals that contextualize predictions against base rates, and establishes a prediction logging + self-evaluation pipeline for calibration tracking.

This phase combines the original Phase 2 (live game state) and Phase 3 (chat agent integration) from the roadmap into a single deliverable, plus adds confidence and self-evaluation capabilities.

## Glossary

- **Sandy**: The MLB prediction system; the Python package providing prediction functions, feature engineering, and data pipelines.
- **Live_State_Client**: The module responsible for fetching current game state from the MLB Stats API on demand.
- **MCP_Server**: The Model Context Protocol server that wraps Sandy's functions as callable tools for external agents.
- **OpenCLAW**: The user's existing LLM agent running on EC2 (127.0.0.1:18789), connected to Telegram via bot token.
- **Confidence_Assessor**: The module that compares a prediction to the relevant base rate and produces a confidence classification.
- **Prediction_Logger**: The module that persists every prediction Sandy makes and later reconciles outcomes for calibration analysis.
- **Calibration_Reporter**: The module that computes accuracy, calibration, and performance breakdowns from logged predictions.
- **MLB_Stats_API**: The free, unauthenticated MLB statistics API (statsapi.mlb.com).
- **Base_Rate**: The historical average probability for a given prediction target (e.g., ~72% for reached_base per inning).
- **Shutdown_Features**: Additional features designed to detect situations where a pitcher is dominating and baserunners are unlikely.
- **Game_State**: A structured representation of the current state of a live MLB game (inning, score, pitcher, batters due up, recent events).

## Requirements

### Requirement 1: On-Demand Live Game State Fetching

**User Story:** As a user watching a live game, I want Sandy to fetch the current game state when I ask a question, so that predictions are based on what is actually happening right now.

#### Acceptance Criteria

1. WHEN a team code is provided, THE Live_State_Client SHALL fetch the current game state from the MLB_Stats_API for the active game involving that team.
2. WHEN the MLB_Stats_API returns game data, THE Live_State_Client SHALL extract and return: current inning number, inning half (top/bottom), home score, away score, current pitcher name and pitch count, the next three batters due up, and a summary of what happened in the previous inning.
3. WHEN no active game exists for the requested team, THE Live_State_Client SHALL return a structured error indicating no game is in progress.
4. THE Live_State_Client SHALL include a staleness indicator showing the UTC timestamp of when the data was fetched and the elapsed seconds since fetch.
5. WHEN the MLB_Stats_API is unreachable or returns an error, THE Live_State_Client SHALL raise a descriptive exception without crashing the process.
6. THE Live_State_Client SHALL make exactly one HTTP request per invocation (on-demand, no background polling).
7. THE Live_State_Client SHALL be importable as a Python function: `get_live_game_state(team_code: str) -> LiveGameState`.

### Requirement 2: Live Game State Data Structure

**User Story:** As a developer integrating Sandy with agents, I want a well-defined data structure for live game state, so that downstream consumers can reliably access game information.

#### Acceptance Criteria

1. THE Live_State_Client SHALL return a frozen dataclass `LiveGameState` containing: game_pk, inning_number, inning_half, home_team_code, away_team_code, home_score, away_score, current_pitcher_name, current_pitcher_id, pitch_count, batters_due_up (list of up to 3 player names), previous_inning_summary (string), fetched_at_utc (datetime), and is_final (bool).
2. WHEN the game is in a state where batters due up cannot be determined, THE Live_State_Client SHALL return an empty list for batters_due_up.
3. WHEN the game has not yet started, THE Live_State_Client SHALL return inning_number as 0 and scores as 0.
4. FOR ALL valid LiveGameState objects, serializing to JSON then deserializing SHALL produce an equivalent object (round-trip property).

### Requirement 3: MCP Server Exposing Sandy Tools

**User Story:** As a user chatting on Telegram, I want OpenCLAW to call Sandy's prediction functions via MCP, so that I get natural-language answers to baseball questions without running CLI commands.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose the following tools: `get_todays_schedule`, `get_live_game_state`, `predict_reached_base`, `predict_game_winner`, `predict_runs`, and `get_player_stats`.
2. THE MCP_Server SHALL accept tool invocations over stdio transport (compatible with OpenCLAW's MCP client).
3. WHEN a tool is invoked with valid parameters, THE MCP_Server SHALL call the corresponding Sandy Python function and return the result as structured JSON.
4. WHEN a tool is invoked with invalid parameters, THE MCP_Server SHALL return a structured error message without crashing the server process.
5. THE MCP_Server SHALL include the live game state in prediction tool responses so the calling agent can present it to the user for verification.
6. THE MCP_Server SHALL be registerable in OpenCLAW's MCP configuration file as a local stdio server.

### Requirement 4: MCP Tool Schemas and Contracts

**User Story:** As a developer maintaining the MCP integration, I want each tool to have a well-defined input/output schema, so that OpenCLAW can correctly invoke tools and parse responses.

#### Acceptance Criteria

1. THE MCP_Server SHALL define JSON Schema for each tool's input parameters and output structure.
2. WHEN `predict_reached_base` is invoked, THE MCP_Server SHALL accept parameters: team_code (string), opponent_code (string, optional), and inning (integer, optional); and SHALL return probability, confidence assessment, top features, and the live game state used.
3. WHEN `predict_game_winner` is invoked, THE MCP_Server SHALL accept parameters: team_code (string) and opponent_code (string, optional); and SHALL return win probability and confidence assessment.
4. WHEN `predict_runs` is invoked, THE MCP_Server SHALL accept parameters: team_code (string) and opponent_code (string, optional); and SHALL return expected runs and confidence assessment.
5. WHEN `get_live_game_state` is invoked, THE MCP_Server SHALL accept parameter: team_code (string); and SHALL return the full LiveGameState structure.
6. WHEN `get_todays_schedule` is invoked with no parameters, THE MCP_Server SHALL return the list of today's scheduled games with probable pitchers.
7. WHEN `get_player_stats` is invoked, THE MCP_Server SHALL accept parameter: player_name (string); and SHALL return the player's season statistics relevant to predictions.

### Requirement 5: Confidence Signal Generation

**User Story:** As a user receiving predictions, I want to know whether Sandy's prediction is meaningfully different from the base rate, so that I can tell when the model is actually detecting something interesting versus just returning noise.

#### Acceptance Criteria

1. THE Confidence_Assessor SHALL classify every prediction as either HIGH confidence or LOW confidence.
2. WHEN a prediction is within ±5 percentage points of the base rate for its target, THE Confidence_Assessor SHALL classify it as LOW confidence with explanation "close to base rate, nothing unusual detected."
3. WHEN a prediction deviates more than 5 percentage points from the base rate for its target, THE Confidence_Assessor SHALL classify it as HIGH confidence with a natural-language explanation of the primary contributing factors.
4. THE Confidence_Assessor SHALL use the following base rates: reached_base = 0.72, game_winner = 0.50, runs (mean per team per game) = 4.5.
5. THE Confidence_Assessor SHALL include the base rate value and the deviation magnitude in its output.
6. FOR ALL predictions, THE Confidence_Assessor SHALL produce a confidence result (the function is total — it never fails for valid prediction inputs).

### Requirement 6: Shutdown Detection Features

**User Story:** As a user, I want Sandy to detect "shutdown" situations where a pitcher is dominating, so that the model can identify below-average reached-base probabilities with high confidence.

#### Acceptance Criteria

1. WHEN live game state is available, THE Live_State_Client SHALL compute shutdown features: pitcher_zero_baserunner_innings (count of consecutive innings with no baserunners), is_bottom_of_order (true if batting order spots 7-8-9 are due up), pitcher_k_rate_vs_team_k_rate (ratio of pitcher's game K-rate to the batting team's season K-rate), and is_fresh_reliever (true if current pitcher has thrown fewer than 20 pitches in the game and is not the starter).
2. WHEN live game state is not available (pre-game prediction), THE Live_State_Client SHALL return null for all shutdown features.
3. THE Confidence_Assessor SHALL reference active shutdown features in its HIGH confidence explanation when the prediction is below base rate.
4. WHEN pitcher_zero_baserunner_innings is 3 or greater, THE Confidence_Assessor SHALL flag this as a primary factor in the confidence explanation.

### Requirement 7: Prediction Logging

**User Story:** As a user who wants Sandy to improve over time, I want every prediction logged with its context, so that Sandy can later evaluate its own accuracy.

#### Acceptance Criteria

1. THE Prediction_Logger SHALL persist every prediction to a Postgres table `derived.prediction_log` with columns: id (serial PK), game_pk, target (string), team_code, inning_number (nullable), probability, confidence_level (HIGH/LOW), features_snapshot (JSONB), predicted_at_utc (timestamp), actual_outcome (nullable), outcome_filled_at_utc (nullable), was_correct (nullable boolean).
2. WHEN a prediction is made via the MCP_Server, THE Prediction_Logger SHALL automatically log it without requiring explicit caller action.
3. WHEN a prediction is made via the CLI, THE Prediction_Logger SHALL automatically log it without requiring explicit caller action.
4. IF the database is unreachable when logging a prediction, THEN THE Prediction_Logger SHALL log a warning and continue without blocking the prediction response.

### Requirement 8: Outcome Reconciliation

**User Story:** As a user, I want Sandy to automatically fill in what actually happened after a game finishes, so that calibration data accumulates without manual effort.

#### Acceptance Criteria

1. WHEN a game reaches "Final" status, THE Prediction_Logger SHALL fill in actual_outcome and was_correct for all predictions logged against that game_pk.
2. FOR reached_base predictions, THE Prediction_Logger SHALL set actual_outcome to true if the team reached base in that inning, false otherwise.
3. FOR game_winner predictions, THE Prediction_Logger SHALL set actual_outcome to true if the predicted team won, false otherwise.
4. FOR runs predictions, THE Prediction_Logger SHALL set actual_outcome to the actual runs scored by the team.
5. THE Prediction_Logger SHALL provide a `reconcile_outcomes()` function that can be called on-demand or scheduled to backfill outcomes for all unresolved predictions whose games have finished.
6. WHEN reconcile_outcomes is called, THE Prediction_Logger SHALL only update rows where actual_outcome is currently null and the game status is "Final".

### Requirement 9: Calibration and Self-Evaluation Reporting

**User Story:** As a user, I want Sandy to report on its own accuracy and calibration, so that I know which predictions to trust and which to take with a grain of salt.

#### Acceptance Criteria

1. THE Calibration_Reporter SHALL compute accuracy (percentage correct) grouped by: target, confidence level, and inning number.
2. THE Calibration_Reporter SHALL compute calibration buckets: for predictions binned into 10-percentage-point ranges (0-10%, 10-20%, ..., 90-100%), report the actual outcome rate within each bucket.
3. THE Calibration_Reporter SHALL produce a summary report as a Python dataclass containing: total predictions, accuracy by target, accuracy by confidence level, calibration buckets, and date range covered.
4. WHEN fewer than 10 predictions exist in a bucket, THE Calibration_Reporter SHALL mark that bucket as "insufficient data" rather than reporting a potentially misleading percentage.
5. THE Calibration_Reporter SHALL expose a `get_calibration_report(days: int = 7) -> CalibrationReport` function importable from Python.
6. THE MCP_Server SHALL expose a `get_calibration_report` tool so agents can query Sandy's self-assessment.

### Requirement 10: Agent-Consumable Calibration Metadata

**User Story:** As an agent (OpenCLAW), I want to know what Sandy is good and bad at, so that I can qualify my answers to the user appropriately.

#### Acceptance Criteria

1. THE Calibration_Reporter SHALL produce a natural-language summary suitable for inclusion in agent system prompts (e.g., "Sandy is well-calibrated for reached_base HIGH confidence predictions but tends to overestimate game_winner probabilities").
2. WHEN the calibration report shows a target with accuracy below 55% over the reporting window, THE Calibration_Reporter SHALL flag it as "unreliable for this target."
3. WHEN the calibration report shows a target with accuracy above 65% for HIGH confidence predictions, THE Calibration_Reporter SHALL flag it as "reliable for this target at high confidence."
4. THE MCP_Server SHALL include the calibration summary in the metadata returned alongside predictions, so the agent can present caveats to the user.

### Requirement 11: Infrastructure and Deployment Constraints

**User Story:** As the operator, I want Phase 2 to run on the same EC2 instance with the same Postgres database, so that there is no additional infrastructure cost or complexity.

#### Acceptance Criteria

1. THE MCP_Server SHALL run as a local process on the same EC2 instance as OpenCLAW (no network hops for MCP communication).
2. THE Prediction_Logger SHALL use the same Postgres instance and connection configuration as existing Sandy modules.
3. THE Live_State_Client SHALL make unauthenticated HTTP requests to the MLB_Stats_API (no API keys or tokens required).
4. THE MCP_Server SHALL not require any paid LLM API tokens (all prediction math is local LightGBM inference).
5. WHEN the MCP_Server starts, THE MCP_Server SHALL validate that required model artifacts exist and the database is reachable, logging errors if not.

### Requirement 12: Prediction with Live Context

**User Story:** As a user asking "will the Mariners get on base next inning?" during a live game, I want Sandy to automatically use the current game state for its prediction rather than requiring me to specify all parameters manually.

#### Acceptance Criteria

1. WHEN a prediction is requested via MCP with only a team_code and the team has an active game, THE MCP_Server SHALL automatically fetch live game state and use it to determine the current inning, opposing pitcher, and lineup context.
2. WHEN live game state is used for a prediction, THE MCP_Server SHALL include the game state in the response so the calling agent can present it to the user for verification (e.g., "Based on: Top of 5th, Buehler pitching, 67 pitches, due up: Rodríguez, Haniger, Suárez").
3. WHEN the user has not specified an inning, THE MCP_Server SHALL predict for the next half-inning (the team's next at-bat opportunity).
4. IF live game state cannot be fetched but pre-game data is available, THEN THE MCP_Server SHALL fall back to pre-game prediction using schedule data and indicate that live state was unavailable.

### Requirement 13: Total Game Runs Prediction

**User Story:** As a user, I want Sandy to predict the total combined runs for a game (both teams), so that I can assess whether a game will be high-scoring or low-scoring.

#### Acceptance Criteria

1. THE Predictor SHALL provide a `predict_total_runs` function that returns the expected total runs (home + away combined) for a given matchup.
2. THE Predictor SHALL also return per-team expected runs alongside the total (e.g., SEA: 4.2, LAD: 3.8, Total: 8.0).
3. THE MCP_Server SHALL expose `predict_total_runs` as a tool accepting team_code and opponent_code parameters.
4. THE CLI SHALL display total runs alongside per-team runs in `sandy predict --target runs` and `sandy predict-all` output.

### Requirement 14: Over/Under Threshold Probabilities

**User Story:** As a user, I want Sandy to tell me the probability of the total runs being over various thresholds (e.g., over 7.5, over 8.5), so that I can evaluate different scenarios.

#### Acceptance Criteria

1. WHEN a total runs prediction is made, THE Predictor SHALL also compute P(total > threshold) for thresholds: 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5.
2. THE over/under probabilities SHALL be derived from the model's predicted distribution (using the regression model's output + historical variance to estimate the probability of exceeding each threshold).
3. THE MCP_Server SHALL include over/under probabilities in the `predict_total_runs` tool response.
4. THE CLI SHALL display over/under probabilities as a table when `--target runs` is used.
5. WHEN displayed, THE over/under table SHALL show each threshold and its corresponding probability, sorted by threshold ascending.

### Requirement 15: Reached-Base Model Improvement

**User Story:** As a user, I want the reached_base model to be better at detecting innings where a team will NOT reach base, so that predictions below the base rate are more reliable and actionable.

#### Acceptance Criteria

1. THE Feature_Builder SHALL add the following shutdown detection features to the inning-level feature set (bumping FEATURE_SCHEMA_VERSION):
   - `pitcher_zero_baserunner_innings`: count of consecutive innings the current pitcher has allowed zero baserunners in this game (0 for inning 1 or prediction without live state)
   - `is_bottom_of_order`: 1 if batting order spots 7, 8, or 9 are all among the three due up, 0 otherwise
   - `pitcher_game_k_rate`: pitcher's strikeout rate in this game so far (strikeouts / batters faced)
   - `team_season_k_rate`: batting team's season strikeout rate (strikeouts / plate appearances)
   - `is_fresh_reliever`: 1 if current pitcher has thrown fewer than 20 pitches and is not the game's starting pitcher, 0 otherwise
2. WHEN live game state is not available (pre-game or historical training), THE Feature_Builder SHALL compute these features from the play-by-play data using the same cutoff_ts leakage prevention as existing features.
3. AFTER adding shutdown features, THE Trainer SHALL retrain the reached_base model and log the new validation ROC AUC.
4. THE new reached_base model SHALL achieve a validation ROC AUC of at least 0.55 (improvement over the current 0.54).
5. THE Confidence_Assessor SHALL reference active shutdown features when explaining HIGH confidence below-base-rate predictions.

### Requirement 16: Automated Daily Data Refresh

**User Story:** As a user asking about today's games, I want the database to be automatically updated with recent game results, so that predictions use the latest team form and pitcher stats without manual intervention.

#### Acceptance Criteria

1. THE system SHALL provide a scheduled task (cron job or systemd timer) that runs `sandy ingest incremental` daily at a configured time (default: 06:00 local time).
2. AFTER incremental ingestion completes, THE scheduled task SHALL run `sandy labels build` and `sandy features build` to update derived tables with the new data.
3. WHEN the scheduled task completes successfully, THE system SHALL log a summary line with games added, labels generated, and features built.
4. IF the scheduled task fails (DB unreachable, API error), THE system SHALL log the error and retry once after 30 minutes.
5. THE scheduled task SHALL NOT retrain models automatically — retraining is a manual operator decision (models are retrained weekly or when the operator chooses).
6. THE operator SHALL be able to trigger an immediate refresh by running `sandy refresh` from the CLI (equivalent to: ingest incremental + labels build + features build for all targets).
7. THE scheduled task SHALL complete within 10 minutes under normal conditions (fewer than 30 new games per day during the season).
