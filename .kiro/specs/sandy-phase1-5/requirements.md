# Requirements Document

## Introduction

Sandy Phase 1.5 extends the existing MLB prediction system with additional prediction targets (game winner, total runs), a today's-schedule lookup, batch prediction for all of today's games, and automatic starter resolution from the MLB API. All new functionality reuses the existing 3-season backfill data in Postgres and runs on the same EC2 instance. The architecture exposes all prediction logic as an importable Python API so Phase 2+ agents can call functions directly.

## Glossary

- **Sandy**: The MLB prediction system (Python package `sandy`)
- **Predictor**: The module responsible for loading a model artifact and producing predictions from feature vectors
- **Label_Generator**: A pure function that reads raw game data from Postgres and produces training labels for a specific prediction target
- **Model_Artifact**: A serialized LightGBM model file on disk, identified by target name and feature schema version
- **Schedule_Client**: The component that fetches today's MLB schedule and probable pitchers from the MLB Stats API
- **Game_Feature_Builder**: The component that constructs game-level feature vectors (as opposed to inning-level) for game_winner and runs predictions
- **CLI**: The Click-based command-line interface (`sandy` command group)
- **Prediction_Target**: One of: `reached_base`, `game_winner`, `runs` — selects which model and predict function to use
- **Probable_Pitcher**: The starting pitcher announced by MLB for an upcoming game, available via the `/v1/schedule` endpoint

## Requirements

### Requirement 1: Game Winner Label Generation

**User Story:** As a developer, I want to generate game-winner labels from existing game data, so that I can train a binary classifier for P(home team wins).

#### Acceptance Criteria

1. WHEN a completed game exists in `raw.games` with status 'Final', THE Label_Generator SHALL produce one label row with `home_team_wins = (home_score > away_score)` for that game_pk
2. WHEN a game has status other than 'Final', THE Label_Generator SHALL produce no label row for that game_pk
3. THE Label_Generator SHALL be a pure function that takes a DB connection and game_pk and returns a label dataclass without side effects
4. THE Label_Generator SHALL exclude postseason games (game_type != 'R') from label generation
5. WHEN home_score equals away_score (tie, suspended game), THE Label_Generator SHALL produce no label row for that game_pk

### Requirement 2: Runs Label Generation

**User Story:** As a developer, I want to generate per-team runs labels from existing game data, so that I can train a regression model predicting expected runs.

#### Acceptance Criteria

1. WHEN a completed game exists in `raw.games` with status 'Final', THE Label_Generator SHALL produce two label rows: one for the home team with `runs = home_score` and one for the away team with `runs = away_score`
2. WHEN a game has status other than 'Final', THE Label_Generator SHALL produce no label rows for that game_pk
3. THE Label_Generator SHALL be a pure function that takes a DB connection and game_pk and returns a list of runs label dataclasses without side effects
4. THE Label_Generator SHALL exclude postseason games (game_type != 'R') from label generation

### Requirement 3: Game-Level Feature Builder

**User Story:** As a developer, I want to build game-level feature vectors from existing data, so that game_winner and runs models have appropriate inputs.

#### Acceptance Criteria

1. THE Game_Feature_Builder SHALL produce one feature vector per (game_pk, team_code) containing: home starter ERA, home starter WHIP, away starter ERA, away starter WHIP, home team trailing-15 RPG, away team trailing-15 RPG, home team season OBP, away team season OBP, ballpark_id, and is_home indicator
2. THE Game_Feature_Builder SHALL compute pitcher stats using only data from games before the target game_date (no future leakage)
3. THE Game_Feature_Builder SHALL compute team trailing-15 stats using only the 15 most recent completed games before the target game_date
4. THE Game_Feature_Builder SHALL be a pure function that takes a DB connection, game_pk, and team_code and returns a feature vector dataclass
5. WHEN a starter has fewer than 3 appearances in the dataset, THE Game_Feature_Builder SHALL fall back to the league-average ERA and WHIP for that season
6. THE Game_Feature_Builder SHALL define a GAME_FEATURE_NAMES constant listing all game-level features in a fixed order

### Requirement 4: Game Winner Model Training

**User Story:** As a developer, I want to train a LightGBM binary classifier for game winner prediction, so that Sandy can predict P(home team wins).

#### Acceptance Criteria

1. THE Trainer SHALL fit a LightGBM model with `objective: binary` on the game-level features joined with game_winner labels
2. THE Trainer SHALL use chronological splitting (no same-season leakage between train and validation sets)
3. THE Trainer SHALL log validation ROC AUC, log loss, and Brier score as a single structured JSON line
4. IF validation ROC AUC is below 0.52, THEN THE Trainer SHALL raise a TrainingQualityError
5. THE Trainer SHALL serialize the model artifact to a path determined by target name: `{model_dir}/game_winner.pkl`
6. THE Trainer SHALL record the game feature schema version in the artifact metadata

### Requirement 5: Runs Model Training

**User Story:** As a developer, I want to train a LightGBM regression model for runs prediction, so that Sandy can predict expected runs per team per game.

#### Acceptance Criteria

1. THE Trainer SHALL fit a LightGBM model with `objective: regression` on the game-level features joined with runs labels
2. THE Trainer SHALL use chronological splitting consistent with the game_winner model
3. THE Trainer SHALL log validation MAE and RMSE as a single structured JSON line
4. THE Trainer SHALL serialize the model artifact to `{model_dir}/runs.pkl`
5. THE Trainer SHALL record the game feature schema version in the artifact metadata

### Requirement 6: Multi-Target Prediction API

**User Story:** As a developer, I want a unified prediction API that dispatches to the correct model based on target name, so that Phase 2+ agents can call `predict(target="game_winner", ...)` directly.

#### Acceptance Criteria

1. THE Predictor SHALL accept a `target` parameter with values `reached_base`, `game_winner`, or `runs`
2. WHEN target is `game_winner`, THE Predictor SHALL load the game_winner model artifact and return a probability between 0.0 and 1.0 representing P(home team wins)
3. WHEN target is `runs`, THE Predictor SHALL load the runs model artifact and return a float representing expected runs for the specified team
4. WHEN target is `reached_base`, THE Predictor SHALL behave identically to the existing Phase 1 predictor (backward compatible)
5. IF the model artifact for the requested target does not exist, THEN THE Predictor SHALL raise a MissingArtifactError with the target name in the message
6. THE Predictor SHALL expose a `predict_game()` function importable from `sandy.predict` that takes team, opponent, starter names, and target — suitable for direct invocation by Phase 2+ agents
7. WHEN target is `game_winner` or `runs`, THE Predictor SHALL not require an inning parameter

### Requirement 7: Schedule Lookup from MLB API

**User Story:** As a user, I want to see today's MLB games and probable pitchers, so that I know what matchups are available for prediction.

#### Acceptance Criteria

1. THE Schedule_Client SHALL fetch today's games from the MLB Stats API endpoint `/v1/schedule` with `sportId=1` and `hydrate=probablePitcher`
2. THE Schedule_Client SHALL parse the response into a list of game records containing: game_pk, home team code, away team code, home probable pitcher name, away probable pitcher name, game time, and game status
3. WHEN a probable pitcher is not yet announced for a team, THE Schedule_Client SHALL set that pitcher field to None
4. THE Schedule_Client SHALL reuse the existing MlbStatsClient (rate limiter, retry logic) from `sandy.ingest.client`
5. IF the MLB API returns an error or is unreachable, THEN THE Schedule_Client SHALL raise a descriptive error without crashing the process
6. THE Schedule_Client SHALL be importable as a Python function from `sandy.schedule` for direct use by Phase 2+ agents

### Requirement 8: Auto-Resolve Starters

**User Story:** As a user, I want Sandy to automatically look up the probable pitcher when I don't specify one, so that I can predict games without manually finding starter names.

#### Acceptance Criteria

1. WHEN the user does not provide a `--starter` option and the target is `game_winner` or `runs`, THE Predictor SHALL look up today's schedule and resolve the probable pitcher for the specified matchup
2. WHEN the user does not provide a `--starter` option and the target is `reached_base`, THE CLI SHALL require the `--starter` option and exit with an error if missing
3. IF no probable pitcher is announced for the specified matchup today, THEN THE Predictor SHALL return an error message indicating no probable pitcher is available
4. IF the specified team matchup does not appear in today's schedule, THEN THE Predictor SHALL return an error message indicating the matchup was not found in today's games
5. THE auto-resolve logic SHALL match teams by team code (case-insensitive comparison)

### Requirement 9: CLI — Today's Schedule Command

**User Story:** As a user, I want a `sandy today` command that shows today's MLB games in a readable table, so that I can quickly see what's playing.

#### Acceptance Criteria

1. WHEN the user runs `sandy today`, THE CLI SHALL display a table with columns: game time, away team, home team, away probable pitcher, home probable pitcher, and game status
2. WHEN a probable pitcher is not announced, THE CLI SHALL display "TBD" in that column
3. THE CLI SHALL format game times in the user's local timezone
4. IF no games are scheduled today, THEN THE CLI SHALL display a message "No MLB games scheduled for today"
5. THE CLI SHALL exit with code 0 on success and code 1 on API errors

### Requirement 10: CLI — Predict Command Extension

**User Story:** As a user, I want to run `sandy predict --target game_winner --team SEA --opp LAD` to get a win probability, so that I can predict any supported target from the command line.

#### Acceptance Criteria

1. THE CLI SHALL accept a `--target` option with choices `reached_base`, `game_winner`, `runs` (default: `reached_base`)
2. WHEN target is `game_winner`, THE CLI SHALL output JSON with a `win_probability` field (float 0.0–1.0) and `top_features` list
3. WHEN target is `runs`, THE CLI SHALL output JSON with an `expected_runs` field (float) and `top_features` list
4. WHEN target is `game_winner` or `runs` and `--starter` is not provided, THE CLI SHALL auto-resolve the starter from today's schedule
5. WHEN target is `game_winner` or `runs`, THE CLI SHALL not require the `--inning` option
6. WHEN target is `reached_base`, THE CLI SHALL require both `--inning` and `--starter` (preserving Phase 1 behavior)
7. THE CLI SHALL exit with code 2 for invalid input, code 3 for missing model artifact, and code 1 for other errors

### Requirement 11: CLI — Batch Predict All Today's Games

**User Story:** As a user, I want to run `sandy predict-all` to get game_winner and runs predictions for every game today, so that I get a full day's predictions in one command.

#### Acceptance Criteria

1. WHEN the user runs `sandy predict-all`, THE CLI SHALL fetch today's schedule, resolve probable pitchers, and run both game_winner and runs predictions for every game that has both probable pitchers announced
2. THE CLI SHALL output a summary table with columns: away team, home team, P(home wins), away expected runs, home expected runs, away starter, home starter
3. WHEN a game is missing one or both probable pitchers, THE CLI SHALL skip that game and note it as "skipped (TBD pitcher)" in the output
4. THE CLI SHALL output valid JSON when `--json` flag is provided (array of prediction objects)
5. IF no games have both probable pitchers announced, THEN THE CLI SHALL display a message indicating no games are predictable
6. THE CLI SHALL exit with code 0 on success, code 3 if model artifacts are missing, and code 1 on API errors

### Requirement 12: Label and Feature Persistence

**User Story:** As a developer, I want game-level labels and features stored in derived tables, so that training can join them efficiently.

#### Acceptance Criteria

1. THE Runner SHALL persist game_winner labels to a `derived.game_winner_labels` table with columns: game_pk (PK), home_team_wins (boolean)
2. THE Runner SHALL persist runs labels to a `derived.runs_labels` table with columns: game_pk (PK), team_code (PK), runs (integer)
3. THE Runner SHALL persist game-level features to a `derived.game_features` table with columns: game_pk (PK), team_code (PK), feature_schema_version, and one column per game-level feature
4. THE Runner SHALL use UPSERT (INSERT ON CONFLICT UPDATE) to ensure idempotent writes
5. THE Runner SHALL process games in chronological order to support incremental builds

### Requirement 13: Model Path Configuration

**User Story:** As a developer, I want model artifacts stored by target name, so that multiple models coexist without conflict.

#### Acceptance Criteria

1. THE Config SHALL resolve model paths as `{model_dir}/{target_name}.pkl` where model_dir defaults to `./models/`
2. THE Config SHALL support an `MLB_MODEL_DIR` environment variable to override the model directory
3. WHEN loading a model for a specific target, THE Predictor SHALL construct the path from model_dir and target name
4. THE existing `MLB_MODEL_PATH` environment variable SHALL continue to work for the `reached_base` target (backward compatibility)

### Requirement 14: Serializer Round-Trip for New Models

**User Story:** As a developer, I want game_winner and runs model artifacts to survive serialization round-trips, so that saved models produce identical predictions after reload.

#### Acceptance Criteria

1. FOR ALL valid game_winner ModelArtifact objects, saving then loading SHALL produce a ModelArtifact that generates identical predictions on the same input (round-trip property)
2. FOR ALL valid runs ModelArtifact objects, saving then loading SHALL produce a ModelArtifact that generates identical predictions on the same input (round-trip property)
3. THE serializer SHALL include the target name in the artifact metadata so loaders can verify they loaded the correct model type
