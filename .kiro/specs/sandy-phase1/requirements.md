# Requirements Document — Sandy, Phase 1

## Introduction

Sandy is an MLB game-day assistant whose long-term vision (see `.kiro/specs/sandy-roadmap.md`) is a Telegram-accessible, multi-agent system that predicts and discusses live-game props. This document scopes **Phase 1 only**: the offline foundation — historical data ingestion, relational storage, a trained gradient-boosted classifier, and a local Python CLI that produces per-inning "reaches base" probabilities for a specified team. No live-game features, no agent orchestration, no chat surface are in scope for this phase. The system runs on a single AWS EC2 instance (Amazon Linux 2023, t3.small) with PostgreSQL in docker-compose, and is developed from the user's Mac against that EC2. Success for Phase 1 is an operator being able to run `sandy predict --team SEA --opp LAD --inning 3 --starter "Walker Buehler"` and receive a well-formed probability with top contributing features, backed by a model trained on three seasons of historical data.

## Glossary

- **MLB_Stats_API**: The public MLB Stats API at `statsapi.mlb.com`, requiring no authentication.
- **Ingestion_Service**: Python component that fetches data from MLB_Stats_API and writes it to the Database.
- **Database**: PostgreSQL instance running in docker-compose on the EC2 host, storing raw and derived data.
- **Raw_Schema**: Database tables holding canonical, source-of-truth records from MLB_Stats_API (games, plays, players, teams).
- **Derived_Schema**: Database tables holding computed rows: per-inning labels and model features.
- **Label_Generator**: Component that reads play-by-play rows from Raw_Schema and produces binary `reached_base` labels in Derived_Schema.
- **Reaches_Base_Event**: A plate appearance with outcome in the set {single, double, triple, home_run, walk, hit_by_pitch, reached_on_error}.
- **Feature_Builder**: Component that assembles the feature vector for a (game, team, inning) training row or a prediction request.
- **Trainer**: Component that fits a LightGBM binary classifier over labeled feature rows and writes a Model_Artifact.
- **Model_Artifact**: A pickle file on local disk containing the fitted LightGBM model plus metadata (feature list, training window, schema version, created_at timestamp).
- **Predictor**: Component that loads a Model_Artifact and returns a probability plus feature contributions for a prediction request.
- **CLI**: The `sandy` command with subcommands `predict`, `ingest`, and `train`, implemented in Python.
- **Team_Code**: A three-letter MLB team abbreviation used by MLB_Stats_API (e.g., `SEA`, `LAD`).
- **Season**: A single MLB regular season identified by calendar year (e.g., 2023). Postseason games are excluded from Phase 1.
- **Target_Seasons**: The three most recent complete regular seasons as of system initialization.
- **Operator**: The single human user running the system from a shell on the EC2 host.

## Requirements

### Requirement 1: Historical Data Backfill

**User Story:** As the Operator, I want to backfill three complete MLB regular seasons from MLB_Stats_API into the Database, so that I have enough historical data to train a model.

#### Acceptance Criteria

1. WHEN the Operator runs the backfill command, THE Ingestion_Service SHALL fetch every regular-season game from the three most recent complete seasons and persist game, play-by-play, starting pitcher, and lineup records into the Raw_Schema.
2. WHEN the backfill command is run a second time against the same Database, THE Ingestion_Service SHALL produce an end state byte-equivalent to the first run for all Raw_Schema tables (idempotent backfill).
3. WHEN MLB_Stats_API returns an HTTP 429 or 5xx response, THE Ingestion_Service SHALL retry the failing request with exponential backoff up to 5 attempts with a base delay of 1 second.
4. IF a request to MLB_Stats_API fails after 5 retry attempts, THEN THE Ingestion_Service SHALL record the failed `game_pk` in an `ingest_failures` table with the error reason and continue processing remaining games.
5. WHILE the backfill is running, THE Ingestion_Service SHALL emit a progress log line at minimum every 50 games containing games-processed, games-remaining, and elapsed seconds.
6. IF the backfill process is interrupted and rerun, THEN THE Ingestion_Service SHALL skip games whose `game_pk` is already present in the Raw_Schema and resume with the next unprocessed game.
7. THE Ingestion_Service SHALL issue no more than 10 requests per second to MLB_Stats_API.

### Requirement 2: Incremental Daily Ingestion

**User Story:** As the Operator, I want a daily incremental ingestion command, so that the Database stays current without re-downloading history.

#### Acceptance Criteria

1. WHEN the Operator runs the incremental ingestion command, THE Ingestion_Service SHALL fetch only games with a `gameDate` strictly greater than the maximum `gameDate` already present in the Raw_Schema for completed games.
2. WHEN a game is already present in the Raw_Schema with status `Final`, THE Ingestion_Service SHALL skip that game.
3. WHEN a game is present in the Raw_Schema with a non-final status and MLB_Stats_API now reports the status as `Final`, THE Ingestion_Service SHALL replace the existing rows for that `game_pk` with the final data.
4. IF the incremental ingestion command is run twice in immediate succession, THEN THE Ingestion_Service SHALL produce the same end state in the Raw_Schema (idempotent incremental ingestion).
5. WHEN incremental ingestion completes, THE Ingestion_Service SHALL write a summary line containing games-added, games-updated, and games-skipped counts.

### Requirement 3: Data Storage Schema

**User Story:** As a developer, I want a well-defined relational schema, so that ingestion, labeling, and training can share a single source of truth.

#### Acceptance Criteria

1. THE Database SHALL provide tables `games`, `plays`, `teams`, `players`, and `pitcher_game_stats` in the Raw_Schema with a primary key on each table.
2. THE Database SHALL provide tables `inning_labels` and `inning_features` in the Derived_Schema with a composite primary key of `(game_pk, team_code, inning_number)`.
3. THE Database SHALL declare a foreign key from `plays.game_pk` to `games.game_pk`.
4. WHEN a row is written to `inning_labels` or `inning_features` for an existing `(game_pk, team_code, inning_number)`, THE Database SHALL overwrite the existing row rather than inserting a duplicate.
5. THE Database SHALL run inside a docker-compose service named `postgres` with a named volume for data persistence.
6. THE Ingestion_Service SHALL connect to the Database using credentials supplied via environment variables.

### Requirement 4: Label Generation

**User Story:** As a developer, I want correct binary labels for every (game, team, inning) pair, so that the Trainer can learn from true outcomes.

#### Acceptance Criteria

1. WHEN the Operator runs the label-generation command, THE Label_Generator SHALL emit exactly one row in `inning_labels` for every `(game_pk, team_code, inning_number)` tuple present in the Raw_Schema where that team batted in that inning.
2. WHEN at least one Reaches_Base_Event exists in the play-by-play for a given `(game_pk, team_code, inning_number)`, THE Label_Generator SHALL set `reached_base = true` for that row.
3. IF no Reaches_Base_Event exists in the play-by-play for a given `(game_pk, team_code, inning_number)`, THEN THE Label_Generator SHALL set `reached_base = false` for that row.
4. WHEN the label-generation command is run repeatedly over the same Raw_Schema state, THE Label_Generator SHALL produce byte-equivalent `inning_labels` rows on each run (idempotent labeling).
5. WHEN a new Reaches_Base_Event is added to the play-by-play for an inning that previously had `reached_base = false`, and label generation is rerun, THE Label_Generator SHALL set `reached_base = true` for that row (label monotonicity under added baserunner events).
6. THE Label_Generator SHALL exclude innings from games whose status is not `Final`.

### Requirement 5: Feature Engineering

**User Story:** As a developer, I want a deterministic feature-building step that uses only information available before the target inning, so that the model is trained without leakage.

#### Acceptance Criteria

1. WHEN the Feature_Builder constructs features for a `(game_pk, team_code, inning_number)` row, THE Feature_Builder SHALL use only data from events with a timestamp strictly earlier than the first pitch of the target inning.
2. THE Feature_Builder SHALL include the following features in every feature vector: opposing starter season ERA, opposing starter season WHIP, opposing starter season K/9, opposing starter pitches thrown in the current game prior to the target inning, lineup spots due up for the batting team in the target inning (three integer slots in 1..9), home-or-away indicator for the batting team, ballpark identifier, inning number, batting team runs-per-game over the trailing 15 games, and batting team on-base-percentage over the trailing 15 games.
3. WHEN the Feature_Builder is invoked twice with identical Raw_Schema inputs for the same `(game_pk, team_code, inning_number)`, THE Feature_Builder SHALL produce identical feature vectors (determinism).
4. IF any feature in the feature vector cannot be computed for a training row, THEN THE Feature_Builder SHALL omit that row from `inning_features` and log the omitted `(game_pk, team_code, inning_number)` with the missing feature name.
5. WHEN the Feature_Builder writes rows to `inning_features`, THE Feature_Builder SHALL include a `feature_schema_version` integer column whose value matches the current code-declared feature schema version.

### Requirement 6: Model Training

**User Story:** As the Operator, I want to train a LightGBM binary classifier on the labeled feature rows, so that I have a usable predictive model.

#### Acceptance Criteria

1. WHEN the Operator runs the train command, THE Trainer SHALL fit a LightGBM binary classifier using all rows in `inning_features` joined to `inning_labels` whose `game_date` falls within the Target_Seasons.
2. THE Trainer SHALL split training data chronologically such that the most recent 15 percent of games by `game_date` form the validation set and the remainder forms the training set.
3. WHEN training completes, THE Trainer SHALL log the following metrics computed on the validation set: ROC AUC, log loss, Brier score, and the count of positive and negative examples.
4. IF the validation ROC AUC is strictly less than 0.52, THEN THE Trainer SHALL exit with a non-zero status and write a warning message naming the observed AUC.
5. WHEN training completes successfully, THE Trainer SHALL persist a Model_Artifact to a configured output path.
6. WHERE the Operator supplies a `--seed` option, THE Trainer SHALL use that integer to seed the LightGBM random state and any data-shuffling step.

### Requirement 7: Model Artifact Persistence

**User Story:** As a developer, I want the saved model to round-trip cleanly through the filesystem, so that the Predictor can reliably load it.

#### Acceptance Criteria

1. THE Trainer SHALL serialize the Model_Artifact as a single pickle file containing a dictionary with keys `model`, `feature_names`, `feature_schema_version`, `training_window_start`, `training_window_end`, and `created_at`.
2. WHEN the Predictor loads a Model_Artifact written by the Trainer, THE Predictor SHALL expose a model object whose predicted probabilities on a fixed input match those from the in-memory model at serialization time to within 1e-9 absolute tolerance (serializer round-trip).
3. IF a loaded Model_Artifact has a `feature_schema_version` different from the currently declared feature schema version, THEN THE Predictor SHALL exit with a non-zero status and emit an error message naming both versions.

### Requirement 8: Prediction CLI

**User Story:** As the Operator, I want a `predict` CLI command, so that I can get a probability for any team, opponent, inning, and starting pitcher combination from my shell.

#### Acceptance Criteria

1. WHEN the Operator runs `sandy predict --team <code> --opp <code> --inning <n> --starter "<name>"`, THE CLI SHALL load the most recent Model_Artifact and emit on stdout a JSON object with keys `probability` and `top_features`.
2. THE CLI SHALL ensure the emitted `probability` value is a floating-point number in the inclusive interval [0.0, 1.0].
3. THE CLI SHALL ensure `top_features` is a list of up to five objects, each with keys `name` and `contribution`, ordered by descending absolute `contribution`.
4. IF either `--team` or `--opp` is not a recognized Team_Code in the Database, THEN THE CLI SHALL exit with status code 2 and write an error to stderr naming the unrecognized code.
5. IF `--inning` is not an integer in the inclusive range 1 through 9, THEN THE CLI SHALL exit with status code 2 and write an error to stderr naming the invalid value.
6. IF `--starter` does not match exactly one player in the Database via case-insensitive name lookup, THEN THE CLI SHALL exit with status code 2 and write an error to stderr listing up to five closest matches.
7. WHERE the Operator supplies `--as-of <YYYY-MM-DD>`, THE CLI SHALL compute features using only data dated strictly before the supplied date.
8. WHEN no Model_Artifact exists at the configured path, THE CLI SHALL exit with status code 3 and write an error to stderr instructing the Operator to run the train command.

### Requirement 9: Configuration Management

**User Story:** As the Operator, I want runtime configuration to come from environment variables and a single config file, so that I can move between dev and EC2 without code changes.

#### Acceptance Criteria

1. THE CLI SHALL read Database connection parameters from the environment variables `MLB_DB_HOST`, `MLB_DB_PORT`, `MLB_DB_NAME`, `MLB_DB_USER`, and `MLB_DB_PASSWORD`.
2. IF any required Database environment variable is unset at CLI startup, THEN THE CLI SHALL exit with status code 4 and write an error to stderr naming the missing variable.
3. THE CLI SHALL read the Model_Artifact path from the environment variable `MLB_MODEL_PATH` and SHALL fall back to `./models/latest.pkl` when that variable is unset.
4. THE CLI SHALL expose a `--config <path>` flag that loads a TOML file whose values override defaults but are overridden by environment variables and explicit command-line flags.

### Requirement 10: Logging and Observability

**User Story:** As the Operator, I want consistent logs from every long-running operation, so that I can see what the system did after it finishes.

#### Acceptance Criteria

1. THE Ingestion_Service, Label_Generator, Feature_Builder, and Trainer SHALL write structured log lines to stdout in JSON format with fields `timestamp`, `level`, `component`, and `message`.
2. WHEN an ingestion, labeling, feature-build, or training run completes, THE corresponding component SHALL write a final log line at level `INFO` containing the run duration in seconds and the count of rows read and written.
3. WHERE the environment variable `MLB_LOG_LEVEL` is set to a recognized level (`DEBUG`, `INFO`, `WARN`, `ERROR`), THE CLI SHALL emit log lines at or above that level.

### Requirement 11: Testing and Validation

**User Story:** As a developer, I want an automated test suite covering critical correctness properties, so that I can refactor without fear.

#### Acceptance Criteria

1. THE project SHALL include a property-based test asserting that the `probability` returned by the Predictor falls in the closed interval [0.0, 1.0] for every generated input.
2. THE project SHALL include a property-based test asserting that running the Ingestion_Service twice over the same fixed set of mocked MLB_Stats_API responses produces byte-equivalent Raw_Schema rows on both runs (idempotent ingestion).
3. THE project SHALL include a property-based test asserting that for any synthetic play-by-play where at least one Reaches_Base_Event exists in an inning, the Label_Generator produces `reached_base = true` for that inning (label monotonicity).
4. THE project SHALL include a property-based test asserting that saving a trained LightGBM model via the Trainer's serializer and then loading it via the Predictor's deserializer produces predictions matching the in-memory model within 1e-9 absolute tolerance on a fixed input (serializer round-trip).
5. THE project SHALL include an integration test that executes the CLI end-to-end against a seeded test Database and verifies the `predict` command emits valid JSON containing a numeric `probability` in [0.0, 1.0].
6. WHEN the test suite is run via `uv run pytest`, THE project SHALL exit with status code 0 if and only if every test passes.

### Requirement 12: Environment and Packaging

**User Story:** As the Operator, I want dependencies managed by `uv` and the runtime defined by docker-compose, so that setup on a fresh EC2 is reproducible.

#### Acceptance Criteria

1. THE project SHALL declare all Python dependencies in a single `pyproject.toml` file managed by `uv`.
2. THE project SHALL pin the Python version in `pyproject.toml` to a single minor version in the 3.11.x or 3.12.x series.
3. THE project SHALL provide a `docker-compose.yml` whose `postgres` service starts a PostgreSQL instance bound to a named volume on the EC2 host.
4. WHEN the Operator runs `uv sync` followed by `docker compose up -d` in a fresh clone on the EC2, THE project SHALL reach a state where the `sandy predict`, `sandy ingest`, and `sandy train` commands are invokable without additional manual setup.
