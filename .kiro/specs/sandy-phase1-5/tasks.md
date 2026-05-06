# Implementation Plan: Sandy — Phase 1.5 (Additional Prediction Targets)

## Overview

This plan extends Sandy with game-winner and runs prediction targets, a today's-schedule lookup, batch prediction, and automatic starter resolution. The build order is strictly bottom-up — DB schema → dataclasses → label generators → game-level features → multi-target training → schedule client → multi-target predictor → CLI extensions — so every stage is independently runnable and testable before the one that depends on it.

All code targets Python 3.11, managed by `uv`. Reuses the existing EC2 + Postgres infrastructure, `MlbStatsClient`, `db.py`, and `config.py`. Property-based tests are mandatory (not optional) and validate the 12 correctness properties defined in the design.

## Tasks

- [ ] 1. DB schema changes — new derived tables
  - [x] 1.1 Add DDL for `derived.game_winner_labels` table with columns: `game_pk` (PK, FK → raw.games), `home_team_wins` (BOOLEAN NOT NULL), `labeled_at` (TIMESTAMPTZ DEFAULT now())
    - Add to the `bootstrap_schema()` DDL in `sandy/db.py`
    - Use `CREATE TABLE IF NOT EXISTS` for idempotent bootstrap
    - _Requirements: 12.1_
  - [ ] 1.2 Add DDL for `derived.runs_labels` table with columns: `game_pk` (NOT NULL, FK → raw.games), `team_code` (CHAR(3) NOT NULL), `runs` (INTEGER NOT NULL), `labeled_at` (TIMESTAMPTZ DEFAULT now()), PRIMARY KEY (game_pk, team_code)
    - _Requirements: 12.2_
  - [ ] 1.3 Add DDL for `derived.game_features` table with columns: `game_pk` (NOT NULL, FK → raw.games), `team_code` (CHAR(3) NOT NULL), `feature_schema_version` (INTEGER NOT NULL), all 10 game-level feature columns (REAL/INTEGER/BOOLEAN), `computed_at` (TIMESTAMPTZ DEFAULT now()), PRIMARY KEY (game_pk, team_code)
    - _Requirements: 12.3_

- [ ] 2. New dataclasses and schema constants
  - [ ] 2.1 Add `GameWinnerLabel`, `RunsLabel`, `GameFeatureVector`, and `ScheduledGame` frozen dataclasses to `sandy/schemas.py`
    - `GameWinnerLabel(game_pk: int, home_team_wins: bool)`
    - `RunsLabel(game_pk: int, team_code: str, runs: int)`
    - `GameFeatureVector(game_pk: int | None, team_code: str, feature_schema_version: int, values: dict[str, float | int | bool])`
    - `ScheduledGame(game_pk: int, home_team_code: str, away_team_code: str, home_probable_pitcher: str | None, away_probable_pitcher: str | None, game_time_utc: datetime, status: str)`
    - _Requirements: 1.3, 2.3, 3.4, 7.2_
  - [ ] 2.2 Add `target_name: str` field to the existing `ModelArtifact` dataclass (default `"reached_base"` for backward compatibility)
    - _Requirements: 14.3_
  - [ ] 2.3 Add `GAME_FEATURE_SCHEMA_VERSION = 1` and `GAME_FEATURE_NAMES` (10 names) to `sandy/features/schema.py`
    - Names: `home_starter_era`, `home_starter_whip`, `away_starter_era`, `away_starter_whip`, `home_trailing15_rpg`, `away_trailing15_rpg`, `home_season_obp`, `away_season_obp`, `ballpark_id`, `is_home`
    - _Requirements: 3.1, 3.6_

- [ ] 3. Checkpoint — Schema and dataclasses
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Game winner label generator
  - [ ] 4.1 Implement `generate_game_winner_label()` (pure function) in `sandy/labels/game_winner_generator.py`
    - Takes a DB connection and `game_pk`, returns `GameWinnerLabel | None`
    - Returns `None` if status != 'Final', game_type != 'R', or home_score == away_score
    - Returns `GameWinnerLabel(game_pk, home_score > away_score)` otherwise
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ] 4.2 Write property-based test for game winner label correctness
    - **Property 1: Game winner label correctness**
    - Generate random (home_score, away_score, status, game_type) tuples via Hypothesis
    - Assert: Final + regular-season + home_score > away_score → `home_team_wins=True`
    - Assert: Final + regular-season + home_score < away_score → `home_team_wins=False`
    - Assert: non-Final, non-regular-season, or tied → returns None
    - **Validates: Requirements 1.1, 1.2, 1.4, 1.5**

- [ ] 5. Runs label generator
  - [ ] 5.1 Implement `generate_runs_labels()` (pure function) in `sandy/labels/runs_generator.py`
    - Takes a DB connection and `game_pk`, returns `list[RunsLabel]`
    - Returns `[]` if status != 'Final' or game_type != 'R'
    - Returns exactly 2 labels: `RunsLabel(game_pk, home_team_code, home_score)` and `RunsLabel(game_pk, away_team_code, away_score)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ] 5.2 Write property-based test for runs label correctness
    - **Property 2: Runs label correctness**
    - Generate random game rows with scores, status, game_type via Hypothesis
    - Assert: Final + regular-season → exactly 2 labels with correct runs values
    - Assert: non-Final or non-regular-season → empty list
    - **Validates: Requirements 2.1, 2.2, 2.4**

- [ ] 6. Extend labels runner for game-level labels
  - [ ] 6.1 Extend `sandy/labels/runner.py` with `run_game_winner_labels()` and `run_runs_labels()` functions
    - Iterate all Final regular-season games, call the respective generator, UPSERT into `derived.game_winner_labels` / `derived.runs_labels`
    - Use `ON CONFLICT DO UPDATE` for idempotent writes
    - Emit structured JSON log line with `duration_seconds`, `rows_written`
    - _Requirements: 12.1, 12.2, 12.4, 12.5_
  - [ ] 6.2 Write property-based test for idempotent label writes
    - **Property 11: Idempotent label and feature writes**
    - Generate random label data, write twice, assert table state is identical after both writes
    - Extend existing `tests/test_pbt_idempotent_ingest.py` or create new test file
    - **Validates: Requirements 12.4**

- [ ] 7. Game-level feature builder
  - [ ] 7.1 Implement `build_game_feature_vector()` (pure function) in `sandy/features/game_builder.py`
    - Takes: conn, game_pk (or None for prediction), team_code, opp_team_code, home_starter_id, away_starter_id, game_date, venue_id
    - Returns `GameFeatureVector` with all 10 features from `GAME_FEATURE_NAMES`
    - Computes pitcher stats using only games before `game_date` (no future leakage)
    - Computes trailing-15 stats using only the 15 most recent completed games before `game_date`
    - Falls back to league-average ERA/WHIP when starter has < 3 appearances
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [ ] 7.2 Write property-based test for game feature vector completeness
    - **Property 3: Game feature vector completeness**
    - Generate random game contexts with varying pitcher histories
    - Assert: output contains all 10 fields from GAME_FEATURE_NAMES with no None values
    - **Validates: Requirements 3.1, 3.5**
  - [ ] 7.3 Write property-based test for no future leakage in game features
    - **Property 4: No future leakage in game features**
    - Generate games with known date boundaries, seed DB with games on specific dates
    - Assert: all pitcher stats and team stats use only data from games with game_date < target game_date
    - **Validates: Requirements 3.2, 3.3**

- [ ] 8. Extend features runner for game-level features
  - [ ] 8.1 Implement `run_game_features()` in `sandy/features/runner.py`
    - Iterate all Final regular-season games, call `build_game_feature_vector()` for each (game_pk, team_code) pair
    - UPSERT into `derived.game_features` with `feature_schema_version` stamped
    - Process in chronological order for incremental builds
    - Emit structured JSON log line with `duration_seconds`, `rows_written`
    - _Requirements: 12.3, 12.4, 12.5_

- [ ] 9. Checkpoint — Labels and features pipelines
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Multi-target training
  - [x] 10.1 Extend `sandy/train/trainer.py` with `train_game_winner_model()` function
    - Load game_winner_labels ⨝ game_features join from Postgres
    - Fit LightGBM with `objective: binary`, chronological split, deterministic training
    - Log validation ROC AUC, log loss, Brier score as structured JSON
    - Raise `TrainingQualityError` if ROC AUC < 0.52
    - Return `ModelArtifact` with `target_name="game_winner"`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_
  - [x] 10.2 Extend `sandy/train/trainer.py` with `train_runs_model()` function
    - Load runs_labels ⨝ game_features join from Postgres
    - Fit LightGBM with `objective: regression`, chronological split, deterministic training
    - Log validation MAE and RMSE as structured JSON
    - Return `ModelArtifact` with `target_name="runs"`
    - _Requirements: 5.1, 5.2, 5.3, 5.5_
  - [x] 10.3 Extend `sandy/train/artifact.py` to include `target_name` in serialized metadata and resolve paths by target
    - `save_artifact()` stores `target_name` in the pickle dict
    - `load_artifact()` verifies `target_name` matches expected (raises `TargetMismatchError` if not)
    - Path resolution: `{model_dir}/{target_name}.pkl`
    - _Requirements: 4.5, 5.4, 13.1, 13.3, 14.3_
  - [ ] 10.4 Write property-based test for model artifact round-trip (game_winner and runs)
    - **Property 12: Model artifact round-trip**
    - Generate random model artifacts with target metadata, save and reload
    - Assert: reloaded artifact produces numerically identical predictions (within 1e-9)
    - Extend existing `tests/test_pbt_serializer_roundtrip.py`
    - **Validates: Requirements 14.1, 14.2**
  - [ ] 10.5 Write property-based test for model path construction
    - **Property 7: Model path construction**
    - Generate random target names from {"reached_base", "game_winner", "runs"} and random model_dir paths
    - Assert: resolved path equals `model_dir / f"{target_name}.pkl"`
    - **Validates: Requirements 4.5, 5.4, 13.1, 13.3**

- [ ] 11. Config extension for multi-target model paths
  - [x] 11.1 Extend `ModelConfig` in `sandy/config.py` to support `model_dir` (directory) with `artifact_path(target: str) -> Path` helper
    - `MLB_MODEL_DIR` env var overrides the model directory
    - `MLB_MODEL_PATH` continues to work for `reached_base` (backward compat)
    - Default model_dir: `./models/`
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [ ] 12. Schedule client
  - [ ] 12.1 Create `sandy/schedule/__init__.py` and `sandy/schedule/client.py`
    - Implement `get_todays_schedule(config: Config | None = None) -> list[ScheduledGame]`
    - Fetch from MLB Stats API `/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher`
    - Reuse existing `MlbStatsClient` for rate limiting and retries
    - Parse response into `ScheduledGame` dataclass list
    - Set pitcher fields to `None` when not announced
    - Raise descriptive error on API failure without crashing
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [ ] 12.2 Implement `resolve_starter_for_matchup(schedule, team, opp) -> tuple[str, str]`
    - Case-insensitive team code matching
    - Returns `(home_starter, away_starter)` for the given matchup
    - Raises `InvalidInputError` if matchup not found or pitcher TBD
    - _Requirements: 8.1, 8.3, 8.4, 8.5_
  - [ ] 12.3 Write property-based test for schedule response parsing
    - **Property 8: Schedule response parsing completeness**
    - Generate random valid MLB schedule API response JSON structures
    - Assert: each ScheduledGame has non-None game_pk, home_team_code, away_team_code, game_time_utc, status
    - Assert: pitcher fields are None when not present in response
    - **Validates: Requirements 7.2, 7.3**
  - [ ] 12.4 Write property-based test for case-insensitive team matching
    - **Property 9: Case-insensitive team matching in auto-resolve**
    - Generate random case variations of team codes
    - Assert: all case variations find the same matchup in the schedule
    - **Validates: Requirements 8.5**

- [ ] 13. Checkpoint — Training and schedule client
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Multi-target predictor
  - [x] 14.1 Implement `predict_game()` in `sandy/predict/predictor.py`
    - Accepts: team, opp, target ("game_winner" | "runs"), starter (optional), opp_starter (optional), as_of, config
    - Dispatches to correct model artifact and feature builder based on target
    - Auto-resolves starters from today's schedule when not provided (for game_winner/runs)
    - Returns `PredictionResult` with probability (game_winner) or expected_runs (runs)
    - Importable from `sandy.predict` for Phase 2+ agents
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 8.1, 8.2, 8.3, 8.4_
  - [ ] 14.2 Write property-based test for game winner probability range
    - **Property 5: Game winner probability range**
    - Generate random valid game feature vectors, predict with a trained game_winner model
    - Assert: probability ∈ [0.0, 1.0]
    - Extend existing `tests/test_pbt_probability_range.py`
    - **Validates: Requirements 6.2**
  - [ ] 14.3 Write property-based test for runs prediction non-negativity
    - **Property 6: Runs prediction non-negativity**
    - Generate random valid game feature vectors, predict with a trained runs model
    - Assert: expected_runs >= 0.0
    - **Validates: Requirements 6.3**

- [ ] 15. CLI extensions
  - [x] 15.1 Extend `sandy predict` command with `--target` option (choices: `reached_base`, `game_winner`, `runs`; default: `reached_base`)
    - When target is `game_winner`/`runs`: `--inning` not required, `--starter` optional (auto-resolves)
    - When target is `reached_base`: `--inning` and `--starter` required (Phase 1 behavior preserved)
    - Output JSON with `win_probability` (game_winner) or `expected_runs` (runs) field
    - Exit codes: 2 (invalid input), 3 (missing artifact), 1 (other errors)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  - [x] 15.2 Implement `sandy today` command in `sandy/cli/today_cmd.py`
    - Display table: game time (local TZ), away team, home team, away pitcher, home pitcher, status
    - Show "TBD" for unannounced pitchers
    - Show "No MLB games scheduled for today" when empty
    - Exit code 0 on success, 1 on API errors
    - Register in `sandy/cli/main.py`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [x] 15.3 Implement `sandy predict-all` command in `sandy/cli/predict_all_cmd.py`
    - Fetch today's schedule, resolve probable pitchers
    - Run game_winner + runs predictions for every game with both pitchers announced
    - Output summary table: away team, home team, P(home wins), away expected runs, home expected runs, away starter, home starter
    - Skip games with TBD pitchers (note in output)
    - Support `--json` flag for JSON array output
    - Exit code 0 on success, 3 if model artifacts missing, 1 on API errors
    - Register in `sandy/cli/main.py`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  - [ ] 15.4 Write property-based test for predict-all JSON serialization
    - **Property 10: Predict-all JSON serialization**
    - Generate random lists of prediction results
    - Assert: `--json` output is valid JSON array with required fields per element
    - **Validates: Requirements 11.4**
  - [x] 15.5 Extend `sandy labels build` and `sandy features build` CLI commands to support `--target` flag dispatching to game-level runners
    - `sandy labels build --target game_winner` → `run_game_winner_labels()`
    - `sandy labels build --target runs` → `run_runs_labels()`
    - `sandy features build --target game_winner` (or `runs`) → `run_game_features()`
    - Default behavior (no `--target` or `--target reached_base`) unchanged
    - _Requirements: 12.1, 12.2, 12.3_
  - [x] 15.6 Extend `sandy train` CLI command to support `--target` flag
    - `sandy train --target game_winner` → `train_game_winner_model()`
    - `sandy train --target runs` → `train_runs_model()`
    - Default (`--target reached_base`) unchanged
    - _Requirements: 4.5, 5.4_

- [ ] 16. Checkpoint — CLI and predictor
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. End-to-end integration test
  - [ ] 17.1 Write integration test covering the full Phase 1.5 pipeline
    - Spin up Postgres via `testcontainers[postgres]`, bootstrap schema
    - Seed minimal dataset: two teams, two players as starters, 20+ Final regular-season games with scores
    - Run `sandy labels build --target game_winner` and `sandy labels build --target runs`
    - Run `sandy features build --target game_winner`
    - Run `sandy train --target game_winner --seed 42` and `sandy train --target runs --seed 42`
    - Invoke `sandy predict --target game_winner --team ... --opp ... --starter "..."` via subprocess
    - Assert: exit code 0, valid JSON on stdout, `win_probability` ∈ [0.0, 1.0]
    - Invoke `sandy predict --target runs --team ... --opp ... --starter "..."` via subprocess
    - Assert: exit code 0, valid JSON on stdout, `expected_runs` >= 0.0
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 6.2, 6.3_

- [ ] 18. Final checkpoint — Full suite
  - Ensure all tests pass (`uv run pytest` exits 0), ask the user if questions arise.

## Notes

- All property-based tests (tasks 4.2, 5.2, 6.2, 7.2, 7.3, 10.4, 10.5, 12.3, 12.4, 14.2, 14.3, 15.4) are mandatory — they validate the 12 correctness properties from the design document.
- Each property-based test uses Hypothesis with `@settings(max_examples=100)` minimum.
- Checkpoints (tasks 3, 9, 13, 16, 18) catch integration problems early before the next layer is built.
- The existing `reached_base` pipeline is never modified — all changes are additive. Backward compatibility is maintained via default parameter values and config fallbacks.
- The `predict_game()` function is the primary importable API for Phase 2+ agents.
- Test files follow the existing naming convention: `tests/test_pbt_{property_name}.py`.
