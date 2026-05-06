# Implementation Plan: Sandy — Phase 2 (Live Assistant + MCP + Confidence + Self-Evaluation)

## Overview

This plan transforms Sandy from an offline prediction system into a live, conversational assistant. The build order is strictly bottom-up — DB schema → dataclasses → live client → shutdown features → confidence assessor → over/under computation → prediction logger + reconciler → calibration reporter → MCP server → OpenCLAW registration → daily refresh → property-based tests → integration tests → model retrain — so every stage is independently runnable and testable before the one that depends on it.

All code targets Python 3.11, managed by `uv`. Reuses the existing EC2 + Postgres infrastructure. The MCP server uses stdio transport (no network hops). Property-based tests are mandatory and validate the 11 correctness properties defined in the design.

## Tasks

- [ ] 1. DB schema — prediction_log table
  - [ ] 1.1 Add DDL for `derived.prediction_log` table to `sandy/db.py`
    - Columns: `id` (BIGSERIAL PK), `game_pk` (INTEGER NOT NULL, FK → raw.games), `target` (TEXT NOT NULL), `team_code` (CHAR(3) NOT NULL), `inning_number` (SMALLINT nullable), `probability` (REAL NOT NULL), `confidence_level` (TEXT NOT NULL, CHECK IN ('HIGH','LOW')), `features_snapshot` (JSONB NOT NULL), `predicted_at_utc` (TIMESTAMPTZ NOT NULL DEFAULT now()), `actual_outcome` (TEXT nullable), `outcome_filled_at_utc` (TIMESTAMPTZ nullable), `was_correct` (BOOLEAN nullable)
    - Add indexes: `prediction_log_game_idx` on (game_pk), `prediction_log_target_idx` on (target, predicted_at_utc), `prediction_log_unresolved_idx` on (game_pk) WHERE actual_outcome IS NULL
    - Use `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` for idempotent bootstrap
    - _Requirements: 7.1_

- [ ] 2. New dataclasses
  - [ ] 2.1 Create `sandy/live/schemas.py` with `LiveGameState` and `ShutdownFeatures` frozen dataclasses
    - `LiveGameState`: game_pk, inning_number, inning_half, home_team_code, away_team_code, home_score, away_score, current_pitcher_name, current_pitcher_id, pitch_count, batters_due_up (list[str]), previous_inning_summary (str), fetched_at_utc (datetime), is_final (bool)
    - Include `staleness_seconds()` method, `to_dict()` and `from_dict()` for serialization round-trip
    - `ShutdownFeatures`: pitcher_zero_baserunner_innings (int), is_bottom_of_order (bool), pitcher_game_k_rate (float), team_season_k_rate (float), is_fresh_reliever (bool)
    - Create `sandy/live/__init__.py`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.1_
  - [ ] 2.2 Create `sandy/confidence/__init__.py` and add `ConfidenceResult` frozen dataclass to `sandy/confidence/assessor.py`
    - Fields: level (str "HIGH"|"LOW"), base_rate (float), deviation (float), explanation (str), shutdown_factors (list[str])
    - _Requirements: 5.1, 5.5_
  - [ ] 2.3 Create `sandy/evaluation/__init__.py` and add `CalibrationBucket` and `CalibrationReport` frozen dataclasses to `sandy/evaluation/reporter.py`
    - `CalibrationBucket`: range_start, range_end, prediction_count, actual_rate (float|None), is_sufficient (bool)
    - `CalibrationReport`: total_predictions, date_range_start, date_range_end, accuracy_by_target (dict), accuracy_by_confidence (dict), calibration_buckets (list), natural_language_summary (str)
    - _Requirements: 9.3, 9.4_
  - [ ] 2.4 Add `TotalRunsResult` and `OverUnderLine` frozen dataclasses to `sandy/schemas.py`
    - `OverUnderLine`: threshold (float), probability_over (float)
    - `TotalRunsResult`: home_expected_runs (float), away_expected_runs (float), total_expected_runs (float), over_under_lines (list[OverUnderLine]), residual_std (float)
    - _Requirements: 13.1, 13.2, 14.1_

- [ ] 3. Checkpoint — Schema and dataclasses
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Live game state client
  - [ ] 4.1 Implement `get_live_game_state(team_code: str, config: Config | None = None) -> LiveGameState` in `sandy/live/client.py`
    - Resolve game_pk from today's schedule via `/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,linescore`
    - Fetch full live feed from `/v1.1/game/{game_pk}/feed/live`
    - Parse response into `LiveGameState` dataclass
    - Reuse existing `MlbStatsClient` for rate limiting and retries
    - Raise `NoActiveGameError` when no game in progress for the team
    - Raise `LiveStateError` on API failure without crashing the process
    - Make exactly one HTTP request per invocation (on-demand, no polling)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  - [ ] 4.2 Implement `compute_shutdown_features(live_state: LiveGameState, conn: Connection, team_code: str) -> ShutdownFeatures` in `sandy/live/client.py`
    - Compute pitcher_zero_baserunner_innings from live feed play-by-play
    - Compute is_bottom_of_order from batters_due_up lineup positions
    - Compute pitcher_game_k_rate from live pitcher stats
    - Compute team_season_k_rate from DB (raw.plays aggregate)
    - Compute is_fresh_reliever from pitch_count < 20 and not starter
    - Return null/default ShutdownFeatures when live state is unavailable
    - _Requirements: 6.1, 6.2_

- [ ] 5. Shutdown features — extend feature builder to v4
  - [ ] 5.1 Extend `FEATURE_NAMES` in `sandy/features/schema.py` with 5 new shutdown features, bump `FEATURE_SCHEMA_VERSION` to 4
    - Add: `pitcher_zero_baserunner_innings`, `is_bottom_of_order`, `pitcher_game_k_rate`, `team_season_k_rate`, `is_fresh_reliever`
    - Update assertion to `len(FEATURE_NAMES) == 25`
    - _Requirements: 15.1_
  - [ ] 5.2 Extend `build_feature_vector()` in `sandy/features/builder.py` to compute shutdown features from play-by-play data
    - For historical training: compute from `raw.plays` and `raw.pitcher_game_stats` with `cutoff_ts` leakage prevention
    - For live predictions: accept optional `ShutdownFeatures` parameter and use directly
    - Default to 0/False when live state is unavailable (pre-game predictions)
    - _Requirements: 15.1, 15.2_

- [ ] 6. Confidence assessor
  - [ ] 6.1 Implement `ConfidenceAssessor` class in `sandy/confidence/assessor.py`
    - `BASE_RATES = {"reached_base": 0.72, "game_winner": 0.50, "runs": 4.5}`
    - `THRESHOLD = 0.05` (±5 percentage points)
    - `assess(prediction, target, top_features=None, shutdown_features=None) -> ConfidenceResult`
    - LOW if `abs(prediction - base_rate) <= THRESHOLD`, explanation = "close to base rate, nothing unusual detected."
    - HIGH if deviation > THRESHOLD, explanation built from top contributing features + active shutdown factors
    - Total function — never raises for valid inputs (clamp NaN/Inf, handle unknown targets)
    - When prediction below base rate and `pitcher_zero_baserunner_innings >= 3`, reference consecutive shutout innings in explanation and populate `shutdown_factors`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.3, 6.4_

- [ ] 7. Over/under probability computation
  - [ ] 7.1 Implement `predict_total_runs()` in `sandy/predict/predictor.py`
    - Accept: conn, home_team_code, away_team_code, game_date, config
    - Get per-team predicted runs from existing `runs` model (home and away separately)
    - Sum for total: `μ = home_predicted + away_predicted`
    - Use historical residual std from artifact metadata (default σ ≈ 2.8)
    - Compute `P(total > t) = 1 - Φ((t - μ) / σ)` for thresholds [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    - Return `TotalRunsResult` with per-team runs, total, over_under_lines, residual_std
    - _Requirements: 13.1, 13.2, 14.1, 14.2, 14.5_

- [ ] 8. Checkpoint — Core prediction logic
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Prediction logger
  - [ ] 9.1 Implement `PredictionLogger` class in `sandy/evaluation/logger.py`
    - `__init__(self, engine: Engine)`
    - `log_prediction(game_pk, target, team_code, inning_number, probability, confidence_level, features_snapshot) -> int`
    - INSERT into `derived.prediction_log`, return row ID
    - On DB failure: log WARNING, return -1 (non-blocking)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 10. Outcome reconciler
  - [ ] 10.1 Implement `reconcile_outcomes(engine: Engine) -> int` in `sandy/evaluation/reconciler.py`
    - Only update rows where `actual_outcome IS NULL` and game status is "Final"
    - For reached_base: set actual_outcome to "true"/"false" based on whether team reached base in that inning
    - For game_winner: set actual_outcome to "true"/"false" based on whether predicted team won
    - For runs: set actual_outcome to string of actual runs scored
    - Set `was_correct` consistently (probability > 0.5 matched actual for binary targets)
    - Set `outcome_filled_at_utc` to now()
    - Use single transaction per game_pk
    - Return count of updated rows
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 11. Calibration reporter
  - [ ] 11.1 Implement `get_calibration_report(engine: Engine, days: int = 7) -> CalibrationReport` in `sandy/evaluation/reporter.py`
    - Compute accuracy grouped by target, confidence level
    - Compute calibration buckets (10pp ranges: 0-10%, 10-20%, ..., 90-100%)
    - Mark buckets with < 10 predictions as `is_sufficient=False`, `actual_rate=None`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [ ] 11.2 Implement `get_calibration_summary(report: CalibrationReport) -> str` in `sandy/evaluation/reporter.py`
    - Generate natural-language summary suitable for agent system prompts
    - Flag targets with accuracy < 55% as "unreliable"
    - Flag targets with HIGH-confidence accuracy > 65% as "reliable"
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 12. Checkpoint — Evaluation pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. MCP server + tool definitions
  - [ ] 13.1 Add `mcp` Python SDK as a dependency in `pyproject.toml`
    - `uv add mcp`
    - _Requirements: 3.2, 11.1_
  - [ ] 13.2 Create `sandy/mcp/__init__.py` and `sandy/mcp/server.py`
    - Implement MCP server main loop using stdio transport
    - Read JSON-RPC from stdin, write responses to stdout
    - Register tool definitions with JSON Schema for inputs/outputs
    - Catch all exceptions within handlers — never crash the process
    - Validate model artifacts and DB connectivity at startup (log warnings if missing)
    - Entry point: `python -m sandy.mcp.server`
    - _Requirements: 3.2, 3.4, 11.1, 11.4, 11.5_
  - [ ] 13.3 Implement tool handlers in `sandy/mcp/tools.py`
    - `get_todays_schedule`: no params → list of ScheduledGame
    - `get_live_game_state`: team_code → LiveGameState
    - `predict_reached_base`: team_code, opponent_code?, inning? → PredictionResponse (probability, confidence, top_features, live_state)
    - `predict_game_winner`: team_code, opponent_code? → PredictionResponse (win_probability, confidence)
    - `predict_total_runs`: team_code, opponent_code? → TotalRunsResponse (expected runs, over/under lines)
    - `get_player_stats`: player_name → PlayerStatsResponse
    - `get_calibration_report`: days? → CalibrationReport
    - Auto-fetch live game state when only team_code provided and game is active
    - Determine next half-inning for the team's next at-bat when inning not specified
    - Include live game state in prediction responses for agent verification
    - Log every prediction via PredictionLogger automatically
    - Fall back to pre-game prediction when live state unavailable
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 7.2, 7.3, 9.6, 10.4, 12.1, 12.2, 12.3, 12.4, 13.3, 14.3_

- [ ] 14. OpenCLAW registration
  - [ ] 14.1 Document MCP server registration in OpenCLAW's config
    - Create/update OpenCLAW MCP config to include Sandy as a stdio server: `{"mcpServers": {"sandy": {"command": "python", "args": ["-m", "sandy.mcp.server"], "env": {}}}}`
    - Verify the server starts and responds to `tools/list` JSON-RPC call
    - _Requirements: 3.6, 11.1, 11.2_

- [ ] 15. Daily refresh (cron/CLI)
  - [ ] 15.1 Implement `sandy refresh` CLI command in `sandy/cli/refresh_cmd.py`
    - Run `ingest incremental` + `labels build` + `features build` for all targets in sequence
    - Log summary line with games added, labels generated, features built
    - On failure: log error, retry once after 30 seconds (simplified from 30 min for CLI context)
    - Never trigger model retraining automatically
    - Register in `sandy/cli/main.py`
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  - [ ] 15.2 Create systemd timer or cron configuration file for daily refresh
    - Default schedule: 06:00 local time
    - Run `sandy refresh` with appropriate config path
    - Log output to journald/syslog
    - Complete within 10 minutes under normal conditions
    - _Requirements: 16.1, 16.4, 16.7_

- [ ] 16. Checkpoint — MCP server and daily refresh
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Property-based tests (11 properties)
  - [ ] 17.1 Write property test: LiveGameState serialization round-trip
    - **Property 1: LiveGameState Serialization Round-Trip**
    - Generate random LiveGameState instances (inning 0-20, scores 0-99, team codes, pitcher names, batters_due_up 0-3 items, timestamps)
    - Assert: `LiveGameState.from_dict(state.to_dict()) == state`
    - File: `tests/test_pbt_live_state_roundtrip.py`
    - **Validates: Requirements 2.4**
  - [ ] 17.2 Write property test: Confidence classification correctness
    - **Property 2: Confidence Classification Correctness**
    - Generate random predictions in [0.0, 1.0] and random targets ("reached_base", "game_winner", "runs")
    - Assert: LOW if `abs(prediction - base_rate) <= 0.05`, HIGH otherwise
    - Assert: `base_rate` equals target's known base rate, `deviation` equals `prediction - base_rate`
    - Assert: never raises (totality)
    - File: `tests/test_pbt_confidence_classification.py`
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6**
  - [ ] 17.3 Write property test: Shutdown features in confidence explanation
    - **Property 3: Shutdown Features in Confidence Explanation**
    - Generate random predictions below base rate classified as HIGH confidence
    - Generate random ShutdownFeatures with `pitcher_zero_baserunner_innings >= 3`
    - Assert: explanation references consecutive shutout innings, `shutdown_factors` is non-empty
    - File: `tests/test_pbt_confidence_shutdown.py`
    - **Validates: Requirements 6.3, 6.4, 15.5**
  - [ ] 17.4 Write property test: Outcome reconciliation correctness
    - **Property 4: Outcome Reconciliation Correctness**
    - Generate random prediction sets with known game outcomes
    - Assert: reached_base → actual_outcome matches whether team reached base
    - Assert: game_winner → actual_outcome matches whether team won
    - Assert: runs → actual_outcome equals actual runs scored
    - Assert: was_correct set consistently
    - File: `tests/test_pbt_reconciliation_correctness.py`
    - **Validates: Requirements 8.2, 8.3, 8.4**
  - [ ] 17.5 Write property test: Reconciliation idempotence
    - **Property 5: Reconciliation Idempotence**
    - Generate random prediction sets, run reconcile twice
    - Assert: second call updates zero rows, all values unchanged
    - File: `tests/test_pbt_reconciliation_idempotence.py`
    - **Validates: Requirements 8.6**
  - [ ] 17.6 Write property test: Calibration accuracy computation
    - **Property 6: Calibration Accuracy Computation**
    - Generate random sets of (probability, was_correct) pairs
    - Assert: computed accuracy equals `count(was_correct=True) / count(total)` for each grouping
    - File: `tests/test_pbt_calibration_accuracy.py`
    - **Validates: Requirements 9.1**
  - [ ] 17.7 Write property test: Calibration bucket assignment
    - **Property 7: Calibration Bucket Assignment**
    - Generate random prediction sets with varying bucket sizes
    - Assert: each prediction lands in exactly one bucket
    - Assert: buckets with < 10 predictions have `is_sufficient=False` and `actual_rate=None`
    - File: `tests/test_pbt_calibration_buckets.py`
    - **Validates: Requirements 9.2, 9.4**
  - [ ] 17.8 Write property test: Calibration summary flags
    - **Property 8: Calibration Summary Flags**
    - Generate random CalibrationReports with controlled accuracy values
    - Assert: accuracy < 55% → summary contains "unreliable"
    - Assert: HIGH-confidence accuracy > 65% → summary contains "reliable"
    - File: `tests/test_pbt_calibration_summary.py`
    - **Validates: Requirements 10.2, 10.3**
  - [ ] 17.9 Write property test: Over/under probability computation
    - **Property 9: Over/Under Probability Computation**
    - Generate random (μ, σ) pairs with μ > 0 and σ > 0
    - Assert: `P(total > t) == 1 - Φ((t - μ) / σ)` for each threshold (verify against scipy.stats.norm.sf)
    - Assert: probabilities are monotonically decreasing as thresholds increase
    - File: `tests/test_pbt_over_under.py`
    - **Validates: Requirements 14.1, 14.5**
  - [ ] 17.10 Write property test: Next half-inning determination
    - **Property 10: Next Half-Inning Determination**
    - Generate random LiveGameState (inning 1-9, top/bottom, home/away team)
    - Assert: computed next at-bat inning is correct based on which half the team bats in
    - File: `tests/test_pbt_next_inning.py`
    - **Validates: Requirements 12.3**
  - [ ] 17.11 Write property test: Total runs summation
    - **Property 11: Total Runs Summation**
    - Generate random non-negative float pairs (home_expected, away_expected)
    - Assert: `total_expected_runs == home_expected_runs + away_expected_runs` exactly
    - File: `tests/test_pbt_total_runs.py`
    - **Validates: Requirements 13.1**

- [ ] 18. Integration tests
  - [ ] 18.1 Write integration test: MCP → Sandy → DB full flow
    - Send tool invocation via stdin to MCP server, verify JSON-RPC response on stdout
    - Verify prediction logged in `derived.prediction_log`
    - File: `tests/test_e2e_mcp_flow.py`
    - _Requirements: 3.2, 3.3, 7.2_
  - [ ] 18.2 Write integration test: Prediction → Log → Reconcile → Calibration
    - Make prediction, mark game Final in DB, run reconcile, verify was_correct
    - Run calibration report, verify accuracy computation
    - File: `tests/test_e2e_mcp_flow.py`
    - _Requirements: 7.1, 8.1, 8.5, 9.1_
  - [ ] 18.3 Write integration test: Live state → Prediction with mocked MLB API
    - Mock MLB API responses, invoke `predict_reached_base` with only team_code
    - Verify live state is fetched and used for prediction context
    - File: `tests/test_e2e_mcp_flow.py`
    - _Requirements: 1.1, 12.1, 12.2_
  - [ ] 18.4 Write integration test: Daily refresh end-to-end
    - Run `sandy refresh` against test DB
    - Verify games ingested + labels + features built
    - File: `tests/test_e2e_mcp_flow.py`
    - _Requirements: 16.1, 16.2, 16.3_

- [ ] 19. Retrain reached_base model with shutdown features
  - [ ] 19.1 Rebuild inning features with v4 schema (includes shutdown features)
    - Run `sandy features build` to recompute all inning_features rows with the 5 new shutdown columns
    - Verify feature_schema_version = 4 in all new rows
    - _Requirements: 15.1, 15.2_
  - [ ] 19.2 Retrain reached_base model on v4 features
    - Run `sandy train --target reached_base --seed 42`
    - Log validation ROC AUC — must achieve >= 0.55 (improvement over current 0.54)
    - Save new artifact with feature_schema_version = 4
    - _Requirements: 15.3, 15.4_

- [ ] 20. Final checkpoint — Full suite
  - Ensure all tests pass (`uv run pytest` exits 0), ask the user if questions arise.

## Notes

- All property-based tests (tasks 17.1–17.11) are mandatory — they validate the 11 correctness properties from the design document.
- Each property-based test uses Hypothesis with `@settings(max_examples=100)` minimum.
- Checkpoints (tasks 3, 8, 12, 16, 20) catch integration problems early before the next layer is built.
- The MCP Python SDK (`mcp`) must be added as a dependency via `uv add mcp`.
- The existing `reached_base` pipeline continues to work until task 19 bumps the feature schema to v4.
- The `predict_from_features()` and `predict_game()` functions from Phase 1/1.5 remain the stable entry points — Phase 2 wraps them with live context and confidence signals.
- Test files follow the existing naming convention: `tests/test_pbt_{property_name}.py` for PBT, `tests/test_e2e_{flow}.py` for integration.
- The MCP server entry point is `python -m sandy.mcp.server` (stdio transport, no network listener).
