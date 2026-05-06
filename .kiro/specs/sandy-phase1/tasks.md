# Implementation Plan: Sandy — Phase 1

## Overview

This plan converts the Phase 1 design into incremental coding tasks. The build order is strictly bottom-up — scaffolding → DB → ingestion → labels → features → training → predict → CLI — so every stage is independently runnable and testable before the one that depends on it is written. The four property-based tests mandated by requirement 11 and the end-to-end integration test are included as required sub-tasks alongside the implementation they validate.

All code targets Python 3.11, managed by `uv`. Postgres runs via docker-compose; integration tests use `testcontainers[postgres]`. Optional sub-tasks are marked with `*` and can be skipped for a minimal MVP; the four PBT tasks and the E2E integration test are intentionally non-optional.

## Tasks

- [ ] 1. Scaffold project, package layout, and runtime
  - [x] 1.1 Create `pyproject.toml` managed by `uv` with Python pinned to `==3.11.*`, runtime deps (click, lightgbm, numpy, pandas, psycopg[binary], scikit-learn, sqlalchemy, tomli), dev deps (hypothesis, pytest, pytest-cov, testcontainers[postgres]), and the `sandy = "sandy.cli.main:cli"` console script
    - _Requirements: 12.1, 12.2, 12.4_
  - [x] 1.2 Create `docker-compose.yml` with a `postgres:16` service bound to a named volume `sandy_pgdata`, healthcheck, and env-var-driven credentials
    - _Requirements: 3.5, 12.3_
  - [x] 1.3 Create the `sandy/` package skeleton (`config.py`, `logging.py`, `db.py`, `schemas.py`, and empty `ingest/`, `labels/`, `features/`, `train/`, `predict/`, `cli/` subpackages with `__init__.py` files) plus a `sandy.example.toml`
    - _Requirements: 9.4, 12.4_
  - [ ]* 1.4 Configure `ruff` and `mypy` (line length 100, target `py311`) and add a pre-commit entry point
    - _Requirements: 12.1_

- [ ] 2. Implement configuration and structured logging
  - [x] 2.1 Implement the `Config` dataclass hierarchy (`DatabaseConfig`, `ModelConfig`, `IngestConfig`, `TrainingConfig`, `LoggingConfig`) and `load_config()` with precedence defaults < TOML < env < CLI flags, including fallback of `MLB_MODEL_PATH` to `./models/latest.pkl`
    - _Requirements: 9.1, 9.3, 9.4_
  - [x] 2.2 Implement JSON-formatter logging (`sandy/logging.py`) emitting `timestamp`, `level`, `component`, `message` with extras, honoring `MLB_LOG_LEVEL` and `--log-level`
    - _Requirements: 10.1, 10.3_
  - [x] 2.3 Implement the "missing required env var" guard that exits with status 4 and names the missing variable
    - _Requirements: 9.2_
  - [ ]* 2.4 Write unit tests for config precedence and the missing-env-var exit path
    - _Requirements: 9.1, 9.2, 9.4_

- [ ] 3. Implement the database layer
  - [x] 3.1 Implement `sandy/db.py`: SQLAlchemy engine factory from `Config.database`, session context manager, and a `bootstrap_schema()` helper that runs DDL idempotently
    - _Requirements: 3.5, 3.6, 9.1_
  - [x] 3.2 Write DDL for the `raw` schema: `teams`, `players`, `games` (with FKs, indexes, `raw_payload_hash`), `plays` (composite PK, `is_reaches_base`, JSONB `raw`, `ON DELETE CASCADE` to games), `pitcher_game_stats`, and `ingest_failures`
    - _Requirements: 3.1, 3.3_
  - [x] 3.3 Write DDL for the `derived` schema: `inning_labels` and `inning_features` with composite PK `(game_pk, team_code, inning_number)` and the `feature_schema_version` column on `inning_features`
    - _Requirements: 3.2, 3.4, 5.5_
  - [x] 3.4 Define `TypedDict` / frozen-dataclass row types in `sandy/schemas.py` for every raw and derived table, plus `FeatureVector`, `ModelArtifact`, `TopFeature`, `PredictionResult`
    - _Requirements: 3.1, 3.2, 7.1_

- [x] 4. Checkpoint — DB and config bootstrap
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement MLB Stats API client and parsers
  - [x] 5.1 Implement `MlbStatsClient` in `sandy/ingest/client.py` with a token-bucket limiter (≤ `config.ingest.max_rps`, default 10), exponential-backoff retry (base 1.0s, factor 2, jitter ±25%, max 5 attempts) on 429 and 5xx, and non-retryable classification for other 4xx / JSON decode errors
    - _Requirements: 1.3, 1.7_
  - [x] 5.2 Implement pure parsers in `sandy/ingest/parsers.py` converting `/v1/schedule`, `/v1.1/game/{pk}/feed/live`, `/v1/teams`, and `/v1/people` responses into the `TypedDict` row types from task 3.4, including derivation of `is_reaches_base` from the MLB event code set `{single, double, triple, home_run, walk, hit_by_pitch, field_error}` and `raw_payload_hash` (sha256 of the source JSON)
    - _Requirements: 1.1, 4.2_
  - [ ]* 5.3 Write unit tests for the parsers using recorded fixture JSON for a known `game_pk`
    - _Requirements: 1.1_

- [ ] 6. Implement the ingestion service
  - [x] 6.1 Implement `backfill_seasons()` in `sandy/ingest/service.py`: enumerate regular-season `game_pk`s via `/v1/schedule` in one-month windows for the three most recent complete seasons, skip games already in `raw.games` with `status='Final'`, and UPSERT each game in a single transaction (`DELETE plays WHERE game_pk=?; UPSERT games/players/pitcher_game_stats; INSERT plays`) with a progress log line every 50 games containing `games_processed`, `games_remaining`, `elapsed_seconds`
    - _Requirements: 1.1, 1.2, 1.5, 1.6, 10.2_
  - [x] 6.2 Implement the ingest-failure path: after 5 exhausted retries or a non-retryable error, insert into `raw.ingest_failures` with `game_pk`, `endpoint`, `error_reason`, `http_status`, `retries` and continue processing remaining games
    - _Requirements: 1.4_
  - [x] 6.3 Implement `incremental_ingest()`: compute `max_final_date` from `raw.games`, fetch the schedule from that date through today, skip games already `Final`, replace non-final→Final games with the same transaction as backfill, and emit the summary line with `games_added`, `games_updated`, `games_skipped`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 6.4 Write property-based test for idempotent ingestion
    - **Property 1: Idempotent ingestion** — running the Ingestion_Service twice over the same fixed set of mocked MLB Stats API responses produces byte-equivalent rows across all `raw.*` tables.
    - Use Hypothesis to generate arbitrary ordered sequences of mocked `game_pk` responses and assert that the second run's `SELECT * ORDER BY PK` output is identical to the first's.
    - **Validates: Requirements 1.2, 2.4, 11.2**
  - [ ]* 6.5 Add a `--dry-run` flag to ingest that fetches and parses but skips DB writes, logging what would have been written
    - _Requirements: 10.1_

- [ ] 7. Implement the label generator
  - [-] 7.1 Implement `generate_labels_for_game()` (pure) in `sandy/labels/generator.py`: read `raw.plays` for a `game_pk`, skip if `raw.games.status != 'Final'`, group by `(batting_team_code, inning)`, emit one `InningLabel` per group with `reached_base = BOOL_OR(is_reaches_base)`
    - _Requirements: 4.1, 4.2, 4.3, 4.6_
  - [-] 7.2 Implement the labels runner in `sandy/labels/runner.py` iterating all Final games, calling the pure generator, and UPSERTing `derived.inning_labels` keyed by `(game_pk, team_code, inning_number)` with a final log line containing `duration_seconds`, `rows_read`, `rows_written`
    - _Requirements: 3.4, 4.4, 10.1, 10.2_
  - [x] 7.3 Write property-based test for label monotonicity
    - **Property 2: Label monotonicity** — for any synthetic play-by-play where at least one reach-base event (`single`, `double`, `triple`, `home_run`, `walk`, `hit_by_pitch`, `field_error`) exists for a `(game_pk, team_code, inning)`, the Label_Generator produces `reached_base = true` for that row.
    - Use Hypothesis to generate arbitrary lists of play rows with at least one reach-base event injected per target inning, run the generator against an in-memory DB, and assert `reached_base` is `true`.
    - **Validates: Requirements 4.2, 4.5, 11.3**

- [ ] 8. Implement the feature builder
  - [x] 8.1 Implement `sandy/features/schema.py` with `FEATURE_SCHEMA_VERSION = 1` and `FEATURE_NAMES` (the 12 names from the design: `opp_starter_era`, `opp_starter_whip`, `opp_starter_k9`, `opp_starter_pitches_before`, `lineup_spot_1..3`, `is_home`, `ballpark_id`, `inning_number_feat`, `trailing15_rpg`, `trailing15_obp`)
    - _Requirements: 5.2, 5.5_
  - [x] 8.2 Implement `build_feature_vector()` (pure) in `sandy/features/builder.py` using a `cutoff_ts` computed from `min(start_time_utc)` of the target half-inning, bounding every same-game aggregate to rows strictly before `cutoff_ts` and cross-game trailing-15 aggregates to `game_date < :game_date`; handle the `game_pk=None` / `as_of` path for prediction with empty same-game history (pitches_before=0, lineup_spots=(1,2,3))
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 8.3 Implement the features runner in `sandy/features/runner.py` iterating all Final-game innings, omitting any row where a feature is uncomputable and logging the omitted `(game_pk, team_code, inning_number)` with the missing feature name, then UPSERTing `derived.inning_features` with `feature_schema_version` stamped on every row
    - _Requirements: 3.4, 5.4, 5.5, 10.1, 10.2_
  - [ ]* 8.4 Write unit tests asserting feature-builder determinism (identical inputs → identical outputs) and leakage prevention (no event with `start_time_utc >= cutoff_ts` influences the result)
    - _Requirements: 5.1, 5.3_

- [ ] 9. Checkpoint — Ingestion, labels, and features pipelines
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement the training pipeline
  - [x] 10.1 Implement `chronological_split()` (pure) in `sandy/train/split.py`: sort unique `(game_date, game_pk)` ascending, define the last 15% of games by date as the validation set, and assign rows to training or validation by their `game_pk`
    - _Requirements: 6.2_
  - [x] 10.2 Implement `train_model()` in `sandy/train/trainer.py`: load the labels⨝features frame for the training window, split via 10.1, fit LightGBM with `objective=binary`, `deterministic=True`, `force_col_wise=True`, `seed=<--seed>`, `num_boost_round=500`, `early_stopping_rounds=50`; compute ROC AUC, log loss, Brier score, positive/negative counts on validation and log them as a single INFO JSON line; raise `TrainingQualityError` if validation ROC AUC < 0.52
    - _Requirements: 6.1, 6.3, 6.4, 6.6_
  - [x] 10.3 Implement `save_artifact()` (atomic `.tmp` + `replace`) and `load_artifact()` in `sandy/train/artifact.py`, serializing via LightGBM `model_to_string()` inside a pickle dict with keys `model`, `feature_names`, `feature_schema_version`, `training_window_start`, `training_window_end`, `created_at`; raise `FeatureSchemaMismatch` on load if the stored version ≠ current
    - _Requirements: 6.5, 7.1, 7.3_
  - [x] 10.4 Write property-based test for model serializer round-trip
    - **Property 3: Serializer round-trip** — saving a trained LightGBM model via `save_artifact()` and loading it via `load_artifact()` produces predictions matching the in-memory model within 1e-9 absolute tolerance on a fixed input.
    - Use Hypothesis to generate fixed input feature matrices, fit a deterministic LightGBM, save and reload, then assert `abs(reloaded.predict(x) - original.predict(x)) < 1e-9` element-wise.
    - **Validates: Requirements 7.2, 11.4**

- [ ] 11. Implement the predictor
  - [x] 11.1 Implement `predict_from_features()` (pure) in `sandy/predict/predictor.py`: verify `features.feature_schema_version == artifact.feature_schema_version`, compute probability via `artifact.model.predict(x)`, compute SHAP-style contributions via `pred_contrib=True`, return a `PredictionResult` with `probability` and the top-5 features by `|contribution|` descending
    - _Requirements: 7.3, 8.2, 8.3_
  - [x] 11.2 Implement the high-level `predict()` wrapper: validate `inning ∈ 1..9` (exit 2), resolve `--team` and `--opp` against `raw.teams.team_code` case-insensitively (exit 2), resolve `--starter` via exact case-insensitive match with a Jaro-Winkler fuzzy fallback returning top-5 candidates on ambiguity or miss (exit 2), honor `--as-of`, call `build_feature_vector(game_pk=None, as_of=...)`, load the latest artifact (exit 3 if missing), and return the `PredictionResult`
    - _Requirements: 8.1, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x] 11.3 Write property-based test for probability range
    - **Property 4: Probability range** — the `probability` returned by the Predictor falls in the closed interval `[0.0, 1.0]` for every generated input.
    - Use Hypothesis to generate arbitrary valid `FeatureVector`s against a fixed artifact and assert `0.0 <= result.probability <= 1.0`.
    - **Validates: Requirements 8.2, 11.1**

- [ ] 12. Wire up the `sandy` CLI with Click
  - [x] 12.1 Implement the top-level Click group in `sandy/cli/main.py` with `--config`, `--log-level`, and `--version`, wiring `load_config()` and `configure_logging()` into the group's context
    - _Requirements: 9.4, 10.3_
  - [x] 12.2 Implement `sandy ingest backfill [--seasons N] [--start-season YYYY]` and `sandy ingest incremental` subcommands delegating to `ingest.service`, emitting the final `duration_seconds` / `rows_read` / `rows_written` log line
    - _Requirements: 1.1, 2.1, 10.2_
  - [x] 12.3 Implement `sandy labels build [--game-pk INT]` delegating to `labels.runner`
    - _Requirements: 4.1, 10.2_
  - [ ] 12.4 Implement `sandy features build [--game-pk INT]` delegating to `features.runner`
    - _Requirements: 5.1, 10.2_
  - [ ] 12.5 Implement `sandy train [--seed INT] [--output PATH]` delegating to `train.trainer` and `train.artifact`, translating `TrainingQualityError` to exit code 1 with a warning message naming the observed AUC
    - _Requirements: 6.4, 6.5, 6.6, 10.2_
  - [ ] 12.6 Implement `sandy predict --team --opp --inning --starter [--as-of]` printing `PredictionResult.to_json()` to stdout and translating errors to exit codes 2 (invalid input), 3 (missing artifact), 4 (missing env var)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.2_
  - [ ]* 12.7 Write unit tests for the CLI exit-code contracts (exit 2 for bad team/inning/starter; exit 3 for missing artifact; exit 4 for missing env var)
    - _Requirements: 8.4, 8.5, 8.6, 8.8, 9.2_

- [ ] 13. End-to-end integration
  - [x] 13.1 Write the end-to-end integration test: spin up a Postgres via `testcontainers[postgres]`, bootstrap the schema, seed a minimal dataset (one team pair, one player as starter, one Final game with a handful of plays across three innings), run `sandy labels build`, `sandy features build`, `sandy train --seed 42`, then invoke `sandy predict --team ... --opp ... --inning 3 --starter "..."` via `subprocess.run` and assert exit code 0, valid JSON on stdout, and a numeric `probability` in `[0.0, 1.0]`
    - _Requirements: 11.5, 11.6_

- [ ] 14. Final checkpoint — Full suite
  - Ensure all tests pass (`uv run pytest` exits 0), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional sub-tasks and may be skipped for a minimal MVP. All four property-based tests (6.4, 7.3, 10.4, 11.3) and the end-to-end integration test (13.1) are intentionally non-optional per the Phase 1 spec.
- Each property-based test is annotated with its property number (1–4, aligned with requirements 11.1–11.4 in order of appearance in this plan) and the requirements clauses it validates.
- Checkpoints (tasks 4, 9, 14) exist to catch integration problems early before the next pipeline is built on top.
- All ingest, labels, features, and training components write a final JSON log line with `duration_seconds`, `rows_read`, and `rows_written` per requirement 10.2.
- The CLI's JSON output on `sandy predict` is the Phase 1 success criterion; the integration test in 13.1 is the authoritative check that the whole stack wires together correctly.
