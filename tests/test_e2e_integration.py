"""End-to-end integration test for Sandy Phase 1.

Task 13.1: Verifies the entire pipeline works from ingestion through
prediction by:
1. Spinning up a throwaway Postgres via testcontainers
2. Bootstrapping the schema
3. Seeding minimal data (2 teams, players, multiple games with plays)
4. Running labels build, features build, train, predict
5. Asserting valid JSON output with probability in [0.0, 1.0]

This test uses subprocess.run to invoke the CLI exactly as an operator
would, proving the full stack wires together correctly.

Validates: Requirements 11.5, 11.6
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import text

from sandy.config import Config, DatabaseConfig, IngestConfig, LoggingConfig, ModelConfig, TrainingConfig
from sandy.db import bootstrap_schema, create_engine, get_connection


@pytest.fixture(scope="module")
def e2e_env():
    """Set up a complete test environment: Postgres + seeded data + env vars."""
    with PostgresContainer("postgres:16") as pg:
        # Build config
        cfg = Config(
            database=DatabaseConfig(
                host=pg.get_container_host_ip(),
                port=int(pg.get_exposed_port(5432)),
                name=pg.dbname,
                user=pg.username,
                password=pg.password,
            ),
            model=ModelConfig(path=Path(tempfile.mkdtemp()) / "test_model.pkl"),
            ingest=IngestConfig(),
            training=TrainingConfig(),
            logging=LoggingConfig(),
        )

        engine = create_engine(cfg)

        # Bootstrap schema
        with get_connection(engine) as conn:
            bootstrap_schema(conn)

        # Seed data: enough games to train a model
        with get_connection(engine) as conn:
            _seed_test_data(conn)

        # Build env vars for subprocess calls
        env = {
            **os.environ,
            "MLB_DB_HOST": cfg.database.host,
            "MLB_DB_PORT": str(cfg.database.port),
            "MLB_DB_NAME": cfg.database.name,
            "MLB_DB_USER": cfg.database.user,
            "MLB_DB_PASSWORD": cfg.database.password,
            "MLB_MODEL_PATH": str(cfg.model.path),
            "MLB_LOG_LEVEL": "ERROR",  # suppress log noise in tests
        }

        yield cfg, env


def _seed_test_data(conn):
    """Seed minimal but sufficient data for the full pipeline.

    Creates 30 synthetic games with plays across 9 innings each,
    giving ~540 training rows (enough for LightGBM to fit).
    """
    # Teams
    conn.execute(text("""
        INSERT INTO raw.teams (team_code, team_id, name, venue_id)
        VALUES ('SEA', 136, 'Seattle Mariners', 680),
               ('LAD', 119, 'Los Angeles Dodgers', 22)
        ON CONFLICT (team_code) DO NOTHING
    """))

    # Players (starter + batters)
    conn.execute(text("""
        INSERT INTO raw.players (player_id, full_name, primary_position, throws, bats)
        VALUES (100001, 'Test Pitcher', 'P', 'R', 'R'),
               (100002, 'Test Batter 1', 'CF', 'R', 'L'),
               (100003, 'Test Batter 2', '1B', 'L', 'L'),
               (100004, 'Test Batter 3', 'SS', 'R', 'R')
        ON CONFLICT (player_id) DO NOTHING
    """))

    # Generate 30 games across 3 months (enough for trailing-15 + train/val split)
    event_codes = ["single", "double", "walk", "strikeout", "field_out",
                   "home_run", "hit_by_pitch", "field_error", "flyout"]

    import random
    random.seed(42)

    for game_idx in range(30):
        game_pk = 900000 + game_idx
        game_date = date(2023, 4 + (game_idx // 10), 1 + (game_idx % 28))

        # Insert game
        conn.execute(text("""
            INSERT INTO raw.games (
                game_pk, game_date, season, game_type, status,
                home_team_code, away_team_code, venue_id,
                first_pitch_utc, home_score, away_score,
                home_starter_id, away_starter_id, raw_payload_hash
            ) VALUES (
                :game_pk, :game_date, 2023, 'R', 'Final',
                'SEA', 'LAD', 680,
                :first_pitch, :home_score, :away_score,
                100001, 100001, :hash
            )
            ON CONFLICT (game_pk) DO NOTHING
        """), {
            "game_pk": game_pk,
            "game_date": game_date,
            "first_pitch": datetime(2023, game_date.month, game_date.day, 19, 0, tzinfo=timezone.utc),
            "home_score": random.randint(1, 8),
            "away_score": random.randint(0, 7),
            "hash": f"hash_{game_pk}",
        })

        # Insert pitcher game stats
        conn.execute(text("""
            INSERT INTO raw.pitcher_game_stats (
                game_pk, pitcher_id, team_code,
                pitches_thrown, outs_recorded, runs_allowed,
                walks, hits_allowed, strikeouts, is_starter
            ) VALUES (
                :game_pk, 100001, 'SEA',
                :pitches, :outs, :runs, :walks, :hits, :ks, true
            )
            ON CONFLICT (game_pk, pitcher_id) DO NOTHING
        """), {
            "game_pk": game_pk,
            "pitches": random.randint(60, 100),
            "outs": random.randint(12, 21),
            "runs": random.randint(1, 5),
            "walks": random.randint(1, 4),
            "hits": random.randint(3, 8),
            "ks": random.randint(3, 10),
        })

        # Insert plays: 4-6 per inning, 9 innings, both teams
        at_bat_idx = 0
        for inning in range(1, 10):
            for half, batting, pitching in [("top", "LAD", "SEA"), ("bottom", "SEA", "LAD")]:
                n_plays = random.randint(3, 5)
                for play_idx in range(n_plays):
                    event = random.choice(event_codes)
                    is_reach = event in ("single", "double", "home_run", "walk", "hit_by_pitch", "field_error")
                    batter_id = random.choice([100002, 100003, 100004])
                    batting_order = (play_idx % 9) + 1

                    ts = datetime(2023, game_date.month, game_date.day,
                                  19, inning, play_idx * 3, tzinfo=timezone.utc)

                    conn.execute(text("""
                        INSERT INTO raw.plays (
                            game_pk, at_bat_index, inning, half_inning,
                            batting_team_code, pitching_team_code,
                            batter_id, pitcher_id, batting_order,
                            event_type, event_code, is_reaches_base,
                            pitches_in_pa, start_time_utc, end_time_utc, raw
                        ) VALUES (
                            :game_pk, :at_bat, :inning, :half,
                            :batting, :pitching,
                            :batter_id, 100001, :batting_order,
                            :event_type, :event_code, :is_reach,
                            :pitches, :start_ts, :end_ts, '{}'::jsonb
                        )
                        ON CONFLICT (game_pk, at_bat_index) DO NOTHING
                    """), {
                        "game_pk": game_pk,
                        "at_bat": at_bat_idx,
                        "inning": inning,
                        "half": half,
                        "batting": batting,
                        "pitching": pitching,
                        "batter_id": batter_id,
                        "batting_order": batting_order,
                        "event_type": "hit" if is_reach else "out",
                        "event_code": event,
                        "is_reach": is_reach,
                        "pitches": random.randint(1, 7),
                        "start_ts": ts,
                        "end_ts": ts,
                    })
                    at_bat_idx += 1


def _run_sandy(args: list[str], env: dict) -> subprocess.CompletedProcess:
    """Run a sandy CLI command via subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "sandy.cli.main"] + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


class TestEndToEnd:
    """End-to-end integration test: seed → labels → features → train → predict."""

    def test_full_pipeline(self, e2e_env):
        cfg, env = e2e_env

        # Step 1: Build labels
        result = _run_sandy(["labels", "build"], env)
        assert result.returncode == 0, f"labels build failed:\n{result.stderr}"

        # Step 2: Build features
        result = _run_sandy(["features", "build"], env)
        assert result.returncode == 0, f"features build failed:\n{result.stderr}"

        # Step 3: Train model
        result = _run_sandy(["train", "--seed", "42", "--output", str(cfg.model.path)], env)
        assert result.returncode == 0, f"train failed:\n{result.stderr}\n{result.stdout}"

        # Step 4: Predict
        result = _run_sandy([
            "predict",
            "--team", "SEA",
            "--opp", "LAD",
            "--inning", "3",
            "--starter", "Test Pitcher",
        ], env)

        # Assertions
        assert result.returncode == 0, (
            f"predict failed with exit code {result.returncode}:\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

        # Parse JSON output
        output = result.stdout.strip()
        data = json.loads(output)

        assert "probability" in data, f"Missing 'probability' key in output: {data}"
        assert "top_features" in data, f"Missing 'top_features' key in output: {data}"

        prob = data["probability"]
        assert isinstance(prob, (int, float)), f"probability is not numeric: {prob}"
        assert 0.0 <= prob <= 1.0, f"probability {prob} outside [0.0, 1.0]"

        # Verify top_features structure
        assert isinstance(data["top_features"], list)
        assert len(data["top_features"]) <= 5
        for feat in data["top_features"]:
            assert "name" in feat
            assert "contribution" in feat
