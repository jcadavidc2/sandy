"""CLI subcommand: sandy predict-all — batch predict all today's games.

Phase 1.5, Task 15.3: Fetches today's schedule, runs game_winner + runs
predictions for every game with both pitchers announced.

Requirements: 11.1–11.6
"""
from __future__ import annotations

import json
import sys

import click

from sandy.cli._helpers import require_config
from sandy.logging import configure_logging


@click.command("predict-all")
@click.option("--json-output", "json_flag", is_flag=True, default=False, help="Output as JSON array")
@click.pass_context
def predict_all(ctx: click.Context, json_flag: bool) -> None:
    """Predict game_winner + runs for all today's games with announced pitchers."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    try:
        from sandy.schedule.client import get_todays_schedule

        schedule = get_todays_schedule(cfg)
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not schedule:
        click.echo("No MLB games scheduled for today.")
        return

    from sandy.predict.predictor import (
        InvalidInputError,
        MissingArtifactError,
        predict_game,
    )
    from sandy.train.artifact import FeatureSchemaMismatch, TargetMismatchError

    results = []
    skipped = []

    for game in sorted(schedule, key=lambda g: g.game_time_utc):
        if game.home_probable_pitcher is None or game.away_probable_pitcher is None:
            skipped.append(game)
            continue

        try:
            # Game winner prediction
            gw_result = predict_game(
                team=game.home_team_code,
                opp=game.away_team_code,
                target="game_winner",
                starter=game.home_probable_pitcher,
                opp_starter=game.away_probable_pitcher,
                config=cfg,
            )

            # Runs predictions (home)
            home_runs_result = predict_game(
                team=game.home_team_code,
                opp=game.away_team_code,
                target="runs",
                starter=game.home_probable_pitcher,
                opp_starter=game.away_probable_pitcher,
                config=cfg,
            )

            # Runs predictions (away)
            away_runs_result = predict_game(
                team=game.away_team_code,
                opp=game.home_team_code,
                target="runs",
                starter=game.away_probable_pitcher,
                opp_starter=game.home_probable_pitcher,
                config=cfg,
            )

            results.append({
                "home_team": game.home_team_code,
                "away_team": game.away_team_code,
                "win_probability": round(gw_result.probability, 4),
                "home_expected_runs": round(home_runs_result.probability, 2),
                "away_expected_runs": round(away_runs_result.probability, 2),
                "home_starter": game.home_probable_pitcher,
                "away_starter": game.away_probable_pitcher,
            })

        except MissingArtifactError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(3)
        except (FeatureSchemaMismatch, TargetMismatchError) as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(3)
        except InvalidInputError as exc:
            # Skip games where we can't resolve inputs
            click.echo(f"Warning: skipping {game.away_team_code}@{game.home_team_code}: {exc}", err=True)
            continue
        except RuntimeError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

    if json_flag:
        click.echo(json.dumps(results, indent=2))
    else:
        # Table output
        if results:
            header = (
                f"{'Away':<5} {'Home':<5} {'P(Home)':<9} "
                f"{'Away R':<8} {'Home R':<8} "
                f"{'Away Starter':<22} {'Home Starter':<22}"
            )
            click.echo(header)
            click.echo("-" * len(header))

            for r in results:
                click.echo(
                    f"{r['away_team']:<5} "
                    f"{r['home_team']:<5} "
                    f"{r['win_probability']:<9.4f} "
                    f"{r['away_expected_runs']:<8.2f} "
                    f"{r['home_expected_runs']:<8.2f} "
                    f"{r['away_starter']:<22} "
                    f"{r['home_starter']:<22}"
                )

        if skipped:
            click.echo(f"\nSkipped {len(skipped)} game(s) with TBD pitchers:")
            for game in skipped:
                click.echo(f"  {game.away_team_code} @ {game.home_team_code}")
