"""CLI subcommand for prediction.

Task 12.6: sandy predict --team --opp --inning --starter [--as-of]
Phase 1.5: extended with --target option for game_winner/runs predictions.

Requirements: 8.1–8.8, 9.2, 10.1–10.7
"""
from __future__ import annotations

import json
import sys
from datetime import date

import click

from sandy.cli._helpers import require_config
from sandy.logging import configure_logging
from sandy.predict.predictor import (
    InvalidInputError,
    MissingArtifactError,
    predict,
    predict_game,
)
from sandy.train.artifact import FeatureSchemaMismatch, TargetMismatchError


@click.command()
@click.option("--team", required=True, help="Team code (e.g. SEA)")
@click.option("--opp", required=True, help="Opposing team code (e.g. LAD)")
@click.option(
    "--target",
    type=click.Choice(["reached_base", "game_winner", "runs"], case_sensitive=False),
    default="reached_base",
    help="Prediction target (default: reached_base)",
)
@click.option("--inning", type=int, default=None, help="Target inning (1-9, required for reached_base)")
@click.option("--starter", default=None, help="Starting pitcher name (required for reached_base, optional for game targets)")
@click.option("--opp-starter", default=None, help="Opposing starting pitcher name (optional for game targets)")
@click.option("--as-of", type=str, default=None, help="Date ceiling YYYY-MM-DD (default: today)")
@click.pass_context
def predict_cmd(
    ctx: click.Context,
    team: str,
    opp: str,
    target: str,
    inning: int | None,
    starter: str | None,
    opp_starter: str | None,
    as_of: str | None,
) -> None:
    """Emit prediction as JSON (reached_base, game_winner, or runs)."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    # Parse as_of date
    as_of_date: date | None = None
    if as_of is not None:
        try:
            as_of_date = date.fromisoformat(as_of)
        except ValueError:
            click.echo(
                f"Error: invalid --as-of date format: '{as_of}'. Use YYYY-MM-DD.",
                err=True,
            )
            sys.exit(2)

    try:
        if target == "reached_base":
            # Phase 1 behavior: require --inning and --starter
            if inning is None:
                click.echo(
                    "Error: --inning is required when --target is reached_base.",
                    err=True,
                )
                sys.exit(2)
            if starter is None:
                click.echo(
                    "Error: --starter is required when --target is reached_base.",
                    err=True,
                )
                sys.exit(2)

            result = predict(
                team=team,
                opp=opp,
                inning=inning,
                starter=starter,
                as_of=as_of_date,
                config=cfg,
            )
            click.echo(result.to_json())

        else:
            # Phase 1.5: game_winner or runs
            result = predict_game(
                team=team,
                opp=opp,
                target=target,
                starter=starter,
                opp_starter=opp_starter,
                as_of=as_of_date,
                config=cfg,
            )

            # Output format depends on target
            if target == "game_winner":
                output = {
                    "target": "game_winner",
                    "team": team.upper(),
                    "opponent": opp.upper(),
                    "win_probability": result.probability,
                    "top_features": [
                        {"name": f.name, "contribution": f.contribution}
                        for f in result.top_features
                    ],
                }
            else:  # runs
                output = {
                    "target": "runs",
                    "team": team.upper(),
                    "opponent": opp.upper(),
                    "expected_runs": result.probability,
                    "top_features": [
                        {"name": f.name, "contribution": f.contribution}
                        for f in result.top_features
                    ],
                }
            click.echo(json.dumps(output))

    except InvalidInputError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)
    except MissingArtifactError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(3)
    except (FeatureSchemaMismatch, TargetMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(3)
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
