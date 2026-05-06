"""CLI subcommand for prediction.

Task 12.6: sandy predict --team --opp --inning --starter [--as-of]

Requirements: 8.1–8.8, 9.2
"""
from __future__ import annotations

import sys
from datetime import date

import click

from sandy.cli.main import _require_config
from sandy.logging import configure_logging
from sandy.predict.predictor import (
    InvalidInputError,
    MissingArtifactError,
    predict,
)
from sandy.train.artifact import FeatureSchemaMismatch


@click.command()
@click.option("--team", required=True, help="Batting team code (e.g. SEA)")
@click.option("--opp", required=True, help="Opposing team code (e.g. LAD)")
@click.option("--inning", required=True, type=int, help="Target inning (1-9)")
@click.option("--starter", required=True, help="Opposing starting pitcher name")
@click.option("--as-of", type=str, default=None, help="Date ceiling YYYY-MM-DD (default: today)")
@click.pass_context
def predict_cmd(
    ctx: click.Context,
    team: str,
    opp: str,
    inning: int,
    starter: str,
    as_of: str | None,
) -> None:
    """Emit per-inning reach-base probability as JSON."""
    cfg = _require_config(ctx)
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
        result = predict(
            team=team,
            opp=opp,
            inning=inning,
            starter=starter,
            as_of=as_of_date,
            config=cfg,
        )
    except InvalidInputError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)
    except MissingArtifactError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(3)
    except FeatureSchemaMismatch as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Output JSON to stdout (requirement 8.1)
    click.echo(result.to_json())
