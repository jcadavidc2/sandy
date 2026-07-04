"""Sandy CLI entry point.

Phase 1, task 2.3: adds the top-level Click group options (``--config``,
``--log-level``, ``--version``) and a lazy ``_require_config`` helper that
subcommands use to materialize a resolved :class:`~sandy.config.Config`. The
helper catches :class:`~sandy.config.MissingEnvVarError` and exits with
status 4 naming the missing variable (requirement 9.2). No real subcommands
are wired yet; they arrive in tasks 12.2–12.6.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from sandy.config import Config, MissingEnvVarError, load_config

__version__ = "0.1.0"


@click.group()
@click.version_option(__version__, prog_name="sandy")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to a TOML config file (overridden by env vars and CLI flags).",
)
@click.option(
    "--log-level",
    type=str,
    default=None,
    help="Log level: DEBUG, INFO, WARN, ERROR. Overrides MLB_LOG_LEVEL.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, log_level: str | None) -> None:
    """sandy — MLB per-inning reach-base predictor."""
    ctx.ensure_object(dict)
    ctx.obj["_config_path"] = config_path
    ctx.obj["_log_level"] = log_level


def _require_config(ctx: click.Context) -> Config:
    """Resolve a :class:`Config` for a subcommand, translating config errors.

    On :class:`MissingEnvVarError` we write a clear message to stderr and
    exit with status 4 (requirement 9.2). ``--help`` and ``--version`` never
    call this, so introspection commands keep working even with no env set.
    """
    overrides: dict[str, Any] = {}
    log_level = ctx.obj.get("_log_level") if ctx.obj else None
    if log_level:
        overrides["logging.level"] = log_level

    config_path_str = ctx.obj.get("_config_path") if ctx.obj else None
    toml_path = Path(config_path_str) if config_path_str else None

    try:
        return load_config(toml_path=toml_path, cli_overrides=overrides or None)
    except MissingEnvVarError as exc:
        click.echo(
            f"Error: missing required environment variable: {exc.name}",
            err=True,
        )
        sys.exit(4)


# ---------------------------------------------------------------------------
# Register subcommands (imported here to avoid circular imports)
# ---------------------------------------------------------------------------

from sandy.cli.ingest_cmd import ingest  # noqa: E402
from sandy.cli.labels_cmd import labels  # noqa: E402
from sandy.cli.features_cmd import features  # noqa: E402
from sandy.cli.train_cmd import train  # noqa: E402
from sandy.cli.predict_cmd import predict_cmd  # noqa: E402
from sandy.cli.today_cmd import today  # noqa: E402
from sandy.cli.predict_all_cmd import predict_all  # noqa: E402
from sandy.cli.refresh_cmd import refresh  # noqa: E402
from sandy.cli.over_under_cmd import over_under  # noqa: E402
from sandy.cli.football_cmd import football
from sandy.cli.mls_cmd import mls
from sandy.cli.nhl_cmd import nhl  # noqa: E402

cli.add_command(ingest)
cli.add_command(labels)
cli.add_command(features)
cli.add_command(train)
cli.add_command(predict_cmd, name="predict")
cli.add_command(today)
cli.add_command(predict_all)
cli.add_command(refresh)
cli.add_command(over_under)
cli.add_command(football)
cli.add_command(mls)
cli.add_command(nhl)


if __name__ == "__main__":  # pragma: no cover
    cli()
