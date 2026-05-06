"""Shared CLI helpers — avoids circular imports between main.py and subcommands."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from sandy.config import Config, MissingEnvVarError, load_config


def require_config(ctx: click.Context) -> Config:
    """Resolve a Config for a subcommand, translating config errors.

    On MissingEnvVarError we write a clear message to stderr and exit with
    status 4 (requirement 9.2).
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
