"""Sandy CLI entry point.

Phase 1, task 1.1 stub: declares the `sandy` console script wired through
`pyproject.toml`. Real subcommands (predict, ingest, train, labels, features)
are implemented in later tasks.
"""
from __future__ import annotations

import click

__version__ = "0.1.0"


@click.group()
@click.version_option(__version__, prog_name="sandy")
def cli() -> None:
    """sandy — MLB per-inning reach-base predictor."""


if __name__ == "__main__":  # pragma: no cover
    cli()
