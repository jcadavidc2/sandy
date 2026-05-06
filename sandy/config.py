"""Configuration loading for Sandy.

Implements the :class:`Config` dataclass hierarchy and :func:`load_config`
with the precedence order: defaults < TOML < environment variables < CLI
overrides (requirement 9.4).

Required database connection values come from the ``MLB_DB_*`` environment
variables (requirement 9.1). TOML or explicit CLI overrides may also supply
them. If a required value is still missing after all layers merge,
:class:`MissingConfigError` is raised; the CLI layer (task 2.3) translates
that to exit code 4 (requirement 9.2).

``MLB_MODEL_PATH`` falls back to ``./models/latest.pkl`` when not supplied
by TOML, env, or CLI (requirement 9.3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    model_dir: Path = Path("./models")

    def artifact_path(self, target: str = "reached_base") -> Path:
        """Resolve the artifact path for a given target name.

        Returns model_dir / f"{target}.pkl".
        For backward compat, if MLB_MODEL_PATH was set and target is
        reached_base, self.path is used directly.
        """
        return self.model_dir / f"{target}.pkl"


@dataclass(frozen=True)
class IngestConfig:
    max_rps: float = 10.0
    max_retries: int = 5
    retry_base_delay_seconds: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    num_boost_round: int = 500
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class Config:
    database: DatabaseConfig
    model: ModelConfig
    ingest: IngestConfig = field(default_factory=IngestConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingConfigError(RuntimeError):
    """Raised when a required configuration value is not resolvable.

    The ``name`` attribute carries the ``MLB_DB_*`` env-var name the operator
    needs to export (or set via TOML/CLI). Requirement 9.2 mandates the
    operator-facing error reference the env var they'd normally set.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"Missing required environment variable: {name}")
        self.name = name


# Back-compat alias so existing callers importing ``MissingEnvVarError``
# (e.g. ``sandy/cli/main.py``) continue to work.
MissingEnvVarError = MissingConfigError


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------


# Checked in this order so the error message is deterministic when multiple
# variables are missing.
REQUIRED_DB_ENV_VARS: tuple[str, ...] = (
    "MLB_DB_HOST",
    "MLB_DB_PORT",
    "MLB_DB_NAME",
    "MLB_DB_USER",
    "MLB_DB_PASSWORD",
)

_DB_ENV_TO_FIELD: dict[str, str] = {
    "MLB_DB_HOST": "host",
    "MLB_DB_PORT": "port",
    "MLB_DB_NAME": "name",
    "MLB_DB_USER": "user",
    "MLB_DB_PASSWORD": "password",
}

_DEFAULT_MODEL_PATH = Path("./models/latest.pkl")

# Sections we understand; anything else in TOML is ignored (forward-compat).
_KNOWN_SECTIONS: tuple[str, ...] = (
    "database",
    "model",
    "ingest",
    "training",
    "logging",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomli.load(f)


def _apply_cli_overrides(
    working: dict[str, dict[str, Any]],
    overrides: dict[str, Any],
) -> None:
    """Merge a flat dotted-key mapping (e.g. ``{"logging.level": "DEBUG"}``).

    ``None`` values are skipped so CLI layers can unconditionally pass
    options whose flags weren't provided.
    """
    for dotted, value in overrides.items():
        if value is None:
            continue
        if "." not in dotted:
            raise ValueError(
                f"CLI override key must be 'section.field', got {dotted!r}"
            )
        section, field_name = dotted.split(".", 1)
        working.setdefault(section, {})[field_name] = value


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(
    *,
    toml_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """Resolve a :class:`Config` from defaults, TOML, env, and CLI overrides.

    Precedence (lowest first, requirement 9.4):

    1. Dataclass defaults (from the frozen dataclass definitions above).
    2. TOML at ``toml_path`` (or ``./sandy.toml`` in cwd if present).
    3. Environment variables (``MLB_DB_*``, ``MLB_MODEL_PATH``,
       ``MLB_LOG_LEVEL``).
    4. ``cli_overrides`` — flat dotted-key mapping, e.g.
       ``{"logging.level": "DEBUG"}``.

    Raises :class:`MissingConfigError` if any ``MLB_DB_*`` value is not
    supplied by TOML, env, or CLI after the merge (requirements 9.1, 9.2).
    The error's ``name`` attribute is the env-var name.

    Falls back to ``./models/latest.pkl`` for ``model.path`` if none of
    TOML/env/CLI provides it (requirement 9.3).
    """
    working: dict[str, dict[str, Any]] = {
        section: {} for section in _KNOWN_SECTIONS
    }

    # ---- Layer 2: TOML ----
    if toml_path is None:
        candidate = Path("./sandy.toml")
        if candidate.is_file():
            toml_path = candidate
    if toml_path is not None:
        toml_data = _load_toml(Path(toml_path))
        for section, fields in toml_data.items():
            if section not in working or not isinstance(fields, dict):
                # Unknown sections are silently ignored for forward-compat.
                continue
            working[section].update(fields)

    # ---- Layer 3: environment variables ----
    for env_name, field_name in _DB_ENV_TO_FIELD.items():
        value = os.environ.get(env_name)
        if value:  # treats both unset and empty string as "not supplied"
            working["database"][field_name] = value

    model_env = os.environ.get("MLB_MODEL_PATH")
    if model_env:
        working["model"]["path"] = model_env

    model_dir_env = os.environ.get("MLB_MODEL_DIR")
    if model_dir_env:
        working["model"]["model_dir"] = model_dir_env

    log_env = os.environ.get("MLB_LOG_LEVEL")
    if log_env:
        working["logging"]["level"] = log_env

    # ---- Layer 4: explicit CLI overrides (highest precedence) ----
    if cli_overrides:
        _apply_cli_overrides(working, cli_overrides)

    # ---- Validation: required DB values must be present ----
    db_working = working["database"]
    for env_name in REQUIRED_DB_ENV_VARS:
        field_name = _DB_ENV_TO_FIELD[env_name]
        raw = db_working.get(field_name)
        if raw is None or (isinstance(raw, str) and raw == ""):
            raise MissingConfigError(env_name)

    # ---- Fallback: model path (requirement 9.3) ----
    if "path" not in working["model"] or working["model"]["path"] in (None, ""):
        working["model"]["path"] = _DEFAULT_MODEL_PATH

    # ---- Coerce and construct typed sections ----
    database = DatabaseConfig(
        host=str(db_working["host"]),
        port=int(db_working["port"]),
        name=str(db_working["name"]),
        user=str(db_working["user"]),
        password=str(db_working["password"]),
    )
    model = ModelConfig(
        path=Path(working["model"]["path"]),
        model_dir=Path(working["model"].get("model_dir", "./models")),
    )
    ingest = IngestConfig(
        **{
            k: v
            for k, v in working["ingest"].items()
            if k in {"max_rps", "max_retries", "retry_base_delay_seconds"}
        }
    )
    training = TrainingConfig(
        **{
            k: v
            for k, v in working["training"].items()
            if k in {"seed", "num_boost_round", "early_stopping_rounds"}
        }
    )
    logging_cfg = LoggingConfig(
        **{k: v for k, v in working["logging"].items() if k in {"level"}}
    )

    return Config(
        database=database,
        model=model,
        ingest=ingest,
        training=training,
        logging=logging_cfg,
    )


__all__ = [
    "Config",
    "DatabaseConfig",
    "IngestConfig",
    "LoggingConfig",
    "MissingConfigError",
    "MissingEnvVarError",  # back-compat alias
    "ModelConfig",
    "REQUIRED_DB_ENV_VARS",
    "TrainingConfig",
    "load_config",
]
