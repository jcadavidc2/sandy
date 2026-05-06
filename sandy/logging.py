"""Structured JSON logging for Sandy.

Emits one JSON object per line on stdout with the required fields
``timestamp`` (ISO-8601 UTC with ``Z`` suffix), ``level``, ``component``, and
``message``; any keyword arguments passed via ``extra=...`` are flattened onto
the top-level object (requirement 10.1).

``configure_logging`` is idempotent: calling it repeatedly does not stack
handlers. Level resolution accepts either the `MLB_LOG_LEVEL` env var or an
explicit argument, and unknown levels quietly fall back to ``INFO`` rather
than crashing the program (requirement 10.3).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Keys already covered by dedicated fields in the JSON output. Anything else
# on the LogRecord is treated as "extra" and flattened onto the output dict.
_STANDARD_LOGRECORD_ATTRS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "component",
        "message",
        # logging internals / derived:
        "asctime",
    }
)

_VALID_LEVEL_NAMES: frozenset[str] = frozenset({"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"})

_HANDLER_FLAG = "_sandy_json_handler"


class JsonFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a single-line JSON object.

    Standard keys: ``timestamp`` (UTC ISO-8601, ``Z`` suffix), ``level``,
    ``component``, ``message``. Any additional attributes set on the record
    via ``logger.info(..., extra={...})`` are merged at the top level.
    """

    def format(self, record: logging.LogRecord) -> str:
        component = getattr(record, "component", None) or record.name.rsplit(".", 1)[-1]
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "component": component,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOGRECORD_ATTRS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a value to something ``json.dumps`` will accept."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _resolve_level(level: str | None) -> int:
    """Resolve a level name to a numeric level, falling back to INFO.

    Precedence: explicit ``level`` arg > ``MLB_LOG_LEVEL`` env var > ``INFO``.
    Unknown names fall back to INFO rather than raising (requirement 10.3).
    """
    candidate = level or os.environ.get("MLB_LOG_LEVEL") or "INFO"
    name = candidate.upper()
    if name not in _VALID_LEVEL_NAMES:
        return logging.INFO
    if name == "WARN":
        return logging.WARNING
    return getattr(logging, name, logging.INFO)


def configure_logging(level: str = "INFO") -> None:
    """Attach a JSON :class:`StreamHandler` on stdout to the root logger.

    Idempotent: subsequent calls adjust the level but do not add duplicate
    handlers.
    """
    root = logging.getLogger()
    resolved = _resolve_level(level)
    root.setLevel(resolved)

    for handler in root.handlers:
        if getattr(handler, _HANDLER_FLAG, False):
            handler.setLevel(resolved)
            return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(resolved)
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)


def get_logger(component: str) -> logging.Logger:
    """Return a logger whose name is ``component``.

    The root handler installed by :func:`configure_logging` formats every
    record; callers can still pass ``extra={"component": "..."}`` on a single
    call site to override the default, which is the last dotted segment of
    the logger name.
    """
    return logging.getLogger(component)


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
