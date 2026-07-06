"""Per-league tuned hyperparameters, persisted as small JSON artifacts.

`models/hyper_{league}.json` holds walk-forward-validated hyperparameters
(chosen by scripts/tune_base.py: tuned on the earlier 75% of the league's own
history, gated on the untouched final-25% judge window). Production code reads
them through :func:`load_hyper` with the CURRENT hardcoded values as defaults,
so a missing/partial/corrupt file changes nothing and nightlies pick up new
values automatically.

Only keys present in ``defaults`` are honored (typo-proof), and ``null`` values
are ignored (explicit "keep the default").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sandy.config import Config, load_config

logger = logging.getLogger(__name__)

_cache: dict[Path, dict] = {}


def load_hyper(league: str, defaults: dict, cfg: Config | None = None) -> dict:
    """Return ``defaults`` overlaid with models/hyper_{league}.json (if any)."""
    cfg = cfg or load_config()
    path = Path(cfg.model.model_dir) / f"hyper_{league}.json"
    if path not in _cache:
        data: dict = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    logger.warning("hyper_%s.json is not an object — ignoring", league)
            except (ValueError, OSError) as e:
                logger.warning("hyper_%s.json unreadable (%s) — using defaults", league, e)
        _cache[path] = data
    out = dict(defaults)
    out.update({k: v for k, v in _cache[path].items() if k in defaults and v is not None})
    return out


def clear_cache() -> None:
    """Testing/tuning hook: force re-read of hyper files."""
    _cache.clear()


__all__ = ["load_hyper", "clear_cache"]
