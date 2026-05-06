"""Model artifact serialization for Sandy.

Task 10.3: save_artifact() and load_artifact() for ModelArtifact.

Serialization uses LightGBM's model_to_string() (text format) rather than
pickling the Booster directly. This makes the round-trip property testable:
the text form is deterministic and lgb.Booster(model_str=...) reconstructs
a numerically identical model (requirement 7.2).

The artifact is written atomically via a .tmp file + os.replace() so a
crash mid-write never leaves a corrupt artifact (requirement 7.1).

FeatureSchemaMismatch is raised on load if the stored feature_schema_version
differs from the current code version (requirement 7.3).

Requirements: 6.5, 7.1, 7.2, 7.3
"""
from __future__ import annotations

import pickle
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb

from sandy.features.schema import FEATURE_SCHEMA_VERSION
from sandy.schemas import ModelArtifact


class FeatureSchemaMismatch(Exception):
    """Raised when a loaded artifact's feature schema version doesn't match.

    The CLI layer catches this and exits with a non-zero status, naming both
    versions so the operator knows to retrain (requirement 7.3).
    """

    def __init__(self, loaded: int, current: int) -> None:
        super().__init__(
            f"Model artifact has feature_schema_version={loaded} but "
            f"current code expects version={current}. "
            f"Please retrain the model with 'sandy train'."
        )
        self.loaded = loaded
        self.current = current


def save_artifact(artifact: ModelArtifact, path: Path) -> None:
    """Serialize a ModelArtifact to *path* atomically.

    Uses LightGBM's model_to_string() for the model itself so the artifact
    is portable across Python versions and pickle protocols (requirement 7.1).

    Write is atomic: data goes to path.with_suffix('.pkl.tmp') first, then
    os.replace() swaps it in — a crash mid-write leaves the old file intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": artifact.model.model_to_string(),
        "feature_names": artifact.feature_names,
        "feature_schema_version": artifact.feature_schema_version,
        "training_window_start": artifact.training_window_start.isoformat(),
        "training_window_end": artifact.training_window_end.isoformat(),
        "created_at": artifact.created_at.isoformat(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp.replace(path)  # atomic on POSIX; near-atomic on Windows


def load_artifact(path: Path) -> ModelArtifact:
    """Load a ModelArtifact from *path*.

    Raises
    ------
    FileNotFoundError:    if *path* does not exist.
    FeatureSchemaMismatch: if the stored feature_schema_version differs from
                           the current FEATURE_SCHEMA_VERSION (requirement 7.3).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            f"Run 'sandy train' to create one."
        )

    with path.open("rb") as f:
        payload = pickle.load(f)

    stored_version = payload["feature_schema_version"]
    if stored_version != FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaMismatch(
            loaded=stored_version,
            current=FEATURE_SCHEMA_VERSION,
        )

    booster = lgb.Booster(model_str=payload["model"])

    return ModelArtifact(
        model=booster,
        feature_names=payload["feature_names"],
        feature_schema_version=stored_version,
        training_window_start=date.fromisoformat(payload["training_window_start"]),
        training_window_end=date.fromisoformat(payload["training_window_end"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


__all__ = ["FeatureSchemaMismatch", "load_artifact", "save_artifact"]
