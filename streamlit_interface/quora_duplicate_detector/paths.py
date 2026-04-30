"""Path helpers for locating project artifacts from the Streamlit app."""

from pathlib import Path
from typing import Optional, Union

from .config import DEFAULT_ARTIFACT_DIR_NAME, METADATA_FILENAME, MODEL_FILENAME


def project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def default_artifact_dir() -> Path:
    return project_root() / DEFAULT_ARTIFACT_DIR_NAME


def artifact_paths(artifact_dir: Optional[Union[str, Path]] = None) -> tuple[Path, Path]:
    """Return classifier and metadata paths for an artifact directory."""
    base_dir = Path(artifact_dir).expanduser() if artifact_dir else default_artifact_dir()
    if not base_dir.is_absolute():
        base_dir = project_root() / base_dir

    return base_dir / MODEL_FILENAME, base_dir / METADATA_FILENAME
