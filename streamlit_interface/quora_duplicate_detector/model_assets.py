"""Load model artifacts and metadata."""

import json
from pathlib import Path
from typing import Union

import joblib

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD


def load_metadata(metadata_path: Union[str, Path]) -> dict:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_classifier(model_path: Union[str, Path]):
    return joblib.load(model_path)


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def metadata_value(metadata: dict, key: str, default):
    return metadata.get(key, default)


def model_settings(metadata: dict) -> dict:
    """Normalize optional metadata fields used by the UI."""
    return {
        "embedding_model_name": metadata_value(
            metadata,
            "embedding_model_name",
            DEFAULT_EMBEDDING_MODEL,
        ),
        "best_threshold": float(metadata_value(metadata, "best_threshold", DEFAULT_THRESHOLD)),
        "best_model_name": metadata_value(metadata, "best_model_name", DEFAULT_MODEL_NAME),
        "test_metrics": metadata_value(metadata, "test_metrics", {}),
        "feature_summary": metadata_value(metadata, "feature_summary", {}),
    }
