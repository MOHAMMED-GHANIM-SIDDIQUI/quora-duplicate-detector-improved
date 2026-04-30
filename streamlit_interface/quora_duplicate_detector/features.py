"""Feature construction shared by single and batch predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .text_features import (
    char_count,
    jaccard_similarity,
    token_overlap_ratio,
    word_count,
)


@dataclass(frozen=True)
class FeatureBatch:
    """Model-ready features plus display-friendly intermediate values."""

    matrix: np.ndarray
    lexical_features: pd.DataFrame
    cosine_similarity: np.ndarray


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
    return np.sum(left_norm * right_norm, axis=1)


def build_lexical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the eight lexical features used by the trained model."""
    question_1 = frame["question1"]
    question_2 = frame["question2"]

    q1_word_counts = question_1.apply(word_count).values
    q2_word_counts = question_2.apply(word_count).values
    q1_char_counts = question_1.apply(char_count).values
    q2_char_counts = question_2.apply(char_count).values

    return pd.DataFrame(
        {
            "q1_word_count": q1_word_counts,
            "q2_word_count": q2_word_counts,
            "q1_char_count": q1_char_counts,
            "q2_char_count": q2_char_counts,
            "word_count_diff": (q1_word_counts - q2_word_counts).astype(float),
            "char_count_diff": (q1_char_counts - q2_char_counts).astype(float),
            "jaccard_similarity": [
                jaccard_similarity(left, right)
                for left, right in zip(question_1, question_2)
            ],
            "token_overlap_ratio": [
                token_overlap_ratio(left, right)
                for left, right in zip(question_1, question_2)
            ],
        }
    )


def build_features_from_embeddings(
    frame: pd.DataFrame,
    question_1_embeddings: np.ndarray,
    question_2_embeddings: np.ndarray,
) -> FeatureBatch:
    """Combine semantic, lexical, and embedding-interaction features.

    Feature order must match the order used when the saved classifier was trained:
    cosine, lexical features, absolute embedding difference, element-wise product.
    """
    cosine_values = cosine_similarity_matrix(
        question_1_embeddings,
        question_2_embeddings,
    )
    lexical_features = build_lexical_features(frame)
    absolute_difference = np.abs(question_1_embeddings - question_2_embeddings)
    elementwise_product = question_1_embeddings * question_2_embeddings

    matrix = np.hstack(
        [
            cosine_values.reshape(-1, 1),
            lexical_features.values.astype(float),
            absolute_difference,
            elementwise_product,
        ]
    )

    return FeatureBatch(
        matrix=matrix,
        lexical_features=lexical_features,
        cosine_similarity=cosine_values,
    )


def build_feature_batch(frame: pd.DataFrame, embedder, batch_size: int = 128) -> FeatureBatch:
    """Encode question pairs and return model-ready features."""
    working_frame = frame.copy()
    working_frame["question1"] = working_frame["question1"].astype(str)
    working_frame["question2"] = working_frame["question2"].astype(str)

    question_1_embeddings = embedder.encode(
        working_frame["question1"].tolist(),
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    question_2_embeddings = embedder.encode(
        working_frame["question2"].tolist(),
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    return build_features_from_embeddings(
        working_frame,
        question_1_embeddings,
        question_2_embeddings,
    )


def build_single_pair_features(question_1: str, question_2: str, embedder) -> FeatureBatch:
    frame = pd.DataFrame({"question1": [question_1], "question2": [question_2]})
    return build_feature_batch(frame, embedder)

