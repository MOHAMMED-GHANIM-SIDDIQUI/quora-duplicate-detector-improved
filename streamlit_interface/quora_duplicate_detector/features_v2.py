"""Improved feature engineering for retraining a stronger duplicate detector.

The deployed artifact still expects the original 777-feature layout from
features.py. This module is for a new model version and should be used only
after retraining and saving a matching artifact.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import cosine_similarity_matrix
from .text_features import char_count, normalize_text, token_set, word_count


FEATURE_SCHEMA_VERSION = "pairwise_embedding_v2"


@dataclass(frozen=True)
class FeatureMatrixV2:
    matrix: np.ndarray
    feature_names: list[str]
    diagnostics: pd.DataFrame


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.clip(denominator, 1e-12, None)


def _token_stats(question_1: object, question_2: object) -> dict[str, float]:
    tokens_1 = token_set(question_1)
    tokens_2 = token_set(question_2)
    intersection = tokens_1.intersection(tokens_2)
    union = tokens_1.union(tokens_2)

    intersection_count = len(intersection)
    q1_unique = len(tokens_1)
    q2_unique = len(tokens_2)
    union_count = len(union)

    return {
        "common_token_count": float(intersection_count),
        "q1_unique_token_count": float(q1_unique),
        "q2_unique_token_count": float(q2_unique),
        "jaccard_similarity": float(intersection_count / union_count) if union_count else 0.0,
        "overlap_min_ratio": (
            float(intersection_count / min(q1_unique, q2_unique))
            if min(q1_unique, q2_unique)
            else 0.0
        ),
        "overlap_max_ratio": (
            float(intersection_count / max(q1_unique, q2_unique))
            if max(q1_unique, q2_unique)
            else 0.0
        ),
        "dice_similarity": (
            float((2 * intersection_count) / (q1_unique + q2_unique))
            if (q1_unique + q2_unique)
            else 0.0
        ),
        "q1_contained_in_q2": float(intersection_count / q1_unique) if q1_unique else 0.0,
        "q2_contained_in_q1": float(intersection_count / q2_unique) if q2_unique else 0.0,
    }


def build_lexical_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """Build richer symmetric lexical features for a retrained model."""
    question_1 = frame["question1"].astype(str)
    question_2 = frame["question2"].astype(str)

    q1_word_counts = question_1.apply(word_count).astype(float).values
    q2_word_counts = question_2.apply(word_count).astype(float).values
    q1_char_counts = question_1.apply(char_count).astype(float).values
    q2_char_counts = question_2.apply(char_count).astype(float).values

    token_rows = [
        _token_stats(left, right)
        for left, right in zip(question_1, question_2)
    ]
    token_features = pd.DataFrame(token_rows)

    lexical = pd.DataFrame(
        {
            "q1_word_count": q1_word_counts,
            "q2_word_count": q2_word_counts,
            "q1_char_count": q1_char_counts,
            "q2_char_count": q2_char_counts,
            "abs_word_count_diff": np.abs(q1_word_counts - q2_word_counts),
            "abs_char_count_diff": np.abs(q1_char_counts - q2_char_counts),
            "word_count_ratio": _safe_divide(
                np.minimum(q1_word_counts, q2_word_counts),
                np.maximum(q1_word_counts, q2_word_counts),
            ),
            "char_count_ratio": _safe_divide(
                np.minimum(q1_char_counts, q2_char_counts),
                np.maximum(q1_char_counts, q2_char_counts),
            ),
            "q1_avg_word_length": _safe_divide(q1_char_counts, q1_word_counts),
            "q2_avg_word_length": _safe_divide(q2_char_counts, q2_word_counts),
            "normalized_exact_match": [
                float(normalize_text(left) == normalize_text(right))
                for left, right in zip(question_1, question_2)
            ],
        }
    )

    return pd.concat([lexical, token_features], axis=1)


def build_features_v2_from_embeddings(
    frame: pd.DataFrame,
    question_1_embeddings: np.ndarray,
    question_2_embeddings: np.ndarray,
) -> FeatureMatrixV2:
    """Create a stronger feature matrix for a new model version.

    Additions over v1:
    - symmetric absolute length differences instead of signed-only differences
    - length ratios, containment, Dice similarity, and exact normalized match
    - L2 and L1 embedding distances alongside cosine similarity
    """
    lexical = build_lexical_features_v2(frame)
    embedding_difference = question_1_embeddings - question_2_embeddings
    absolute_difference = np.abs(embedding_difference)
    elementwise_product = question_1_embeddings * question_2_embeddings

    cosine_values = cosine_similarity_matrix(question_1_embeddings, question_2_embeddings)
    l2_distance = np.linalg.norm(embedding_difference, axis=1)
    l1_distance = np.sum(absolute_difference, axis=1)

    scalar_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding_cosine_similarity": cosine_values,
                    "embedding_l2_distance": l2_distance,
                    "embedding_l1_distance": l1_distance,
                }
            ),
            lexical,
        ],
        axis=1,
    )

    matrix = np.hstack(
        [
            scalar_features.values.astype(float),
            absolute_difference,
            elementwise_product,
        ]
    )

    embedding_dim = question_1_embeddings.shape[1]
    feature_names = list(scalar_features.columns)
    feature_names.extend(f"embedding_abs_diff_{index}" for index in range(embedding_dim))
    feature_names.extend(f"embedding_product_{index}" for index in range(embedding_dim))

    return FeatureMatrixV2(
        matrix=matrix,
        feature_names=feature_names,
        diagnostics=scalar_features,
    )


def build_feature_matrix_v2(frame: pd.DataFrame, embedder, batch_size: int = 128) -> FeatureMatrixV2:
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

    return build_features_v2_from_embeddings(
        working_frame,
        question_1_embeddings,
        question_2_embeddings,
    )

