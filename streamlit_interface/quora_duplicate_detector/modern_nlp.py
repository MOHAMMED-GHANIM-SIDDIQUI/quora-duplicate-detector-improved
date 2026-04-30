"""Modern transformer-based semantic similarity scoring.

This module complements the saved XGBoost artifact with two direct NLP scoring
strategies:

1. Bi-encoder sentence embeddings for fast semantic similarity.
2. Cross-encoder pair scoring for higher-accuracy duplicate-question decisions.
"""

from dataclasses import dataclass

import numpy as np

from .config import CROSS_ENCODER_THRESHOLD, SEMANTIC_SIMILARITY_THRESHOLD
from .text_features import jaccard_similarity, token_overlap_ratio


@dataclass(frozen=True)
class ModernSimilarityResult:
    question_1: str
    question_2: str
    semantic_similarity: float
    embedding_distance: float
    jaccard_similarity: float
    token_overlap_ratio: float
    cross_encoder_score: float | None
    final_score: float
    threshold: float
    label: str
    method: str


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _as_probability(score: float) -> float:
    """Normalize cross-encoder outputs that may be logits instead of probabilities."""
    if 0.0 <= score <= 1.0:
        return float(score)
    return _sigmoid(score)


def encode_normalized_pair(question_1: str, question_2: str, embedder) -> tuple[np.ndarray, np.ndarray]:
    embeddings = embedder.encode(
        [question_1, question_2],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings[0], embeddings[1]


def embedding_similarity(question_1: str, question_2: str, embedder) -> tuple[float, float]:
    q1_embedding, q2_embedding = encode_normalized_pair(question_1, question_2, embedder)
    cosine_similarity = float(np.dot(q1_embedding, q2_embedding))
    euclidean_distance = float(np.linalg.norm(q1_embedding - q2_embedding))
    return cosine_similarity, euclidean_distance


def cross_encoder_similarity(question_1: str, question_2: str, cross_encoder) -> float:
    raw_score = float(cross_encoder.predict([(question_1, question_2)])[0])
    return _as_probability(raw_score)


def compare_questions_modern(
    question_1: str,
    question_2: str,
    embedder,
    cross_encoder=None,
) -> ModernSimilarityResult:
    """Compare two questions with modern transformer similarity techniques."""
    semantic_score, embedding_distance = embedding_similarity(question_1, question_2, embedder)
    lexical_jaccard = jaccard_similarity(question_1, question_2)
    lexical_overlap = token_overlap_ratio(question_1, question_2)

    cross_score = None
    if cross_encoder is not None:
        cross_score = cross_encoder_similarity(question_1, question_2, cross_encoder)
        final_score = cross_score
        threshold = CROSS_ENCODER_THRESHOLD
        method = "Cross-Encoder Pair Classifier"
    else:
        final_score = semantic_score
        threshold = SEMANTIC_SIMILARITY_THRESHOLD
        method = "Bi-Encoder Sentence Embeddings"

    return ModernSimilarityResult(
        question_1=question_1,
        question_2=question_2,
        semantic_similarity=float(semantic_score),
        embedding_distance=float(embedding_distance),
        jaccard_similarity=float(lexical_jaccard),
        token_overlap_ratio=float(lexical_overlap),
        cross_encoder_score=cross_score,
        final_score=float(final_score),
        threshold=float(threshold),
        label="Duplicate" if final_score >= threshold else "Not Duplicate",
        method=method,
    )
