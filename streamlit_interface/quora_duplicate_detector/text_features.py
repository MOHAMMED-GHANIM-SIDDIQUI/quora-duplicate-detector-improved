"""Text normalization and lexical similarity features."""

import re


def normalize_text(text: object) -> str:
    """Normalize text for token-overlap features."""
    normalized = str(text).lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def token_set(text: object) -> set[str]:
    return set(normalize_text(text).split())


def word_count(text: object) -> int:
    return len(str(text).split())


def char_count(text: object) -> int:
    return len(str(text))


def jaccard_similarity(question_1: object, question_2: object) -> float:
    tokens_1 = token_set(question_1)
    tokens_2 = token_set(question_2)
    union = tokens_1.union(tokens_2)

    if not union:
        return 0.0

    return len(tokens_1.intersection(tokens_2)) / len(union)


def token_overlap_ratio(question_1: object, question_2: object) -> float:
    tokens_1 = token_set(question_1)
    tokens_2 = token_set(question_2)
    denominator = min(len(tokens_1), len(tokens_2))

    if denominator == 0:
        return 0.0

    return len(tokens_1.intersection(tokens_2)) / denominator

