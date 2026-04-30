"""Prediction helpers for single-row and batch inference."""

import numpy as np
import pandas as pd

from .features import FeatureBatch, build_feature_batch, build_single_pair_features


def probability_band(probability: float) -> str:
    if probability >= 0.85:
        return "Very High"
    if probability >= 0.70:
        return "High"
    if probability >= 0.50:
        return "Moderate"
    if probability >= 0.30:
        return "Low"
    return "Very Low"


def predict_from_features(feature_batch: FeatureBatch, classifier, threshold: float) -> np.ndarray:
    probabilities = classifier.predict_proba(feature_batch.matrix)[:, 1]
    return (probabilities >= threshold).astype(int)


def predict_single_pair(
    question_1: str,
    question_2: str,
    embedder,
    classifier,
    threshold: float,
) -> dict:
    feature_batch = build_single_pair_features(question_1, question_2, embedder)
    probability = float(classifier.predict_proba(feature_batch.matrix)[:, 1][0])
    prediction = int(probability >= threshold)
    lexical = feature_batch.lexical_features.iloc[0]

    return {
        "question1": question_1,
        "question2": question_2,
        "duplicate_probability": probability,
        "prediction": prediction,
        "label": "Duplicate" if prediction == 1 else "Not Duplicate",
        "cosine_similarity": float(feature_batch.cosine_similarity[0]),
        "jaccard_similarity": float(lexical["jaccard_similarity"]),
        "token_overlap_ratio": float(lexical["token_overlap_ratio"]),
        "q1_word_count": int(lexical["q1_word_count"]),
        "q2_word_count": int(lexical["q2_word_count"]),
    }


def predict_batch(frame: pd.DataFrame, embedder, classifier, threshold: float) -> pd.DataFrame:
    feature_batch = build_feature_batch(frame, embedder)
    probabilities = classifier.predict_proba(feature_batch.matrix)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = frame.copy()
    result["duplicate_probability"] = probabilities
    result["prediction"] = predictions
    result["label"] = np.where(predictions == 1, "Duplicate", "Not Duplicate")
    result["cosine_similarity"] = feature_batch.cosine_similarity
    result["jaccard_similarity"] = feature_batch.lexical_features["jaccard_similarity"].values
    result["token_overlap_ratio"] = feature_batch.lexical_features["token_overlap_ratio"].values
    return result

