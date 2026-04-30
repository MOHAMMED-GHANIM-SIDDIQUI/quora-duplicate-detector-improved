"""Evaluation and threshold tuning utilities for duplicate detection."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_probabilities(y_true, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def tune_threshold(y_true, probabilities: np.ndarray, metric: str = "f1") -> tuple[float, pd.DataFrame]:
    rows = [
        evaluate_probabilities(y_true, probabilities, threshold)
        for threshold in np.arange(0.10, 0.91, 0.01)
    ]
    threshold_table = pd.DataFrame(rows)
    best_row = threshold_table.sort_values(metric, ascending=False).iloc[0]
    return float(best_row["threshold"]), threshold_table

