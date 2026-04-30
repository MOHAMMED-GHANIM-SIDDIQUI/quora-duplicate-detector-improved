"""Train a stronger duplicate-question classifier with the v2 feature set.

Example:
    python -m quora_duplicate_detector.training_v2 --data-path data/quora.csv

Expected columns:
    question1, question2, is_duplicate
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_probabilities, tune_threshold
from .features_v2 import FEATURE_SCHEMA_VERSION, build_feature_matrix_v2

TARGET_COLUMN = "is_duplicate"
QUESTION_COLUMNS = ["question1", "question2"]


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(QUESTION_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    cleaned = frame[QUESTION_COLUMNS + [TARGET_COLUMN]].copy()
    cleaned = cleaned.dropna(subset=QUESTION_COLUMNS + [TARGET_COLUMN])
    cleaned["question1"] = cleaned["question1"].astype(str)
    cleaned["question2"] = cleaned["question2"].astype(str)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    cleaned = cleaned[cleaned["question1"].str.strip().ne("")]
    cleaned = cleaned[cleaned["question2"].str.strip().ne("")]
    return cleaned.reset_index(drop=True)


def candidate_models(random_state: int) -> dict:
    models = {
        "regularized_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.05,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        pass

    return models


def train_and_select_model(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    random_state: int,
) -> tuple[str, object, float, pd.DataFrame]:
    rows = []
    fitted_models = {}

    for model_name, model in candidate_models(random_state).items():
        model.fit(train_features, train_labels)
        probabilities = model.predict_proba(validation_features)[:, 1]
        threshold, threshold_table = tune_threshold(validation_labels, probabilities)
        metrics = evaluate_probabilities(validation_labels, probabilities, threshold)
        metrics["model_name"] = model_name
        rows.append(metrics)
        fitted_models[model_name] = (model, threshold, threshold_table)

    leaderboard = pd.DataFrame(rows).sort_values("f1", ascending=False)
    best_model_name = leaderboard.iloc[0]["model_name"]
    best_model, best_threshold, _ = fitted_models[best_model_name]
    return best_model_name, best_model, best_threshold, leaderboard


def run_training(args: argparse.Namespace) -> None:
    data = clean_training_frame(pd.read_csv(args.data_path))

    train_df, test_df = train_test_split(
        data,
        test_size=args.test_size,
        stratify=data[TARGET_COLUMN],
        random_state=args.random_state,
    )
    train_df, validation_df = train_test_split(
        train_df,
        test_size=args.validation_size,
        stratify=train_df[TARGET_COLUMN],
        random_state=args.random_state,
    )

    embedder = SentenceTransformer(args.embedding_model)
    train_batch = build_feature_matrix_v2(train_df, embedder, batch_size=args.batch_size)
    validation_batch = build_feature_matrix_v2(validation_df, embedder, batch_size=args.batch_size)
    test_batch = build_feature_matrix_v2(test_df, embedder, batch_size=args.batch_size)

    best_model_name, best_model, best_threshold, leaderboard = train_and_select_model(
        train_batch.matrix,
        train_df[TARGET_COLUMN].values,
        validation_batch.matrix,
        validation_df[TARGET_COLUMN].values,
        args.random_state,
    )

    test_probabilities = best_model.predict_proba(test_batch.matrix)[:, 1]
    test_metrics = evaluate_probabilities(
        test_df[TARGET_COLUMN].values,
        test_probabilities,
        best_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, output_dir / "quora_duplicate_classifier_v2.joblib")
    leaderboard.to_csv(output_dir / "model_leaderboard_v2.csv", index=False)

    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "embedding_model_name": args.embedding_model,
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "feature_count": int(train_batch.matrix.shape[1]),
        "feature_names": train_batch.feature_names,
        "test_metrics": test_metrics,
        "row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
    }
    with (output_dir / "metadata_v2.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Best model:", best_model_name)
    print("Best threshold:", round(best_threshold, 3))
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print("Artifacts written to:", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 Quora duplicate detector.")
    parser.add_argument("--data-path", required=True, help="CSV with question1, question2, is_duplicate.")
    parser.add_argument("--output-dir", default="artifacts_v2", help="Directory for v2 artifacts.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())

