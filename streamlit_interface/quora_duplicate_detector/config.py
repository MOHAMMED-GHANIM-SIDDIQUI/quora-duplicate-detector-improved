"""Shared configuration defaults for training artifacts and inference."""

MODEL_FILENAME = "quora_duplicate_classifier.joblib"
METADATA_FILENAME = "metadata.json"

DEFAULT_ARTIFACT_DIR_NAME = "artifacts"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.50
DEFAULT_MODEL_NAME = "Unknown"

REQUIRED_BATCH_COLUMNS = {"question1", "question2"}

# Modern NLP comparison defaults. These are independent from the saved v1
# XGBoost artifact, so they can be used without retraining the old classifier.
MODERN_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/quora-distilroberta-base"
SEMANTIC_SIMILARITY_THRESHOLD = 0.75
CROSS_ENCODER_THRESHOLD = 0.50
