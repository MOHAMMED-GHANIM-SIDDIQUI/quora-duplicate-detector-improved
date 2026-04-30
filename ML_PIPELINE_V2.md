# Improved ML Pipeline

This project currently ships a trained v1 model that expects the original
777-feature layout. The new `features_v2.py` and `training_v2.py` files define a
cleaner retraining path without breaking the existing Streamlit app.

## What Changed

- Keeps separate sentence embeddings for `question1` and `question2`.
- Adds stronger symmetric lexical features:
  - absolute length differences
  - word and character length ratios
  - token containment in both directions
  - Dice similarity
  - exact normalized text match
- Adds embedding distance features:
  - cosine similarity
  - L2 distance
  - L1 distance
- Keeps high-signal embedding interactions:
  - absolute embedding difference
  - element-wise embedding product
- Benchmarks several model choices:
  - regularized Logistic Regression
  - HistGradientBoosting
  - Random Forest
  - XGBoost when installed
- Tunes the decision threshold on validation data instead of relying on `0.5`.

## Why This Is Better

The original pipeline is a good prototype, but the model relies heavily on a
single notebook and a duplicated inference implementation. The v2 pipeline makes
the feature schema explicit, saves feature names into metadata, and makes model
selection repeatable from a command.

## How To Train

```bash
cd streamlit_interface
python -m quora_duplicate_detector.training_v2 ^
  --data-path ..\data\quora.csv ^
  --output-dir ..\artifacts_v2
```

The input CSV must contain:

```text
question1
question2
is_duplicate
```

The output directory will contain:

```text
quora_duplicate_classifier_v2.joblib
metadata_v2.json
model_leaderboard_v2.csv
```

## Better Model Choices To Try Next

For highest pair-classification accuracy, fine-tune a cross-encoder transformer
that reads both questions together. For large-scale duplicate search, use a
two-stage system: bi-encoder retrieval followed by cross-encoder reranking.

