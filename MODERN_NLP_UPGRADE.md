# Modern NLP Upgrade

This upgrade adds a transformer-first comparison path alongside the original
XGBoost artifact.

## Added Techniques

### Bi-Encoder Sentence Embeddings

The app can now compare questions with `BAAI/bge-small-en-v1.5`, a modern
sentence embedding model. It encodes each question separately, normalizes the
vectors, and uses cosine similarity via dot product.

This is fast and works well when you need to compare many questions.

### Cross-Encoder Pair Scoring

The app can optionally use `cross-encoder/quora-distilroberta-base`. A
cross-encoder reads both questions together and directly predicts whether the
pair is a duplicate.

This is usually more accurate for pair classification than comparing two
separate vectors, because the transformer can attend across both questions.

## How This Differs From The Original Model

Original path:

```text
question1 + question2
-> sentence embeddings
-> handcrafted pair features
-> XGBoost
-> thresholded prediction
```

Modern NLP path:

```text
question1 + question2
-> modern embedding similarity
-> optional cross-encoder duplicate score
-> thresholded prediction
```

The original model remains available in the Single Prediction and Batch
Prediction tabs. The Modern NLP tab is independent and does not require
retraining the old XGBoost artifact.

## When To Use Each

- Use the original model when you want the exact saved artifact behavior.
- Use embedding similarity when you need fast semantic comparison.
- Use the cross-encoder when you want stronger duplicate-question judgment for
  one pair at a time.

For a production rebuild, the strongest design is often a two-stage system:

```text
bi-encoder embeddings retrieve candidate duplicates
-> cross-encoder reranks the candidates
-> final duplicate decision
```

