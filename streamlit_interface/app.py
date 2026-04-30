from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from quora_duplicate_detector.config import (
    CROSS_ENCODER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
    MODERN_EMBEDDING_MODEL,
    REQUIRED_BATCH_COLUMNS,
)
from quora_duplicate_detector.inference import (
    predict_batch,
    predict_single_pair,
    probability_band,
)
from quora_duplicate_detector.modern_nlp import compare_questions_modern
from quora_duplicate_detector.model_assets import (
    load_classifier,
    load_cross_encoder,
    load_embedder,
    load_metadata,
    model_settings,
)
from quora_duplicate_detector.paths import artifact_paths, default_artifact_dir
from quora_duplicate_detector.text_features import normalize_text


MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 1000
MAX_BATCH_ROWS = 5000

EXAMPLES = {
    "Python duplicate": (
        "How can I learn Python quickly?",
        "What is the fastest way to learn Python?",
    ),
    "Interview duplicate": (
        "How do I prepare for a data science interview?",
        "What is the best way to get ready for data science interviews?",
    ),
    "Not duplicate": (
        "How do I learn machine learning?",
        "What are the best tourist places in Nepal?",
    ),
    "Borderline": (
        "How can I improve my English speaking skills?",
        "How can I improve my public speaking skills?",
    ),
}


st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #155e75 55%, #14532d 100%);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 1.5rem 1.7rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }

    .hero h1 {
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
    }

    .hero p {
        color: #dbeafe;
        margin: 0;
        font-size: 1rem;
        max-width: 780px;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.22);
        padding: 0.9rem 1rem;
        border-radius: 8px;
        min-height: 90px;
    }

    .metric-title {
        color: #bae6fd;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: white;
        font-size: 1.35rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .result-box {
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin: 0.75rem 0 1rem 0;
    }

    .result-success {
        background: rgba(16, 185, 129, 0.13);
        border: 1px solid rgba(16, 185, 129, 0.45);
    }

    .result-warning {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid rgba(245, 158, 11, 0.45);
    }

    .result-title {
        font-size: 1.2rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .result-note {
        color: #cbd5e1;
        font-size: 0.92rem;
    }

    .section-note {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def cached_metadata(metadata_path: str) -> dict:
    return load_metadata(metadata_path)


@st.cache_resource
def cached_classifier(model_path: str):
    return load_classifier(model_path)


@st.cache_resource
def cached_embedder(model_name: str):
    return load_embedder(model_name)


@st.cache_resource
def cached_cross_encoder(model_name: str):
    return load_cross_encoder(model_name)


def load_runtime_assets() -> tuple[dict, object, Path, Path]:
    model_path, metadata_path = artifact_paths(default_artifact_dir())

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = cached_metadata(str(metadata_path))
    settings = model_settings(metadata)
    classifier = cached_classifier(str(model_path))
    return settings, classifier, model_path, metadata_path


def render_metric_card(title: str, value: str, font_size: str = "1.35rem") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="font-size:{font_size};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision(label: str, score: float, threshold: float, note: str) -> None:
    is_duplicate = label == "Duplicate"
    css_class = "result-success" if is_duplicate else "result-warning"
    st.markdown(
        f"""
        <div class="result-box {css_class}">
            <div class="result-title">{label}</div>
            <div class="result-note">
                Score: <b>{score:.3f}</b> | Threshold: <b>{threshold:.3f}</b><br>
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(max(0.0, min(1.0, float(score))))


def validate_question_pair(question_1: str, question_2: str) -> list[str]:
    errors = []
    q1 = question_1.strip()
    q2 = question_2.strip()

    if not q1 or not q2:
        errors.append("Both question fields are required.")
    if q1 and len(q1) < MIN_QUESTION_CHARS:
        errors.append(f"Question 1 must be at least {MIN_QUESTION_CHARS} characters.")
    if q2 and len(q2) < MIN_QUESTION_CHARS:
        errors.append(f"Question 2 must be at least {MIN_QUESTION_CHARS} characters.")
    if len(q1) > MAX_QUESTION_CHARS:
        errors.append(f"Question 1 must be {MAX_QUESTION_CHARS} characters or fewer.")
    if len(q2) > MAX_QUESTION_CHARS:
        errors.append(f"Question 2 must be {MAX_QUESTION_CHARS} characters or fewer.")
    if q1 and q2 and normalize_text(q1) == normalize_text(q2):
        errors.append("The two questions are exactly the same after normalization.")

    return errors


def show_validation_errors(errors: list[str]) -> bool:
    if not errors:
        return False

    for error in errors:
        st.error(error)
    return True


def validate_batch_frame(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing_columns = REQUIRED_BATCH_COLUMNS.difference(frame.columns)

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(sorted(missing_columns))}.")
        return errors
    if frame.empty:
        errors.append("CSV contains no rows to score.")
        return errors
    if len(frame) > MAX_BATCH_ROWS:
        errors.append(f"CSV has {len(frame):,} rows. Limit uploads to {MAX_BATCH_ROWS:,} rows.")

    blank_rows = (
        frame["question1"].isna()
        | frame["question2"].isna()
        | frame["question1"].astype(str).str.strip().eq("")
        | frame["question2"].astype(str).str.strip().eq("")
    )
    if blank_rows.any():
        errors.append(f"CSV contains {int(blank_rows.sum()):,} rows with blank questions.")

    return errors


def sample_csv_bytes() -> bytes:
    sample = pd.DataFrame(
        [
            {
                "question1": "How can I learn Python quickly?",
                "question2": "What is the fastest way to learn Python?",
            },
            {
                "question1": "How do I learn machine learning?",
                "question2": "What are the best tourist places in Nepal?",
            },
        ]
    )
    return sample.to_csv(index=False).encode("utf-8")


st.sidebar.title("Model Status")
st.sidebar.markdown("**Artifact folder**")
st.sidebar.code(str(default_artifact_dir()))

try:
    settings, classifier, model_path, metadata_path = load_runtime_assets()
except Exception as error:
    settings = {
        "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
        "best_threshold": DEFAULT_THRESHOLD,
        "best_model_name": DEFAULT_MODEL_NAME,
        "test_metrics": {},
        "feature_summary": {},
    }
    classifier = None
    st.sidebar.error(f"Artifact loading error:\n{error}")


st.markdown(
    """
    <div class="hero">
        <h1>Quora Duplicate Question Detector</h1>
        <p>
            Compare two questions, inspect the confidence signal, or score a CSV
            of question pairs with the trained duplicate-detection model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if classifier is None:
    st.error(
        "Artifacts were not loaded. Place `quora_duplicate_classifier.joblib` "
        "and `metadata.json` inside the repo-level `artifacts/` folder."
    )
    st.stop()


best_threshold = settings["best_threshold"]
best_model_name = settings["best_model_name"]
embedding_model_name = settings["embedding_model_name"]
test_metrics = settings["test_metrics"]
feature_summary = settings["feature_summary"]

st.sidebar.success("Artifacts loaded")
st.sidebar.caption(f"Model: {best_model_name}")
st.sidebar.caption(f"Embedding: {embedding_model_name}")
st.sidebar.caption(f"Threshold: {best_threshold:.3f}")

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric_card("Model", best_model_name)
with metric_columns[1]:
    render_metric_card("Embedding", embedding_model_name, font_size="0.95rem")
with metric_columns[2]:
    render_metric_card("Threshold", f"{best_threshold:.3f}")
with metric_columns[3]:
    f1_value = test_metrics.get("f1")
    render_metric_card("Test F1", f"{f1_value:.4f}" if f1_value is not None else "N/A")


tab_single, tab_modern, tab_batch, tab_details = st.tabs(
    ["Single Prediction", "Modern NLP", "Batch Prediction", "Project Details"]
)


with tab_single:
    st.subheader("Single Prediction")
    st.markdown(
        '<div class="section-note">Use the saved trained classifier for the project\'s original prediction flow.</div>',
        unsafe_allow_html=True,
    )

    selected_example = st.selectbox("Load example", list(EXAMPLES.keys()), key="single_example")
    default_q1, default_q2 = EXAMPLES[selected_example]

    with st.form("single_prediction_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            question_1 = st.text_area(
                "Question 1",
                value=default_q1,
                height=130,
                max_chars=MAX_QUESTION_CHARS,
            )
        with col_right:
            question_2 = st.text_area(
                "Question 2",
                value=default_q2,
                height=130,
                max_chars=MAX_QUESTION_CHARS,
            )

        custom_threshold = st.slider(
            "Prediction threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(best_threshold),
            step=0.01,
            help="Higher thresholds require stronger evidence before returning Duplicate.",
        )
        run_prediction = st.form_submit_button("Predict duplicate status", use_container_width=True)

    if run_prediction:
        errors = validate_question_pair(question_1, question_2)
        if not show_validation_errors(errors):
            with st.spinner("Building features and scoring the pair..."):
                result = predict_single_pair(
                    question_1=question_1,
                    question_2=question_2,
                    embedder=cached_embedder(embedding_model_name),
                    classifier=classifier,
                    threshold=custom_threshold,
                )

            probability = result["duplicate_probability"]
            render_decision(
                result["label"],
                probability,
                custom_threshold,
                "The saved model combines semantic embeddings, lexical overlap, and XGBoost probability.",
            )

            result_columns = st.columns(4)
            result_columns[0].metric("Probability", f"{probability:.2%}")
            result_columns[1].metric("Confidence Band", probability_band(probability))
            result_columns[2].metric("Cosine Similarity", f"{result['cosine_similarity']:.4f}")
            result_columns[3].metric("Jaccard Similarity", f"{result['jaccard_similarity']:.4f}")

            with st.expander("Feature snapshot"):
                feature_columns = st.columns(3)
                feature_columns[0].metric("Q1 Word Count", result["q1_word_count"])
                feature_columns[1].metric("Q2 Word Count", result["q2_word_count"])
                feature_columns[2].metric(
                    "Token Overlap Ratio",
                    f"{result['token_overlap_ratio']:.4f}",
                )
                st.json(
                    {
                        "question1": result["question1"],
                        "question2": result["question2"],
                        "duplicate_probability": round(result["duplicate_probability"], 6),
                        "prediction": result["prediction"],
                        "label": result["label"],
                    }
                )


with tab_modern:
    st.subheader("Modern NLP")
    st.markdown(
        '<div class="section-note">Compare the same task with transformer-native semantic similarity and optional pair classification.</div>',
        unsafe_allow_html=True,
    )

    selected_modern_example = st.selectbox(
        "Load example",
        list(EXAMPLES.keys()),
        key="modern_example",
    )
    modern_default_q1, modern_default_q2 = EXAMPLES[selected_modern_example]

    with st.form("modern_nlp_form"):
        modern_left, modern_right = st.columns(2)
        with modern_left:
            modern_question_1 = st.text_area(
                "Question 1",
                value=modern_default_q1,
                height=130,
                max_chars=MAX_QUESTION_CHARS,
                key="modern_q1",
            )
        with modern_right:
            modern_question_2 = st.text_area(
                "Question 2",
                value=modern_default_q2,
                height=130,
                max_chars=MAX_QUESTION_CHARS,
                key="modern_q2",
            )

        use_cross_encoder = st.toggle(
            "Use cross-encoder pair classifier",
            value=True,
            help="More accurate for one pair at a time, but slower than embedding similarity.",
        )
        run_modern_prediction = st.form_submit_button("Compare with modern NLP", use_container_width=True)

    st.caption(
        f"Embedding model: `{MODERN_EMBEDDING_MODEL}`"
        + (f" | Cross-encoder: `{CROSS_ENCODER_MODEL}`" if use_cross_encoder else "")
    )

    if run_modern_prediction:
        errors = validate_question_pair(modern_question_1, modern_question_2)
        if not show_validation_errors(errors):
            try:
                with st.spinner("Loading transformer scorer and comparing questions..."):
                    modern_embedder = cached_embedder(MODERN_EMBEDDING_MODEL)
                    modern_cross_encoder = (
                        cached_cross_encoder(CROSS_ENCODER_MODEL)
                        if use_cross_encoder
                        else None
                    )
                    modern_result = compare_questions_modern(
                        modern_question_1,
                        modern_question_2,
                        embedder=modern_embedder,
                        cross_encoder=modern_cross_encoder,
                    )

                render_decision(
                    modern_result.label,
                    modern_result.final_score,
                    modern_result.threshold,
                    f"Decision method: {modern_result.method}.",
                )

                modern_metrics = st.columns(4)
                modern_metrics[0].metric("Final Score", f"{modern_result.final_score:.3f}")
                modern_metrics[1].metric(
                    "Embedding Similarity",
                    f"{modern_result.semantic_similarity:.3f}",
                )
                modern_metrics[2].metric(
                    "Embedding Distance",
                    f"{modern_result.embedding_distance:.3f}",
                )
                modern_metrics[3].metric(
                    "Cross-Encoder Score",
                    (
                        f"{modern_result.cross_encoder_score:.3f}"
                        if modern_result.cross_encoder_score is not None
                        else "N/A"
                    ),
                )

                with st.expander("Similarity details"):
                    st.metric("Jaccard Similarity", f"{modern_result.jaccard_similarity:.3f}")
                    st.metric("Token Overlap", f"{modern_result.token_overlap_ratio:.3f}")
                    st.json(
                        {
                            "method": modern_result.method,
                            "label": modern_result.label,
                            "final_score": round(modern_result.final_score, 6),
                            "threshold": modern_result.threshold,
                            "semantic_similarity": round(
                                modern_result.semantic_similarity,
                                6,
                            ),
                            "cross_encoder_score": (
                                round(modern_result.cross_encoder_score, 6)
                                if modern_result.cross_encoder_score is not None
                                else None
                            ),
                        }
                    )
            except Exception as error:
                st.error(f"Modern NLP scoring failed: {error}")


with tab_batch:
    st.subheader("Batch Prediction")
    st.markdown(
        '<div class="section-note">Upload a CSV with `question1` and `question2` columns. Rows with missing questions are rejected before scoring.</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download sample CSV",
        data=sample_csv_bytes(),
        file_name="sample_question_pairs.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read CSV: {error}")
            batch_df = None

        if batch_df is not None:
            errors = validate_batch_frame(batch_df)
            if show_validation_errors(errors):
                st.stop()

            st.success(f"File loaded successfully. Rows: {len(batch_df):,}")

            preview_rows = st.slider(
                "Preview rows",
                min_value=1,
                max_value=min(50, len(batch_df)),
                value=min(10, len(batch_df)),
            )
            st.dataframe(batch_df.head(preview_rows), use_container_width=True)

            if st.button("Run batch prediction", use_container_width=True):
                with st.spinner("Encoding text and generating predictions..."):
                    result_df = predict_batch(
                        batch_df,
                        embedder=cached_embedder(embedding_model_name),
                        classifier=classifier,
                        threshold=best_threshold,
                    )

                st.subheader("Batch Output")
                duplicate_count = int((result_df["prediction"] == 1).sum())
                average_probability = result_df["duplicate_probability"].mean()

                batch_columns = st.columns(4)
                batch_columns[0].metric("Rows Scored", f"{len(result_df):,}")
                batch_columns[1].metric("Duplicates", f"{duplicate_count:,}")
                batch_columns[2].metric(
                    "Duplicate Rate",
                    f"{duplicate_count / len(result_df):.2%}",
                )
                batch_columns[3].metric(
                    "Avg Probability",
                    f"{average_probability:.2%}",
                )

                display_filter = st.radio(
                    "Result filter",
                    ["All", "Duplicate", "Not Duplicate"],
                    horizontal=True,
                )
                filtered_df = result_df
                if display_filter != "All":
                    filtered_df = result_df[result_df["label"] == display_filter]

                st.dataframe(filtered_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions CSV",
                    data=csv_bytes,
                    file_name="quora_duplicate_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


with tab_details:
    st.subheader("Project Details")

    st.markdown("#### Runtime model")
    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.write("Features used by the saved classifier:")
        st.write("- Cosine similarity between question embeddings")
        st.write("- Word counts and character counts")
        st.write("- Word-count and character-count differences")
        st.write("- Jaccard similarity")
        st.write("- Token overlap ratio")
        st.write("- Absolute embedding difference")
        st.write("- Element-wise embedding product")

    with detail_columns[1]:
        st.write("Saved metadata:")
        st.json(
            {
                "embedding_model_name": embedding_model_name,
                "best_model_name": best_model_name,
                "best_threshold": best_threshold,
                "feature_summary": feature_summary,
                "test_metrics": test_metrics,
            }
        )

    st.info(
        "The Single Prediction and Batch Prediction tabs use the saved classifier. "
        "The Modern NLP tab is independent and compares questions with transformer-based similarity."
    )
