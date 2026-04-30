from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import math
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from quora_duplicate_detector.text_features import (
    jaccard_similarity,
    normalize_text,
    token_overlap_ratio,
    word_count,
)


MAX_QUESTION_CHARS = 1000
MAX_BATCH_ROWS = 5000
DEFAULT_THRESHOLD = 0.42
RECALL_THRESHOLD = 0.30
BALANCED_THRESHOLD = 0.38
STRICT_THRESHOLD = 0.52
REQUIRED_COLUMNS = {"question1", "question2"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "best", "can", "do", "does", "for",
    "from", "get", "how", "i", "in", "is", "it", "my", "of", "on", "or",
    "should", "the", "to", "way", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}
SYNONYM_GROUPS = {
    "learn": {"learn", "study", "master", "understand"},
    "fast": {"fast", "quick", "quickly", "rapid", "rapidly", "faster", "fastest"},
    "prepare": {"prepare", "ready", "practice", "prep"},
    "job": {"job", "career", "interview", "interviews"},
    "good": {"good", "great", "best", "better"},
    "improve": {"improve", "increase", "enhance", "boost"},
}
SYNONYM_LOOKUP = {
    alias: canonical
    for canonical, aliases in SYNONYM_GROUPS.items()
    for alias in aliases
}

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
    page_title="Quora Duplicate AI",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 18% 10%, rgba(34, 211, 238, .22), transparent 28%),
            radial-gradient(circle at 84% 16%, rgba(16, 185, 129, .16), transparent 26%),
            linear-gradient(135deg, #020617 0%, #07111f 52%, #020617 100%);
    }
    .block-container { max-width: 1180px; padding-top: 1.5rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        border-right: 1px solid rgba(34, 211, 238, .22);
    }
    h1, h2, h3, label, p, .stMarkdown, .stCaption { color: #e5f9ff; }
    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, .92) 0%, rgba(8, 47, 73, .88) 52%, rgba(20, 83, 45, .72) 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(34, 211, 238, .34);
        box-shadow: 0 0 32px rgba(34, 211, 238, .13), inset 0 0 32px rgba(15, 23, 42, .35);
    }
    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(90deg, black, transparent 75%);
        pointer-events: none;
    }
    .hero h1 { margin: 0 0 .4rem 0; font-size: 2.15rem; letter-spacing: 0; }
    .hero p { margin: 0; color: #bae6fd; max-width: 800px; }
    .cyber-pill {
        display: inline-block;
        color: #67e8f9;
        border: 1px solid rgba(103, 232, 249, .45);
        border-radius: 999px;
        padding: .22rem .65rem;
        font-size: .78rem;
        margin-bottom: .65rem;
        background: rgba(8, 47, 73, .4);
    }
    .result {
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin: .75rem 0 1rem 0;
        border: 1px solid rgba(148, 163, 184, .28);
        color: #e5f9ff;
        box-shadow: 0 12px 28px rgba(0,0,0,.22);
    }
    .duplicate { background: rgba(16, 185, 129, .16); border-color: rgba(16, 185, 129, .62); }
    .not-duplicate { background: rgba(245, 158, 11, .16); border-color: rgba(245, 158, 11, .62); }
    .small-note { color: #bae6fd; font-size: .95rem; }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, .72);
        border: 1px solid rgba(34, 211, 238, .2);
        border-radius: 10px;
        padding: .75rem;
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(34, 211, 238, .55);
        background: linear-gradient(135deg, #0891b2 0%, #10b981 100%);
        color: white;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


def word_ngrams(text: str) -> list[str]:
    tokens = canonical_tokens(text)
    unigrams = tokens
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return unigrams + bigrams


def canonical_tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = normalize_text(text).split()
    normalized = []
    for token in tokens:
        token = SYNONYM_LOOKUP.get(token, token)
        if keep_stopwords or token not in STOPWORDS:
            normalized.append(token)
    return normalized


def content_overlap(question_1: str, question_2: str) -> float:
    left = set(canonical_tokens(question_1))
    right = set(canonical_tokens(question_2))
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / min(len(left), len(right))


def content_jaccard(question_1: str, question_2: str) -> float:
    left = set(canonical_tokens(question_1))
    right = set(canonical_tokens(question_2))
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def char_ngrams(text: str, n: int = 3) -> list[str]:
    normalized = normalize_text(text).replace(" ", "")
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[index : index + n] for index in range(len(normalized) - n + 1)]


def hashed_vector(terms: list[str], buckets: int = 4096) -> Counter:
    vector = Counter()
    for term in terms:
        digest = hashlib.md5(term.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % buckets
        vector[bucket] += 1.0
    return vector


def normalized_dot(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0

    dot_product = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def word_vector_similarity(question_1: str, question_2: str) -> float:
    return normalized_dot(hashed_vector(word_ngrams(question_1)), hashed_vector(word_ngrams(question_2)))


def char_vector_similarity(question_1: str, question_2: str) -> float:
    return normalized_dot(hashed_vector(char_ngrams(question_1)), hashed_vector(char_ngrams(question_2)))


def length_similarity(question_1: str, question_2: str) -> float:
    left = max(word_count(question_1), 1)
    right = max(word_count(question_2), 1)
    return min(left, right) / max(left, right)


def sequence_similarity(question_1: str, question_2: str) -> float:
    return SequenceMatcher(None, normalize_text(question_1), normalize_text(question_2)).ratio()


def duplicate_score(question_1: str, question_2: str, threshold: float) -> dict:
    word_similarity = word_vector_similarity(question_1, question_2)
    char_similarity = char_vector_similarity(question_1, question_2)
    sequence_score = sequence_similarity(question_1, question_2)
    jaccard = jaccard_similarity(question_1, question_2)
    overlap = token_overlap_ratio(question_1, question_2)
    content_token_overlap = content_overlap(question_1, question_2)
    content_token_jaccard = content_jaccard(question_1, question_2)
    length_score = length_similarity(question_1, question_2)
    exact_match = float(normalize_text(question_1) == normalize_text(question_2))

    score = (
        0.18 * word_similarity
        + 0.18 * char_similarity
        + 0.16 * sequence_score
        + 0.17 * overlap
        + 0.20 * content_token_overlap
        + 0.06 * content_token_jaccard
        + 0.03 * jaccard
        + 0.02 * length_score
        + 0.05 * exact_match
    )
    score = max(0.0, min(1.0, float(score)))

    recall_rule = (
        content_token_overlap >= 0.50
        and (char_similarity >= 0.32 or sequence_score >= 0.34 or word_similarity >= 0.25)
    ) or (
        overlap >= 0.45 and char_similarity >= 0.28
    )
    if recall_rule:
        score = max(score, threshold + 0.03)

    return {
        "duplicate_probability": score,
        "label": "Duplicate" if score >= threshold else "Not Duplicate",
        "word_similarity": word_similarity,
        "char_similarity": char_similarity,
        "sequence_similarity": sequence_score,
        "jaccard_similarity": jaccard,
        "token_overlap_ratio": overlap,
        "content_overlap": content_token_overlap,
        "content_jaccard": content_token_jaccard,
        "length_similarity": length_score,
    }


def validate_pair(question_1: str, question_2: str) -> list[str]:
    errors = []
    if not question_1.strip() or not question_2.strip():
        errors.append("Both questions are required.")
    if len(question_1) > MAX_QUESTION_CHARS or len(question_2) > MAX_QUESTION_CHARS:
        errors.append(f"Each question must be {MAX_QUESTION_CHARS} characters or fewer.")
    return errors


def render_result(result: dict, threshold: float) -> None:
    css_class = "duplicate" if result["label"] == "Duplicate" else "not-duplicate"
    st.markdown(
        f"""
        <div class="result {css_class}">
            <h3 style="margin:0 0 .25rem 0;">{result["label"]}</h3>
            <div>Score: <b>{result["duplicate_probability"]:.3f}</b> | Threshold: <b>{threshold:.3f}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(result["duplicate_probability"])


def score_batch(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    scores = [
        duplicate_score(str(row.question1), str(row.question2), threshold)
        for row in result.itertuples(index=False)
    ]
    result["duplicate_probability"] = [item["duplicate_probability"] for item in scores]
    result["label"] = [item["label"] for item in scores]
    result["word_similarity"] = [item["word_similarity"] for item in scores]
    result["char_similarity"] = [item["char_similarity"] for item in scores]
    result["sequence_similarity"] = [item["sequence_similarity"] for item in scores]
    result["jaccard_similarity"] = [item["jaccard_similarity"] for item in scores]
    result["token_overlap_ratio"] = [item["token_overlap_ratio"] for item in scores]
    result["content_overlap"] = [item["content_overlap"] for item in scores]
    return result


st.markdown(
    """
    <div class="hero">
        <div class="cyber-pill">SEMANTIC MATCH ENGINE / CLOUD MODE</div>
        <h1>Quora Duplicate AI</h1>
        <p>
            A cloud-ready semantic question matcher by Mohammed Ghanim Siddiqui.
            Compare question pairs with a recall-first similarity engine, inspect signals,
            and score CSV uploads in a fast cyber-style dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("App Mode")
st.sidebar.success("Cloud deployment mode")
st.sidebar.caption("Recall-first lightweight semantic scoring")
st.sidebar.caption("Full transformer/XGBoost pipeline remains in the repository for local use.")

tab_single, tab_batch, tab_about = st.tabs(["Single Prediction", "Batch Prediction", "About"])

with tab_single:
    st.subheader("Single Prediction")
    selected_example = st.selectbox("Load example", list(EXAMPLES.keys()))
    default_q1, default_q2 = EXAMPLES[selected_example]

    with st.form("single_form"):
        left, right = st.columns(2)
        with left:
            question_1 = st.text_area("Question 1", value=default_q1, height=130, max_chars=MAX_QUESTION_CHARS)
        with right:
            question_2 = st.text_area("Question 2", value=default_q2, height=130, max_chars=MAX_QUESTION_CHARS)

        sensitivity = st.radio(
            "Sensitivity mode",
            ["High Recall", "Balanced", "Strict"],
            horizontal=True,
            help="High Recall catches more duplicates. Strict reduces false positives.",
        )
        default_threshold = {
            "High Recall": RECALL_THRESHOLD,
            "Balanced": BALANCED_THRESHOLD,
            "Strict": STRICT_THRESHOLD,
        }[sensitivity]
        threshold = st.slider("Duplicate threshold", 0.10, 0.90, default_threshold, 0.01)
        submitted = st.form_submit_button("Compare questions", use_container_width=True)

    if submitted:
        errors = validate_pair(question_1, question_2)
        if errors:
            for error in errors:
                st.error(error)
        else:
            result = duplicate_score(question_1, question_2, threshold)
            render_result(result, threshold)

            metrics = st.columns(4)
            metrics[0].metric("Word Similarity", f"{result['word_similarity']:.3f}")
            metrics[1].metric("Char Similarity", f"{result['char_similarity']:.3f}")
            metrics[2].metric("Sequence Similarity", f"{result['sequence_similarity']:.3f}")
            metrics[3].metric("Content Overlap", f"{result['content_overlap']:.3f}")

with tab_batch:
    st.subheader("Batch Prediction")
    st.markdown('<div class="small-note">Upload a CSV containing `question1` and `question2` columns.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_mode = st.radio(
        "Batch sensitivity",
        ["High Recall", "Balanced", "Strict"],
        horizontal=True,
    )
    batch_default = {
        "High Recall": RECALL_THRESHOLD,
        "Balanced": BALANCED_THRESHOLD,
        "Strict": STRICT_THRESHOLD,
    }[batch_mode]
    batch_threshold = st.slider(
        "Batch duplicate threshold",
        min_value=0.10,
        max_value=0.90,
        value=batch_default,
        step=0.01,
    )
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Could not read CSV: {error}")
            frame = None

        if frame is not None:
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            elif frame.empty:
                st.error("CSV contains no rows.")
            elif len(frame) > MAX_BATCH_ROWS:
                st.error(f"Please upload {MAX_BATCH_ROWS:,} rows or fewer.")
            else:
                st.success(f"Loaded {len(frame):,} rows")
                st.dataframe(frame.head(min(20, len(frame))), use_container_width=True)

                if st.button("Run batch prediction", use_container_width=True):
                    output = score_batch(frame, batch_threshold)
                    duplicate_count = int((output["label"] == "Duplicate").sum())
                    cols = st.columns(3)
                    cols[0].metric("Rows", f"{len(output):,}")
                    cols[1].metric("Duplicates", f"{duplicate_count:,}")
                    cols[2].metric("Duplicate Rate", f"{duplicate_count / len(output):.2%}")
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        "quora_duplicate_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )

with tab_about:
    st.subheader("About this deployment")
    st.write(
        "This public Streamlit deployment uses a lightweight semantic scoring engine "
        "so the hosted app starts quickly on Community Cloud. The repository also includes "
        "the full local transformer/XGBoost implementation, model artifacts, and v2 training pipeline."
    )
    st.markdown(
        "- Author: **Mohammed Ghanim Siddiqui**\n"
        "- GitHub: [MOHAMMED-GHANIM-SIDDIQUI](https://github.com/MOHAMMED-GHANIM-SIDDIQUI)\n"
        "- Repository: [quora-duplicate-detector-improved](https://github.com/MOHAMMED-GHANIM-SIDDIQUI/quora-duplicate-detector-improved)"
    )
